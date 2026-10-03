"""DOCX extraction with python-docx.

How python-docx reads a document
--------------------------------
A .docx is a zip; `word/document.xml` holds the body as an ordered sequence of
<w:p> (paragraph) and <w:tbl> (table) elements. python-docx wraps these with
styles resolved from styles.xml. Unlike PDF there is nothing to *detect*:
- reading order is the XML order;
- a table's rows/cells are explicit (<w:tr>/<w:tc>), so cell text is exact and
  merged cells are flagged (gridSpan / vMerge);
- images are <w:drawing> runs referencing parts in word/media/, so the exact
  original bytes are available, no cropping;
- captions are ordinary paragraphs: we take the paragraph immediately after an
  image when it matches the caption pattern.

What it cannot give: page numbers and positions. See ae.extract.pagemap.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import docx
from docx.document import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from ae.extract.captions import CAPTION_RE, figure_id
from ae.extract.native.figures import FIG_DIR, ocr_labels
from ae.extract.pagemap import BlockPlacer, PageLocator, assemble_pages, render_to_pdf, tables_by_page
from ae.log import get_logger
from ae.schema import Block, FigureBlock, ParsedDocument, Role, TextBlock

log = get_logger(__name__)

EMU_PER_INCH = 914400
"""OOXML drawing extents are in English Metric Units."""
POINTS_PER_INCH = 72
"""PDF points per inch."""
HEADING_MAX_CHARS = 90
"""A bold numbered/upper-case paragraph shorter than this is a heading even without a heading style."""


def parse_docx(path: Path) -> ParsedDocument:
    """Parse a DOCX from its XML into the shared schema.

    Content comes from the XML; pages and boxes come from a LibreOffice render when one is
    available (`meta["page_mapping"]` says which). Raises ValueError when the document
    declares no page geometry.
    """
    path = Path(path)
    document = docx.Document(str(path))
    page_w, page_h, left, top, right = _page_geometry(document, path)

    pdf = render_to_pdf(path)
    locator = PageLocator(pdf) if pdf else None
    walker = _BodyWalker(document, path, BlockPlacer(locator, x0=left, x1=right, top=top))
    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            walker.paragraph(child)
        elif tag == "tbl":
            walker.table(child)

    log.debug(f"{path.name}: {len(walker.items)} blocks, {walker.fig_i} figures, {walker.tbl_i} tables")
    return ParsedDocument(
        doc=path.name,
        source_path=str(path),
        pages=assemble_pages(walker.items, locator, (page_w, page_h)),
        meta={
            "page_mapping": "libreoffice_render" if locator else "unavailable",
            "rendered_pdf": str(pdf) if pdf else None,
        },
    )


def _page_geometry(document: Document, path: Path) -> tuple[float, float, float, float, float]:
    """(page width, page height, left margin x, top margin y, right margin x) of the first section, in points."""
    if not document.sections:
        raise ValueError(f"{path.name}: the document has no section properties (w:sectPr), so its page size is unknown")
    sec = document.sections[0]
    dims = (sec.page_width, sec.page_height, sec.left_margin, sec.top_margin, sec.right_margin)
    if any(d is None for d in dims):
        raise ValueError(
            f"{path.name}: the first section does not declare its page size and margins (w:pgSz / w:pgMar); "
            "re-save the file from Word or LibreOffice so they are written out"
        )
    page_w, page_h, left, top, right = (d.pt for d in dims if d is not None)
    return page_w, page_h, left, top, page_w - right


def _paragraph_role(p: Paragraph, text: str) -> Role:
    """Heading from the style name or bold numbered/upper-case text; caption from the caption pattern."""
    style = ((p.style.name if p.style is not None else None) or "").lower()
    runs = [r for r in p.runs if r.text.strip()]
    all_bold = bool(runs) and all(
        bool(r.bold) or (r.style is not None and "strong" in (r.style.name or "").lower()) for r in runs
    )
    looks_heading = (
        len(text) < HEADING_MAX_CHARS and (re.match(r"^\d+(\.\d+)*\.?\s+\S", text) or text.isupper()) and all_bold
    )
    if style.startswith(("heading", "title")) or looks_heading:
        return "heading"
    return "caption" if CAPTION_RE.match(text) else "body"


def _table_rows(t: Table) -> list[list[str]]:
    """Cell text per row, one entry per merged cell; all-empty rows dropped."""
    rows = []
    for r in t.rows:
        cells, prev = [], None
        for c in r.cells:
            # python-docx returns the same _tc for horizontally merged cells; keep one copy.
            txt = re.sub(r"\s+", " ", c.text).strip()
            if c._tc is prev:
                continue
            prev = c._tc
            cells.append(txt)
        rows.append(cells)
    return [r for r in rows if any(r)]


class _BodyWalker:
    """Turns body elements into (page, block) items in document order.

    State carried between elements: the figure waiting for its caption paragraph, the
    last heading (a table's title), and figure/table counters for file names and ids.
    """

    def __init__(self, document: Document, path: Path, placer: BlockPlacer) -> None:
        self.document = document
        self.path = path
        self.placer = placer
        self.items: list[tuple[int, Block]] = []
        self.last_heading: str | None = None
        self.pending_figure: FigureBlock | None = None
        self.fig_i = 0
        self.tbl_i = 0

    def paragraph(self, child: Any) -> None:
        """Add a paragraph: a figure if it holds an image, a caption for the pending figure, or text."""
        blips = child.findall(".//" + qn("a:blip"))
        if blips:
            self._figure(child, blips[0])
            return
        p = Paragraph(child, self.document)
        text = re.sub(r"\s+", " ", p.text).strip()
        if not text:
            return
        role = _paragraph_role(p, text)
        if role == "heading":
            self.last_heading = text
        if role == "caption" and self.pending_figure is not None and self.pending_figure.caption is None:
            self.pending_figure.caption = text
            self.pending_figure.figure_id = figure_id(text)
            self.pending_figure = None
            return
        self.pending_figure = None
        pno, bbox = self.placer.place_text(text)
        self.items.append((pno, TextBlock(bbox=bbox, text=text, role=role)))

    def _figure(self, child: Any, blip: Any) -> None:
        """Save the paragraph's (first) image with its original bytes and add it as a figure."""
        self.fig_i += 1
        part = self.document.part.related_parts[blip.get(qn("r:embed"))]
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        img_path = FIG_DIR / f"{self.path.stem}_fig{self.fig_i}{Path(str(part.partname)).suffix}"
        img_path.write_bytes(part.blob)
        extent = child.find(".//" + qn("wp:extent"))
        if extent is not None:
            pno, bbox = self.placer.place_image(int(extent.get("cy")) / EMU_PER_INCH * POINTS_PER_INCH)
        else:
            pno, bbox = self.placer.place_image()
        self.pending_figure = FigureBlock(bbox=bbox, image_path=str(img_path), ocr_labels=ocr_labels(img_path))
        self.items.append((pno, self.pending_figure))

    def table(self, child: Any) -> None:
        """Add a table, split into one piece per rendered page, titled by the last heading."""
        rows = _table_rows(Table(child, self.document))
        self.tbl_i += 1
        fallback = self.placer.table_fallback(len(rows))
        table_id = f"{self.path.stem}#t{self.tbl_i}"
        self.items += tables_by_page(rows, self.placer.locator, self.last_heading, table_id, fallback)

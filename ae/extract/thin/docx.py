"""DOCX extraction (thin backend) with python-docx.

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

import docx
from docx.oxml.ns import qn

from ae.extract.cache import PageCache
from ae.extract.pagemap import PageLocator, render_to_pdf, tables_by_page
from ae.extract.thin.figures import figure_id, ocr_labels
from ae.extract.thin.text import CAPTION_RE
from ae.schema import BBox, Block, FigureBlock, Page, ParsedDocument, TableBlock, TextBlock

VERSION = "1"
FIG_DIR = Path("data/extracted/thin/figures")


def parse_docx(path: Path, use_cache: bool = True, backend_name: str = "thin") -> ParsedDocument:
    path = Path(path)
    d = docx.Document(str(path))
    sec = d.sections[0]
    page_w, page_h = sec.page_width.pt, sec.page_height.pt
    margins = (sec.left_margin.pt, sec.top_margin.pt, page_w - sec.right_margin.pt)

    pdf = render_to_pdf(path)
    locator = PageLocator(pdf) if pdf else None
    cursor_y = margins[1]
    fig_i = tbl_i = 0
    items: list[tuple[int, Block]] = []  # (page_no, block) in document order
    last_heading: str | None = None
    pending_figure: FigureBlock | None = None

    def flow_bbox(height: float) -> BBox:
        nonlocal cursor_y
        b = BBox(x0=margins[0], y0=cursor_y, x1=margins[2], y1=cursor_y + height)
        cursor_y += height + 6
        return b

    def place_text(text: str, est_h: float) -> tuple[int, BBox]:
        if locator:
            hit = locator.locate_text(text)
            if hit:
                return hit
        return 1, flow_bbox(est_h)

    for child in d.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            p = docx.text.paragraph.Paragraph(child, d)
            blips = child.findall(".//" + qn("a:blip"))
            text = re.sub(r"\s+", " ", p.text).strip()
            if blips:
                fig_i += 1
                part = d.part.related_parts[blips[0].get(qn("r:embed"))]
                FIG_DIR.mkdir(parents=True, exist_ok=True)
                img_path = FIG_DIR / f"{path.stem}_fig{fig_i}{Path(str(part.partname)).suffix}"
                img_path.write_bytes(part.blob)
                ext = child.find(".//" + qn("wp:extent"))
                h = (int(ext.get("cy")) / 914400 * 72) if ext is not None else 200.0
                hit = locator.next_image() if locator else None
                pno, bbox = hit if hit else (1, flow_bbox(h))
                pending_figure = FigureBlock(bbox=bbox, image_path=str(img_path), ocr_labels=ocr_labels(img_path))
                items.append((pno, pending_figure))
                continue
            if not text:
                continue
            style = (p.style.name or "").lower()
            role = "heading" if style.startswith(("heading", "title")) else ("caption" if CAPTION_RE.match(text) else "body")
            if role == "heading":
                last_heading = text
            if role == "caption" and pending_figure is not None and pending_figure.caption is None:
                pending_figure.caption = text
                pending_figure.figure_id = figure_id(text)
                pending_figure = None
                continue
            pending_figure = None
            lines = max(1, len(text) // 95 + 1)
            pno, bbox = place_text(text, 14.0 * lines)
            items.append((pno, TextBlock(bbox=bbox, text=text, role=role)))  # type: ignore[arg-type]
        elif tag == "tbl":
            t = docx.table.Table(child, d)
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
            rows = [r for r in rows if any(r)]
            tbl_i += 1
            fallback = (1, flow_bbox(16.0 * len(rows)))
            fallback = (fallback[0], BBox(x0=margins[0], y0=fallback[1].y0, x1=margins[2], y1=fallback[1].y1))
            items += tables_by_page(rows, locator, last_heading, f"{path.stem}#t{tbl_i}", fallback)

    n_pages = len(locator.sizes) if locator else 1
    pages = []
    for pno in range(1, n_pages + 1):
        w, h = locator.sizes[pno - 1] if locator else (page_w, page_h)
        pages.append(Page(number=pno, width=w, height=h, blocks=[b for p, b in items if p == pno], backend=backend_name))
    return ParsedDocument(
        doc=path.name,
        source_path=str(path),
        backend=backend_name,
        pages=pages,
        meta={"page_mapping": "libreoffice_render" if locator else "unavailable", "rendered_pdf": str(pdf) if pdf else None},
    )

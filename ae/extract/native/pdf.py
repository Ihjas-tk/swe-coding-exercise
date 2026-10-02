"""Native backend orchestrator: one PDF -> ParsedDocument, page by page.

Per page:
  1. scanned?  -> Tesseract blocks            else -> PyMuPDF text-layer blocks
  2. merge + order blocks, tag roles            (text.py)
  3. tables via pdfplumber, drop duplicate text (tables.py)
  4. figures: crop, caption link, label OCR     (figures.py)
  5. assemble blocks in reading order; cache the page
After all pages: resolve captions that landed on the page after their figure.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pymupdf

from ae.extract.cache import PageCache
from ae.extract.captions import FIG_TITLE_RE, figure_id
from ae.extract.native import figures as figmod
from ae.extract.native import ocr as ocrmod
from ae.extract.native import tables as tabmod
from ae.extract.native import text as textmod
from ae.extract.native.figures import FIG_DIR
from ae.log import get_logger
from ae.schema import BBox, Block, FigureBlock, Page, ParsedDocument, TableBlock, TextBlock

log = get_logger(__name__)

VERSION = "10"
"""Page-cache version of this backend; bump it whenever a change alters parsed pages."""

# Figures on scanned pages (located from the OCR layout) ----------------------------------
SCAN_PARAGRAPH_MIN_WORDS = 8
"""An OCR block with fewer words is a figure label, not the prose that bounds a figure."""
SCAN_FIGURE_MIN_GAP = 60.0
"""Points of empty band between prose and a caption for the band to hold a figure."""
SCAN_FIGURE_PAD = 4.0
"""Points the figure box is grown beyond the bounding prose/caption edges."""
SCAN_CAPTION_CLEARANCE = 2.0
"""Points left between the figure box bottom and its caption."""

# Captions on the page after their figure -------------------------------------------------
ORPHAN_FIGURE_MIN_Y_FRAC = 0.6
"""An uncaptioned figure whose bottom is below this fraction of the page may be captioned on the next page."""
CONTINUED_CAPTION_MAX_Y_FRAC = 0.25
"""... by a caption whose top is above this fraction of the next page."""


def parse_pdf(path: Path, use_cache: bool = True, ocr_labels: bool = True) -> ParsedDocument:
    """Parse a PDF into the shared schema, page by page.

    Pages are read from / written to the per-page cache (`use_cache=False` re-parses but
    still refreshes the cache). `ocr_labels=False` skips Tesseract on figure crops.
    """
    path = Path(path)
    cache = PageCache("native", VERSION, path)
    pages: list[Page] = []
    with pymupdf.open(path) as doc, pdfplumber.open(path) as pl:
        for pno in range(len(doc)):
            cached = cache.get(pno + 1) if use_cache else None
            if cached is not None:
                log.debug(f"{path.name} p{pno + 1}: page cache hit")
                pages.append(cached)
                continue
            page = _parse_page(doc[pno], pl.pages[pno], path.stem, ocr_labels)
            cache.put(page)
            pages.append(page)
    _link_cross_page_captions(pages)
    return ParsedDocument(
        doc=path.name, source_path=str(path), backend="native", pages=pages, meta={"n_pages": len(pages)}
    )


def _parse_page(page: pymupdf.Page, pl_page: pdfplumber.page.Page, doc_stem: str, do_ocr_labels: bool) -> Page:
    """Parse one uncached page (steps 1-5 of the module docstring)."""
    w, h = page.rect.width, page.rect.height
    reason = ocrmod.needs_ocr(page)
    scanned = reason is not None
    if scanned:
        log.warning(f"{doc_stem} p{page.number + 1}: OCR fallback ({reason})")
    raw = ocrmod.ocr_page(page) if scanned else textmod.raw_blocks_from_page(page)
    text_blocks = textmod.build_text_blocks(raw, h)

    tables: list[TableBlock] = [] if scanned else tabmod.extract_tables(pl_page, page, text_blocks)
    text_blocks = tabmod.drop_text_inside_tables(text_blocks, tables)

    if scanned:
        # The whole page is one image; figures are located from the OCR layout instead.
        figs, used_ids = _figures_from_scan(page, text_blocks, doc_stem, do_ocr_labels)
    else:
        figs, used_ids = figmod.extract_figures(page, text_blocks, tables, FIG_DIR, doc_stem, do_ocr_labels)

    blocks: list[Block] = [*(b for b in text_blocks if id(b) not in used_ids), *tables, *figs]
    # Final reading order over the mixed block list (same column/band logic as for text).
    blocks = [b for b, _ in textmod.order_blocks(blocks, h)]
    log.debug(
        f"{doc_stem} p{page.number + 1}: {len(text_blocks)} text, {len(tables)} tables, {len(figs)} figures"
        + (" (OCR)" if scanned else "")
    )
    return Page(
        number=page.number + 1,
        width=w,
        height=h,
        blocks=blocks,
        is_scanned=scanned,
        ocr_reason=reason,
        backend="native",
    )


def _figures_from_scan(
    page: pymupdf.Page, text_blocks: list[TextBlock], doc_stem: str, do_ocr_labels: bool
) -> tuple[list[FigureBlock], set[int]]:
    """On a scanned page, a figure is the band above a caption that holds no prose.

    Heuristic: for each caption block, the figure spans from the last paragraph-sized
    block above it in the same column down to the caption. Good enough for patent block
    diagrams; documented as a limitation for dense scans. Returns the figures and the
    `id()`s of the text blocks they consumed (caption and OCR'd label fragments).
    """
    figs: list[FigureBlock] = []
    used: set[int] = set()
    caps = [b for b in text_blocks if b.role == "caption" and not FIG_TITLE_RE.match(b.text)]
    for i, cap in enumerate(caps):
        bbox = _scan_figure_bbox(cap, text_blocks, page.rect.width)
        if bbox is None:
            continue
        path = FIG_DIR / f"{doc_stem}_p{page.number + 1}_fig{i + 1}.png"
        figmod.crop_figure(page, bbox, path)
        inside = [b for b in text_blocks if bbox.contains_centre(b.bbox)]
        used.update(id(b) for b in inside)  # OCR'd label fragments become figure labels, not body text
        labels = [word for b in inside for word in b.text.split()]
        if do_ocr_labels:
            labels += figmod.ocr_labels(path)
        used.add(id(cap))
        figs.append(
            FigureBlock(
                bbox=bbox,
                image_path=str(path),
                caption=cap.text,
                figure_id=figure_id(cap.text),
                ocr_labels=sorted(set(labels)),
            )
        )
    return figs, used


def _scan_figure_bbox(cap: TextBlock, text_blocks: list[TextBlock], page_width: float) -> BBox | None:
    """Box of the figure above caption `cap` on a scanned page, or None when the band is too short."""
    same_col = [b for b in text_blocks if b is not cap and b.column == cap.column and b.bbox.y1 <= cap.bbox.y0]
    # Figure labels OCR'd as tiny blocks sit inside the figure; the real "previous text"
    # is the last block of normal paragraph width above a sizeable gap.
    prose_above = [
        b
        for b in same_col
        if len(b.text.split()) >= SCAN_PARAGRAPH_MIN_WORDS
        and cap.bbox.y0 - b.bbox.y1 > SCAN_FIGURE_MIN_GAP
        and not FIG_TITLE_RE.match(b.text)
    ]
    top = max((b.bbox.y1 for b in prose_above), default=0.0) + SCAN_FIGURE_PAD
    if cap.bbox.y0 - top < figmod.MIN_FIGURE_PTS:
        return None
    x0 = min([cap.bbox.x0] + [b.bbox.x0 for b in same_col]) - SCAN_FIGURE_PAD
    x1 = max([cap.bbox.x1] + [b.bbox.x1 for b in same_col]) + SCAN_FIGURE_PAD
    return BBox(x0=max(0, x0), y0=top, x1=min(page_width, x1), y1=cap.bbox.y0 - SCAN_CAPTION_CLEARANCE)


def _link_cross_page_captions(pages: list[Page]) -> None:
    """Give uncaptioned figures at the bottom of a page the caption at the top of the next page.

    Mutates `pages`: the figure gets the caption, its id and `caption_page`; the caption
    text block is removed from the next page.
    """
    for i in range(len(pages) - 1):
        orphans = [
            b
            for b in pages[i].blocks
            if b.kind == "figure" and not b.caption and b.bbox.y1 > ORPHAN_FIGURE_MIN_Y_FRAC * pages[i].height
        ]
        if not orphans:
            continue
        nxt = pages[i + 1]
        for b in list(nxt.blocks):
            if b.kind == "text" and b.role == "caption" and b.bbox.y0 < CONTINUED_CAPTION_MAX_Y_FRAC * nxt.height:
                fig = orphans.pop(0)
                fig.caption = b.text
                fig.caption_page = nxt.number
                fig.figure_id = figure_id(b.text)
                nxt.blocks.remove(b)
                if not orphans:
                    break

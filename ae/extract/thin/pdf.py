"""Thin backend orchestrator: one PDF -> ParsedDocument, page by page.

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
from ae.extract.thin import figures as figmod
from ae.extract.thin import ocr as ocrmod
from ae.extract.thin import tables as tabmod
from ae.extract.thin import text as textmod
from ae.schema import Block, FigureBlock, Page, ParsedDocument, TableBlock, TextBlock

VERSION = "5"
FIG_DIR = Path("data/extracted/thin/figures")


def parse_pdf(path: Path, use_cache: bool = True, ocr_labels: bool = True) -> ParsedDocument:
    path = Path(path)
    cache = PageCache("thin", VERSION, path)
    doc = pymupdf.open(path)
    pages: list[Page] = []
    with pdfplumber.open(path) as pl:
        for pno in range(len(doc)):
            cached = cache.get(pno + 1) if use_cache else None
            if cached is not None:
                pages.append(cached)
                continue
            page = _parse_page(doc[pno], pl.pages[pno], path.stem, ocr_labels)
            cache.put(page)
            pages.append(page)
    _link_cross_page_captions(pages)
    return ParsedDocument(doc=path.name, source_path=str(path), backend="thin", pages=pages, meta={"n_pages": len(pages)})


def _parse_page(page: pymupdf.Page, pl_page, doc_stem: str, do_ocr_labels: bool) -> Page:
    w, h = page.rect.width, page.rect.height
    reason = ocrmod.needs_ocr(page)
    scanned = reason is not None
    raw = ocrmod.ocr_page(page) if scanned else textmod.raw_blocks_from_page(page)
    text_blocks = textmod.build_text_blocks(raw, w, h)

    tables: list[TableBlock] = [] if scanned else tabmod.extract_tables(pl_page, page, text_blocks)
    text_blocks = tabmod.drop_text_inside_tables(text_blocks, tables)

    if scanned:
        # The whole page is one image; figures are located from the OCR layout instead.
        figs, used_ids = _figures_from_scan(page, text_blocks, doc_stem, do_ocr_labels)
    else:
        tx0 = min((b.bbox.x0 for b in text_blocks), default=0.0)
        tx1 = max((b.bbox.x1 for b in text_blocks), default=w)
        columns = textmod.detect_columns([textmod.RawBlock(b.bbox, b.text, 0, False) for b in text_blocks], tx0, tx1)
        figs, used_ids = figmod.extract_figures(page, text_blocks, tables, columns, FIG_DIR, doc_stem, do_ocr_labels)

    blocks: list[Block] = [tb for tb in text_blocks if id(tb) not in used_ids] + tables + figs
    # Final reading order over the mixed block list (same column/band logic as for text).
    blocks = [b for b, _ in textmod.order_blocks(blocks, w, h)]
    return Page(number=page.number + 1, width=w, height=h, blocks=blocks, is_scanned=scanned, ocr_reason=reason, backend="thin")


def _figures_from_scan(page: pymupdf.Page, text_blocks: list[TextBlock], doc_stem: str, do_ocr_labels: bool):
    """On a scanned page, a figure is the largest region with a caption and little OCR text.

    Heuristic: take the caption block(s); the figure sits in the band above the caption
    bounded by the previous text block in the same column. Good enough for patent
    block diagrams; documented as a limitation for dense scans.
    """
    figs: list[FigureBlock] = []
    used: set[int] = set()
    caps = [tb for tb in text_blocks if tb.role == "caption" and not textmod.FIG_TITLE_RE.match(tb.text)]
    for i, cap in enumerate(caps):
        same_col = [tb for tb in text_blocks if tb is not cap and tb.column == cap.column and tb.bbox.y1 <= cap.bbox.y0]
        # Figure labels OCR'd as tiny blocks sit inside the figure; the real "previous text"
        # is the last block of normal paragraph width above a sizeable gap.
        prev = [
            tb
            for tb in same_col
            if len(tb.text.split()) >= 8 and cap.bbox.y0 - tb.bbox.y1 > 60 and not textmod.FIG_TITLE_RE.match(tb.text)
        ]
        top = max((tb.bbox.y1 for tb in prev), default=0.0) + 4
        if cap.bbox.y0 - top < figmod.MIN_FIGURE_PTS:
            continue
        x0 = min([cap.bbox.x0] + [tb.bbox.x0 for tb in same_col]) - 4
        x1 = max([cap.bbox.x1] + [tb.bbox.x1 for tb in same_col]) + 4
        bbox = figmod.BBox(x0=max(0, x0), y0=top, x1=min(page.rect.width, x1), y1=cap.bbox.y0 - 2)
        path = figmod.Path(FIG_DIR) / f"{doc_stem}_p{page.number + 1}_fig{i + 1}.png"
        figmod.crop_figure(page, bbox, path)
        inside = [tb for tb in text_blocks if bbox.x0 <= tb.bbox.cx <= bbox.x1 and bbox.y0 <= tb.bbox.cy <= bbox.y1]
        for tb in inside:
            used.add(id(tb))  # OCR'd label fragments become figure labels, not body text
        labels = [w for tb in inside for w in tb.text.split()]
        if do_ocr_labels:
            labels += figmod.ocr_labels(path)
        used.add(id(cap))
        figs.append(
            FigureBlock(bbox=bbox, image_path=str(path), caption=cap.text, figure_id=figmod.figure_id(cap.text), ocr_labels=sorted(set(labels)))
        )
    return figs, used


def _link_cross_page_captions(pages: list[Page]) -> None:
    for i in range(len(pages) - 1):
        orphans = [b for b in pages[i].blocks if b.kind == "figure" and not b.caption and b.bbox.y1 > 0.6 * pages[i].height]
        if not orphans:
            continue
        nxt = pages[i + 1]
        for b in list(nxt.blocks):
            if b.kind == "text" and b.role == "caption" and b.bbox.y0 < 0.25 * nxt.height:
                fig = orphans.pop(0)
                fig.caption = b.text
                fig.figure_id = figmod.figure_id(b.text)
                nxt.blocks.remove(b)
                if not orphans:
                    break

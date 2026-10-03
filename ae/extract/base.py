"""Extraction entry point. `parse(path)` is the only function ingest and the evaluation call.

PDF: native extraction (PyMuPDF text layer, pdfplumber tables, Tesseract OCR on pages
without a usable text layer) on every page; on the pages that needed OCR, Docling's layout
model adds the tables and figures it finds from pixels (`ae.extract.layout_model`).
DOCX: python-docx plus a LibreOffice render for page numbers (`ae.extract.native.docx`).

The extraction modules are imported on first use: they pull in PyMuPDF/pdfplumber/Tesseract
bindings and, for the layout pass, torch and the Docling models.
"""

from __future__ import annotations

from pathlib import Path

from ae.corpus import DOCUMENT_SUFFIXES
from ae.log import get_logger
from ae.schema import Block, FigureBlock, Page, ParsedDocument, TableBlock

log = get_logger(__name__)

DUPLICATE_FIGURE_IOU = 0.3
"""A Docling figure overlapping a native figure by more than this IoU is the same figure (native's is kept)."""


def parse(path: Path, use_cache: bool = True) -> ParsedDocument:
    """Extract one PDF or DOCX into the shared schema.

    `use_cache=False` re-parses PDF pages instead of reading the page cache. Raises
    ValueError for an unsupported file type.
    """
    path = Path(path)
    if path.suffix.lower() not in DOCUMENT_SUFFIXES:
        raise ValueError(
            f"{path.name}: cannot extract {path.suffix or 'extension-less'} files; expected {DOCUMENT_SUFFIXES}"
        )
    if path.suffix.lower() == ".docx":
        from ae.extract.native.docx import parse_docx

        doc = parse_docx(path)
    else:
        doc = parse_pdf(path, use_cache=use_cache)
    log.info(f"extracted {path.name}: {len(doc.pages)} pages, {sum(1 for p in doc.pages if p.is_scanned)} OCR pages")
    return doc


def parse_pdf(path: Path, use_cache: bool = True) -> ParsedDocument:
    """Native extraction on every page; Docling's tables and figures merged in on OCR'd pages.

    Measured on this corpus and on the OmniDocBench scanned slice: native's OCR text is far
    cleaner (scanned patent CER 0.13 % vs 21.8 % for Docling driving Tesseract, which
    duplicates regions), while Docling's layout model finds tables and figures from pixels
    that native cannot (TEDS 0.53 vs 0.0, 36/50 vs 0/50 figures). So native supplies the
    text and reading order of every page and Docling adds tables and figures on scanned
    pages. Native text blocks that fall inside a Docling table are dropped (the table holds
    them); Docling figures overlapping a native figure are dropped (native's caption linking
    is stronger). Both passes are page-cached, so the composition costs nothing on re-runs.
    """
    from ae.extract.native.pdf import parse_pdf as native_parse

    doc = native_parse(path, use_cache=use_cache)
    needs = [p.number for p in doc.pages if p.is_scanned]
    if needs:
        from ae.extract.layout_model import detect_layout

        layout = detect_layout(path, use_cache=use_cache)
        for page in doc.pages:
            if page.number in needs and page.number in layout:
                _merge_layout(page, layout[page.number])
    doc.meta["docling_layout_pages"] = needs
    return doc


def _merge_layout(page: Page, layout_page: Page) -> None:
    """Add Docling's tables and new figures to a native page in place and re-order its blocks."""
    from ae.extract.native.tables import inside_any_table
    from ae.extract.native.text import order_blocks

    tables = [b for b in layout_page.blocks if isinstance(b, TableBlock) and b.rows]
    figs = [b for b in layout_page.blocks if isinstance(b, FigureBlock)]
    own_figs = [b for b in page.blocks if isinstance(b, FigureBlock)]
    keep: list[Block] = [b for b in page.blocks if not (b.kind == "text" and inside_any_table(b.bbox, tables))]
    added: list[Block] = list(tables)
    for f in figs:
        if not any(f.bbox.iou(o.bbox) > DUPLICATE_FIGURE_IOU for o in own_figs):
            added.append(f)
    blocks = keep + added
    page.blocks = [b for b, _ in order_blocks(blocks, page.height)]

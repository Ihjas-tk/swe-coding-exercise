"""Backend registry. `parse(path, backend)` is the only entry point ingest uses.

Backend modules are imported on first use: the CLI imports this module for BACKENDS on
every command, and the backends pull in PyMuPDF/pdfplumber/Tesseract bindings (native)
or torch and the layout models (Docling).
"""

from __future__ import annotations

from pathlib import Path

from ae.corpus import DOCUMENT_SUFFIXES
from ae.log import get_logger
from ae.schema import Block, FigureBlock, Page, ParsedDocument, TableBlock

log = get_logger(__name__)

BACKENDS = ("native", "docling", "hybrid")
"""Valid `backend` names."""
HYBRID_DUPLICATE_FIGURE_IOU = 0.3
"""A Docling figure overlapping a native figure by more than this IoU is the same figure (native's is kept)."""


def parse(path: Path, backend: str = "native", use_cache: bool = True) -> ParsedDocument:
    """Extract one PDF or DOCX with the named backend into the shared schema.

    `use_cache=False` re-parses PDF pages instead of reading the page cache. Raises
    ValueError for an unknown backend or an unsupported file type.
    """
    path = Path(path)
    doc = _parse(path, backend, use_cache)
    log.info(
        f"extracted {path.name} ({backend}): {len(doc.pages)} pages, "
        f"{sum(1 for p in doc.pages if p.is_scanned)} OCR pages"
    )
    return doc


def _parse(path: Path, backend: str, use_cache: bool) -> ParsedDocument:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
    if path.suffix.lower() not in DOCUMENT_SUFFIXES:
        raise ValueError(
            f"{path.name}: cannot extract {path.suffix or 'extension-less'} files; expected {DOCUMENT_SUFFIXES}"
        )
    if path.suffix.lower() == ".docx":
        if backend == "docling":
            from ae.extract.docling_backend import parse_docx as docling_docx

            return docling_docx(path)
        from ae.extract.native.docx import parse_docx

        return parse_docx(path, backend_name=backend)  # native and hybrid share it
    if backend == "native":
        from ae.extract.native.pdf import parse_pdf as native_pdf

        return native_pdf(path, use_cache=use_cache)
    if backend == "docling":
        from ae.extract.docling_backend import parse_pdf as docling_pdf

        return docling_pdf(path, use_cache=use_cache)
    return parse_hybrid(path, use_cache=use_cache)


def parse_hybrid(path: Path, use_cache: bool = True) -> ParsedDocument:
    """Compose per page: native everywhere, Docling's tables and figures on OCR'd pages.

    Measured on this corpus and on the OmniDocBench scanned slice: native's OCR text is far
    cleaner (scanned patent CER 0.13 % vs 21.8 % for Docling driving Tesseract, which
    duplicates regions), while Docling's layout model finds tables and figures from pixels
    that native cannot (TEDS 0.53 vs 0.0, 36/50 vs 0/50 figures). So each is used where it
    measured best: native supplies the text and reading order of every page; Docling adds
    tables and figures on scanned pages. Native text blocks that fall inside a Docling table
    are dropped (the table holds them); Docling figures overlapping a native figure are
    dropped (native's caption linking is stronger). Both backends are page-cached, so the
    composition costs nothing on re-runs.
    """
    from ae.extract.native.pdf import parse_pdf as native_parse

    native_doc = native_parse(path, use_cache=use_cache)
    needs = [p.number for p in native_doc.pages if p.is_scanned]
    if needs:
        from ae.extract.docling_backend import parse_pdf as docling_parse

        dl = {p.number: p for p in docling_parse(path, use_cache=use_cache).pages}
        for page in native_doc.pages:
            if page.number in needs and page.number in dl:
                _merge_layout(page, dl[page.number])
    native_doc.backend = "hybrid"
    native_doc.meta["docling_layout_pages"] = needs
    return native_doc


def _merge_layout(page: Page, dl_page: Page) -> None:
    """Add Docling's tables and new figures to a native page in place and re-order its blocks."""
    from ae.extract.native.tables import inside_any_table
    from ae.extract.native.text import order_blocks

    tables = [b for b in dl_page.blocks if isinstance(b, TableBlock) and b.rows]
    figs = [b for b in dl_page.blocks if isinstance(b, FigureBlock)]
    own_figs = [b for b in page.blocks if isinstance(b, FigureBlock)]
    keep: list[Block] = [b for b in page.blocks if not (b.kind == "text" and inside_any_table(b.bbox, tables))]
    added: list[Block] = list(tables)
    for f in figs:
        if not any(f.bbox.iou(o.bbox) > HYBRID_DUPLICATE_FIGURE_IOU for o in own_figs):
            added.append(f)
    blocks = keep + added
    page.blocks = [b for b, _ in order_blocks(blocks, page.height)]
    page.backend = "hybrid"

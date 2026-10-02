"""Backend registry. `parse(path, backend)` is the only entry point ingest uses."""
from __future__ import annotations

from pathlib import Path

from ae.log import get_logger, timed
from ae.schema import ParsedDocument

BACKENDS = ("thin", "docling", "hybrid")
log = get_logger(__name__)


def parse(path: Path, backend: str = "thin", use_cache: bool = True) -> ParsedDocument:
    path = Path(path)
    with timed("extract", path.name, backend=backend) as t:
        doc = _parse(path, backend, use_cache)
        t["pages"] = len(doc.pages)
        t["ocr_pages"] = sum(1 for p in doc.pages if p.is_scanned)
    return doc


def _parse(path: Path, backend: str, use_cache: bool) -> ParsedDocument:
    if path.suffix.lower() == ".docx":
        if backend == "docling":
            from ae.extract.docling_backend import parse_docx as docling_docx

            return docling_docx(path, use_cache=use_cache)
        from ae.extract.thin.docx import parse_docx

        return parse_docx(path, use_cache=use_cache, backend_name=backend)  # thin and hybrid share it
    if backend == "thin":
        from ae.extract.thin.pdf import parse_pdf

        return parse_pdf(path, use_cache=use_cache)
    if backend == "docling":
        from ae.extract.docling_backend import parse_pdf

        return parse_pdf(path, use_cache=use_cache)
    if backend == "hybrid":
        return parse_hybrid(path, use_cache=use_cache)
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")


def parse_hybrid(path: Path, use_cache: bool = True) -> ParsedDocument:
    """Per-page composition: thin everywhere; on pages that needed OCR, Docling's table and
    figure blocks are merged into thin's page.

    Measured on this corpus and on the OmniDocBench scanned slice: thin's OCR text is far
    cleaner (scanned patent CER 0.13 % vs 21.8 % for Docling driving Tesseract, which
    duplicates regions), while Docling's layout model finds tables and figures from pixels
    that thin cannot (TEDS 0.53 vs 0.0, 36/50 vs 0/50 figures). So each is used where it
    measured best: thin supplies the text and reading order of every page; Docling adds
    tables and figures on scanned pages. Thin text blocks that fall inside a Docling table
    are dropped (the table holds them); Docling figures overlapping a thin figure are
    dropped (thin's caption linking is stronger). Both backends are page-cached, so the
    composition costs nothing on re-runs.
    """
    from ae.extract.thin.pdf import parse_pdf as thin_parse

    thin_doc = thin_parse(path, use_cache=use_cache)
    needs = [p.number for p in thin_doc.pages if p.is_scanned]
    if needs:
        from ae.extract.docling_backend import parse_pdf as docling_parse

        dl = {p.number: p for p in docling_parse(path, use_cache=use_cache).pages}
        for page in thin_doc.pages:
            if page.number in needs and page.number in dl:
                _merge_layout(page, dl[page.number])
    thin_doc.backend = "hybrid"
    thin_doc.meta["docling_layout_pages"] = needs
    return thin_doc


def _merge_layout(page, dl_page) -> None:
    from ae.extract.thin.text import order_blocks

    tables = [b for b in dl_page.blocks if b.kind == "table" and b.rows]
    figs = [b for b in dl_page.blocks if b.kind == "figure"]
    own_figs = [b for b in page.blocks if b.kind == "figure"]
    keep = []
    for b in page.blocks:
        if b.kind == "text" and any(t.bbox.x0 - 2 <= b.bbox.cx <= t.bbox.x1 + 2 and t.bbox.y0 - 2 <= b.bbox.cy <= t.bbox.y1 + 2 for t in tables):
            continue
        keep.append(b)
    added = list(tables)
    for f in figs:
        if not any(_iou(f.bbox, o.bbox) > 0.3 for o in own_figs):
            added.append(f)
    blocks = keep + added
    page.blocks = [b for b, _ in order_blocks(blocks, page.width, page.height)]
    page.backend = "hybrid"


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a.x0, b.x0), max(a.y0, b.y0), min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0

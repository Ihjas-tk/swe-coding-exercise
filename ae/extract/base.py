"""Backend registry. `parse(path, backend)` is the only entry point ingest uses."""
from __future__ import annotations

from pathlib import Path

from ae.schema import ParsedDocument

BACKENDS = ("thin", "docling", "hybrid")


def parse(path: Path, backend: str = "thin", use_cache: bool = True) -> ParsedDocument:
    path = Path(path)
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
    """Per-page routing: thin for pages with a usable text layer, Docling for the rest.

    Rationale (measured on this corpus, see README): the thin stack is faster and
    more faithful when PDF structure exists (glyphs, ruled tables, image bboxes,
    captions); a layout model earns its keep when there is none, i.e. on scans or
    on pages whose text layer fails the quality gate, where figures and tables must
    be found from pixels. Both backends are cached per page, so routing costs
    nothing extra on re-runs.
    """
    from ae.extract.thin.pdf import parse_pdf as thin_parse

    thin_doc = thin_parse(path, use_cache=use_cache)
    needs = [p.number for p in thin_doc.pages if p.is_scanned]
    if needs:
        from ae.extract.docling_backend import parse_pdf as docling_parse

        dl = {p.number: p for p in docling_parse(path, use_cache=use_cache).pages}
        thin_doc.pages = [dl.get(p.number, p) if p.number in needs else p for p in thin_doc.pages]
    thin_doc.backend = "hybrid"
    thin_doc.meta["routed_to_docling"] = needs
    return thin_doc

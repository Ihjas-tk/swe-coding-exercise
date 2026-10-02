"""End-to-end ingest: extract -> structured load -> numerals -> (VLM) -> chunks -> index (-> embeddings).

One index file per backend (`ae.index.store.index_db`). Every stage is cached upstream
(pages, renders, VLM descriptions, embeddings), so re-ingesting an unchanged corpus only
rebuilds the SQLite index.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from ae.corpus import DOCUMENT_SUFFIXES, STRUCTURED_SUFFIXES, corpus_files
from ae.extract.base import parse
from ae.extract.structured import load_structured
from ae.index.chunker import Chunk, chunk_document, structured_chunks
from ae.index.embed import embed_chunks
from ae.index.numerals import NumeralIndex, reconcile_labels
from ae.index.store import IndexStore, index_db
from ae.index.vlm import describe_figure, description_text
from ae.log import get_logger
from ae.schema import FigureBlock, Page, ParsedDocument

log = get_logger(__name__)


def ingest(
    backend: str = "native",
    files: list[Path] | None = None,
    use_cache: bool = True,
    embed_model: str | None = None,
    vlm: bool = False,
) -> dict[str, Any]:
    """Build (replace) the index for `backend` from `files` (default: the corpus) and return its stats.

    PDF/DOCX go through extraction, CSV/XLSX through the structured loader. `vlm=True`
    adds figure descriptions (needs ANTHROPIC_API_KEY; skipped with a warning otherwise);
    `embed_model` adds dense vectors. `use_cache=False` re-parses PDF pages.
    """
    files = files or corpus_files()
    db = index_db(backend)
    db.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"ingest backend={backend} files={len(files)} index={db} embed={embed_model or '-'} vlm={vlm}")
    docs = [parse(f, backend=backend, use_cache=use_cache) for f in files if f.suffix.lower() in DOCUMENT_SUFFIXES]
    structured = [f for f in files if f.suffix.lower() in STRUCTURED_SUFFIXES]
    if structured:
        load_structured(structured, db)
    numerals = NumeralIndex()
    for d in docs:
        numerals.add_document(d)
    numerals.save(db)
    if vlm:
        _describe_figures(docs, numerals, db)
    chunks: list[Chunk] = []
    for d in docs:
        chunks += chunk_document(d, numerals)
    if structured:
        chunks += structured_chunks(db)
    store = IndexStore(db)
    try:
        store.rebuild(chunks, backend)
        log.info(
            f"indexed {len(chunks)} chunks, {len(numerals.entries)} numerals, {len(numerals.figures)} figure defs -> {db}"
        )
        if embed_model:
            n = embed_chunks(store, embed_model)
            log.info(f"embedded {n} chunks with {embed_model}")
        return store.stats()
    finally:
        store.close()


def _describe_figures(docs: list[ParsedDocument], numerals: NumeralIndex, db: Path) -> None:
    """Fill FigureBlock.description from the VLM (cached) and record each description in the index.

    Stops the stage at the first figure the VLM cannot describe at all (no API key);
    figures whose calls failed are logged and skipped.
    """
    n = failed = 0
    with closing(sqlite3.connect(db)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS figure_descriptions "
            "(doc TEXT, page INTEGER, figure_id TEXT, image_path TEXT, json TEXT, PRIMARY KEY (doc, page, image_path))"
        )
        for d in docs:
            defined = numerals.defined(d.doc)
            for p in d.pages:
                for b in p.blocks:
                    if not isinstance(b, FigureBlock) or not b.image_path or not Path(b.image_path).exists():
                        continue
                    desc = describe_figure(b.image_path, b.caption, _figure_glossary(b, p, d.doc, numerals, defined))
                    if desc is None:
                        log.warning("VLM stage skipped (no API key)")
                        return
                    if desc.get("_failed"):
                        failed += 1
                        log.warning(f"figure description failed, skipped: {b.image_path}")
                        continue
                    b.description = description_text(desc)
                    conn.execute(
                        "INSERT OR REPLACE INTO figure_descriptions VALUES (?,?,?,?,?)",
                        (d.doc, p.number, b.figure_id, b.image_path, json.dumps(desc)),
                    )
                    n += 1
        conn.commit()
    log.info(f"described {n} figures with the VLM" + (f" ({failed} failed and skipped)" if failed else ""))


def _figure_glossary(figure: FigureBlock, page: Page, doc: str, numerals: NumeralIndex, defined: set[str]) -> list[str]:
    """'160 = control unit' lines for the numerals OCR confirmed in the figure plus those defined on its page."""
    rec = reconcile_labels(figure.ocr_labels, defined)
    present = rec["matched"] + rec["repaired"]
    same_page = [e.numeral for e in numerals.entries.values() if e.doc == doc and page.number in e.pages]
    return numerals.glossary(doc, list(dict.fromkeys(present + same_page)))

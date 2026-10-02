"""End-to-end ingest: extract -> structured load -> numerals -> chunks -> index (-> embeddings)."""
from __future__ import annotations

from pathlib import Path

from ae.cli import corpus_files
from ae.extract.base import parse
from ae.extract.structured import load_structured
from ae.index.chunker import chunk_document, structured_chunks
from ae.index.numerals import NumeralIndex
from ae.index.store import IndexStore, index_db


def ingest(backend: str = "thin", files: list[Path] | None = None, use_cache: bool = True, embed_model: str | None = None, log=print) -> dict:
    files = files or corpus_files()
    db = index_db(backend)
    db.parent.mkdir(parents=True, exist_ok=True)
    docs = []
    for f in files:
        if f.suffix.lower() in (".pdf", ".docx"):
            d = parse(f, backend=backend, use_cache=use_cache)
            docs.append(d)
            log(f"extracted {f.name}: {len(d.pages)} pages")
    structured = [f for f in files if f.suffix.lower() in (".csv", ".xlsx")]
    if structured:
        for t in load_structured(structured, db):
            log(f"loaded {t.doc} -> {t.name} ({t.n_rows} rows)")
    numerals = NumeralIndex()
    for d in docs:
        numerals.add_document(d)
    numerals.save(db)
    chunks = []
    for d in docs:
        chunks += chunk_document(d, numerals)
    chunks += structured_chunks(db) if structured else []
    store = IndexStore(db)
    store.rebuild(chunks, backend)
    log(f"indexed {len(chunks)} chunks, {len(numerals.entries)} numerals, {len(numerals.figures)} figure defs -> {db}")
    if embed_model:
        from ae.index.embed import embed_chunks

        n = embed_chunks(store, embed_model, log=log)
        log(f"embedded {n} chunks with {embed_model}")
    stats = store.stats()
    store.close()
    return stats

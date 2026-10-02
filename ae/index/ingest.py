"""End-to-end ingest: extract -> structured load -> numerals -> chunks -> index (-> embeddings)."""
from __future__ import annotations

from pathlib import Path

from ae.cli import corpus_files
from ae.extract.base import parse
from ae.extract.structured import load_structured
from ae.index.chunker import chunk_document, structured_chunks
from ae.index.numerals import NumeralIndex
from ae.index.store import IndexStore, index_db
from ae.log import TIMINGS, flush_timings, get_logger, timed, timing_table

log = get_logger(__name__)


def ingest(backend: str = "thin", files: list[Path] | None = None, use_cache: bool = True, embed_model: str | None = None, vlm: bool = False, log=None) -> dict:
    """Extract -> structured -> numerals -> (VLM) -> chunk -> index -> (embed). Every stage is
    timed per document (see data/logs/timings_<index>_<backend>.json and the table printed
    at the end)."""
    logger = globals()["log"]
    files = files or corpus_files()
    db = index_db(backend)
    db.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"ingest backend={backend} files={len(files)} index={db} embed={embed_model or '-'} vlm={vlm}")
    TIMINGS.clear()
    docs = []
    with timed("ingest.total"):
        for f in files:
            if f.suffix.lower() in (".pdf", ".docx"):
                d = parse(f, backend=backend, use_cache=use_cache)
                docs.append(d)
        structured = [f for f in files if f.suffix.lower() in (".csv", ".xlsx")]
        if structured:
            load_structured(structured, db)
        with timed("numerals") as t:
            numerals = NumeralIndex()
            for d in docs:
                numerals.add_document(d)
            numerals.save(db)
            t["numerals"] = len(numerals.entries)
        if vlm:
            _describe_figures(docs, numerals, db, logger)
        chunks = []
        for d in docs:
            with timed("chunk", d.doc) as t:
                cs = chunk_document(d, numerals)
                t["chunks"] = len(cs)
            chunks += cs
        if structured:
            with timed("chunk.structured_rows") as t:
                rows = structured_chunks(db)
                t["chunks"] = len(rows)
            chunks += rows
        store = IndexStore(db)
        with timed("index", chunks=len(chunks)):
            store.rebuild(chunks, backend)
        logger.info(f"indexed {len(chunks)} chunks, {len(numerals.entries)} numerals, {len(numerals.figures)} figure defs -> {db}")
        if embed_model:
            from ae.index.embed import embed_chunks

            n = embed_chunks(store, embed_model)
            logger.info(f"embedded {n} chunks with {embed_model}")
        stats = store.stats()
        store.close()
    from ae import config

    flush_timings(Path("data/logs") / f"timings_{config.INDEX or 'default'}_{backend}.json")
    table = timing_table()
    if table:
        logger.info("per-document timings (ms):\n" + table)
    stats["timings_table"] = table
    return stats


def _describe_figures(docs, numerals: NumeralIndex, db: Path, log) -> None:
    """Fill FigureBlock.description with a cached VLM description and persist the JSON."""
    import json
    import sqlite3

    from ae.index.numerals import reconcile_labels
    from ae.index.vlm import describe_figure, description_text

    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE IF NOT EXISTS figure_descriptions (doc TEXT, page INTEGER, figure_id TEXT, image_path TEXT, json TEXT, PRIMARY KEY (doc, page, image_path))")
    n = failed = 0
    for d in docs:
      defined = numerals.defined(d.doc)
      with timed("vlm", d.doc) as t:
        for p in d.pages:
            for b in p.blocks:
                if b.kind != "figure" or not b.image_path or not Path(b.image_path).exists():
                    continue
                rec = reconcile_labels(b.ocr_labels, defined)
                present = rec["matched"] + rec["repaired"]
                same_page = [e.numeral for e in numerals.entries.values() if e.doc == d.doc and p.number in e.pages]
                gl = numerals.glossary(d.doc, list(dict.fromkeys(present + same_page)))
                desc = describe_figure(b.image_path, b.caption, gl)
                if desc is None:
                    log.warning("VLM stage skipped (no API key)")
                    conn.close()
                    return
                if desc.get("_failed"):
                    failed += 1
                    log.warning(f"figure description failed, skipped: {b.image_path}")
                    continue
                b.description = description_text(desc)
                conn.execute("INSERT OR REPLACE INTO figure_descriptions VALUES (?,?,?,?,?)", (d.doc, p.number, b.figure_id, b.image_path, json.dumps(desc)))
                n += 1
        t["figures"] = sum(1 for p in d.pages for b in p.blocks if b.kind == "figure")
    conn.commit()
    conn.close()
    log.info(f"described {n} figures with the VLM" + (f" ({failed} failed and skipped)" if failed else ""))

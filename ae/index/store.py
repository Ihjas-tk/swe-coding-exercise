"""SQLite index: chunks + FTS5 keyword indexes + embedding blobs. One file per backend.

Keyword search uses three FTS5 tables because one table cannot mix tokenizers:
- chunks_fts  : prefix + text, `unicode61 remove_diacritics 2` with Porter stemming, for prose;
- chunks_ids  : the normalised identifier column, `unicode61 tokenchars '-'` WITHOUT stemming,
                so "fvt-esc-40a", "fvtesc40a", "450a", "fig2", "ref160" match exactly;
- chunks_tri  : the same identifier column with the trigram tokenizer, for partial or
                misspelled ids ("ESC-40", "8485576").
Queries are built as quoted terms joined with OR: a bare FTS5 MATCH means AND, and an
unquoted hyphen or dot is a syntax error or a NOT. bm25() returns lower-is-better, so
scores are negated. Each component's scores are normalised by that component's best
score before combining (prose 1.0, exact ids ID_WEIGHT, trigram TRI_WEIGHT): raw bm25
magnitudes differ between tables, and an un-normalised identifier bonus let every chunk
that merely mentions "EV-BMS-100" outrank the table row that answers the question.

Dense search is brute-force cosine over float32 blobs loaded into one numpy matrix per
model; fine to ~100k chunks, which is far beyond a few hundred pages.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ae import config
from ae.index.chunker import Chunk
from ae.index.identifiers import extract_identifiers
from ae.schema import BBox

INDEX_DIR = Path("data/index")
"""Root of the per-backend index files (namespaced by config.INDEX)."""
PROSE_WEIGHT = 1.0
"""Weight of the max-normalised prose BM25 component."""
ID_WEIGHT = 0.6
"""Weight of the exact-identifier component, relative to the best prose score."""
TRI_WEIGHT = 0.2
"""Weight of the trigram (partial identifier) component."""
PROSE_COLUMN_WEIGHTS = (0.5, 1.0)
"""bm25() weights of the chunks_fts columns (prefix, text): the prefix repeats on every chunk of a section."""
PROSE_CANDIDATE_FACTOR = 2
"""The prose component fetches k x this many candidates before fusion."""
TRIGRAM_MIN_CHARS = 3
"""The trigram tokenizer cannot match shorter identifiers."""
NORM_EPSILON = 1e-9
"""Added to vector norms so a zero vector never divides by zero."""
STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "is",
    "are",
    "what",
    "which",
    "how",
    "does",
    "do",
    "and",
    "or",
    "with",
    "by",
    "at",
    "it",
    "its",
    "this",
    "that",
    "be",
    "as",
    "from",
}
"""Question words dropped from the prose query (BM25 would otherwise reward them)."""


def index_db(backend: str, name: str | None = None) -> Path:
    """Path of a backend's index file within the index namespace (`name`, default config.INDEX)."""
    name = config.INDEX if name is None else name
    return (INDEX_DIR / name if name else INDEX_DIR) / f"{backend}.sqlite"


@dataclass
class Hit:
    """A scored chunk id from one search."""

    chunk_id: str
    score: float
    source: str  # "bm25" | "dense" | "ids"


class IndexStore:
    """Chunks, keyword indexes and embeddings in one SQLite file."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._matrix: dict[str, tuple[list[str], np.ndarray]] = {}
        self._numerals: set[str] | None = None
        self._ensure()

    # ------------------------------------------------------------------ schema
    def _ensure(self) -> None:
        """Create the tables and FTS indexes if this is a new file."""
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, doc TEXT, page INTEGER, kind TEXT, text TEXT, prefix TEXT,
                section TEXT, bbox TEXT, ids TEXT, meta TEXT, backend TEXT);
            CREATE INDEX IF NOT EXISTS chunks_doc_page ON chunks(doc, page);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(prefix, text, content='chunks', content_rowid='rowid',
                tokenize='porter unicode61 remove_diacritics 2');
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_ids USING fts5(ids, content='chunks', content_rowid='rowid',
                tokenize="unicode61 tokenchars '-'");
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_tri USING fts5(ids, content='chunks', content_rowid='rowid', tokenize='trigram');
            CREATE TABLE IF NOT EXISTS embeddings (chunk_id TEXT, model TEXT, dim INTEGER, vec BLOB, PRIMARY KEY (chunk_id, model));
            CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )

    # ------------------------------------------------------------------ writes
    def rebuild(self, chunks: list[Chunk], backend: str) -> None:
        """Replace all chunks and rebuild the FTS indexes; keeps embeddings of surviving chunks."""
        c = self.conn
        c.execute("DELETE FROM chunks")
        for t in ("chunks_fts", "chunks_ids", "chunks_tri"):
            c.execute(f"INSERT INTO {t}({t}) VALUES('delete-all')")
        for ch in chunks:
            c.execute(
                "INSERT INTO chunks(id, doc, page, kind, text, prefix, section, bbox, ids, meta, backend) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ch.id,
                    ch.doc,
                    ch.page,
                    ch.kind,
                    ch.text,
                    ch.prefix,
                    ch.section,
                    ch.bbox.model_dump_json() if ch.bbox else None,
                    " ".join(ch.identifiers),
                    json.dumps(ch.meta),
                    ch.backend or backend,
                ),
            )
        for t in ("chunks_fts", "chunks_ids", "chunks_tri"):
            c.execute(f"INSERT INTO {t}({t}) VALUES('rebuild')")
        # Keep embeddings of every model for chunks that still exist (ablations swap models).
        c.execute("DELETE FROM embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks)")
        self._matrix.clear()
        c.execute("INSERT OR REPLACE INTO index_meta VALUES ('backend', ?)", (backend,))
        c.execute("INSERT OR REPLACE INTO index_meta VALUES ('n_chunks', ?)", (str(len(chunks)),))
        c.commit()

    def add_embeddings(self, model: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        """Store vectors for chunk ids under a model name."""
        vectors = np.asarray(vectors, dtype=np.float32)
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
            [(cid, model, int(vectors.shape[1]), vectors[i].tobytes()) for i, cid in enumerate(chunk_ids)],
        )
        self.conn.execute("INSERT OR REPLACE INTO index_meta VALUES ('embedding_model', ?)", (model,))
        self.conn.commit()
        self._matrix.pop(model, None)

    # ------------------------------------------------------------------ reads
    def get(self, chunk_ids: list[str]) -> list[Chunk]:
        """Chunks by id, in the order given (missing ids are skipped)."""
        if not chunk_ids:
            return []
        q = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(f"SELECT * FROM chunks WHERE id IN ({q})", chunk_ids).fetchall()
        by_id = {r["id"]: self._to_chunk(r) for r in rows}
        return [by_id[i] for i in chunk_ids if i in by_id]

    def page_chunks(self, doc: str, page: int) -> list[Chunk]:
        """Chunks of one page in index order."""
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE doc = ? AND page = ? ORDER BY rowid", (doc, page)
        ).fetchall()
        return [self._to_chunk(r) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        """Every chunk in index order."""
        return [self._to_chunk(r) for r in self.conn.execute("SELECT * FROM chunks ORDER BY rowid")]

    def docs(self) -> list[str]:
        """Distinct document names."""
        return [r[0] for r in self.conn.execute("SELECT DISTINCT doc FROM chunks ORDER BY doc")]

    def has_identifier(self, ident: str) -> bool:
        """Return True when any chunk carries the identifier."""
        return (
            self.conn.execute("SELECT 1 FROM chunks_ids WHERE chunks_ids MATCH ? LIMIT 1", (f'"{ident}"',)).fetchone()
            is not None
        )

    @staticmethod
    def _to_chunk(r: sqlite3.Row) -> Chunk:
        return Chunk(
            id=r["id"],
            doc=r["doc"],
            page=r["page"],
            kind=r["kind"],
            text=r["text"],
            prefix=r["prefix"],
            section=r["section"],
            bbox=BBox.model_validate_json(r["bbox"]) if r["bbox"] else None,
            identifiers=(r["ids"] or "").split(),
            meta=json.loads(r["meta"] or "{}"),
            backend=r["backend"],
        )

    # ------------------------------------------------------------------ keyword search
    @staticmethod
    def _prose_query(question: str) -> str | None:
        """FTS5 query OR-ing the question's quoted content terms, or None when nothing is left."""
        terms = [
            t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.'/-]*", question.lower()) if t not in STOP and len(t) > 1
        ]
        terms = [t.replace('"', "") for t in terms]
        return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms)) if terms else None

    def known_numerals(self) -> set[str]:
        """Every reference numeral the index defines (cached)."""
        if self._numerals is None:
            try:
                self._numerals = {r[0] for r in self.conn.execute("SELECT DISTINCT numeral FROM numerals")}
            except sqlite3.OperationalError:
                self._numerals = set()
        return self._numerals

    def query_identifiers(self, question: str) -> list[str]:
        """Return the identifiers in a question, including bare reference numerals the index knows."""
        return extract_identifiers(question, self.known_numerals())

    def _component(self, sql: str, params: list[object], weight: float) -> dict[str, float]:
        """Run one bm25 query and return {chunk id: weight x score / best score}."""
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return {}
        best = max(-r["s"] for r in rows) or 1.0
        return {r["id"]: weight * (-r["s"]) / best for r in rows}

    def keyword_search(self, question: str, k: int = 50, doc: str | None = None) -> list[Hit]:
        """Top-k chunks by keyword score, optionally within one document.

        The score sums three max-normalised components: BM25 over prose (stemmed), exact
        identifier match (ID_WEIGHT) and trigram partial-identifier match (TRI_WEIGHT).
        """
        parts: list[dict[str, float]] = []
        pq = self._prose_query(question)
        if pq:
            parts.append(self._fts_component("chunks_fts", pq, doc, k * PROSE_CANDIDATE_FACTOR, PROSE_WEIGHT))
        idents = self.query_identifiers(question)
        if idents:
            iq = " OR ".join(f'"{i}"' for i in idents)
            parts.append(self._fts_component("chunks_ids", iq, doc, k, ID_WEIGHT))
            tq = " OR ".join(f'"{i}"' for i in idents if len(i) >= TRIGRAM_MIN_CHARS)
            if tq:
                parts.append(self._fts_component("chunks_tri", tq, doc, k, TRI_WEIGHT))
        scores: dict[str, float] = {}
        for comp in parts:
            for cid, sc in comp.items():
                scores[cid] = scores.get(cid, 0.0) + sc
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(cid, s, "bm25") for cid, s in ranked]

    def _fts_component(self, table: str, match: str, doc: str | None, limit: int, weight: float) -> dict[str, float]:
        """One keyword component: bm25 over FTS table `table` for query `match`, optionally within `doc`."""
        bm25 = (
            f"bm25({table}, {', '.join(map(str, PROSE_COLUMN_WEIGHTS))})" if table == "chunks_fts" else f"bm25({table})"
        )
        doc_filter = " AND c.doc = ?" if doc else ""
        sql = (
            f"SELECT c.id AS id, {bm25} AS s FROM {table} JOIN chunks c ON c.rowid = {table}.rowid "
            f"WHERE {table} MATCH ?{doc_filter} ORDER BY s LIMIT ?"
        )
        return self._component(sql, [match, *([doc] if doc else []), limit], weight)

    # ------------------------------------------------------------------ dense search
    def _load_matrix(self, model: str) -> tuple[list[str], np.ndarray]:
        """Chunk ids and their L2-normalised vectors for `model` (loaded once, then cached)."""
        if model not in self._matrix:
            rows = self.conn.execute("SELECT chunk_id, dim, vec FROM embeddings WHERE model = ?", (model,)).fetchall()
            ids = [r["chunk_id"] for r in rows]
            mat = (
                np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
                if rows
                else np.zeros((0, 1), dtype=np.float32)
            )
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + NORM_EPSILON
            self._matrix[model] = (ids, mat / norms)
        return self._matrix[model]

    def dense_search(self, query_vec: np.ndarray, model: str, k: int = 50, doc: str | None = None) -> list[Hit]:
        """Top-k chunks by cosine similarity to the query vector."""
        ids, mat = self._load_matrix(model)
        if not ids:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + NORM_EPSILON)
        sims = mat @ q
        order = np.argsort(-sims)
        out: list[Hit] = []
        allowed = None
        if doc:
            allowed = {r[0] for r in self.conn.execute("SELECT id FROM chunks WHERE doc = ?", (doc,))}
        for i in order:
            if allowed is not None and ids[i] not in allowed:
                continue
            out.append(Hit(ids[i], float(sims[i]), "dense"))
            if len(out) >= k:
                break
        return out

    def embedding_models(self) -> list[str]:
        """Models that have embeddings in this index."""
        return [r[0] for r in self.conn.execute("SELECT DISTINCT model FROM embeddings")]

    def stats(self) -> dict[str, Any]:
        """Index metadata plus chunk counts by kind and embeddings by model."""
        m = dict(self.conn.execute("SELECT key, value FROM index_meta"))
        m["kinds"] = dict(self.conn.execute("SELECT kind, COUNT(*) FROM chunks GROUP BY kind"))
        m["embeddings"] = dict(self.conn.execute("SELECT model, COUNT(*) FROM embeddings GROUP BY model"))
        return m

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

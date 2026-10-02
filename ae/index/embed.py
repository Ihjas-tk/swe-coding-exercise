"""Embeddings: local sentence-transformers models, cached per (model, text hash).

Default is ibm-granite/granite-embedding-english-r2 (149M params, 768-d, 8k context,
Apache-2.0, no query/passage prefixes): best retrieval and table-retrieval scores among
Apache base-size models in the research notes. BAAI/bge-small-en-v1.5 (33M, 384-d) is the
fast fallback and the model-swap ablation; it wants a query instruction prefix.

Dense retrieval's job in this system is paraphrase recall ("IP rating" vs "ingress
protection", "how hot can it run" vs "operating temperature range"). Exact values and
identifiers come from BM25; embedders score barely above chance on numeric detail.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ae.index.chunker import Chunk
from ae.index.store import IndexStore
from ae.log import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = get_logger(__name__)

DEFAULT_MODEL = "ibm-granite/granite-embedding-english-r2"
"""Model used when a caller names none (the CLI passes config.EMBED_MODEL explicitly)."""
CACHE_DB = Path("data/cache/embeddings.sqlite")
"""Vector cache shared by every index and backend (keyed on model + text)."""
SQL_BATCH = 500
"""Cache lookups per query; stays under SQLite's bound-parameter limit."""
CACHE_KEY_HEX_CHARS = 32
"""128 bits of SHA-256 per cached text."""


@dataclass(frozen=True)
class ModelSpec:
    """Per-model prefixes and sequence length."""

    name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512


SPECS = {
    "ibm-granite/granite-embedding-english-r2": ModelSpec(
        "ibm-granite/granite-embedding-english-r2", max_seq_length=1024
    ),
    "ibm-granite/granite-embedding-small-english-r2": ModelSpec(
        "ibm-granite/granite-embedding-small-english-r2", max_seq_length=1024
    ),
    "BAAI/bge-small-en-v1.5": ModelSpec(
        "BAAI/bge-small-en-v1.5", query_prefix="Represent this sentence for searching relevant passages: "
    ),
    "BAAI/bge-base-en-v1.5": ModelSpec(
        "BAAI/bge-base-en-v1.5", query_prefix="Represent this sentence for searching relevant passages: "
    ),
}

_models: dict[str, SentenceTransformer] = {}


def spec_for(name: str) -> ModelSpec:
    """Known spec for a model, or defaults."""
    return SPECS.get(name, ModelSpec(name))


def _model(name: str) -> SentenceTransformer:
    """Load a model once per process, on CPU."""
    if name not in _models:
        # Deferred: sentence-transformers imports torch (seconds); BM25-only runs never need it.
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(name, device="cpu")
        m.max_seq_length = spec_for(name).max_seq_length
        _models[name] = m
    return _models[name]


class EmbeddingCache:
    """SQLite cache of vectors keyed on (model, text hash)."""

    def __init__(self, path: Path = CACHE_DB) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (model TEXT, h TEXT, dim INTEGER, vec BLOB, PRIMARY KEY (model, h))"
        )

    @staticmethod
    def key(text: str) -> str:
        """Cache key for a text (model-independent; the model is a separate key column)."""
        return hashlib.sha256(text.encode()).hexdigest()[:CACHE_KEY_HEX_CHARS]

    def get_many(self, model: str, texts: list[str]) -> dict[str, np.ndarray]:
        """Return cached vectors for the texts that have one, keyed by text hash."""
        keys = [self.key(t) for t in texts]
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), SQL_BATCH):
            batch = keys[i : i + SQL_BATCH]
            q = ",".join("?" * len(batch))
            for h, vec in self.conn.execute(
                f"SELECT h, vec FROM cache WHERE model = ? AND h IN ({q})", [model, *batch]
            ):
                out[h] = np.frombuffer(vec, dtype=np.float32)
        return out

    def put_many(self, model: str, texts: list[str], vecs: np.ndarray) -> None:
        """Store vectors for the texts."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
            [
                (model, self.key(t), int(vecs.shape[1]), np.asarray(vecs[i], dtype=np.float32).tobytes())
                for i, t in enumerate(texts)
            ],
        )
        self.conn.commit()


def embed_texts(
    texts: list[str], model: str = DEFAULT_MODEL, kind: str = "passage", batch_size: int = 32
) -> np.ndarray:
    """Embed passages or queries (`kind` "passage" | "query") and return one row per text.

    The model's query/passage prefix is prepended before hashing, so cached vectors are
    only reused for the same role. Vectors are L2-normalised.
    """
    sp = spec_for(model)
    prefix = sp.query_prefix if kind == "query" else sp.passage_prefix
    inputs = [prefix + t for t in texts]
    cache = EmbeddingCache()
    have = cache.get_many(model, inputs)
    todo = [t for t in inputs if cache.key(t) not in have]
    if todo:
        log.info(f"embedding {len(todo)} new texts with {model} ({len(have)} cached)")
        vecs = _model(model).encode(
            todo, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        cache.put_many(model, todo, vecs)
        have.update({cache.key(t): np.asarray(vecs[i], dtype=np.float32) for i, t in enumerate(todo)})
    return np.stack([have[cache.key(t)] for t in inputs])


def embed_query(question: str, model: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed one question (with the model's query prefix)."""
    return embed_texts([question], model, kind="query")[0]


def embed_chunks(store: IndexStore, model: str = DEFAULT_MODEL) -> int:
    """Embed every chunk in the index (per document, cached) and store the vectors; returns the count."""
    chunks = store.all_chunks()
    if not chunks:
        return 0
    _model(model)
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc, []).append(c)
    total = 0
    for cs in by_doc.values():
        vecs = embed_texts([c.full_text for c in cs], model, kind="passage")
        store.add_embeddings(model, [c.id for c in cs], vecs)
        total += len(cs)
    return total

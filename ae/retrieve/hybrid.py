"""Hybrid retrieval: BM25 ∪ dense -> RRF(k=60) -> page scoring -> top pages with neighbours.

- Both lists are top-50; fused with reciprocal rank fusion, k=60 (safe default; a tuned
  convex blend needs labelled queries we don't have).
- Chunks containing a parsed identifier that exists in the index are always included
  (exact-ID guarantee).
- Pages are scored as best chunk + 0.1 x sum of the next three fused scores on the page, plus a soft
  boost when the question names the document. Citations are page-level, so pages are the
  unit returned; each page carries its hit chunks first, then same-page neighbours.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ae import config
from ae.index.chunker import Chunk
from ae.index.store import IndexStore
from ae.log import get_logger, timed
from ae.retrieve.query import ParsedQuery

log = get_logger(__name__)

RRF_K = 60
TOP_N = 50
MAX_PAGES = 8
MAX_CHUNKS_PER_PAGE = 10
DOC_BOOST = 0.004  # ~ one RRF rank step at the top


@dataclass
class PageHit:
    doc: str
    page: int
    score: float
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class RetrievalResult:
    pages: list[PageHit]
    fused: dict[str, float]  # chunk id -> fused score
    bm25_ids: list[str]
    dense_ids: list[str]
    gate: dict = field(default_factory=dict)  # signals for the not-found gate


def rrf(lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    s: dict[str, float] = defaultdict(float)
    for lst in lists:
        for r, cid in enumerate(lst):
            s[cid] += 1.0 / (k + r + 1)
    return dict(s)


def retrieve(store: IndexStore, pq: ParsedQuery, embed_model: str | None = None, mode: str = "hybrid", max_pages: int = MAX_PAGES) -> RetrievalResult:
    embed_model = embed_model or config.EMBED_MODEL
    bm25_ids: list[str] = []
    if mode in ("hybrid", "bm25"):
        with timed("query.retrieve.bm25", level=10):
            bm25_ids = [h.chunk_id for h in store.keyword_search(pq.question, k=TOP_N, doc=pq.doc)]
    dense_ids: list[str] = []
    if mode in ("hybrid", "dense") and embed_model in store.embedding_models():
        from ae.index.embed import embed_query

        with timed("query.retrieve.embed_query", level=10):
            qv = embed_query(pq.question, embed_model)
        with timed("query.retrieve.dense", level=10):
            dense_ids = [h.chunk_id for h in store.dense_search(qv, embed_model, k=TOP_N, doc=pq.doc)]
    elif mode in ("hybrid", "dense"):
        log.warning(f"no embeddings for {embed_model} in this index; running keyword-only")
    fused = rrf([lst for lst in (bm25_ids, dense_ids) if lst])
    # exact-identifier guarantee: anything the id index matches is in the candidate set
    for ident in pq.identifiers:
        if store.has_identifier(ident):
            for h in store.keyword_search(ident, k=20, doc=pq.doc):
                fused.setdefault(h.chunk_id, 1.0 / (RRF_K + TOP_N))
    chunks = {c.id: c for c in store.get(list(fused))}
    by_page: dict[tuple[str, int], list[tuple[float, Chunk]]] = defaultdict(list)
    for cid, sc in fused.items():
        c = chunks.get(cid)
        if c:
            by_page[(c.doc, c.page)].append((sc, c))
    pages: list[PageHit] = []
    for (doc, page), items in by_page.items():
        items.sort(key=lambda t: -t[0])
        score = items[0][0] + 0.1 * sum(s for s, _ in items[1:4])  # best + a little support; capped so 46 log rows can't flood a page
        if doc in pq.doc_boost:
            score += DOC_BOOST
        hits = [c for _, c in items][:MAX_CHUNKS_PER_PAGE]
        pages.append(PageHit(doc, page, score, hits))
    pages.sort(key=lambda p: -p.score)
    pages = pages[:max_pages]
    # neighbours: fill each page with its remaining chunks (reading order) up to the cap
    for p in pages:
        have = {c.id for c in p.chunks}
        for c in store.page_chunks(p.doc, p.page):
            if len(p.chunks) >= MAX_CHUNKS_PER_PAGE:
                break
            if c.id not in have:
                p.chunks.append(c)
                have.add(c.id)
    scores = sorted((v for v in fused.values()), reverse=True)
    gate = {
        "top1": scores[0] if scores else 0.0,
        "gap_1_5": (scores[0] - scores[4]) if len(scores) >= 5 else (scores[0] if scores else 0.0),
        "n_candidates": len(scores),
        "top_page_score": pages[0].score if pages else 0.0,
        "both_lists_agree": bool(bm25_ids and dense_ids and set(bm25_ids[:5]) & set(dense_ids[:5])),
    }
    return RetrievalResult(pages, fused, bm25_ids, dense_ids, gate)

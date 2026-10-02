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
from ae.log import get_logger
from ae.retrieve.query import ParsedQuery

log = get_logger(__name__)

RRF_K = 60  # the standard RRF constant; flattens rank differences so neither list dominates
TOP_N = 50  # depth of each ranked list; deeper lists only add noise to a ~300-chunk corpus
MAX_PAGES = 8  # pages handed to the generator: fits the prompt budget with neighbours attached
MAX_CHUNKS_PER_PAGE = 10  # hits + neighbours per page; long BOM/test-log pages would otherwise flood the prompt
DOC_BOOST = 0.004  # ~ one RRF rank step at the top: breaks ties toward the named document, never overrides relevance
SUPPORT_WEIGHT = 0.1  # weight of a page's supporting chunks relative to its best chunk
SUPPORT_CHUNKS = 3  # supporting chunks counted, so 46 log rows on one page cannot outscore one strong hit
ID_SEARCH_K = 20  # chunks fetched per exact identifier for the exact-ID guarantee
ID_FLOOR_SCORE = 1.0 / (RRF_K + TOP_N)  # an identifier-only hit ranks just below the last fused rank
GATE_GAP_RANK = 5  # the not-found gate compares the top fused score with the 5th
AGREEMENT_TOP = 5  # BM25 and dense "agree" when their top-5 lists share a chunk
RETRIEVAL_MODES = ("hybrid", "bm25", "dense")


@dataclass
class PageHit:
    """A retrieved page with its score and chunks (hits first, then neighbours)."""

    doc: str
    page: int
    score: float
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Ranked pages plus the raw rankings and not-found gate signals."""

    pages: list[PageHit]
    fused: dict[str, float]  # chunk id -> fused score
    bm25_ids: list[str]
    dense_ids: list[str]
    gate: dict = field(default_factory=dict)  # signals for the not-found gate


def rrf(lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal rank fusion of ranked id lists (insertion order follows first appearance)."""
    s: dict[str, float] = defaultdict(float)
    for lst in lists:
        for r, cid in enumerate(lst):
            s[cid] += 1.0 / (k + r + 1)
    return dict(s)


def retrieve(
    store: IndexStore, pq: ParsedQuery, embed_model: str | None = None, mode: str = "hybrid", max_pages: int = MAX_PAGES
) -> RetrievalResult:
    """BM25 and dense search fused with RRF, then scored and grouped by page.

    `mode` is an ablation switch (bm25 / dense / hybrid). An index without embeddings for
    `embed_model` silently degrades to keyword-only, with a warning.
    """
    bm25_ids, dense_ids = _rankings(store, pq, embed_model or config.EMBED_MODEL, mode)
    fused = rrf([lst for lst in (bm25_ids, dense_ids) if lst])
    _add_identifier_hits(store, pq, fused)
    pages = _score_pages(store, pq, fused)[:max_pages]
    _fill_neighbours(store, pages)
    gate = _gate_signals(fused, pages, bm25_ids, dense_ids)
    log.debug(f"retrieved bm25={len(bm25_ids)} dense={len(dense_ids)} fused={len(fused)} pages={len(pages)}")
    return RetrievalResult(pages, fused, bm25_ids, dense_ids, gate)


def _rankings(store: IndexStore, pq: ParsedQuery, embed_model: str, mode: str) -> tuple[list[str], list[str]]:
    """Top-N chunk ids from BM25 and from dense search (an empty list for a disabled side)."""
    bm25_ids: list[str] = []
    if mode in ("hybrid", "bm25"):
        bm25_ids = [h.chunk_id for h in store.keyword_search(pq.question, k=TOP_N, doc=pq.doc)]
    dense_ids: list[str] = []
    if mode in ("hybrid", "dense") and embed_model in store.embedding_models():
        from ae.index.embed import embed_query

        qv = embed_query(pq.question, embed_model)
        dense_ids = [h.chunk_id for h in store.dense_search(qv, embed_model, k=TOP_N, doc=pq.doc)]
    elif mode in ("hybrid", "dense"):
        log.warning(f"no embeddings for {embed_model} in this index; running keyword-only")
    return bm25_ids, dense_ids


def _add_identifier_hits(store: IndexStore, pq: ParsedQuery, fused: dict[str, float]) -> None:
    """Exact-identifier guarantee: anything the identifier index matches joins the candidate set."""
    for ident in pq.identifiers:
        if store.has_identifier(ident):
            for h in store.keyword_search(ident, k=ID_SEARCH_K, doc=pq.doc):
                fused.setdefault(h.chunk_id, ID_FLOOR_SCORE)


def _score_pages(store: IndexStore, pq: ParsedQuery, fused: dict[str, float]) -> list[PageHit]:
    """Group fused chunks by page and rank pages: best chunk + a little capped support + doc boost."""
    chunks = {c.id: c for c in store.get(list(fused))}
    by_page: dict[tuple[str, int], list[tuple[float, Chunk]]] = defaultdict(list)
    for cid, sc in fused.items():
        c = chunks.get(cid)
        if c:
            by_page[(c.doc, c.page)].append((sc, c))
    pages: list[PageHit] = []
    for (doc, page), items in by_page.items():
        items.sort(key=lambda t: -t[0])
        score = items[0][0] + SUPPORT_WEIGHT * sum(s for s, _ in items[1 : 1 + SUPPORT_CHUNKS])
        if doc in pq.doc_boost:
            score += DOC_BOOST
        hits = [c for _, c in items][:MAX_CHUNKS_PER_PAGE]
        pages.append(PageHit(doc, page, score, hits))
    pages.sort(key=lambda p: -p.score)
    return pages


def _fill_neighbours(store: IndexStore, pages: list[PageHit]) -> None:
    """Append each page's remaining chunks (reading order) after its hits, up to the per-page cap."""
    for p in pages:
        have = {c.id for c in p.chunks}
        for c in store.page_chunks(p.doc, p.page):
            if len(p.chunks) >= MAX_CHUNKS_PER_PAGE:
                break
            if c.id not in have:
                p.chunks.append(c)
                have.add(c.id)


def _gate_signals(fused: dict[str, float], pages: list[PageHit], bm25_ids: list[str], dense_ids: list[str]) -> dict:
    """Score-shape signals for the retrieval not-found gate (top score, top-vs-5th gap, list agreement)."""
    scores = sorted(fused.values(), reverse=True)
    top1 = scores[0] if scores else 0.0
    return {
        "top1": top1,
        "gap_1_5": (scores[0] - scores[GATE_GAP_RANK - 1]) if len(scores) >= GATE_GAP_RANK else top1,
        "n_candidates": len(scores),
        "top_page_score": pages[0].score if pages else 0.0,
        "both_lists_agree": bool(
            bm25_ids and dense_ids and set(bm25_ids[:AGREEMENT_TOP]) & set(dense_ids[:AGREEMENT_TOP])
        ),
    }

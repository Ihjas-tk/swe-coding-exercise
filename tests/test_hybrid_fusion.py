"""Reciprocal rank fusion and page scoring, with a fake keyword-only store (no index, no embeddings)."""

import pytest

from ae.index.chunker import Chunk
from ae.index.store import Hit
from ae.retrieve.hybrid import DOC_BOOST, MAX_CHUNKS_PER_PAGE, MAX_EVIDENCE_CHUNKS, RRF_K, TOP_N, retrieve, rrf
from ae.retrieve.query import ParsedQuery


def test_rrf_single_list_scores_by_rank():
    s = rrf([["a", "b", "c"]])
    assert s == pytest.approx({"a": 1 / 61, "b": 1 / 62, "c": 1 / 63})


def test_rrf_rewards_agreement_between_lists():
    s = rrf([["a", "b", "c"], ["c", "b", "x"]])
    order = sorted(s, key=lambda k: -s[k])
    assert order[:2] == ["c", "b"]  # c: 1/61 + 1/63 > b: 2/62 > a: 1/61
    assert s["c"] > s["b"] > s["a"] > s["x"]


def test_rrf_k_controls_rank_decay():
    assert rrf([["a", "b"]], k=0) == pytest.approx({"a": 1.0, "b": 0.5})


def test_rrf_empty():
    assert rrf([]) == {} and rrf([[]]) == {}


def chunk(cid: str, doc: str, page: int) -> Chunk:
    return Chunk(id=cid, doc=doc, page=page, kind="text", text=cid, prefix="")


class FakeStore:
    """Keyword-only store: a fixed ranking, an identifier lookup and per-page neighbours."""

    def __init__(self, ranking, chunks, id_hits=None):
        self.ranking = ranking
        self.chunks = {c.id: c for c in chunks}
        self.id_hits = id_hits or {}

    def keyword_search(self, question, k=50, doc=None):
        ids = self.id_hits.get(question, self.ranking)
        return [Hit(cid, 1.0, "bm25") for cid in ids if doc is None or self.chunks[cid].doc == doc][:k]

    def has_identifier(self, ident):
        return ident in self.id_hits

    def get(self, ids):
        return [self.chunks[i] for i in ids if i in self.chunks]

    def page_chunks(self, doc, page):
        return [c for c in self.chunks.values() if c.doc == doc and c.page == page]

    def embedding_models(self):
        return []  # no vectors: retrieval runs keyword-only


CHUNKS = [
    chunk("a1", "A.pdf", 1),
    chunk("a2", "A.pdf", 1),
    chunk("a3", "A.pdf", 1),
    chunk("a4", "A.pdf", 1),
    chunk("a5", "A.pdf", 1),
    chunk("a6", "A.pdf", 1),
    chunk("b1", "B.pdf", 2),
    chunk("b2", "B.pdf", 2),
    chunk("b9", "B.pdf", 2),  # never ranked: a neighbour
    chunk("c1", "C.pdf", 5),
]
RANKING = ["b1", "a1", "a2", "a3", "a4", "a5", "c1", "b2"]


def r(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def test_page_score_is_best_chunk_plus_tenth_of_next_three():
    res = retrieve(FakeStore(RANKING, CHUNKS), ParsedQuery("q"))
    pages = {(p.doc, p.page): p.score for p in res.pages}
    assert pages[("A.pdf", 1)] == pytest.approx(r(2) + 0.1 * (r(3) + r(4) + r(5)))  # a5 (rank 6) not counted
    assert pages[("B.pdf", 2)] == pytest.approx(r(1) + 0.1 * r(8))
    assert pages[("C.pdf", 5)] == pytest.approx(r(7))
    assert [(p.doc, p.page) for p in res.pages] == [("A.pdf", 1), ("B.pdf", 2), ("C.pdf", 5)]


def test_named_document_gets_a_soft_boost():
    res = retrieve(FakeStore(RANKING, CHUNKS), ParsedQuery("q", doc_boost=["C.pdf"]))
    c = next(p for p in res.pages if p.doc == "C.pdf")
    assert c.score == pytest.approx(r(7) + DOC_BOOST)


def test_page_lists_hits_first_then_neighbours():
    res = retrieve(FakeStore(RANKING, CHUNKS), ParsedQuery("q"))
    b = next(p for p in res.pages if p.doc == "B.pdf")
    assert [c.id for c in b.chunks] == ["b1", "b2", "b9"]
    assert all(len(p.chunks) <= MAX_CHUNKS_PER_PAGE for p in res.pages)
    assert sum(len(p.chunks) for p in res.pages) <= MAX_EVIDENCE_CHUNKS


def test_max_pages_limits_the_result():
    res = retrieve(FakeStore(RANKING, CHUNKS), ParsedQuery("q"), max_pages=1)
    assert [(p.doc, p.page) for p in res.pages] == [("A.pdf", 1)]


def test_exact_identifier_hit_is_always_a_candidate():
    store = FakeStore(["a1"], CHUNKS, id_hits={"fvt-esc-40a": ["c1"]})
    res = retrieve(store, ParsedQuery("q", identifiers=["fvt-esc-40a"]))
    assert res.fused["c1"] == pytest.approx(1.0 / (RRF_K + TOP_N))
    assert ("C.pdf", 5) in {(p.doc, p.page) for p in res.pages}


def test_gate_signals():
    res = retrieve(FakeStore(RANKING, CHUNKS), ParsedQuery("q"))
    assert res.gate["top1"] == pytest.approx(r(1))
    assert res.gate["gap_1_5"] == pytest.approx(r(1) - r(5))
    assert res.gate["n_candidates"] == len(RANKING)
    assert res.gate["both_lists_agree"] is False  # keyword-only run
    assert res.bm25_ids == RANKING and res.dense_ids == []


def test_empty_retrieval():
    res = retrieve(FakeStore([], CHUNKS), ParsedQuery("q"))
    assert res.pages == [] and res.gate["top1"] == 0.0 and res.gate["top_page_score"] == 0.0

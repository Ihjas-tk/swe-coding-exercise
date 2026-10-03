"""Query parsing/routing and keyword retrieval on the built index (no LLM calls)."""

import pytest

from ae.index.store import IndexStore, index_db

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not index_db("").exists(), reason="run `make ingest` first"),
]


@pytest.fixture(scope="module")
def store():
    s = IndexStore(index_db(""))
    yield s
    s.close()


def test_routes(store):
    from ae.retrieve.query import doc_aliases, parse_query

    al = doc_aliases(store)
    q = parse_query("What component is labelled 160 in FIG. 2 of US8,485,576 B2?", store, al)
    assert (
        q.route == "numeral"
        and q.numerals == ["160"]
        and q.fig_id == "FIG. 2"
        and q.doc == "US8485576B2_robotic_gripper.pdf"
    )
    q = parse_query("How many test runs failed in the test log?", store, al)
    assert q.route == "sql"
    q = parse_query("What is the maximum discharge current rating of the EV-BMS-100?", store, al)
    assert q.route == "hybrid" and q.doc is None and "EV-BMS-100_design_document.pdf" in q.doc_boost
    q = parse_query("How many M12 connector ports are shown on the enclosure in Figure 2?", store, al)
    assert q.visual


def test_keyword_search_hits_table_row(store):
    hits = store.keyword_search("maximum discharge current rating EV-BMS-100", k=5)
    top2 = store.get([h.chunk_id for h in hits[:2]])
    # the spec-table row ties with a test-log row for the same measurement; both are legitimate
    assert any(c.kind == "table_row" and "320 A" in c.text for c in top2)


def test_identifier_column(store):
    assert store.has_identifier("fvt-esc-40a") and store.has_identifier("fvtesc40a")
    assert not store.has_identifier("zzz-999")

"""Question parsing and routing against a duck-typed fake store (no SQLite index)."""

import pytest

from ae.index.chunker import Chunk
from ae.index.identifiers import extract_identifiers
from ae.retrieve.query import doc_aliases, parse_query

GRIPPER = "US8485576B2_robotic_gripper.pdf"
TRANSFER = "US2988237A_programmed_article_transfer.pdf"
UAV = "US20170320570A1_uav_vtol.pdf"
BMS = "EV-BMS-100_design_document.pdf"


class FakeStore:
    """Implements only what parse_query / doc_aliases call."""

    def __init__(self):
        self.numerals = {"160", "112", "18", "74"}
        self._docs = [GRIPPER, TRANSFER, UAV, BMS, "test_log.csv", "bill_of_materials.xlsx"]
        heading = "EV-BMS-100 Battery Management System"
        self._page1 = {
            BMS: [
                Chunk(
                    id="c",
                    doc=BMS,
                    page=1,
                    kind="text",
                    text=heading,
                    prefix="",
                    identifiers=extract_identifiers(heading),
                )
            ]
        }

    def docs(self):
        return self._docs

    def page_chunks(self, doc, page):
        return self._page1.get(doc, []) if page == 1 else []

    def known_numerals(self):
        return self.numerals

    def query_identifiers(self, question):
        return extract_identifiers(question, self.numerals)

    def has_identifier(self, ident):
        return False


@pytest.fixture(scope="module")
def store():
    return FakeStore()


@pytest.fixture(scope="module")
def aliases(store):
    return doc_aliases(store)


def ask(q, store, aliases):
    return parse_query(q, store, aliases)


def test_aliases_include_patent_number_variants_and_name_parts(aliases):
    assert {"us8485576", "us8485576b2", "8485576", "robotic", "gripper"} <= aliases[GRIPPER]
    assert {"us20170320570", "us20170320570a1"} <= aliases[UAV]
    assert "evbms100" in aliases[BMS]


def test_numeral_question_pins_patent_and_figure(store, aliases):
    q = ask("What component is labelled 160 in FIG. 2 of US8,485,576 B2?", store, aliases)
    assert (q.route, q.numerals, q.fig_id, q.doc) == ("numeral", ["160"], "FIG. 2", GRIPPER)


def test_patent_number_with_commas_pins_the_document(store, aliases):
    q = ask("What is element 112 in US 2,988,237?", store, aliases)
    assert q.doc == TRANSFER and q.route == "numeral"


def test_publication_number_pins_the_document(store, aliases):
    q = ask("What does 18 in FIG. 3 of US 2017/0320570 A1 denote?", store, aliases)
    assert (q.doc, q.numerals, q.fig_id) == (UAV, ["18"], "FIG. 3")


def test_patent_number_with_bare_kind_letter_pins_the_document(store, aliases):
    assert ask("What is the gate in US2988237A?", store, aliases).doc == TRANSFER


def test_suffixed_numeral_routes_even_when_unknown(store, aliases):
    assert ask("What is component 120a?", store, aliases).numerals == ["120a"]


def test_unknown_plain_numeral_does_not_route_to_numeral(store, aliases):
    q = ask("What is part 999 in the gripper?", store, aliases)
    assert q.route == "hybrid" and q.numerals == []


@pytest.mark.parametrize(
    "question",
    ["How many test runs failed in the test log?", "Which part has the highest unit cost in the BOM?"],
)
def test_aggregate_with_structured_vocabulary_routes_to_sql(store, aliases, question):
    assert ask(question, store, aliases).route == "sql"


def test_aggregate_without_structured_vocabulary_stays_hybrid(store, aliases):
    assert ask("How many claims does the gripper patent have in total?", store, aliases).route == "hybrid"


def test_product_name_is_a_soft_boost_not_a_pin(store, aliases):
    q = ask("What is the maximum discharge current rating of the EV-BMS-100?", store, aliases)
    assert q.route == "hybrid" and q.doc is None and q.doc_boost == [BMS]


def test_visual_question_about_a_figure_is_flagged(store, aliases):
    q = ask("How many M12 connector ports are shown on the enclosure in Figure 2?", store, aliases)
    assert q.visual and q.fig_id == "FIG. 2" and q.route == "hybrid"


def test_visual_words_without_a_figure_are_not_visual(store, aliases):
    assert not ask("What is the color of the housing?", store, aliases).visual


def test_aliases_are_computed_when_not_supplied(store):
    assert parse_query("What is element 112 in US 2,988,237?", store).doc == TRANSFER

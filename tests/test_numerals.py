"""Reference-numeral extraction, figure definitions and OCR label reconciliation on synthetic text."""

import pytest

from ae.index.numerals import NumeralIndex, extract_figure_defs, extract_numerals, reconcile_labels
from ae.schema import BBox, Page, ParsedDocument, TextBlock

PATENT_PAGES = [
    ["FIG. 1 is a perspective view of the gripper. FIG. 2 is a block diagram of the control system."],
    [
        "The gripper comprises a control unit 160 connected to valves 74 and 76. "
        "The wings 14, 16 are attached to the body. An electromagnet 120a holds the part.",
        "As recited in claim 1, the sensor samples at 10 Hz with a 16-bit converter. The control unit 160 is powered.",
    ],
]


def make_doc(pages: list[list[str]], name: str = "X.pdf", role: str = "body") -> ParsedDocument:
    return ParsedDocument(
        doc=name,
        source_path=name,
        backend="native",
        pages=[
            Page(
                number=i + 1,
                width=612,
                height=792,
                blocks=[
                    TextBlock(bbox=BBox(x0=72, y0=100 + 30 * j, x1=540, y1=120 + 30 * j), text=t, role=role)
                    for j, t in enumerate(texts)
                ],
            )
            for i, texts in enumerate(pages)
        ],
    )


@pytest.fixture(scope="module")
def entries():
    return {e.numeral: e for e in extract_numerals(make_doc(PATENT_PAGES))}


def test_noun_phrase_before_numeral_is_the_part_name(entries):
    assert entries["160"].name == "control unit"
    assert entries["160"].count == 2 and entries["160"].pages == [2]


def test_and_list_expands_to_each_numeral(entries):
    assert entries["74"].name == entries["76"].name == "valves"


def test_comma_list_expands_to_each_numeral(entries):
    assert entries["14"].name == entries["16"].name == "wings"


def test_single_occurrence_suffixed_numeral_is_kept(entries):
    assert entries["120a"].name == "electromagnet"


def test_claims_units_and_bit_widths_are_not_numerals(entries):
    assert not {"1", "10", "16-bit", "2"} & set(entries)
    assert set(entries) == {"14", "16", "74", "76", "120a", "160"}


def test_header_and_footer_blocks_are_ignored():
    assert extract_numerals(make_doc(PATENT_PAGES, role="header")) == []


def test_figure_definitions_are_extracted_with_page_and_title():
    defs = {f.fig_id: f for f in extract_figure_defs(make_doc(PATENT_PAGES))}
    assert set(defs) == {"FIG. 1", "FIG. 2"}
    assert defs["FIG. 2"].page == 1 and defs["FIG. 2"].title == "block diagram of the control system"


def test_first_figure_definition_wins():
    doc = make_doc([["FIG. 3 shows the hub."], ["FIG. 3 shows something else entirely."]])
    (f,) = extract_figure_defs(doc)
    assert f.title == "hub" and f.page == 1


def test_design_doc_without_figure_defs_keeps_only_repeated_or_suffixed_numerals():
    doc = make_doc(
        [
            [
                "The pack uses about 96 cells. Rev. C was released. The controller 12 sends data. The housing 40 holds the board.",
                "The bracket 7b is fixed. The controller 12 again. The connector 33 mates.",
            ]
        ]
    )
    assert {e.numeral: e.name for e in extract_numerals(doc)} == {"12": "controller", "7b": "bracket"}


def test_numeral_index_lookup_defined_and_glossary():
    idx = NumeralIndex()
    idx.add_document(make_doc(PATENT_PAGES))
    assert idx.lookup("X.pdf", "160").name == "control unit"
    assert idx.lookup("X.pdf", "999") is None
    assert idx.defined("X.pdf") == {"14", "16", "74", "76", "120a", "160"}
    assert idx.glossary("X.pdf", ["160", "999", "74"]) == ["160 = control unit", "74 = valves"]
    assert ("X.pdf", "FIG. 2") in idx.figures


def test_numeral_index_round_trips_through_sqlite(tmp_path):
    idx = NumeralIndex()
    idx.add_document(make_doc(PATENT_PAGES))
    db = tmp_path / "idx.sqlite"
    idx.save(db)
    loaded = NumeralIndex.load(db)
    assert loaded.entries == idx.entries and loaded.figures == idx.figures


DEFINED = {"160", "74", "11", "120", "220"}


def test_reconcile_exact_match():
    assert reconcile_labels(["160", "74."], DEFINED) == {"matched": ["160", "74"], "repaired": [], "unmatched": []}


def test_reconcile_repairs_dropped_leading_digit():
    assert reconcile_labels(["60"], DEFINED)["repaired"] == ["160"]


def test_reconcile_repairs_one_seven_confusion():
    assert reconcile_labels(["17"], DEFINED)["repaired"] == ["11"]


def test_reconcile_ambiguous_repair_is_unmatched():
    assert reconcile_labels(["20"], DEFINED) == {"matched": [], "repaired": [], "unmatched": ["20"]}
    assert reconcile_labels(["71"], {"11", "77"})["unmatched"] == ["71"]


def test_reconcile_skips_non_numeral_tokens_and_reports_unknowns():
    rec = reconcile_labels(["FIG.", "housing", "999"], DEFINED)
    assert rec == {"matched": [], "repaired": [], "unmatched": ["999"]}

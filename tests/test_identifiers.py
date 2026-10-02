"""Exact-identifier extraction shared by the chunker, query parser and verifier."""

import pytest

from ae.index.identifiers import extract_identifiers, norm, norm_hyphen, numbers_in


def test_part_number_is_stored_in_hyphenated_and_compact_forms():
    ids = extract_identifiers("Replace the FVT-ESC-40A with a BMS-CONN-M12-4.")
    assert {"fvt-esc-40a", "fvtesc40a", "bms-conn-m12-4", "bmsconnm124"} <= set(ids)


def test_identifiers_are_deduplicated_in_first_seen_order():
    ids = extract_identifiers("EV-BMS-100 and again EV-BMS-100")
    assert ids[:2] == ["ev-bms-100", "evbms100"]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See US 8,485,576 B2 for details", "us8485576b2"),
        ("Published as US2017/0320570A1.", "us20170320570a1"),
        ("US 2,988,237 describes the transfer", "us2988237"),
    ],
)
def test_patent_numbers_normalise_to_compact_form(text, expected):
    assert expected in extract_identifiers(text)


def test_patent_number_with_bare_kind_letter_is_recognised():
    assert "us2988237a" in extract_identifiers("What does US2988237A describe?")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("FIG. 2", "fig2"), ("Figure 3A", "fig3a"), ("FIGS. 4", "fig4"), ("fig 12", "fig12")],
)
def test_figure_ids_normalise_to_fig_prefix(text, expected):
    assert expected in extract_identifiers(text)


def test_value_with_unit_yields_compact_and_spaced_forms():
    ids = extract_identifiers("rated 320 A for 10 s at 3.3 V and 1,000 rpm")
    assert {"320a", "320 a", "10s", "10 s", "33v", "3.3 v", "1000rpm", "1,000 rpm"} <= set(ids)


def test_part_number_suffix_is_not_a_value_with_unit():
    ids = extract_identifiers("FVT-ESC-40A")
    assert "40a" not in ids and "40 a" not in ids


def test_reference_numerals_only_when_known():
    text = "the control unit 160 drives the electromagnet 120a"
    assert not any(i.startswith("ref") for i in extract_identifiers(text))
    ids = extract_identifiers(text, known_numerals={"160", "120a"})
    assert "ref160" in ids and "ref120a" in ids


def test_unknown_numerals_are_not_tagged():
    ids = extract_identifiers("valves 74 and 76", known_numerals={"74"})
    assert "ref74" in ids and "ref76" not in ids


def test_numerals_glued_to_part_numbers_or_units_are_excluded():
    ids = extract_identifiers("The EV-BMS-100 samples at 10 Hz", known_numerals={"100", "10"})
    assert "ref100" not in ids and "ref10" not in ids
    assert "ev-bms-100" in ids and "10hz" in ids


def test_text_without_identifiers_returns_empty_list():
    assert extract_identifiers("the quick brown fox") == []


def test_numbers_in_strips_thousands_separators_and_keeps_decimals():
    assert numbers_in("1,000 units at 4.25 USD, total 12,345.67") == {"1000", "4.25", "12345.67"}


def test_numbers_in_splits_ratios_and_glued_units():
    assert numbers_in("gear ratio 101:1 at 3.3V") == {"101", "1", "3.3"}


def test_numbers_in_empty_for_text_without_digits():
    assert numbers_in("no digits here") == set()


@pytest.mark.parametrize(
    ("raw", "compact", "hyphenated"),
    [
        ("FVT-ESC-40A", "fvtesc40a", "fvt-esc-40a"),
        ("US 8,485,576 B2", "us8485576b2", "us-8485576-b2"),
        ("BMS CONN/M12", "bmsconnm12", "bms-conn-m12"),
    ],
)
def test_norm_and_norm_hyphen(raw, compact, hyphenated):
    assert norm(raw) == compact
    assert norm_hyphen(raw) == hyphenated

"""Deterministic post-hoc answer checks with hand-built drafts and evidence."""

import pytest

from ae.answer.generate import Draft, EvidenceItem
from ae.answer.verify import verify
from ae.retrieve.query import ParsedQuery


def ev(eid: str, text: str, doc: str = "A.pdf", page: int = 1, kind: str = "table_row") -> EvidenceItem:
    return EvidenceItem(eid, None, text, doc, page, kind)


def draft(answer: str, citations: list[str], evidence: list[EvidenceItem], not_found=False, sufficient=True) -> Draft:
    return Draft(answer, citations, not_found, sufficient, "high", "", {}, evidence)


EVIDENCE = [
    ev("e1", "Max discharge current | 320 A continuous, 450 A for 10 s"),
    ev("e2", "Supply voltage range OV to 5 V", doc="B.pdf", page=3),
]
PQ = ParsedQuery("What is the maximum discharge current?")


def test_valid_citation_passes():
    v = verify(PQ, draft("The maximum discharge current is 320 A continuous.", ["e1"], EVIDENCE))
    assert v.ok and not v.not_found and [c.eid for c in v.citations] == ["e1"] and v.failures == []


def test_invalid_citation_id_is_dropped_with_warning():
    v = verify(PQ, draft("The maximum discharge current is 320 A continuous.", ["e1", "e9"], EVIDENCE))
    assert v.ok and [c.eid for c in v.citations] == ["e1"]
    assert any("e9" in w for w in v.warnings)


def test_only_invalid_citations_is_not_found():
    v = verify(PQ, draft("320 A continuous.", ["e9"], EVIDENCE))
    assert not v.ok and v.not_found and v.citations == [] and v.failures == ["no valid citations"]


def test_citations_outside_the_named_document_are_not_found():
    v = verify(ParsedQuery("q", doc="C.pdf"), draft("320 A continuous.", ["e1"], EVIDENCE))
    assert v.not_found and "named document C.pdf" in v.failures[0]


def test_named_document_keeps_only_in_scope_citations():
    v = verify(ParsedQuery("q", doc="A.pdf"), draft("320 A continuous.", ["e1", "e2"], EVIDENCE))
    assert v.ok and [c.eid for c in v.citations] == ["e1"]


def test_numeral_lookup_evidence_is_always_in_scope():
    lookup = ev("e4", "Reference numeral 160 = control unit", doc="A.pdf", kind="numeral_lookup")
    v = verify(ParsedQuery("What is labelled 160?", doc="C.pdf"), draft("It is the control unit.", ["e4"], [lookup]))
    assert v.ok


def test_number_missing_from_evidence_is_not_found():
    v = verify(PQ, draft("The maximum discharge current is 500 A.", ["e1"], EVIDENCE))
    assert v.not_found and v.failures == ["numbers not in cited evidence: ['500']"]


def test_numbers_from_the_question_need_no_grounding():
    lookup = ev("e4", "control unit", kind="numeral_lookup")
    assert verify(
        ParsedQuery("What is labelled 160?"), draft("Component 160 is the control unit.", ["e4"], [lookup])
    ).ok


@pytest.mark.parametrize("label", ["Table 7", "Figure 3", "FIG. 12", "page 9", "claim 4", "row 22"])
def test_label_numbers_in_the_answer_are_ignored(label):
    assert verify(PQ, draft(f"{label} lists 320 A continuous.", ["e1"], EVIDENCE)).ok


def test_ocr_letter_o_for_zero_in_evidence_is_tolerated():
    assert verify(PQ, draft("The supply ranges from 0 V to 5 V.", ["e2"], EVIDENCE)).ok


def test_thousands_separators_and_trailing_zeros_compare_equal():
    items = [ev("e5", "mass 1200 g, thickness 4.5 mm")]
    assert verify(PQ, draft("It weighs 1,200 g and is 4.50 mm thick.", ["e5"], items)).ok


def test_word_number_in_answer_matches_digit_in_evidence():
    items = [ev("e3", "The enclosure has 4 M12 ports")]
    assert verify(PQ, draft("The enclosure has four ports.", ["e3"], items)).ok


def test_digit_in_answer_matches_word_number_in_evidence():
    items = [ev("e3", "The enclosure has four M12 ports")]
    assert verify(PQ, draft("The enclosure has 4 ports.", ["e3"], items)).ok


def test_wrong_word_number_is_not_found():
    items = [ev("e3", "The enclosure has 4 M12 ports")]
    v = verify(PQ, draft("The enclosure has five ports.", ["e3"], items))
    assert v.not_found and "'5'" in v.failures[0]


def test_model_decline_is_passed_through():
    v = verify(PQ, draft("Not found in the provided documents.", [], EVIDENCE, not_found=True))
    assert v.ok and v.not_found and v.warnings == ["model declined"]
    assert verify(PQ, draft("maybe", ["e1"], EVIDENCE, sufficient=False)).not_found


def test_low_content_overlap_is_only_a_warning():
    v = verify(PQ, draft("Gearbox lubrication requires synthetic grease monthly.", ["e1"], EVIDENCE))
    assert v.ok and any("low content overlap" in w for w in v.warnings)

"""Answer-stage scoring helpers: numeric match, failure attribution and the summary."""

import pytest

from ae.answer.pipeline import Answer
from ae.eval.answers import attribute_failure, summarise
from ae.eval.scoring import nums_match, score_question


@pytest.mark.parametrize(
    ("reference", "answer"),
    [
        ("320 A", "The rating is 320 A."),
        ("320 A", "The rating is 322 A."),  # within 1 %
        ("1,000 rpm", "1000 rpm"),
        ("1000 rpm", "about 1,000 rpm"),
        ("0.5 mm", "0.504 mm"),
        ("There are 15 units", "fifteen units"),  # word numbers expanded on the answer side
        ("7 runs failed", "seven runs failed"),
        ("no numbers at all", "anything"),
        ("four ports", "6 ports"),  # reference word numbers are prose, not required values
    ],
)
def test_nums_match_accepts(reference, answer):
    assert nums_match(reference, answer)


@pytest.mark.parametrize(
    ("reference", "answer"),
    [
        ("320 A", "The rating is 330 A."),  # 3 % off
        ("7 and 18", "only 7"),  # every reference number is required
        ("15 units", "many units"),
        ("2.5 V", "25 V"),
    ],
)
def test_nums_match_rejects(reference, answer):
    assert not nums_match(reference, answer)


QUESTION = {
    "id": "q1",
    "type": "table",
    "question": "What is the max discharge current?",
    "reference_answer": "320 A",
    "evidence": [{"doc": "A.pdf", "page": 2}],
}


def test_score_question_without_grader():
    a = Answer(
        "320 A continuous.",
        [{"doc": "A.pdf", "page": 2}, {"doc": "A.pdf", "page": 3}],
        False,
        {"pages": [("A.pdf", 2, 0.03), ("B.pdf", 1, 0.02)], "route": "hybrid"},
    )
    row = score_question(QUESTION, a, use_grader=False)
    assert (row["retrieved"], row["page_hit"], row["nums_ok"], row["nf_ok"]) == (True, True, True, True)
    assert row["citation_precision"] == 0.5 and "correct" not in row


def test_score_question_unanswerable_decline_is_correct():
    q = {**QUESTION, "evidence": []}
    row = score_question(q, Answer("Not found in the provided documents.", [], True), use_grader=True)
    assert row["correct"] is True and row["nf_ok"] and row["retrieved"] is None and row["nums_ok"] is None


def test_score_question_declined_answerable_is_wrong_without_calling_the_grader():
    row = score_question(QUESTION, Answer("Not found.", [], True), use_grader=True)
    assert row["correct"] is False and row["grader"] == "declined" and row["nums_ok"] is False


ROW = {"correct": False, "page_hit": False, "retrieved": True, "not_found": False}
FACTUAL = {"type": "factual", "evidence": [{"doc": "A.pdf", "page": 1}]}
TABLE = {"type": "table", "evidence": [{"doc": "A.pdf", "page": 1}]}
EXTRACTION = {"tables": {"tables": [{"doc": "A.pdf", "cell_acc": 0.9}, {"doc": "B.pdf", "cell_acc": 1.0}]}}


@pytest.mark.parametrize(
    ("row", "q", "report", "stage"),
    [
        ({**ROW, "correct": True, "page_hit": True}, FACTUAL, None, None),
        ({**ROW, "correct": True}, FACTUAL, None, "citation"),
        ({**ROW, "retrieved": False}, FACTUAL, None, "retrieval"),
        ({**ROW, "retrieved": False, "not_found": True}, FACTUAL, None, "retrieval"),
        (ROW, TABLE, EXTRACTION, "extraction"),
        (ROW, {**TABLE, "evidence": [{"doc": "B.pdf", "page": 1}]}, EXTRACTION, "generation"),
        (ROW, TABLE, None, "generation"),
        ({**ROW, "not_found": True}, FACTUAL, None, "abstention"),
        (ROW, FACTUAL, EXTRACTION, "generation"),  # extraction errors only blame table questions
    ],
)
def test_failure_attribution(row, q, report, stage):
    assert attribute_failure(row, q, report) == stage


def dev(qtype, correct, page_hit=True, not_found=False, precision=1.0, stage=None):
    return {
        "set": "dev",
        "type": qtype,
        "correct": correct,
        "page_hit": page_hit,
        "not_found": not_found,
        "citation_precision": precision,
        "stage": stage,
    }


def adv(category, correct, not_found):
    return {"set": "adversarial", "type": "unanswerable", "category": category, "correct": correct, "not_found": not_found,
            "stage": None if correct else "abstention"}  # fmt: skip


@pytest.fixture
def rows():
    return [
        dev("factual", True),
        dev("factual", False, page_hit=False, precision=0.0, stage="retrieval"),  # answered wrongly
        dev("table", False, page_hit=False, not_found=True, precision=0.0, stage="abstention"),  # wrong decline
        dev("structured", True, precision=0.5),
        {"set": "dev", "type": "unanswerable", "correct": True, "not_found": True, "stage": None},
        adv("fake_part", True, True),
        adv("fake_part", False, False),
    ]


def test_summarise_answerable(rows):
    s = summarise(rows)["answerable"]
    assert s == {"n": 4, "correct": 2, "page_hit": 2, "wrong_decline": 1, "citation_precision": 0.375}


def test_summarise_by_type(rows):
    by_type = summarise(rows)["by_type"]
    assert set(by_type) == {"factual", "table", "structured"}
    assert by_type["factual"] == {"n": 2, "correct": 1, "page_hit": 1, "wrong_decline": 0}


def test_summarise_unanswerable_includes_dev_and_adversarial(rows):
    u = summarise(rows)["unanswerable"]
    assert u == {
        "n": 3,
        "correct": 2,
        "correct_decline": 2,
        "answered_wrongly": 1,
        "by_category": {"dev": "1/1", "fake_part": "1/2"},
    }


def test_penalty_score_counts_wrong_answers_double(rows):
    s = summarise(rows)
    # +2 correct, 0 for the wrong decline, -2 for the wrong answer, -2 for the answered unanswerable
    assert s["penalty_score"] == 2 - 2 - 2 and s["penalty_max"] == 4


def test_summarise_stage_counts(rows):
    assert summarise(rows)["stages"] == {
        "retrieval": 1,
        "extraction": 0,
        "citation": 0,
        "generation": 0,
        "abstention": 2,
    }


def test_summarise_empty():
    s = summarise([])
    assert s["answerable"]["n"] == 0 and s["answerable"]["citation_precision"] is None and s["penalty_score"] == 0

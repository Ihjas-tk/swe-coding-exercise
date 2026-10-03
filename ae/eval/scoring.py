"""The one per-question scorer shared by the quick harness (`ae eval --quick`), the full answer stage and smoke.

For a question in the dev-set format ({id, type, question, reference_answer, evidence}) and
the system's `Answer`:
- retrieved: a gold (doc, page) is among the retrieved pages (retrieval stage ok)
- page_hit: a cited (doc, page) is in the gold evidence
- nums_ok: every digit-written number in the reference appears in the answer within 1 %
- nf_ok: not_found matches whether the question is answerable
- correct (only with the grader): nums_ok AND the LLM grader says the answer states the
  reference's key facts. Unanswerable questions are correct when declined.
Fields that do not apply are None (retrieved/page_hit/nums_ok for unanswerable questions).
"""

from __future__ import annotations

import re

from ae import config
from ae.answer.pipeline import Answer
from ae.answer.verify import NUMBER_WORDS
from ae.index.identifiers import numbers_in
from ae.llm import complete, complete_json

NUM_TOLERANCE = 0.01  # relative; absorbs rounding in prose ("3.3 V" vs "3.30 V") without accepting other values
GRADER_MAX_TOKENS = 200

GRADER = """You grade an answer against a reference answer for an engineering question.
"correct" means the answer states the same key fact(s) as the reference (same values, units, names); extra correct detail is fine; a missing key value or a contradicting value is incorrect.
Reply with ONLY a JSON object on one line, no prose: {"correct": true, "reason": "..."}"""

# Spelled-out numbers the scorer accepts on the answer side ("fifteen" satisfies "15").
WORDS = {w: str(i) for i, w in enumerate(NUMBER_WORDS)}


def _close(a: str, b: str, tol: float = NUM_TOLERANCE) -> bool:
    """Numeric equality within a relative tolerance; non-numbers must match exactly."""
    try:
        x, y = float(a.replace(",", "")), float(b.replace(",", ""))
    except ValueError:
        return a == b
    return abs(x - y) <= tol * max(abs(y), 1e-9)


def _with_word_numbers(text: str) -> set[str]:
    """Digit-written numbers in the text plus the digits of spelled-out ones."""
    nums = set(numbers_in(text))
    for w, d in WORDS.items():
        if re.search(rf"\b{w}\b", text.lower()):
            nums.add(d)
    return nums


def nums_match(reference: str, answer: str) -> bool:
    """Check that every digit-written number in the reference appears in the answer within 1 %.

    Word numbers are expanded on the answer side only ("fifteen" satisfies "15"); a
    reference's "two in the wings, one in the fuselage" is prose, not a required value.
    """
    ref = set(numbers_in(reference))
    ans = _with_word_numbers(answer)
    return all(any(_close(a, r) for a in ans) for r in ref)


def grade(question: str, reference: str, answer: str) -> dict:
    """LLM-grade an answer against the reference.

    If the grader replies in prose instead of JSON, ask once more for a one-word verdict;
    if that fails too the answer counts as incorrect with `grader_failed` set.
    """
    try:
        out = complete_json(
            GRADER,
            f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}\n\nJSON:",
            model=config.GRADER_MODEL,
            max_tokens=GRADER_MAX_TOKENS,
        )
        return {"correct": bool(out.get("correct")), "reason": str(out.get("reason", ""))}
    except Exception:  # prose instead of JSON: fall back to a one-word verdict
        try:
            verdict = complete(
                GRADER.split("Reply with")[0] + "Answer with exactly one word: CORRECT or INCORRECT.",
                f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}",
                model=config.GRADER_MODEL,
                max_tokens=5,
                use_cache=False,
            )
            return {
                "correct": verdict.strip().upper().startswith("CORRECT"),
                "reason": f"one-word verdict: {verdict.strip()[:20]}",
            }
        except Exception as e:
            return {"correct": False, "reason": f"grader failed: {type(e).__name__}", "grader_failed": True}


def score_question(q: dict, a: Answer, use_grader: bool) -> dict:
    """Score one answered question; returns a row with the fields described in the module docstring.

    The grader is called only for answerable questions that were not declined.
    """
    answerable = bool(q["evidence"])
    gold = {(e["doc"], e["page"]) for e in q["evidence"]}
    cited = {(c["doc"], c["page"]) for c in a.citations}
    retrieved = {(d, p) for d, p, _ in a.debug.get("pages", [])}
    row: dict = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "reference": q["reference_answer"],
        "answer": a.answer,
        "citations": a.citations,
        "not_found": a.not_found,
        "route": a.debug.get("route"),
        "retrieved": bool(gold & retrieved) if answerable else None,
        "page_hit": bool(gold & cited) if answerable else None,
        "nums_ok": (not a.not_found and nums_match(q["reference_answer"], a.answer)) if answerable else None,
        "nf_ok": a.not_found == (not answerable),
        "verify": a.debug.get("verify"),
        "ms": a.debug.get("ms"),
    }
    if answerable:
        row["citation_precision"] = (len(gold & cited) / len(cited)) if cited else 0.0
    if use_grader:
        if not answerable:
            row["correct"] = a.not_found
        elif a.not_found:
            row.update(correct=False, grader="declined")
        else:
            g = grade(q["question"], q["reference_answer"], a.answer)
            row.update(correct=bool(row["nums_ok"] and g.get("correct")), grader=g.get("reason", ""))
    return row

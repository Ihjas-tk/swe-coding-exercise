"""Abstention and citation checks (deterministic; run after generation).

Order of gates, from the research notes:
1. scope gate (before retrieval): a named document or figure that is not in the corpus.
2. retrieval gate: top fused score / page score below a threshold -> not_found.
3. post-hoc checks, decline if a HARD check fails:
   - cited ids must be a subset of the supplied ids (hard; empty after filtering -> not_found)
   - cited evidence must come from the named document when one was named (hard)
   - every number in the answer must appear in the cited evidence (hard; word-numbers
     like "four" are compared to digits too)
   - answer content words should overlap the cited evidence (soft: lowers confidence)
Thresholds are configuration; the eval sets them on the dev set to cap wrong declines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ae.answer.generate import Draft, EvidenceItem
from ae.index.identifiers import numbers_in
from ae.retrieve.query import ParsedQuery

# Spelled-out numbers, shared with the evaluation scorer (ae.eval.scoring).
NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]
# The verifier expands only zero..twelve: the range it was tuned with on the dev set.
WORD_NUMS = {w: str(i) for i, w in enumerate(NUMBER_WORDS[:13])}
# Words ignored by the content-overlap check (function words and the decline phrase).
STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "is",
        "are",
        "and",
        "or",
        "with",
        "by",
        "at",
        "it",
        "its",
        "this",
        "that",
        "be",
        "as",
        "from",
        "which",
        "has",
        "have",
        "per",
        "not",
        "found",
        "provided",
        "documents",
    ]
)
# OCR confusions undone before comparing numbers: "l2"/"i2" -> "12" (l or i before a digit) ...
OCR_ONE_RE = re.compile(r"(?<![a-z])[li](?=\d)")
# ... and a standalone "o" (or one before a short unit such as "OV", "omm") -> "0".
OCR_ZERO_RE = re.compile(r"(?<![a-z])[o]((?=[a-z]{1,3}\b)|\b)")
# Numbers that only label a table/figure/page/claim/row in the answer are not facts to ground.
LABEL_NUMBER_RE = re.compile(
    r"\b(table|figure|fig\.?|figs\.?|page|p\.|row|item|claim|section|sheet|no\.)\s*\d+[a-z]?\b", re.I
)
# Content words for the overlap check: 4+ letters (hyphens allowed).
CONTENT_WORD_RE = re.compile(r"[a-z][a-z\-]{3,}")
MIN_CONTENT_OVERLAP = 0.4  # below this share of answer words found in the cited text, warn (soft check)
STRUCTURED_KINDS = ("numeral_lookup", "sql_result")  # evidence that is in scope whatever document was named


@dataclass
class Verdict:
    """Outcome of the post-hoc checks: kept citations, hard failures and soft warnings."""

    ok: bool
    not_found: bool
    citations: list[EvidenceItem]
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _norm_num(s: str) -> str:
    """Canonical form of a number string: no thousands separators, integral floats without ".0"."""
    s = s.replace(",", "")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def verify(pq: ParsedQuery, draft: Draft) -> Verdict:
    """Run the post-hoc checks on a draft; a failed hard check turns it into not_found.

    A model decline is passed through as ok + not_found (nothing to check).
    """
    if draft.not_found or not draft.sufficient:
        return Verdict(True, True, [], [], ["model declined"])
    failures: list[str] = []
    warnings: list[str] = []
    cited = _valid_citations(pq, draft, failures, warnings)
    if not cited:
        return Verdict(False, True, [], failures, warnings)
    cited_text = "\n".join(c.text for c in cited).lower()
    missing = _ungrounded_numbers(draft.answer, cited_text, pq.question)
    if missing:
        failures.append(f"numbers not in cited evidence: {missing}")
    overlap = _content_overlap(draft.answer, cited_text)
    if overlap is not None and overlap < MIN_CONTENT_OVERLAP:
        warnings.append(f"low content overlap with cited evidence ({overlap:.2f})")
    return Verdict(not failures, bool(failures), cited, failures, warnings)


def _valid_citations(pq: ParsedQuery, draft: Draft, failures: list[str], warnings: list[str]) -> list[EvidenceItem]:
    """Cited evidence after dropping unknown ids and (when a document is named) out-of-scope items.

    An empty result means a hard failure, recorded in `failures`.
    """
    by_id = {it.eid: it for it in draft.evidence}
    cited = [by_id[c] for c in draft.citations if c in by_id]
    bad = [c for c in draft.citations if c not in by_id]
    if bad:
        warnings.append(f"dropped invalid citation ids {bad}")
    if not cited:
        failures.append("no valid citations")
        return []
    if pq.doc:
        in_scope = [c for c in cited if c.doc == pq.doc or c.kind in STRUCTURED_KINDS]
        if not in_scope:
            failures.append(f"citations not from named document {pq.doc}")
        return in_scope
    return cited


def _ungrounded_numbers(answer: str, cited_text: str, question: str) -> list[str]:
    """Numbers in the answer that appear neither in the cited evidence nor in the question (sorted).

    Word numbers ("four") count on both sides; OCR letter/digit confusions in the evidence are undone.
    """
    cited_nums = {_norm_num(n) for n in numbers_in(cited_text)}
    ocr_fixed = OCR_ZERO_RE.sub("0", OCR_ONE_RE.sub("1", cited_text))
    cited_nums |= {_norm_num(n) for n in numbers_in(ocr_fixed)}
    ans_nums = {_norm_num(n) for n in numbers_in(LABEL_NUMBER_RE.sub(" ", answer))}
    for w, d in WORD_NUMS.items():
        if re.search(rf"\b{w}\b", answer.lower()):
            ans_nums.add(d)
        if re.search(rf"\b{w}\b", cited_text):
            cited_nums.add(d)
    question_nums = {_norm_num(x) for x in numbers_in(question)}
    return sorted(n for n in ans_nums if n not in cited_nums and n not in question_nums)


def _content_overlap(answer: str, cited_text: str) -> float | None:
    """Share of the answer's content words found in the cited text (None when it has none)."""
    words = [w for w in CONTENT_WORD_RE.findall(answer.lower()) if w not in STOP]
    if not words:
        return None
    return sum(1 for w in words if w in cited_text or w.rstrip("s") in cited_text) / len(words)

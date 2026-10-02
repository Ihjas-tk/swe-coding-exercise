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

WORD_NUMS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}
STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "is", "are", "and", "or", "with", "by", "at", "it", "its", "this", "that", "be", "as", "from", "which", "has", "have", "per", "not", "found", "provided", "documents"}


@dataclass
class Verdict:
    ok: bool
    not_found: bool
    citations: list[EvidenceItem]
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _norm_num(s: str) -> str:
    s = s.replace(",", "")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def verify(pq: ParsedQuery, draft: Draft, min_top_score: float = 0.0) -> Verdict:
    failures: list[str] = []
    warnings: list[str] = []
    by_id = {it.eid: it for it in draft.evidence}
    if draft.not_found or not draft.sufficient:
        return Verdict(True, True, [], [], ["model declined"])
    cited = [by_id[c] for c in draft.citations if c in by_id]
    bad = [c for c in draft.citations if c not in by_id]
    if bad:
        warnings.append(f"dropped invalid citation ids {bad}")
    if not cited:
        failures.append("no valid citations")
        return Verdict(False, True, [], failures, warnings)
    if pq.doc:
        in_scope = [c for c in cited if c.doc == pq.doc or c.kind in ("numeral_lookup", "sql_result")]
        if not in_scope:
            failures.append(f"citations not from named document {pq.doc}")
            return Verdict(False, True, [], failures, warnings)
        cited = in_scope
    cited_text = "\n".join(c.text for c in cited).lower()
    cited_nums = {_norm_num(n) for n in numbers_in(cited_text)}
    # OCR'd evidence may spell numbers with letters ("OV" for "0V", "l2" for "12"); accept
    # the answer's number if the evidence contains it after the usual confusions are undone.
    ocr_fixed = re.sub(r"(?<![a-z])[o]((?=[a-z]{1,3}\b)|\b)", "0", re.sub(r"(?<![a-z])[li](?=\d)", "1", cited_text))
    cited_nums |= {_norm_num(n) for n in numbers_in(ocr_fixed)}
    # Numbers that only label a table/figure/page/claim/row in the answer are not facts to ground.
    answer_wo_labels = re.sub(r"\b(table|figure|fig\.?|figs\.?|page|p\.|row|item|claim|section|sheet|no\.)\s*\d+[a-z]?\b", " ", draft.answer, flags=re.I)
    ans_nums = {_norm_num(n) for n in numbers_in(answer_wo_labels)}
    for w, d in WORD_NUMS.items():
        if re.search(rf"\b{w}\b", draft.answer.lower()):
            ans_nums.add(d)
        if re.search(rf"\b{w}\b", cited_text):
            cited_nums.add(d)
    missing = sorted(n for n in ans_nums if n not in cited_nums and n not in {_norm_num(x) for x in numbers_in(pq.question)})
    if missing:
        failures.append(f"numbers not in cited evidence: {missing}")
    words = [w for w in re.findall(r"[a-z][a-z\-]{3,}", draft.answer.lower()) if w not in STOP]
    if words:
        overlap = sum(1 for w in words if w in cited_text or w.rstrip("s") in cited_text) / len(words)
        if overlap < 0.4:
            warnings.append(f"low content overlap with cited evidence ({overlap:.2f})")
    return Verdict(not failures, bool(failures), cited, failures, warnings)

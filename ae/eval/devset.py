"""Run the dev set end to end and score answers (quick harness; the full staged eval builds on it).

Scoring per question:
- page_hit: any cited (doc, page) is in the gold evidence
- retrieved: gold page among the retrieved pages (retrieval stage ok)
- nums_ok: every number in the reference answer appears in the system answer (numeric tolerance 1%)
- nf_ok: not_found matches whether the question is answerable
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ae.answer.pipeline import Engine
from ae.index.identifiers import numbers_in


def _close(a: str, b: str, tol: float = 0.01) -> bool:
    try:
        x, y = float(a.replace(",", "")), float(b.replace(",", ""))
    except ValueError:
        return a == b
    return abs(x - y) <= tol * max(abs(y), 1e-9)


WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}


def _with_word_numbers(text: str) -> set[str]:
    nums = set(numbers_in(text))
    for w, d in WORDS.items():
        if re.search(rf"\b{w}\b", text.lower()):
            nums.add(d)
    return nums


def nums_match(reference: str, answer: str) -> bool:
    """Every digit-written number in the reference appears in the answer within 1 %.
    Word numbers are expanded on the answer side only ("fifteen" satisfies "15"); a
    reference's "two in the wings, one in the fuselage" is prose, not a required value."""
    ref = set(numbers_in(reference))
    ans = _with_word_numbers(answer)
    return all(any(_close(a, r) for a in ans) for r in ref)


def run(questions: Path = Path("dev_set/questions.json"), backend: str | None = None, mode: str = "hybrid", out: Path | None = None) -> list[dict]:
    eng = Engine(backend=backend, mode=mode)
    rows = []
    for q in json.load(open(questions)):
        a = eng.ask(q["question"], debug=True)
        gold = {(e["doc"], e["page"]) for e in q["evidence"]}
        cited = {(c["doc"], c["page"]) for c in a.citations}
        retrieved = {(d, p) for d, p, _ in a.debug.get("pages", [])}
        answerable = bool(q["evidence"])
        rows.append({
            "id": q["id"], "type": q["type"], "question": q["question"], "reference": q["reference_answer"], "answer": a.answer,
            "citations": a.citations, "not_found": a.not_found, "route": a.debug.get("route"),
            "retrieved": bool(gold & retrieved) if answerable else None,
            "page_hit": bool(gold & cited) if answerable else None,
            "nums_ok": nums_match(q["reference_answer"], a.answer) if answerable and not a.not_found else (None if not answerable else False),
            "nf_ok": a.not_found == (not answerable),
            "verify": a.debug.get("verify"), "ms": a.debug.get("ms"),
        })
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return rows


def summary(rows: list[dict]) -> str:
    def rate(key, subset):
        vals = [r[key] for r in subset if r[key] is not None]
        return f"{sum(vals)}/{len(vals)}" if vals else "-"

    lines = [f"{'type':11} {'n':>2} {'retrieved':>9} {'page_hit':>8} {'nums_ok':>7} {'nf_ok':>5}"]
    types = sorted({r["type"] for r in rows}, key=lambda t: ["factual", "table", "figure", "structured", "unanswerable"].index(t) if t in ["factual", "table", "figure", "structured", "unanswerable"] else 9)
    for t in types + ["all"]:
        sub = rows if t == "all" else [r for r in rows if r["type"] == t]
        lines.append(f"{t:11} {len(sub):>2} {rate('retrieved', sub):>9} {rate('page_hit', sub):>8} {rate('nums_ok', sub):>7} {rate('nf_ok', sub):>5}")
    return "\n".join(lines)

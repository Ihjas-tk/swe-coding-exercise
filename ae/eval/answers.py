"""Answer-stage evaluation: correctness, citation accuracy, not-found behaviour, attribution.

Correctness of an answerable question = numeric check (every number in the reference
appears in the answer within 1%) AND an LLM grader comparing answer to reference (the
grader sees question, reference and answer; it is asked only whether the answer conveys
the reference's facts). Free-text judgements were also read by hand for the dev set.

Not-found behaviour is measured on the dev set's unanswerable questions plus the 30
adversarial questions in gold/unanswerable.json (each paired with an answerable twin in
the dev set). A penalty score (+1 correct, 0 declined, -2 wrong) summarises the trade-off
the brief describes: a confident wrong answer is worse than no answer.

Failure attribution (per failed question): retrieval (gold page not retrieved),
extraction (the gold table/figure has known extraction errors), citation (answer right,
page convention differs), generation (evidence retrieved, answer wrong), abstention
(wrong decline / failed decline).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ae.answer.pipeline import Engine
from ae.eval.devset import nums_match
from ae.llm import complete_json

GRADER = """You grade an answer against a reference answer for an engineering question.
"correct" means the answer states the same key fact(s) as the reference (same values, units, names); extra correct detail is fine; a missing key value or a contradicting value is incorrect.
Reply with ONLY a JSON object on one line, no prose: {"correct": true, "reason": "..."}"""


def grade(question: str, reference: str, answer: str) -> dict:
    try:
        out = complete_json(GRADER, f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}\n\nJSON:", max_tokens=200)
        return {"correct": bool(out.get("correct")), "reason": str(out.get("reason", ""))}
    except Exception:  # noqa: BLE001  prose instead of JSON: fall back to a one-word verdict
        from ae.llm import complete

        try:
            verdict = complete(GRADER.split("Reply with")[0] + "Answer with exactly one word: CORRECT or INCORRECT.", f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}", max_tokens=5, use_cache=False)
            return {"correct": verdict.strip().upper().startswith("CORRECT"), "reason": f"one-word verdict: {verdict.strip()[:20]}"}
        except Exception as e:  # noqa: BLE001
            return {"correct": False, "reason": f"grader failed: {type(e).__name__}", "grader_failed": True}


def run(backend: str, mode: str = "hybrid", extraction_report: dict | None = None, questions: Path = Path("dev_set/questions.json"), adversarial: bool = True) -> dict:
    eng = Engine(backend=backend, mode=mode)
    dev = json.load(open(questions))
    unans = json.load(open("gold/unanswerable.json")) if adversarial else []
    rows = []
    for q in dev:
        a = eng.ask(q["question"], debug=True)
        answerable = bool(q["evidence"])
        gold = {(e["doc"], e["page"]) for e in q["evidence"]}
        cited = {(c["doc"], c["page"]) for c in a.citations}
        retrieved = {(d, p) for d, p, _ in a.debug.get("pages", [])}
        row = {"id": q["id"], "type": q["type"], "set": "dev", "question": q["question"], "reference": q["reference_answer"], "answer": a.answer, "citations": a.citations, "not_found": a.not_found, "route": a.debug.get("route"), "ms": a.debug.get("ms")}
        if answerable:
            nums = nums_match(q["reference_answer"], a.answer) if not a.not_found else False
            g = grade(q["question"], q["reference_answer"], a.answer) if not a.not_found else {"correct": False, "reason": "declined"}
            row.update(correct=bool(nums and g.get("correct")), grader=g.get("reason", ""), nums_ok=nums, retrieved=bool(gold & retrieved), page_hit=bool(gold & cited), citation_precision=(len(gold & cited) / len(cited)) if cited else 0.0)
            row["stage"] = _attribute(row, q, extraction_report)
        else:
            row.update(correct=a.not_found, retrieved=None, page_hit=None, stage=None if a.not_found else "abstention")
        rows.append(row)
    for u in unans:
        a = eng.ask(u["question"], debug=True)
        ok = a.not_found or (u.get("accept_zero") and re.search(r"\b(0|zero|no|none)\b", a.answer.lower()) is not None) or (u.get("accept_visual") and any(c["doc"].startswith("EV-BMS") for c in a.citations))
        rows.append({"id": u["id"], "type": "unanswerable", "set": "adversarial", "category": u["category"], "question": u["question"], "answer": a.answer, "citations": a.citations, "not_found": a.not_found, "route": a.debug.get("route"), "correct": bool(ok), "stage": None if ok else "abstention", "ms": a.debug.get("ms")})
    from ae.log import latency_summary

    s = summarise(rows)
    s["latency_ms"] = latency_summary()
    return {"backend": backend, "mode": mode, "rows": rows, "summary": s}


def _attribute(row: dict, q: dict, extraction_report: dict | None) -> str | None:
    if row["correct"] and row["page_hit"]:
        return None
    if row["correct"] and not row["page_hit"]:
        return "citation"
    if not row["retrieved"]:
        return "retrieval"
    if extraction_report and q["type"] == "table":
        for t in extraction_report.get("tables", {}).get("tables", []):
            if t["doc"] == q["evidence"][0]["doc"] and t["cell_acc"] < 1.0:
                return "extraction"
    if row["not_found"]:
        return "abstention"
    return "generation"


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["set"] == "dev" and r["type"] != "unanswerable"]
    un = [r for r in rows if r["type"] == "unanswerable"]
    by_type = {}
    for t in ("factual", "table", "figure", "structured"):
        sub = [r for r in ans if r["type"] == t]
        if sub:
            by_type[t] = {"n": len(sub), "correct": sum(r["correct"] for r in sub), "page_hit": sum(r["page_hit"] for r in sub), "wrong_decline": sum(r["not_found"] for r in sub)}
    n_un = len(un)
    declined = sum(r["not_found"] for r in un)
    wrong_answers = sum(1 for r in un if not r["correct"])
    score = sum(1 for r in ans if r["correct"]) + 0 * sum(r["not_found"] for r in ans) - 2 * sum(1 for r in ans if not r["correct"] and not r["not_found"]) - 2 * wrong_answers
    return {
        "answerable": {"n": len(ans), "correct": sum(r["correct"] for r in ans), "page_hit": sum(r["page_hit"] for r in ans), "wrong_decline": sum(r["not_found"] for r in ans), "citation_precision": round(sum(r["citation_precision"] for r in ans) / len(ans), 3) if ans else None},
        "by_type": by_type,
        "unanswerable": {"n": n_un, "correct": n_un - wrong_answers, "correct_decline": declined, "answered_wrongly": wrong_answers, "by_category": _by_cat(un)},
        "penalty_score": score, "penalty_max": len(ans),
        "stages": {s: sum(1 for r in rows if r.get("stage") == s) for s in ("retrieval", "extraction", "citation", "generation", "abstention")},
    }


def _by_cat(un: list[dict]) -> dict:
    out: dict[str, list[int]] = {}
    for r in un:
        c = r.get("category", "dev")
        out.setdefault(c, [0, 0])
        out[c][1] += 1
        out[c][0] += int(r["correct"])
    return {k: f"{v[0]}/{v[1]}" for k, v in out.items()}

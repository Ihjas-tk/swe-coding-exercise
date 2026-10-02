"""Answer-stage evaluation: correctness, citation accuracy, not-found behaviour, attribution.

Correctness of an answerable question = numeric check (every number in the reference
appears in the answer within 1%) AND an LLM grader comparing answer to reference (the
grader sees question, reference and answer; it is asked only whether the answer conveys
the reference's facts). Free-text judgements were also read by hand for the dev set.

Not-found behaviour is measured on the dev set's unanswerable questions plus the 30
adversarial questions in gold/unanswerable.json (each paired with an answerable twin in
the dev set). A penalty score (+1 correct, 0 declined, -2 wrong) summarises the trade-off
the brief describes: a confident wrong answer is worse than no answer.

Failure attribution (per failed question, see `attribute_failure`): retrieval, extraction,
citation, generation, abstention.

Per-question scoring is `ae.eval.scoring.score_question`, shared with the quick harness.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ae.answer.pipeline import Answer, Engine
from ae.eval.scoring import score_question
from ae.log import get_logger

log = get_logger(__name__)

# An adversarial "how many X failed" question may be answered "0"/"none" instead of declined.
ZERO_ANSWER_RE = re.compile(r"\b(0|zero|no|none)\b")
# Penalty score: a confident wrong answer costs twice what a correct one earns; a decline is neutral.
PENALTY_CORRECT, PENALTY_DECLINED, PENALTY_WRONG = 1, 0, -2
QUESTION_TYPES = ("factual", "table", "figure", "structured")
STAGES = ("retrieval", "extraction", "citation", "generation", "abstention")


def run(
    backend: str,
    mode: str = "hybrid",
    extraction_report: dict | None = None,
    questions: Path = Path("dev_set/questions.json"),
    adversarial: bool = True,
) -> dict:
    """Answer the question file (and the adversarial set) on one backend and score every question."""
    eng = Engine(backend=backend, mode=mode)
    dev = json.loads(questions.read_text())
    unans = json.loads(Path("gold/unanswerable.json").read_text()) if adversarial else []
    log.info(f"answer stage: backend={backend} questions={len(dev)} adversarial={len(unans)}")
    rows = []
    for q in dev:
        row = {"set": "dev", **score_question(q, eng.ask(q["question"]), use_grader=True)}
        row["stage"] = attribute_failure(row, q, extraction_report) if q["evidence"] else _abstention_stage(row)
        rows.append(row)
    rows += [_adversarial_row(u, eng.ask(u["question"])) for u in unans]
    return {"backend": backend, "mode": mode, "rows": rows, "summary": summarise(rows)}


def _adversarial_row(u: dict, a: Answer) -> dict:
    """Score an adversarial unanswerable question: a decline is correct, as are the accepted alternatives.

    `accept_zero`: "0"/"none" is a correct answer; `accept_visual`: an answer citing the
    EV-BMS document (the figure does show it) is accepted.
    """
    ok = (
        a.not_found
        or (u.get("accept_zero") and ZERO_ANSWER_RE.search(a.answer.lower()) is not None)
        or (u.get("accept_visual") and any(c["doc"].startswith("EV-BMS") for c in a.citations))
    )
    return {
        "id": u["id"],
        "type": "unanswerable",
        "set": "adversarial",
        "category": u["category"],
        "question": u["question"],
        "answer": a.answer,
        "citations": a.citations,
        "not_found": a.not_found,
        "route": a.debug.get("route"),
        "correct": bool(ok),
        "stage": None if ok else "abstention",
        "ms": a.debug.get("ms"),
    }


def _abstention_stage(row: dict) -> str | None:
    """Stage of a dev-set unanswerable question: None when declined, else abstention."""
    return None if row["not_found"] else "abstention"


def attribute_failure(row: dict, q: dict, extraction_report: dict | None) -> str | None:
    """Attribute a scored answerable question to the first pipeline stage that explains its failure.

    Checked in order, so each failure lands in exactly one stage:
    - None: correct and a gold page cited;
    - citation: correct answer, but no gold page cited (page convention differs);
    - retrieval: no gold page among the retrieved pages;
    - extraction: a table question whose gold document has a table with cell accuracy < 1
      in the extraction report (the evidence itself was wrong);
    - abstention: retrieved but declined (wrong decline);
    - generation: retrieved and answered, but wrong.
    """
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
    """Aggregate per-question rows into the answer-stage summary."""
    ans = [r for r in rows if r["set"] == "dev" and r["type"] != "unanswerable"]
    un = [r for r in rows if r["type"] == "unanswerable"]
    by_type: dict[str, dict] = {}
    for t in QUESTION_TYPES:
        sub = [r for r in ans if r["type"] == t]
        if sub:
            by_type[t] = {
                "n": len(sub),
                "correct": sum(r["correct"] for r in sub),
                "page_hit": sum(r["page_hit"] for r in sub),
                "wrong_decline": sum(r["not_found"] for r in sub),
            }
    n_un = len(un)
    declined = sum(r["not_found"] for r in un)
    wrong_answers = sum(1 for r in un if not r["correct"])
    wrong_on_answerable = sum(1 for r in ans if not r["correct"] and not r["not_found"])
    score = (
        PENALTY_CORRECT * sum(1 for r in ans if r["correct"])
        + PENALTY_DECLINED * sum(r["not_found"] for r in ans)
        + PENALTY_WRONG * (wrong_on_answerable + wrong_answers)
    )
    return {
        "answerable": {
            "n": len(ans),
            "correct": sum(r["correct"] for r in ans),
            "page_hit": sum(r["page_hit"] for r in ans),
            "wrong_decline": sum(r["not_found"] for r in ans),
            "citation_precision": round(sum(r["citation_precision"] for r in ans) / len(ans), 3) if ans else None,
        },
        "by_type": by_type,
        "unanswerable": {
            "n": n_un,
            "correct": n_un - wrong_answers,
            "correct_decline": declined,
            "answered_wrongly": wrong_answers,
            "by_category": _by_cat(un),
        },
        "penalty_score": score,
        "penalty_max": len(ans),
        "stages": {s: sum(1 for r in rows if r.get("stage") == s) for s in STAGES},
    }


def _by_cat(un: list[dict]) -> dict:
    """Unanswerable questions handled correctly per adversarial category ("dev" for the dev set's own)."""
    out: dict[str, list[int]] = {}
    for r in un:
        c = r.get("category", "dev")
        out.setdefault(c, [0, 0])
        out[c][1] += 1
        out[c][0] += int(r["correct"])
    return {k: f"{v[0]}/{v[1]}" for k, v in out.items()}

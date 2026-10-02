"""Quick harness (`ae eval --quick`, `ae smoke`): answer a question file and score it numerically.

Uses the shared scorer (`ae.eval.scoring.score_question`) without the LLM grader, so the
only LLM calls are the answers themselves. The full staged evaluation (`ae eval`) adds the
grader, the adversarial set and failure attribution on top of the same scorer.
"""

from __future__ import annotations

import json
from pathlib import Path

from ae.answer.pipeline import Engine
from ae.eval.scoring import score_question

TYPE_ORDER = ("factual", "table", "figure", "structured", "unanswerable")


def run(
    questions: Path = Path("dev_set/questions.json"),
    backend: str | None = None,
    mode: str = "hybrid",
    out: Path | None = None,
) -> list[dict]:
    """Answer every question in the file and return one scored row per question (also written to `out`)."""
    eng = Engine(backend=backend, mode=mode)
    rows = [score_question(q, eng.ask(q["question"]), use_grader=False) for q in json.loads(questions.read_text())]
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return rows


def summary(rows: list[dict]) -> str:
    """Render per-type pass rates as a fixed-width text table."""

    def rate(key: str, subset: list[dict]) -> str:
        vals = [r[key] for r in subset if r[key] is not None]
        return f"{sum(vals)}/{len(vals)}" if vals else "-"

    lines = [f"{'type':11} {'n':>2} {'retrieved':>9} {'page_hit':>8} {'nums_ok':>7} {'nf_ok':>5}"]
    types = sorted({r["type"] for r in rows}, key=lambda t: TYPE_ORDER.index(t) if t in TYPE_ORDER else len(TYPE_ORDER))
    for t in [*types, "all"]:
        sub = rows if t == "all" else [r for r in rows if r["type"] == t]
        lines.append(
            f"{t:11} {len(sub):>2} {rate('retrieved', sub):>9} {rate('page_hit', sub):>8} {rate('nums_ok', sub):>7} {rate('nf_ok', sub):>5}"
        )
    return "\n".join(lines)

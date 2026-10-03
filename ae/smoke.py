"""`ae smoke`: a 5-minute end-to-end check on one file of each type.

Builds data/smoke/corpus (scanned patent, born-digital patent, design-doc PDF, DOCX, CSV,
XLSX), ingests it into the `smoke` index (data/index/smoke/), and answers the
dev-set questions whose evidence lies in those files. Meant as the first thing a reviewer
runs after `make check`; `make eval` is the full run.

What fails the smoke run (non-zero exit): a crash, a wrong number (a number of the
reference answer missing from the answer), a wrong decline (an answerable question
declined) or a missed decline (an unanswerable question answered). Cited pages are
reported as information only: two dev questions (q02, q07) cite a page the dev set does
not list although the answer is right (README, failure analysis).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ae import config
from ae.log import get_logger

log = get_logger(__name__)

ROOT = Path("data/smoke")
INDEX_NAME = "smoke"
PICKS = [
    "patents/US2988237A_programmed_article_transfer.pdf",  # scanned -> OCR path + Docling layout pass
    "patents/US8485576B2_robotic_gripper.pdf",  # born-digital two-column + numerals
    "design_docs/EV-BMS-100_design_document.pdf",  # ruled spec tables, cross-page caption
    "design_docs/RJA-40_actuator_design_document.docx",  # DOCX + LibreOffice page mapping
    "structured/test_log.csv",
    "structured/bill_of_materials.xlsx",
]
KNOWN_PAGE_DISAGREEMENTS = {"q02", "q07"}  # right answer, page the dev set does not list (README failure analysis)


@dataclass
class SmokeResult:
    """Outcome of a smoke run: index size and one scored row per question (empty without an API key)."""

    chunks: int
    questions: int
    rows: list[dict] = field(default_factory=list)
    answered: bool = False  # False when no API key: ingest-only smoke

    @property
    def failures(self) -> dict[str, str]:
        """Question id -> reason, for wrong numbers and wrong or missed declines."""
        out: dict[str, str] = {}
        for r in self.rows:
            if not r["nf_ok"]:
                out[r["id"]] = "wrong decline" if r["not_found"] else "answered an unanswerable question"
            elif r["nums_ok"] is False:
                out[r["id"]] = "wrong numbers"
        return out


def run() -> SmokeResult:
    """Ingest the smoke corpus into the `smoke` index and answer the matching dev-set questions."""
    corpus = ROOT / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for rel in PICKS:
        src = Path(rel)
        if src.exists():
            shutil.copy(src, corpus / src.name)
    docs = {Path(p).name for p in PICKS}
    config.INDEX = INDEX_NAME
    from ae.corpus import corpus_files
    from ae.index.ingest import ingest

    log.info(f"smoke: ingesting {len(PICKS)} files")
    stats = ingest(files=corpus_files(corpus=corpus), embed_model=config.EMBED_MODEL, vlm=bool(config.api_key()))
    dev = json.loads(Path("dev_set/questions.json").read_text())
    qs = [q for q in dev if not q["evidence"] or all(e["doc"] in docs for e in q["evidence"])]
    qfile = ROOT / "questions.json"
    qfile.write_text(json.dumps(qs, indent=1))
    result = SmokeResult(sum(stats.get("kinds", {}).values()), len(qs))
    if config.api_key():
        from ae.eval.devset import run as run_dev

        log.info(f"smoke: answering {len(qs)} dev-set questions")
        result.rows = run_dev(questions=qfile, out=ROOT / "results.json")
        result.answered = True
    return result


def render(r: SmokeResult) -> str:
    """Plain-text summary: counts by question type, page hits as information, then the verdict."""
    lines = [f"smoke: {r.chunks} chunks indexed, {r.questions} questions"]
    if not r.answered:
        lines.append("no ANTHROPIC_API_KEY: ingest-only smoke (put the key in .env to answer questions)")
        return "\n".join(lines)
    from ae.eval.devset import TYPE_ORDER

    for t in [t for t in TYPE_ORDER if any(row["type"] == t for row in r.rows)]:
        sub = [row for row in r.rows if row["type"] == t]
        ok = sum(1 for row in sub if row["id"] not in r.failures)
        lines.append(f"  {t:12} {ok}/{len(sub)} ok")
    answerable = [row for row in r.rows if row["page_hit"] is not None]
    misses = sorted(row["id"] for row in answerable if not row["page_hit"])
    known = [m for m in misses if m in KNOWN_PAGE_DISAGREEMENTS]
    note = f" (known gold-page disagreements: {', '.join(known)})" if known else ""
    lines.append(
        f"page hits (information only): {len(answerable) - len(misses)}/{len(answerable)}"
        + (f"; other page cited for {', '.join(misses)}{note}" if misses else "")
    )
    fails = r.failures
    lines.append(
        "result: OK" if not fails else "result: FAILED " + "; ".join(f"{k}: {v}" for k, v in sorted(fails.items()))
    )
    return "\n".join(lines)

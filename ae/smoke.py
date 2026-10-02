"""`ae smoke`: a 5-minute end-to-end check on one file of each type.

Builds data/smoke/corpus (scanned patent, born-digital patent, design-doc PDF, DOCX, CSV,
XLSX), ingests it into the `smoke` index with the default backend, asks one question, and
scores the dev-set questions whose evidence lies in those files. Meant as the first thing
a reviewer runs after `make check`; `make eval` is the full run.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ae import config
from ae.log import get_logger, timing_table

log = get_logger(__name__)

PICKS = [
    "patents/US2988237A_programmed_article_transfer.pdf",  # scanned -> OCR path (+ Docling layout in hybrid)
    "patents/US8485576B2_robotic_gripper.pdf",  # born-digital two-column + numerals
    "design_docs/EV-BMS-100_design_document.pdf",  # ruled spec tables, cross-page caption
    "design_docs/RJA-40_actuator_design_document.docx",  # DOCX + LibreOffice page mapping
    "structured/test_log.csv",
    "structured/bill_of_materials.xlsx",
]


def run(backend: str | None = None) -> dict:
    backend = backend or config.BACKEND
    root = Path("data/smoke")
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for rel in PICKS:
        src = Path(rel)
        if src.exists():
            shutil.copy(src, corpus / src.name)
    docs = {Path(p).name for p in PICKS}
    config.INDEX = "smoke"
    from ae.cli import corpus_files
    from ae.index.ingest import ingest

    t0 = time.time()
    stats = ingest(backend=backend, files=corpus_files(corpus=corpus), embed_model=config.EMBED_MODEL, vlm=bool(config.api_key()))
    t_ingest = time.time() - t0
    log.info(f"smoke ingest done in {t_ingest:.0f} s: {stats.get('kinds')}")
    qs = [q for q in json.load(open("dev_set/questions.json")) if not q["evidence"] or all(e["doc"] in docs for e in q["evidence"])]
    qfile = root / "questions.json"
    qfile.write_text(json.dumps(qs, indent=1))
    result = {"ingest_s": round(t_ingest), "chunks": sum(stats.get("kinds", {}).values()), "questions": len(qs), "timings": timing_table()}
    if config.api_key():
        from ae.eval.devset import run as run_dev, summary

        t1 = time.time()
        rows = run_dev(questions=qfile, backend=backend, out=root / "results.json")
        result["eval_s"] = round(time.time() - t1)
        result["summary"] = summary(rows)
        result["failures"] = [r["id"] for r in rows if not (r["nums_ok"] in (True, None) and r["page_hit"] in (True, None) and r["nf_ok"])]
    else:
        result["summary"] = "no ANTHROPIC_API_KEY: ingest-only smoke"
    return result

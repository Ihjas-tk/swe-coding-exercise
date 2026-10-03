"""Run every evaluation stage and write data/eval/RESULTS.md (+ JSON) with one summary table.

Also renders the external question-set report (`ae eval-external` -> data/eval/EXTERNAL.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from ae import config
from ae.eval import answers as answers_eval
from ae.eval import extraction as extraction_eval
from ae.eval import omnidocbench as odb_eval
from ae.eval import retrieval as retrieval_eval
from ae.log import get_logger

log = get_logger(__name__)
OUT = Path("data/eval")


def run_all(with_answers: bool = True, with_external: bool = True) -> dict:
    """Run every stage on the active index and write data/eval/RESULTS.md and results.json.

    A missing index is built first. The OmniDocBench stage runs when its extraction dumps
    exist (see `ae.eval.omnidocbench`).
    """
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_index(config.EMBED_MODEL)
    report: dict = {"embed_model": config.EMBED_MODEL, "external": None, "answers": None}
    log.info("stage 1/4 extraction")
    report["extraction"] = extraction_eval.evaluate()
    if with_external and odb_eval.available():
        log.info("stage 2/4 OmniDocBench slice")
        report["external"] = odb_eval.evaluate()
    log.info("stage 3/4 retrieval")
    report["retrieval"] = retrieval_eval.recall_at_k(config.EMBED_MODEL)
    if with_answers:
        log.info("stage 4/4 answers (LLM calls are cached)")
        report["answers"] = answers_eval.run(extraction_report=report["extraction"])
    (OUT / "results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    (OUT / "RESULTS.md").write_text(to_markdown(report))
    log.info(f"wrote {OUT / 'RESULTS.md'} and {OUT / 'results.json'}")
    return report


def ensure_index(embed_model: str, require_embeddings: bool = True) -> None:
    """Ingest when the active index is missing, empty, or (if required) lacks embeddings for `embed_model`.

    Ingest embeds with `embed_model`, and adds figure descriptions (VLM) when an API key is
    set, as `make ingest` does.
    """
    from ae.index.ingest import ingest
    from ae.index.store import IndexStore, index_db

    db = index_db()
    have = False
    if db.exists():
        s = IndexStore(db)
        have = bool(s.stats()["kinds"]) and (not require_embeddings or embed_model in s.embedding_models())
        s.close()
    if have:
        log.info(f"index found: {db}")
    else:
        log.info(f"index missing or unembedded: running ingest first ({db})")
        ingest(embed_model=embed_model, vlm=bool(config.api_key()))


def run_external(questions: Path, out: Path) -> list[str]:
    """Score an external question set on the active index; writes `out` (Markdown) and its .json; returns the lines."""
    report = answers_eval.run(questions=questions, adversarial=False)
    lines = external_markdown(report, questions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    return lines


def external_markdown(report: dict, questions: Path) -> list[str]:
    """Render the external evaluation: one summary table (14 lines incl. header) and the failure list."""
    s = report["summary"]
    lines = [f"# External evaluation ({questions})", "", "| Metric | Value |", "|---|---|"]

    def row(label: str, value: object) -> None:
        lines.append(f"| {label} | {value} |")

    row("Answerable questions", s["answerable"]["n"])
    row("Correct (numbers within 1 % AND grader)", s["answerable"]["correct"])
    row("Cited page in gold evidence", s["answerable"]["page_hit"])
    row("Wrong declines", s["answerable"]["wrong_decline"])
    row("Unanswerable handled correctly", f"{s['unanswerable']['correct']}/{s['unanswerable']['n']}")
    for t in answers_eval.QUESTION_TYPES:
        bt = s["by_type"].get(t, {})
        row(f"Correct, {t}", f"{bt.get('correct', '-')}/{bt.get('n', '-')}")
    row("Failures by stage", s["stages"])
    lines += [
        "",
        "## Failures",
        "",
        "| id | type | route | answer | stage | grader |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        if r.get("stage"):
            lines.append(
                f"| {r['id']} | {r['type']} | {r.get('route')} | {r['answer'][:80]} | {r['stage']} | {str(r.get('grader', ''))[:80]} |"
            )
    return lines


def to_markdown(r: dict) -> str:
    """Render the evaluation report as Markdown: summary table, retrieval, answers, extraction detail."""
    out: list[str] = ["# Evaluation results", ""]
    out += _summary_table(r)
    out += _retrieval_section(r)
    out += _answer_section(r)
    out += _extraction_section(r)
    return "\n".join(out)


Metric = tuple[str, object]  # row label, cell value


def _summary_table(r: dict) -> list[str]:
    """Render the one-table summary: one row per metric."""
    out = ["## Summary (one table)", "", "| Metric | Value |", "|---|---|"]
    for name, value in _extraction_metrics(r) + _retrieval_metrics(r) + _answer_metrics(r):
        out.append(f"| {name} | {value} |")
    out.append("")
    return out


def _extraction_metrics(r: dict) -> list[Metric]:
    """Summary rows for the gold extraction set and (when run) the OmniDocBench slice."""
    ex = r["extraction"]
    figs = ex["figures"]
    rows: list[Metric] = [
        ("Extraction: OCR CER (scanned page)", _first(ex["ocr"], "cer")),
        ("Extraction: OCR WER (scanned page)", _first(ex["ocr"], "wer")),
        ("Extraction: table cell accuracy (4 tables)", f"{ex['tables']['cell_acc']:.3f}"),
        ("Extraction: figures found / caption linked (9)", f"{figs['found']} / {figs['caption_linked']}"),
        ("Extraction: figure label recall, OCR / VLM", f"{figs['label_recall_ocr']} / {figs['label_recall_vlm']}"),
        (
            "Extraction: reading order pair accuracy (3 pages)",
            _avg([v["pair_order_acc"] for v in ex["reading_order"].values() if v["pair_order_acc"] is not None]),
        ),
    ]
    xo = r["external"]
    if xo:
        rows += [
            ("OmniDocBench (106 scanned pages): text similarity", xo.get("text_similarity", "-")),
            ("OmniDocBench: reading order pair accuracy", xo.get("reading_order_pair_acc", "-")),
            ("OmniDocBench: table TEDS (19 tables)", xo.get("table_teds", "-")),
            ("OmniDocBench: figure recall @IoU0.5 (50)", xo.get("figure_recall", "-")),
        ]
    return rows


def _retrieval_metrics(r: dict) -> list[Metric]:
    """Summary rows: overall page Recall@1 and @5."""
    rec = r["retrieval"]["recall"].get("all", {})
    return [(f"Retrieval: page Recall@{k}", rec.get(f"R@{k}", "-")) for k in (1, 5)]


def _answer_metrics(r: dict) -> list[Metric]:
    """Summary rows for the answer stage (empty when it was skipped)."""
    if not r["answers"]:
        return []
    an = r["answers"]["summary"]
    a, u = an["answerable"], an["unanswerable"]
    return [
        ("Answers: correct (20 answerable)", f"{a['correct']}/{a['n']}"),
        ("Answers: cited page correct", f"{a['page_hit']}/{a['n']}"),
        ("Answers: citation precision", a["citation_precision"]),
        ("Answers: wrong declines on answerable", a["wrong_decline"]),
        ("Not-found: handled correctly (2 dev + 30 adversarial)", f"{u['correct']}/{u['n']}"),
        ("Not-found: explicit declines / answered wrongly", f"{u['correct_decline']} / {u['answered_wrongly']}"),
        ("Penalty score (+1 / 0 / -2)", f"{an['penalty_score']} of {an['penalty_max']}"),
    ]


def _retrieval_section(r: dict) -> list[str]:
    """Page Recall@k by question type."""
    x = r["retrieval"]
    model = (x["embed_model"] or "").split("/")[-1]
    out = [
        f"## Retrieval: page Recall@k by question type (BM25 ∪ dense {model}, RRF)",
        "",
        "| type | n | R@1 | R@3 | R@5 | R@10 |",
        "|---|---|---|---|---|---|",
    ]
    for t, v in x["recall"].items():
        out.append(f"| {t} | {v['n']} | {v['R@1']} | {v['R@3']} | {v['R@5']} | {v['R@10']} |")
    out.append("")
    return out


def _answer_section(r: dict) -> list[str]:
    """Answers by type, unanswerable categories and the failure table."""
    a = r["answers"]
    if not a:
        return []
    s = a["summary"]
    out = ["## Answers", "", "| type | n | correct | page hit | wrong declines |", "|---|---|---|---|---|"]
    for t, v in s["by_type"].items():
        out.append(f"| {t} | {v['n']} | {v['correct']} | {v['page_hit']} | {v['wrong_decline']} |")
    out += [
        "",
        f"Unanswerable by category: {s['unanswerable']['by_category']}",
        "",
        "### Failure analysis",
        "",
        "| id | question | what happened | stage |",
        "|---|---|---|---|",
    ]
    for row_ in a["rows"]:
        if row_.get("stage"):
            what = (row_["answer"][:90] + ("…" if len(row_["answer"]) > 90 else "")) + (
                f" | cited {[(c['doc'][:18], c['page']) for c in row_['citations']]}" if row_["citations"] else ""
            )
            out.append(f"| {row_['id']} | {row_['question'][:70]} | {what} | {row_['stage']} |")
    out.append("")
    return out


def _extraction_section(r: dict) -> list[str]:
    """Table, figure and reading-order detail."""
    e = r["extraction"]
    out = [
        "## Extraction detail",
        "",
        "Tables: "
        + "; ".join(
            f"{t['doc'][:24]} {t['title']}: {t['cells']} cells, rows {t['rows_matched']}" for t in e["tables"]["tables"]
        ),
        "",
        "Figures: "
        + "; ".join(
            f"{f['doc'][:20]} p{f['page']} {f['figure_id']}: found={f['found']} caption={f['caption_linked']} labels OCR/VLM={f['label_recall_ocr']}/{f['label_recall_vlm']}"
            for f in e["figures"]["figures"]
        ),
        "",
        "Reading order: "
        + "; ".join(
            f"{d[:24]} p{v['page']}: {v['pair_order_acc']} (missing {v['missing']})"
            for d, v in e["reading_order"].items()
        ),
        "",
    ]
    return out


def _first(d: dict, key: str) -> str:
    """Format `key` of the first entry (the single scanned gold page), or "-"."""
    return next((f"{v[key]:.4f}" for v in d.values()), "-")


def _avg(xs: list[float]) -> str:
    """Format the mean, or "-" for an empty list."""
    return f"{sum(xs) / len(xs):.3f}" if xs else "-"

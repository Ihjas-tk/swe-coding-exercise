"""Run every evaluation stage and write data/eval/RESULTS.md (+ JSON) with one summary table.

Also renders the external question-set report (`ae eval-external` -> data/eval/EXTERNAL.md).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ae import config
from ae.eval import answers as answers_eval
from ae.eval import extraction as extraction_eval
from ae.eval import omnidocbench as odb_eval
from ae.eval import retrieval as retrieval_eval
from ae.log import get_logger

log = get_logger(__name__)
OUT = Path("data/eval")


def run_all(
    backends: list[str], embed_models: list[str] | None = None, with_answers: bool = True, with_external: bool = True
) -> dict:
    """Run every stage for the given backends and write data/eval/RESULTS.md and results.json.

    Missing indexes are built first. The first embedding model is the one answers use; the
    others only appear in the retrieval ablations.
    """
    embed_models = embed_models or [config.EMBED_MODEL]
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_indexes(backends, embed_models[0])
    report: dict = {"backends": backends, "extraction": {}, "external": {}, "retrieval": [], "answers": {}}
    for be in backends:
        log.info(f"stage 1/4 extraction: backend={be}")
        report["extraction"][be] = extraction_eval.evaluate(be)
        if with_external and (Path("data/external/omnidocbench/extracted") / be).exists():
            log.info(f"stage 2/4 OmniDocBench slice: backend={be}")
            report["external"][be] = odb_eval.evaluate(be)
    log.info(f"stage 3/4 retrieval ablations: backends={backends}")
    report["retrieval"] = retrieval_eval.ablations(backends, embed_models)
    if with_answers:
        for be in backends:
            log.info(f"stage 4/4 answers: backend={be} (LLM calls are cached)")
            report["answers"][be] = answers_eval.run(be, extraction_report=report["extraction"][be])
    (OUT / "results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    (OUT / "RESULTS.md").write_text(to_markdown(report))
    log.info(f"wrote {OUT / 'RESULTS.md'} and {OUT / 'results.json'}")
    return report


def ensure_indexes(backends: list[str], embed_model: str, require_embeddings: bool = True) -> None:
    """Ingest every backend whose index is missing, empty, or (if required) lacks embeddings for `embed_model`.

    Ingest embeds with `embed_model`, and adds figure descriptions (VLM) when an API key is
    set, as `make ingest` does.
    """
    from ae.index.ingest import ingest
    from ae.index.store import IndexStore, index_db

    for be in backends:
        db = index_db(be)
        have = False
        if db.exists():
            s = IndexStore(db)
            have = bool(s.stats()["kinds"]) and (not require_embeddings or embed_model in s.embedding_models())
            s.close()
        if have:
            log.info(f"index for backend {be!r} found: {db}")
        else:
            log.info(f"index for backend {be!r} missing or unembedded: running ingest first ({db})")
            ingest(backend=be, embed_model=embed_model, vlm=bool(config.api_key()))


def run_external(backends: list[str], questions: Path, out: Path) -> list[str]:
    """Score an external question set on each backend; writes `out` (Markdown) and its .json; returns the lines."""
    reports = {be: answers_eval.run(be, questions=questions, adversarial=False) for be in backends}
    lines = external_markdown(reports, questions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    out.with_suffix(".json").write_text(json.dumps(reports, indent=1, ensure_ascii=False, default=str))
    return lines


def external_markdown(reports: dict[str, dict], questions: Path) -> list[str]:
    """Render the external evaluation: one summary table (14 lines incl. header) and the failure list."""
    bes = list(reports)
    lines = [
        f"# External evaluation ({questions})",
        "",
        "| Metric | " + " | ".join(bes) + " |",
        "|---|" + "---|" * len(bes),
    ]

    def row(label: str, fn: Callable[..., object]) -> None:
        lines.append(f"| {label} | " + " | ".join(str(fn(reports[be]["summary"])) for be in bes) + " |")

    row("Answerable questions", lambda s: s["answerable"]["n"])
    row("Correct (numbers within 1 % AND grader)", lambda s: s["answerable"]["correct"])
    row("Cited page in gold evidence", lambda s: s["answerable"]["page_hit"])
    row("Wrong declines", lambda s: s["answerable"]["wrong_decline"])
    row("Unanswerable handled correctly", lambda s: f"{s['unanswerable']['correct']}/{s['unanswerable']['n']}")
    for t in answers_eval.QUESTION_TYPES:
        row(
            f"Correct, {t}",
            lambda s, t=t: f"{s['by_type'].get(t, {}).get('correct', '-')}/{s['by_type'].get(t, {}).get('n', '-')}",
        )
    row("Failures by stage", lambda s: s["stages"])
    lines += [
        "",
        "## Failures (all backends)",
        "",
        "| id | backend | type | route | answer | stage | grader |",
        "|---|---|---|---|---|---|---|",
    ]
    for be in bes:
        for r in reports[be]["rows"]:
            if r.get("stage"):
                lines.append(
                    f"| {r['id']} | {be} | {r['type']} | {r.get('route')} | {r['answer'][:80]} | {r['stage']} | {str(r.get('grader', ''))[:80]} |"
                )
    return lines


def to_markdown(r: dict) -> str:
    """Render the evaluation report as Markdown: summary table, retrieval, answers, extraction detail."""
    out: list[str] = ["# Evaluation results", ""]
    out += _summary_table(r)
    out += _retrieval_sections(r)
    out += _answer_sections(r)
    out += _extraction_sections(r)
    return "\n".join(out)


Metric = tuple[str, Callable[[str], object]]  # row label, backend -> cell value


def _summary_table(r: dict) -> list[str]:
    """Render the one-table summary: one row per metric, one column per backend."""
    bes = r["backends"]
    out = ["## Summary (one table)", "", "| Metric | " + " | ".join(bes) + " |", "|---|" + "---|" * len(bes)]
    for name, fn in _extraction_metrics(r) + _retrieval_metrics(r) + _answer_metrics(r):
        out.append(f"| {name} | " + " | ".join(str(fn(be)) for be in bes) + " |")
    out.append("")
    return out


def _extraction_metrics(r: dict) -> list[Metric]:
    """Summary rows for the gold extraction set and (when run) the OmniDocBench slice."""
    ex = r["extraction"]
    rows: list[Metric] = [
        ("Extraction: OCR CER (scanned page)", lambda be: _first(ex[be]["ocr"], "cer")),
        ("Extraction: OCR WER (scanned page)", lambda be: _first(ex[be]["ocr"], "wer")),
        ("Extraction: table cell accuracy (4 tables)", lambda be: f"{ex[be]['tables']['cell_acc']:.3f}"),
        (
            "Extraction: figures found / caption linked (9)",
            lambda be: f"{ex[be]['figures']['found']} / {ex[be]['figures']['caption_linked']}",
        ),
        (
            "Extraction: figure label recall, OCR / VLM",
            lambda be: f"{ex[be]['figures']['label_recall_ocr']} / {ex[be]['figures']['label_recall_vlm']}",
        ),
        (
            "Extraction: reading order pair accuracy (3 pages)",
            lambda be: _avg(
                [v["pair_order_acc"] for v in ex[be]["reading_order"].values() if v["pair_order_acc"] is not None]
            ),
        ),
    ]
    if r["external"]:
        xo = r["external"]
        rows += [
            (
                "OmniDocBench (106 scanned pages): text similarity",
                lambda be: xo.get(be, {}).get("text_similarity", "-"),
            ),
            ("OmniDocBench: reading order pair accuracy", lambda be: xo.get(be, {}).get("reading_order_pair_acc", "-")),
            ("OmniDocBench: table TEDS (19 tables)", lambda be: xo.get(be, {}).get("table_teds", "-")),
            ("OmniDocBench: figure recall @IoU0.5 (50)", lambda be: xo.get(be, {}).get("figure_recall", "-")),
        ]
    return rows


def _retrieval_metrics(r: dict) -> list[Metric]:
    """Summary rows: overall page Recall@1 and @5 per retrieval mode (no chunk-kind ablations)."""
    ret = {(x["backend"], x["mode"], x["embed_model"], tuple(x["exclude_kinds"])): x for x in r["retrieval"]}
    emb = next((x["embed_model"] for x in r["retrieval"] if x["mode"] == "hybrid"), None)

    def recall(be: str, mode: str, k: int) -> object:
        cfg = ret.get((be, mode, None if mode == "bm25" else emb, ()), {})
        return cfg.get("recall", {}).get("all", {}).get(f"R@{k}", "-")

    def cell(mode: str, k: int) -> Callable[[str], object]:
        return lambda be: recall(be, mode, k)

    return [
        (f"Retrieval: page Recall@{k}, {mode}", cell(mode, k)) for mode in ("bm25", "dense", "hybrid") for k in (1, 5)
    ]


def _answer_metrics(r: dict) -> list[Metric]:
    """Summary rows for the answer stage (empty when it was skipped)."""
    if not r["answers"]:
        return []
    an = {be: a["summary"] for be, a in r["answers"].items()}
    return [
        (
            "Answers: correct (20 answerable)",
            lambda be: f"{an[be]['answerable']['correct']}/{an[be]['answerable']['n']}",
        ),
        ("Answers: cited page correct", lambda be: f"{an[be]['answerable']['page_hit']}/{an[be]['answerable']['n']}"),
        ("Answers: citation precision", lambda be: an[be]["answerable"]["citation_precision"]),
        ("Answers: wrong declines on answerable", lambda be: an[be]["answerable"]["wrong_decline"]),
        (
            "Not-found: handled correctly (2 dev + 30 adversarial)",
            lambda be: f"{an[be]['unanswerable']['correct']}/{an[be]['unanswerable']['n']}",
        ),
        (
            "Not-found: explicit declines / answered wrongly",
            lambda be: f"{an[be]['unanswerable']['correct_decline']} / {an[be]['unanswerable']['answered_wrongly']}",
        ),
        ("Penalty score (+1 / 0 / -2)", lambda be: f"{an[be]['penalty_score']} of {an[be]['penalty_max']}"),
    ]


def _retrieval_sections(r: dict) -> list[str]:
    """Page Recall@k by question type for every retrieval configuration."""
    out = ["## Retrieval: page Recall@k by question type", ""]
    for x in r["retrieval"]:
        tag = (
            f"{x['backend']} / {x['mode']}"
            + (f" / {x['embed_model'].split('/')[-1]}" if x["embed_model"] else "")
            + (f" / without {','.join(x['exclude_kinds'])}" if x["exclude_kinds"] else "")
        )
        out.append(f"**{tag}**  ")
        out.append("| type | n | R@1 | R@3 | R@5 | R@10 |")
        out.append("|---|---|---|---|---|---|")
        for t, v in x["recall"].items():
            out.append(f"| {t} | {v['n']} | {v['R@1']} | {v['R@3']} | {v['R@5']} | {v['R@10']} |")
        out.append("")
    return out


def _answer_sections(r: dict) -> list[str]:
    """Per-backend answers by type, unanswerable categories and the failure table."""
    out: list[str] = []
    for be, a in r["answers"].items():
        s = a["summary"]
        out += [f"## Answers ({be})", "", "| type | n | correct | page hit | wrong declines |", "|---|---|---|---|---|"]
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


def _extraction_sections(r: dict) -> list[str]:
    """Per-backend table, figure and reading-order detail."""
    out: list[str] = []
    for be, e in r["extraction"].items():
        out += [
            f"## Extraction detail ({be})",
            "",
            "Tables: "
            + "; ".join(
                f"{t['doc'][:24]} {t['title']}: {t['cells']} cells, rows {t['rows_matched']}"
                for t in e["tables"]["tables"]
            ),
            "",
        ]
        out.append(
            "Figures: "
            + "; ".join(
                f"{f['doc'][:20]} p{f['page']} {f['figure_id']}: found={f['found']} caption={f['caption_linked']} labels OCR/VLM={f['label_recall_ocr']}/{f['label_recall_vlm']}"
                for f in e["figures"]["figures"]
            )
        )
        out.append("")
        out.append(
            "Reading order: "
            + "; ".join(
                f"{d[:24]} p{v['page']}: {v['pair_order_acc']} (missing {v['missing']})"
                for d, v in e["reading_order"].items()
            )
        )
        out.append("")
    return out


def _first(d: dict, key: str) -> str:
    """Format `key` of the first entry (the single scanned gold page), or "-"."""
    return next((f"{v[key]:.4f}" for v in d.values()), "-")


def _avg(xs: list[float]) -> str:
    """Format the mean, or "-" for an empty list."""
    return f"{sum(xs) / len(xs):.3f}" if xs else "-"

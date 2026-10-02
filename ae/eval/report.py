"""Run every evaluation stage and write data/eval/RESULTS.md (+ JSON) with one summary table."""
from __future__ import annotations

import json
from pathlib import Path

from ae import config
from ae.eval import answers as answers_eval
from ae.eval import extraction as extraction_eval
from ae.eval import omnidocbench as odb_eval
from ae.eval import retrieval as retrieval_eval

OUT = Path("data/eval")


def run_all(backends: list[str], embed_models: list[str] | None = None, with_answers: bool = True, with_external: bool = True) -> dict:
    embed_models = embed_models or [config.EMBED_MODEL]
    OUT.mkdir(parents=True, exist_ok=True)
    from ae.index.ingest import ingest
    from ae.index.store import IndexStore, index_db
    from ae.log import get_logger

    log = get_logger(__name__)
    for be in backends:
        db = index_db(be)
        have = False
        if db.exists():
            s = IndexStore(db)
            have = bool(s.stats()["kinds"]) and embed_models[0] in s.embedding_models()
            s.close()
        if not have:
            log.info(f"index for backend {be!r} missing or unembedded: running ingest first")
            ingest(backend=be, embed_model=embed_models[0], vlm=bool(config.api_key()))
    report: dict = {"backends": backends, "extraction": {}, "external": {}, "retrieval": [], "answers": {}}
    for be in backends:
        report["extraction"][be] = extraction_eval.evaluate(be)
        if with_external and (Path("data/external/omnidocbench/extracted") / be).exists():
            report["external"][be] = odb_eval.evaluate(be)
    report["retrieval"] = retrieval_eval.ablations(backends, embed_models)
    if with_answers:
        for be in backends:
            report["answers"][be] = answers_eval.run(be, extraction_report=report["extraction"][be])
    (OUT / "results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    (OUT / "RESULTS.md").write_text(to_markdown(report))
    return report


def to_markdown(r: dict) -> str:
    L: list[str] = ["# Evaluation results", ""]
    bes = r["backends"]
    # ---- single summary table
    L += ["## Summary (one table)", "", "| Metric | " + " | ".join(bes) + " |", "|---|" + "---|" * len(bes)]

    def row(name, fn):
        L.append(f"| {name} | " + " | ".join(str(fn(be)) for be in bes) + " |")

    ex = r["extraction"]
    row("Extraction: OCR CER (scanned page)", lambda be: _first(ex[be]["ocr"], "cer"))
    row("Extraction: OCR WER (scanned page)", lambda be: _first(ex[be]["ocr"], "wer"))
    row("Extraction: table cell accuracy (4 tables)", lambda be: f'{ex[be]["tables"]["cell_acc"]:.3f}')
    row("Extraction: figures found / caption linked (9)", lambda be: f'{ex[be]["figures"]["found"]} / {ex[be]["figures"]["caption_linked"]}')
    row("Extraction: figure label recall, OCR / VLM", lambda be: f'{ex[be]["figures"]["label_recall_ocr"]} / {ex[be]["figures"]["label_recall_vlm"]}')
    row("Extraction: reading order pair accuracy (3 pages)", lambda be: _avg([v["pair_order_acc"] for v in ex[be]["reading_order"].values() if v["pair_order_acc"] is not None]))
    if r["external"]:
        xo = r["external"]
        row("OmniDocBench (106 scanned pages): text similarity", lambda be: xo.get(be, {}).get("text_similarity", "-"))
        row("OmniDocBench: reading order pair accuracy", lambda be: xo.get(be, {}).get("reading_order_pair_acc", "-"))
        row("OmniDocBench: table TEDS (19 tables)", lambda be: xo.get(be, {}).get("table_teds", "-"))
        row("OmniDocBench: figure recall @IoU0.5 (50)", lambda be: xo.get(be, {}).get("figure_recall", "-"))
    ret = {(x["backend"], x["mode"], x["embed_model"], tuple(x["exclude_kinds"])): x for x in r["retrieval"]}
    emb = next((x["embed_model"] for x in r["retrieval"] if x["mode"] == "hybrid"), None)
    for mode in ("bm25", "dense", "hybrid"):
        for k in (1, 5):
            row(f"Retrieval: page Recall@{k}, {mode}", lambda be, mode=mode, k=k: ret.get((be, mode, None if mode == "bm25" else emb, ()), {}).get("recall", {}).get("all", {}).get(f"R@{k}", "-"))
    if r["answers"]:
        an = r["answers"]
        row("Answers: correct (20 answerable)", lambda be: f'{an[be]["summary"]["answerable"]["correct"]}/{an[be]["summary"]["answerable"]["n"]}')
        row("Answers: cited page correct", lambda be: f'{an[be]["summary"]["answerable"]["page_hit"]}/{an[be]["summary"]["answerable"]["n"]}')
        row("Answers: citation precision", lambda be: an[be]["summary"]["answerable"]["citation_precision"])
        row("Answers: wrong declines on answerable", lambda be: an[be]["summary"]["answerable"]["wrong_decline"])
        row("Not-found: handled correctly (2 dev + 30 adversarial)", lambda be: f'{an[be]["summary"]["unanswerable"]["correct"]}/{an[be]["summary"]["unanswerable"]["n"]}')
        row("Not-found: explicit declines / answered wrongly", lambda be: f'{an[be]["summary"]["unanswerable"]["correct_decline"]} / {an[be]["summary"]["unanswerable"]["answered_wrongly"]}')
        row("Penalty score (+1 / 0 / -2)", lambda be: f'{an[be]["summary"]["penalty_score"]} of {an[be]["summary"]["penalty_max"]}')
        row("Query latency p50 / p95 (ms, end to end)", lambda be: f'{an[be]["summary"].get("latency_ms", {}).get("total", {}).get("p50", "-")} / {an[be]["summary"].get("latency_ms", {}).get("total", {}).get("p95", "-")}')
        row("Latency p50 by stage (ms): retrieve / generate / sql", lambda be: " / ".join(str(an[be]["summary"].get("latency_ms", {}).get(k, {}).get("p50", "-")) for k in ("retrieve", "generate", "sql")))
    L.append("")
    # ---- retrieval by type
    L += ["## Retrieval: page Recall@k by question type", ""]
    for x in r["retrieval"]:
        tag = f'{x["backend"]} / {x["mode"]}' + (f' / {x["embed_model"].split("/")[-1]}' if x["embed_model"] else "") + (f' / without {",".join(x["exclude_kinds"])}' if x["exclude_kinds"] else "")
        L.append(f"**{tag}**  ")
        L.append("| type | n | R@1 | R@3 | R@5 | R@10 |")
        L.append("|---|---|---|---|---|---|")
        for t, v in x["recall"].items():
            L.append(f'| {t} | {v["n"]} | {v["R@1"]} | {v["R@3"]} | {v["R@5"]} | {v["R@10"]} |')
        L.append("")
    # ---- answers by type + failures
    for be, a in r["answers"].items():
        s = a["summary"]
        L += [f"## Answers ({be})", "", "| type | n | correct | page hit | wrong declines |", "|---|---|---|---|---|"]
        for t, v in s["by_type"].items():
            L.append(f'| {t} | {v["n"]} | {v["correct"]} | {v["page_hit"]} | {v["wrong_decline"]} |')
        L += ["", f'Unanswerable by category: {s["unanswerable"]["by_category"]}', "", "### Failure analysis", "", "| id | question | what happened | stage |", "|---|---|---|---|"]
        for row_ in a["rows"]:
            if row_.get("stage"):
                what = (row_["answer"][:90] + ("…" if len(row_["answer"]) > 90 else "")) + (f' | cited {[(c["doc"][:18], c["page"]) for c in row_["citations"]]}' if row_["citations"] else "")
                L.append(f'| {row_["id"]} | {row_["question"][:70]} | {what} | {row_["stage"]} |')
        L.append("")
    # ---- extraction detail
    for be, e in r["extraction"].items():
        L += [f"## Extraction detail ({be})", "", "Tables: " + "; ".join(f'{t["doc"][:24]} {t["title"]}: {t["cells"]} cells, rows {t["rows_matched"]}' for t in e["tables"]["tables"]), ""]
        L.append("Figures: " + "; ".join(f'{f["doc"][:20]} p{f["page"]} {f["figure_id"]}: found={f["found"]} caption={f["caption_linked"]} labels OCR/VLM={f["label_recall_ocr"]}/{f["label_recall_vlm"]}' for f in e["figures"]["figures"]))
        L.append("")
        L.append("Reading order: " + "; ".join(f'{d[:24]} p{v["page"]}: {v["pair_order_acc"]} (missing {v["missing"]})' for d, v in e["reading_order"].items()))
        L.append("")
    return "\n".join(L)


def _first(d: dict, key: str):
    return next((f"{v[key]:.4f}" for v in d.values()), "-")


def _avg(xs):
    return f"{sum(xs) / len(xs):.3f}" if xs else "-"

"""Command line entry points. `make` targets wrap these."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint

app = typer.Typer(no_args_is_help=True, add_completion=False)

CORPUS_DIRS = ("patents", "design_docs", "structured")


SUPPORTED = (".pdf", ".docx", ".csv", ".xlsx")


def corpus_files(root: Path = Path("."), corpus: Path | None = None) -> list[Path]:
    """Files to ingest: the built-in corpus dirs, or every supported file under `corpus` (recursive)."""
    from ae import config

    corpus = corpus or (Path(config.CORPUS) if config.CORPUS else None)
    if corpus:
        return sorted(p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith("."))
    files: list[Path] = []
    for d in CORPUS_DIRS:
        files += sorted(p for p in (root / d).glob("*") if p.is_file() and not p.name.startswith("."))
    return files


def _set_index(name: str | None) -> None:
    from ae import config

    if name is not None:
        config.INDEX = name


@app.command()
def extract(
    path: list[Path] = typer.Argument(None, help="Files to extract; default: whole corpus (PDFs only in this phase)."),
    backend: str = typer.Option("thin", help="thin | docling | hybrid"),
    out: Path = typer.Option(Path("data/extracted"), help="Output root; writes <out>/<backend>/<doc>.json and .md"),
    no_cache: bool = typer.Option(False, help="Ignore the page cache and re-parse."),
):
    """Run extraction only and dump ParsedDocument JSON + Markdown for inspection."""
    from ae.extract.base import parse

    files = path or [p for p in corpus_files() if p.suffix.lower() in (".pdf", ".docx")]
    out_dir = out / backend
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        doc = parse(f, backend=backend, use_cache=not no_cache)
        (out_dir / f"{f.stem}.json").write_text(doc.model_dump_json(indent=1))
        (out_dir / f"{f.stem}.md").write_text(doc.to_markdown())
        n_tab = sum(1 for p in doc.pages for b in p.blocks if b.kind == "table")
        n_fig = sum(1 for p in doc.pages for b in p.blocks if b.kind == "figure")
        n_ocr = sum(1 for p in doc.pages if p.is_scanned)
        note = f" [yellow](page mapping {doc.meta['page_mapping']})[/yellow]" if "page_mapping" in doc.meta else ""
        rprint(f"[green]{f.name}[/green]: {len(doc.pages)} pages, {n_tab} tables, {n_fig} figures, {n_ocr} OCR pages{note} -> {out_dir / f.stem}.md")


@app.command()
def prep_external():
    """Wrap the OmniDocBench subset images into single-page PDFs under data/external/omnidocbench/pdfs."""
    from ae.eval.external import prepare_omnidocbench

    pdfs = prepare_omnidocbench()
    rprint(f"[green]{len(pdfs)} benchmark PDFs ready[/green]")


@app.command()
def load_structured(
    path: list[Path] = typer.Argument(None, help="CSV/XLSX files; default: structured/ in the corpus."),
    sql: str = typer.Option(None, help="Optional SELECT to run after loading."),
):
    """Load CSV/XLSX files into typed SQLite tables and print the schema the LLM will see."""
    from ae.extract.structured import describe_schema, load_structured as _load, query_sql

    files = path or [p for p in corpus_files() if p.suffix.lower() in (".csv", ".xlsx")]
    for t in _load(files):
        rprint(f"[green]{t.doc}[/green] -> table [bold]{t.name}[/bold]: {t.n_rows} rows, {len(t.columns)} columns")
    rprint(describe_schema())
    if sql:
        res = query_sql(sql)
        rprint(res["columns"])
        for r in res["rows"]:
            rprint(r)


@app.command()
def ingest(
    backend: str = typer.Option("thin", help="thin | docling | hybrid"),
    embed: str = typer.Option(None, help="Embedding model id (e.g. ibm-granite/granite-embedding-english-r2); omit to skip embeddings."),
    vlm: bool = typer.Option(False, help="Describe each figure with the multimodal LLM (needs ANTHROPIC_API_KEY; cached)."),
    no_cache: bool = typer.Option(False, help="Re-parse instead of using the page cache."),
    corpus: Path = typer.Option(None, help="Directory of files to ingest instead of the built-in corpus (recursive)."),
    name: str = typer.Option(None, help="Index namespace, e.g. 'ext' -> data/index/ext/<backend>.sqlite (default: AE_INDEX)."),
):
    """Extract, load structured files, build numeral index, chunk, and index into data/index/[<name>/]<backend>.sqlite."""
    from ae.index.ingest import ingest as _ingest

    _set_index(name)
    stats = _ingest(backend=backend, files=corpus_files(corpus=corpus), use_cache=not no_cache, embed_model=embed, vlm=vlm)
    table = stats.pop("timings_table", "")
    rprint(stats)
    if table:
        print(table)


@app.command()
def search(
    question: str,
    backend: str = typer.Option("thin"),
    k: int = typer.Option(8),
    doc: str = typer.Option(None, help="Restrict to one document file name."),
):
    """Keyword (BM25 + identifier) search over the index; prints the top chunks."""
    from ae.index.store import IndexStore, index_db

    store = IndexStore(index_db(backend))
    hits = store.keyword_search(question, k=k, doc=doc)
    for h in hits:
        c = store.get([h.chunk_id])[0]
        rprint(f"[cyan]{h.score:6.2f}[/cyan] [bold]{c.kind:13}[/bold] {c.doc} p{c.page} | {c.text[:110]}")


@app.command()
def ask(
    question: str,
    backend: str = typer.Option(None, help="thin | docling | hybrid (default: AE_BACKEND or thin)"),
    debug: bool = typer.Option(False, help="Include routing/retrieval/verification details."),
    mode: str = typer.Option("hybrid", help="hybrid | bm25 | dense (retrieval ablation)"),
    name: str = typer.Option(None, help="Index namespace (see ingest --name)."),
):
    """Answer a question; prints {"answer", "citations": [{"doc", "page"}], "not_found"}."""
    from ae.answer.pipeline import Engine

    _set_index(name)
    eng = Engine(backend=backend, mode=mode)
    a = eng.ask(question, debug=debug)
    out = a.to_json(debug=debug)
    if not debug:
        out["latency_ms"] = a.debug.get("timing_ms", {}).get("total")
    print(json.dumps(out, indent=2, ensure_ascii=False))


@app.command()
def eval_dev(
    backend: str = typer.Option(None),
    mode: str = typer.Option("hybrid"),
    out: Path = typer.Option(Path("data/eval/devset_results.json")),
    questions: Path = typer.Option(Path("dev_set/questions.json"), help="Question file in the dev-set format."),
    name: str = typer.Option(None, help="Index namespace (see ingest --name)."),
):
    """Run a question set end to end and print per-type scores (quick harness)."""
    from ae.eval.devset import run, summary

    from ae.log import latency_summary

    _set_index(name)
    rows = run(questions=questions, backend=backend, mode=mode, out=out)
    for r in rows:
        flag = "" if (r["nums_ok"] in (True, None) and r["page_hit"] in (True, None) and r["nf_ok"]) else "  <-- check"
        rprint(f"[bold]{r['id']}[/bold] {r['type']:12} route={r['route']:7} ret={r['retrieved']} page={r['page_hit']} nums={r['nums_ok']} nf={r['nf_ok']}{flag}")
        rprint(f"      [dim]{r['answer'][:150]}[/dim]  cites={[(c['doc'][:14], c['page']) for c in r['citations']]}")
    print(summary(rows))
    lat = latency_summary()
    if lat:
        print("latency ms (p50 / p95 / max): " + ", ".join(f"{k} {v['p50']}/{v['p95']}/{v['max']}" for k, v in lat.items()))


@app.command()
def eval(
    backend: list[str] = typer.Option(None, help="Backends to evaluate (default: thin docling hybrid)."),
    no_answers: bool = typer.Option(False, help="Skip the answer stage (no LLM calls)."),
    no_external: bool = typer.Option(False, help="Skip the OmniDocBench slice."),
):
    """Run the staged evaluation (extraction, external benchmark, retrieval ablations, answers) -> data/eval/RESULTS.md"""
    from ae.eval.report import run_all

    bes = backend or ["thin", "docling", "hybrid"]
    run_all(bes, with_answers=not no_answers, with_external=not no_external)
    rprint((Path("data/eval/RESULTS.md").read_text().split("## Retrieval")[0]))
    rprint("[green]full report: data/eval/RESULTS.md[/green]")


@app.command()
def eval_external(
    questions: Path = typer.Option(Path("data/external/evalset/questions.json")),
    name: str = typer.Option("ext"),
    backend: list[str] = typer.Option(None, help="default: thin docling hybrid"),
    out: Path = typer.Option(Path("data/eval/EXTERNAL.md")),
):
    """Score an external question set with the numeric check + LLM grader and write a gap report by stage."""
    from ae.eval.answers import run

    _set_index(name)
    bes = backend or ["thin", "docling", "hybrid"]
    reports = {be: run(be, questions=questions, adversarial=False) for be in bes}
    L = [f"# External evaluation ({questions})", "", "| Metric | " + " | ".join(bes) + " |", "|---|" + "---|" * len(bes)]
    def row(label, fn):
        L.append(f"| {label} | " + " | ".join(str(fn(reports[be]["summary"])) for be in bes) + " |")
    row("Answerable questions", lambda s: s["answerable"]["n"])
    row("Correct (numbers within 1 % AND grader)", lambda s: s["answerable"]["correct"])
    row("Cited page in gold evidence", lambda s: s["answerable"]["page_hit"])
    row("Wrong declines", lambda s: s["answerable"]["wrong_decline"])
    row("Unanswerable handled correctly", lambda s: f'{s["unanswerable"]["correct"]}/{s["unanswerable"]["n"]}')
    for t in ("factual", "table", "figure", "structured"):
        row(f"Correct, {t}", lambda s, t=t: f'{s["by_type"].get(t, {}).get("correct", "-")}/{s["by_type"].get(t, {}).get("n", "-")}')
    row("Failures by stage", lambda s: s["stages"])
    row("Query latency p50 / p95 ms", lambda s: f'{s.get("latency_ms", {}).get("total", {}).get("p50", "-")} / {s.get("latency_ms", {}).get("total", {}).get("p95", "-")}')
    L += ["", "## Failures (all backends)", "", "| id | backend | type | route | answer | stage | grader |", "|---|---|---|---|---|---|---|"]
    for be in bes:
        for r in reports[be]["rows"]:
            if r.get("stage"):
                L.append(f'| {r["id"]} | {be} | {r["type"]} | {r.get("route")} | {r["answer"][:80]} | {r["stage"]} | {str(r.get("grader", ""))[:80]} |')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    (out.with_suffix(".json")).write_text(json.dumps(reports, indent=1, ensure_ascii=False, default=str))
    rprint("\n".join(L[:14]))
    rprint(f"[green]full report: {out}[/green]")


@app.command()
def fetch_external(manifest: Path = typer.Option(Path("gold/external_manifest.json"))):
    """Download the external evaluation corpus listed in the manifest (verifies sha256); ICDAR zip handled too."""
    import hashlib
    import io
    import shutil
    import urllib.request
    import zipfile

    man = json.loads(manifest.read_text())
    root = Path(man["root"])
    for f in man["files"]:
        dst = root / f["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == f["sha256"]:
            rprint(f"[dim]ok {f['path']}[/dim]")
            continue
        req = urllib.request.Request(f["url"], headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=120).read()
        dst.write_bytes(data)
        h = hashlib.sha256(data).hexdigest()
        rprint(f"{'[green]fetched[/green]' if h == f['sha256'] else '[yellow]fetched (checksum differs: publisher updated the file)[/yellow]'} {f['path']}")
    ic = man["icdar2013"]
    raw = Path("data/external/icdar2013_raw")
    if not raw.exists():
        req = urllib.request.Request(ic["zip"], headers={"User-Agent": "Mozilla/5.0"})
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=300).read())).extractall(raw)
    (root / "icdar2013").mkdir(parents=True, exist_ok=True)
    for name in ic["selected"]:
        d = name.replace("icdar2013_", "").replace(".pdf", "")
        src = raw / f"competition-dataset-{d.split('-')[0]}" / f"{d}.pdf"
        if src.exists() and not (root / "icdar2013" / name).exists():
            shutil.copy(src, root / "icdar2013" / name)
    rprint(f"[green]external corpus ready under {root}[/green]")


@app.command()
def check():
    """Verify system packages, Python packages, the API key, cached models and indexes; exit 1 on failures."""
    from ae.check import render, run_checks

    res = run_checks()
    print(render(res))
    raise typer.Exit(code=1 if any(r[0] == "fail" for r in res) else 0)


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8080), backend: str = typer.Option(None)):
    """Minimal HTTP API: POST /ask {"question": ...} -> the same JSON as `ask`; models stay warm."""
    from ae.serve import serve as _serve

    _serve(host, port, backend)


@app.command()
def smoke(backend: str = typer.Option(None, help="default: AE_BACKEND / hybrid")):
    """Fast end-to-end check on one file of each type (data/smoke, index 'smoke'); then run `make eval` for the full run."""
    from ae.smoke import run

    r = run(backend)
    print(r.pop("timings", ""))
    print(r.pop("summary", ""))
    rprint(r)
    raise typer.Exit(code=1 if r.get("failures") else 0)


@app.command()
def show(path: Path, page: int = typer.Option(0, help="1-based page; 0 = all")):
    """Pretty-print a ParsedDocument JSON produced by `extract`."""
    from ae.schema import ParsedDocument

    doc = ParsedDocument.model_validate_json(path.read_text())
    for p in doc.pages:
        if page and p.number != page:
            continue
        src = f"OCR: {p.ocr_reason}" if p.is_scanned else "text layer"
        rprint(f"[bold]--- page {p.number} ({src}; backend={p.backend}) ---[/bold]")
        for b in p.blocks:
            if b.kind == "text":
                rprint(f"[cyan]{b.role:8}[/cyan] col={b.column} | {b.text[:110]}")
            elif b.kind == "table":
                rprint(f"[magenta]table   [/magenta] {b.n_rows}x{b.n_cols} title={b.title!r}")
                rprint("  " + "\n  ".join(json.dumps(r, ensure_ascii=False) for r in b.rows))
            else:
                rprint(f"[yellow]figure  [/yellow] id={b.figure_id!r} labels={b.ocr_labels[:12]} caption={b.caption!r}")


if __name__ == "__main__":
    app()

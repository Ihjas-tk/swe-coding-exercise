"""Command line entry points (`ae <command>`). `make` targets wrap these.

Shared options mean the same in every command that takes them:
  --name     index namespace: data/index/NAME/index.sqlite. --name wins over AE_INDEX;
             with neither, data/index/default/index.sqlite.
  --corpus   directory of files used instead of the built-in patents/ design_docs/ structured/
             (recursive). --corpus wins over AE_CORPUS.
Exit codes: 0 ok, 1 failure (missing index or API key, failed check, smoke failure), 2 usage error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich import print as rprint
from rich.console import Console

from ae import config
from ae.corpus import corpus_files

HELP = """Answer engine for engineering documents: extract, ingest, ask, evaluate.

Shared options: --name (index namespace data/index/NAME/; --name wins over AE_INDEX),
--corpus (directory instead of the built-in corpus; --corpus wins over AE_CORPUS).
Exit codes: 0 ok, 1 failure, 2 usage error."""

app = typer.Typer(no_args_is_help=True, add_completion=False, help=HELP, rich_markup_mode="markdown")
_stderr = Console(stderr=True)

DOC_SUFFIXES = (".pdf", ".docx")
STRUCTURED_SUFFIXES = (".csv", ".xlsx")
NO_KEY_MESSAGE = (
    "ANTHROPIC_API_KEY is not set: add the line ANTHROPIC_API_KEY=<your key> to .env in the repo root "
    "(copy .env.example); cached answers work without it"
)


# ------------------------------------------------------------------ shared options
NAME_HELP = "Index namespace -> data/index/NAME/index.sqlite; wins over AE_INDEX. Default: AE_INDEX, else 'default'."
CORPUS_HELP = "Directory of files to use instead of the built-in corpus (recursive); wins over AE_CORPUS."

NameOpt = Annotated[str | None, typer.Option("--name", help=NAME_HELP)]
CorpusOpt = Annotated[Path | None, typer.Option("--corpus", help=CORPUS_HELP, file_okay=False, exists=True)]


def _fail(msg: str) -> NoReturn:
    """Print an actionable error to stderr and exit 1."""
    _stderr.print(f"[red]error:[/red] {msg}", soft_wrap=True)
    raise typer.Exit(1)


def _set_index(name: str | None) -> None:
    """Apply --name over AE_INDEX for everything that resolves index paths afterwards."""
    if name is not None:
        config.INDEX = name


def _require_index() -> None:
    """Exit 1 with the build command when the index of the active namespace is missing."""
    from ae.answer.pipeline import ingest_hint
    from ae.index.store import index_db

    db = index_db()
    if not db.exists():
        _fail(f"no index at {db}; build it with {ingest_hint()}")


def _warn_without_key() -> None:
    """Answering without a key only works from the LLM cache; say so before a long run."""
    if not config.api_key():
        _stderr.print(
            f"[yellow]warning:[/yellow] {NO_KEY_MESSAGE}; uncached questions will score as declined", soft_wrap=True
        )


# ------------------------------------------------------------------ build
@app.command()
def extract(
    path: Annotated[
        list[Path] | None, typer.Argument(help="Files to extract. Default: every PDF/DOCX in the corpus.")
    ] = None,
    corpus: CorpusOpt = None,
    out: Annotated[Path, typer.Option(help="Output directory; writes OUT/DOC.json and OUT/DOC.md")] = Path(
        "data/extracted"
    ),
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Ignore the page cache and re-parse.")] = False,
) -> None:
    """Run extraction only and dump ParsedDocument JSON + Markdown for inspection."""
    from ae.extract.base import parse

    files = path or [p for p in corpus_files(corpus=corpus) if p.suffix.lower() in DOC_SUFFIXES]
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        doc = parse(f, use_cache=not no_cache)
        (out / f"{f.stem}.json").write_text(doc.model_dump_json(indent=1))
        (out / f"{f.stem}.md").write_text(doc.to_markdown())
        n_tab = sum(1 for p in doc.pages for b in p.blocks if b.kind == "table")
        n_fig = sum(1 for p in doc.pages for b in p.blocks if b.kind == "figure")
        n_ocr = sum(1 for p in doc.pages if p.is_scanned)
        note = f" [yellow](page mapping {doc.meta['page_mapping']})[/yellow]" if "page_mapping" in doc.meta else ""
        rprint(
            f"[green]{f.name}[/green]: {len(doc.pages)} pages, {n_tab} tables, {n_fig} figures, {n_ocr} OCR pages{note} -> {out / f.stem}.md"
        )


@app.command()
def ingest(
    name: NameOpt = None,
    corpus: CorpusOpt = None,
    embed: Annotated[
        str | None,
        typer.Option(
            help="Embedding model id (e.g. ibm-granite/granite-embedding-english-r2); omit to skip embeddings."
        ),
    ] = None,
    vlm: Annotated[
        bool,
        typer.Option("--vlm", help="Describe each figure with the multimodal LLM (needs ANTHROPIC_API_KEY; cached)."),
    ] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Re-parse instead of using the page cache.")] = False,
) -> None:
    """Extract, load structured files, build the numeral index, chunk and index into data/index/NAME/index.sqlite."""
    from ae.index.ingest import ingest as _ingest

    _set_index(name)
    stats = _ingest(
        files=corpus_files(corpus=corpus),
        use_cache=not no_cache,
        embed_model=embed,
        vlm=vlm,
    )
    rprint(stats)


@app.command()
def load_structured(
    path: Annotated[list[Path] | None, typer.Argument(help="CSV/XLSX files. Default: those in the corpus.")] = None,
    corpus: CorpusOpt = None,
    sql: Annotated[str | None, typer.Option(help="Optional SELECT to run after loading.")] = None,
) -> None:
    """Load CSV/XLSX files into typed SQLite tables and print the schema the LLM will see."""
    from ae.extract.structured import describe_schema, query_sql
    from ae.extract.structured import load_structured as _load

    files = path or [p for p in corpus_files(corpus=corpus) if p.suffix.lower() in STRUCTURED_SUFFIXES]
    for t in _load(files):
        rprint(f"[green]{t.doc}[/green] -> table [bold]{t.name}[/bold]: {t.n_rows} rows, {len(t.columns)} columns")
    rprint(describe_schema())
    if sql:
        res = query_sql(sql)
        rprint(res["columns"])
        for r in res["rows"]:
            rprint(r)


@app.command()
def models(
    embed: Annotated[str | None, typer.Option(help="Embedding model to fetch. Default: AE_EMBED_MODEL.")] = None,
) -> None:
    """Download the local models into the Hugging Face cache so ingest does not stall on a download."""
    try:
        from ae.models import prefetch
    except ImportError:
        _fail("ae.models is missing from this checkout; models download on first use instead")
    try:
        prefetch(embed or config.EMBED_MODEL)
    except RuntimeError as e:
        _fail(str(e))


# ------------------------------------------------------------------ query
@app.command()
def ask(
    question: str,
    name: NameOpt = None,
    debug: Annotated[bool, typer.Option("--debug", help="Include routing/retrieval/verification details.")] = False,
) -> None:
    """Answer a question; prints {"answer", "citations": [{"doc", "page"}], "not_found"}."""
    from ae.answer.pipeline import Engine

    _set_index(name)
    _require_index()
    ans = Engine().ask(question)
    error = ans.debug.get("error")
    if error:
        _fail(NO_KEY_MESSAGE if "ANTHROPIC_API_KEY" in error else f"answer generation failed: {error}")
    typer.echo(json.dumps(ans.to_json(debug=debug), indent=2, ensure_ascii=False))


@app.command()
def search(
    question: str,
    name: NameOpt = None,
    k: Annotated[int, typer.Option(help="Number of chunks to show.")] = 8,
    doc: Annotated[str | None, typer.Option(help="Restrict to one document file name.")] = None,
) -> None:
    """Keyword (BM25 + identifier) search over the index; prints the top chunks (no LLM)."""
    from ae.index.store import IndexStore, index_db

    _set_index(name)
    _require_index()
    store = IndexStore(index_db())
    for h in store.keyword_search(question, k=k, doc=doc):
        c = store.get([h.chunk_id])[0]
        rprint(f"[cyan]{h.score:6.2f}[/cyan] [bold]{c.kind:13}[/bold] {c.doc} p{c.page} | {c.text[:110]}")


# ------------------------------------------------------------------ evaluate
@app.command(name="eval")
def eval_(
    name: NameOpt = None,
    quick: Annotated[
        bool,
        typer.Option("--quick", help="Quick harness: answer a question file, numeric scores per type (no grader)."),
    ] = False,
    questions: Annotated[
        Path | None, typer.Option(help="With --quick: question file. Default: dev_set/questions.json.", exists=True)
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="With --quick: rows JSON. Default: data/eval/devset_results.json.")
    ] = None,
    no_answers: Annotated[bool, typer.Option("--no-answers", help="Skip the answer stage (no LLM calls).")] = False,
    no_external: Annotated[bool, typer.Option("--no-external", help="Skip the OmniDocBench slice.")] = False,
) -> None:
    """Staged evaluation (extraction, OmniDocBench, retrieval, answers) -> data/eval/RESULTS.md.

    Builds the index first when it is missing. With --quick, only answers a question file
    (default: the dev set) and prints per-question and per-type scores; --questions/--out
    apply only to --quick.
    """
    _set_index(name)
    if quick:
        if no_answers or no_external:
            raise typer.BadParameter("--no-answers/--no-external apply to the full eval only", param_hint="'--quick'")
        _eval_quick(questions or Path("dev_set/questions.json"), out or Path("data/eval/devset_results.json"))
        return
    if questions or out:
        raise typer.BadParameter("--questions/--out apply only with --quick", param_hint="'--quick'")
    from ae.eval.report import run_all

    if not no_answers:
        _warn_without_key()
    run_all(with_answers=not no_answers, with_external=not no_external)
    rprint(Path("data/eval/RESULTS.md").read_text().split("## Retrieval")[0])
    rprint("[green]full report: data/eval/RESULTS.md[/green]")


def _eval_quick(questions: Path, out: Path) -> None:
    """Quick harness: one line per question, then the per-type table."""
    from ae.eval.devset import run, summary
    from ae.eval.report import ensure_index

    _warn_without_key()
    ensure_index(config.EMBED_MODEL, require_embeddings=False)
    rows = run(questions=questions, out=out)
    rprint(f"[bold]{questions}[/bold] -> {out}")
    for r in rows:
        ok = r["nums_ok"] in (True, None) and r["page_hit"] in (True, None) and r["nf_ok"]
        rprint(
            f"[bold]{r['id']}[/bold] {r['type']:12} route={r['route']:7} ret={r['retrieved']} page={r['page_hit']} nums={r['nums_ok']} nf={r['nf_ok']}{'' if ok else '  <-- check'}"
        )
        rprint(f"      [dim]{r['answer'][:150]}[/dim]  cites={[(c['doc'][:14], c['page']) for c in r['citations']]}")
    typer.echo(summary(rows))


@app.command()
def eval_external(
    questions: Annotated[Path, typer.Option(help="Question file in the dev-set format.", exists=True)] = Path(
        "data/external/evalset/questions.json"
    ),
    name: Annotated[str, typer.Option("--name", help="Index namespace of the external corpus.")] = "ext",
    out: Annotated[Path, typer.Option(help="Markdown report; a .json with every row is written next to it.")] = Path(
        "data/eval/EXTERNAL.md"
    ),
) -> None:
    """Score an external question set (numeric check + LLM grader) and write a gap report by stage."""
    from ae.eval.report import run_external

    _set_index(name)
    _require_index()
    _warn_without_key()
    lines = run_external(questions, out)
    rprint("\n".join(lines[:14]))
    rprint(f"[green]full report: {out}[/green]")


@app.command()
def fetch_external(
    manifest: Annotated[Path, typer.Option(help="Manifest of files, URLs and checksums.", exists=True)] = Path(
        "gold/external_manifest.json"
    ),
) -> None:
    """Download the external evaluation corpus listed in the manifest (verifies sha256; ICDAR zip handled too)."""
    from ae.eval.external import fetch_corpus

    root = fetch_corpus(manifest)
    rprint(f"[green]external corpus ready under {root}[/green]")


@app.command()
def prep_external() -> None:
    """Wrap the OmniDocBench subset images into single-page PDFs under data/external/omnidocbench/pdfs."""
    from ae.eval.external import prepare_omnidocbench

    pdfs = prepare_omnidocbench()
    rprint(f"[green]{len(pdfs)} benchmark PDFs ready[/green]")


# ------------------------------------------------------------------ inspect / health
@app.command()
def show(
    path: Annotated[
        Path, typer.Argument(help="ParsedDocument JSON written by `ae extract`.", exists=True, dir_okay=False)
    ],
    page: Annotated[int, typer.Option(help="1-based page; 0 = all")] = 0,
) -> None:
    """Pretty-print a ParsedDocument JSON produced by `extract`."""
    from ae.schema import ParsedDocument

    doc = ParsedDocument.model_validate_json(path.read_text())
    for p in doc.pages:
        if page and p.number != page:
            continue
        src = f"OCR: {p.ocr_reason}" if p.is_scanned else "text layer"
        rprint(f"[bold]--- page {p.number} ({src}) ---[/bold]")
        for b in p.blocks:
            if b.kind == "text":
                rprint(f"[cyan]{b.role:8}[/cyan] col={b.column} | {b.text[:110]}")
            elif b.kind == "table":
                rprint(f"[magenta]table   [/magenta] {b.n_rows}x{b.n_cols} title={b.title!r}")
                rprint("  " + "\n  ".join(json.dumps(r, ensure_ascii=False) for r in b.rows))
            else:
                rprint(f"[yellow]figure  [/yellow] id={b.figure_id!r} labels={b.ocr_labels[:12]} caption={b.caption!r}")


@app.command()
def check() -> None:
    """Verify system packages, Python packages, the API key, cached models and the index; exit 1 on failures."""
    from ae.check import render, run_checks

    res = run_checks()
    typer.echo(render(res))
    raise typer.Exit(code=1 if any(r[0] == "fail" for r in res) else 0)


@app.command()
def smoke() -> None:
    """Fast end-to-end check on one file of each type (index 'smoke'); exit 1 on wrong numbers or declines.

    Cited pages are reported as information only. `make eval` is the full run.
    """
    from ae.smoke import render, run

    result = run()
    typer.echo(render(result))
    raise typer.Exit(code=1 if result.failures else 0)


if __name__ == "__main__":
    app()

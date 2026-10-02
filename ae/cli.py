"""Command line entry points. `make` targets wrap these."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint

app = typer.Typer(no_args_is_help=True, add_completion=False)

CORPUS_DIRS = ("patents", "design_docs", "structured")


def corpus_files(root: Path = Path(".")) -> list[Path]:
    files: list[Path] = []
    for d in CORPUS_DIRS:
        files += sorted(p for p in (root / d).glob("*") if p.is_file() and not p.name.startswith("."))
    return files


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

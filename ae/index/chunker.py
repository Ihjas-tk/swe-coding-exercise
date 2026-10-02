"""ParsedDocument -> retrieval chunks.

Chunk kinds and the evidence behind them (see research notes):
- text: consecutive body blocks within ONE section and ONE page, packed to ~300 tokens
  (cap 450, floor 60), no overlap. Simple size-bounded packing is as good as "smart"
  chunking in controlled studies; never crossing a page means a chunk cites one page.
- claim: each numbered patent claim is its own chunk (claims are self-contained units).
- table_row: one chunk per row, "Table: <title> | <col>: <value> | ..." with units kept
  next to values; row boundaries preserved (large MRR win in table-retrieval studies).
- table: one whole-table markdown chunk (title + rows) so questions that span rows can
  land on the table. Tables above ~400 tokens emit a header chunk instead.
- figure: one chunk per figure: caption, the "FIG. n is a ..." definition, OCR'd labels
  reconciled against the numeral index, and a glossary "160 = control unit" so a numeral
  question can match the figure by text. The VLM description (optional, later stage) is a
  marked field appended when present.
- structured_row: one chunk per CSV/XLSX row, from the loader's side table.

Every chunk carries a deterministic prefix "<doc title> › <section> › p.<n>" (the one
chunking trick with consistent measured gains) and an identifier list for the exact-match
column of the keyword index.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ae.index.identifiers import extract_identifiers
from ae.index.numerals import NumeralIndex, reconcile_labels
from ae.schema import BBox, FigureBlock, Page, ParsedDocument, TableBlock, TextBlock

TARGET_TOKENS, MAX_TOKENS, MIN_TOKENS = 300, 450, 60
BIG_TABLE_TOKENS = 400
CLAIM_RE = re.compile(r"^\s*(\d{1,3})\.\s+(?:The|A|An|In|Apparatus|Method|System)\b")

Kind = Literal["text", "claim", "table_row", "table", "figure", "structured_row"]


class Chunk(BaseModel):
    id: str
    doc: str
    page: int
    kind: Kind
    text: str  # content
    prefix: str  # "doc title › section › p.n"
    section: str | None = None
    bbox: BBox | None = None
    identifiers: list[str] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)
    backend: str | None = None

    @property
    def full_text(self) -> str:
        return f"{self.prefix}\n{self.text}"


def n_tokens(text: str) -> int:
    return int(len(text.split()) * 1.33) + 1


# ----------------------------------------------------------------------------- helpers


def doc_title(doc: ParsedDocument) -> str:
    for b in doc.pages[0].blocks if doc.pages else []:
        if b.kind == "text" and b.role == "heading" and 3 < len(b.text) < 120:
            return b.text
    return Path(doc.doc).stem


def running_lines(doc: ParsedDocument) -> set[str]:
    """Short lines that repeat on 3+ pages (running headers/footers) -> dropped from chunks."""
    if len(doc.pages) < 3:
        return set()
    seen: Counter = Counter()
    for p in doc.pages:
        lines = {re.sub(r"\d+", "#", b.text.strip()) for b in p.blocks if b.kind == "text" and len(b.text) < 80}
        seen.update(lines)
    return {l for l, c in seen.items() if c >= 3}


def row_text(table: TableBlock, row: list[str]) -> str:
    header = table.rows[0] if table.has_header and len(table.rows) > 1 else None
    parts = [f"Table: {table.title}" if table.title else "Table"]
    for i, v in enumerate(row):
        if not v:
            continue
        col = header[i] if header and i < len(header) and header[i] else f"col{i + 1}"
        parts.append(f"{col}: {v}")
    return " | ".join(parts)


# ----------------------------------------------------------------------------- main


def chunk_document(doc: ParsedDocument, numerals: NumeralIndex | None = None) -> list[Chunk]:
    title = doc_title(doc)
    drop = running_lines(doc)
    known = numerals.defined(doc.doc) if numerals else set()
    chunks: list[Chunk] = []
    section: str | None = None
    counter: Counter = Counter()

    def new_id(page: int, kind: str) -> str:
        counter[(page, kind)] += 1
        return f"{doc.doc}#p{page}#{kind}{counter[(page, kind)]}"

    def prefix(page: int) -> str:
        return " › ".join([title] + ([section] if section else []) + [f"p.{page}"])

    def emit(page: int, kind: Kind, text: str, bbox: BBox | None, meta: dict | None = None) -> None:
        chunks.append(
            Chunk(
                id=new_id(page, kind),
                doc=doc.doc,
                page=page,
                kind=kind,
                text=text,
                prefix=prefix(page),
                section=section,
                bbox=bbox,
                identifiers=extract_identifiers(text, known),
                meta=meta or {},
                backend=doc.backend,
            )
        )

    for page in doc.pages:
        buf: list[TextBlock] = []

        def flush() -> None:
            nonlocal buf
            if not buf:
                return
            text = "\n".join(b.text for b in buf)
            bbox = BBox(x0=min(b.bbox.x0 for b in buf), y0=min(b.bbox.y0 for b in buf), x1=max(b.bbox.x1 for b in buf), y1=max(b.bbox.y1 for b in buf))
            emit(page.number, "text", text, bbox)
            buf = []

        for b in page.blocks:
            if b.kind == "text":
                if b.role in ("header", "footer") or re.sub(r"\d+", "#", b.text.strip()) in drop:
                    continue
                if b.role == "heading":
                    flush()
                    section = b.text
                    continue
                if CLAIM_RE.match(b.text) and (section or "").lower().startswith("claim"):
                    flush()
                    emit(page.number, "claim", b.text, b.bbox, {"claim": int(CLAIM_RE.match(b.text).group(1))})
                    continue
                # Pack: start a new chunk when adding this block would exceed the cap, or when
                # the buffer is already at target size.
                if buf and (n_tokens("\n".join(x.text for x in buf)) >= TARGET_TOKENS or n_tokens("\n".join(x.text for x in buf) + b.text) > MAX_TOKENS):
                    flush()
                buf.append(b)
            elif b.kind == "table":
                flush()
                _emit_table(b, page, emit)
            elif b.kind == "figure":
                flush()
                _emit_figure(b, page, doc, numerals, emit)
        flush()
    # Merge undersized text chunks into their predecessor on the same page/section.
    return _merge_small(chunks)


def _emit_table(t: TableBlock, page: Page, emit) -> None:
    meta = {"table_id": t.table_id, "continued": t.continued, "title": t.title}
    body = t.rows[1:] if t.has_header and len(t.rows) > 1 else t.rows
    for i, row in enumerate(body):
        if any(row):
            emit(page.number, "table_row", row_text(t, row), t.bbox, {**meta, "row": i + 1, "cells": row})
    md = (f"{t.title}\n" if t.title else "") + t.to_markdown()
    if n_tokens(md) > BIG_TABLE_TOKENS and t.has_header:
        md = (f"{t.title}\n" if t.title else "") + "Columns: " + " | ".join(t.rows[0]) + f"\n({len(body)} rows)"
    emit(page.number, "table", md, t.bbox, {**meta, "n_rows": len(body)})


def _emit_figure(f: FigureBlock, page: Page, doc: ParsedDocument, numerals: NumeralIndex | None, emit) -> None:
    cap = f.caption or ""
    head = cap if (not f.figure_id or cap.lower().startswith(f.figure_id.lower().rstrip("."))) else f"{f.figure_id}: {cap}".strip(": ")
    parts = [head or "Figure"]
    labels_meta: dict = {}
    if numerals:
        fdef = numerals.figures.get((doc.doc, f.figure_id or ""))
        if fdef and fdef.title.lower() not in (f.caption or "").lower():
            parts.append(f"Description: {fdef.title}")
        rec = reconcile_labels(f.ocr_labels, numerals.defined(doc.doc))
        labels_meta = rec
        present = rec["matched"] + rec["repaired"]
        if present:
            parts.append("Labels: " + ", ".join(present))
            gl = numerals.glossary(doc.doc, present)
            if gl:
                parts.append("; ".join(gl))
        # OCR misses labels (it never read "160" on the gripper figure), so numerals the text
        # defines on this page are added as a marked secondary field. The answerer treats
        # "also defined on this page" as weaker evidence than a confirmed label.
        same_page = [e.numeral for e in numerals.entries.values() if e.doc == doc.doc and page.number in e.pages and e.numeral not in present]
        gl2 = numerals.glossary(doc.doc, same_page)
        if gl2:
            parts.append("Also defined on this page: " + "; ".join(gl2))
    words = [t for t in f.ocr_labels if re.search(r"[A-Za-z]{3,}", t)]
    if words:
        parts.append("Text in figure: " + " ".join(dict.fromkeys(words)))
    if f.description:
        parts.append(f"VLM description: {f.description}")
    emit(page.number, "figure", " | ".join(parts), f.bbox, {"figure_id": f.figure_id, "image_path": f.image_path, "labels": labels_meta, "caption_page": f.caption_page, "has_vlm": bool(f.description)})


def _merge_small(chunks: list[Chunk]) -> list[Chunk]:
    out: list[Chunk] = []
    for c in chunks:
        if c.kind == "text" and out and out[-1].kind == "text" and out[-1].page == c.page and out[-1].section == c.section and n_tokens(c.text) < MIN_TOKENS and n_tokens(out[-1].text + c.text) <= MAX_TOKENS:
            prev = out[-1]
            prev.text += "\n" + c.text
            prev.identifiers = list(dict.fromkeys(prev.identifiers + c.identifiers))
            if prev.bbox and c.bbox:
                prev.bbox = BBox(x0=min(prev.bbox.x0, c.bbox.x0), y0=min(prev.bbox.y0, c.bbox.y0), x1=max(prev.bbox.x1, c.bbox.x1), y1=max(prev.bbox.y1, c.bbox.y1))
            continue
        out.append(c)
    return out


def structured_chunks(db_path: Path) -> list[Chunk]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = []
    for doc, tbl, row, text in conn.execute("SELECT doc, tbl, row, text FROM _structured_rows ORDER BY doc, row"):
        out.append(Chunk(id=f"{doc}#{tbl}#row{row}", doc=doc, page=1, kind="structured_row", text=text, prefix=f"{doc} › {tbl} › row {row}", identifiers=extract_identifiers(text), meta={"table": tbl, "row": row}))
    conn.close()
    return out

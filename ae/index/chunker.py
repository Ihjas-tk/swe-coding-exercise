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
from typing import Any, Literal

from pydantic import BaseModel, Field

from ae.index.identifiers import extract_identifiers
from ae.index.numerals import NumeralIndex, reconcile_labels
from ae.schema import BBox, FigureBlock, Page, ParsedDocument, TableBlock, TextBlock

TARGET_TOKENS = 300
"""A text chunk is closed once it reaches this size ..."""
MAX_TOKENS = 450
"""... and never grows past this one (a block that would overflow starts a new chunk)."""
MIN_TOKENS = 60
"""Text chunks smaller than this are merged into the previous chunk of the same page and section."""
BIG_TABLE_TOKENS = 400
"""Tables larger than this get a header-only "table" chunk (their rows are chunked individually anyway)."""
TOKENS_PER_WORD = 1.33
"""Whitespace words -> subword tokens, a typical ratio for English technical prose."""
TITLE_MAX_CHARS = 120
"""A page-1 heading longer than this is a paragraph, not the document title."""
RUNNING_LINE_MIN_PAGES = 3
"""A short line repeated on at least this many pages is a running header/footer."""
RUNNING_LINE_MAX_CHARS = 80
"""Only lines shorter than this are considered running headers/footers."""
CLAIM_RE = re.compile(r"^\s*(\d{1,3})\.\s+(?:The|A|An|In|Apparatus|Method|System)\b")
"""Start of a numbered patent claim ("12. The gripper of claim 1, ...")."""

Kind = Literal["text", "claim", "table_row", "table", "figure", "structured_row"]


class Chunk(BaseModel):
    """One retrieval unit with its citation location and identifier list."""

    id: str
    doc: str
    page: int
    kind: Kind
    text: str  # content
    prefix: str  # "doc title › section › p.n"
    section: str | None = None
    bbox: BBox | None = None
    identifiers: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    backend: str | None = None

    @property
    def full_text(self) -> str:
        """Prefix and content, as indexed and embedded."""
        return f"{self.prefix}\n{self.text}"


def n_tokens(text: str) -> int:
    """Cheap token estimate (whitespace words x TOKENS_PER_WORD, rounded down, plus one)."""
    return int(len(text.split()) * TOKENS_PER_WORD) + 1


# ----------------------------------------------------------------------------- helpers


def doc_title(doc: ParsedDocument) -> str:
    """First page-1 heading of a plausible title length, or the file stem."""
    for b in doc.pages[0].blocks if doc.pages else []:
        if b.kind == "text" and b.role == "heading" and 3 < len(b.text) < TITLE_MAX_CHARS:
            return b.text
    return Path(doc.doc).stem


def _split_oversized(b: TextBlock) -> list[TextBlock]:
    """Split a text block whose text alone exceeds MAX_TOKENS into sentence-bounded pieces."""
    if n_tokens(b.text) <= MAX_TOKENS:
        return [b]
    sentences = re.split(r"(?<=[.!?;])\s+", b.text)
    pieces: list[str] = []
    current = ""
    for s in sentences:
        if current and n_tokens(current + " " + s) > TARGET_TOKENS:
            pieces.append(current)
            current = s
        else:
            current = (current + " " + s).strip()
        while n_tokens(current) > MAX_TOKENS:  # a single run-on sentence: cut on words
            words = current.split()
            cut = int(TARGET_TOKENS / 1.33)
            pieces.append(" ".join(words[:cut]))
            current = " ".join(words[cut:])
    if current:
        pieces.append(current)
    return [b.model_copy(update={"text": t}) for t in pieces]


def _running_key(text: str) -> str:
    """Line text with digits masked, so "Page 3 of 9" and "Page 4 of 9" compare equal."""
    return re.sub(r"\d+", "#", text.strip())


def running_lines(doc: ParsedDocument) -> set[str]:
    """Digit-masked short lines that repeat on RUNNING_LINE_MIN_PAGES+ pages (running headers/footers)."""
    if len(doc.pages) < RUNNING_LINE_MIN_PAGES:
        return set()
    seen: Counter[str] = Counter()
    for p in doc.pages:
        lines = {_running_key(b.text) for b in p.blocks if b.kind == "text" and len(b.text) < RUNNING_LINE_MAX_CHARS}
        seen.update(lines)
    return {line for line, c in seen.items() if c >= RUNNING_LINE_MIN_PAGES}


def row_text(table: TableBlock, row: list[str]) -> str:
    """Render a table row as 'Table: <title> | <col>: <value> | ...' (empty cells skipped)."""
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
    """Split a parsed document into text, claim, table, table-row and figure chunks.

    Chunk ids are "<doc>#p<page>#<kind><n>" (n counts per page and kind), stable across
    runs for the same extraction. `numerals` (optional) adds figure label glossaries and
    reference-numeral identifiers.
    """
    return _DocumentChunker(doc, numerals).run()


class _DocumentChunker:
    """Walks a document's blocks in reading order, tracking the current section."""

    def __init__(self, doc: ParsedDocument, numerals: NumeralIndex | None) -> None:
        self.doc = doc
        self.numerals = numerals
        self.title = doc_title(doc)
        self.drop = running_lines(doc)
        self.known = numerals.defined(doc.doc) if numerals else set()
        self.chunks: list[Chunk] = []
        self.section: str | None = None
        self.counter: Counter[tuple[int, str]] = Counter()
        self.buffer: list[TextBlock] = []

    def run(self) -> list[Chunk]:
        for page in self.doc.pages:
            self.buffer = []
            for b in page.blocks:
                if b.kind == "text":
                    self._text(b, page.number)
                elif b.kind == "table":
                    self._flush(page.number)
                    self._table(b, page)
                elif b.kind == "figure":
                    self._flush(page.number)
                    self._figure(b, page)
            self._flush(page.number)
        # Merge undersized text chunks into their predecessor on the same page/section.
        return _merge_small(self.chunks)

    # ------------------------------------------------------------------ emitting
    def _new_id(self, page: int, kind: str) -> str:
        self.counter[(page, kind)] += 1
        return f"{self.doc.doc}#p{page}#{kind}{self.counter[(page, kind)]}"

    def _prefix(self, page: int) -> str:
        return " › ".join([self.title] + ([self.section] if self.section else []) + [f"p.{page}"])

    def _emit(self, page: int, kind: Kind, text: str, bbox: BBox | None, meta: dict[str, Any] | None = None) -> None:
        self.chunks.append(
            Chunk(
                id=self._new_id(page, kind),
                doc=self.doc.doc,
                page=page,
                kind=kind,
                text=text,
                prefix=self._prefix(page),
                section=self.section,
                bbox=bbox,
                identifiers=extract_identifiers(text, self.known),
                meta=meta or {},
                backend=self.doc.backend,
            )
        )

    def _flush(self, page_no: int) -> None:
        """Emit the buffered body blocks as one text chunk."""
        buf = self.buffer
        if not buf:
            return
        text = "\n".join(b.text for b in buf)
        bbox = BBox(
            x0=min(b.bbox.x0 for b in buf),
            y0=min(b.bbox.y0 for b in buf),
            x1=max(b.bbox.x1 for b in buf),
            y1=max(b.bbox.y1 for b in buf),
        )
        self._emit(page_no, "text", text, bbox)
        self.buffer = []

    # ------------------------------------------------------------------ block kinds
    def _text(self, b: TextBlock, page_no: int) -> None:
        """Skip running lines, open a section on a heading, emit claims whole, else pack into the buffer."""
        if b.role in ("header", "footer") or _running_key(b.text) in self.drop:
            return
        if b.role == "heading":
            self._flush(page_no)
            self.section = b.text
            return
        if (claim := CLAIM_RE.match(b.text)) and (self.section or "").lower().startswith("claim"):
            self._flush(page_no)
            self._emit(page_no, "claim", b.text, b.bbox, {"claim": int(claim.group(1))})
            return
        # Pack: start a new chunk when the buffer is already at target size, or when adding
        # this block would exceed the cap (estimated on the texts joined without a separator).
        buffered = "\n".join(x.text for x in self.buffer)
        if self.buffer and (n_tokens(buffered) >= TARGET_TOKENS or n_tokens(buffered + b.text) > MAX_TOKENS):
            self._flush(page_no)
        # A single block above the cap (a long OCR paragraph) is split on sentence boundaries so
        # no chunk exceeds MAX_TOKENS; each piece keeps the block's bbox and role.
        for piece in _split_oversized(b):
            if self.buffer and n_tokens("\n".join(x.text for x in self.buffer) + piece.text) > MAX_TOKENS:
                self._flush(page_no)
            self.buffer.append(piece)

    def _table(self, t: TableBlock, page: Page) -> None:
        """One table_row chunk per non-empty body row, then one whole-table (or header-only) chunk."""
        meta = {"table_id": t.table_id, "continued": t.continued, "title": t.title}
        body = t.rows[1:] if t.has_header and len(t.rows) > 1 else t.rows
        for i, row in enumerate(body):
            if any(row):
                self._emit(page.number, "table_row", row_text(t, row), t.bbox, {**meta, "row": i + 1, "cells": row})
        title_line = f"{t.title}\n" if t.title else ""
        md = title_line + t.to_markdown()
        if n_tokens(md) > BIG_TABLE_TOKENS and t.has_header:
            md = title_line + "Columns: " + " | ".join(t.rows[0]) + f"\n({len(body)} rows)"
        self._emit(page.number, "table", md, t.bbox, {**meta, "n_rows": len(body)})

    def _figure(self, f: FigureBlock, page: Page) -> None:
        """One figure chunk: caption, text definition, confirmed labels with glossary, OCR words, VLM text."""
        cap = f.caption or ""
        head = (
            cap
            if (not f.figure_id or cap.lower().startswith(f.figure_id.lower().rstrip(".")))
            else f"{f.figure_id}: {cap}".strip(": ")
        )
        parts = [head or "Figure"]
        labels_meta: dict[str, list[str]] = {}
        if self.numerals:
            labels_meta = self._numeral_parts(self.numerals, f, page, parts)
        words = [t for t in f.ocr_labels if re.search(r"[A-Za-z]{3,}", t)]
        if words:
            parts.append("Text in figure: " + " ".join(dict.fromkeys(words)))
        if f.description:
            parts.append(f"VLM description: {f.description}")
        self._emit(
            page.number,
            "figure",
            " | ".join(parts),
            f.bbox,
            {
                "figure_id": f.figure_id,
                "image_path": f.image_path,
                "labels": labels_meta,
                "caption_page": f.caption_page,
                "has_vlm": bool(f.description),
            },
        )

    def _numeral_parts(
        self, numerals: NumeralIndex, f: FigureBlock, page: Page, parts: list[str]
    ) -> dict[str, list[str]]:
        """Append the figure's definition, labels and glossaries to `parts`; return the label reconciliation."""
        doc_name = self.doc.doc
        figure_def = numerals.figures.get((doc_name, f.figure_id or ""))
        if figure_def and figure_def.title.lower() not in (f.caption or "").lower():
            parts.append(f"Description: {figure_def.title}")
        rec = reconcile_labels(f.ocr_labels, numerals.defined(doc_name))
        present = list(
            dict.fromkeys(rec["matched"] + rec["repaired"])
        )  # an OCR token and its repair may name the same numeral
        if present:
            parts.append("Labels: " + ", ".join(present))
            glossary = numerals.glossary(doc_name, present)
            if glossary:
                parts.append("; ".join(glossary))
        # OCR misses labels (it never read "160" on the gripper figure), so numerals the text
        # defines on this page are added as a marked secondary field. The answerer treats
        # "also defined on this page" as weaker evidence than a confirmed label.
        same_page = [
            e.numeral
            for e in numerals.entries.values()
            if e.doc == doc_name and page.number in e.pages and e.numeral not in present
        ]
        same_page_glossary = numerals.glossary(doc_name, same_page)
        if same_page_glossary:
            parts.append("Also defined on this page: " + "; ".join(same_page_glossary))
        return rec


def _merge_small(chunks: list[Chunk]) -> list[Chunk]:
    """Fold text chunks under MIN_TOKENS into the preceding text chunk of the same page and section."""
    out: list[Chunk] = []
    for c in chunks:
        if (
            c.kind == "text"
            and out
            and out[-1].kind == "text"
            and out[-1].page == c.page
            and out[-1].section == c.section
            and n_tokens(c.text) < MIN_TOKENS
            and n_tokens(out[-1].text + c.text) <= MAX_TOKENS
        ):
            prev = out[-1]
            prev.text += "\n" + c.text
            prev.identifiers = list(dict.fromkeys(prev.identifiers + c.identifiers))
            if prev.bbox and c.bbox:
                prev.bbox = BBox(
                    x0=min(prev.bbox.x0, c.bbox.x0),
                    y0=min(prev.bbox.y0, c.bbox.y0),
                    x1=max(prev.bbox.x1, c.bbox.x1),
                    y1=max(prev.bbox.y1, c.bbox.y1),
                )
            continue
        out.append(c)
    return out


def structured_chunks(db_path: Path) -> list[Chunk]:
    """Return one structured_row chunk per CSV/XLSX row in the loader's `_structured_rows` table (page 1)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = []
    for doc, tbl, row, text in conn.execute("SELECT doc, tbl, row, text FROM _structured_rows ORDER BY doc, row"):
        out.append(
            Chunk(
                id=f"{doc}#{tbl}#row{row}",
                doc=doc,
                page=1,
                kind="structured_row",
                text=text,
                prefix=f"{doc} › {tbl} › row {row}",
                identifiers=extract_identifiers(text),
                meta={"table": tbl, "row": row},
            )
        )
    conn.close()
    return out

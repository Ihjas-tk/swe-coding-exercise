"""Grounded answer generation with chunk-id citations.

The model only ever sees short evidence ids (e1, e2, ...) that we map back to document and
page ourselves; page numbers written by the model are never trusted. not_found is a
first-class answer. Figure crops are attached only for visual questions (counts, locations,
appearance) or when the top evidence is a figure whose text does not contain the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ae.index.chunker import Chunk
from ae.llm import complete_json, image_block
from ae.retrieve.hybrid import RetrievalResult
from ae.retrieve.query import ParsedQuery
from ae.retrieve.sql import SQLResult

ANSWER_MAX_TOKENS = 700  # a one-or-two-sentence answer plus citations and a short reasoning field
SQL_MAX_COLS = 15  # wider SQL results are cut; the model needs the asked-for columns, not 500 sensor columns
SQL_MAX_ROWS = 25  # more rows are summarised as a count; aggregates should return few rows anyway
MAX_IMAGES = 2  # figure crops attached per question (image tokens are expensive)

SYSTEM = """You answer engineering questions strictly from the EVIDENCE provided. The evidence comes from patents, design documents, a bill of materials and a test log.
Rules:
1. Use only the evidence. If it does not contain the answer, set "not_found": true and leave "answer" as "Not found in the provided documents." Guessing is worse than declining.
2. Cite the evidence ids (e.g. "e3") that support each fact. Cite only ids from the list. Prefer the most specific evidence (a table row over a whole table; a figure over prose about it).
3. Copy numbers, units and identifiers exactly as they appear in the evidence.
4. Answer in one or two sentences; include the key value(s) and qualifiers (e.g. "320 A continuous, 450 A for 10 s").
5. A "Reference numeral lookup" block is authoritative for "what is labelled N" questions. A "Structured query result" block is authoritative for counts/aggregates and must be cited by its id.
6. If an image is attached, you may read values from it; say so in the answer only if the text evidence disagrees.
Reply with ONE JSON object:
{"sufficient": true|false, "answer": "...", "citations": ["e1"], "not_found": false, "confidence": "high|medium|low", "reasoning": "one sentence"}"""


@dataclass
class EvidenceItem:
    """One numbered piece of evidence shown to the model, mapped back to document and page by us."""

    eid: str
    chunk: Chunk | None
    text: str
    doc: str
    page: int
    kind: str
    extra_pages: list[int] = field(default_factory=list)  # caption page / other pages of a split table


@dataclass
class Draft:
    """The model's answer before verification."""

    answer: str
    citations: list[str]
    not_found: bool
    sufficient: bool
    confidence: str
    reasoning: str
    raw: dict
    evidence: list[EvidenceItem]


def build_evidence(
    result: RetrievalResult, sql: SQLResult | None, numeral_block: dict | None, split_tables: dict[str, list[int]]
) -> list[EvidenceItem]:
    """Assign evidence ids e1..en: numeral lookup first, then the SQL result, then retrieved chunks in page order.

    `split_tables` maps a table id to every page it spans, so a cited row carries the other
    pages of its table as `extra_pages` (citation policy for tables split across pages).
    """
    items: list[EvidenceItem] = []
    if numeral_block:
        items.append(_numeral_item(f"e{len(items) + 1}", numeral_block))
    if sql and sql.sql and not sql.error:
        items.append(_sql_item(f"e{len(items) + 1}", sql))
    for p in result.pages:
        for c in p.chunks:
            items.append(_chunk_item(f"e{len(items) + 1}", c, split_tables))
    return items


def _numeral_item(eid: str, block: dict) -> EvidenceItem:
    """Wrap the numeral-index lookup as one authoritative evidence item."""
    return EvidenceItem(
        eid, None, block["text"], block["doc"], block["page"], "numeral_lookup", block.get("extra_pages", [])
    )


def _sql_item(eid: str, sql: SQLResult) -> EvidenceItem:
    """Render the SQL and its (capped) rows as one evidence item, cited as the source file's page 1."""
    ncol = min(len(sql.columns), SQL_MAX_COLS)
    head = ", ".join(sql.columns[:ncol]) + (
        f", ... ({len(sql.columns) - ncol} more columns)" if len(sql.columns) > ncol else ""
    )
    rows = "\n".join(", ".join(str(v) for v in r[:ncol]) for r in sql.rows[:SQL_MAX_ROWS])
    more = f"\n... ({len(sql.rows)} rows total)" if len(sql.rows) > SQL_MAX_ROWS else ""
    text = f"SQL: {sql.sql}\nResult columns: {head}\n{rows}{more}"
    return EvidenceItem(eid, None, text, sql.docs[0] if sql.docs else "structured", 1, "sql_result")


def _chunk_item(eid: str, c: Chunk, split_tables: dict[str, list[int]]) -> EvidenceItem:
    """Wrap a retrieved chunk; figures add their caption page, split tables their other pages."""
    extra: list[int] = []
    if c.kind == "figure" and c.meta.get("caption_page") and c.meta["caption_page"] != c.page:
        extra.append(c.meta["caption_page"])
    if c.kind in ("table_row", "table") and c.meta.get("table_id") in split_tables:
        extra += [pg for pg in split_tables[c.meta["table_id"]] if pg != c.page]
    return EvidenceItem(eid, c, c.full_text, c.doc, c.page, c.kind, extra)


def _figure_images(items: list[EvidenceItem], pq: ParsedQuery, max_images: int = MAX_IMAGES) -> list[dict]:
    """Image blocks for figure evidence of a visual question, skipping figures other than the one named."""
    blocks: list[dict] = []
    if not pq.visual:
        return blocks
    for it in items:
        path = it.chunk.meta.get("image_path") if it.kind == "figure" and it.chunk else None
        if not path or not Path(path).exists():
            continue
        fig = it.chunk.meta.get("figure_id") if it.chunk else None
        if pq.fig_id and fig and pq.fig_id.split()[-1] != fig.split()[-1]:
            continue
        blocks.append({"type": "text", "text": f"Image for evidence {it.eid} ({it.doc} p.{it.page}):"})
        blocks.append(image_block(path))
        if len(blocks) >= 2 * max_images:
            break
    return blocks


def generate(pq: ParsedQuery, items: list[EvidenceItem]) -> Draft:
    """Ask the LLM for a grounded answer over the evidence (figure crops attached for visual questions)."""
    ev_text = "\n\n".join(f"[{it.eid}] ({it.doc}, page {it.page}, {it.kind})\n{it.text}" for it in items)
    scope = (
        f"\nThe question names the document {pq.doc}; only evidence from it can answer the question." if pq.doc else ""
    )
    content: list[dict] = [{"type": "text", "text": f"QUESTION: {pq.question}{scope}\n\nEVIDENCE:\n{ev_text}"}]
    content += _figure_images(items, pq)
    out = complete_json(SYSTEM, content, max_tokens=ANSWER_MAX_TOKENS)
    cits = [c for c in out.get("citations", []) if isinstance(c, str)]
    return Draft(
        answer=str(out.get("answer", "")).strip(),
        citations=cits,
        not_found=bool(out.get("not_found", False)),
        sufficient=bool(out.get("sufficient", not out.get("not_found", False))),
        confidence=str(out.get("confidence", "medium")),
        reasoning=str(out.get("reasoning", "")),
        raw=out,
        evidence=items,
    )

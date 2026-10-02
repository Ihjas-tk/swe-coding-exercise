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
    eid: str
    chunk: Chunk | None
    text: str
    doc: str
    page: int
    kind: str
    extra_pages: list[int] = field(default_factory=list)  # caption page / other pages of a split table


@dataclass
class Draft:
    answer: str
    citations: list[str]
    not_found: bool
    sufficient: bool
    confidence: str
    reasoning: str
    raw: dict
    evidence: list[EvidenceItem]


def build_evidence(result: RetrievalResult, sql: SQLResult | None, numeral_block: dict | None, split_tables: dict[str, list[int]]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    n = 0
    if numeral_block:
        n += 1
        items.append(EvidenceItem(f"e{n}", None, numeral_block["text"], numeral_block["doc"], numeral_block["page"], "numeral_lookup", numeral_block.get("extra_pages", [])))
    if sql and sql.sql and not sql.error:
        n += 1
        ncol = min(len(sql.columns), 15)  # cap very wide results
        head = ", ".join(sql.columns[:ncol]) + (f", ... ({len(sql.columns) - ncol} more columns)" if len(sql.columns) > ncol else "")
        rows = "\n".join(", ".join(str(v) for v in r[:ncol]) for r in sql.rows[:25])
        more = f"\n... ({len(sql.rows)} rows total)" if len(sql.rows) > 25 else ""
        items.append(EvidenceItem(f"e{n}", None, f"SQL: {sql.sql}\nResult columns: {head}\n{rows}{more}", sql.docs[0] if sql.docs else "structured", 1, "sql_result"))
    for p in result.pages:
        for c in p.chunks:
            n += 1
            extra = []
            if c.kind == "figure" and c.meta.get("caption_page") and c.meta["caption_page"] != c.page:
                extra.append(c.meta["caption_page"])
            if c.kind in ("table_row", "table") and c.meta.get("table_id") in split_tables:
                extra += [pg for pg in split_tables[c.meta["table_id"]] if pg != c.page]
            items.append(EvidenceItem(f"e{n}", c, c.full_text, c.doc, c.page, c.kind, extra))
    return items


def _figure_images(items: list[EvidenceItem], pq: ParsedQuery, max_images: int = 2) -> list[dict]:
    blocks = []
    if not pq.visual:
        return blocks
    for it in items:
        if it.kind == "figure" and it.chunk and it.chunk.meta.get("image_path") and Path(it.chunk.meta["image_path"]).exists():
            if pq.fig_id and it.chunk.meta.get("figure_id") and pq.fig_id.split()[-1] != it.chunk.meta["figure_id"].split()[-1]:
                continue
            blocks.append({"type": "text", "text": f"Image for evidence {it.eid} ({it.doc} p.{it.page}):"})
            blocks.append(image_block(it.chunk.meta["image_path"]))
            if len(blocks) >= 2 * max_images:
                break
    return blocks


def generate(pq: ParsedQuery, items: list[EvidenceItem]) -> Draft:
    ev_text = "\n\n".join(f"[{it.eid}] ({it.doc}, page {it.page}, {it.kind})\n{it.text}" for it in items)
    scope = f"\nThe question names the document {pq.doc}; only evidence from it can answer the question." if pq.doc else ""
    content: list[dict] = [{"type": "text", "text": f"QUESTION: {pq.question}{scope}\n\nEVIDENCE:\n{ev_text}"}]
    content += _figure_images(items, pq)
    out = complete_json(SYSTEM, content, max_tokens=700)
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

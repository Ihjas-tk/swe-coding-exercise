"""Question parsing and routing.

Rules first (a TF-IDF/rule router beat embedding classifiers in routing benchmarks):
- identifiers in the question are resolved against per-document aliases (patent numbers,
  product names from file names and page-1 headings). A patent number pins the document
  hard; a product name is a soft boost because it also appears in BOM/test-log rows.
- "labelled 160 in FIG. 2" / "component 18" / "numeral 204"  -> numeral route
- aggregate words (how many / count / highest / total / average / distinct ...) together
  with structured vocabulary (test log, BOM, part number, unit cost, failed, serial ...)
  -> sql route
- otherwise hybrid. Hybrid always runs as well, so a wrong route cannot fail silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ae.index.identifiers import FIG_RE, extract_identifiers, norm
from ae.index.store import IndexStore

NUMERAL_Q_RE = re.compile(r"\b(?:label(?:l?ed)?|numeral|reference|component|element|part|item|number)\s+(?:as\s+|number\s+)?(\d{1,4}[a-z]?)\b", re.I)
NUMERAL_Q2_RE = re.compile(r"\b(\d{1,4}[a-z]?)\s+(?:in|of|on)\s+FIG", re.I)
AGG_RE = re.compile(r"\b(how many|count|number of|total|sum|average|mean|highest|lowest|maximum|minimum|most expensive|cheapest|largest|smallest|distinct|unique|percentage|percent|rate of|ratio of|earliest|latest|first|last)\b", re.I)
STRUCT_VOCAB = re.compile(r"\b(test log|test runs?|tests?|bill of materials|bom|part numbers?|line items?|unit cost|extended cost|quantity|serial|units? (were|was) tested|operator|failed|passed|pass|fail|measured|nominal|tolerance|manufacturer|supplier|cost)\b", re.I)
VISUAL_RE = re.compile(r"\b(how many|count|shown|visible|depicted|where is|located|position|colou?r|shape|appearance|look like|arranged|orientation)\b", re.I)


@dataclass
class ParsedQuery:
    question: str
    identifiers: list[str] = field(default_factory=list)
    doc: str | None = None  # hard restriction (patent number matched exactly one document)
    doc_boost: list[str] = field(default_factory=list)  # soft: product name mentioned
    fig_id: str | None = None
    numerals: list[str] = field(default_factory=list)
    route: str = "hybrid"  # numeral | sql | hybrid
    visual: bool = False  # attach figure image at answer time
    reasons: list[str] = field(default_factory=list)


def doc_aliases(store: IndexStore) -> dict[str, set[str]]:
    """doc -> normalised alias tokens (file name parts, page-1 headings, patent number variants)."""
    out: dict[str, set[str]] = {}
    for doc in store.docs():
        al = set()
        stem = re.sub(r"\.(pdf|docx|csv|xlsx)$", "", doc, flags=re.I)
        for part in re.split(r"[_\s]+", stem):
            n = norm(part)
            if len(n) >= 4:
                al.add(n)
        m = re.match(r"(US)?(\d{7,11})([AB]\d)?", stem.split("_")[0], re.I)
        if m:
            digits = m.group(2)
            al.update({"us" + digits, digits, "us" + digits + (m.group(3) or "").lower()})
        for c in store.page_chunks(doc, 1)[:3]:
            for ident in c.identifiers:
                if re.fullmatch(r"[a-z]{2,}[a-z0-9]*(-?[a-z0-9]+)+", ident) and len(ident) >= 5:
                    al.add(norm(ident))
        out[doc] = al
    return out


def parse_query(question: str, store: IndexStore, aliases: dict[str, set[str]] | None = None) -> ParsedQuery:
    aliases = aliases or doc_aliases(store)
    pq = ParsedQuery(question=question, identifiers=store.query_identifiers(question))
    q_norm = {norm(i) for i in pq.identifiers}
    # document resolution
    patent_like = {i for i in q_norm if re.fullmatch(r"us\d{7,11}[ab]?\d?", i) or re.fullmatch(r"\d{7,11}", i)}
    hard = [d for d, al in aliases.items() if al & patent_like or any(p.rstrip("ab0123456789") and (p in al) for p in patent_like)]
    hard = [d for d, al in aliases.items() if any(_patent_match(p, al) for p in patent_like)]
    if len(hard) == 1:
        pq.doc = hard[0]
        pq.reasons.append(f"patent number -> {pq.doc}")
    soft = [d for d, al in aliases.items() if (al & q_norm) and d not in hard and not d.lower().endswith((".csv", ".xlsx"))]
    pq.doc_boost = soft
    # figure / numeral
    m = FIG_RE.search(question)
    if m:
        pq.fig_id = f"FIG. {m.group(1).upper()}"
    nums = [m.group(1) for m in NUMERAL_Q_RE.finditer(question)] + [m.group(1) for m in NUMERAL_Q2_RE.finditer(question)]
    known = store.known_numerals()
    pq.numerals = [n for n in dict.fromkeys(nums) if n in known or re.search(r"[a-z]$", n)]
    if pq.numerals:
        pq.route = "numeral"
        pq.reasons.append(f"numeral pattern {pq.numerals}")
    elif AGG_RE.search(question) and STRUCT_VOCAB.search(question):
        pq.route = "sql"
        pq.reasons.append("aggregate + structured vocabulary")
    pq.visual = bool(VISUAL_RE.search(question)) and bool(pq.fig_id or re.search(r"\b(figure|fig\.?|drawing|diagram|rendering|photo|image)\b", question, re.I))
    return pq


def _patent_match(p: str, aliases: set[str]) -> bool:
    if p in aliases:
        return True
    digits = re.sub(r"\D", "", p)
    return bool(digits) and any(re.sub(r"\D", "", a) == digits and a.startswith("us") for a in aliases)

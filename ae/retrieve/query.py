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

from ae.index.identifiers import FIG_RE, norm
from ae.index.store import IndexStore
from ae.log import get_logger

log = get_logger(__name__)

# --- numeral route ---------------------------------------------------------------------
# "labelled 160", "reference numeral 18", "component 204a", "part number 12": a cue word
# followed by a 1-4 digit numeral (optional letter suffix as in patents' "204a").
NUMERAL_AFTER_CUE_RE = re.compile(
    r"\b(?:label(?:l?ed)?|numeral|reference|component|element|part|item|number)\s+(?:as\s+|number\s+)?(\d{1,4}[a-z]?)\b",
    re.I,
)
# "160 in FIG. 2" / "18 of FIG 3": a bare numeral directly tied to a figure.
NUMERAL_BEFORE_FIG_RE = re.compile(r"\b(\d{1,4}[a-z]?)\s+(?:in|of|on)\s+FIG", re.I)
# A numeral candidate with a letter suffix ("204a") is accepted even if the numeral index
# does not list it: suffixed numerals are never part numbers or quantities.
LETTER_SUFFIX_RE = re.compile(r"[a-z]$")

# --- sql route -------------------------------------------------------------------------
# Aggregate intent: counting, extremes, totals, averages, ratios, ordering in time.
AGGREGATE_RE = re.compile(
    r"\b(how many|count|number of|total|sum|average|mean|highest|lowest|maximum|minimum|most expensive|cheapest|largest|smallest|distinct|unique|percentage|percent|rate of|ratio of|earliest|latest|first|last)\b",
    re.I,
)
# Vocabulary of the structured files (test log, bill of materials). Aggregate words alone
# are not enough ("maximum discharge current" is a spec-table lookup), so both must match.
STRUCTURED_VOCAB_RE = re.compile(
    r"\b(test log|test runs?|tests?|bill of materials|bom|part numbers?|line items?|unit cost|extended cost|quantity|serial|units? (were|was) tested|operator|failed|passed|pass|fail|measured|nominal|tolerance|manufacturer|supplier|cost)\b",
    re.I,
)

# --- visual questions (attach the figure crop at answer time) ---------------------------
# Questions whose answer is read off the picture: counts, positions, appearance.
VISUAL_CUE_RE = re.compile(
    r"\b(how many|count|shown|visible|depicted|where is|located|position|colou?r|shape|appearance|look like|arranged|orientation)\b",
    re.I,
)
# A visual cue only counts when the question is about a figure (named FIG. id or one of these words).
FIGURE_WORD_RE = re.compile(r"\b(figure|fig\.?|drawing|diagram|rendering|photo|image)\b", re.I)

# --- document aliases ------------------------------------------------------------------
# Normalised patent numbers as they appear among a question's identifiers: "us8485576b2" or bare digits.
PATENT_ID_RE = re.compile(r"us\d{7,11}[ab]?\d?")
PATENT_DIGITS_RE = re.compile(r"\d{7,11}")
# Corpus file extensions, stripped from a document name before splitting it into alias parts.
FILE_EXT_RE = re.compile(r"\.(pdf|docx|csv|xlsx)$", re.I)
# File-name part separators ("EV-BMS-100_design_document" -> "EV-BMS-100", "design", "document").
NAME_SEP_RE = re.compile(r"[_\s]+")
# A patent file name starts with its publication number: optional "US", digits, optional kind code ("B2", "A1").
PATENT_STEM_RE = re.compile(r"(US)?(\d{7,11})([AB]\d)?", re.I)
# Product-style identifiers on page 1 ("ev-bms-100", "rja-40"): letters, then hyphenated alphanumeric parts.
PRODUCT_ID_RE = re.compile(r"[a-z]{2,}[a-z0-9]*(-?[a-z0-9]+)+")
NON_DIGIT_RE = re.compile(r"\D")

MIN_ALIAS_LEN = 4  # shorter file-name parts ("doc", "v2") match too many questions
MIN_PRODUCT_ID_LEN = 5  # shorter page-1 identifiers are generic ("ab-1")
PAGE1_ALIAS_CHUNKS = 3  # the title block: the first chunks of page 1
STRUCTURED_SUFFIXES = (".csv", ".xlsx")  # never soft-boosted: product names recur in their rows


@dataclass
class ParsedQuery:
    """What the parser learned about a question and the route it chose."""

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
    """Doc -> normalised alias tokens (file name parts, page-1 headings, patent number variants)."""
    out: dict[str, set[str]] = {}
    for doc in store.docs():
        al: set[str] = set()
        stem = FILE_EXT_RE.sub("", doc)
        for part in NAME_SEP_RE.split(stem):
            n = norm(part)
            if len(n) >= MIN_ALIAS_LEN:
                al.add(n)
        m = PATENT_STEM_RE.match(stem.split("_")[0])
        if m:
            digits = m.group(2)
            al.update({"us" + digits, digits, "us" + digits + (m.group(3) or "").lower()})
        for c in store.page_chunks(doc, 1)[:PAGE1_ALIAS_CHUNKS]:
            for ident in c.identifiers:
                if PRODUCT_ID_RE.fullmatch(ident) and len(ident) >= MIN_PRODUCT_ID_LEN:
                    al.add(norm(ident))
        out[doc] = al
    return out


def parse_query(question: str, store: IndexStore, aliases: dict[str, set[str]] | None = None) -> ParsedQuery:
    """Resolve documents, figures and numerals in a question and pick its route.

    Numeral wins over sql: a "what is labelled 12" question can contain aggregate words.
    """
    aliases = aliases or doc_aliases(store)
    pq = ParsedQuery(question=question, identifiers=store.query_identifiers(question))
    _resolve_documents(pq, aliases)
    m = FIG_RE.search(question)
    if m:
        pq.fig_id = f"FIG. {m.group(1).upper()}"
    pq.numerals = _known_numerals(question, store.known_numerals())
    if pq.numerals:
        pq.route = "numeral"
        pq.reasons.append(f"numeral pattern {pq.numerals}")
    elif AGGREGATE_RE.search(question) and STRUCTURED_VOCAB_RE.search(question):
        pq.route = "sql"
        pq.reasons.append("aggregate + structured vocabulary")
    pq.visual = bool(VISUAL_CUE_RE.search(question)) and bool(pq.fig_id or FIGURE_WORD_RE.search(question))
    log.debug(f"route={pq.route} doc={pq.doc} boost={pq.doc_boost} fig={pq.fig_id} numerals={pq.numerals}")
    return pq


def _resolve_documents(pq: ParsedQuery, aliases: dict[str, set[str]]) -> None:
    """Set the hard document (a patent number naming exactly one doc) and the soft-boosted docs."""
    q_norm = {norm(i) for i in pq.identifiers}
    patent_like = {i for i in q_norm if PATENT_ID_RE.fullmatch(i) or PATENT_DIGITS_RE.fullmatch(i)}
    hard = [d for d, al in aliases.items() if any(_patent_match(p, al) for p in patent_like)]
    if len(hard) == 1:
        pq.doc = hard[0]
        pq.reasons.append(f"patent number -> {pq.doc}")
    pq.doc_boost = [
        d
        for d, al in aliases.items()
        if (al & q_norm) and d not in hard and not d.lower().endswith(STRUCTURED_SUFFIXES)
    ]


def _known_numerals(question: str, known: set[str]) -> list[str]:
    """Numerals the question asks about that the numeral index knows (or that carry a letter suffix)."""
    nums = [m.group(1) for m in NUMERAL_AFTER_CUE_RE.finditer(question)] + [
        m.group(1) for m in NUMERAL_BEFORE_FIG_RE.finditer(question)
    ]
    return [n for n in dict.fromkeys(nums) if n in known or LETTER_SUFFIX_RE.search(n)]


def _patent_match(p: str, aliases: set[str]) -> bool:
    """Match a question's patent number to a doc alias, ignoring kind codes ("B2") on either side."""
    if p in aliases:
        return True
    digits = NON_DIGIT_RE.sub("", p)
    return bool(digits) and any(NON_DIGIT_RE.sub("", a) == digits and a.startswith("us") for a in aliases)

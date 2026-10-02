"""Reference-numeral index for patent-style documents.

Patents define every drawing label in the specification text: "a control unit 160",
"valves 74 and 76". Published OCR/VLM results on patent drawings are too weak to be the
source of truth (best competition part-label F1 ~71%), so the mapping numeral -> part
name is built from the text, and the figure crops' OCR only *confirms* that a numeral is
present in a given figure.

Three things are extracted per document:
1. numerals: (numeral -> canonical name, name variants, pages, count)
   via a regex for "<noun phrase> <numeral>" with list expansion ("wings 14, 16",
   "valves 74 and 76") and a stoplist that rejects "claim 1", "FIG. 2", "all 96",
   "10 Hz", dates and patent numbers.
2. figure definitions: "FIG. 2 is a block diagram of ..." -> (fig id -> page, title)
3. figure label reconciliation: OCR tokens from a figure crop matched to the document's
   defined numerals; a token is repaired only when exactly one defined numeral explains
   it (a dropped leading digit, or a 1/7, 6/8, 0/8 confusion). The source of every label
   (ocr | repaired | vlm) is kept.

Design docs have no reference numerals; the stoplist and the "defined at least once as
<noun> <numeral>" requirement keep false positives low there (measured in tests).
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ae.schema import ParsedDocument

NUMERAL = r"\d{1,4}[a-z]?'{0,2}"
WORD = r"[A-Za-z][A-Za-z\-/]*"
# up to 4 words then a numeral (and optionally a list of further numerals)
DEF_RE = re.compile(rf"((?:{WORD}\s+){{1,4}})({NUMERAL})((?:\s*,\s*{NUMERAL})*)(?:\s*,?\s*(?:and|or)\s+({NUMERAL}))?(?![\w.:/,-]*\d)")
FIG_DEF_RE = re.compile(r"\b(FIG(?:URE)?\.?|Figure)\s*(\d{1,3}[A-Za-z]?)\s+(?:is|shows|illustrates|depicts|represents)\s+(an?\s+|the\s+)?([^.;]{3,200})", re.I)
UNIT_AFTER_RE = re.compile(r"^\s*(?:mA|kA|A|mV|kV|V|W|kW|Hz|kHz|MHz|°C|°F|mm|cm|km|m|kg|g|N·m|Nm|rpm|ms|s|kbps|Mbps|arcmin|dB|%|bar|psi|kOhm|Ω|x|×|-bit|bit|hours?|days?|years?|times?|percent|seconds?|minutes?)\b")

STOP_BEFORE = {
    "claim", "claims", "fig", "figs", "figure", "figures", "page", "pages", "step", "steps", "rev", "no", "nos", "us", "about", "approximately",
    "than", "least", "all", "at", "of", "to", "in", "on", "by", "for", "with", "from", "is", "are", "was", "were", "be", "and", "or", "the",
    "a", "an", "over", "under", "within", "between", "every", "each", "per", "up", "only", "some", "any", "has", "have", "having", "comprising",
    "includes", "include", "including", "example", "section", "part", "number", "col", "column", "line", "lines", "patent", "serial", "version",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "january", "february", "march", "april", "june",
    "july", "august", "september", "october", "november", "december", "filed", "dated", "class", "cl", "rated", "total", "approx", "nominal",
    "minimum", "maximum", "min", "max", "ratio", "resolution",
}
STRIP_LEADING = {
    "a", "an", "the", "said", "of", "each", "its", "their", "respective", "one", "or", "more", "at", "least", "two", "three", "four", "plurality",
    "pair", "set", "to", "and", "with", "by", "in", "on", "from", "for", "as", "such", "is", "are", "be", "that", "which", "when", "where", "via",
    "through", "into", "onto", "about", "comprising", "comprises", "includes", "including", "has", "having", "further", "also", "so", "then",
    "first", "second", "third", "another", "other", "same", "corresponding", "respective", "adjacent", "this", "these", "those", "shown",
}


@dataclass
class NumeralEntry:
    doc: str
    numeral: str
    name: str
    variants: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    count: int = 0


@dataclass
class FigureDef:
    doc: str
    fig_id: str  # "FIG. 2"
    page: int
    title: str


def _clean_name(words: list[str]) -> str | None:
    w = [x.lower().strip("-/") for x in words if x.strip("-/")]
    # The part name is whatever follows the last function word: "under control of gate" -> "gate",
    # "energized by a wire" -> "wire", "a distal end" -> "distal end".
    cut = max((i for i, x in enumerate(w) if x in STRIP_LEADING), default=-1)
    w = w[cut + 1 :]
    if not w or w[-1] in STOP_BEFORE:
        return None
    if any(c.isdigit() for c in w[-1]):
        return None
    return " ".join(w[-3:])  # at most three words: "left thrust-generating device"


def extract_numerals(doc: ParsedDocument) -> list[NumeralEntry]:
    names: dict[str, Counter] = defaultdict(Counter)
    pages: dict[str, set[int]] = defaultdict(set)
    for page in doc.pages:
        for b in page.blocks:
            if b.kind != "text" or b.role in ("header", "footer"):
                continue
            text = b.text
            for m in DEF_RE.finditer(text):
                head = m.group(1).split()
                if head and head[-1].lower().rstrip(".") in STOP_BEFORE:
                    continue
                tail = text[m.end() :]
                if UNIT_AFTER_RE.match(tail) or re.match(r"-[A-Za-z]", tail):
                    continue  # "10 Hz", "16-bit", "402-style" are not reference numerals
                first = m.group(2)
                if int(re.match(r"\d+", first).group(0)) >= 2000 and first.isdigit():
                    continue  # years, patent numbers
                name = _clean_name(head)
                if not name:
                    continue
                nums = [first] + re.findall(NUMERAL, m.group(3) or "") + ([m.group(4)] if m.group(4) else [])
                if len(nums) > 1 and not name.endswith("s"):
                    pass  # "wings 14, 16" -> plural already; singular names with lists are kept as-is
                for n in nums:
                    names[n][name] += 1
                    pages[n].add(page.number)
    # Document-level gate: a document "bears numerals" when it defines figures ("FIG. n is
    # a ...") or repeats several numerals. In such documents every definition is kept, even
    # single occurrences ("an electromagnet 120a"). Elsewhere (design docs) only repeated
    # or suffixed numerals survive, which keeps stray matches like "Rev. C" / "about 96" out.
    repeated = sum(1 for ctr in names.values() if sum(ctr.values()) >= 2)
    bears = bool(extract_figure_defs(doc)) or repeated >= 3
    out = []
    for n, ctr in names.items():
        canon, _ = ctr.most_common(1)[0]
        total = sum(ctr.values())
        if not bears and total < 2 and n.isdigit():
            continue
        out.append(NumeralEntry(doc.doc, n, canon, [v for v in ctr if v != canon], sorted(pages[n]), total))
    return sorted(out, key=lambda e: (len(e.numeral.rstrip("abc'")), e.numeral))


def extract_figure_defs(doc: ParsedDocument) -> list[FigureDef]:
    out, seen = [], set()
    for page in doc.pages:
        for b in page.blocks:
            if b.kind != "text":
                continue
            for m in FIG_DEF_RE.finditer(b.text):
                fid = f"FIG. {m.group(2).upper()}"
                if fid in seen:
                    continue
                seen.add(fid)
                out.append(FigureDef(doc.doc, fid, page.number, m.group(4).strip()))
    return out


def reconcile_labels(ocr_tokens: list[str], defined: set[str]) -> dict[str, list[str]]:
    """Match OCR tokens from a figure crop to the document's defined numerals.

    Returns {"matched": [...], "repaired": [...], "unmatched": [...]}. A repair is applied only
    when exactly one defined numeral explains the token.
    """
    matched, repaired, unmatched = [], [], []
    confusions = {"1": "7", "7": "1", "6": "8", "8": "6", "0": "8"}
    for tok in ocr_tokens:
        t = tok.strip(".,;:()[]")
        if not re.fullmatch(r"\d{1,4}[a-z]?", t):
            continue
        if t in defined:
            matched.append(t)
            continue
        cands = set()
        for d in defined:
            if d.endswith(t) and len(d) == len(t) + 1:  # dropped leading digit: "20" -> "120"
                cands.add(d)
        for i, ch in enumerate(t):
            if ch in confusions:
                alt = t[:i] + confusions[ch] + t[i + 1 :]
                if alt in defined:
                    cands.add(alt)
        if len(cands) == 1:
            repaired.append(cands.pop())
        else:
            unmatched.append(t)
    return {"matched": sorted(set(matched)), "repaired": sorted(set(repaired)), "unmatched": sorted(set(unmatched))}


class NumeralIndex:
    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], NumeralEntry] = {}
        self.figures: dict[tuple[str, str], FigureDef] = {}

    def add_document(self, doc: ParsedDocument) -> None:
        for e in extract_numerals(doc):
            self.entries[(e.doc, e.numeral)] = e
        for f in extract_figure_defs(doc):
            self.figures[(f.doc, f.fig_id)] = f

    def defined(self, doc: str) -> set[str]:
        return {n for d, n in self.entries if d == doc}

    def lookup(self, doc: str, numeral: str) -> NumeralEntry | None:
        return self.entries.get((doc, numeral))

    def glossary(self, doc: str, numerals: list[str]) -> list[str]:
        """'160 = control unit' lines for the numerals present in a figure."""
        out = []
        for n in numerals:
            e = self.entries.get((doc, n))
            if e:
                out.append(f"{n} = {e.name}")
        return out

    def save(self, db_path: Path) -> None:
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS numerals (doc TEXT, numeral TEXT, name TEXT, variants TEXT, pages TEXT, count INTEGER, PRIMARY KEY (doc, numeral))")
        conn.execute("CREATE TABLE IF NOT EXISTS figure_defs (doc TEXT, fig_id TEXT, page INTEGER, title TEXT, PRIMARY KEY (doc, fig_id))")
        for e in self.entries.values():
            conn.execute("INSERT OR REPLACE INTO numerals VALUES (?,?,?,?,?,?)", (e.doc, e.numeral, e.name, "|".join(e.variants), ",".join(map(str, e.pages)), e.count))
        for f in self.figures.values():
            conn.execute("INSERT OR REPLACE INTO figure_defs VALUES (?,?,?,?)", (f.doc, f.fig_id, f.page, f.title))
        conn.commit()
        conn.close()

    @classmethod
    def load(cls, db_path: Path) -> "NumeralIndex":
        idx = cls()
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        for doc, n, name, variants, pages, count in conn.execute("SELECT * FROM numerals"):
            idx.entries[(doc, n)] = NumeralEntry(doc, n, name, [v for v in variants.split("|") if v], [int(p) for p in pages.split(",") if p], count)
        for doc, fid, page, title in conn.execute("SELECT * FROM figure_defs"):
            idx.figures[(doc, fid)] = FigureDef(doc, fid, page, title)
        conn.close()
        return idx

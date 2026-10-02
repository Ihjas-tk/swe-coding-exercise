"""Exact identifiers in questions and chunks (part numbers, patent numbers, figure ids,
reference numerals, value+unit). Shared by the chunker (identifier column for BM25),
the query parser (routing / ID boost) and the verifier (numbers in answers must appear
in cited text).

Every identifier is stored in two normalised forms so BM25 can match however the user
types it: "fvt-esc-40a" (hyphens kept) and "fvtesc40a" (alphanumerics only).
"""
from __future__ import annotations

import re

PART_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)+\b")  # FVT-ESC-40A, EV-BMS-100, BMS-CONN-M12-4
PATENT_RE = re.compile(r"\bUS\s?-?\s?(?:\d{1,2}[,/]?\d{3}[,/]?\d{3,4}|\d{4}/\d{7})\s?(?:[AB]\d)?\b", re.I)  # US 8,485,576 B2 / US2017/0320570A1
FIG_RE = re.compile(r"\b(?:FIG(?:URE)?S?\.?|Figures?)\s*(\d{1,3}[A-Za-z]?)\b", re.I)
UNIT = r"(?:mA|kA|A|mV|kV|V|mW|kW|W|Hz|kHz|MHz|GHz|°C|°F|mm|cm|km|m|kg|g|N·m|N\.m|Nm|rpm|ms|µs|s|kbps|Mbps|arcmin|dB|%|bar|psi|kOhm|Ohm|Ω|mAh|Ah|Wh|kWh)"
VALUE_UNIT_RE = re.compile(rf"(?<![\w.])(\d+(?:[.,]\d+)?)\s*({UNIT})(?![\w])")
NUMERAL_TOKEN_RE = re.compile(r"\b(\d{1,4}[a-z]?'{0,2})\b")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def norm_hyphen(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "", s.lower().replace(" ", "-").replace("/", "-"))


def extract_identifiers(text: str, known_numerals: set[str] | None = None) -> list[str]:
    """Normalised identifier tokens found in `text` (deduplicated, order preserved)."""
    out: list[str] = []

    def add(*forms: str) -> None:
        for f in forms:
            if f and f not in out:
                out.append(f)

    for m in PART_RE.finditer(text):
        add(norm_hyphen(m.group(0)), norm(m.group(0)))
    for m in PATENT_RE.finditer(text):
        add(norm(m.group(0)))
    for m in FIG_RE.finditer(text):
        add("fig" + m.group(1).lower())
    for m in VALUE_UNIT_RE.finditer(text):
        add(norm(m.group(1) + m.group(2)), f"{m.group(1)} {m.group(2)}".lower())
    if known_numerals:
        for m in NUMERAL_TOKEN_RE.finditer(text):
            if m.group(1) in known_numerals:
                add("ref" + m.group(1))
    return out


def numbers_in(text: str) -> set[str]:
    """All numeric tokens (for answer verification): '320', '450', '4.25', '101:1' -> parts."""
    return {t.replace(",", "") for t in re.findall(r"\d+(?:[.,]\d+)?", text)}

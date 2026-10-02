"""Common document schema shared by every extraction backend.

Design rule: every block carries its source document, page number (1-based) and
bounding box in PDF points with a top-left origin. Downstream stages (chunking,
indexing, citation) only ever see this schema, so backends are interchangeable
and can be compared head to head in the evaluation.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Axis-aligned box in PDF points; origin top-left, y grows downward."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def overlaps(self, other: "BBox", pad: float = 0.0) -> bool:
        return not (
            self.x1 + pad < other.x0
            or other.x1 + pad < self.x0
            or self.y1 + pad < other.y0
            or other.y1 + pad < self.y0
        )


class TextBlock(BaseModel):
    kind: Literal["text"] = "text"
    bbox: BBox
    text: str
    role: Literal["body", "heading", "caption", "header", "footer", "other"] = "body"
    column: int | None = None  # 0 = left/full, 1 = right, ... (thin backend only)
    source: Literal["text_layer", "ocr"] = "text_layer"
    ocr_confidence: float | None = None  # mean word confidence 0-100 when source == "ocr"


class TableBlock(BaseModel):
    kind: Literal["table"] = "table"
    bbox: BBox
    rows: list[list[str]]  # rows[r][c]; header row included as rows[0] when has_header
    has_header: bool = True
    caption: str | None = None
    title: str | None = None  # nearest preceding heading, e.g. "3. Electrical Specifications"
    table_id: str | None = None  # same id on every per-page piece of a table that spans pages
    continued: bool = False  # True on the second and later pieces (header row is repeated)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        w = self.n_cols
        norm = [r + [""] * (w - len(r)) for r in self.rows]
        esc = lambda s: s.replace("|", "\\|").replace("\n", " ")
        lines = ["| " + " | ".join(esc(c) for c in norm[0]) + " |", "|" + "---|" * w]
        lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in norm[1:]]
        return "\n".join(lines)


class FigureBlock(BaseModel):
    kind: Literal["figure"] = "figure"
    bbox: BBox
    image_path: str | None = None  # cropped PNG on disk, relative to project root
    caption: str | None = None  # e.g. "FIG. 2 — Perspective cross section of ..."
    figure_id: str | None = None  # normalised, e.g. "FIG. 2" / "Figure 1"
    ocr_labels: list[str] = Field(default_factory=list)  # tokens read off the image (e.g. "160", "control unit")
    description: str | None = None  # optional VLM description (filled in a later phase)


Block = TextBlock | TableBlock | FigureBlock


class Page(BaseModel):
    number: int  # 1-based
    width: float
    height: float
    blocks: list[Block] = Field(default_factory=list)  # in reading order
    is_scanned: bool = False  # True when the page had no usable text layer and went through OCR
    ocr_reason: str | None = None  # "no_text_layer" | "low_text_quality" | None
    backend: str | None = None  # which backend produced this page (hybrid sets it per page)

    def text(self) -> str:
        """Plain text of the page in reading order (tables as markdown, figures as captions)."""
        parts: list[str] = []
        for b in self.blocks:
            if b.kind == "text":
                parts.append(b.text)
            elif b.kind == "table":
                parts.append(b.to_markdown())
            elif b.kind == "figure" and b.caption:
                parts.append(b.caption)
        return "\n\n".join(parts)


class ParsedDocument(BaseModel):
    doc: str  # file name as it should appear in citations, e.g. "EV-BMS-100_design_document.pdf"
    source_path: str
    backend: str  # "thin" | "docling"
    pages: list[Page]
    meta: dict = Field(default_factory=dict)

    def to_markdown(self) -> str:
        out = []
        for p in self.pages:
            out.append(f"\n\n<!-- page {p.number} -->\n\n{p.text()}")
        return "".join(out).strip()

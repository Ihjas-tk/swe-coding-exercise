"""OmniDocBench slice (English, double-column; 106 pages) for all backends.

All pages are images wrapped as PDFs, so every backend runs its OCR path here; this
benchmark therefore measures scanned-page behaviour: text accuracy, reading order,
table structure from pixels, figure detection. Metrics follow the benchmark's spirit with
our own implementations (the official toolkit expects its own output format):
- text NED: 1 - normalised Levenshtein between the page text in reading order and the
  ground-truth blocks concatenated in annotated `order` (text_block, title, captions,
  list, footnotes; equations and masked regions excluded). Higher is better.
- reading order: pair-order accuracy of GT blocks located in our text (as in extraction.py).
- tables: TEDS (tree edit distance similarity, apted) between our table HTML and the GT
  HTML, matched by page; a GT table with no extracted table scores 0.
- figures: a GT figure counts as detected when an extracted figure bbox overlaps it with
  IoU >= 0.5 (GT polygons are in image pixels; our bboxes in points, scaled by page width).
"""

from __future__ import annotations

import json
import re
from itertools import pairwise
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from ae.eval.extraction import _norm
from ae.schema import BBox, ParsedDocument, TableBlock

ROOT = Path("data/external/omnidocbench")
TEXT_CATS = {
    "text_block",
    "title",
    "figure_caption",
    "table_caption",
    "list_group",
    "page_footnote",
    "table_footnote",
    "figure_footnote",
    "equation_caption",
    "reference",
    "code_txt",
}


def _gt_pages() -> list[dict]:
    return json.loads((ROOT / "subset_en_double_column.json").read_text())


def _gt_text(pg: dict) -> tuple[str, list[str]]:
    dets = [d for d in pg["layout_dets"] if d["category_type"] in TEXT_CATS and d.get("text") and not d.get("ignore")]
    dets.sort(key=lambda d: (d.get("order") is None, d.get("order", 0)))
    blocks = [_norm(d["text"]) for d in dets if _norm(d["text"])]
    return " ".join(blocks), blocks


def _our_text(doc: ParsedDocument) -> str:
    parts = []
    for b in doc.pages[0].blocks:
        if b.kind == "text":
            parts.append(b.text)
        elif b.kind == "figure" and b.caption:
            parts.append(b.caption)
        elif b.kind == "table":
            parts.append(" ".join(" ".join(r) for r in b.rows))
    return _norm(" ".join(parts))


def _table_html(t: TableBlock) -> str:
    return "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in t.rows) + "</table>"


def teds(html_a: str, html_b: str) -> float:
    """Tree-edit-distance similarity between two simple <table><tr><td> HTML strings."""
    from apted import APTED, Config
    from apted.helpers import Tree

    def parse(html: str) -> Tree:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
        root = Tree("table")
        for r in rows:
            tr = Tree("tr")
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S):
                tr.children.append(Tree(_norm(re.sub(r"<[^>]+>", " ", c)).lower() or "·"))
            root.children.append(tr)
        return root

    class Cfg(Config):
        def rename(self, n1: Tree, n2: Tree) -> float:
            """Relabel cost: free for equal labels, full for structure tags, else text edit distance."""
            if n1.name == n2.name:
                return 0.0
            if n1.name in ("table", "tr") or n2.name in ("table", "tr"):
                return 1.0
            return 1.0 - Levenshtein.normalized_similarity(n1.name, n2.name)

    a, b = parse(html_a), parse(html_b)

    def size(t: Tree) -> int:
        return 1 + sum(size(c) for c in t.children)

    d = APTED(a, b, Cfg()).compute_edit_distance()
    return max(0.0, 1.0 - d / max(size(a), size(b)))


def evaluate(backend: str) -> dict:
    """Score one backend's OmniDocBench extractions (text, reading order, TEDS, figure recall)."""
    gts = _gt_pages()
    ex_dir = ROOT / "extracted" / backend
    text_scores, order_scores, teds_scores, fig_hits, fig_total, n = [], [], [], 0, 0, 0
    for pg in gts:
        stem = Path(pg["page_info"]["image_path"]).stem
        f = ex_dir / f"{stem}.json"
        if not f.exists():
            continue
        doc = ParsedDocument.model_validate_json(f.read_text())
        n += 1
        gt_text, gt_blocks = _gt_text(pg)
        ours = _our_text(doc)
        if gt_text:
            text_scores.append(Levenshtein.normalized_similarity(gt_text.lower(), ours.lower()))
            pos = [ours.lower().find(b.lower()[:30]) for b in gt_blocks if len(b) >= 12]
            pos = [p for p in pos if p >= 0]
            pairs = list(pairwise(pos))
            if pairs:
                order_scores.append(sum(1 for a, b in pairs if b > a) / len(pairs))
        # scale: GT polygons are in image pixels; our page width is 612 pt
        scale = 612.0 / pg["page_info"]["width"]
        our_tables = [b for b in doc.pages[0].blocks if b.kind == "table"]
        for d in pg["layout_dets"]:
            if d["category_type"] == "table" and d.get("html") and not d.get("ignore"):
                best = max((teds(_table_html(t), d["html"]) for t in our_tables), default=0.0)
                teds_scores.append(best)
            if d["category_type"] == "figure" and not d.get("ignore"):
                fig_total += 1
                xs, ys = d["poly"][0::2], d["poly"][1::2]
                gbox = BBox(x0=min(xs) * scale, y0=min(ys) * scale, x1=max(xs) * scale, y1=max(ys) * scale)
                if any(gbox.iou(b.bbox) >= 0.5 for b in doc.pages[0].blocks if b.kind == "figure"):
                    fig_hits += 1

    def avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "backend": backend,
        "pages": n,
        "text_similarity": avg(text_scores),
        "reading_order_pair_acc": avg(order_scores),
        "table_teds": avg(teds_scores),
        "tables": len(teds_scores),
        "figure_detection": f"{fig_hits}/{fig_total}",
        "figure_recall": round(fig_hits / fig_total, 3) if fig_total else None,
    }

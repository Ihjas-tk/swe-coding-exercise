"""Extraction-stage evaluation against the hand-annotated gold set (gold/).

Metrics (all per backend):
- OCR: character and word error rate (jiwer) of the scanned page's text in reading order,
  after light normalisation (whitespace, quotes, dashes).
- Tables: cell accuracy. Gold rows are aligned to extracted rows by their first cell
  (fuzzy, >= 85); a cell counts as correct when its normalised text matches exactly.
  Reported as correct / total gold cells, plus how many gold tables were found at all.
- Figures: found (a figure on the gold page whose caption starts with the gold caption or
  prefix), caption linked, and label recall (gold numerals present among the extracted
  figure's confirmed labels: OCR matched + repaired, or VLM parts when available).
- Reading order: for each gold anchor list, the anchors are located in the extracted page
  text; the score is the fraction of consecutive anchor pairs that appear in the right order
  (1.0 = perfect), plus the count of anchors not found at all.
"""

from __future__ import annotations

import json
import re
from itertools import pairwise
from pathlib import Path

from rapidfuzz import fuzz

from ae.index.numerals import NumeralIndex, reconcile_labels
from ae.schema import FigureBlock, ParsedDocument

GOLD = Path("gold")


def _norm(s: str) -> str:
    s = s.replace("—", "-").replace("–", "-").replace("’", "'").replace("“", '"').replace("”", '"').replace("·", "·")
    return re.sub(r"\s+", " ", s).strip()


def _load_docs(backend: str) -> dict[str, ParsedDocument]:
    """Parse the corpus through the backend's page cache (built by `make ingest`)."""
    from ae.corpus import corpus_files
    from ae.extract.base import parse

    out = {}
    for f in corpus_files():
        if f.suffix.lower() in (".pdf", ".docx"):
            d = parse(f, backend=backend)
            out[d.doc] = d
    return out


def _page_text(doc: ParsedDocument, page: int) -> str:
    p = doc.pages[page - 1]
    parts = []
    for b in p.blocks:
        if b.kind == "text":
            parts.append(b.text)
        elif b.kind == "figure" and b.caption:
            parts.append(b.caption)
        elif b.kind == "table":
            parts.append(" ".join(" ".join(r) for r in b.rows))
    return "\n".join(parts)


def eval_ocr(docs: dict[str, ParsedDocument]) -> dict:
    """CER/WER of each gold OCR page."""
    import jiwer

    out = {}
    for f in (GOLD / "ocr").glob("*.txt"):
        m = re.match(r"(.+)_p(\d+)$", f.stem)
        if m is None:
            raise ValueError(f"gold OCR file must be named <doc>_p<page>.txt: {f.name}")
        doc_stem, page = m.group(1), int(m.group(2))
        doc = next((d for name, d in docs.items() if name.startswith(doc_stem)), None)
        if not doc:
            continue
        hyp, ref = _norm(_page_text(doc, page)), _norm(f.read_text())
        out[f.stem] = {
            "cer": round(jiwer.cer(ref, hyp), 4),
            "wer": round(jiwer.wer(ref, hyp), 4),
            "ref_chars": len(ref),
            "hyp_chars": len(hyp),
        }
    return out


def eval_tables(docs: dict[str, ParsedDocument]) -> dict:
    """Cell accuracy of each gold table."""
    gold = json.loads((GOLD / "tables.json").read_text())
    results = []
    for g in gold:
        doc = docs.get(g["doc"])
        found_rows: list[list[str]] = []
        if doc:
            pieces = []
            for p in doc.pages:
                if p.number in g["pages"]:
                    for b in p.blocks:
                        # every table on the gold pages counts; rows are matched below
                        if b.kind == "table" and b.rows:
                            pieces.append(b)
            # merge pieces of a split table (header repeated)
            for i, b in enumerate(pieces):
                rows = b.rows[1:] if (i > 0 and b.rows and b.rows[0] == pieces[0].rows[0]) else b.rows
                found_rows += rows
        gold_rows = g["rows"]
        total = sum(len(r) for r in gold_rows)
        correct = 0
        matched_rows = 0
        for gr in gold_rows:
            best = max(
                found_rows,
                key=lambda fr: fuzz.ratio(_norm(fr[0]).lower(), _norm(gr[0]).lower()) if fr else 0,
                default=None,
            )
            if best is None or fuzz.ratio(_norm(best[0]).lower(), _norm(gr[0]).lower()) < 85:
                continue
            matched_rows += 1
            for i, gc in enumerate(gr):
                if i < len(best) and _norm(best[i]) == _norm(gc):
                    correct += 1
        results.append(
            {
                "doc": g["doc"],
                "title": g["title"],
                "found": bool(found_rows),
                "rows_matched": f"{matched_rows}/{len(gold_rows)}",
                "cell_acc": round(correct / total, 3),
                "cells": f"{correct}/{total}",
            }
        )
    agg_c = sum(int(r["cells"].split("/")[0]) for r in results)
    agg_t = sum(int(r["cells"].split("/")[1]) for r in results)
    return {
        "tables": results,
        "cell_acc": round(agg_c / agg_t, 3),
        "tables_found": f"{sum(r['found'] for r in results)}/{len(results)}",
    }


def eval_figures(docs: dict[str, ParsedDocument], db_path: Path | None = None) -> dict:
    """Figure found / caption linked / label recall for each gold figure."""
    gold = json.loads((GOLD / "figures.json").read_text())
    numerals = NumeralIndex()
    for d in docs.values():
        numerals.add_document(d)
    vlm_parts = _vlm_labels(db_path)
    rows = []
    for g in gold:
        fig = _match_figure(docs.get(g["doc"]), g)
        cap_ok = bool(fig is not None and fig.caption and _norm(fig.caption).lower().startswith(_gold_caption(g)))
        label_recall_ocr = label_recall_vlm = None
        if fig is not None and g["labels"]:
            rec = reconcile_labels(fig.ocr_labels, numerals.defined(g["doc"]))
            have = set(rec["matched"] + rec["repaired"])
            label_recall_ocr = round(len(have & set(g["labels"])) / len(g["labels"]), 2)
            vp = vlm_parts.get((g["doc"], g["page"]))
            if vp is not None:
                label_recall_vlm = round(len(vp & set(g["labels"])) / len(g["labels"]), 2)
        rows.append(
            {
                "doc": g["doc"],
                "page": g["page"],
                "figure_id": g["figure_id"],
                "found": fig is not None,
                "caption_linked": cap_ok,
                "label_recall_ocr": label_recall_ocr,
                "label_recall_vlm": label_recall_vlm,
            }
        )
    n = len(rows)
    lab = [r["label_recall_ocr"] for r in rows if r["label_recall_ocr"] is not None]
    labv = [r["label_recall_vlm"] for r in rows if r["label_recall_vlm"] is not None]
    return {
        "figures": rows,
        "found": f"{sum(r['found'] for r in rows)}/{n}",
        "caption_linked": f"{sum(r['caption_linked'] for r in rows)}/{n}",
        "label_recall_ocr": round(sum(lab) / len(lab), 2) if lab else None,
        "label_recall_vlm": round(sum(labv) / len(labv), 2) if labv else None,
    }


def _vlm_labels(db_path: Path | None) -> dict[tuple[str, int], set[str]]:
    """(doc, page) -> part labels the VLM figure descriptions list (empty without an index or descriptions)."""
    vlm_parts: dict[tuple[str, int], set[str]] = {}
    if not (db_path and db_path.exists()):
        return vlm_parts
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for doc, page, js in conn.execute("SELECT doc, page, json FROM figure_descriptions"):
            d = json.loads(js)
            vlm_parts.setdefault((doc, page), set()).update(
                str(x.get("label")) for x in d.get("labelled_parts", []) if x.get("label")
            )
    except sqlite3.OperationalError:
        pass
    conn.close()
    return vlm_parts


def _gold_caption(g: dict) -> str:
    """Normalised first 40 characters of the gold caption (or caption prefix): the match key."""
    return _norm(g.get("caption") or g["caption_prefix"]).lower()[:40]


def _match_figure(doc: ParsedDocument | None, g: dict) -> FigureBlock | None:
    """Find the extracted figure for a gold one: any figure on the gold page, preferring a caption/id match."""
    if not doc or len(doc.pages) < g["page"]:
        return None
    figs_on_page = [b for b in doc.pages[g["page"] - 1].blocks if isinstance(b, FigureBlock)]
    want = _gold_caption(g)
    fig = next(
        (
            b
            for b in figs_on_page
            if _norm(b.caption or "").lower().startswith(want) or (b.figure_id and b.figure_id == g["figure_id"])
        ),
        None,
    )
    return fig if fig is not None else (figs_on_page[0] if figs_on_page else None)


def eval_reading_order(docs: dict[str, ParsedDocument]) -> dict:
    """Pair-order accuracy of gold anchors on each gold page."""
    gold = json.loads((GOLD / "reading_order.json").read_text())
    out = {}
    for doc_name, g in gold.items():
        doc = docs.get(doc_name)
        if not doc:
            continue
        text = _norm(_page_text(doc, g["page"])).lower()
        pos = []
        missing = 0
        for a in g["anchors"]:
            i = text.find(_norm(a).lower())
            if i < 0:
                # tolerate OCR noise: fuzzy locate the anchor
                best: float = 0
                best_i = -1
                a_n = _norm(a).lower()
                for j in range(0, max(1, len(text) - len(a_n)), 4):
                    sc = fuzz.ratio(text[j : j + len(a_n)], a_n)
                    if sc > best:
                        best, best_i = sc, j
                i = best_i if best >= 85 else -1
            if i < 0:
                missing += 1
            else:
                pos.append(i)
        pairs = list(pairwise(pos))
        ordered = sum(1 for a, b in pairs if b > a)
        out[doc_name] = {
            "page": g["page"],
            "anchors": len(g["anchors"]),
            "missing": missing,
            "pair_order_acc": round(ordered / len(pairs), 3) if pairs else None,
        }
    return out


def evaluate(backend: str) -> dict:
    """Run every extraction metric for one backend."""
    docs = _load_docs(backend)
    from ae.index.store import index_db

    return {
        "backend": backend,
        "ocr": eval_ocr(docs),
        "tables": eval_tables(docs),
        "figures": eval_figures(docs, index_db(backend)),
        "reading_order": eval_reading_order(docs),
    }

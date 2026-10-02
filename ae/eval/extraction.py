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
from pathlib import Path

from rapidfuzz import fuzz

from ae.index.numerals import NumeralIndex, reconcile_labels
from ae.schema import ParsedDocument

GOLD = Path("gold")


def _norm(s: str) -> str:
    s = s.replace("—", "-").replace("–", "-").replace("’", "'").replace("“", '"').replace("”", '"').replace("·", "·")
    return re.sub(r"\s+", " ", s).strip()


def _load_docs(backend: str) -> dict[str, ParsedDocument]:
    """Parsed documents for the corpus via the backend's page cache (built by `make ingest`)."""
    from ae.cli import corpus_files
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
    import jiwer

    out = {}
    for f in (GOLD / "ocr").glob("*.txt"):
        m = re.match(r"(.+)_p(\d+)$", f.stem)
        doc_stem, page = m.group(1), int(m.group(2))
        doc = next((d for name, d in docs.items() if name.startswith(doc_stem)), None)
        if not doc:
            continue
        hyp, ref = _norm(_page_text(doc, page)), _norm(f.read_text())
        out[f.stem] = {"cer": round(jiwer.cer(ref, hyp), 4), "wer": round(jiwer.wer(ref, hyp), 4), "ref_chars": len(ref), "hyp_chars": len(hyp)}
    return out


def eval_tables(docs: dict[str, ParsedDocument]) -> dict:
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
                        if b.kind == "table" and b.rows and (not g.get("title") or (b.title and fuzz.partial_ratio(b.title.lower(), g["title"].lower()) >= 80) or True):
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
            best = max(found_rows, key=lambda fr: fuzz.ratio(_norm(fr[0]).lower(), _norm(gr[0]).lower()) if fr else 0, default=None)
            if best is None or fuzz.ratio(_norm(best[0]).lower(), _norm(gr[0]).lower()) < 85:
                continue
            matched_rows += 1
            for i, gc in enumerate(gr):
                if i < len(best) and _norm(best[i]) == _norm(gc):
                    correct += 1
        results.append({"doc": g["doc"], "title": g["title"], "found": bool(found_rows), "rows_matched": f"{matched_rows}/{len(gold_rows)}", "cell_acc": round(correct / total, 3), "cells": f"{correct}/{total}"})
    agg_c = sum(int(r["cells"].split("/")[0]) for r in results)
    agg_t = sum(int(r["cells"].split("/")[1]) for r in results)
    return {"tables": results, "cell_acc": round(agg_c / agg_t, 3), "tables_found": f"{sum(r['found'] for r in results)}/{len(results)}"}


def eval_figures(docs: dict[str, ParsedDocument], db_path: Path | None = None) -> dict:
    gold = json.loads((GOLD / "figures.json").read_text())
    numerals = NumeralIndex()
    for d in docs.values():
        numerals.add_document(d)
    vlm_parts: dict[tuple[str, int], set[str]] = {}
    if db_path and db_path.exists():
        import sqlite3

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            for doc, page, js in conn.execute("SELECT doc, page, json FROM figure_descriptions"):
                d = json.loads(js)
                vlm_parts.setdefault((doc, page), set()).update(str(x.get("label")) for x in d.get("labelled_parts", []) if x.get("label"))
        except sqlite3.OperationalError:
            pass
        conn.close()
    rows = []
    for g in gold:
        doc = docs.get(g["doc"])
        fig = None
        if doc and len(doc.pages) >= g["page"]:
            figs_on_page = [b for b in doc.pages[g["page"] - 1].blocks if b.kind == "figure"]
            want = _norm(g.get("caption") or g.get("caption_prefix")).lower()[:40]
            # "found" = a figure block exists on the gold page; prefer the one whose caption matches
            fig = next((b for b in figs_on_page if _norm(b.caption or "").lower().startswith(want) or (b.figure_id and b.figure_id == g["figure_id"])), None)
            if fig is None and figs_on_page:
                fig = figs_on_page[0]
        found = fig is not None
        cap_ok = bool(found and fig.caption and _norm(fig.caption).lower().startswith(_norm(g.get("caption") or g.get("caption_prefix")).lower()[:40]))
        label_recall_ocr = label_recall_vlm = None
        if found and g["labels"]:
            rec = reconcile_labels(fig.ocr_labels, numerals.defined(g["doc"]))
            have = set(rec["matched"] + rec["repaired"])
            label_recall_ocr = round(len(have & set(g["labels"])) / len(g["labels"]), 2)
            vp = vlm_parts.get((g["doc"], g["page"]))
            if vp is not None:
                label_recall_vlm = round(len(vp & set(g["labels"])) / len(g["labels"]), 2)
        rows.append({"doc": g["doc"], "page": g["page"], "figure_id": g["figure_id"], "found": found, "caption_linked": cap_ok, "label_recall_ocr": label_recall_ocr, "label_recall_vlm": label_recall_vlm})
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


def eval_reading_order(docs: dict[str, ParsedDocument]) -> dict:
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
                best, best_i = 0, -1
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
        pairs = list(zip(pos, pos[1:]))
        ordered = sum(1 for a, b in pairs if b > a)
        out[doc_name] = {"page": g["page"], "anchors": len(g["anchors"]), "missing": missing, "pair_order_acc": round(ordered / len(pairs), 3) if pairs else None}
    return out


def evaluate(backend: str) -> dict:
    docs = _load_docs(backend)
    from ae.index.store import index_db

    return {
        "backend": backend,
        "ocr": eval_ocr(docs),
        "tables": eval_tables(docs),
        "figures": eval_figures(docs, index_db(backend)),
        "reading_order": eval_reading_order(docs),
    }

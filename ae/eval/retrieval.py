"""Retrieval-stage evaluation: page-level Recall@k by question type, with ablations.

Unit is the page (citations are page-level): the ranked list of distinct (doc, page) pairs
from the fused chunk ranking. Ablations: retrieval mode (bm25 / dense / hybrid), embedding
model, extraction backend, and chunk kinds excluded (e.g. without table_row chunks).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ae.index.store import IndexStore, index_db
from ae.retrieve.hybrid import retrieve
from ae.retrieve.query import doc_aliases, parse_query

KS = (1, 3, 5, 10)
TYPES = ("factual", "table", "figure", "structured")


def recall_at_k(
    backend: str,
    mode: str = "hybrid",
    embed_model: str | None = None,
    exclude_kinds: tuple[str, ...] = (),
    questions: Path = Path("dev_set/questions.json"),
) -> dict:
    """Page-level Recall@k by question type for one retrieval configuration."""
    store = IndexStore(index_db(backend))
    aliases = doc_aliases(store)
    qs = [q for q in json.loads(questions.read_text()) if q["evidence"]]
    hits: defaultdict[str, dict[int, int]] = defaultdict(lambda: dict.fromkeys(KS, 0))
    n: defaultdict[str, int] = defaultdict(int)
    ranks: dict[str, int | None] = {}
    for q in qs:
        gold = {(e["doc"], e["page"]) for e in q["evidence"]}
        pq = parse_query(q["question"], store, aliases)
        res = retrieve(store, pq, embed_model, mode=mode, max_pages=10)
        chunks = {c.id: c for c in store.get(list(res.fused))}
        ranked = sorted(res.fused.items(), key=lambda kv: -kv[1])
        pages = []
        for cid, _ in ranked:
            c = chunks.get(cid)
            if not c or c.kind in exclude_kinds:
                continue
            if (c.doc, c.page) not in pages:
                pages.append((c.doc, c.page))
        n[q["type"]] += 1
        n["all"] += 1
        rank = next((i + 1 for i, p in enumerate(pages) if p in gold), None)
        ranks[q["id"]] = rank
        for k in KS:
            if rank and rank <= k:
                hits[q["type"]][k] += 1
                hits["all"][k] += 1
    store.close()
    table = {t: {f"R@{k}": round(hits[t][k] / n[t], 3) for k in KS} | {"n": n[t]} for t in (*TYPES, "all") if n[t]}
    return {
        "backend": backend,
        "mode": mode,
        "embed_model": embed_model,
        "exclude_kinds": list(exclude_kinds),
        "recall": table,
        "ranks": ranks,
    }


def ablations(backends: list[str], embed_models: list[str]) -> list[dict]:
    """Recall@k for every mode, embedding model and chunk-kind ablation per backend."""
    out = []
    for be in backends:
        store = IndexStore(index_db(be))
        available = store.embedding_models()
        store.close()
        out.append(recall_at_k(be, "bm25"))
        for m in embed_models:
            if m in available:
                out.append(recall_at_k(be, "dense", m))
                out.append(recall_at_k(be, "hybrid", m))
        if embed_models and embed_models[0] in available:
            out.append(recall_at_k(be, "hybrid", embed_models[0], exclude_kinds=("table_row",)))
            out.append(recall_at_k(be, "hybrid", embed_models[0], exclude_kinds=("table",)))
            out.append(recall_at_k(be, "hybrid", embed_models[0], exclude_kinds=("figure",)))
    return out

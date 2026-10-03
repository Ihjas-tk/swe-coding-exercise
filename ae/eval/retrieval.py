"""Retrieval-stage evaluation: page-level Recall@k by question type.

Unit is the page (citations are page-level): the ranked list of distinct (doc, page) pairs
from the fused BM25 + dense chunk ranking of the active index, exactly as `ask` retrieves.
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


def recall_at_k(embed_model: str | None = None, questions: Path = Path("dev_set/questions.json")) -> dict:
    """Page-level Recall@k by question type (and "all") over the questions that have evidence."""
    store = IndexStore(index_db())
    aliases = doc_aliases(store)
    qs = [q for q in json.loads(questions.read_text()) if q["evidence"]]
    hits: defaultdict[str, dict[int, int]] = defaultdict(lambda: dict.fromkeys(KS, 0))
    n: defaultdict[str, int] = defaultdict(int)
    ranks: dict[str, int | None] = {}
    for q in qs:
        gold = {(e["doc"], e["page"]) for e in q["evidence"]}
        pq = parse_query(q["question"], store, aliases)
        res = retrieve(store, pq, embed_model, max_pages=10)
        chunks = {c.id: c for c in store.get(list(res.fused))}
        ranked = sorted(res.fused.items(), key=lambda kv: -kv[1])
        pages = []
        for cid, _ in ranked:
            c = chunks.get(cid)
            if c and (c.doc, c.page) not in pages:
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
    return {"embed_model": embed_model, "recall": table, "ranks": ranks}

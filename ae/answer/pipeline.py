"""End-to-end `ask`: parse -> route -> retrieve -> generate -> verify -> JSON in the brief's shape."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from ae import config
from ae.answer.generate import build_evidence, generate
from ae.answer.verify import verify
from ae.index.numerals import NumeralIndex
from ae.index.store import IndexStore, index_db
from ae.log import TIMINGS, get_logger, timed
from ae.retrieve.hybrid import retrieve

log = get_logger(__name__)
from ae.retrieve.query import ParsedQuery, doc_aliases, parse_query
from ae.retrieve.sql import SQLResult, answer_sql

NOT_FOUND_TEXT = "Not found in the provided documents."


@dataclass
class Answer:
    answer: str
    citations: list[dict]
    not_found: bool
    debug: dict = field(default_factory=dict)

    def to_json(self, debug: bool = False) -> dict:
        out = {"answer": self.answer, "citations": self.citations, "not_found": self.not_found}
        if debug:
            out["debug"] = self.debug
        return out


class Engine:
    def __init__(self, backend: str | None = None, embed_model: str | None = None, mode: str = "hybrid", min_top_score: float = 0.0):
        self.backend = backend or config.BACKEND
        self.db = index_db(self.backend)
        if not self.db.exists():
            raise FileNotFoundError(f"index {self.db} not found; run `make ingest BACKEND={self.backend}` first")
        self.store = IndexStore(self.db)
        self.aliases = doc_aliases(self.store)
        self.numerals = NumeralIndex.load(self.db)
        self.embed_model = embed_model or config.EMBED_MODEL
        self.mode = mode
        self.min_top_score = min_top_score
        self.split_tables = self._split_tables()

    def _split_tables(self) -> dict[str, list[int]]:
        out: dict[str, set[int]] = {}
        for c in self.store.all_chunks():
            tid = c.meta.get("table_id") if c.kind in ("table", "table_row") else None
            if tid:
                out.setdefault(tid, set()).add(c.page)
        return {k: sorted(v) for k, v in out.items() if len(v) > 1}

    # ---------------------------------------------------------------- routes
    def _numeral_lookup(self, pq: ParsedQuery) -> dict | None:
        docs = [pq.doc] if pq.doc else (pq.doc_boost or self.store.docs())
        for n in pq.numerals:
            hits = [self.numerals.lookup(d, n) for d in docs]
            hits = [h for h in hits if h]
            if len(hits) == 1:
                e = hits[0]
                fig = self.numerals.figures.get((e.doc, pq.fig_id)) if pq.fig_id else None
                seen = self._numeral_seen_in_figure(e.doc, pq.fig_id, n) if pq.fig_id else None
                page = fig.page if fig else e.pages[0]
                note = ""
                if pq.fig_id and seen is False:
                    note = f" (label {n} was not read on the {pq.fig_id} image; the definition comes from the specification text)"
                text = f"Reference numeral {n} in {e.doc} is defined in the specification as: {e.name}" + (f" (also written: {', '.join(e.variants)})" if e.variants else "") + f". Defined on page(s) {', '.join(map(str, e.pages))}." + (f" {pq.fig_id} is described as: {fig.title} (page {fig.page})." if fig else "") + note
                # cite the figure's page (or the first definition page); other definition pages are not cited
                return {"doc": e.doc, "page": page, "text": text, "extra_pages": [], "numeral": n, "name": e.name, "seen_in_figure": seen}
            if len(hits) > 1:
                return {"doc": hits[0].doc, "page": hits[0].pages[0], "text": "Reference numeral " + n + " is defined in several documents: " + "; ".join(f"{h.doc}: {h.name}" for h in hits) + ". The question must name the document.", "extra_pages": [], "ambiguous": True}
        return None

    def _numeral_seen_in_figure(self, doc: str, fig_id: str | None, n: str) -> bool | None:
        for c in self.store.all_chunks():
            if c.doc == doc and c.kind == "figure" and c.meta.get("figure_id") == fig_id:
                labels = c.meta.get("labels", {})
                return n in (labels.get("matched", []) + labels.get("repaired", []))
        return None

    # ---------------------------------------------------------------- main
    def ask(self, question: str, debug: bool = False) -> Answer:
        t0 = time.time()
        n_before = len(TIMINGS)
        with timed("query.total", level=10):
            ans = self._ask(question, t0)
        stages = {r["stage"][len("query."):]: r["ms"] for r in TIMINGS[n_before:] if r["stage"].startswith("query.")}
        ans.debug["timing_ms"] = stages
        log.info(f"answered in {stages.get('total', 0)} ms route={ans.debug.get('route')} not_found={ans.not_found} | " + " ".join(f"{k}={v}" for k, v in stages.items() if k != "total") + f" | q={question[:70]!r}")
        return ans

    def _ask(self, question: str, t0: float) -> Answer:
        with timed("query.parse", level=10):
            pq = parse_query(question, self.store, self.aliases)
        dbg: dict = {"route": pq.route, "reasons": pq.reasons, "doc": pq.doc, "doc_boost": pq.doc_boost, "fig_id": pq.fig_id, "numerals": pq.numerals, "identifiers": pq.identifiers, "visual": pq.visual, "backend": self.backend}
        # scope gate: a named figure that the named document does not define
        if pq.doc and pq.fig_id and self.numerals.figures and not any(d == pq.doc for d, _ in self.numerals.figures) is False:
            pass
        with timed("query.retrieve", level=10):
            result = retrieve(self.store, pq, self.embed_model, mode=self.mode)
        dbg["gate"] = result.gate
        dbg["pages"] = [(p.doc, p.page, round(p.score, 4)) for p in result.pages]
        numeral_block = None
        if pq.route == "numeral":
            with timed("query.numeral_lookup", level=10):
                numeral_block = self._numeral_lookup(pq)
        sql: SQLResult | None = None
        if pq.route == "sql":
            try:
                with timed("query.sql", level=10):
                    sql = answer_sql(question, self.db)
            except Exception as e:  # noqa: BLE001  (text-to-SQL failure falls back to hybrid evidence)
                dbg["sql_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                log.warning(f"sql route failed, falling back to retrieval evidence: {dbg['sql_error']}")
        if sql:
            dbg["sql"] = {"sql": sql.sql, "rows": sql.rows[:5], "error": sql.error}
        if numeral_block:
            dbg["numeral"] = {k: v for k, v in numeral_block.items() if k != "text"}
        if self.min_top_score and result.gate["top_page_score"] < self.min_top_score and not numeral_block and not (sql and sql.rows):
            dbg["gate_decline"] = True
            return Answer(NOT_FOUND_TEXT, [], True, {**dbg, "ms": int((time.time() - t0) * 1000)})
        items = build_evidence(result, sql, numeral_block, self.split_tables)
        try:
            with timed("query.generate", level=10, evidence=len(items)):
                draft = generate(pq, items)
        except Exception as e:  # noqa: BLE001  (LLM/API failure must not abort an evaluation run)
            dbg["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            dbg["ms"] = int((time.time() - t0) * 1000)
            log.error(f"generation failed: {dbg['error']}")
            return Answer(NOT_FOUND_TEXT, [], True, dbg)
        with timed("query.verify", level=10):
            verdict = verify(pq, draft, self.min_top_score)
        if verdict.failures:
            log.info(f"verification declined the draft: {verdict.failures}")
        dbg["draft"] = {"answer": draft.answer, "citations": draft.citations, "confidence": draft.confidence, "reasoning": draft.reasoning}
        dbg["verify"] = {"ok": verdict.ok, "failures": verdict.failures, "warnings": verdict.warnings}
        dbg["ms"] = int((time.time() - t0) * 1000)
        if verdict.not_found:
            return Answer(NOT_FOUND_TEXT, [], True, dbg)
        cits: list[dict] = []
        for it in verdict.citations:
            for pg in [it.page, *it.extra_pages]:
                c = {"doc": it.doc, "page": pg}
                if c not in cits:
                    cits.append(c)
        return Answer(draft.answer, cits, False, dbg)

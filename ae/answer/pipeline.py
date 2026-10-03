"""End-to-end `ask`: parse -> route -> retrieve -> generate -> verify -> JSON in the brief's shape."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ae import config
from ae.answer.generate import Draft, build_evidence, generate
from ae.answer.verify import verify
from ae.index.numerals import NumeralEntry, NumeralIndex
from ae.index.store import IndexStore, index_db
from ae.log import get_logger
from ae.retrieve.hybrid import retrieve
from ae.retrieve.query import ParsedQuery, doc_aliases, parse_query
from ae.retrieve.sql import SQLResult, answer_sql

log = get_logger(__name__)

NOT_FOUND_TEXT = "Not found in the provided documents."


def ingest_hint() -> str:
    """Return the command that builds `index_db()`, the index of the active namespace."""
    name = config.INDEX
    if not name:
        return "`make ingest`"
    if name == "smoke":
        return "`make smoke`"
    if name == "ext":
        return "`make external`"
    return f"`uv run ae ingest --corpus <dir> --name {name}`"


@dataclass
class Answer:
    """Final answer in the brief's JSON shape, plus debug details."""

    answer: str
    citations: list[dict]
    not_found: bool
    debug: dict = field(default_factory=dict)

    def to_json(self, debug: bool = False) -> dict:
        """Return the brief's JSON shape; include `debug` when asked."""
        out = {"answer": self.answer, "citations": self.citations, "not_found": self.not_found}
        if debug:
            out["debug"] = self.debug
        return out


class Engine:
    """Loads one index and answers questions against it (parse, route, retrieve, generate, verify)."""

    def __init__(self, embed_model: str | None = None, min_top_score: float = 0.0):
        self.db = index_db()
        if not self.db.exists():
            raise FileNotFoundError(f"index {self.db} not found; build it with {ingest_hint()}")
        self.store = IndexStore(self.db)
        self.aliases = doc_aliases(self.store)
        self.numerals = NumeralIndex.load(self.db)
        self.embed_model = embed_model or config.EMBED_MODEL
        self.min_top_score = min_top_score
        self.split_tables = self._split_tables()

    def _split_tables(self) -> dict[str, list[int]]:
        """Table id -> every page it spans, for tables split across pages (citation policy)."""
        out: dict[str, set[int]] = {}
        for c in self.store.all_chunks():
            tid = c.meta.get("table_id") if c.kind in ("table", "table_row") else None
            if tid:
                out.setdefault(tid, set()).add(c.page)
        return {k: sorted(v) for k, v in out.items() if len(v) > 1}

    # ---------------------------------------------------------------- routes
    def _numeral_lookup(self, pq: ParsedQuery) -> dict | None:
        """Evidence block for the first asked numeral defined in the candidate documents.

        Candidates are the named document, else the boosted ones, else all. A numeral defined
        in several candidates yields an "ambiguous" block that tells the model to decline.
        """
        docs = [pq.doc] if pq.doc else (pq.doc_boost or self.store.docs())
        for n in pq.numerals:
            hits = [h for d in docs if (h := self.numerals.lookup(d, n)) is not None]
            if len(hits) == 1:
                return self._numeral_block(pq, n, hits[0])
            if len(hits) > 1:
                return {
                    "doc": hits[0].doc,
                    "page": hits[0].pages[0],
                    "text": "Reference numeral "
                    + n
                    + " is defined in several documents: "
                    + "; ".join(f"{h.doc}: {h.name}" for h in hits)
                    + ". The question must name the document.",
                    "extra_pages": [],
                    "ambiguous": True,
                }
        return None

    def _numeral_block(self, pq: ParsedQuery, n: str, e: NumeralEntry) -> dict:
        """One numeral's definition; cites the named figure's page, else the first definition page."""
        fig = self.numerals.figures.get((e.doc, pq.fig_id)) if pq.fig_id else None
        seen = self._numeral_seen_in_figure(e.doc, pq.fig_id, n) if pq.fig_id else None
        note = ""
        if pq.fig_id and seen is False:
            note = (
                f" (label {n} was not read on the {pq.fig_id} image; the definition comes from the specification text)"
            )
        text = (
            f"Reference numeral {n} in {e.doc} is defined in the specification as: {e.name}"
            + (f" (also written: {', '.join(e.variants)})" if e.variants else "")
            + f". Defined on page(s) {', '.join(map(str, e.pages))}."
            + (f" {pq.fig_id} is described as: {fig.title} (page {fig.page})." if fig else "")
            + note
        )
        return {
            "doc": e.doc,
            "page": fig.page if fig else e.pages[0],
            "text": text,
            "extra_pages": [],
            "numeral": n,
            "name": e.name,
            "seen_in_figure": seen,
        }

    def _numeral_seen_in_figure(self, doc: str, fig_id: str | None, n: str) -> bool | None:
        """Whether label `n` was read on the figure's image (OCR matched or repaired); None if no such figure."""
        for c in self.store.all_chunks():
            if c.doc == doc and c.kind == "figure" and c.meta.get("figure_id") == fig_id:
                labels = c.meta.get("labels", {})
                return n in (labels.get("matched", []) + labels.get("repaired", []))
        return None

    def _sql(self, question: str, dbg: dict) -> SQLResult | None:
        """Text-to-SQL result, or None when the route failed (the retrieval evidence still answers)."""
        try:
            return answer_sql(question, self.db)
        except Exception as e:  # text-to-SQL failure falls back to hybrid evidence
            dbg["sql_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            log.warning(f"sql route failed, falling back to retrieval evidence: {dbg['sql_error']}")
            return None

    # ---------------------------------------------------------------- main
    def ask(self, question: str) -> Answer:
        """Answer one question; routing/retrieval/verification details are kept in `Answer.debug`.

        Never raises for LLM/API failures: those become not_found with `debug["error"]` set,
        so an evaluation run is not aborted by one bad call.
        """
        t0 = time.time()
        ans = self._ask(question)
        ans.debug["ms"] = int((time.time() - t0) * 1000)
        log.info(f"answered route={ans.debug.get('route')} not_found={ans.not_found} q={question[:70]!r}")
        return ans

    def _ask(self, question: str) -> Answer:
        """Parse -> retrieve -> route evidence -> (gate) -> generate -> verify."""
        pq = parse_query(question, self.store, self.aliases)
        dbg = _parse_debug(pq)
        result = retrieve(self.store, pq, self.embed_model)
        dbg["gate"] = result.gate
        dbg["pages"] = [(p.doc, p.page, round(p.score, 4)) for p in result.pages]
        numeral_block = self._numeral_lookup(pq) if pq.route == "numeral" else None
        sql = self._sql(question, dbg) if pq.route == "sql" else None
        if sql:
            dbg["sql"] = {"sql": sql.sql, "rows": sql.rows[:5], "error": sql.error}
        if numeral_block:
            dbg["numeral"] = {k: v for k, v in numeral_block.items() if k != "text"}
        if self._gate_declines(result.gate, numeral_block, sql):
            dbg["gate_decline"] = True
            return Answer(NOT_FOUND_TEXT, [], True, dbg)
        items = build_evidence(result, sql, numeral_block, self.split_tables)
        try:
            draft = generate(pq, items)
        except Exception as e:  # LLM/API failure must not abort an evaluation run
            dbg["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            log.error(f"generation failed: {dbg['error']}")
            return Answer(NOT_FOUND_TEXT, [], True, dbg)
        return self._finalise(pq, draft, dbg)

    def _gate_declines(self, gate: dict, numeral_block: dict | None, sql: SQLResult | None) -> bool:
        """Apply the retrieval not-found gate: weak top page and no numeral/SQL evidence (off when 0)."""
        return bool(
            self.min_top_score
            and gate["top_page_score"] < self.min_top_score
            and not numeral_block
            and not (sql and sql.rows)
        )

    def _finalise(self, pq: ParsedQuery, draft: Draft, dbg: dict) -> Answer:
        """Verify the draft and map its evidence ids to page citations (incl. caption / split-table pages)."""
        verdict = verify(pq, draft)
        if verdict.failures:
            log.debug(f"verification declined the draft: {verdict.failures}")
        dbg["draft"] = {
            "answer": draft.answer,
            "citations": draft.citations,
            "confidence": draft.confidence,
            "reasoning": draft.reasoning,
        }
        dbg["verify"] = {"ok": verdict.ok, "failures": verdict.failures, "warnings": verdict.warnings}
        if verdict.not_found:
            return Answer(NOT_FOUND_TEXT, [], True, dbg)
        cits: list[dict] = []
        for it in verdict.citations:
            for pg in [it.page, *it.extra_pages]:
                c = {"doc": it.doc, "page": pg}
                if c not in cits:
                    cits.append(c)
        return Answer(draft.answer, cits, False, dbg)


def _parse_debug(pq: ParsedQuery) -> dict:
    """Record the parser's decisions for `Answer.debug`."""
    return {
        "route": pq.route,
        "reasons": pq.reasons,
        "doc": pq.doc,
        "doc_boost": pq.doc_boost,
        "fig_id": pq.fig_id,
        "numerals": pq.numerals,
        "identifiers": pq.identifiers,
        "visual": pq.visual,
    }

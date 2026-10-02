"""Structured route: text-to-SQL over the loaded CSV/XLSX tables, read-only, one retry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ae.extract.structured import SQLError, describe_schema, list_tables, query_sql
from ae.llm import complete_json
from ae.log import get_logger

log = get_logger(__name__)

SQL_MAX_TOKENS = 1500  # one SELECT plus a short explanation; wide schemas need long column lists

SYSTEM = """You translate a question into ONE SQLite SELECT statement over the tables below.
Rules: use only listed tables/columns; match text values exactly as listed (case-sensitive); never modify data;
prefer explicit aggregates (COUNT, COUNT(DISTINCT ...), SUM, MAX, ORDER BY ... LIMIT 1); include the columns a reader needs to verify the answer
(e.g. part_number together with its cost). Spreadsheet-derived tables often contain total/subtotal/section rows whose label columns are NULL:
exclude them (WHERE <label column> IS NOT NULL) for per-item questions such as "largest", "most expensive" or "which item". If the question cannot be answered from these tables, set "sql" to null.
Reply with JSON: {"sql": "...", "tables": ["..."], "explanation": "one sentence"}"""


@dataclass
class SQLResult:
    """The generated SQL and its result (or error)."""

    sql: str | None
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)  # source files of the tables used
    error: str | None = None
    explanation: str = ""


def answer_sql(question: str, db_path: Path) -> SQLResult:
    """Text-to-SQL over the structured tables, with one repair attempt on error.

    The repair prompt shows the model its failed SQL and the SQLite error; a second failure
    is returned as `SQLResult.error` and the pipeline falls back to retrieval evidence.
    """
    schema = describe_schema(db_path)
    table_docs = {name: doc for name, doc, _ in list_tables(db_path)}
    prompt = f"SCHEMA:\n{schema}\n\nQUESTION: {question}"
    out = complete_json(SYSTEM, prompt, max_tokens=SQL_MAX_TOKENS)
    sql = out.get("sql")
    if not sql:
        return SQLResult(None, explanation=out.get("explanation", ""))
    try:
        return _run(sql, out, db_path, table_docs)
    except SQLError as e:
        log.debug(f"sql failed, asking for a repair: {e}")
        out = complete_json(
            SYSTEM,
            prompt + f"\n\nYour previous SQL failed with: {e}\nPrevious SQL: {sql}\nFix it.",
            max_tokens=SQL_MAX_TOKENS,
        )
        sql = out.get("sql") or sql
    try:
        return _run(sql, out, db_path, table_docs)
    except SQLError as e:
        return SQLResult(sql, error=str(e), explanation=out.get("explanation", ""))


def _run(sql: str, out: dict, db_path: Path, table_docs: dict[str, str]) -> SQLResult:
    """Execute the SQL read-only; the source files are those whose table names appear in it."""
    res = query_sql(sql, db_path)
    used = [t for t in table_docs if t.lower() in sql.lower()]
    return SQLResult(sql, res["columns"], res["rows"], [table_docs[t] for t in used], None, out.get("explanation", ""))

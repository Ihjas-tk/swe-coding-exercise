"""Structured route: text-to-SQL over the loaded CSV/XLSX tables, read-only, one retry."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ae.extract.structured import SQLError, describe_schema, list_tables, query_sql
from ae.llm import complete_json

SYSTEM = """You translate a question into ONE SQLite SELECT statement over the tables below.
Rules: use only listed tables/columns; match text values exactly as listed (case-sensitive); never modify data;
prefer explicit aggregates (COUNT, COUNT(DISTINCT ...), SUM, MAX, ORDER BY ... LIMIT 1); include the columns a reader needs to verify the answer
(e.g. part_number together with its cost). Spreadsheet-derived tables often contain total/subtotal/section rows whose label columns are NULL:
exclude them (WHERE <label column> IS NOT NULL) for per-item questions such as "largest", "most expensive" or "which item". If the question cannot be answered from these tables, set "sql" to null.
Reply with JSON: {"sql": "...", "tables": ["..."], "explanation": "one sentence"}"""


@dataclass
class SQLResult:
    sql: str | None
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)  # source files of the tables used
    error: str | None = None
    explanation: str = ""


def answer_sql(question: str, db_path: Path) -> SQLResult:
    schema = describe_schema(db_path)
    table_docs = {name: doc for name, doc, _ in list_tables(db_path)}
    prompt = f"SCHEMA:\n{schema}\n\nQUESTION: {question}"
    out = complete_json(SYSTEM, prompt, max_tokens=1500)
    sql = out.get("sql")
    if not sql:
        return SQLResult(None, explanation=out.get("explanation", ""))
    for attempt in range(2):
        try:
            res = query_sql(sql, db_path)
            used = [t for t in table_docs if t.lower() in sql.lower()]
            return SQLResult(sql, res["columns"], res["rows"], [table_docs[t] for t in used], None, out.get("explanation", ""))
        except SQLError as e:
            if attempt == 1:
                return SQLResult(sql, error=str(e), explanation=out.get("explanation", ""))
            out = complete_json(SYSTEM, prompt + f"\n\nYour previous SQL failed with: {e}\nPrevious SQL: {sql}\nFix it.", max_tokens=1500)
            sql = out.get("sql") or sql
    return SQLResult(sql, error="unreachable")

"""Structured files (CSV / XLSX) -> typed SQLite tables with row-level provenance.

Why SQL and not text: "how many runs failed" and "highest unit cost" are
aggregates; embedding rows as prose cannot answer them reliably. Each file
becomes one table (one per sheet for multi-sheet workbooks) with inferred column
types, plus:
- `_row`: 1-based row number in the source (provenance; the page is always 1);
- a `_structured_tables` catalogue (table -> source file, schema, row count) that
  the text-to-SQL prompt is built from;
- a `_structured_rows` table holding one text rendering per row, so rows are
  also findable by keyword/vector search (the BOM description "planetary +
  harmonic gearset" or a test-log note only exist here).

Both extraction backends share this loader: Docling would flatten these files
into a document, which is the failure mode the brief warns about.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import openpyxl

DB_PATH = Path("data/index/answer_engine.sqlite")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SQL_TYPES = {"int": "INTEGER", "float": "REAL", "date": "TEXT", "text": "TEXT"}


@dataclass
class ColumnInfo:
    name: str
    type: str  # int | float | date | text
    samples: list[str] = field(default_factory=list)  # distinct examples (low-cardinality text columns)
    n_distinct: int = 0


@dataclass
class StructuredTable:
    name: str
    doc: str
    columns: list[ColumnInfo]
    n_rows: int


# ----------------------------------------------------------------------------- reading


def _read_csv(path: Path) -> list[tuple[str, list[list[Any]]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = [r for r in csv.reader(f, dialect)]
    return [(path.stem, rows)]


def _read_xlsx(path: Path) -> list[tuple[str, list[list[Any]]]]:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)  # data_only: formula results, not formulas
    out = []
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        rows = [r for r in rows if any(v is not None and str(v).strip() != "" for v in r)]
        if not rows:
            continue
        name = path.stem if len(wb.worksheets) == 1 else f"{path.stem}__{_ident(ws.title)}"
        out.append((name, rows))
    return out


def _ident(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(s).strip()).strip("_").lower()
    return s or "col"


# ----------------------------------------------------------------------------- typing


def _header_index(rows: list[list[Any]]) -> int:
    """First row whose cells are all non-empty strings is the header (skips title rows)."""
    for i, r in enumerate(rows[:10]):
        if r and all(isinstance(v, str) and v.strip() for v in r):
            return i
    return 0


def _coerce(v: Any, typ: str) -> Any:
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    if typ == "int":
        return int(float(v))
    if typ == "float":
        return float(v)
    if typ == "date":
        return v.date().isoformat() if isinstance(v, datetime) else (v.isoformat() if isinstance(v, date) else str(v))
    return str(v).strip()


def _infer_type(values: list[Any]) -> str:
    vals = [v for v in values if v is not None and not (isinstance(v, str) and v.strip() == "")]
    if not vals:
        return "text"
    if all(isinstance(v, (date, datetime)) or (isinstance(v, str) and DATE_RE.match(v.strip())) for v in vals):
        return "date"
    try:
        nums = [float(v) for v in vals]
    except (TypeError, ValueError):
        return "text"
    if all(isinstance(v, int) or float(v).is_integer() for v in vals) and all(not isinstance(v, str) or "." not in v for v in vals):
        return "int"
    return "float"


# ----------------------------------------------------------------------------- loading


def load_structured(paths: list[Path], db_path: Path = DB_PATH) -> list[StructuredTable]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS _structured_tables (name TEXT PRIMARY KEY, doc TEXT, n_rows INTEGER, schema_json TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS _structured_rows (doc TEXT, tbl TEXT, row INTEGER, text TEXT, PRIMARY KEY (doc, tbl, row))")
    loaded: list[StructuredTable] = []
    for path in paths:
        sheets = _read_csv(path) if path.suffix.lower() == ".csv" else _read_xlsx(path)
        for name, rows in sheets:
            tbl = _load_table(conn, _ident(name), path.name, rows)
            loaded.append(tbl)
    conn.commit()
    conn.close()
    return loaded


def _load_table(conn: sqlite3.Connection, name: str, doc: str, rows: list[list[Any]]) -> StructuredTable:
    h = _header_index(rows)
    header = [_ident(c) for c in rows[h]]
    # de-duplicate column names
    seen: dict[str, int] = {}
    for i, c in enumerate(header):
        if c in seen:
            seen[c] += 1
            header[i] = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
    body = [r + [None] * (len(header) - len(r)) for r in rows[h + 1 :]]
    cols: list[ColumnInfo] = []
    for i, c in enumerate(header):
        values = [r[i] for r in body]
        typ = _infer_type(values)
        distinct = sorted({str(_coerce(v, typ)) for v in values if _coerce(v, typ) is not None})
        samples = distinct if (typ == "text" and len(distinct) <= 20) else distinct[:3]
        cols.append(ColumnInfo(c, typ, samples, len(distinct)))

    conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    ddl = ", ".join([f'"_row" INTEGER PRIMARY KEY'] + [f'"{c.name}" {SQL_TYPES[c.type]}' for c in cols])
    conn.execute(f'CREATE TABLE "{name}" ({ddl})')
    placeholders = ", ".join(["?"] * (len(cols) + 1))
    conn.execute("DELETE FROM _structured_rows WHERE doc = ? AND tbl = ?", (doc, name))
    for ri, r in enumerate(body, start=1):
        typed = [_coerce(r[i], cols[i].type) for i in range(len(cols))]
        conn.execute(f'INSERT INTO "{name}" VALUES ({placeholders})', [ri] + typed)
        text = f"{name} row {ri}: " + "; ".join(f"{cols[i].name}={typed[i]}" for i in range(len(cols)) if typed[i] not in (None, ""))
        conn.execute("INSERT INTO _structured_rows VALUES (?, ?, ?, ?)", (doc, name, ri, text))
    tbl = StructuredTable(name=name, doc=doc, columns=cols, n_rows=len(body))
    conn.execute(
        "INSERT OR REPLACE INTO _structured_tables VALUES (?, ?, ?, ?)",
        (name, doc, len(body), json.dumps([c.__dict__ for c in cols])),
    )
    return tbl


# ----------------------------------------------------------------------------- querying


class SQLError(Exception):
    pass


def query_sql(sql: str, db_path: Path = DB_PATH, limit: int = 200, timeout_s: float = 5.0) -> dict:
    """Run a single read-only SELECT. Returns {"columns": [...], "rows": [[...], ...], "truncated": bool}."""
    stmt = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", stmt, re.I) or ";" in stmt:
        raise SQLError("only a single SELECT statement is allowed")
    if re.search(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace)\b", stmt, re.I):
        raise SQLError("statement contains a write/DDL keyword")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    deadline_steps = int(timeout_s * 100_000)
    conn.set_progress_handler(lambda: 1, deadline_steps)  # abort after ~N VM steps
    try:
        cur = conn.execute(stmt)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(limit + 1)
    except sqlite3.OperationalError as e:
        raise SQLError(str(e)) from e
    finally:
        conn.close()
    truncated = len(rows) > limit
    return {"columns": cols, "rows": [list(r) for r in rows[:limit]], "truncated": truncated}


def describe_schema(db_path: Path = DB_PATH) -> str:
    """Human/LLM-readable schema with types, row counts and sample values (for text-to-SQL)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = []
    for name, doc, n_rows, schema_json in conn.execute("SELECT name, doc, n_rows, schema_json FROM _structured_tables ORDER BY name"):
        out.append(f'TABLE "{name}"  -- source file: {doc}, {n_rows} rows; column _row = 1-based source row number')
        for c in json.loads(schema_json):
            ex = ", ".join(c["samples"][:20])
            note = f"values: {ex}" if c["type"] == "text" and c["n_distinct"] <= 20 else f"e.g. {ex}"
            out.append(f'  "{c["name"]}" {SQL_TYPES[c["type"]]}  -- {c["n_distinct"]} distinct; {note}')
    conn.close()
    return "\n".join(out)


def list_tables(db_path: Path = DB_PATH) -> list[tuple[str, str, int]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute("SELECT name, doc, n_rows FROM _structured_tables ORDER BY name").fetchall()
    conn.close()
    return rows

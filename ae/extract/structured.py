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
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import openpyxl

from ae.log import get_logger

log = get_logger(__name__)

DB_PATH = Path("data/index/answer_engine.sqlite")
"""Default database for the stand-alone `load-structured` command (ingest passes the index file)."""
MAX_ROW_TEXT_COLS = 30
"""Columns rendered into a row's search text; wide sensor tables stay readable chunks."""
CSV_SNIFF_BYTES = 4096
"""Bytes of a CSV the dialect sniffer reads to guess the delimiter."""
HEADER_SCAN_ROWS = 15
"""The header row is searched for among the first this-many rows (title rows can precede it)."""
NUMERIC_COLUMN_FRAC = 0.9
"""A column with at least this fraction of numeric cells is numeric; the rest become NULL."""
MAX_ENUM_VALUES = 20
"""A text column with at most this many distinct values is listed in full in the schema prompt."""
NUMERIC_SAMPLES = 3
"""Example values shown for numeric/date/high-cardinality columns."""
WIDE_TABLE_COLS = 30
"""Tables wider than this are summarised in the schema prompt ..."""
WIDE_HEAD_COLS, WIDE_TAIL_COLS = 20, 8
"""... as their first and last few columns."""
SQL_STEPS_PER_SECOND = 100_000
"""Approximate SQLite VM steps per second, to turn a timeout into a progress-handler step budget."""
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SQL_TYPES = {"int": "INTEGER", "float": "REAL", "date": "TEXT", "text": "TEXT"}


@dataclass
class ColumnInfo:
    """Inferred column name, type and sample values."""

    name: str
    type: str  # int | float | date | text
    samples: list[str] = field(default_factory=list)  # distinct examples (low-cardinality text columns)
    n_distinct: int = 0


@dataclass
class StructuredTable:
    """A loaded table and where it came from."""

    name: str
    doc: str
    columns: list[ColumnInfo]
    n_rows: int


# ----------------------------------------------------------------------------- reading


def _read_csv(path: Path) -> list[tuple[str, list[list[Any]]]]:
    """One (table name, rows) pair; the delimiter is sniffed, falling back to Excel CSV."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        sample = f.read(CSV_SNIFF_BYTES)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows: list[list[Any]] = list(csv.reader(f, dialect))
    return [(path.stem, rows)]


def _read_xlsx(path: Path) -> list[tuple[str, list[list[Any]]]]:
    """One (table name, rows) pair per non-empty sheet; multi-sheet tables are named <file>__<sheet>."""
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
    """SQL-safe lower-case identifier ("Unit Cost (€)" -> "unit_cost")."""
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(s).strip()).strip("_").lower()
    return s or "col"


# ----------------------------------------------------------------------------- typing


def _header_index(rows: list[list[Any]]) -> int:
    """Pick the header: the row among the first HEADER_SCAN_ROWS with the most non-empty *string* cells.

    At least two such cells are required. Title rows above the header are short (one merged cell) and
    data rows are mostly numeric, so this picks real headers even when a column has no
    header at all (OLSK BOM: cost column unnamed, header on row 6).
    """
    best, best_n = 0, -1
    for i, r in enumerate(rows[:HEADER_SCAN_ROWS]):
        n = sum(1 for v in r if isinstance(v, str) and v.strip() and not re.fullmatch(r"[\d.,%€$£-]+", v.strip()))
        if n > best_n and n >= 2:
            best, best_n = i, n
    return best if best_n >= 2 else 0


def _coerce(v: Any, typ: str) -> Any:
    """Convert a cell to the column's SQL value; blanks and non-numeric cells of numeric columns become None."""
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    if typ in ("int", "float"):
        if not _is_num(v):
            return None  # minority non-numeric cell in a numeric column
        f = float(str(v).replace(",", ""))
        return int(f) if typ == "int" else f
    if typ == "date":
        return v.date().isoformat() if isinstance(v, datetime) else (v.isoformat() if isinstance(v, date) else str(v))
    return str(v).strip()


def _is_num(v: Any) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except ValueError:
        return False


def _infer_type(values: list[Any]) -> str:
    """Infer int / float / date / text for a column.

    A column that is >= 90 % numeric is numeric; its few text cells (e.g. "16H" in a cost column) become NULL so aggregates still work.
    """
    vals = [v for v in values if v is not None and not (isinstance(v, str) and v.strip() == "")]
    if not vals:
        return "text"
    if all(isinstance(v, (date, datetime)) or (isinstance(v, str) and DATE_RE.match(v.strip())) for v in vals):
        return "date"
    numeric = [v for v in vals if _is_num(v)]
    if len(numeric) < NUMERIC_COLUMN_FRAC * len(vals) or not numeric:
        return "text"
    if all(isinstance(v, int) or float(str(v).replace(",", "")).is_integer() for v in numeric) and all(
        not isinstance(v, str) or "." not in v for v in numeric
    ):
        return "int"
    return "float"


# ----------------------------------------------------------------------------- loading


def load_structured(paths: list[Path], db_path: Path = DB_PATH) -> list[StructuredTable]:
    """Load each CSV/XLSX (one table per sheet) into SQLite and return the tables loaded.

    Re-loading a file replaces its tables and their `_structured_rows` entries.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _structured_tables (name TEXT PRIMARY KEY, doc TEXT, n_rows INTEGER, schema_json TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _structured_rows (doc TEXT, tbl TEXT, row INTEGER, text TEXT, PRIMARY KEY (doc, tbl, row))"
    )
    loaded: list[StructuredTable] = []
    for path in paths:
        sheets = _read_csv(path) if path.suffix.lower() == ".csv" else _read_xlsx(path)
        for name, rows in sheets:
            tbl = _load_table(conn, _ident(name), path.name, rows)
            loaded.append(tbl)
            log.info(f"{path.name} -> table {tbl.name}: {tbl.n_rows} rows, {len(tbl.columns)} columns")
    conn.commit()
    conn.close()
    return loaded


def _load_table(conn: sqlite3.Connection, name: str, doc: str, rows: list[list[Any]]) -> StructuredTable:
    """Create table `name` from raw rows (header detected), plus its row texts and catalogue entry."""
    h = _header_index(rows)
    header = _column_names(rows[h])
    body = [r + [None] * (len(header) - len(r)) for r in rows[h + 1 :]]
    cols = [_column_info(c, [r[i] for r in body]) for i, c in enumerate(header)]

    conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    ddl = ", ".join(['"_row" INTEGER PRIMARY KEY'] + [f'"{c.name}" {SQL_TYPES[c.type]}' for c in cols])
    conn.execute(f'CREATE TABLE "{name}" ({ddl})')
    placeholders = ", ".join(["?"] * (len(cols) + 1))
    conn.execute("DELETE FROM _structured_rows WHERE doc = ? AND tbl = ?", (doc, name))
    for ri, r in enumerate(body, start=1):
        typed = [_coerce(r[i], cols[i].type) for i in range(len(cols))]
        conn.execute(f'INSERT INTO "{name}" VALUES ({placeholders})', [ri, *typed])
        conn.execute(
            "INSERT INTO _structured_rows VALUES (?, ?, ?, ?)", (doc, name, ri, _row_text(name, ri, cols, typed))
        )
    conn.execute(
        "INSERT OR REPLACE INTO _structured_tables VALUES (?, ?, ?, ?)",
        (name, doc, len(body), json.dumps([asdict(c) for c in cols])),
    )
    return StructuredTable(name=name, doc=doc, columns=cols, n_rows=len(body))


def _column_names(header_row: list[Any]) -> list[str]:
    """Return SQL column names for a header row: blanks become col_<i>, duplicates get _1, _2, ..."""
    header = [_ident(c) if (c is not None and str(c).strip()) else f"col_{i + 1}" for i, c in enumerate(header_row)]
    seen: dict[str, int] = {}
    for i, c in enumerate(header):
        if c in seen:
            seen[c] += 1
            header[i] = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
    return header


def _column_info(name: str, values: list[Any]) -> ColumnInfo:
    """Infer a column's type and the sample values the schema prompt shows."""
    typ = _infer_type(values)
    distinct = sorted({str(_coerce(v, typ)) for v in values if _coerce(v, typ) is not None})
    samples = distinct if (typ == "text" and len(distinct) <= MAX_ENUM_VALUES) else distinct[:NUMERIC_SAMPLES]
    return ColumnInfo(name, typ, samples, len(distinct))


def _row_text(name: str, row_no: int, cols: list[ColumnInfo], typed: list[Any]) -> str:
    """Search text for one row: "<table> row <n>: col=value; ..." over its non-empty cells.

    Wide tables (hundreds of sensor columns) are capped at MAX_ROW_TEXT_COLS so a row stays
    a readable chunk; aggregates over the dropped columns still work through SQL.
    """
    filled = [i for i in range(len(cols)) if typed[i] not in (None, "")]
    shown = filled[:MAX_ROW_TEXT_COLS]
    text = f"{name} row {row_no}: " + "; ".join(f"{cols[i].name}={typed[i]}" for i in shown)
    if len(shown) < len(filled):
        text += f"; … ({len(cols) - len(shown)} more columns)"
    return text


# ----------------------------------------------------------------------------- querying


class SQLError(Exception):
    """A rejected or failing SQL statement."""


def query_sql(sql: str, db_path: Path = DB_PATH, limit: int = 200, timeout_s: float = 5.0) -> dict:
    """Run a single read-only SELECT with a row limit and a time budget.

    Returns {"columns": [...], "rows": [[...], ...], "truncated": bool}. Raises SQLError for
    anything but one SELECT/WITH statement, for write keywords, and for SQLite errors
    (including the progress-handler abort when `timeout_s` is exceeded).
    """
    stmt = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", stmt, re.I) or ";" in stmt:
        raise SQLError("only a single SELECT statement is allowed")
    without_literals = re.sub(r"'(?:[^']|'')*'", "''", stmt)  # keywords inside string literals are data
    if re.search(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace)\b", without_literals, re.I):
        raise SQLError("statement contains a write/DDL keyword")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    deadline_steps = int(timeout_s * SQL_STEPS_PER_SECOND)
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
    for name, doc, n_rows, schema_json in conn.execute(
        "SELECT name, doc, n_rows, schema_json FROM _structured_tables ORDER BY name"
    ):
        out.append(f'TABLE "{name}"  -- source file: {doc}, {n_rows} rows; column _row = 1-based source row number')
        cols = json.loads(schema_json)
        # Wide tables (hundreds of sensor columns) are summarised: first and last few columns.
        wide = len(cols) > WIDE_TABLE_COLS
        shown = [*cols[:WIDE_HEAD_COLS], None, *cols[-WIDE_TAIL_COLS:]] if wide else cols
        for c in shown:
            if c is None:
                first_hidden, last_hidden = cols[WIDE_HEAD_COLS], cols[-WIDE_TAIL_COLS - 1]
                out.append(
                    f"  ... {len(cols) - WIDE_HEAD_COLS - WIDE_TAIL_COLS} more columns named like the ones above "
                    f"({first_hidden['name']} .. {last_hidden['name']}), all {SQL_TYPES[first_hidden['type']]}"
                )
                continue
            ex = ", ".join(c["samples"][:MAX_ENUM_VALUES])
            note = f"values: {ex}" if c["type"] == "text" and c["n_distinct"] <= MAX_ENUM_VALUES else f"e.g. {ex}"
            out.append(f'  "{c["name"]}" {SQL_TYPES[c["type"]]}  -- {c["n_distinct"]} distinct; {note}')
    conn.close()
    return "\n".join(out)


def list_tables(db_path: Path = DB_PATH) -> list[tuple[str, str, int]]:
    """(name, source file, row count) for every loaded table."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute("SELECT name, doc, n_rows FROM _structured_tables ORDER BY name").fetchall()
    conn.close()
    return rows

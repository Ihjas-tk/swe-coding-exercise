"""Structured loader must answer the dev set's structured questions exactly."""

from pathlib import Path

import pytest

from ae.extract.structured import SQLError, describe_schema, load_structured, query_sql

CORPUS = Path(__file__).resolve().parents[1] / "structured"


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "t.sqlite"
    load_structured([CORPUS / "test_log.csv", CORPUS / "bill_of_materials.xlsx"], db)
    return db


def q(db, sql):
    return query_sql(sql, db)["rows"]


def test_q17_failed_runs(db):
    assert q(db, "SELECT COUNT(*) FROM test_log WHERE result = 'FAIL'") == [[7]]


def test_q18_distinct_bms_units(db):
    assert q(db, "SELECT COUNT(DISTINCT unit_sn) FROM test_log WHERE unit_type = 'EV-BMS-100'") == [[18]]


def test_q19_highest_unit_cost(db):
    rows = q(db, "SELECT part_number, unit_cost_usd FROM bill_of_materials ORDER BY unit_cost_usd DESC LIMIT 1")
    assert rows == [["RJA-GEAR-101", 315.75]]


def test_q20_extended_cost(db):
    rows = q(
        db, "SELECT quantity, unit_cost_usd, extended_cost_usd FROM bill_of_materials WHERE part_number = 'FVT-ESC-40A'"
    )
    assert rows == [[8, 22.5, 180.0]]


def test_types_and_provenance(db):
    schema = describe_schema(db)
    assert '"measured_value" REAL' in schema and '"quantity" INTEGER' in schema and '"date" TEXT' in schema
    assert q(db, "SELECT _row FROM test_log WHERE test_id = 'T-1001'") == [[2]]
    assert q(db, "SELECT COUNT(*) FROM _structured_rows WHERE doc = 'test_log.csv'") == [[109]]


def test_read_only_guard(db):
    with pytest.raises(SQLError):
        query_sql("DELETE FROM test_log", db)
    with pytest.raises(SQLError):
        query_sql("SELECT 1; DROP TABLE test_log", db)

from pathlib import Path
import sys

import pytest


def _ensure_src_on_path() -> None:
    """Ensure that the 'src' directory is available on sys.path for imports."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

from text2sql.eval.schema import (  # noqa: E402  # isort: skip
    parse_create_table_context,
    referenced_identifiers,
    schema_adherence,
)


SCHEMA_CONTEXT = """
CREATE TABLE head (
    id INTEGER PRIMARY KEY,
    age INTEGER,
    department TEXT
);

CREATE TABLE manager (
    id INTEGER PRIMARY KEY,
    name TEXT
);
"""


def test_parse_create_table_context_extracts_tables_and_columns() -> None:
    parsed = parse_create_table_context(SCHEMA_CONTEXT)
    tables = parsed["tables"]
    columns_by_table = parsed["columns_by_table"]

    assert "head" in tables
    assert "manager" in tables

    assert "age" in columns_by_table["head"]
    assert "department" in columns_by_table["head"]
    assert "name" in columns_by_table["manager"]


def test_referenced_identifiers_finds_tables_and_columns() -> None:
    sql = "SELECT department, COUNT(*) FROM head WHERE age > 56"
    refs = referenced_identifiers(sql)

    assert "head" in refs["tables"]
    # We only track column names (lowercased, without table prefixes).
    assert "department" in refs["columns"]
    assert "age" in refs["columns"]


def test_schema_adherence_true_for_known_tables_and_columns() -> None:
    sql = "SELECT department FROM head WHERE age > 56"
    assert schema_adherence(sql, SCHEMA_CONTEXT) is True


def test_schema_adherence_false_for_unknown_table() -> None:
    sql = "SELECT department FROM unknown_table WHERE age > 56"
    assert schema_adherence(sql, SCHEMA_CONTEXT) is False


def test_schema_adherence_false_for_unknown_column() -> None:
    sql = "SELECT salary FROM head WHERE age > 56"
    assert schema_adherence(sql, SCHEMA_CONTEXT) is False
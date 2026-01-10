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


@pytest.fixture()
def schema_context() -> str:
    return (
        "CREATE TABLE employee (id INTEGER, name TEXT, department_id INTEGER); "
        "CREATE TABLE department (id INTEGER, name TEXT);"
    )


def test_parse_create_table_context_extracts_tables_and_columns(schema_context: str) -> None:
    info = parse_create_table_context(schema_context)
    assert "employee" in info["tables"]
    assert "department" in info["tables"]

    employee_cols = info["columns_by_table"]["employee"]
    assert {"id", "name", "department_id"}.issubset(employee_cols)


def test_schema_adherence_true_when_only_known_tables_and_columns_used(
    schema_context: str,
) -> None:
    sql = "SELECT e.name, d.name FROM employee e JOIN department d ON e.department_id = d.id"
    assert schema_adherence(sql, schema_context)


def test_schema_adherence_false_for_unknown_table(schema_context: str) -> None:
    sql = "SELECT name FROM unknown_table"
    assert not schema_adherence(sql, schema_context)


def test_schema_adherence_false_for_unknown_column(schema_context: str) -> None:
    sql = "SELECT bogus FROM employee"
    assert not schema_adherence(sql, schema_context)


def test_referenced_identifiers_extracts_tables_and_columns() -> None:
    sql = "SELECT e.name, d.name FROM employee e JOIN department d ON e.department_id = d.id"
    refs = referenced_identifiers(sql)
    assert "employee" in refs["tables"]
    assert "department" in refs["tables"]
    assert "name" in refs["columns"]
    assert "department_id" in refs["columns"]
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql.eval.schema import (  # noqa: E402
    parse_create_table_context,
    referenced_identifiers,
    schema_adherence,
)


SCHEMA_CONTEXT = """
CREATE TABLE head (
  head_id INTEGER PRIMARY KEY,
  name TEXT,
  born_state TEXT,
  age INTEGER
);

CREATE TABLE department (
  department_id INTEGER PRIMARY KEY,
  name TEXT,
  creation INTEGER
);
"""


def test_parse_create_table_context_extracts_tables_and_columns():
    schema = parse_create_table_context(SCHEMA_CONTEXT)

    tables = schema["tables"]
    columns_by_table = schema["columns_by_table"]

    assert "head" in tables
    assert "department" in tables
    assert "age" in columns_by_table["head"]
    assert "name" in columns_by_table["department"]


def test_referenced_identifiers_finds_tables_and_columns():
    sql = "SELECT name, age FROM head"
    ids = referenced_identifiers(sql)

    assert "head" in ids["tables"]
    assert "name" in ids["columns"]
    assert "age" in ids["columns"]


def test_schema_adherence_true_for_valid_query():
    sql = "SELECT name, age FROM head WHERE age > 30"
    assert schema_adherence(sql, SCHEMA_CONTEXT) is True


def test_schema_adherence_false_for_unknown_column():
    sql = "SELECT foo FROM head"
    assert schema_adherence(sql, SCHEMA_CONTEXT) is False


def test_schema_adherence_false_for_unknown_table():
    sql = "SELECT age FROM unknown_table"
    assert schema_adherence(sql, SCHEMA_CONTEXT) is False
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

from text2sql.data_prep import (  # noqa: E402  # isort: skip
    build_input_text,
    format_record,
    normalize_sql,
)


@pytest.fixture()
def sample_fields() -> dict:
    return {
        "question": "How many heads of the departments are older than 56 ?",
        "context": "CREATE TABLE head (age INTEGER)",
        "answer": "SELECT  COUNT(*)  \nFROM head\nWHERE  age  >  56",
    }


def test_format_record_contains_required_keys(sample_fields: dict) -> None:
    record = format_record(
        question=sample_fields["question"],
        context=sample_fields["context"],
        answer=sample_fields["answer"],
    )

    for key in ("instruction", "input", "output", "source"):
        assert key in record, f"Missing key '{key}' in formatted record."

    assert record["source"] == "b-mc2/sql-create-context"


def test_build_input_text_includes_schema_and_question(sample_fields: dict) -> None:
    text = build_input_text(
        context=sample_fields["context"],
        question=sample_fields["question"],
    )

    # Basic structure markers.
    assert "Schema" in text
    assert "Question" in text

    # Ensure schema and question content are present.
    assert "CREATE TABLE head" in text
    assert "How many heads of the departments are older than 56 ?" in text


def test_normalize_sql_collapses_whitespace(sample_fields: dict) -> None:
    normalized = normalize_sql(sample_fields["answer"])

    # No leading/trailing whitespace.
    assert normalized == normalized.strip()
    # Internal whitespace collapsed.
    assert "  " not in normalized
    # Still contains critical SQL tokens.
    assert "SELECT COUNT(*) FROM head WHERE age > 56" in normalized
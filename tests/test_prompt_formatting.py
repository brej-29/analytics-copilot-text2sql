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

from text2sql.training.formatting import (  # noqa: E402  # isort: skip
    build_prompt,
    ensure_sql_only,
)


@pytest.fixture()
def sample_data() -> dict:
    return {
        "instruction": "Write a SQL query that answers the user's question using ONLY the tables and columns provided in the schema.",
        "input": "### Schema:\nCREATE TABLE head (age INTEGER)\n\n### Question:\nHow many heads of the departments are older than 56 ?",
        "output": "```sql\nSELECT  COUNT(*)  \nFROM head\nWHERE  age  >  56\n```",
    }


def test_build_prompt_includes_markers_and_content(sample_data: dict) -> None:
    prompt = build_prompt(sample_data["instruction"], sample_data["input"])

    assert "Instruction" in prompt
    assert "Input" in prompt
    assert "Response" in prompt

    assert sample_data["instruction"] in prompt
    assert sample_data["input"] in prompt


def test_ensure_sql_only_strips_fences_and_normalizes_whitespace(
    sample_data: dict,
) -> None:
    cleaned = ensure_sql_only(sample_data["output"])

    # Fences should be removed
    assert "```" not in cleaned.lower()

    # Whitespace should be normalized (no double spaces)
    assert "  " not in cleaned

    # SQL content should remain intact
    assert cleaned.startswith("SELECT")
    assert "FROM head" in cleaned
    assert "WHERE age > 56" in cleaned
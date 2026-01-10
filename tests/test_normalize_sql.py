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

from text2sql.eval.normalize import (  # noqa: E402  # isort: skip
    normalize_sql,
    normalize_sql_no_values,
)


def test_normalize_sql_strips_whitespace_and_trailing_semicolons() -> None:
    raw = "  \n\tSELECT  *  FROM  head  ; \n"
    normalized = normalize_sql(raw)

    assert normalized == "SELECT * FROM head"
    assert normalized.endswith(";") is False
    assert "  " not in normalized


def test_normalize_sql_no_values_replaces_literals() -> None:
    raw = "SELECT * FROM head WHERE name = 'Alice' AND age >= 30;"
    normalized = normalize_sql_no_values(raw)

    # Should normalize structural SQL.
    assert normalized.startswith("SELECT * FROM head WHERE")
    # String literal should be replaced.
    assert "Alice" not in normalized
    assert "__STR__" in normalized
    # Numeric literal should be replaced.
    assert "30" not in normalized
    assert "__NUM__" in normalized
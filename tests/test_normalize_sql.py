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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SELECT * FROM t;", "SELECT * FROM t"),
        ("  SELECT  a,  b  FROM   t  ;  ", "SELECT a, b FROM t"),
        ("\nSELECT\n*\nFROM\n t  \n;;", "SELECT * FROM t"),
    ],
)
def test_normalize_sql_strips_and_collapses_whitespace(raw: str, expected: str) -> None:
    assert normalize_sql(raw) == expected


def test_normalize_sql_no_values_replaces_literals() -> None:
    sql1 = "SELECT * FROM t WHERE name = 'Alice' AND age > 42;"
    sql2 = "SELECT * FROM t WHERE name = 'Bob' AND age > 99"
    norm1 = normalize_sql_no_values(sql1)
    norm2 = normalize_sql_no_values(sql2)

    # Both queries differ only in literal values, so they should be equal
    # under the no-values normalization.
    assert norm1 == norm2
    assert "__str__" in norm1
    assert "__num__" in norm1
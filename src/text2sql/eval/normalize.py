import re
from typing import Final


_WHITESPACE_RE: Final = re.compile(r"\s+")
_TRAILING_SEMICOLONS_RE: Final = re.compile(r";+\s*$")
_SINGLE_QUOTED_STRING_RE: Final = re.compile(r"'(?:''|[^'])*'")
_DOUBLE_QUOTED_STRING_RE: Final = re.compile(r'"(?:\\"|[^"])*"')
_FLOAT_RE: Final = re.compile(r"\b\d+\.\d+\b")
_INT_RE: Final = re.compile(r"\b\d+\b")


def normalize_sql(sql: str | None) -> str:
    """
    Normalize a SQL string for comparison.

    The normalization is intentionally conservative and focuses on:
    - Stripping leading/trailing whitespace.
    - Removing trailing semicolons.
    - Collapsing runs of whitespace (spaces, tabs, newlines) into a single space.
    """
    if not sql:
        return ""

    text = sql.strip()
    # Drop any number of trailing semicolons.
    text = _TRAILING_SEMICOLONS_RE.sub("", text)
    # Collapse all internal whitespace.
    text = _WHITESPACE_RE.sub(" ", text)

    return text


def normalize_sql_no_values(sql: str | None) -> str:
    """
    Normalize a SQL string while abstracting away literal values.

    This is used for "no-values" exact match, where differences in literal
    numbers or strings are ignored. The transformation is:

    - Apply ``normalize_sql``.
    - Replace single- and double-quoted string literals with a placeholder.
    - Replace integer and floating-point numeric literals with a placeholder.
    """
    normalized = normalize_sql(sql)

    if not normalized:
        return ""

    # Replace string literals first so that digits inside them are not
    # touched by the numeric regexes.
    normalized = _SINGLE_QUOTED_STRING_RE.sub("'__str__'", normalized)
    normalized = _DOUBLE_QUOTED_STRING_RE.sub('"__str__"', normalized)

    # Replace numeric literals (floats first to avoid double replacement).
    normalized = _FLOAT_RE.sub("__num__", normalized)
    normalized = _INT_RE.sub("__num__", normalized)

    return normalized
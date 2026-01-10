from __future__ import annotations

import re

from text2sql.data_prep import normalize_sql as _base_normalize_sql


def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL string for string-based comparisons.

    This wraps the lighter normalization used during data preparation and adds:

    - Removal of trailing semicolons.
    - Final trim of leading/trailing whitespace.

    The goal is to make logically equivalent queries compare equal even if they
    differ in trailing semicolons or whitespace layout.
    """
    base = _base_normalize_sql(sql)
    if not base:
        return ""

    # Remove one or more trailing semicolons plus any surrounding whitespace.
    normalized = re.sub(r";+\s*$", "", base)
    return normalized.strip()


_VALUE_NUMERIC_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
_VALUE_STRING_PATTERN = re.compile(r"('([^']*)'|\"([^\"]*)\")")


def normalize_sql_no_values(sql: str) -> str:
    """
    Normalize SQL while masking literal values (strings and numbers).

    This is used for "no-values EM", where we care about the query structure
    but do not penalize differences in literal constants.

    Steps:
    - Apply `normalize_sql`.
    - Replace quoted string literals with a generic placeholder token.
    - Replace numeric literals with the same placeholder token.
    """
    text = normalize_sql(sql)
    if not text:
        return ""

    # Replace quoted string literals (single or double quotes) with a placeholder.
    text = _VALUE_STRING_PATTERN.sub("'<value>'", text)

    # Replace numeric literals with the same placeholder.
    text = _VALUE_NUMERIC_PATTERN.sub("<value>", text)

    # Collapse any whitespace runs introduced by substitutions.
    text = re.sub(r"\s+", " ", text)
    return text.strip()
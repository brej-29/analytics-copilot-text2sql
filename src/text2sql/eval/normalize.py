import re
from typing import Final

# Placeholders used when stripping literal values from SQL.
_NUM_PLACEHOLDER: Final[str] = "__NUM__"
_STR_PLACEHOLDER: Final[str] = "__STR__"


def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL string for string-based comparison.

    The normalization is intentionally conservative and focuses on:
    - Stripping leading/trailing whitespace.
    - Removing trailing semicolons.
    - Collapsing runs of whitespace (spaces, tabs, newlines) into a single space.

    Parameters
    ----------
    sql : str
        Raw SQL string.

    Returns
    -------
    str
        Normalized SQL string.
    """
    if sql is None:
        return ""

    text = sql.strip()
    if not text:
        return ""

    # Remove one or more trailing semicolons plus any following whitespace.
    text = re.sub(r";\s*$", "", text)

    # Collapse all whitespace (spaces, tabs, newlines) into a single space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_sql_no_values(sql: str) -> str:
    """
    Normalize a SQL string while stripping literal values.

    This function builds on `normalize_sql` and additionally:
    - Replaces single-quoted string literals with a placeholder.
    - Replaces numeric literals (integers and decimals, optionally negative)
      with a placeholder.

    The goal is to support "no-values" exact match evaluations that ignore
    concrete literal values but still compare query structure.

    Parameters
    ----------
    sql : str
        Raw SQL string.

    Returns
    -------
    str
        Normalized SQL string with literal values replaced by placeholders.
    """
    text = normalize_sql(sql)
    if not text:
        return ""

    # Replace single-quoted string literals, e.g., 'Alice' -> '__STR__'.
    # This is a best-effort approach and does not handle all SQL dialect edge cases.
    text = re.sub(r"'([^']*)'", f"'{_STR_PLACEHOLDER}'", text)

    # Replace numeric literals, e.g., 42, -3.14 -> '__NUM__'.
    # We use word boundaries to avoid touching identifiers like col1 or t2_name.
    text = re.sub(r"\b-?\d+(?:\.\d+)?\b", _NUM_PLACEHOLDER, text)

    # Collapse whitespace again in case substitutions introduced irregular spacing.
    text = re.sub(r"\s+", " ", text)

    return text.strip()
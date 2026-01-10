"""
Evaluation utilities for text-to-SQL models.

This subpackage contains helpers for:
- SQL normalization (with and without literal values).
- Simple schema parsing from CREATE TABLE context.
- Best-effort schema adherence checks.
- Aggregate evaluation metrics (exact match, parse success, etc.).
"""

from .normalize import normalize_sql, normalize_sql_no_values
from .schema import parse_create_table_context, referenced_identifiers, schema_adherence
from .metrics import exact_match, aggregate_metrics

__all__ = [
    "normalize_sql",
    "normalize_sql_no_values",
    "parse_create_table_context",
    "referenced_identifiers",
    "schema_adherence",
    "exact_match",
    "aggregate_metrics",
]
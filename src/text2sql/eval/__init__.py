"""
Evaluation utilities for the Analytics Copilot (Text-to-SQL) project.

This subpackage contains:
- SQL normalization helpers (normalize_sql, normalize_sql_no_values).
- Schema parsing and adherence checks for CREATE TABLE context.
- Aggregate metrics for text-to-SQL evaluation.
- Spider-specific prompt construction helpers.
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
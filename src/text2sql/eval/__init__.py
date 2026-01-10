"""
Evaluation utilities for the Analytics Copilot (Text-to-SQL) project.

This subpackage provides:
- SQL normalization helpers.
- Schema parsing and adherence checks.
- Aggregated evaluation metrics.

The modules are intentionally lightweight so they can be reused in
both offline tests and full evaluation scripts.
"""

from .normalize import normalize_sql, normalize_sql_no_values  # noqa: F401
from .schema import (  # noqa: F401
    parse_create_table_context,
    referenced_identifiers,
    schema_adherence,
)
from .metrics import exact_match, aggregate_metrics  # noqa: F401

__all__ = [
    "normalize_sql",
    "normalize_sql_no_values",
    "parse_create_table_context",
    "referenced_identifiers",
    "schema_adherence",
    "exact_match",
    "aggregate_metrics",
]
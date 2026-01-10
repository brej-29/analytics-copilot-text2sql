from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import sqlglot
from sqlglot.errors import ParseError

from .normalize import normalize_sql, normalize_sql_no_values
from .schema import schema_adherence


def exact_match(pred: str, gold: str) -> bool:
    """
    Return True if `pred` and `gold` match under normalized comparison.

    Normalization strips leading/trailing whitespace, collapses whitespace runs,
    and removes trailing semicolons.
    """
    return normalize_sql(pred) == normalize_sql(gold)


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def aggregate_metrics(
    preds: Sequence[str],
    golds: Sequence[str],
    contexts: Optional[Sequence[str]] = None,
    compute_schema_adherence: bool = False,
) -> Dict[str, Any]:
    """
    Compute aggregate evaluation metrics over a set of predictions.

    Metrics:
    - em: exact match on normalized SQL.
    - no_values_em: exact match after masking literals.
    - parse_success_rate: fraction of predictions that sqlglot can parse.
    - schema_adherence_rate: fraction of predictions that respect the schema
      defined in `contexts` (only if `compute_schema_adherence=True`).

    Parameters
    ----------
    preds : Sequence[str]
        Predicted SQL strings.
    golds : Sequence[str]
        Gold/reference SQL strings. Must be the same length as `preds`.
    contexts : Optional[Sequence[str]]
        Optional CREATE TABLE context strings corresponding to each example.
        Required when `compute_schema_adherence` is True.
    compute_schema_adherence : bool
        Whether to compute schema adherence.

    Returns
    -------
    dict
        Dictionary of aggregate metrics.
    """
    if len(preds) != len(golds):
        raise ValueError(
            f"preds and golds must have the same length; "
            f"got {len(preds)} and {len(golds)}."
        )

    if compute_schema_adherence:
        if contexts is None or len(contexts) != len(preds):
            raise ValueError(
                "When compute_schema_adherence=True, `contexts` must be provided "
                "and have the same length as preds/golds."
            )

    total = len(preds)
    em_count = 0
    no_values_em_count = 0
    parse_success_count = 0
    schema_ok_count = 0

    for idx, (pred, gold) in enumerate(zip(preds, golds)):
        if exact_match(pred, gold):
            em_count += 1

        if normalize_sql_no_values(pred) == normalize_sql_no_values(gold):
            no_values_em_count += 1

        try:
            sqlglot.parse_one(pred)
            parse_success_count += 1
        except ParseError:
            pass
        except Exception:
            # Any unexpected failure to parse is treated as a parse failure.
            pass

        if compute_schema_adherence and contexts is not None:
            context = contexts[idx]
            if schema_adherence(pred, context):
                schema_ok_count += 1

    metrics: Dict[str, Any] = {
        "num_examples": total,
        "em": _rate(em_count, total),
        "no_values_em": _rate(no_values_em_count, total),
        "parse_success_rate": _rate(parse_success_count, total),
    }

    if compute_schema_adherence:
        metrics["schema_adherence_rate"] = _rate(schema_ok_count, total)

    return metrics
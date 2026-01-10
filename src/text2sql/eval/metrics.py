from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .normalize import normalize_sql, normalize_sql_no_values
from .schema import is_parsable_sql, schema_adherence


@dataclass
class MetricFlags:
    """Configuration flags for aggregate_metrics."""

    compute_schema_adherence: bool = False


def exact_match(pred: str | None, gold: str | None) -> bool:
    """
    Exact match on normalized SQL strings.

    Normalization:
    - Strip leading/trailing whitespace.
    - Remove trailing semicolons.
    - Collapse internal whitespace runs.
    """
    return normalize_sql(pred) == normalize_sql(gold)


def no_values_exact_match(pred: str | None, gold: str | None) -> bool:
    """
    Exact match on SQL strings after normalizing and abstracting away values.

    Uses :func:`normalize_sql_no_values` on both prediction and gold.
    """
    return normalize_sql_no_values(pred) == normalize_sql_no_values(gold)


def aggregate_metrics(
    preds: Sequence[str],
    golds: Sequence[str],
    *,
    contexts: Optional[Sequence[str]] = None,
    flags: Optional[MetricFlags] = None,
) -> Dict[str, float]:
    """
    Aggregate core evaluation metrics over a batch of predictions.

    Parameters
    ----------
    preds:
        Model-predicted SQL strings.
    golds:
        Gold/reference SQL strings.
    contexts:
        Optional iterable of schema contexts (CREATE TABLE statements) aligned
        with preds/golds. Required if ``flags.compute_schema_adherence`` is True.
    flags:
        Optional MetricFlags controlling which extra metrics to compute.

    Returns
    -------
    dict
        A mapping with keys such as:
        - num_examples
        - exact_match
        - no_values_exact_match
        - parse_success_rate
        - schema_adherence_rate (if requested)
    """
    if len(preds) != len(golds):
        raise ValueError(
            f"preds and golds must have the same length, "
            f"got {len(preds)} and {len(golds)}."
        )

    if flags is None:
        flags = MetricFlags()

    n = len(preds)
    if n == 0:
        return {
            "num_examples": 0,
            "exact_match": 0.0,
            "no_values_exact_match": 0.0,
            "parse_success_rate": 0.0,
            **({"schema_adherence_rate": 0.0} if flags.compute_schema_adherence else {}),
        }

    if flags.compute_schema_adherence and contexts is not None:
        if len(contexts) != n:
            raise ValueError(
                f"contexts length must match preds/golds when computing schema adherence, "
                f"got {len(contexts)} vs {n}."
            )

    em_count = 0
    no_values_em_count = 0
    parse_success_count = 0
    schema_adherence_count = 0

    for idx in range(n):
        pred = preds[idx]
        gold = golds[idx]

        if exact_match(pred, gold):
            em_count += 1

        if no_values_exact_match(pred, gold):
            no_values_em_count += 1

        if is_parsable_sql(pred):
            parse_success_count += 1

        if flags.compute_schema_adherence and contexts is not None:
            context = contexts[idx]
            if schema_adherence(pred, context):
                schema_adherence_count += 1

    metrics: Dict[str, float] = {
        "num_examples": float(n),
        "exact_match": em_count / n,
        "no_values_exact_match": no_values_em_count / n,
        "parse_success_rate": parse_success_count / n,
    }

    if flags.compute_schema_adherence and contexts is not None:
        metrics["schema_adherence_rate"] = schema_adherence_count / n

    return metrics
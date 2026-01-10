from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import sqlglot

from .normalize import normalize_sql, normalize_sql_no_values
from .schema import schema_adherence as _schema_adherence


def exact_match(pred: str, gold: str) -> bool:
    """
    Exact match on normalized SQL strings.

    Both prediction and gold SQL are normalized with `normalize_sql` before
    comparison to make the metric robust to trivial formatting differences
    (whitespace, trailing semicolons).
    """
    return normalize_sql(pred) == normalize_sql(gold)


def _parse_success(sql: str) -> bool:
    """
    Return True if sqlglot can parse the given SQL string.

    This is used as a lightweight proxy for syntactic validity of the query.
    """
    if not sql or not sql.strip():
        return False
    try:
        sqlglot.parse_one(sql)
        return True
    except Exception:  # noqa: BLE001
        return False


def aggregate_metrics(
    predictions: Sequence[str],
    golds: Sequence[str],
    *,
    contexts: Optional[Sequence[str]] = None,
    compute_schema_adherence: bool = False,
) -> Dict[str, object]:
    """
    Aggregate core text-to-SQL evaluation metrics over a set of examples.

    Parameters
    ----------
    predictions : Sequence[str]
        Model-predicted SQL strings.
    golds : Sequence[str]
        Gold/reference SQL strings.
    contexts : Optional[Sequence[str]], optional
        Schema context strings (e.g., CREATE TABLE statements). Required if
        `compute_schema_adherence` is True.
    compute_schema_adherence : bool, optional
        Whether to compute the schema adherence rate using the provided
        contexts, by default False.

    Returns
    -------
    dict
        Dictionary containing counts and rates for:
        - n_examples
        - exact_match
        - no_values_em
        - parse_success
        - schema_adherence (if requested)
    """
    if len(predictions) != len(golds):
        raise ValueError(
            f"Expected predictions and golds to have the same length, "
            f"got {len(predictions)} and {len(golds)}."
        )

    if compute_schema_adherence:
        if contexts is None:
            raise ValueError(
                "contexts must be provided when compute_schema_adherence is True."
            )
        if len(contexts) != len(predictions):
            raise ValueError(
                f"Expected contexts to have the same length as predictions, "
                f"got {len(contexts)} and {len(predictions)}."
            )

    n_examples = len(predictions)
    if n_examples == 0:
        return {
            "n_examples": 0,
            "exact_match": {"count": 0, "rate": 0.0},
            "no_values_em": {"count": 0, "rate": 0.0},
            "parse_success": {"count": 0, "rate": 0.0},
            "schema_adherence": {"count": 0, "rate": 0.0}
            if compute_schema_adherence
            else None,
        }

    em_count = 0
    no_values_em_count = 0
    parse_success_count = 0
    schema_adherence_count = 0

    for idx, (pred, gold) in enumerate(zip(predictions, golds)):
        if exact_match(pred, gold):
            em_count += 1

        pred_no_vals = normalize_sql_no_values(pred)
        gold_no_vals = normalize_sql_no_values(gold)
        if pred_no_vals == gold_no_vals:
            no_values_em_count += 1

        if _parse_success(pred):
            parse_success_count += 1

        if compute_schema_adherence and contexts is not None:
            ctx = contexts[idx]
            if _schema_adherence(pred, ctx):
                schema_adherence_count += 1

    def _rate(count: int) -> float:
        return float(count) / float(n_examples) if n_examples else 0.0

    result: Dict[str, object] = {
        "n_examples": n_examples,
        "exact_match": {"count": em_count, "rate": _rate(em_count)},
        "no_values_em": {
            "count": no_values_em_count,
            "rate": _rate(no_values_em_count),
        },
        "parse_success": {
            "count": parse_success_count,
            "rate": _rate(parse_success_count),
        },
    }

    if compute_schema_adherence:
        result["schema_adherence"] = {
            "count": schema_adherence_count,
            "rate": _rate(schema_adherence_count),
        }

    return result
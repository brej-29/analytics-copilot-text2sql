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

from text2sql.eval.metrics import (  # noqa: E402  # isort: skip
    MetricFlags,
    aggregate_metrics,
)


def test_aggregate_metrics_computes_em_and_no_values_em() -> None:
    preds = [
        "SELECT a FROM t;",
        "SELECT a FROM t WHERE b = 1;",
        "SELECT * FROM x",
    ]
    golds = [
        "SELECT a FROM t",
        "SELECT a FROM t WHERE b = 2",
        "SELECT * FROM x",
    ]
    contexts = [
        "CREATE TABLE t (a INT, b INT);",
        "CREATE TABLE t (a INT, b INT);",
        "CREATE TABLE x (id INT);",
    ]

    flags = MetricFlags(compute_schema_adherence=True)
    metrics = aggregate_metrics(preds, golds, contexts=contexts, flags=flags)

    assert metrics["num_examples"] == 3.0
    # First and third are exact matches after normalization.
    assert pytest.approx(metrics["exact_match"], rel=1e-6) == 2.0 / 3.0
    # All three should match when values are abstracted away.
    assert pytest.approx(metrics["no_values_exact_match"], rel=1e-6) == 1.0
    # All three queries are syntactically valid.
    assert pytest.approx(metrics["parse_success_rate"], rel=1e-6) == 1.0
    # All references are within the schema.
    assert pytest.approx(metrics["schema_adherence_rate"], rel=1e-6) == 1.0
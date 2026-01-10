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

from text2sql.eval.metrics import aggregate_metrics  # noqa: E402  # isort: skip


SCHEMA_CONTEXT = """
CREATE TABLE head (
    id INTEGER PRIMARY KEY,
    age INTEGER,
    department TEXT
);
"""


def test_aggregate_metrics_basic_counts_and_rates() -> None:
    preds = [
        "SELECT department FROM head WHERE age > 56",
        "SELECT age FROM head WHERE age > 30",
    ]
    golds = [
        "SELECT department FROM head WHERE age > 56",
        "SELECT age FROM head WHERE age > 40",
    ]
    contexts = [SCHEMA_CONTEXT, SCHEMA_CONTEXT]

    metrics = aggregate_metrics(
        predictions=preds,
        golds=golds,
        contexts=contexts,
        compute_schema_adherence=True,
    )

    assert metrics["n_examples"] == 2

    # First example is exact match; second differs by literal.
    em = metrics["exact_match"]
    assert em["count"] == 1
    assert em["rate"] == pytest.approx(0.5)

    # No-values EM should treat both as matches.
    nvem = metrics["no_values_em"]
    assert nvem["count"] == 2
    assert nvem["rate"] == pytest.approx(1.0)

    # All predictions should parse and adhere to the schema.
    parse = metrics["parse_success"]
    assert parse["count"] == 2
    assert parse["rate"] == pytest.approx(1.0)

    schema = metrics["schema_adherence"]
    assert schema["count"] == 2
    assert schema["rate"] == pytest.approx(1.0)
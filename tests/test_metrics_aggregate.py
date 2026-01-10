import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql.eval.metrics import aggregate_metrics, exact_match  # noqa: E402


def test_exact_match_with_normalization():
    gold = "SELECT *  FROM  head WHERE age > 56;"
    pred = "  SELECT * FROM head   WHERE age  > 56  ;  "
    assert exact_match(pred, gold)


def test_aggregate_metrics_basic_with_schema_adherence():
    preds = [
        "SELECT age FROM head",
        "SELECT creation FROM department GROUP BY creation",
    ]
    golds = [
        "SELECT age FROM head",
        "SELECT creation FROM department GROUP BY creation",
    ]
    contexts = [
        "CREATE TABLE head (age INTEGER);",
        "CREATE TABLE department (creation INTEGER);",
    ]

    metrics = aggregate_metrics(
        preds=preds,
        golds=golds,
        contexts=contexts,
        compute_schema_adherence=True,
    )

    assert metrics["num_examples"] == 2
    assert metrics["em"] == 1.0
    assert metrics["no_values_em"] == 1.0
    assert metrics["parse_success_rate"] == 1.0
    assert metrics["schema_adherence_rate"] == 1.0
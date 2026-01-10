import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql.eval.normalize import normalize_sql, normalize_sql_no_values  # noqa: E402


def test_normalize_sql_trims_and_collapses_whitespace_and_semicolons():
    raw = "  SELECT  *  FROM   head   WHERE  age > 56 ;  "
    normalized = normalize_sql(raw)
    assert normalized == "SELECT * FROM head WHERE age > 56"


def test_normalize_sql_no_values_masks_literals():
    raw = "SELECT * FROM head WHERE age > 56 AND name = 'Alice';"
    masked = normalize_sql_no_values(raw)

    # No raw literals should remain.
    assert "56" not in masked
    assert "Alice" not in masked

    # Structure should be preserved.
    assert "SELECT * FROM head WHERE age >" in masked
    assert "AND name =" in masked
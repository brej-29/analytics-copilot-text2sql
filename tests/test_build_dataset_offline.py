from pathlib import Path
import json
import subprocess
import sys

import pytest


def test_build_dataset_offline_uses_fixture_and_writes_jsonl(tmp_path: Path) -> None:
    """
    Ensure the build_dataset script can run in offline mode using a local JSONL fixture.

    This test does NOT require internet access. It uses the --input_jsonl flag
    to bypass Hugging Face download and writes output into a temporary directory.
    """
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "build_dataset.py"
    fixture_path = root / "tests" / "fixtures" / "sql_create_context_sample.jsonl"

    assert script_path.is_file(), f"Script not found at {script_path}"
    assert fixture_path.is_file(), f"Fixture not found at {fixture_path}"

    out_dir = tmp_path / "processed"

    cmd = [
        sys.executable,
        str(script_path),
        "--input_jsonl",
        str(fixture_path),
        "--out_dir",
        str(out_dir),
        "--val_ratio",
        "0.4",
        "--seed",
        "123",
        "--overwrite",
    ]

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(
            f"build_dataset.py failed with exit code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    assert train_path.is_file(), "Train JSONL was not created."
    assert val_path.is_file(), "Val JSONL was not created."

    def _read_jsonl(path: Path) -> list[dict]:
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)

    assert train_rows, "Train JSONL is empty."
    assert val_rows, "Val JSONL is empty."

    for row in train_rows + val_rows:
        for key in ("id", "instruction", "input", "output", "source", "meta"):
            assert key in row, f"Missing key '{key}' in output record."
        assert isinstance(row["meta"], dict), "meta must be a dict."
        assert row["source"] == "b-mc2/sql-create-context"
from pathlib import Path
import json
import subprocess
import sys

import pytest


def test_prompt_building_spider_mock_uses_fixtures_and_writes_reports(tmp_path: Path) -> None:
    """
    Ensure the Spider external evaluation script can run in --mock mode
    using local fixtures only, and that it writes a JSON report containing
    prompts built with the expected schema/question structure.
    """
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "evaluate_spider_external.py"
    assert script_path.is_file(), f"Script not found at {script_path}"

    out_dir = tmp_path / "reports"

    cmd = [
        sys.executable,
        str(script_path),
        "--mock",
        "--max_examples",
        "2",
        "--out_dir",
        str(out_dir),
    ]

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(
            f"evaluate_spider_external.py failed with exit code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    json_path = out_dir / "eval_spider.json"
    assert json_path.is_file(), "Spider evaluation JSON report was not created."

    data = json.loads(json_path.read_text(encoding="utf-8"))
    examples = data.get("examples")
    assert examples, "No examples found in Spider evaluation JSON."

    example = examples[0]
    prompt = example.get("prompt", "")
    question = example.get("question", "")

    assert "### Schema:" in prompt
    assert "### Question:" in prompt
    assert "### Instruction:" in prompt
    assert question in prompt
from pathlib import Path
import json
import sys

import pytest


def _ensure_src_on_path() -> None:
    """Ensure that the 'src' directory is available on sys.path for imports."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

from text2sql.eval.spider import (  # noqa: E402  # isort: skip
    build_schema_map,
    build_spider_prompt,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@pytest.fixture()
def spider_fixtures() -> tuple[list[dict], dict[str, str]]:
    root = Path(__file__).resolve().parents[1]
    fixtures_dir = root / "tests" / "fixtures"
    spider_examples_path = fixtures_dir / "spider_sample.jsonl"
    spider_schema_path = fixtures_dir / "spider_schema_sample.jsonl"

    examples = _load_jsonl(spider_examples_path)
    schema_records = _load_jsonl(spider_schema_path)
    schema_map = build_schema_map(schema_records)

    return examples, schema_map


def test_build_spider_prompt_uses_schema_and_question(spider_fixtures: tuple[list[dict], dict[str, str]]) -> None:
    examples, schema_map = spider_fixtures
    example = examples[0]

    db_id = example["db_id"]
    question = example["question"]
    schema_context = schema_map[db_id]

    prompt = build_spider_prompt(schema_context=schema_context, question=question)

    # Prompt should include markers from training-style formatting.
    assert "Instruction" in prompt
    assert "Input" in prompt
    assert "Response" in prompt

    # Ensure schema and question are present.
    assert "### Schema:" in prompt
    assert "CREATE TABLE" in prompt
    assert "### Question:" in prompt
    assert question in prompt
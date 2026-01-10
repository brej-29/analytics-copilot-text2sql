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
    build_spider_prompt,
    load_spider_schema_map,
    spider_schema_to_pseudo_ddl,
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
    schema_map = load_spider_schema_map(schema_records)

    return examples, schema_map


def test_build_spider_prompt_uses_schema_and_question(
    spider_fixtures: tuple[list[dict], dict[str, str]]
) -> None:
    examples, schema_map = spider_fixtures
    example = examples[0]

    db_id = example["db_id"]
    question = example["question"]
    raw_schema_text = schema_map[db_id]
    schema_context = spider_schema_to_pseudo_ddl(raw_schema_text)

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


def test_load_spider_schema_map_handles_real_column_name(
    spider_fixtures: tuple[list[dict], dict[str, str]]
) -> None:
    _, schema_map = spider_fixtures
    # Ensure we have mappings for known db_ids from the fixtures.
    for db_id in ("concert_singer", "academic", "college_1"):
        assert db_id in schema_map
        assert isinstance(schema_map[db_id], str)
        assert schema_map[db_id]


def test_intersection_nonzero_with_fixtures(
    spider_fixtures: tuple[list[dict], dict[str, str]]
) -> None:
    examples, schema_map = spider_fixtures
    spider_db_ids = {ex["db_id"] for ex in examples}
    schema_db_ids = set(schema_map.keys())
    intersection = spider_db_ids & schema_db_ids
    assert intersection, "Expected non-empty intersection between Spider and schema db_ids"


def test_spider_schema_to_pseudo_ddl_nonempty(
    spider_fixtures: tuple[list[dict], dict[str, str]]
) -> None:
    _, schema_map = spider_fixtures
    # Take one schema text and ensure we can build a non-empty pseudo-DDL string.
    raw_schema_text = next(iter(schema_map.values()))
    pseudo_ddl = spider_schema_to_pseudo_ddl(raw_schema_text)
    assert pseudo_ddl
    assert "CREATE TABLE" in pseudo_ddl
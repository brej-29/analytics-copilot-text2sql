import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.evaluate_spider_external import (  # noqa: E402
    _schema_text_to_create_table,
    build_spider_input,
    build_spider_prompt,
)
from text2sql.data_prep import INSTRUCTION_TEXT  # noqa: E402


FIXTURES_DIR = ROOT / "tests" / "fixtures"


def _load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def test_schema_text_to_create_table_and_prompt_structure():
    spider_records = _load_jsonl(FIXTURES_DIR / "spider_sample.jsonl")
    schema_records = _load_jsonl(FIXTURES_DIR / "spider_schema_sample.jsonl")

    schema_by_db = {row["db_id"]: row["schema"] for row in schema_records}

    example = spider_records[0]
    db_id = example["db_id"]
    question = example["question"]

    schema_text = schema_by_db[db_id]
    schema_ddl = _schema_text_to_create_table(schema_text)

    # Basic sanity checks on DDL shape.
    assert "CREATE TABLE" in schema_ddl
    assert db_id.split("_")[0].lower() not in schema_ddl.lower() or "(" in schema_ddl

    # Build input and prompt.
    input_text = build_spider_input(schema_ddl=schema_ddl, question=question)
    prompt = build_spider_prompt(
        instruction=INSTRUCTION_TEXT,
        schema_ddl=schema_ddl,
        question=question,
    )

    # The input text should contain schema + question markers.
    assert "### Schema:" in input_text
    assert "### Question:" in input_text
    assert question in input_text

    # The full prompt should wrap the input in the standard instruction template.
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:" in prompt
    assert input_text in prompt
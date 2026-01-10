from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

from text2sql.data_prep import INSTRUCTION_TEXT, build_input_text
from text2sql.training.formatting import build_prompt


def build_spider_prompt(schema_context: str, question: str) -> str:
    """
    Build a text-to-SQL prompt for a Spider example.

    This reuses the same instruction text and input formatting as the
    internal b-mc2/sql-create-context pipeline:

    - Instruction: INSTRUCTION_TEXT from text2sql.data_prep.
    - Input: built via build_input_text(schema_context, question).
    - Prompt: wrapped using training.formatting.build_prompt.
    """
    input_text = build_input_text(context=schema_context, question=question)
    return build_prompt(INSTRUCTION_TEXT, input_text)


def build_schema_map(
    records: Sequence[Mapping[str, object]],
    *,
    db_id_field: str = "db_id",
    schema_field: str = "create_table_context",
) -> Dict[str, str]:
    """
    Build a mapping {db_id -> schema_context} from Spider-schema records.

    Parameters
    ----------
    records : Sequence[Mapping[str, object]]
        Iterable of records that each contain at least `db_id_field` and
        `schema_field`.
    db_id_field : str, optional
        Name of the database id field, by default "db_id".
    schema_field : str, optional
        Name of the schema/context field, by default "create_table_context".

    Returns
    -------
    dict
        Mapping from db_id (stringified) to schema/context string.
    """
    schema_map: Dict[str, str] = {}
    for row in records:
        db_id = row.get(db_id_field)
        schema_context = row.get(schema_field)
        if db_id is None or schema_context is None:
            continue
        db_id_str = str(db_id)
        schema_map[db_id_str] = str(schema_context)
    return schema_map
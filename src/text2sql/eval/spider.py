from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, Mapping, Sequence

from text2sql.data_prep import INSTRUCTION_TEXT, build_input_text
from text2sql.training.formatting import build_prompt

logger = logging.getLogger(__name__)


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

    This helper is kept for backwards compatibility in tests or scripts that
    already provide a concrete `schema_field` (e.g. a pre-built
    `create_table_context`). For the real Spider schema helper dataset, use
    ``load_spider_schema_map`` instead, which can automatically detect the
    schema text column.
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


_SCHEMA_FIELD_CANDIDATES: tuple[str, ...] = (
    # Exact field name used by richardr1126/spider-schema.
    "Schema (values (type))",
    # Common alternatives seen in other helper datasets.
    "schema",
    "schema_text",
    "ddl",
    "create_table",
)


def _infer_column_names(schema_ds: Any) -> list[str]:
    """
    Best-effort extraction of column names from a dataset-like object.

    Supports:
    - Hugging Face datasets with ``column_names``.
    - Simple sequences of dicts (e.g. JSONL fixture rows).
    """
    if hasattr(schema_ds, "column_names"):
        return list(schema_ds.column_names)  # type: ignore[no-any-return]

    if isinstance(schema_ds, (list, tuple)) and schema_ds:
        first = schema_ds[0]
        if isinstance(first, Mapping):
            return list(first.keys())

    # Fallback: materialise a small iterator if needed.
    try:
        iterator = iter(schema_ds)
    except TypeError:
        return []
    rows = list(iterator)
    if not rows:
        return []
    first = rows[0]
    if isinstance(first, Mapping):
        return list(first.keys())
    return []


def load_spider_schema_map(schema_ds: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """
    Load a mapping {db_id -> schema_text} from a Spider schema dataset.

    The richardr1126/spider-schema helper uses a non-standard column name
    for the serialized schema:

        \"Schema (values (type))\"

    This loader inspects available columns and chooses the first match from
    a fallback list:

        - \"Schema (values (type))\"
        - \"schema\"
        - \"schema_text\"
        - \"ddl\"
        - \"create_table\"

    If none of these are present, a ValueError is raised with the available
    columns listed.
    """
    column_names = _infer_column_names(schema_ds)
    logger.info("Spider schema dataset columns: %s", column_names)

    schema_field: str | None = None
    for candidate in _SCHEMA_FIELD_CANDIDATES:
        if candidate in column_names:
            schema_field = candidate
            break

    if schema_field is None:
        raise ValueError(
            "Could not find a schema text field in Spider schema dataset. "
            f"Available columns: {column_names}. "
            f"Tried: {_SCHEMA_FIELD_CANDIDATES}."
        )

    schema_map: Dict[str, str] = {}
    for row in schema_ds:
        db_id = row.get("db_id")
        if db_id is None:
            continue
        schema_text = row.get(schema_field)
        if not schema_text:
            continue
        db_id_str = str(db_id)
        schema_map[db_id_str] = str(schema_text)

    sample_db_ids = sorted(schema_map.keys())[:5]
    logger.info(
        "Loaded %d Spider schema entries using field '%s'. Sample db_ids: %s",
        len(schema_map),
        schema_field,
        sample_db_ids,
    )
    return schema_map


def spider_schema_to_pseudo_ddl(schema_text: str) -> str:
    """
    Convert a compact Spider schema description into pseudo-DDL.

    The richardr1126/spider-schema dataset serializes each database schema as
    a compact text description, for example::

        \"department : Department_ID (number) , Name (text) ...
         course : Course_ID (number) , Title (text) ...\"

    This function turns that into a more SQL-looking form that is aligned with
    the internal training format, eg::

        CREATE TABLE department (Department_ID NUMBER, Name TEXT, ...);
        CREATE TABLE course (Course_ID NUMBER, Title TEXT, ...);

    The exact SQL dialect is not important for evaluation; it just needs to be
    consistent and readable.
    """
    if not schema_text:
        return ""

    lines = [line.strip() for line in schema_text.splitlines() if line.strip()]
    statements: list[str] = []

    for line in lines:
        # Expected basic pattern: \"table_name : col1 (type) , col2 (type) , ...\"
        if ":" not in line:
            continue

        table_part, cols_part = line.split(":", 1)
        table_name = table_part.strip()
        if not table_name:
            continue

        column_specs: list[str] = []
        for raw_col in cols_part.split(","):
            col = raw_col.strip()
            if not col:
                continue

            # Try to match \"column_name (type)\".
            match = re.match(r"([^()]+)\(([^)]+)\)", col)
            if not match:
                # Fallback: keep the raw column name if parsing fails.
                column_name = col.strip()
                if column_name:
                    column_specs.append(column_name)
                continue

            column_name = match.group(1).strip()
            column_type = match.group(2).strip()

            type_norm = column_type.upper()
            if type_norm in {"NUMBER", "INT", "INTEGER"}:
                type_norm = "NUMBER"
            elif type_norm in {"TEXT", "STRING", "VARCHAR"}:
                type_norm = "TEXT"
            elif type_norm in {"REAL", "FLOAT", "DOUBLE"}:
                type_norm = "REAL"

            if column_name:
                column_specs.append(f"{column_name} {type_norm}")

        if column_specs:
            stmt = f"CREATE TABLE {table_name} ({', '.join(column_specs)});"
            statements.append(stmt)

    return " ".join(statements)
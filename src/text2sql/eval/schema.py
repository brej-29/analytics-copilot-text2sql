from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, MutableMapping, Set

import sqlglot
from sqlglot import expressions as exp


def parse_create_table_context(
    context: str,
) -> Dict[str, object]:
    """
    Parse a CREATE TABLE schema context into tables and columns.

    Parameters
    ----------
    context : str
        One or more CREATE TABLE statements, typically the schema context from
        b-mc2/sql-create-context or the Spider schema helper dataset.

    Returns
    -------
    dict
        A dictionary with:
        - "tables": set[str] of table names (lowercased).
        - "columns_by_table": dict[str, set[str]] mapping table -> set of
          column names (lowercased).
    """
    tables: Set[str] = set()
    columns_by_table: MutableMapping[str, Set[str]] = defaultdict(set)

    if not context or not context.strip():
        return {"tables": tables, "columns_by_table": dict(columns_by_table)}

    try:
        statements: Iterable[exp.Expression] = sqlglot.parse(context)
    except Exception:  # noqa: BLE001
        # Best-effort: if parsing fails entirely, return empty schema so that
        # adherence checks can degrade gracefully.
        return {"tables": tables, "columns_by_table": dict(columns_by_table)}

    for stmt in statements:
        if not isinstance(stmt, exp.Create):
            continue

        # For CREATE TABLE, sqlglot represents the schema as stmt.this (Schema),
        # whose .this is the Table expression and .expressions are ColumnDef nodes.
        schema_expr = stmt.this
        if not isinstance(schema_expr, exp.Schema):
            continue

        table_expr = schema_expr.this
        if not isinstance(table_expr, exp.Table):
            continue

        table_name = table_expr.name
        if not table_name:
            continue

        table_key = table_name.lower()
        tables.add(table_key)

        for col_def in schema_expr.expressions:
            if isinstance(col_def, exp.ColumnDef) and col_def.this is not None:
                col_ident = col_def.this
                col_name = getattr(col_ident, "this", None)
                if col_name:
                    columns_by_table[table_key].add(str(col_name).lower())

    return {"tables": tables, "columns_by_table": dict(columns_by_table)}


def referenced_identifiers(sql: str) -> Dict[str, Set[str]]:
    """
    Extract referenced table and column identifiers from a SQL query.

    Parameters
    ----------
    sql : str
        SQL query string.

    Returns
    -------
    dict
        Dictionary with:
        - "tables": set[str] of referenced table names (lowercased).
        - "columns": set[str] of referenced column names (lowercased).
    """
    tables: Set[str] = set()
    columns: Set[str] = set()

    if not sql or not sql.strip():
        return {"tables": tables, "columns": columns}

    try:
        expr = sqlglot.parse_one(sql)
    except Exception:  # noqa: BLE001
        # If parsing fails, we conservatively return empty sets; callers can
        # combine this with parse_success metrics separately.
        return {"tables": tables, "columns": columns}

    for table_expr in expr.find_all(exp.Table):
        name = table_expr.name
        if name:
            tables.add(name.lower())

    for col_expr in expr.find_all(exp.Column):
        # sqlglot Column expressions expose a `.name` property with the column
        # name, independent of table/alias qualification.
        col_name = getattr(col_expr, "name", None)
        if col_name:
            columns.add(str(col_name).lower())

    return {"tables": tables, "columns": columns}


def schema_adherence(sql: str, context: str) -> bool:
    """
    Check whether a SQL query only references tables/columns in a given schema.

    The check is intentionally conservative and operates purely at the level
    of identifier names:
    - A table reference is considered valid if its (lowercased) name appears
      among the tables parsed from the CREATE TABLE context.
    - A column reference is considered valid if its (lowercased) name appears
      in the union of all column sets from the parsed context.

    If the schema cannot be parsed and the query references identifiers, this
    function returns False. If the query references no identifiers, adherence
    is considered True even if the schema is empty.

    Parameters
    ----------
    sql : str
        SQL query string.
    context : str
        CREATE TABLE context string.

    Returns
    -------
    bool
        True if all referenced tables/columns are present in the schema;
        False otherwise.
    """
    schema = parse_create_table_context(context)
    schema_tables: Set[str] = schema["tables"]  # type: ignore[assignment]
    columns_by_table: Mapping[str, Set[str]] = schema[
        "columns_by_table"
    ]  # type: ignore[assignment]

    refs = referenced_identifiers(sql)
    ref_tables = refs["tables"]
    ref_columns = refs["columns"]

    if not schema_tables and not any(columns_by_table.values()):
        # No schema information available.
        if ref_tables or ref_columns:
            return False
        return True

    # Tables must be a subset of known schema tables.
    for table in ref_tables:
        if table not in schema_tables:
            return False

    # Columns must appear in at least one table's column set.
    all_columns: Set[str] = set()
    for cols in columns_by_table.values():
        all_columns.update(cols)

    for col in ref_columns:
        if col not in all_columns:
            return False

    return True
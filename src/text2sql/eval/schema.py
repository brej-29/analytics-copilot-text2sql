from __future__ import annotations

from typing import Dict, Optional, Set, TypedDict

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError


class SchemaInfo(TypedDict):
    tables: Set[str]
    columns_by_table: Dict[str, Set[str]]


class IdentifierRefs(TypedDict):
    tables: Set[str]
    columns: Set[str]


def _try_parse_sql(sql: str | None) -> Optional[exp.Expression]:
    """
    Best-effort parsing of a SQL string using sqlglot.

    Returns the parsed expression on success, otherwise None.
    """
    text = (sql or "").strip()
    if not text:
        return None

    try:
        # Use SQLite as a generic, permissive dialect for our synthetic schemas.
        return sqlglot.parse_one(text, read="sqlite")
    except (ParseError, ValueError, TypeError):
        return None
    except Exception:
        # Defensive: never let parsing blow up the caller.
        return None


def is_parsable_sql(sql: str | None) -> bool:
    """Return True if the given SQL string can be parsed by sqlglot."""
    return _try_parse_sql(sql) is not None


def parse_create_table_context(context: str | None) -> SchemaInfo:
    """
    Parse a CREATE TABLE schema context into table and column sets.

    Parameters
    ----------
    context:
        Raw context string, typically one or more CREATE TABLE statements.

    Returns
    -------
    SchemaInfo
        A mapping with:
        - tables: set of lowercased table names.
        - columns_by_table: mapping table -> set of lowercased column names.
    """
    tables: Set[str] = set()
    columns_by_table: Dict[str, Set[str]] = {}

    text = (context or "").strip()
    if not text:
        return {"tables": tables, "columns_by_table": columns_by_table}

    try:
        statements = sqlglot.parse(text, read="sqlite")
    except (ParseError, ValueError, TypeError):
        return {"tables": tables, "columns_by_table": columns_by_table}
    except Exception:
        return {"tables": tables, "columns_by_table": columns_by_table}

    for statement in statements:
        for create in statement.find_all(exp.Create):
            # In many dialects, Create.this is a Schema whose .this is the Table.
            schema_or_table = create.this
            table_expr: Optional[exp.Table]

            if isinstance(schema_or_table, exp.Table):
                table_expr = schema_or_table
                col_defs = list(create.expressions or [])
            elif isinstance(schema_or_table, exp.Schema):
                table_expr = schema_or_table.this if isinstance(schema_or_table.this, exp.Table) else None
                col_defs = list(schema_or_table.expressions or [])
            else:
                table_expr = None
                col_defs = list(create.expressions or [])

            if table_expr is None:
                continue

            table_name = (table_expr.name or "").strip()
            if not table_name:
                continue

            table_key = table_name.lower()
            tables.add(table_key)
            cols = columns_by_table.setdefault(table_key, set())

            # Column definitions are stored as ColumnDef expressions.
            for col_def in col_defs:
                if isinstance(col_def, exp.ColumnDef):
                    # ColumnDef.this is usually an Identifier.
                    identifier = col_def.this if isinstance(col_def.this, exp.Identifier) else col_def.find(exp.Identifier)
                    col_name = (identifier.name if identifier is not None else "").strip()
                    if col_name:
                        cols.add(col_name.lower())

    return {"tables": tables, "columns_by_table": columns_by_table}


def referenced_identifiers(sql: str | None) -> IdentifierRefs:
    """
    Return the set of referenced tables and columns in a SQL query.

    This is a best-effort extractor that relies on sqlglot parsing.
    On parse failure it returns empty sets (callers can combine this with
    a separate parse-success metric).
    """
    tables: Set[str] = set()
    columns: Set[str] = set()

    expression = _try_parse_sql(sql)
    if expression is None:
        return {"tables": tables, "columns": columns}

    for table_expr in expression.find_all(exp.Table):
        name = (table_expr.name or "").strip()
        if name:
            tables.add(name.lower())

    for col_expr in expression.find_all(exp.Column):
        # Column.this is usually an Identifier.
        identifier = col_expr.find(exp.Identifier)
        col_name = (identifier.name if identifier is not None else "").strip()
        if col_name:
            columns.add(col_name.lower())

    return {"tables": tables, "columns": columns}


def schema_adherence(sql: str | None, context: str | None) -> bool:
    """
    Check whether a query only references tables/columns present in the schema.

    The check is intentionally simple and name-based:

    - All referenced table names must appear among the CREATE TABLE statements.
    - All referenced column names must appear in at least one table's column set.

    If either the schema or the SQL cannot be parsed, this function returns False.
    """
    if sql is None or context is None:
        return False

    schema_info = parse_create_table_context(context)
    if not schema_info["tables"] and not schema_info["columns_by_table"]:
        return False

    refs = referenced_identifiers(sql)
    if not refs["tables"] and not refs["columns"]:
        # If nothing could be extracted from the SQL, treat as non-adherent.
        return False

    allowed_tables = {t.lower() for t in schema_info["tables"]}
    allowed_columns: Set[str] = set()
    for cols in schema_info["columns_by_table"].values():
        allowed_columns.update(c.lower() for c in cols)

    # If we have no allowed tables but references exist, this is non-adherent.
    if not allowed_tables and refs["tables"]:
        return False

    invalid_tables = {t for t in refs["tables"] if t not in allowed_tables}
    invalid_columns = {c for c in refs["columns"] if c not in allowed_columns}

    return not invalid_tables and not invalid_columns
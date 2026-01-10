from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError


def _iter_create_statements(statements: Iterable[exp.Expression]) -> Iterable[exp.Create]:
    """
    Yield all CREATE TABLE expressions from a sequence of parsed statements.
    """
    for stmt in statements:
        if isinstance(stmt, exp.Create):
            yield stmt
        for create in stmt.find_all(exp.Create):
            # Avoid yielding the same node twice if stmt is itself a Create.
            if create is not stmt:
                yield create


def parse_create_table_context(context: str) -> Dict[str, Any]:
    """
    Parse a CREATE TABLE context string into a simple schema summary.

    Parameters
    ----------
    context : str
        One or more CREATE TABLE statements, typically the `context` field from
        the b-mc2/sql-create-context dataset.

    Returns
    -------
    dict
        {
            "tables": set[str],
            "columns_by_table": dict[str, set[str]],
        }

        All identifiers are lowercased for robust matching.
    """
    tables: Set[str] = set()
    columns_by_table: Dict[str, Set[str]] = {}

    if not context or not context.strip():
        return {"tables": tables, "columns_by_table": columns_by_table}

    try:
        statements: List[exp.Expression] = list(sqlglot.parse(context))
    except ParseError:
        # If parsing fails completely, return empty schema; callers can decide
        # how to handle this.
        return {"tables": tables, "columns_by_table": columns_by_table}

    for create in _iter_create_statements(statements):
        table_expr = create.this
        table_name: Optional[str] = None

        if isinstance(table_expr, exp.Table):
            table_name = table_expr.name
        elif hasattr(table_expr, "name"):
            table_name = getattr(table_expr, "name", None)

        if not table_name:
            continue

        table_norm = str(table_name).lower()
        tables.add(table_norm)
        cols = columns_by_table.setdefault(table_norm, set())

        # Column definitions are represented as ColumnDef expressions under the
        # CREATE expression.
        for col_def in create.find_all(exp.ColumnDef):
            col_expr = col_def.this
            col_name: Optional[str] = None

            if isinstance(col_expr, exp.Column):
                col_name = col_expr.name
            elif isinstance(col_expr, exp.Identifier):
                col_name = col_expr.name
            elif hasattr(col_expr, "name"):
                col_name = getattr(col_expr, "name", None)

            if col_name:
                cols.add(str(col_name).lower())

    return {"tables": tables, "columns_by_table": columns_by_table}


def referenced_identifiers(sql: str) -> Dict[str, Set[str]]:
    """
    Extract referenced table and column identifiers from a SQL query.

    This function performs a best-effort analysis using sqlglot. Identifiers are
    returned in lowercase to make downstream comparisons case-insensitive.

    Parameters
    ----------
    sql : str
        SQL query text.

    Returns
    -------
    dict
        {
            "tables": set[str],
            "columns": set[str],
        }
    """
    tables: Set[str] = set()
    columns: Set[str] = set()

    if not sql or not sql.strip():
        return {"tables": tables, "columns": columns}

    try:
        expr = sqlglot.parse_one(sql)
    except ParseError:
        return {"tables": tables, "columns": columns}

    for table_expr in expr.find_all(exp.Table):
        name = table_expr.name
        if name:
            tables.add(str(name).lower())

    for col_expr in expr.find_all(exp.Column):
        name = col_expr.name
        if name:
            columns.add(str(name).lower())

    return {"tables": tables, "columns": columns}


def schema_adherence(sql: str, context: str) -> bool:
    """
    Check whether a SQL query references only tables/columns present in `context`.

    The context is expected to be one or more CREATE TABLE statements. The check
    is intentionally conservative:

    - If parsing the context fails, returns False only if the SQL parses and
      clearly references something not present in the (empty) schema.
    - If parsing the SQL fails, returns False.
    - Unqualified columns are considered valid if they match any column in the
      schema (across tables).

    Parameters
    ----------
    sql : str
        Predicted SQL query.
    context : str
        CREATE TABLE context string from the dataset.

    Returns
    -------
    bool
        True if all referenced tables and columns appear in the schema, False
        otherwise.
    """
    schema = parse_create_table_context(context)
    schema_tables: Set[str] = {t.lower() for t in schema.get("tables", set())}
    schema_columns_by_table: Dict[str, Set[str]] = {
        t.lower(): {c.lower() for c in cols}
        for t, cols in schema.get("columns_by_table", {}).items()
    }

    # If we have no schema information at all, treat queries as adherent.
    if not schema_tables and not schema_columns_by_table:
        return True

    try:
        expr = sqlglot.parse_one(sql)
    except ParseError:
        return False

    # Map alias -> underlying table, and collect all referenced tables.
    alias_to_table: Dict[str, str] = {}
    referenced_tables: Set[str] = set()

    for table_expr in expr.find_all(exp.Table):
        name = table_expr.name
        if name:
            table_name = str(name).lower()
            referenced_tables.add(table_name)
        else:
            table_name = ""

        alias = table_expr.alias
        if alias:
            alias_to_table[str(alias).lower()] = table_name

    # If any referenced table is not part of the schema, fail early.
    for t in referenced_tables:
        if t and t not in schema_tables:
            return False

    # Build a global set of all known columns for handling unqualified refs.
    all_schema_columns: Set[str] = set()
    for cols in schema_columns_by_table.values():
        all_schema_columns.update(cols)

    for col_expr in expr.find_all(exp.Column):
        col_name = col_expr.name
        if not col_name:
            continue

        col_norm = str(col_name).lower()

        table_ref = col_expr.table
        table_norm: Optional[str] = None
        if table_ref:
            candidate = str(table_ref).lower()
            # Resolve aliases to base tables when possible.
            table_norm = alias_to_table.get(candidate, candidate)

        if table_norm:
            # If the table itself isn't known, fail.
            if table_norm not in schema_columns_by_table:
                return False
            if col_norm not in schema_columns_by_table[table_norm]:
                return False
        else:
            # Unqualified column: accept if it exists in any table.
            if col_norm not in all_schema_columns:
                return False

    return True
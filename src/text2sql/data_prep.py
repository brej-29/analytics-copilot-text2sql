import re
from typing import Dict

INSTRUCTION_TEXT = (
    "Write a SQL query that answers the user's question using ONLY the tables "
    "and columns provided in the schema."
)
SOURCE_NAME = "b-mc2/sql-create-context"


def normalize_sql(sql: str) -> str:
    """
    Lightly normalize a SQL string.

    Normalization is intentionally conservative and focuses on:
    - Stripping leading/trailing whitespace.
    - Collapsing runs of whitespace (spaces, tabs, newlines) into single spaces.

    Parameters
    ----------
    sql : str
        Raw SQL string.

    Returns
    -------
    str
        Normalized SQL string.
    """
    if sql is None:
        return ""
    stripped = sql.strip()
    # Collapse any whitespace (space, tab, newline) into a single space.
    return re.sub(r"\s+", " ", stripped)


def build_input_text(context: str, question: str) -> str:
    """
    Build the instruction input text from schema context and user question.

    The current format is:

    ### Schema:
    <CREATE TABLE ...>

    ### Question:
    <natural language question>
    """
    context = (context or "").strip()
    question = (question or "").strip()

    return f"### Schema:\n{context}\n\n### Question:\n{question}"


def format_record(question: str, context: str, answer: str) -> Dict[str, object]:
    """
    Format a raw dataset record into an Alpaca-style instruction example.

    This function only constructs the logical content fields. The caller
    (e.g., the dataset building script) is responsible for adding:
    - `id`
    - `meta` (row index, split, build info, etc.)

    Parameters
    ----------
    question : str
        Natural language question.
    context : str
        Schema context, typically a CREATE TABLE statement.
    answer : str
        Target SQL query.

    Returns
    -------
    dict
        A mapping with keys:
        - instruction
        - input
        - output
        - source
    """
    input_text = build_input_text(context=context, question=question)
    output_sql = normalize_sql(answer)

    return {
        "instruction": INSTRUCTION_TEXT,
        "input": input_text,
        "output": output_sql,
        "source": SOURCE_NAME,
    }
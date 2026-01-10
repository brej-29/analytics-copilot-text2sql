import re
from typing import Final


PROMPT_TEMPLATE: Final[str] = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def build_prompt(instruction: str, input: str) -> str:
    """
    Build a prompt for supervised fine-tuning from an instruction and input.

    The current format is:

    ### Instruction:
    <instruction>

    ### Input:
    <schema + question or other input>

    ### Response:
    """
    instruction = (instruction or "").strip()
    input_text = (input or "").strip()

    return PROMPT_TEMPLATE.format(instruction=instruction, input=input_text)


def ensure_sql_only(output: str) -> str:
    """
    Ensure that the output text is a clean SQL string.

    This function is intentionally conservative:
    - Strips leading/trailing whitespace.
    - Removes surrounding ```sql ... ``` fences if present.
    - Collapses runs of whitespace (spaces, newlines, tabs) into single spaces.
    """
    if output is None:
        return ""

    text = output.strip()

    # Remove Markdown-style fenced code blocks such as ```sql ... ```
    if text.startswith("```"):
        # Drop leading ```sql (or ```), case-insensitive for 'sql'
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        # Drop trailing ```
        text = re.sub(r"\s*```$", "", text)

    # Collapse all internal whitespace runs into a single space
    text = re.sub(r"\s+", " ", text)

    return text
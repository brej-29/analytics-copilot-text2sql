#!/usr/bin/env python
"""
Local smoke test for the OpenAI fallback path used by the Streamlit app.

This script:
- Loads OpenAI fallback configuration from Streamlit-style settings.
- Prints which config keys are present (masking the API key).
- Runs a single real OpenAI Responses API call with a fixed schema + question.
- Prints the raw output returned by the fallback helper.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

"""from openai import OpenAI  # type: ignore[import]"""


def _ensure_root_on_path() -> None:
    """Ensure that the project root is available on sys.path for imports."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_root_on_path()

from app import streamlit_app  # noqa: E402  # isort: skip  # pylint: disable=wrong-import-position


def _mask_api_key(value: str) -> str:
    """Return a masked representation of an API key for logging/debugging."""
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return value[0] + "..." + value[-1]
    return value[:4] + "..." + value[-4:]


def _load_openai_config() -> Tuple[str, str]:
    """Load OpenAI fallback API key and model name using the app helper."""
    api_key, model_name = streamlit_app._get_openai_settings()  # type: ignore[attr-defined]
    return api_key, model_name


def main() -> None:
    api_key, model_name = _load_openai_config()

    print("OpenAI fallback configuration:")
    print(f"  OPENAI_API_KEY: {_mask_api_key(api_key)}")
    print(f"  OPENAI_FALLBACK_MODEL: {model_name!r}")

    if not api_key:
        print("ERROR: OPENAI_API_KEY is not configured. Aborting smoke test.")
        return

    """client = OpenAI(api_key=api_key)"""

    schema = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    city VARCHAR(50)
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date DATE,
    total_amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
""".strip()

    question = """
What are the full names of the customers who have placed an order with a total amount
greater than 100, and how many orders did each of them place?
""".strip()

    system_prompt, user_prompt = streamlit_app._build_prompt(schema=schema, question=question)  # type: ignore[attr-defined]

    # Use the same fallback helper as the Streamlit app so behavior is consistent.
    raw_sql = streamlit_app._call_openai_fallback(  # type: ignore[attr-defined]
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    print("\nOpenAI fallback raw output:")
    print(raw_sql)


if __name__ == "__main__":
    main()
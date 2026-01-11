from __future__ import annotations

import logging
import os
from typing import Any, Dict

from huggingface_hub import InferenceClient  # type: ignore[import]


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def build_prompt(schema: str, question: str) -> str:
    """
    Build a simple text-to-SQL prompt consistent with the main project.

    The schema and question are formatted as:

        ### Schema:
        <DDL>

        ### Question:
        <natural language question>
    """
    system_prompt = (
        "You are a careful text-to-SQL assistant. "
        "Given a database schema and a question, you respond with a single SQL "
        "query that answers the question. "
        "Return ONLY the SQL query, without explanation or commentary."
    )

    user_prompt = f"""### Schema:
{schema.strip()}

### Question:
{question.strip()}

Return only the SQL query."""

    return f"{system_prompt}\n\n{user_prompt}"


def _extract_generated_text(response: Any) -> str:
    """
    Best-effort extraction of generated text from InferenceClient.text_generation.

    Handles common shapes:
    - str
    - {"generated_text": "..."}
    - [{"generated_text": "..."}]
    """
    if isinstance(response, str):
        return response

    if isinstance(response, dict) and "generated_text" in response:
        return str(response["generated_text"])

    if (
        isinstance(response, list)
        and response
        and isinstance(response[0], dict)
        and "generated_text" in response[0]
    ):
        return str(response[0]["generated_text"])

    return str(response)


def main() -> int:
    """
    Smoke test for a dedicated HF Inference Endpoint with LoRA adapters.

    Reads configuration from environment variables:

        HF_TOKEN        – required
        HF_ENDPOINT_URL – required
        HF_ADAPTER_ID   – required

    Sends a single text_generation request with adapter_id and prints the raw
    response to stdout.
    """
    configure_logging()

    hf_token = os.getenv("HF_TOKEN", "").strip()
    endpoint_url = os.getenv("HF_ENDPOINT_URL", "").strip() or os.getenv(
        "HF_INFERENCE_BASE_URL",
        "",
    ).strip()
    adapter_id = os.getenv("HF_ADAPTER_ID", "").strip()

    if not hf_token:
        logger.error("HF_TOKEN is not set in the environment.")
        return 1
    if not endpoint_url:
        logger.error("HF_ENDPOINT_URL (or HF_INFERENCE_BASE_URL) is not set.")
        return 1
    if not adapter_id:
        logger.error(
            "HF_ADAPTER_ID is not set. This smoke test is intended for "
            "adapter-based endpoints and always sends adapter_id."
        )
        return 1

    logger.info("Using HF endpoint: %s", endpoint_url)
    logger.info("Using adapter_id: %s", adapter_id)

    client = InferenceClient(base_url=endpoint_url, api_key=hf_token, timeout=60)

    # Tiny toy schema + question for a quick smoke test.
    schema = """CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER,
  amount NUMERIC,
  created_at TIMESTAMP
);"""

    question = "Total order amount per customer for the last 7 days."

    prompt = build_prompt(schema=schema, question=question)

    generation_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "max_new_tokens": 128,
        "temperature": 0.0,
        "adapter_id": adapter_id,
    }

    try:
        logger.info("Calling text_generation on the HF endpoint...")
        response = client.text_generation(**generation_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error while calling the HF endpoint.", exc_info=True)
        print(f"ERROR: {exc}")
        return 1

    text = _extract_generated_text(response)
    print("=== Raw text_generation response ===")
    print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
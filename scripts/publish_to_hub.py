"""
Utility script to publish QLoRA adapter artifacts to Hugging Face Hub.

This script is designed to be idempotent and friendly to both local
development environments and CI:

- Validates that a Hugging Face token is available.
- Ensures the target repo exists (creates it if needed).
- Ensures a README.md model card is present under the adapter directory.
- Uploads the entire adapter folder (LoRA/QLoRA artifacts) to the Hub.

Typical usage:

    python scripts/publish_to_hub.py \
      --repo_id your-username/analytics-copilot-text2sql-mistral7b-qlora \
      --adapter_dir outputs/adapters

The script expects that you have already authenticated with Hugging Face, e.g.:

    huggingface-cli login

or by setting an HF_TOKEN environment variable.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import HfApi  # type: ignore[import]
from huggingface_hub.utils import HfHubHTTPError  # type: ignore[import]


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure basic logging for the publish script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the publish-to-hub workflow."""
    parser = argparse.ArgumentParser(
        description=(
            "Publish QLoRA adapter artifacts from a local directory to a "
            "Hugging Face Hub model repository."
        )
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help=(
            "Target Hugging Face Hub repository id "
            "(e.g. 'username/analytics-copilot-text2sql-mistral7b-qlora')."
        ),
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="outputs/adapters",
        help=(
            "Path to the local directory containing QLoRA adapter artifacts. "
            "This directory will be uploaded as the root of the HF model repo."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the target repository as private instead of public.",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Add QLoRA adapter artifacts",
        help="Commit message to use for the upload.",
    )
    parser.add_argument(
        "--include_metrics",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON metrics file (e.g. reports/eval_internal.json "
            "or reports/eval_spider.json). If provided, a summary will be "
            "embedded into the generated README.md model card."
        ),
    )

    return parser.parse_args(argv)


def _require_hf_token(api: HfApi) -> None:
    """
    Ensure that a Hugging Face token is available and valid.

    We call `whoami` as a lightweight check that authentication is correctly
    configured. If this fails, we raise a RuntimeError with a clear message so
    that the caller can surface it and exit with a non-zero status.
    """
    try:
        api.whoami()
    except HfHubHTTPError as exc:
        msg = (
            "Hugging Face authentication failed. "
            "Please run `huggingface-cli login` or set the HF_TOKEN environment "
            "variable before running scripts/publish_to_hub.py."
        )
        logger.error(msg)
        raise RuntimeError(msg) from exc
    except Exception as exc:  # noqa: BLE001
        msg = (
            "Unexpected error while checking Hugging Face authentication. "
            "Ensure you have network connectivity and a valid HF token."
        )
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg) from exc


def _load_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load metrics from a JSON file, handling both raw metric dicts and
    report-style payloads that wrap metrics under a 'metrics' key.
    """
    if not metrics_path.is_file():
        logger.warning(
            "Metrics file '%s' does not exist; skipping metrics section.",
            metrics_path,
        )
        return None

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to parse metrics JSON file '%s'. Error: %s. "
            "Continuing without embedding metrics.",
            metrics_path,
            exc,
        )
        return None

    if isinstance(payload, dict) and "metrics" in payload and isinstance(
        payload["metrics"],
        dict,
    ):
        return payload["metrics"]

    if isinstance(payload, dict):
        return payload

    logger.warning(
        "Metrics file '%s' did not contain a JSON object; skipping metrics section.",
        metrics_path,
    )
    return None


def _format_metrics_section(metrics: Dict[str, Any]) -> str:
    """
    Render a Markdown metrics section from a metrics dict.

    The expected schema for internal / Spider evaluation reports is:
        {
          "n_examples": ...,
          "exact_match": {"count": ..., "rate": ...},
          "no_values_em": {...},
          "parse_success": {...},
          "schema_adherence": {...},
        }
    """
    if not metrics:
        return "_No metrics available._\n"

    lines: list[str] = []

    n_examples = metrics.get("n_examples")
    if n_examples is not None:
        lines.append(f"- **n_examples:** {n_examples}")

    def _fmt(entry_name: str, label: str) -> None:
        entry = metrics.get(entry_name)
        if not isinstance(entry, dict):
            return
        count = entry.get("count")
        rate = entry.get("rate")
        if count is None or rate is None:
            return
        lines.append(f"- **{label}:** {count} examples ({rate:.3f})")

    _fmt("exact_match", "Exact Match (normalized SQL)")
    _fmt("no_values_em", "No-values Exact Match")
    _fmt("parse_success", "SQL parse success rate")
    _fmt("schema_adherence", "Schema adherence rate")

    if not lines:
        return (
            "_Metrics JSON did not match the expected schema; "
            "see raw file for details._\n"
        )

    return "\n".join(lines) + "\n"


def _ensure_readme(
    adapter_dir: Path,
    repo_id: str,
    metrics_path: Optional[Path],
) -> Path:
    """
    Ensure that a README.md model card exists in `adapter_dir`.

    If a README already exists, it will be replaced with a fresh, structured
    model card tailored to this project. This keeps the Hub repo consistent
    even if training scripts generate ad-hoc notes.
    """
    readme_path = adapter_dir / "README.md"
    metrics: Optional[Dict[str, Any]] = None

    if metrics_path is not None:
        metrics = _load_metrics(metrics_path)

    metrics_section = (
        _format_metrics_section(metrics)
        if metrics
        else "_No metrics provided._\n"
    )

    # Use a triple-single-quoted f-string so we can include triple-double-quoted
    # code samples without extra escaping.
    content = f'''# Analytics Copilot Text-to-SQL – QLoRA Adapter

This repository contains a **QLoRA adapter** for the
`mistralai/Mistral-7B-Instruct-v0.1` model, fine-tuned for **text-to-SQL**
generation on the **b-mc2/sql-create-context** dataset.

- **Base model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Task:** Natural-language question → SQL query (schema-aware)
- **Adapters:** LoRA / QLoRA weights stored in this repository
- **Training dataset:** `b-mc2/sql-create-context` (subset used for training/validation)
- **Repo id:** `{repo_id}`

> **Note:** This Hub repo contains **adapters only** (no full model merge).
> To run the model locally you must load the base model and apply these adapters.

---

## How to Use – Remote Inference (Hugging Face InferenceClient)

If you host a merged model or an inference endpoint on Hugging Face, you can
query it remotely using `huggingface_hub.InferenceClient`. This is the
recommended path for lightweight clients (e.g. Streamlit on Community Cloud)
that should not load the model locally.

```python
from huggingface_hub import InferenceClient

# Option 1: Dedicated inference endpoint / TGI base URL
client = InferenceClient(
    base_url="https://your-endpoint-url",  # e.g. Inference Endpoint
    api_key="hf_your_token",
)

# Option 2: Serverless text-generation-inference (for smaller / supported models)
# client = InferenceClient(
#     model="your-username/your-merged-text2sql-model",
#     api_key="hf_your_token",
#     provider="auto",  # or 'tgi', 'hf-inference', etc., depending on your setup
# )

system_prompt = "You are a careful text-to-SQL assistant. Return ONLY SQL."

schema = """CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER,
  amount NUMERIC,
  created_at TIMESTAMP
);"""

question = "Total order amount per customer for the last 7 days."

user_content = f"""### Schema:
{{schema}}

### Question:
{{question}}

Return only the SQL query."""

response = client.chat_completion(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ],
    max_tokens=256,
    temperature=0.0,
)

sql = response.choices[0].message["content"].strip()
print(sql)
```

---

## How to Use – Local Inference with transformers + peft

To apply this QLoRA adapter on top of the base Mistral-7B model locally:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
ADAPTER_REPO = "{repo_id}"  # this Hub repo

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Optional: 4-bit loading for memory efficiency
try:
    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if device == "cuda" else torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
except Exception:
    # Fallback: standard full-precision load (may require more GPU memory).
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
model.eval()

def generate_sql(schema: str, question: str) -> str:
    prompt = f"""### Schema:
{{schema}}

### Question:
{{question}}

Return only the SQL query."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    generated = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()
```

---

## Safety and Usage Notes

This adapter is intended to generate **SQL queries only**. It does **not**
perform any additional safety filtering or content moderation.

- Always run generated SQL against **read-only** or **non-production** databases.
- Apply standard safeguards (row limits, timeouts, query cost guards).
- Consider wrapping the model behind an application layer that enforces:
  - Allowed schemas / tables.
  - Query whitelisting or pattern checks.
  - Output length limits.

The training data is focused on well-formed analytical queries, but the model
may still hallucinate tables/columns or produce invalid SQL, especially when
used on schemas very different from the training distribution.

---

## Metrics

The following metrics were extracted from recent evaluation runs
(e.g. internal validation on `b-mc2/sql-create-context` and/or Spider dev):

{metrics_section}
'''

    readme_path.write_text(content, encoding="utf-8")
    logger.info("Model card written to %s", readme_path)
    return readme_path


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for publishing adapter artifacts to Hugging Face Hub."""
    configure_logging()
    args = parse_args(argv)

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_dir():
        logger.error(
            "Adapter directory '%s' does not exist or is not a directory. "
            "Ensure you have run training and that the path is correct.",
            adapter_dir,
        )
        return 1

    api = HfApi()

    try:
        _require_hf_token(api)
    except RuntimeError:
        # Error already logged with details.
        return 1

    repo_id: str = args.repo_id

    logger.info(
        "Ensuring Hugging Face Hub repo '%s' exists (private=%s).",
        repo_id,
        args.private,
    )
    try:
        api.create_repo(
            repo_id=repo_id,
            private=args.private,
            exist_ok=True,
            repo_type="model",
        )
    except HfHubHTTPError:
        logger.error(
            "Failed to create or access repository '%s' on Hugging Face Hub.",
            repo_id,
            exc_info=True,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected error while creating or accessing repository '%s': %s",
            repo_id,
            exc,
            exc_info=True,
        )
        return 1

    metrics_path: Optional[Path] = None
    if args.include_metrics is not None:
        metrics_path = Path(args.include_metrics)

    _ensure_readme(adapter_dir=adapter_dir, repo_id=repo_id, metrics_path=metrics_path)

    logger.info(
        "Uploading contents of '%s' to Hugging Face Hub repo '%s' "
        "with commit message: %s",
        adapter_dir,
        repo_id,
        args.commit_message,
    )

    try:
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to upload adapter directory '%s' to '%s': %s",
            adapter_dir,
            repo_id,
            exc,
            exc_info=True,
        )
        return 1

    logger.info("Successfully uploaded adapter artifacts to '%s'.", repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
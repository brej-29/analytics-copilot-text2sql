"""
Utility script to publish QLoRA/LoRA adapter artifacts to Hugging Face Hub.

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
            "Publish QLoRA/LoRA adapter artifacts from a local directory to a "
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
            "Path to the local directory containing QLoRA/LoRA adapter artifacts. "
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
    parser.add_argument(
        "--skip_readme",
        action="store_true",
        help=(
            "Skip auto-generating README.md. Existing README.md (if any) is left "
            "untouched and only adapter files are uploaded."
        ),
    )
    parser.add_argument(
        "--strict_readme",
        action="store_true",
        help=(
            "If set, fail the publish if README generation fails. "
            "By default, README generation errors are logged and the upload "
            "continues."
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


def _validate_adapter_dir(adapter_dir: Path) -> Dict[str, Any]:
    """
    Validate that the adapter directory contains the expected files.

    Required:
    - adapter_config.json
    - adapter_model.safetensors OR adapter_model.bin

    Returns the parsed adapter_config.json payload on success.
    """
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"Adapter directory '{adapter_dir}' is missing 'adapter_config.json'. "
            "This file is required to describe the adapter configuration."
        )

    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if not safetensors_path.is_file() and not bin_path.is_file():
        raise RuntimeError(
            f"Adapter directory '{adapter_dir}' is missing adapter weights. "
            "Expected either 'adapter_model.safetensors' or 'adapter_model.bin'."
        )

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to parse adapter config JSON at '{config_path}': {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise RuntimeError(
            f"Adapter config at '{config_path}' must be a JSON object."
        )

    return config


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
    base_model: str,
    metrics_path: Optional[Path],
) -> Path:
    """
    Ensure that a README.md model card exists in `adapter_dir`.

    If README.md already exists, it is left unchanged. Otherwise, a minimal,
    non-LLM README is created that documents the base model, task, evaluation
    scripts, and a recommended Inference Endpoint + Multi-LoRA deployment
    pattern.
    """
    readme_path = adapter_dir / "README.md"
    if readme_path.is_file():
        logger.info("README.md already exists at %s; leaving it unchanged.", readme_path)
        return readme_path

    metrics: Optional[Dict[str, Any]] = None
    if metrics_path is not None:
        metrics = _load_metrics(metrics_path)

    metrics_section = (
        _format_metrics_section(metrics) if metrics else "_No metrics provided._\n"
    )

    lines: list[str] = []

    # Header and basic info.
    lines.append("# Analytics Copilot Text-to-SQL – QLoRA Adapter")
    lines.append("")
    lines.append(
        "This repository contains a LoRA/QLoRA adapter for a Text-to-SQL model. "
        "It is intended to be applied on top of a base language model hosted on "
        "Hugging Face Hub or in an Inference Endpoint."
    )
    lines.append("")
    lines.append(f"- **Base model:** `{base_model}`")
    lines.append("- **Task:** Text-to-SQL (schema + question → SQL)")
    lines.append(f"- **Adapter repo id:** `{repo_id}`")
    lines.append("")
    lines.append(
        "> Note: This Hub repo typically contains adapters only (no full model "
        "merge). To run the model locally you must load the base model and "
        "apply these adapters."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Evaluation instructions.
    lines.append("## Evaluation")
    lines.append("")
    lines.append(
        "You can evaluate the adapter using the internal and external "
        "evaluation scripts in this project."
    )
    lines.append("")
    lines.append("### Internal evaluation (sql-create-context)")
    lines.append("")
    lines.append("Example command (assuming the repo layout from this project):")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/evaluate_internal.py \\")
    lines.append("  --val_path data/processed/val.jsonl \\")
    lines.append(f"  --base_model {base_model} \\")
    lines.append("  --adapter_dir /path/to/local/adapters \\")
    lines.append("  --device auto \\")
    lines.append("  --max_examples 200 \\")
    lines.append("  --out_dir reports/")
    lines.append("```")
    lines.append("")
    lines.append("### External evaluation (Spider dev)")
    lines.append("")
    lines.append("Example command:")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/evaluate_spider_external.py \\")
    lines.append(f"  --base_model {base_model} \\")
    lines.append("  --adapter_dir /path/to/local/adapters \\")
    lines.append("  --device auto \\")
    lines.append("  --spider_source xlangai/spider \\")
    lines.append("  --schema_source richardr1126/spider-schema \\")
    lines.append("  --spider_split validation \\")
    lines.append("  --max_examples 200 \\")
    lines.append("  --out_dir reports/")
    lines.append("```")
    lines.append("")
    lines.append(
        "See `docs/evaluation.md` in the project for a detailed description of "
        "the metrics and evaluation setup."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Deployment instructions (Inference Endpoint + Multi-LoRA).
    lines.append("## Deployment with Hugging Face Inference Endpoints (Multi-LoRA)")
    lines.append("")
    lines.append(
        "A recommended way to serve this adapter is to deploy the base model once "
        "using a Text Generation Inference (TGI) Inference Endpoint, and attach "
        "this adapter via the `LORA_ADAPTERS` environment variable."
    )
    lines.append("")
    lines.append("High-level steps:")
    lines.append("")
    lines.append("1. Create a new Inference Endpoint based on the base model, e.g.:")
    lines.append(f"   - Base model: `{base_model}`")
    lines.append("   - Hardware: choose a GPU instance suitable for Mistral-7B.")
    lines.append("2. In the endpoint configuration, set the environment variable:")
    lines.append("")
    lines.append("   ```bash")
    lines.append(
        "   LORA_ADAPTERS='["
        '{"id": "text2sql-qlora", "source": "' + repo_id + '"}'
        "]'"
    )
    lines.append("   ```")
    lines.append("")
    lines.append(
        "   This tells TGI to load the adapter from this Hub repo under the "
        "logical adapter id `text2sql-qlora`."
    )
    lines.append("3. Deploy the endpoint.")
    lines.append("")
    lines.append(
        "At inference time, you can select the adapter by passing an `adapter_id` "
        "parameter in your request. For example, using the raw HTTP API:"
    )
    lines.append("")
    lines.append("```json")
    lines.append("{")
    lines.append('  "inputs": "### Schema:\\n<DDL here>\\n\\n### Question:\\n<NL question>",')
    lines.append('  "parameters": {')
    lines.append('    "adapter_id": "text2sql-qlora",')
    lines.append('    "max_new_tokens": 256,')
    lines.append('    "temperature": 0.0')
    lines.append("  }")
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append(
        "If you are using `huggingface_hub.InferenceClient`, you can pass "
        "`adapter_id` via the `extra_headers` or specific provider parameters "
        "depending on the TGI configuration."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Metrics section.
    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "If you have run internal or Spider evaluations and passed "
        "`--include_metrics` to `scripts/publish_to_hub.py`, a summary of those "
        "metrics is included below."
    )
    lines.append("")
    if metrics:
        lines.extend(metrics_section.rstrip("\n").splitlines())
    else:
        lines.append("_No metrics provided._")

    content = "\n".join(lines) + "\n"

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

    # Validate adapter contents before hitting the Hub APIs.
    try:
        adapter_config = _validate_adapter_dir(adapter_dir)
    except (RuntimeError, ValueError) as exc:
        logger.error("Adapter validation failed: %s", exc)
        return 1

    base_model = adapter_config.get("base_model_name_or_path", "<unknown-base-model>")

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

    if args.skip_readme:
        logger.info(
            "Skipping README.md generation because --skip_readme was provided."
        )
    else:
        try:
            _ensure_readme(
                adapter_dir=adapter_dir,
                repo_id=repo_id,
                base_model=base_model,
                metrics_path=metrics_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to generate README.md for adapter repo '%s': %s",
                repo_id,
                exc,
                exc_info=True,
            )
            if args.strict_readme:
                logger.error(
                    "Aborting publish because --strict_readme was set and "
                    "README generation failed."
                )
                return 1
            logger.info(
                "Continuing without auto-generated README because "
                "--strict_readme was not set."
            )

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
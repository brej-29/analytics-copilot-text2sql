import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure the src/ directory is on sys.path so that `text2sql` can be imported
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql.training.formatting import build_prompt  # noqa: E402
from text2sql.eval.metrics import (  # noqa: E402
    MetricFlags,
    aggregate_metrics,
)
from text2sql.eval.normalize import normalize_sql  # noqa: E402

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for internal evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned model on the internal validation set."
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/processed/val.jsonl",
        help="Path to the Alpaca-style validation JSONL file.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Base model name or path for inference.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help=(
            "Path to the LoRA adapter directory produced by training. "
            "Required unless --mock is used."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto (default), cuda, or cpu.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=200,
        help="Maximum number of validation examples to evaluate.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="reports",
        help="Output directory for evaluation reports.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per example.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Run in mock mode: skip model loading and use gold SQL as "
            "predictions to validate the metric pipeline."
        ),
    )
    return parser.parse_args(argv)


def _load_val_records(path: Path, max_examples: int) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation JSONL file not found: {path}")

    logger.info("Loading validation records from %s", path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if 0 < max_examples <= len(records):
                break

    if not records:
        raise RuntimeError(f"No records found in validation file: {path}")

    logger.info("Loaded %d validation records.", len(records))
    return records


def _extract_schema_and_question(input_text: str) -> Tuple[str, str]:
    """
    Extract the schema context and question from the Alpaca-style input field.

    Expected format from text2sql.data_prep.build_input_text:

        ### Schema:
        <CREATE TABLE ...>

        ### Question:
        <natural language question>
    """
    text = (input_text or "").strip()
    schema_marker = "### Schema:"
    question_marker = "### Question:"

    schema = ""
    question = ""

    if schema_marker in text and question_marker in text:
        _, after_schema = text.split(schema_marker, 1)
        schema_part, _, question_part = after_schema.partition(question_marker)
        schema = schema_part.strip()
        question = question_part.strip()
    else:
        # Fallback: treat the whole input as the question.
        question = text

    return schema, question


def _generate_predictions(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], List[str], List[Dict[str, Any]]]:
    """
    Generate predictions (or mock predictions) for the given records.

    Returns
    -------
    preds, golds, contexts, examples
    """
    preds: List[str] = []
    golds: List[str] = []
    contexts: List[str] = []
    example_rows: List[Dict[str, Any]] = []

    if not args.mock:
        # Import inside the function so that tests using --mock do not require
        # transformers/torch/peft to be importable.
        from text2sql.infer import load_model_for_inference, generate_sql  # noqa: WPS433

        if not args.adapter_dir:
            raise ValueError(
                "--adapter_dir is required for real evaluation unless --mock is used."
            )

        load_model_for_inference(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            device=args.device,
        )

        def _predict(prompt: str) -> str:
            return generate_sql(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

    else:
        logger.info("Running in --mock mode; using gold SQL as predictions.")

        def _predict(prompt: str) -> str:  # type: ignore[no-redef]
            return ""  # Will be overridden per example with gold SQL.

    for idx, row in enumerate(records):
        instruction = row.get("instruction", "")
        input_text = row.get("input", "")
        gold_sql = row.get("output", "")

        schema, question = _extract_schema_and_question(input_text)
        prompt = build_prompt(instruction=instruction, input=input_text)

        if args.mock:
            pred_sql = gold_sql
        else:
            try:
                pred_sql = _predict(prompt)
            except Exception:  # noqa: BLE001
                logger.exception("Prediction failed for example %d; using empty string.", idx)
                pred_sql = ""

        preds.append(pred_sql)
        golds.append(gold_sql)
        contexts.append(schema)

        if len(example_rows) < 10:
            example_rows.append(
                {
                    "id": row.get("id", f"val-{idx}"),
                    "question": question,
                    "schema": schema,
                    "gold_sql": gold_sql,
                    "pred_sql": pred_sql,
                    "exact_match": pred_sql
                    and gold_sql
                    and normalize_sql(pred_sql) == normalize_sql(gold_sql),
                }
            )

    return preds, golds, contexts, example_rows


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    cfg = payload["config"]
    metrics = payload["metrics"]
    examples = payload.get("examples", [])

    lines: List[str] = []
    lines.append("# Internal Evaluation – b-mc2/sql-create-context val\n")
    lines.append("## Configuration\n")
    lines.append(f"- Val path: `{cfg['val_path']}`")
    lines.append(f"- Base model: `{cfg['base_model']}`")
    lines.append(f"- Adapter dir: `{cfg['adapter_dir']}`")
    lines.append(f"- Device: `{cfg['device']}`")
    lines.append(f"- Max examples: {cfg['max_examples']}")
    lines.append(f"- Mock mode: {cfg['mock']}")
    lines.append(f"- Max new tokens: {cfg['max_new_tokens']}")
    lines.append(f"- Temperature: {cfg['temperature']}")
    lines.append(f"- Top-p: {cfg['top_p']}\n")

    lines.append("## Metrics\n")
    lines.append(f"- Number of examples: {int(metrics['num_examples'])}")
    lines.append(f"- Exact Match (normalized): {metrics['exact_match']:.4f}")
    lines.append(
        f"- No-values Exact Match (normalized, literals abstracted): "
        f"{metrics['no_values_exact_match']:.4f}"
    )
    lines.append(f"- SQL parse success rate: {metrics['parse_success_rate']:.4f}")
    if "schema_adherence_rate" in metrics:
        lines.append(f"- Schema adherence rate: {metrics['schema_adherence_rate']:.4f}")
    lines.append("")

    if examples:
        lines.append("## Example Predictions\n")
        for idx, ex in enumerate(examples, start=1):
            schema_snippet = (ex["schema"] or "").strip()
            if schema_snippet:
                # Keep the snippet reasonably short.
                schema_lines = schema_snippet.splitlines()
                if len(schema_lines) > 12:
                    schema_snippet = "\n".join(schema_lines[:12]) + "\n-- ..."
            lines.append(f"### Example {idx} – ID: `{ex['id']}`\n")
            if ex["question"]:
                lines.append(f"**Question:** {ex['question']}\n")
            if schema_snippet:
                lines.append("**Schema snippet:**")
                lines.append("```sql")
                lines.append(schema_snippet)
                lines.append("```")
            lines.append("**Gold SQL:**")
            lines.append("```sql")
            lines.append(ex["gold_sql"] or "")
            lines.append("```")
            lines.append("**Predicted SQL:**")
            lines.append("```sql")
            lines.append(ex["pred_sql"] or "")
            lines.append("```")
            lines.append(
                f"- Exact match: {bool(ex.get('exact_match'))}\n"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    logger.info(
        "Starting internal evaluation with val_path=%s, base_model=%s, adapter_dir=%s, "
        "device=%s, max_examples=%d, mock=%s",
        args.val_path,
        args.base_model,
        args.adapter_dir,
        args.device,
        args.max_examples,
        args.mock,
    )

    try:
        val_path = Path(args.val_path)
        records = _load_val_records(val_path, max_examples=args.max_examples)

        preds, golds, contexts, examples = _generate_predictions(records, args)

        flags = MetricFlags(compute_schema_adherence=True)
        metrics = aggregate_metrics(
            preds=preds,
            golds=golds,
            contexts=contexts,
            flags=flags,
        )

        out_dir = Path(args.out_dir)
        payload: Dict[str, Any] = {
            "config": {
                "val_path": str(val_path),
                "base_model": args.base_model,
                "adapter_dir": args.adapter_dir,
                "device": args.device,
                "max_examples": len(preds),
                "mock": bool(args.mock),
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            "metrics": metrics,
            "examples": examples,
        }

        json_path = out_dir / "eval_internal.json"
        md_path = out_dir / "eval_internal.md"

        _write_json(json_path, payload)
        _write_markdown(md_path, payload)

        logger.info("Internal evaluation completed. Reports written to %s", out_dir)
        return 0

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during internal evaluation.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the src/ directory is on sys.path so that `text2sql` can be imported
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql.data_prep import INSTRUCTION_TEXT  # noqa: E402  # isort: skip
from text2sql.training.formatting import build_prompt, ensure_sql_only  # noqa: E402  # isort: skip
from text2sql.eval.metrics import aggregate_metrics  # noqa: E402  # isort: skip
from text2sql.eval.normalize import normalize_sql, normalize_sql_no_values  # noqa: E402  # isort: skip
from text2sql.eval.schema import referenced_identifiers, schema_adherence  # noqa: E402  # isort: skip


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
        description=(
            "Evaluate a fine-tuned text-to-SQL model on the internal "
            "b-mc2/sql-create-context validation set."
        )
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
        help="Base model name or path used during fine-tuning.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help=(
            "Path to the LoRA adapter directory produced by training. "
            "Required for non-mock evaluation."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on (default: auto).",
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
        help="Output directory for evaluation artifacts (JSON + Markdown).",
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
        default=0.95,
        help="Top-p nucleus sampling parameter (used when temperature > 0).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Run in mock mode: skip model loading and reuse the gold SQL as "
            "the prediction. Useful for testing the evaluation pipeline "
            "without a GPU or model weights."
        ),
    )

    return parser.parse_args(argv)


def _read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def _split_schema_and_question(input_text: str) -> Tuple[str, str]:
    """
    Extract the schema context and question from the `input` field.

    The expected format matches `build_input_text`:

    ### Schema:
    <CREATE TABLE ...>

    ### Question:
    <question text>
    """
    schema_marker = "### Schema:"
    question_marker = "### Question:"

    schema_start = input_text.find(schema_marker)
    if schema_start == -1:
        return "", input_text.strip()

    after_schema = schema_start + len(schema_marker)
    question_index = input_text.find(question_marker, after_schema)
    if question_index == -1:
        schema_section = input_text[after_schema:].strip()
        return schema_section, ""

    schema_section = input_text[after_schema:question_index].strip()
    question_section = input_text[question_index + len(question_marker) :].strip()
    return schema_section, question_section


def load_eval_records(
    jsonl_path: Path,
    max_examples: int,
    mock: bool,
) -> List[Dict[str, Any]]:
    """
    Load evaluation records from a JSONL file.

    In `--mock` mode, if the requested path does not exist, this function
    falls back to the small fixture:
    `tests/fixtures/eval_internal_sample.jsonl`.
    """
    PathsTried = [jsonl_path]

    if not jsonl_path.is_file() and mock:
        fixture_path = ROOT / "tests" / "fixtures" / "eval_internal_sample.jsonl"
        if fixture_path.is_file():
            logger.info(
                "Validation file '%s' not found; falling back to fixture '%s'.",
                jsonl_path,
                fixture_path,
            )
            PathsTried = [fixture_path]

    for path in PathsTried:
        if path.is_file():
            logger.info("Loading evaluation records from %s", path)
            return _read_jsonl(path, limit=max_examples)

    raise FileNotFoundError(f"Validation JSONL file not found: {jsonl_path}")


def _build_examples(
    records: List[Dict[str, Any]],
    preds: List[str],
) -> List[Dict[str, Any]]:
    """
    Build per-example summaries for inclusion in the Markdown report.
    """
    examples: List[Dict[str, Any]] = []

    for record, pred_sql in zip(records, preds):
        input_text = record.get("input", "")
        gold_sql = record.get("output", "")
        schema_text, question = _split_schema_and_question(input_text)

        schema_snippet_lines = schema_text.splitlines()
        schema_snippet = "\n".join(schema_snippet_lines[:6])

        # Per-example metrics
        em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
        no_values_em = (
            normalize_sql_no_values(pred_sql) == normalize_sql_no_values(gold_sql)
        )
        try:
            sqlglot_ok = True
            import sqlglot  # type: ignore

            sqlglot.parse_one(pred_sql)
        except Exception:  # noqa: BLE001
            sqlglot_ok = False

        adherent = schema_adherence(pred_sql, schema_text)

        examples.append(
            {
                "id": record.get("id"),
                "question": question,
                "schema_snippet": schema_snippet,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "em": em,
                "no_values_em": no_values_em,
                "parse_success": sqlglot_ok,
                "schema_adherent": adherent,
            }
        )

    return examples


def _render_markdown_report(
    path: Path,
    metrics: Dict[str, Any],
    examples: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    """
    Write a human-readable Markdown report with aggregate metrics and examples.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Internal Evaluation – b-mc2/sql-create-context (val)")
    lines.append("")
    lines.append(f"- **Mode:** {'mock' if config.get('mock') else 'model'}")
    lines.append(f"- **Val path:** `{config.get('val_path')}`")
    lines.append(f"- **Base model:** `{config.get('base_model')}`")
    if config.get("adapter_dir"):
        lines.append(f"- **Adapter dir:** `{config.get('adapter_dir')}`")
    lines.append(f"- **Device:** `{config.get('device')}`")
    lines.append(f"- **Max examples:** {config.get('max_examples')}")
    lines.append("")
    lines.append("## Summary metrics")
    lines.append("")
    n = metrics.get("num_examples", 0)
    lines.append(f"- Examples: **{n}**")
    lines.append(f"- Exact Match (normalized): **{metrics.get('em', 0.0):.3f}**")
    lines.append(
        f"- No-values EM (literals masked): **{metrics.get('no_values_em', 0.0):.3f}**"
    )
    lines.append(
        f"- SQL parse success rate: **{metrics.get('parse_success_rate', 0.0):.3f}**"
    )
    if "schema_adherence_rate" in metrics:
        lines.append(
            f"- Schema adherence rate: **{metrics.get('schema_adherence_rate', 0.0):.3f}**"
        )

    lines.append("")
    lines.append("## Examples")
    lines.append("")
    if not examples:
        lines.append("_No examples available._")
    else:
        for idx, ex in enumerate(examples, start=1):
            lines.append(f"### Example {idx}")
            lines.append("")
            if ex.get("id") is not None:
                lines.append(f"**ID:** `{ex['id']}`")
                lines.append("")
            lines.append("**Question**")
            lines.append("")
            lines.append(ex.get("question") or "_(missing)_")
            lines.append("")
            lines.append("**Schema (truncated)**")
            lines.append("")
            lines.append("```sql")
            lines.append(ex.get("schema_snippet") or "")
            lines.append("```")
            lines.append("")
            lines.append("**Gold SQL**")
            lines.append("")
            lines.append("```sql")
            lines.append(ex.get("gold_sql") or "")
            lines.append("```")
            lines.append("")
            lines.append("**Predicted SQL**")
            lines.append("")
            lines.append("```sql")
            lines.append(ex.get("pred_sql") or "")
            lines.append("```")
            lines.append("")
            lines.append(
                f"- EM: `{ex['em']}`  "
                f"- No-values EM: `{ex['no_values_em']}`  "
                f"- Parse success: `{ex['parse_success']}`  "
                f"- Schema adherent: `{ex['schema_adherent']}`"
            )
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute the internal evaluation pipeline using parsed arguments.

    Returns
    -------
    dict
        Summary information including metrics and paths to artifacts.
    """
    val_path = Path(args.val_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_eval_records(val_path, max_examples=args.max_examples, mock=args.mock)
    if not records:
        raise RuntimeError("No evaluation records loaded.")

    # Prepare model if not in mock mode.
    generate_fn = None
    device_used = "cpu"

    if args.mock:
        logger.info("Running in --mock mode; predictions will echo gold SQL.")
    else:
        if not args.adapter_dir:
            raise ValueError(
                "adapter_dir is required for non-mock evaluation. "
                "If you have merged adapters into the base model, consider "
                "writing a small custom script that uses "
                "`text2sql.infer.load_model_for_inference` directly."
            )
        from text2sql.infer import (  # type: ignore  # noqa: E402
            generate_sql,
            load_model_for_inference,
        )

        _, _, device_used = load_model_for_inference(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            device=args.device,
        )
        generate_fn = generate_sql
        logger.info("Model loaded; beginning inference on device '%s'.", device_used)

    preds: List[str] = []
    golds: List[str] = []
    contexts: List[str] = []

    for record in records:
        instruction = record.get("instruction") or INSTRUCTION_TEXT
        input_text = record.get("input", "")
        gold_sql_raw = record.get("output", "")

        schema_text, _question = _split_schema_and_question(input_text)

        gold_sql = ensure_sql_only(gold_sql_raw)

        if args.mock:
            pred_sql = gold_sql
        else:
            assert generate_fn is not None
            prompt = build_prompt(instruction=instruction, input=input_text)
            pred_raw = generate_fn(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            pred_sql = ensure_sql_only(pred_raw)

        preds.append(pred_sql)
        golds.append(gold_sql)
        contexts.append(schema_text)

    metrics = aggregate_metrics(
        preds=preds,
        golds=golds,
        contexts=contexts,
        compute_schema_adherence=True,
    )

    # Build a small set of examples for the Markdown report.
    example_count = min(10, len(records))
    examples = _build_examples(records[:example_count], preds[:example_count])

    # Persist artifacts.
    metrics_path = out_dir / "eval_internal.json"
    report_path = out_dir / "eval_internal.md"

    config: Dict[str, Any] = {
        "val_path": str(val_path),
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "device": device_used if not args.mock else "mock",
        "max_examples": len(records),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "mock": bool(args.mock),
    }

    payload: Dict[str, Any] = {
        "metrics": metrics,
        "config": config,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    _render_markdown_report(report_path, metrics=metrics, examples=examples, config=config)

    logger.info("Wrote JSON metrics to %s", metrics_path)
    logger.info("Wrote Markdown report to %s", report_path)

    summary: Dict[str, Any] = {
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "num_examples": metrics.get("num_examples", 0),
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the internal evaluation script."""
    configure_logging()
    args = parse_args(argv)

    logger.info(
        "Starting internal evaluation with val_path=%s, base_model=%s, "
        "adapter_dir=%s, device=%s, max_examples=%d, mock=%s",
        args.val_path,
        args.base_model,
        args.adapter_dir,
        args.device,
        args.max_examples,
        args.mock,
    )

    try:
        _ = run(args)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1
    except Exception:  # noqa: BLE001
        logger.error("Unexpected error during evaluation.", exc_info=True)
        return 1

    logger.info("Internal evaluation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
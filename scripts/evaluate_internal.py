import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

# Reduce TensorFlow/CUDA log noise if TensorFlow is installed.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Ensure the src/ directory is on sys.path so that `text2sql` can be imported
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql.eval.metrics import aggregate_metrics  # noqa: E402  # isort: skip
from text2sql.eval.normalize import normalize_sql  # noqa: E402  # isort: skip
from text2sql.training.formatting import (  # noqa: E402  # isort: skip
    build_prompt,
    ensure_sql_only,
)


logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """
    Configuration snapshot for an internal evaluation run.

    This is primarily used for logging and for embedding run metadata into the
    JSON / Markdown reports so that results remain reproducible.
    """

    val_path: str
    base_model: str
    adapter_dir: Optional[str]
    device: str
    max_examples: int
    out_dir: str
    temperature: float
    top_p: float
    max_new_tokens: int
    mock: bool
    load_in_4bit: Optional[bool] = None
    dtype: str = "auto"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    smoke: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def configure_logging() -> None:
    """Configure basic logging for the evaluation script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for internal evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a text-to-SQL model on the internal validation set."
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
            "Path to LoRA adapter directory. If omitted, the script will run "
            "with the base model only (or a merged model if base_model points "
            "to a local directory)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', 'cuda', or 'cpu'.",
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
        help="Output directory for reports (JSON and Markdown).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation (0.0 for greedy).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter for generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        dest="load_in_4bit",
        default=True,
        help=(
            "Enable loading the base model in 4-bit (bitsandbytes) for faster "
            "and more memory-efficient inference. By default this is enabled "
            "automatically when running on CUDA and disabled on CPU."
        ),
    )
    parser.add_argument(
        "--no_load_in_4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable 4-bit loading even when running on CUDA.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help=(
            "Model dtype for base weights. 'auto' selects float16 on CUDA and "
            "float32 on CPU."
        ),
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        help=(
            "Quantization type for 4-bit weights (e.g. 'nf4', 'fp4'). "
            "This is passed to transformers.BitsAndBytesConfig when "
            "--load_in_4bit is enabled."
        ),
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16"],
        help=(
            "Compute dtype for 4-bit quantization. "
            "'auto' chooses bfloat16 on CUDA GPUs with bf16 support, "
            "otherwise float16."
        ),
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action="store_true",
        dest="bnb_4bit_use_double_quant",
        default=True,
        help=(
            "Whether to use nested (double) quantization when loading in 4-bit. "
            "Enabled by default for better memory efficiency."
        ),
    )
    parser.add_argument(
        "--no_bnb_4bit_use_double_quant",
        action="store_false",
        dest="bnb_4bit_use_double_quant",
        help="Disable nested (double) quantization when using 4-bit weights.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Run in mock mode: skip model loading and use gold SQL as "
            "predictions to validate the metric pipeline."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a lightweight smoke test over a small subset of validation "
            "examples. On CPU-only environments this will automatically "
            "fall back to --mock to skip GPU-only model loading."
        ),
    )

    return parser.parse_args(argv)


def _load_alpaca_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load an Alpaca-style JSONL file into a list of dicts."""
    if not path.is_file():
        raise FileNotFoundError(f"Validation JSONL file not found: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise RuntimeError(f"No records found in validation file: {path}")

    return records


def _extract_schema_and_question(input_text: str) -> Tuple[str, str]:
    """
    Extract schema context and question from the formatted input text.

    The expected structure is:

    ### Schema:
    <CREATE TABLE ...>

    ### Question:
    <natural language question>
    """
    schema = ""
    question = ""

    if not input_text:
        return schema, question

    schema_marker = "### Schema:"
    question_marker = "### Question:"

    text = input_text

    if schema_marker in text:
        after_schema = text.split(schema_marker, 1)[1]
    else:
        after_schema = text

    if question_marker in after_schema:
        schema_part, question_part = after_schema.split(question_marker, 1)
        schema = schema_part.strip()
        question = question_part.strip()
    else:
        schema = after_schema.strip()
        question = ""

    return schema, question


def _write_json_report(
    out_path: Path,
    config: EvalConfig,
    metrics: Dict[str, Any],
    examples: List[Dict[str, Any]],
) -> None:
    """Write a JSON report with metrics, config, and example predictions."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "metrics": metrics,
        "examples": examples,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_markdown_report(
    out_path: Path,
    config: EvalConfig,
    metrics: Dict[str, Any],
    examples: List[Dict[str, Any]],
) -> None:
    """Write a human-readable Markdown evaluation report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_examples = metrics.get("n_examples", 0)
    em = metrics.get("exact_match", {})
    nvem = metrics.get("no_values_em", {})
    parse = metrics.get("parse_success", {})
    schema = metrics.get("schema_adherence", {})

    def _fmt_rate(entry: Dict[str, Any]) -> str:
        count = entry.get("count", 0)
        rate = entry.get("rate", 0.0)
        return f"{count}/{n_examples} ({rate:.3f})"

    lines: List[str] = []

    lines.append("# Internal Evaluation – b-mc2/sql-create-context val\n")
    lines.append("## Configuration\n")
    lines.append(f"- **val_path:** `{config.val_path}`")
    lines.append(f"- **base_model:** `{config.base_model}`")
    lines.append(f"- **adapter_dir:** `{config.adapter_dir or 'None (base/merged model only)'}`")
    lines.append(f"- **device:** `{config.device}`")
    lines.append(f"- **max_examples:** `{config.max_examples}`")
    lines.append(f"- **temperature:** `{config.temperature}`")
    lines.append(f"- **top_p:** `{config.top_p}`")
    lines.append(f"- **max_new_tokens:** `{config.max_new_tokens}`")
    lines.append(f"- **mock:** `{config.mock}`")
    lines.append(f"- **smoke:** `{config.smoke}`")
    lines.append(f"- **load_in_4bit:** `{config.load_in_4bit}`")
    lines.append(f"- **dtype:** `{config.dtype}`")
    lines.append(f"- **bnb_4bit_quant_type:** `{config.bnb_4bit_quant_type}`")
    lines.append(f"- **bnb_4bit_compute_dtype:** `{config.bnb_4bit_compute_dtype}`")
    lines.append(f"- **bnb_4bit_use_double_quant:** `{config.bnb_4bit_use_double_quant}`")
    lines.append(f"- **n_evaluated_examples:** `{n_examples}`\n")

    lines.append("## Metrics\n")
    lines.append(f"- **Exact Match (normalized SQL):** {_fmt_rate(em)}")
    lines.append(f"- **No-values Exact Match:** {_fmt_rate(nvem)}")
    lines.append(f"- **SQL parse success rate:** {_fmt_rate(parse)}")
    if schema:
        lines.append(f"- **Schema adherence rate:** {_fmt_rate(schema)}")
    lines.append("")

    lines.append("## Notes\n")
    if config.mock:
        lines.append(
            "- This run used `--mock`, so predictions are set equal to gold SQL to "
            "validate the evaluation pipeline. Metrics should be near 1.0 except "
            "for parser robustness."
        )
    else:
        lines.append(
            "- Exact Match and No-values EM are computed on normalized SQL strings "
            "(whitespace collapsed, trailing semicolons removed)."
        )
        lines.append(
            "- No-values EM further replaces string and numeric literals with "
            "placeholders, focusing on query structure rather than concrete values."
        )
    if config.smoke:
        lines.append(
            "- `--smoke` mode was enabled, so only a small subset of validation "
            "examples was evaluated for a quick health check."
        )
    lines.append("")

    lines.append("## Example Predictions\n")
    if not examples:
        lines.append("_No examples available._")
    else:
        for idx, ex in enumerate(examples, start=1):
            lines.append(f"### Example {idx}\n")
            lines.append(f"- **id:** `{ex.get('id', '')}`")
            lines.append(f"- **Question:** {ex.get('question', '').strip()}")
            schema_snippet = ex.get("schema_snippet", "")
            lines.append(f"- **Schema snippet:** `{schema_snippet}`")
            lines.append(f"- **Gold SQL:** `{ex.get('gold_sql', '').strip()}`")
            lines.append(f"- **Predicted SQL:** `{ex.get('pred_sql', '').strip()}`")
            lines.append("")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_eval(args: argparse.Namespace) -> int:
    """Execute the internal evaluation pipeline."""
    val_path = Path(args.val_path)
    records = _load_alpaca_jsonl(val_path)

    original_total = len(records)

    if args.max_examples is not None and args.max_examples > 0:
        records = records[: args.max_examples]

    if args.smoke:
        # In smoke mode we intentionally evaluate only a tiny subset of the
        # validation set to keep the run fast and cheap.
        smoke_limit = min(5, len(records)) if records else 0
        records = records[:smoke_limit]
        logger.info(
            "Running in --smoke mode: evaluating %d example(s) out of %d loaded "
            "(max_examples=%d).",
            len(records),
            original_total,
            args.max_examples,
        )

        # On CPU-only environments, skip GPU-only model loading by switching
        # to --mock semantics automatically.
        if not args.mock:
            try:
                import torch  # type: ignore[import]
            except Exception:  # pragma: no cover - import-time issues
                logger.warning(
                    "PyTorch is not available; --smoke will run in mock mode "
                    "without loading a model."
                )
                args.mock = True
            else:
                if not torch.cuda.is_available():
                    logger.info(
                        "No CUDA device detected; --smoke will run in mock mode "
                        "and skip model loading."
                    )
                    args.mock = True

    logger.info(
        "Loaded %d validation records from %s (max_examples=%d, smoke=%s).",
        len(records),
        val_path,
        args.max_examples,
        args.smoke,
    )

    # Resolve the compute dtype for 4-bit quantization. We accept an 'auto'
    # option that prefers bfloat16 on GPUs with bf16 support and otherwise
    # falls back to float16.
    bnb_compute_dtype = args.bnb_4bit_compute_dtype
    if bnb_compute_dtype == "auto":
        resolved = "float16"
        try:
            import torch  # type: ignore[import]
        except Exception:  # pragma: no cover - import-time issues
            logger.warning(
                "Could not import torch while resolving --bnb_4bit_compute_dtype; "
                "falling back to float16."
            )
        else:
            has_cuda = torch.cuda.is_available()
            if has_cuda and (args.device or "auto").lower() != "cpu":
                is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
                try:
                    if callable(is_bf16_supported) and torch.cuda.is_bf16_supported():
                        resolved = "bfloat16"
                except Exception:  # pragma: no cover - defensive
                    resolved = "float16"
        bnb_compute_dtype = resolved

    logger.info(
        "4-bit loading configuration: load_in_4bit=%s, "
        "bnb_4bit_quant_type=%s, bnb_4bit_compute_dtype=%s, "
        "bnb_4bit_use_double_quant=%s",
        args.load_in_4bit,
        args.bnb_4bit_quant_type,
        bnb_compute_dtype,
        args.bnb_4bit_use_double_quant,
    )

    eval_config = EvalConfig(
        val_path=str(val_path),
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        device=args.device,
        max_examples=args.max_examples,
        out_dir=args.out_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        mock=args.mock,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_compute_dtype,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        smoke=args.smoke,
    )

    gold_sqls: List[str] = []
    pred_sqls: List[str] = []
    contexts: List[str] = []
    questions: List[str] = []
    example_ids: List[str] = []

    for rec in records:
        input_text = rec.get("input", "")
        output_sql = rec.get("output", "")
        rec_id = rec.get("id", "")

        schema, question = _extract_schema_and_question(input_text)
        contexts.append(schema)
        gold_sqls.append(output_sql)
        questions.append(question)
        example_ids.append(str(rec_id))

    if args.mock:
        logger.info("Running in --mock mode; using gold SQL as predictions.")
        pred_sqls = list(gold_sqls)
    else:
        # Import inference helpers lazily so that --mock mode does not require
        # heavy runtime dependencies like torch to be installed.
        from text2sql.infer import (  # type: ignore[import]
            generate_sql,
            load_model_for_inference,
        )

        logger.info(
            "Loading model for inference with base_model=%s, adapter_dir=%s, "
            "device=%s, load_in_4bit=%s, dtype=%s, bnb_4bit_quant_type=%s, "
            "bnb_4bit_compute_dtype=%s, bnb_4bit_use_double_quant=%s",
            args.base_model,
            args.adapter_dir,
            args.device,
            args.load_in_4bit,
            args.dtype,
            args.bnb_4bit_quant_type,
            bnb_compute_dtype,
            args.bnb_4bit_use_double_quant,
        )
        load_model_for_inference(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            device=args.device,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_compute_dtype=bnb_compute_dtype,
            dtype=args.dtype,
        )

        for idx, rec in enumerate(records):
            instruction = rec.get("instruction", "")
            input_text = rec.get("input", "")
            prompt = build_prompt(instruction=instruction, input=input_text)

            logger.info("Generating prediction for example %d/%d", idx + 1, len(records))
            raw_output = generate_sql(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            cleaned_sql = ensure_sql_only(raw_output)
            pred_sqls.append(cleaned_sql)

    metrics = aggregate_metrics(
        predictions=pred_sqls,
        golds=gold_sqls,
        contexts=contexts,
        compute_schema_adherence=True,
    )
    logger.info("Evaluation metrics: %s", metrics)

    # Build up to 10 example entries for the reports.
    example_entries: List[Dict[str, Any]] = []
    num_examples_to_show = min(10, len(records))
    for i in range(num_examples_to_show):
        schema = contexts[i]
        schema_snippet = schema.replace("\n", " ")
        if len(schema_snippet) > 200:
            schema_snippet = schema_snippet[:197] + "..."

        example_entries.append(
            {
                "id": example_ids[i],
                "question": questions[i],
                "schema_snippet": schema_snippet,
                "gold_sql": normalize_sql(gold_sqls[i]),
                "pred_sql": normalize_sql(pred_sqls[i]),
            }
        )

    out_dir = Path(args.out_dir)
    json_path = out_dir / "eval_internal.json"
    md_path = out_dir / "eval_internal.md"

    _write_json_report(json_path, eval_config, metrics, example_entries)
    _write_markdown_report(md_path, eval_config, metrics, example_entries)

    logger.info("Internal evaluation reports written to %s and %s", json_path, md_path)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    try:
        return run_eval(args)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1
    except Exception:  # noqa: BLE001
        logger.error("Unexpected error during evaluation.", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
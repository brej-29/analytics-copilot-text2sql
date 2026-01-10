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
from text2sql.eval.schema import parse_create_table_context  # noqa: E402  # isort: skip
from text2sql.eval.spider import (  # noqa: E402  # isort: skip
    build_schema_map,
    build_spider_prompt,
    load_spider_schema_map,
    spider_schema_to_pseudo_ddl,
)
from text2sql.training.formatting import ensure_sql_only  # noqa: E402  # isort: skip


logger = logging.getLogger(__name__)


@dataclass
class SpiderEvalConfig:
    base_model: str
    adapter_dir: Optional[str]
    device: str
    spider_split: str
    spider_source: str
    schema_source: str
    max_examples: int
    out_dir: str
    temperature: float
    top_p: float
    max_new_tokens: int
    mock: bool
    load_in_4bit: Optional[bool]
    dtype: str

    def todict(self)


def configure_logging() -> None:
    """Configure basic logging for the evaluation script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for Spider external evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a text-to-SQL model on the Spider dev set (external validation)."
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
        "--spider_split",
        type=str,
        default="validation",
        help="Spider split to evaluate on (e.g., 'validation').",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=200,
        help="Maximum number of Spider examples to evaluate.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="reports",
        help="Output directory for reports (JSON and Markdown).",
    )
    parser.add_argument(
        "--schema_source",
        type=str,
        default="richardr1126/spider-schema",
        help="Hugging Face dataset id for Spider schemas.",
    )
    parser.add_argument(
        "--spider_source",
        type=str,
        default="xlangai/spider",
        help="Hugging Face dataset id for Spider examples.",
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
        default=None,
        help=(
            "If set, force loading the base model in 4-bit (bitsandbytes) for "
            "faster and more memory-efficient inference. By default this is "
            "enabled automatically when running on CUDA and disabled on CPU."
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
        "--mock",
        action="store_true",
        help=(
            "Run in mock mode: load small local Spider fixtures from tests/fixtures "
            "and use gold SQL as predictions to validate prompt building and metrics."
        ),
    )

    return parser.parse_args(argv)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No records found in JSONL file: {path}")
    return rows


def _build_examples_with_schema(
    spider_rows: List[Dict[str, Any]],
    schema_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Join Spider examples with their schemas and log intersection diagnostics.
    """
    spider_db_ids = {str(row["db_id"]) for row in spider_rows if "db_id" in row}
    schema_db_ids = set(schema_map.keys())
    intersection = spider_db_ids & schema_db_ids

    logger.info("Total Spider examples: %d", len(spider_rows))
    logger.info("Unique Spider db_ids: %d", len(spider_db_ids))
    logger.info("Total schema db_ids: %d", len(schema_db_ids))
    logger.info("Intersection size (db_ids in both): %d", len(intersection))

    if not intersection:
        spider_samples = sorted(spider_db_ids)[:10]
        schema_samples = sorted(schema_db_ids)[:10]
        logger.error("No matching db_ids between Spider split and schema source.")
        logger.error("Sample Spider db_ids: %s", spider_samples)
        logger.error("Sample schema db_ids: %s", schema_samples)
        raise RuntimeError(
            "No matching db_ids between Spider split and schema source"
        )

    examples: List[Dict[str, Any]] = []
    skipped_due_to_missing_schema = 0

    for row in spider_rows:
        db_id = row.get("db_id")
        question = row.get("question")
        query = row.get("query")
        if db_id is None or question is None or query is None:
            continue

        db_id_str = str(db_id)
        schema_text = schema_map.get(db_id_str)
        if not schema_text:
            skipped_due_to_missing_schema += 1
            continue

        examples.append(
            {
                "db_id": db_id_str,
                "question": str(question),
                "query": str(query),
                "schema_text": str(schema_text),
            }
        )

    logger.info(
        "After joining with schema: evaluated_count=%d, "
        "skipped_due_to_missing_schema=%d",
        len(examples),
        skipped_due_to_missing_schema,
    )
    return examples


def _load_spider_from_fixtures() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Load Spider examples and schema mapping from local test fixtures.

    This is used when running in --mock mode so that tests and local runs do
    not require internet access or full Spider downloads.
    """
    fixtures_dir = ROOT / "tests" / "fixtures"
    spider_path = fixtures_dir / "spider_sample.jsonl"
    schema_path = fixtures_dir / "spider_schema_sample.jsonl"

    spider_rows = _load_jsonl(spider_path)
    schema_records = _load_jsonl(schema_path)
    schema_map = load_spider_schema_map(schema_records)

    examples = _build_examples_with_schema(spider_rows, schema_map)
    return examples, schema_map


def _write_json_report(
    out_path: Path,
    config: SpiderEvalConfig,
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
    config: SpiderEvalConfig,
    metrics: Dict[str, Any],
    examples: List[Dict[str, Any]],
) -> None:
    """Write a human-readable Markdown evaluation report for Spider."""
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

    lines.append("# External Evaluation – Spider dev (lightweight)\n")
    lines.append("## Configuration\n")
    lines.append(f"- **spider_source:** `{config.spider_source}`")
    lines.append(f"- **schema_source:** `{config.schema_source}`")
    lines.append(f"- **spider_split:** `{config.spider_split}`")
    lines.append(f"- **base_model:** `{config.base_model}`")
    lines.append(f"- **adapter_dir:** `{config.adapter_dir or 'None (base/merged model only)'}`")
    lines.append(f"- **device:** `{config.device}`")
    lines.append(f"- **max_examples:** `{config.max_examples}`")
    lines.append(f"- **temperature:** `{config.temperature}`")
    lines.append(f"- **top_p:** `{config.top_p}`")
    lines.append(f"- **max_new_tokens:** `{config.max_new_tokens}`")
    lines.append(f"- **mock:** `{config.mock}`")
    lines.append(f"- **n_evaluated_examples:** `{n_examples}`\n")

    lines.append("## Metrics\n")
    lines.append(f"- **Exact Match (normalized SQL):** {_fmt_rate(em)}")
    lines.append(f"- **No-values Exact Match:** {_fmt_rate(nvem)}")
    lines.append(f"- **SQL parse success rate:** {_fmt_rate(parse)}")
    if schema:
        lines.append(f"- **Schema adherence rate:** {_fmt_rate(schema)}")
    lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- This is a lightweight Spider external validation intended as a "
        "portfolio-style baseline, not a full reproduction of official Spider "
        "evaluation."
    )
    lines.append(
        "- Official Spider metrics include component matching and execution-based "
        "evaluation. Here we report simple logical-form approximations: Exact "
        "Match, No-values EM, and parse success."
    )
    lines.append(
        "- Schema adherence is computed by checking that predicted queries only "
        "reference tables and columns present in the serialized schema context."
    )
    if config.mock:
        lines.append(
            "- This run used local fixtures and `--mock`, so predictions are set "
            "equal to gold SQL to validate prompt construction and metric logic."
        )
    lines.append("")

    lines.append("## Example Predictions\n")
    if not examples:
        lines.append("_No examples available._")
    else:
        for idx, ex in enumerate(examples, start=1):
            lines.append(f"### Example {idx}\n")
            lines.append(f"- **db_id:** `{ex.get('db_id', '')}`")
            lines.append(f"- **Question:** {ex.get('question', '').strip()}")
            schema_snippet = ex.get("schema_snippet", "")
            lines.append(f"- **Schema snippet:** `{schema_snippet}`")
            lines.append(f"- **Gold SQL:** `{ex.get('gold_sql', '').strip()}`")
            lines.append(f"- **Predicted SQL:** `{ex.get('pred_sql', '').strip()}`")
            lines.append("")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _load_spider_from_hub(
    spider_source: str,
    spider_split: str,
    schema_source: str,
    max_examples: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Load Spider examples and schema mapping from Hugging Face datasets.

    Parameters
    ----------
    spider_source : str
        HF dataset id for Spider examples.
    spider_split : str
        Split name for Spider examples (e.g., 'validation').
    schema_source : str
        HF dataset id for Spider schema helper.
    max_examples : int
        Maximum number of examples to keep.

    Returns
    -------
    (examples, schema_map)
        examples: list of records with at least db_id, question, query and
        attached schema text.
        schema_map: mapping {db_id -> raw schema text}.
    """
    from datasets import load_dataset  # Imported lazily to keep tests lightweight.

    logger.info(
        "Loading Spider dataset '%s' (split=%s) and schema '%s' from Hugging Face.",
        spider_source,
        spider_split,
        schema_source,
    )

    spider_ds = load_dataset(spider_source, split=spider_split)
    schema_ds = load_dataset(schema_source, split="train")

    schema_map = load_spider_schema_map(schema_ds)

    # Materialise Spider rows so we can compute intersections and slice.
    spider_rows: List[Dict[str, Any]] = [dict(row) for row in spider_ds]  # type: ignore[arg-type]

    examples = _build_examples_with_schema(spider_rows, schema_map)

    if max_examples is not None and max_examples > 0:
        examples = examples[:max_examples]

    logger.info(
        "Loaded %d Spider examples with matching schema entries.", len(examples)
    )
    return examples, schema_map


def run_eval(args: argparse.Namespace) -> int:
    """Execute the Spider external evaluation pipeline."""
    if args.mock:
        logger.info("Running Spider evaluation in --mock mode using local fixtures.")
        examples, schema_map = _load_spider_from_fixtures()
    else:
        logger.info(
            "Running Spider evaluation with spider_source=%s, schema_source=%s, split=%s",
            args.spider_source,
            args.schema_source,
            args.spider_split,
        )
        examples, schema_map = _load_spider_from_hub(
            spider_source=args.spider_source,
            spider_split=args.spider_split,
            schema_source=args.schema_source,
            max_examples=args.max_examples,
        )

    if not examples:
        raise RuntimeError("No Spider examples available for evaluation.")

    if args.max_examples is not None and args.max_examples > 0:
        examples = examples[: args.max_examples]

    eval_config = SpiderEvalConfig(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        device=args.device,
        spider_split=args.spider_split,
        spider_source=args.spider_source,
        schema_source=args.schema_source,
        max_examples=args.max_examples,
        out_dir=args.out_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        mock=args.mock,
    )

    gold_sqls: List[str] = []
    pred_sqls: List[str] = []
    contexts: List[str] = []
    questions: List[str] = []
    db_ids: List[str] = []

    for ex in examples:
        db_id = ex["db_id"]
        question = ex["question"]
        gold_query = ex["query"]
        raw_schema_text = ex.get("schema_text", "")
        schema_context = spider_schema_to_pseudo_ddl(raw_schema_text)

        db_ids.append(db_id)
        questions.append(question)
        gold_sqls.append(gold_query)
        contexts.append(schema_context)

    if not db_ids:
        raise RuntimeError("After filtering, no Spider examples had matching schemas.")

    if args.mock:
        logger.info("Using gold SQL as predictions in --mock mode.")
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
            "device=%s, load_in_4bit=%s, dtype=%s",
            args.base_model,
            args.adapter_dir,
            args.device,
            args.load_in_4bit,
            args.dtype,
        )
        load_model_for_inference(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            device=args.device,
            load_in_4bit=args.load_in_4bit,
            bnb_compute_dtype="float16",
            dtype=args.dtype,
        )

        for idx, (db_id, question, schema_context) in enumerate(
            zip(db_ids, questions, contexts)
        ):
            prompt = build_spider_prompt(schema_context=schema_context, question=question)
            logger.info(
                "Generating prediction for Spider example %d/%d (db_id=%s)",
                idx + 1,
                len(db_ids),
                db_id,
            )
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
    logger.info("Spider evaluation metrics: %s", metrics)

    # Build up to 10 example entries for the reports.
    example_entries: List[Dict[str, Any]] = []
    num_examples_to_show = min(10, len(db_ids))
    for i in range(num_examples_to_show):
        schema_context = contexts[i]
        schema_snippet = schema_context.replace("\n", " ")
        if len(schema_snippet) > 200:
            schema_snippet = schema_snippet[:197] + "..."

        example_entries.append(
            {
                "db_id": db_ids[i],
                "question": questions[i],
                "schema_snippet": schema_snippet,
                "gold_sql": normalize_sql(gold_sqls[i]),
                "pred_sql": normalize_sql(pred_sqls[i]),
            }
        )

    out_dir = Path(args.out_dir)
    json_path = out_dir / "eval_spider.json"
    md_path = out_dir / "eval_spider.md"

    _write_json_report(json_path, eval_config, metrics, example_entries)
    _write_markdown_report(md_path, eval_config, metrics, example_entries)

    logger.info("Spider external evaluation reports written to %s and %s", json_path, md_path)
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
        logger.error("Spider evaluation failed: %s", exc)
        return 1
    except Exception:  # noqa: BLE001
        logger.error("Unexpected error during Spider evaluation.", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
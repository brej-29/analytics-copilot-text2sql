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

from datasets import DatasetDict, load_dataset  # noqa: E402  # isort: skip

from text2sql.data_prep import INSTRUCTION_TEXT, build_input_text  # noqa: E402  # isort: skip
from text2sql.training.formatting import build_prompt, ensure_sql_only  # noqa: E402  # isort: skip
from text2sql.eval.metrics import aggregate_metrics  # noqa: E402  # isort: skip
from text2sql.eval.normalize import normalize_sql, normalize_sql_no_values  # noqa: E402  # isort: skip


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for Spider external evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "External evaluation of a text-to-SQL model on the Spider dev split "
            "using xlangai/spider and richardr1126/spider-schema."
        )
    )

    parser.add_argument(
        "--spider_source",
        type=str,
        default="xlangai/spider",
        help="Hugging Face dataset name for Spider (default: xlangai/spider).",
    )
    parser.add_argument(
        "--spider_subset",
        type=str,
        default="spider",
        help="Subset/configuration name for Spider (default: 'spider').",
    )
    parser.add_argument(
        "--spider_split",
        type=str,
        default="validation",
        help="Split to use for evaluation (e.g., 'validation', 'train').",
    )
    parser.add_argument(
        "--schema_source",
        type=str,
        default="richardr1126/spider-schema",
        help=(
            "Hugging Face dataset name providing schema text for each db_id "
            "(default: richardr1126/spider-schema)."
        ),
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
        help="Maximum number of Spider examples to evaluate.",
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


def build_spider_input(schema_ddl: str, question: str) -> str:
    """
    Build the `input` field for Spider evaluation examples.

    This mirrors the internal formatting used during training:
    - The schema (serialized as CREATE TABLE statements).
    - The natural language question.
    """
    return build_input_text(context=schema_ddl, question=question)


def build_spider_prompt(
    instruction: str,
    schema_ddl: str,
    question: str,
) -> str:
    """
    Construct the full text-to-SQL prompt for a Spider example.

    This is a thin wrapper to ensure the same prompt structure used for
    internal training is reused for external evaluation.
    """
    input_text = build_spider_input(schema_ddl=schema_ddl, question=question)
    return build_prompt(instruction=instruction, input=input_text)


def _detect_schema_field(dataset: DatasetDict) -> str:
    """
    Heuristically detect the schema text field in the spider-schema dataset.
    """
    column_names = list(dataset["train"].column_names)
    # Prefer columns whose name starts with "schema".
    for name in column_names:
        if name.lower().startswith("schema"):
            return name

    # Fallback: choose the first non-id / non-key column.
    for name in column_names:
        lower = name.lower()
        if lower == "db_id":
            continue
        if "primary" in lower or "foreign" in lower:
            continue
        return name

    raise KeyError(
        f"Could not identify a schema field in spider-schema columns: {column_names}"
    )


def _schema_text_to_create_table(schema_text: str) -> str:
    """
    Convert the compact spider-schema textual format into pseudo CREATE TABLE DDL.

    The spider-schema dataset encodes each database as a single string of the form:

        table1 : col1 (type) , col2 (type) | table2 : ...

    We expand this into a series of CREATE TABLE statements. The exact types are
    not semantically important for the model; the table/column names are.
    """
    if not schema_text:
        return ""

    tables_ddl: List[str] = []

    for table_block in schema_text.split("|"):
        block = table_block.strip()
        if not block:
            continue
        if ":" not in block:
            continue

        table_name_part, cols_part = block.split(":", 1)
        table_name = table_name_part.strip()
        if not table_name:
            continue

        columns: List[str] = []
        for col_chunk in cols_part.split(","):
            col_chunk = col_chunk.strip()
            if not col_chunk:
                continue
            # Expect "ColumnName (type)" pattern most of the time.
            if "(" in col_chunk:
                col_name_part, type_part = col_chunk.split("(", 1)
                col_name = col_name_part.strip()
                col_type = type_part.rstrip(")").strip()
                if col_type:
                    columns.append(f"{col_name} {col_type}")
                else:
                    columns.append(col_name)
            else:
                columns.append(col_chunk)

        if columns:
            ddl = "CREATE TABLE " + table_name + " (\n  " + ",\n  ".join(columns) + "\n);"
        else:
            ddl = f"CREATE TABLE {table_name} ();"
        tables_ddl.append(ddl)

    return "\n".join(tables_ddl)


def _load_spider_from_hub(
    source: str,
    subset: str,
    split: str,
    max_examples: int,
) -> List[Dict[str, Any]]:
    logger.info(
        "Loading Spider dataset from '%s', subset='%s', split='%s'.",
        source,
        subset,
        split,
    )
    ds = load_dataset(source, subset, split=split)
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        records.append(
            {
                "db_id": row["db_id"],
                "question": row["question"],
                "query": row["query"],
            }
        )
        if len(records) >= max_examples:
            break
    return records


def _load_spider_schema_from_hub(source: str) -> Dict[str, str]:
    logger.info("Loading Spider schema dataset from '%s'.", source)
    ds_dict = load_dataset(source)
    schema_field = _detect_schema_field(ds_dict)
    ds = ds_dict["train"]

    schemas: Dict[str, str] = {}
    for row in ds:
        db_id = row["db_id"]
        schema_text = row[schema_field]
        schemas[db_id] = str(schema_text)

    return schemas


def _load_spider_from_fixtures(max_examples: int) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Load Spider examples and schemas from local test fixtures.

    Used in `--mock` mode to avoid any external downloads.
    """
    spider_path = ROOT / "tests" / "fixtures" / "spider_sample.jsonl"
    schema_path = ROOT / "tests" / "fixtures" / "spider_schema_sample.jsonl"

    if not spider_path.is_file() or not schema_path.is_file():
        raise FileNotFoundError(
            f"Spider fixtures not found at {spider_path} and {schema_path}."
        )

    spider_records = _read_jsonl(spider_path, limit=max_examples)
    schema_records = _read_jsonl(schema_path, limit=None)

    schemas: Dict[str, str] = {}
    for row in schema_records:
        db_id = row["db_id"]
        schemas[db_id] = row["schema"]

    return spider_records, schemas


def _build_examples(
    records: List[Dict[str, Any]],
    schema_by_db: Dict[str, str],
    preds: List[str],
) -> List[Dict[str, Any]]:
    """
    Build per-example summaries for inclusion in the Markdown report.
    """
    examples: List[Dict[str, Any]] = []

    for record, pred_sql in zip(records, preds):
        db_id = record["db_id"]
        question = record["question"]
        gold_sql = record["query"]
        schema_text = schema_by_db.get(db_id, "")

        schema_ddl = _schema_text_to_create_table(schema_text)
        schema_snippet_lines = schema_ddl.splitlines()
        schema_snippet = "\n".join(schema_snippet_lines[:6])

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

        examples.append(
            {
                "db_id": db_id,
                "question": question,
                "schema_snippet": schema_snippet,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "em": em,
                "no_values_em": no_values_em,
                "parse_success": sqlglot_ok,
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
    lines.append("# External Evaluation – Spider dev")
    lines.append("")
    lines.append(f"- **Mode:** {'mock' if config.get('mock') else 'model'}")
    lines.append(f"- **Spider source:** `{config.get('spider_source')}`")
    lines.append(f"- **Spider subset:** `{config.get('spider_subset')}`")
    lines.append(f"- **Spider split:** `{config.get('spider_split')}`")
    lines.append(f"- **Schema source:** `{config.get('schema_source')}`")
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
    lines.append("")
    lines.append("## Examples")
    lines.append("")
    if not examples:
        lines.append("_No examples available._")
    else:
        for idx, ex in enumerate(examples, start=1):
            lines.append(f"### Example {idx}")
            lines.append("")
            lines.append(f"**db_id:** `{ex['db_id']}`")
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
                f"- Parse success: `{ex['parse_success']}`"
            )
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute the Spider external evaluation pipeline using parsed arguments.

    Returns
    -------
    dict
        Summary information including metrics and paths to artifacts.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load examples and schemas.
    if args.mock:
        logger.info("Running in --mock mode; loading Spider fixtures only.")
        records, schema_by_db = _load_spider_from_fixtures(max_examples=args.max_examples)
        device_used = "mock"
        generate_fn = None
    else:
        records = _load_spider_from_hub(
            source=args.spider_source,
            subset=args.spider_subset,
            split=args.spider_split,
            max_examples=args.max_examples,
        )
        schema_by_db = _load_spider_schema_from_hub(source=args.schema_source)

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
        logger.info("Model loaded; beginning Spider inference on device '%s'.", device_used)

    if not records:
        raise RuntimeError("No Spider evaluation records loaded.")

    preds: List[str] = []
    golds: List[str] = []

    for record in records:
        db_id = record["db_id"]
        question = record["question"]
        gold_sql_raw = record["query"]

        schema_text = schema_by_db.get(db_id, "")
        schema_ddl = _schema_text_to_create_table(schema_text)

        gold_sql = ensure_sql_only(gold_sql_raw)

        if args.mock:
            pred_sql = gold_sql
        else:
            assert generate_fn is not None
            prompt = build_spider_prompt(
                instruction=INSTRUCTION_TEXT,
                schema_ddl=schema_ddl,
                question=question,
            )
            pred_raw = generate_fn(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            pred_sql = ensure_sql_only(pred_raw)

        preds.append(pred_sql)
        golds.append(gold_sql)

    metrics = aggregate_metrics(
        preds=preds,
        golds=golds,
        contexts=None,
        compute_schema_adherence=False,
    )

    example_count = min(10, len(records))
    examples = _build_examples(records[:example_count], schema_by_db, preds[:example_count])

    metrics_path = out_dir / "eval_spider.json"
    report_path = out_dir / "eval_spider.md"

    config: Dict[str, Any] = {
        "spider_source": args.spider_source,
        "spider_subset": args.spider_subset,
        "spider_split": args.spider_split,
        "schema_source": args.schema_source,
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "device": device_used,
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
    """Entry point for the Spider external evaluation script."""
    configure_logging()
    args = parse_args(argv)

    logger.info(
        "Starting Spider evaluation with spider_source=%s, subset=%s, split=%s, "
        "schema_source=%s, base_model=%s, adapter_dir=%s, device=%s, "
        "max_examples=%d, mock=%s",
        args.spider_source,
        args.spider_subset,
        args.spider_split,
        args.schema_source,
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
        logger.error("Unexpected error during Spider evaluation.", exc_info=True)
        return 1

    logger.info("Spider external evaluation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
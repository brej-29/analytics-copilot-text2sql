import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure the src/ directory is on sys.path so that `text2sql` can be imported
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql.data_prep import INSTRUCTION_TEXT, build_input_text  # noqa: E402
from text2sql.training.formatting import build_prompt  # noqa: E402
from text2sql.eval.metrics import aggregate_metrics  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_FIELD_NAME = "Schema (values (type))"


@dataclass
class SpiderExample:
    db_id: str
    question: str
    gold_sql: str
    schema_ddl: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Secondary external validation on Spider dev set."
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
        "--spider_split",
        type=str,
        default="validation",
        help="Spider split to use (e.g., 'validation' or 'train').",
    )
    parser.add_argument(
        "--spider_source",
        type=str,
        default="xlangai/spider",
        help="Hugging Face dataset identifier for Spider.",
    )
    parser.add_argument(
        "--schema_source",
        type=str,
        default="richardr1126/spider-schema",
        help="Hugging Face dataset identifier for Spider schemas.",
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
            "Run in mock mode: use local Spider fixtures and gold SQL as "
            "predictions to validate the evaluation pipeline. No HF downloads."
        ),
    )
    return parser.parse_args(argv)


def _parse_spider_schema_values(raw: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse the compact 'Schema (values (type))' format into table -> [(col, type)].

    The format is pipe-delimited, e.g.:

        |phone_market|phone : Name (text) , Phone_ID (number) , ... | market : ...

    Only segments that contain parentheses are treated as table definitions;
    later segments describing keys/foreign keys are ignored for our purposes.
    """
    text = (raw or "").strip()
    if not text:
        return {}

    parts = [p.strip() for p in text.split("|") if p.strip()]
    if not parts:
        return {}

    # First part is usually db_id; we ignore it here.
    table_segments = parts[1:]
    tables: Dict[str, List[Tuple[str, str]]] = {}

    for seg in table_segments:
        if "(" not in seg:
            # Likely a primary/foreign key description; skip.
            continue
        if ":" not in seg:
            continue

        table_name_part, cols_part = seg.split(":", 1)
        table_name = table_name_part.strip()
        if not table_name:
            continue

        cols: List[Tuple[str, str]] = []
        for chunk in cols_part.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "(" in chunk:
                name_part, type_part = chunk.split("(", 1)
                col_name = name_part.strip()
                col_type = type_part.rstrip(")").strip()
            else:
                col_name = chunk
                col_type = "text"
            if col_name:
                cols.append((col_name, col_type))

        if cols:
            tables[table_name] = cols

    return tables


def _schema_row_to_ddl(row: Dict[str, Any]) -> str:
    raw_schema = row.get(SCHEMA_FIELD_NAME, "") or ""
    db_id = row.get("db_id", "").strip()
    tables = _parse_spider_schema_values(raw_schema)

    stmts: List[str] = []
    for table_name, cols in tables.items():
        col_defs: List[str] = []
        for col_name, type_str in cols:
            lower_type = type_str.lower()
            if any(t in lower_type for t in ("number", "int", "real", "double", "float")):
                sql_type = "NUMERIC"
            else:
                sql_type = "TEXT"
            col_defs.append(f"{col_name} {sql_type}")
        if col_defs:
            stmts.append(f"CREATE TABLE {table_name} ({', '.join(col_defs)});")

    ddl = " ".join(stmts)
    if not ddl and raw_schema:
        # Fall back to embedding the raw schema text if parsing failed.
        ddl = raw_schema

    return ddl


def _load_spider_from_fixtures(max_examples: int) -> List[SpiderExample]:
    fixture_dir = ROOT / "tests" / "fixtures"
    spider_path = fixture_dir / "spider_sample.jsonl"
    schema_path = fixture_dir / "spider_schema_sample.jsonl"

    if not spider_path.is_file():
        raise FileNotFoundError(f"Spider fixture not found at {spider_path}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"Spider schema fixture not found at {schema_path}")

    logger.info("Loading Spider examples from fixture: %s", spider_path)
    spider_rows: List[Dict[str, Any]] = []
    with spider_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spider_rows.append(json.loads(line))

    logger.info("Loading Spider schema rows from fixture: %s", schema_path)
    schema_by_db: Dict[str, Dict[str, Any]] = {}
    with schema_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            db_id = row.get("db_id", "").strip()
            if db_id:
                schema_by_db[db_id] = row

    examples: List[SpiderExample] = []
    for row in spider_rows:
        if 0 < max_examples <= len(examples):
            break
        db_id = row.get("db_id", "").strip()
        schema_row = schema_by_db.get(db_id)
        if not schema_row:
            continue
        schema_ddl = _schema_row_to_ddl(schema_row)
        examples.append(
            SpiderExample(
                db_id=db_id,
                question=row.get("question", ""),
                gold_sql=row.get("query", ""),
                schema_ddl=schema_ddl,
            )
        )

    if not examples:
        raise RuntimeError("No Spider examples could be constructed from fixtures.")

    logger.info("Loaded %d Spider examples from fixtures.", len(examples))
    return examples[:max_examples] if max_examples > 0 else examples


def _load_spider_from_hf(
    spider_source: str,
    spider_split: str,
    schema_source: str,
    max_examples: int,
) -> List[SpiderExample]:
    # Import lazily so that --mock mode does not require internet or datasets.
    from datasets import load_dataset  # noqa: WPS433

    logger.info(
        "Loading Spider dataset from Hugging Face: %s (split=%s)",
        spider_source,
        spider_split,
    )
    spider_ds = load_dataset(spider_source, split=spider_split)

    logger.info("Loading Spider schema dataset from Hugging Face: %s", schema_source)
    schema_ds = load_dataset(schema_source, split="train")

    schema_by_db: Dict[str, Dict[str, Any]] = {}
    for row in schema_ds:
        db_id = row.get("db_id", "").strip()
        if db_id:
            schema_by_db[db_id] = row

    examples: List[SpiderExample] = []
    for row in spider_ds:
        if 0 < max_examples <= len(examples):
            break
        db_id = row.get("db_id", "").strip()
        if not db_id:
            continue
        schema_row = schema_by_db.get(db_id)
        if not schema_row:
            continue
        schema_ddl = _schema_row_to_ddl(schema_row)
        examples.append(
            SpiderExample(
                db_id=db_id,
                question=row.get("question", ""),
                gold_sql=row.get("query", ""),
                schema_ddl=schema_ddl,
            )
        )

    if not examples:
        raise RuntimeError(
            "No Spider examples could be constructed from Hugging Face datasets. "
            "Ensure that both datasets are accessible and that db_id keys align."
        )

    logger.info("Loaded %d Spider examples from Hugging Face.", len(examples))
    return examples[:max_examples] if max_examples > 0 else examples


def _build_prompt(schema_ddl: str, question: str) -> str:
    """
    Build the instruction-style prompt for a Spider example using the same
    template as internal evaluation/training.
    """
    input_text = build_input_text(context=schema_ddl, question=question)
    return build_prompt(instruction=INSTRUCTION_TEXT, input=input_text)


def _generate_predictions(
    examples: List[SpiderExample],
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], List[str], List[Dict[str, Any]]]:
    preds: List[str] = []
    golds: List[str] = []
    contexts: List[str] = []
    example_rows: List[Dict[str, Any]] = []

    if not args.mock:
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
            return ""

    for idx, ex in enumerate(examples):
        prompt = _build_prompt(ex.schema_ddl, ex.question)

        if args.mock:
            pred_sql = ex.gold_sql
        else:
            try:
                pred_sql = _predict(prompt)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Prediction failed for Spider example %d (db_id=%s); using empty string.",
                    idx,
                    ex.db_id,
                )
                pred_sql = ""

        preds.append(pred_sql)
        golds.append(ex.gold_sql)
        contexts.append(ex.schema_ddl)

        if len(example_rows) < 10:
            example_rows.append(
                {
                    "db_id": ex.db_id,
                    "question": ex.question,
                    "schema": ex.schema_ddl,
                    "prompt": prompt,
                    "gold_sql": ex.gold_sql,
                    "pred_sql": pred_sql,
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
    lines.append("# External Validation – Spider dev\n")
    lines.append("## Configuration\n")
    lines.append(f"- Spider source: `{cfg['spider_source']}`")
    lines.append(f"- Spider split: `{cfg['spider_split']}`")
    lines.append(f"- Schema source: `{cfg['schema_source']}`")
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
    lines.append(f"- SQL parse success rate: {metrics['parse_success_rate']:.4f}\n")

    lines.append("> Note: Official Spider evaluations typically report component-level\n")
    lines.append("> matching and execution accuracy using the reference evaluation\n")
    lines.append("> script. This report provides a lightweight external validation\n")
    lines.append("> focused on logical-form exact match and parseability.\n")

    if examples:
        lines.append("\n## Example Predictions\n")
        for idx, ex in enumerate(examples, start=1):
            schema_snippet = (ex["schema"] or "").strip()
            if schema_snippet:
                schema_lines = schema_snippet.splitlines()
                if len(schema_lines) > 12:
                    schema_snippet = "\n".join(schema_lines[:12]) + "\n-- ..."
            lines.append(f"### Example {idx} – db_id: `{ex['db_id']}`\n")
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

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    logger.info(
        "Starting Spider external evaluation with spider_source=%s, spider_split=%s, "
        "schema_source=%s, base_model=%s, adapter_dir=%s, device=%s, max_examples=%d, "
        "mock=%s",
        args.spider_source,
        args.spider_split,
        args.schema_source,
        args.base_model,
        args.adapter_dir,
        args.device,
        args.max_examples,
        args.mock,
    )

    try:
        if args.mock:
            examples = _load_spider_from_fixtures(max_examples=args.max_examples)
        else:
            examples = _load_spider_from_hf(
                spider_source=args.spider_source,
                spider_split=args.spider_split,
                schema_source=args.schema_source,
                max_examples=args.max_examples,
            )

        preds, golds, contexts, examples_payload = _generate_predictions(examples, args)

        metrics = aggregate_metrics(preds=preds, golds=golds, contexts=None, flags=None)

        out_dir = Path(args.out_dir)
        payload: Dict[str, Any] = {
            "config": {
                "spider_source": args.spider_source,
                "spider_split": args.spider_split,
                "schema_source": args.schema_source,
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
            "examples": examples_payload,
        }

        json_path = out_dir / "eval_spider.json"
        md_path = out_dir / "eval_spider.md"

        _write_json(json_path, payload)
        _write_markdown(md_path, payload)

        logger.info("Spider external evaluation completed. Reports written to %s", out_dir)
        return 0

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        logger.error("Spider evaluation failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during Spider external evaluation.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
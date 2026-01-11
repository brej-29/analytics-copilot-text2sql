import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset

# Ensure the src/ directory is on sys.path so that `text2sql` can be imported
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql.training.config import TrainingConfig  # noqa: E402  # isort: skip
from text2sql.training.formatting import build_prompt, ensure_sql_only  # noqa: E402  # isort: skip


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for QLoRA training."""
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Mistral-7B-Instruct on Text-to-SQL data."
    )

    # Data paths
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/processed/train.jsonl",
        help="Path to the Alpaca-style training JSONL file.",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/processed/val.jsonl",
        help="Path to the Alpaca-style validation JSONL file.",
    )

    # Model + output
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Base model name or path for fine-tuning.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base output directory for adapters, logs, and metadata.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum number of training steps.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay coefficient.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization.",
    )

    # LoRA / QLoRA hyperparameters
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (r).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout probability.",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a smoke test: load datasets and validate formatting. "
            "Model loading is skipped on CPU-only environments."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Run a dry run: load a small batch, build prompts, and exit without training."
        ),
    )

    return parser.parse_args(argv)


def get_git_commit() -> Optional[str]:
    """Return the current git commit hash if available, otherwise None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def load_alpaca_jsonl(jsonl_path: Path) -> Dataset:
    """
    Load an Alpaca-style JSONL file into a datasets.Dataset.

    Each line is expected to be a JSON object containing at least:
    - instruction
    - input
    - output
    """
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    logger.info("Loading Alpaca-style dataset from %s", jsonl_path)
    records: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise RuntimeError(f"No records found in JSONL file: {jsonl_path}")

    ds = Dataset.from_list(records)
    logger.info("Loaded %d records from %s", ds.num_rows, jsonl_path)
    return ds


def build_sft_dataset(ds: Dataset) -> Dataset:
    """
    Convert an Alpaca-style dataset into a format expected by SFTTrainer.

    We build a single `text` field containing:
    - build_prompt(instruction, input)
    - followed immediately by the target SQL (ensure_sql_only(output)).
    """

    def _format_example(example: Dict[str, Any]) -> Dict[str, str]:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        prompt = build_prompt(instruction=instruction, input=input_text)
        sql = ensure_sql_only(output)
        text = prompt + sql

        return {"text": text}

    logger.info("Formatting dataset into SFT text field.")
    formatted = ds.map(_format_example, remove_columns=ds.column_names)
    logger.info("Formatted dataset now has columns: %s", formatted.column_names)
    return formatted


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Write a JSON file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Construct a TrainingConfig from CLI arguments."""
    return TrainingConfig(
        base_model=args.base_model,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )


def summarize_datasets(
    train_ds: Dataset, val_ds: Dataset
) -> Tuple[int, int]:
    """Return the number of examples in train and val datasets."""
    return train_ds.num_rows, val_ds.num_rows


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for QLoRA training."""
    configure_logging()
    args = parse_args(argv)

    logger.info(
        "Starting QLoRA run with base_model=%s, train_path=%s, val_path=%s, "
        "output_dir=%s, max_steps=%d, batch_size=%d, grad_accum=%d, "
        "learning_rate=%.2e, seed=%d, smoke=%s, dry_run=%s",
        args.base_model,
        args.train_path,
        args.val_path,
        args.output_dir,
        args.max_steps,
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.seed,
        args.smoke,
        args.dry_run,
    )

    out_dir = Path(args.output_dir)
    adapters_dir = out_dir / "adapters"
    metrics_path = out_dir / "metrics.json"
    run_meta_path = out_dir / "run_meta.json"

    try:
        train_raw = load_alpaca_jsonl(Path(args.train_path))
        val_raw = load_alpaca_jsonl(Path(args.val_path))

        train_ds = build_sft_dataset(train_raw)
        val_ds = build_sft_dataset(val_raw)

        train_size, val_size = summarize_datasets(train_ds, val_ds)
        logger.info("SFT train size: %d, val size: %d", train_size, val_size)

        training_config = build_training_config(args)
        git_commit = get_git_commit()

        # Prepare run metadata (even for dry_run / smoke).
        run_meta: Dict[str, Any] = {
            "mode": "train"
            if (not args.dry_run and not args.smoke)
            else "dry_run" if args.dry_run else "smoke",
            "train_path": args.train_path,
            "val_path": args.val_path,
            "output_dir": args.output_dir,
            "base_model": args.base_model,
            "config": training_config.to_dict(),
            "num_train_examples": train_size,
            "num_val_examples": val_size,
            "git_commit": git_commit,
        }

        # Dry run: format a small batch and exit without touching the model.
        if args.dry_run:
            logger.info("Running in --dry_run mode; no model will be loaded.")
            sample = train_ds.select(range(min(3, train_size)))
            logger.info("Sample formatted SFT texts:")
            for i, row in enumerate(sample):
                logger.info("Example %d: %s", i, row["text"][:400])
            save_json(run_meta_path, run_meta)
            save_json(metrics_path, {"note": "dry_run - no training performed"})
            logger.info("Dry run completed successfully.")
            return 0

        # Smoke test: validate dataset + basic config; model loading is optional.
        if args.smoke:
            logger.info("Running in --smoke mode; validating dataset and config.")
            logger.info(
                "Train examples: %d, Val examples: %d, Max seq length: %d",
                train_size,
                val_size,
                args.max_seq_length,
            )
            if torch.cuda.is_available():
                logger.info(
                    "CUDA is available; in a full smoke we would load the model here."
                )
            else:
                logger.info(
                    "CUDA is NOT available; skipping model load in smoke test. "
                    "This is expected on CPU-only environments."
                )
            save_json(run_meta_path, run_meta)
            save_json(metrics_path, {"note": "smoke - no training performed"})
            logger.info("Smoke test completed successfully.")
            return 0

        # From this point on we expect to actually train, so CUDA is required.
        if not torch.cuda.is_available():
            logger.error(
                "CUDA required for training. No GPU detected. "
                "Use --dry_run or --smoke to perform non-training checks on CPU."
            )
            return 1

        logger.info("CUDA is available. Proceeding with model loading and training.")

        # Lazy-import heavy training dependencies so that dry_run/smoke remain lightweight.
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        max_seq_length = args.max_seq_length
        dtype = None  # Let Unsloth pick bf16/float16 as appropriate.
        load_in_4bit = True

        logger.info(
            "Loading base model '%s' in 4-bit with Unsloth (max_seq_length=%d).",
            args.base_model,
            max_seq_length,
        )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        logger.info(
            "Applying LoRA adapters (r=%d, alpha=%d, dropout=%.3f).",
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )

        sft_config = SFTConfig(
            output_dir=str(out_dir),
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=max(args.max_steps // 10, 1),
            save_strategy="steps",
            save_steps=max(args.max_steps // 10, 1),
            bf16=torch.cuda.is_bf16_supported(),
        )

        logger.info("Initializing SFTTrainer.")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_text_field="text",
            args=sft_config,
        )

        logger.info("Starting training for up to %d steps.", args.max_steps)
        train_output = trainer.train()
        logger.info("Training completed.")

        # Gather metrics
        metrics: Dict[str, Any] = {}
        if hasattr(train_output, "metrics") and train_output.metrics is not None:
            metrics.update(train_output.metrics)

        try:
            eval_metrics = trainer.evaluate()
            metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
        except Exception:  # noqa: BLE001
            logger.warning("Evaluation failed or was skipped; continuing.", exc_info=True)

        adapters_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving LoRA adapters to %s", adapters_dir)
        trainer.model.save_pretrained(str(adapters_dir))
        tokenizer.save_pretrained(str(adapters_dir))

        save_json(run_meta_path, run_meta)
        save_json(metrics_path, metrics)

        logger.info("Training run completed successfully.")
        return 0

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        logger.error("Training run failed: %s", exc)
        return 1
    except Exception:  # noqa: BLE001
        logger.error("Unexpected error during training run.", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
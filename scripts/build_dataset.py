import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import sys

from typing import TYPE_CHECKING

# Ensure the src/ directory is on sys.path so that `text2sql` can be imported
# when this script is run directly via `python scripts/build_dataset.py`.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from datasets import Dataset, DatasetDict  # noqa: F401

from text2sql.data_prep import format_record  # noqa: E402  # isort: skip


logger = logging.getLogger(__name__)

DATASET_NAME = "b-mc2/sql-create-context"
DEFAULT_VAL_RATIO = 0.08
DEFAULT_SEED = 42


def configure_logging() -> None:
    """Configure basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the dataset builder."""
    parser = argparse.ArgumentParser(
        description=(
            "Build instruction-tuning JSONL files from the b-mc2/sql-create-context dataset."
        )
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed JSONL files (default: data/processed).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=(
            "Fraction of examples to use for the validation split "
            f"(default: {DEFAULT_VAL_RATIO:.2f})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used for deterministic splitting (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help=(
            "Optional maximum number of rows to use from the input dataset. "
            "Useful for quick development runs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if they already exist.",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help=(
            "Optional local JSONL file in raw format "
            "({'question':..., 'context':..., 'answer':...}). "
            "If provided, the script will NOT download from Hugging Face and "
            "will use this file as the input dataset instead."
        ),
    )
    return parser.parse_args(argv)


def load_raw_dataset(
    input_jsonl: Optional[Path], max_rows: Optional[int]
):
    """
    Load the raw dataset, either from a local JSONL file or from Hugging F_codeging Face.

    Parameters
    ----------
    input_jsonl : Optional[Path]
        Path to a local JSONL file containing raw records with keys
        `question`, `context`, and `answer`. If provided, this is used instead
        of downloading the dataset from Hugging Face.
    max_rows : Optional[int]
        Optional maximum number of rows to keep.

    Returns
    -------
    datasets.Dataset
        A datasets.Dataset-like object with at least the keys: question, context,
        answer.

    Raises
    ------
    RuntimeError
        If loading fails for any reason.
    """
    try:
        if input_jsonl is not None:
            logger.info("Loading raw data from local JSONL: %s", input_jsonl)
            records: list[Dict[str, Any]] = []
            with input_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            if not records:
                raise RuntimeError(
                    f"No records found in local JSONL file: {input_jsonl}"
                )

            # Import datasets lazily so that running with --input_jsonl does not
            # require the 'datasets' package to be installed at module import time.
            from datasets import Dataset  # type: ignore[import]

            ds = Dataset.from_list(records)
        else:
            from datasets import DatasetDict, load_dataset  # type: ignore[import]

            logger.info("Loading dataset '%s' from Hugging Face Datasets...", DATASET_NAME)
            ds_dict: DatasetDict = load_dataset(DATASET_NAME)
            if "train" not in ds_dict:
                raise RuntimeError(
                    f"Expected a 'train' split in dataset '{DATASET_NAME}', "
                    f"but found splits: {list(ds_dict.keys())}"
                )
            ds = ds_dict["train"]

        if max_rows is not None:
            max_rows = max(0, max_rows)
            if max_rows == 0:
                raise RuntimeError("--max_rows was set to 0; nothing to process.")
            num_rows = ds.num_rows
            logger.info(
                "Applying max_rows=%d (original dataset size: %d rows).",
                max_rows,
                num_rows,
            )
            max_rows = min(max_rows, num_rows)
            ds = ds.select(range(max_rows))

        logger.info("Loaded raw dataset with %d rows.", ds.num_rows)
        return ds
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load raw dataset.", exc_info=True)
        logger.error(
            "Common causes: missing 'datasets' package, no internet access when "
            "using Hugging Face mode, unreadable JSONL file, or invalid input format."
        )
        raise RuntimeError("Could not load raw dataset.") from exc


def split_dataset(
    ds, val_ratio: float, seed: int
) -> Dict[str, Any]:
    """
    Split the dataset into train and validation sets deterministically.

    Parameters
    ----------
    ds : datasets.Dataset
        Input dataset (typically the 'train' split from HF).
    val_ratio : float
        Fraction of rows to assign to the validation split.
    seed : int
        Random seed for deterministic splitting.

    Returns
    -------
    dict
        A dictionary with keys 'train' and 'val', each a datasets.Dataset-like
        obj_codeecnewt</.
.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}.")

    logger.info(
        "Creating deterministic train/val split with val_ratio=%.3f, seed=%d.",
        val_ratio,
        seed,
    )
    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]

    logger.info(
        "Split sizes -> train: %d, val: %d",
        train_ds.num_rows,
        val_ds.num_rows,
    )

    return {"train": train_ds, "val": val_ds}


def ensure_output_paths(
    out_dir: Path, overwrite: bool
) -> Dict[str, Path]:
    """
    Compute and validate output paths for train/val JSONL files.

    Parameters
    ----------
    out_dir : Path
        Base output directory.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    dict
        Mapping {'train': train_path, 'val': val_path}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    for path in (train_path, val_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {path}. Use --overwrite to replace it."
            )

    return {"train": train_path, "val": val_path}


def write_jsonl(
    ds,
    split_name: str,
    path: Path,
    val_ratio: float,
    seed: int,
    from_local_input: bool,
) -> int:
    """
    Write a dataset split to a JSONL file in instruction-tuning format.

    Parameters
    ----------
    ds : datasets.Dataset
        The dataset split to write.
    split_name : str
        'train' or 'val'.
    path : Path
        Output JSONL file path.
    val_ratio : float
        Validation ratio used when splitting; recorded in meta.
    seed : int
        Random seed used when splitting; recorded in meta.
    from_local_input : bool
        Whether the dataset came from a local JSONL file.

    Returns
    -------
    int
        Number of records written.
    """
    logger.info("Writing %s split to %s", split_name, path)
    count = 0

    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            question = row.get("question", "")
            context = row.get("context", "")
            answer = row.get("answer", "")

            base_record = format_record(
                question=question,
                context=context,
                answer=answer,
            )

            record_id = f"sqlcc-{split_name}-{idx:06d}"
            meta: Dict[str, Any] = {
                "original_split": "train",
                "row": int(idx),
                "split": split_name,
                "val_ratio": float(val_ratio),
                "seed": int(seed),
                "from_local_input": bool(from_local_input),
            }

            full_record: Dict[str, Any] = {
                "id": record_id,
                **base_record,
                "meta": meta,
            }

            f.write(json.dumps(full_record, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Wrote %d records to %s", count, path)
    return count


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute the dataset build pipeline using parsed arguments.

    Returns
    -------
    dict
        Summary information including row counts and output paths.
    """
    out_dir = Path(args.out_dir)
    input_jsonl = Path(args.input_jsonl) if args.input_jsonl else None

    raw_ds = load_raw_dataset(input_jsonl=input_jsonl, max_rows=args.max_rows)
    split = split_dataset(raw_ds, val_ratio=args.val_ratio, seed=args.seed)

    out_paths = ensure_output_paths(out_dir=out_dir, overwrite=args.overwrite)

    from_local_input = input_jsonl is not None
    train_count = write_jsonl(
        ds=split["train"],
        split_name="train",
        path=out_paths["train"],
        val_ratio=args.val_ratio,
        seed=args.seed,
        from_local_input=from_local_input,
    )
    val_count = write_jsonl(
        ds=split["val"],
        split_name="val",
        path=out_paths["val"],
        val_ratio=args.val_ratio,
        seed=args.seed,
        from_local_input=from_local_input,
    )

    summary = {
        "train_count": train_count,
        "val_count": val_count,
        "train_path": str(out_paths["train"]),
        "val_path": str(out_paths["val"]),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "from_local_input": from_local_input,
    }
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the dataset building script."""
    configure_logging()
    args = parse_args(argv)
    logger.info(
        "Starting dataset build with out_dir=%s, val_ratio=%.3f, seed=%d, "
        "max_rows=%s, overwrite=%s, input_jsonl=%s",
        args.out_dir,
        args.val_ratio,
        args.seed,
        args.max_rows,
        args.overwrite,
        args.input_jsonl,
    )

    try:
        summary = run(args)
    except FileExistsError as exc:
        logger.error("%s", exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        logger.error("Dataset build failed: %s", exc)
        return 1
    except Exception:  # noqa: BLE001
        logger.error("Unexpected error during dataset build.", exc_info=True)
        return 1

    # Final human-readable summary.
    print("\n=== Dataset build summary ===")
    print(f"Train examples: {summary['train_count']}")
    print(f"Val examples:   {summary['val_count']}")
    print(f"Train JSONL:    {summary['train_path']}")
    print(f"Val JSONL:      {summary['val_path']}")
    print(f"Val ratio:      {summary['val_ratio']}")
    print(f"Seed:           {summary['seed']}")
    print(f"Local input:    {summary['from_local_input']}")
    print("=== End of summary ===\n")

    logger.info("Dataset build completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
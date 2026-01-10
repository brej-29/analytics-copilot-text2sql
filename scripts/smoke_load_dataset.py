import logging
from typing import Any, Dict

from datasets import DatasetDict, load_dataset


logger = logging.getLogger(__name__)

DATASET_NAME = "b-mc2/sql-create-context"


def configure_logging() -> None:
    """Configure basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def load_sql_create_context() -> DatasetDict:
    """
    Load the b-mc2/sql-create-context dataset from Hugging Face Datasets.

    Returns
    -------
    datasets.DatasetDict
        A dictionary-like object with one or more splits (e.g., 'train').

    Raises
    ------
    RuntimeError
        If the dataset cannot be loaded for any reason.
    """
    try:
        logger.info("Loading dataset: '%s'...", DATASET_NAME)
        ds = load_dataset(DATASET_NAME)
        logger.info("Successfully loaded dataset '%s'.", DATASET_NAME)
        return ds
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to load dataset '%s' from Hugging Face Datasets.",
            DATASET_NAME,
            exc_info=True,
        )
        logger.error(
            "Common causes: missing 'datasets' package, no internet access, "
            "invalid dataset name, or temporary issues with Hugging Face Hub."
        )
        raise RuntimeError(f"Could not load dataset '{DATASET_NAME}'.") from exc


def get_split_sizes(ds: DatasetDict) -> Dict[str, int]:
    """Return the number of examples in each split of the dataset."""
    sizes: Dict[str, int] = {}
    for split_name, split in ds.items():
        sizes[split_name] = len(split)
    return sizes


def main() -> int:
    """Entry point for the smoke dataset loader."""
    configure_logging()
    logger.info("Starting dataset smoke loader for '%s'.", DATASET_NAME)

    try:
        ds = load_sql_create_context()
    except RuntimeError as exc:
        logger.error("Smoke test failed: %s", exc)
        return 1

    sizes = get_split_sizes(ds)
    logger.info("Dataset split sizes: %s", sizes)

    # Print split sizes for quick visual inspection.
    print(f"\n=== Split sizes for dataset '{DATASET_NAME}' ===")
    for split_name, size in sizes.items():
        print(f"{split_name}: {size}")
    print("=== End of split sizes ===")

    # Show a single example from the training split as a basic sanity check.
    try:
        example: Any = ds["train"][0]
    except KeyError:
        logger.error("Training split 'train' not found in dataset.", exc_info=True)
        return 1
    except IndexError:
        logger.error("Training split 'train' is empty.", exc_info=True)
        return 1

    logger.info("Successfully retrieved an example from the 'train' split.")

    # For interactive inspection, we allow a final print of the example.
    print(f"\n=== Example from '{DATASET_NAME}' 'train' split ===")
    print(example)
    print("=== End of example ===\n")

    logger.info("Dataset smoke loader for '%s' completed successfully.", DATASET_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
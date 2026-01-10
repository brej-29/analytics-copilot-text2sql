import logging
import sys
from typing import Any, Dict

from datasets import DatasetDict, load_dataset


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def load_wikisql() -> DatasetDict:
    """
    Load the Salesforce/wikisql dataset from Hugging Face Datasets.

    Returns
    -------
    datasets.DatasetDict
        A dictionary-like object with 'train', 'validation', and 'test' splits.

    Raises
    ------
    RuntimeError
        If the dataset cannot be loaded for any reason.
    """
    try:
        logger.info("Loading WikiSQL dataset: 'Salesforce/wikisql'...")
        ds = load_dataset("Salesforce/wikisql")
        logger.info("Successfully loaded WikiSQL dataset.")
        return ds
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load WikiSQL dataset from Hugging Face.", exc_info=True)
        logger.error(
            "Common causes: missing 'datasets' package, no internet access, "
            "or temporary issues with Hugging Face Hub."
        )
        raise RuntimeError("Could not load 'Salesforce/wikisql' dataset.") from exc


def get_split_sizes(ds: DatasetDict) -> Dict[str, int]:
    """Return the number of examples in each split of the dataset."""
    sizes: Dict[str, int] = {}
    for split_name, split in ds.items():
        sizes[split_name] = len(split)
    return sizes


def main() -> int:
    """Entry point for the smoke dataset loader."""
    configure_logging()
    logger.info("Starting WikiSQL smoke dataset loader.")

    try:
        ds = load_wikisql()
    except RuntimeError as exc:
        logger.error("Smoke test failed: %s", exc)
        return 1

    sizes = get_split_sizes(ds)
    logger.info("Dataset split sizes: %s", sizes)

    # Show a single example from the training split as a basic sanity check.
    try:
        example: Any = ds["train"][0]
    except KeyError as exc:
        logger.error("Training split 'train' not found in dataset.", exc_info=True)
        return 1
    except IndexError as exc:
        logger.error("Training split 'train' is empty.", exc_info=True)
        return 1

    logger.info("Successfully retrieved an example from the 'train' split.")

    # For interactive inspection, we allow a final print of the example.
    print("\n=== WikiSQL example from 'train' split ===")
    print(example)
    print("=== End of example ===\n")

    logger.info("WikiSQL smoke dataset loader completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
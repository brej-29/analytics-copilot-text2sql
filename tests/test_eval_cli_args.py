from pathlib import Path
import sys

import pytest


def _ensure_root_on_path() -> None:
    """Ensure that the project root is available on sys.path for script imports."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_root_on_path()

from scripts import evaluate_internal  # noqa: E402  # isort: skip
from scripts import evaluate_spider_external  # noqa: E402  # isort: skip


def test_evaluate_internal_parses_4bit_flags() -> None:
    args = evaluate_internal.parse_args(
        [
            "--mock",
            "--load_in_4bit",
            "--dtype",
            "float16",
        ]
    )
    assert args.mock is True
    assert args.load_in_4bit is True
    assert args.dtype == "float16"


def test_evaluate_spider_parses_4bit_flags() -> None:
    args = evaluate_spider_external.parse_args(
        [
            "--mock",
            "--load_in_4bit",
            "--dtype",
            "float16",
        ]
    )
    assert args.mock is True
    assert args.load_in_4bit is True
    assert args.dtype == "float16"
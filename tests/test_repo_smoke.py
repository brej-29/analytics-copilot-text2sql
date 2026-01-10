"""
Basic smoke tests for the Analytics Copilot (Text-to-SQL) repository.

These tests are intentionally lightweight and are meant to verify that:
- The src/ layout is correctly configured for imports.
- The `text2sql` package can be imported without errors.
"""

from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    """Ensure that the 'src' directory is on sys.path for imports."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def test_import_text2sql_package() -> None:
    """Smoke test that the core package imports successfully."""
    _ensure_src_on_path()
    import text2sql  # noqa: F401  # type: ignore[import]  # pylint: disable=unused-import

    assert True
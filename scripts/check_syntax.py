from __future__ import annotations

import compileall
from pathlib import Path
import sys


def _discover_python_files(root_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in root_dirs:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            # Skip compiled files and cache dirs just in case.
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    """
    Compile all Python files under src/, scripts/, and app/ to check syntax.

    This uses the Python stdlib `compileall` module and exits non-zero if any
    file fails to compile.
    """
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "src",
        project_root / "scripts",
        project_root / "app",
    ]

    py_files = _discover_python_files(candidates)
    if not py_files:
        print("No Python files found under src/, scripts/, or app/.")
        return 0

    failures: list[Path] = []
    for path in py_files:
        ok = compileall.compile_file(
            str(path),
            ddir=str(path.parent),
            quiet=1,
        )
        if not ok:
            failures.append(path)

    if failures:
        print("Syntax check FAILED for the following files:")
        for path in failures:
            print(f" - {path}")
        print(f"Total: {len(failures)} file(s) with syntax errors.")
        return 1

    print(f"Syntax OK: compiled {len(py_files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
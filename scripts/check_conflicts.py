#!/usr/bin/env python3
"""Scan repository files for unresolved merge conflict markers.

Useful for ensuring the working tree is clean before pushing or running CI.
Skips large or binary files to keep the check fast and reliable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

CONFLICT_MARKERS: tuple[str, ...] = ("<<<<<<<", "=======", ">>>>>>>")
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "outputs", "inputs"}
SKIP_SUFFIXES = {".parquet", ".db", ".sqlite", ".pkl", ".gz", ".png", ".jpg", ".jpeg"}
MAX_SIZE_BYTES = 2_000_000  # Skip files larger than ~2MB


def has_conflict_marker(lines: Iterable[str]) -> bool:
    """Check if any line begins with a conflict marker."""
    for line in lines:
        stripped = line.lstrip()
        for marker in CONFLICT_MARKERS:
            if not stripped.startswith(marker):
                continue
            if stripped == marker or stripped.startswith(f"{marker} "):
                return True
    return False


def is_skipped(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    try:
        return path.stat().st_size > MAX_SIZE_BYTES
    except OSError:
        return True


def scan_path(path: Path) -> list[Path]:
    """Return a list of files that contain merge conflict markers."""
    bad_files: list[Path] = []
    for file_path in path.rglob("*"):
        if not file_path.is_file() or is_skipped(file_path):
            continue
        try:
            with file_path.open(errors="ignore") as fh:
                if has_conflict_marker(fh):
                    bad_files.append(file_path)
        except (UnicodeDecodeError, OSError):
            continue
    return bad_files


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    bad_files = scan_path(repo_root)
    if bad_files:
        print("Found unresolved conflict markers in:")
        for file_path in bad_files:
            print(f" - {file_path.relative_to(repo_root)}")
        return 1
    print("No unresolved merge conflict markers detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

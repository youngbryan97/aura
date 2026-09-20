#!/usr/bin/env python3
"""Reject generated/runtime files that must never become repository source."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import PurePosixPath

_CACHE_PARTS = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "test_vdb",
}
_CACHE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".sqlite", ".sqlite3"}


def hygiene_violation(path_text: str) -> str | None:
    """Return why a tracked path is generated state, or ``None`` if allowed."""

    path = PurePosixPath(path_text)
    if any(part in _CACHE_PARTS for part in path.parts):
        return "generated_cache"
    if path.suffix.lower() in _CACHE_SUFFIXES or path.name.endswith("$py.class"):
        return "generated_cache"
    if (
        len(path.parts) >= 3
        and path.parts[0:2] == ("artifacts", "closeout")
        and "source_snapshots" in path.parts
    ):
        return "duplicated_source_snapshot"
    return None


def collect_violations(paths: Iterable[str]) -> list[tuple[str, str]]:
    return [
        (path, reason)
        for path in paths
        if (reason := hygiene_violation(path)) is not None
    ]


def tracked_paths() -> list[str]:
    completed = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [path for path in completed.stdout.split("\0") if path]


def main() -> int:
    violations = collect_violations(tracked_paths())
    if not violations:
        return 0
    print("Generated source/runtime artifacts are tracked:")
    for path, reason in violations:
        print(f"  {reason}: {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

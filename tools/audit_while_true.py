#!/usr/bin/env python3
"""Audit and harden 'while True' loops for shutdown awareness.

Scans Python files for 'while True:' loops and checks if they have any
shutdown/cancellation guard. Reports which ones are unguarded.

A loop is considered "guarded" if within 20 lines of the 'while True:' it
contains any of:
  - self._shutdown / self._shutting_down / self._running / self._alive
  - asyncio.CancelledError
  - break (conditional)
  - sentinel / stop / cancel / shutdown keywords
"""

import re
import sys
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}

GUARD_PATTERNS = [
    r"_shutdown",
    r"_shutting_down",
    r"_running",
    r"_alive",
    r"_stop",
    r"_cancelled",
    r"CancelledError",
    r"break\b",
    r"sentinel",
    r"\.cancel\b",
    r"shutdown",
    r"\.is_set\(",
    r"Event\(\)",
    r"\.wait\(",
]


def find_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name.startswith("test_"):
            continue
        files.append(path)
    return sorted(files)


def check_file(filepath: Path) -> list[dict]:
    try:
        lines = filepath.read_text(encoding="utf-8").split("\n")
    except (OSError, UnicodeDecodeError):
        return []

    findings = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("while True:"):
            continue

        # Check next 30 lines for guard patterns
        lookahead = "\n".join(lines[i:min(i + 30, len(lines))])
        guarded = any(re.search(pat, lookahead) for pat in GUARD_PATTERNS)

        findings.append({
            "line": i + 1,
            "guarded": guarded,
            "context": stripped,
        })

    return findings


def main():
    root = Path(__file__).resolve().parents[1]
    search_dirs = [root / "core", root / "skills"]
    all_files = []
    for d in search_dirs:
        if d.exists():
            all_files.extend(find_python_files(d))

    unguarded = 0
    guarded = 0

    for filepath in all_files:
        rel = filepath.relative_to(root)
        findings = check_file(filepath)
        for f in findings:
            if f["guarded"]:
                guarded += 1
            else:
                print(f"  ⚠️  UNGUARDED: {rel}:{f['line']}")
                unguarded += 1

    print(f"\nTotal: {guarded + unguarded} while-True loops ({guarded} guarded, {unguarded} UNGUARDED)")


if __name__ == "__main__":
    main()

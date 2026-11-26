#!/usr/bin/env python3
"""Automated broad-exception narrowing tool for Aura codebase hardening.

Scans Python files under core/ and skills/ for 'except Exception' patterns and
replaces them with context-appropriate specific exception types based on the
code within each try block.

This is a one-shot remediation tool — run it, review the diff, compile-check,
then commit.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway

# ── Exception Mapping Rules ──────────────────────────────────────────────
# Maps patterns found INSIDE the try block to the appropriate narrow catches.

IMPORT_CATCHES = "(ImportError, AttributeError)"
SERVICE_CATCHES = "(ImportError, AttributeError, RuntimeError)"
IO_CATCHES = "(OSError, IOError)"
SQLITE_CATCHES = "(sqlite3.Error, OSError)"
HTTP_CATCHES = "(httpx.HTTPError, OSError, ConnectionError, TimeoutError)"
JSON_CATCHES = "(json.JSONDecodeError, KeyError, TypeError, ValueError)"
SUBPROCESS_CATCHES = "(subprocess.SubprocessError, OSError)"
ASYNC_CATCHES = "(RuntimeError, asyncio.CancelledError, TimeoutError)"
GENERIC_RUNTIME = "(RuntimeError, AttributeError, TypeError)"
MEMORY_CATCHES = "(ImportError, AttributeError, RuntimeError, TypeError)"

# Patterns in try-block content → replacement exception tuple
RULES = [
    # Import-based
    (r"from\s+\S+\s+import\s+|import\s+\S+", SERVICE_CATCHES),
    # ServiceContainer.get
    (r"ServiceContainer\.get\(", SERVICE_CATCHES),
    # get_event_bus / event bus
    (r"get_event_bus|event_bus|EventBus", SERVICE_CATCHES),
    # sqlite3
    (r"sqlite3\.|\.execute\(|\.cursor\(|PRAGMA", "(sqlite3.Error, OSError)"),
    # httpx / HTTP
    (r"httpx\.|\.get\(|\.post\(|client\.", "(httpx.HTTPError, OSError, ConnectionError, TimeoutError)"),
    # subprocess
    (r"subprocess\.", "(subprocess.SubprocessError, OSError)"),
    # json parsing
    (r"json\.loads|json\.dumps|\.json\(\)", "(json.JSONDecodeError, TypeError, ValueError)"),
    # asyncio operations
    (r"asyncio\.wait_for|asyncio\.shield|asyncio\.gather|await\s+\S+\.wait", "(RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError)"),
    # File I/O
    (r"\.open\(|\.read\(|\.write\(|Path\(|os\.path\.|shutil\.", "(OSError, IOError)"),
    # psutil
    (r"psutil\.", "(ImportError, OSError, AttributeError)"),
    # General attribute/method access on dynamic objects
    (r"getattr\(|hasattr\(", "(RuntimeError, AttributeError, TypeError)"),
]

# Files to skip (already handled or intentionally broad)
SKIP_FILES = {
    "diagnostics_bundle.py",   # Intentionally broad for crash-safe diagnostics
}

SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}


def find_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.name.startswith("test_"):
            continue
        files.append(path)
    return sorted(files)


def determine_catch_type(try_block_text: str) -> str:
    """Determine the best narrow exception type based on try block content."""
    for pattern, catch_type in RULES:
        if re.search(pattern, try_block_text):
            return catch_type
    # Default fallback — still much better than bare Exception
    return "(RuntimeError, AttributeError, TypeError, ValueError)"


def process_file(filepath: Path, dry_run: bool = False) -> int:
    """Process a single file, replacing broad exceptions. Returns count of changes."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0

    lines = content.split("\n")
    changes = 0
    i = 0
    new_lines = []

    while i < len(lines):
        line = lines[i]

        # Match "except Exception:" or "except Exception as <name>:"
        match = re.match(r"^(\s*)(except)\s+Exception(\s+as\s+\w+)?(\s*:\s*)(.*)$", line)

        if match:
            indent = match.group(1)
            as_clause = match.group(3) or ""
            colon_part = match.group(4)
            rest = match.group(5)

            # Walk backward to find the try block
            try_block_lines = []
            j = i - 1

            # Find matching try
            while j >= 0:
                l = lines[j]
                ls = l.lstrip()
                if ls.startswith("try:") and len(l) - len(ls) <= len(indent):
                    break
                j -= 1

            if j >= 0:
                # Collect try block content
                for k in range(j + 1, i):
                    try_block_lines.append(lines[k])
                try_block_text = "\n".join(try_block_lines)

                catch_type = determine_catch_type(try_block_text)
                new_line = f"{indent}except {catch_type}{as_clause}{colon_part}{rest}"
                new_lines.append(new_line)
                changes += 1
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
        i += 1

    if changes > 0 and not dry_run:
        filepath.write_text("\n".join(new_lines), encoding="utf-8")

    return changes


def main():
    dry_run = "--dry-run" in sys.argv
    root = ROOT

    search_dirs = [root / "core", root / "skills"]
    all_files = []
    for d in search_dirs:
        if d.exists():
            all_files.extend(find_python_files(d))

    total_changes = 0
    changed_files = 0

    for filepath in all_files:
        rel = filepath.relative_to(root)
        count = process_file(filepath, dry_run=dry_run)
        if count > 0:
            print(f"  {'[DRY] ' if dry_run else ''}Fixed {count:3d} catches in {rel}")
            total_changes += count
            changed_files += 1

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Total: {total_changes} broad exceptions narrowed across {changed_files} files.")

    if not dry_run:
        # Compile check all changed files
        python = str(root / ".venv" / "bin" / "python")
        failures = 0
        for filepath in all_files:
            result = get_subprocess_gateway().run(
                [python, "-m", "py_compile", str(filepath)],
                cwd=root,
                timeout=30,
                read_only=True,
                source="narrow_exceptions:py_compile",
                accelerator_capability="none",
            )
            if result.returncode != 0:
                print(f"  COMPILE FAIL: {filepath.relative_to(root)}")
                print(f"     {result.stderr.strip()}")
                failures += 1

        if failures:
            print(f"\n {failures} files failed compilation. Review and fix manually.")
        else:
            print(f"\nAll {len(all_files)} files compile clean.")


if __name__ == "__main__":
    main()

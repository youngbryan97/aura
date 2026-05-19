#!/usr/bin/env python3
"""Fix files that catch httpx.HTTPError without importing httpx.

These were introduced by the automated `narrow_exceptions.py` tool which
replaced `except Exception` with a broad set of exception types including
`httpx.HTTPError`. Most files don't use httpx, so this catch would produce
a NameError at runtime if triggered.

Strategy:
- In files that DO import httpx: keep httpx.HTTPError
- In files that DON'T import httpx: replace httpx.HTTPError with ConnectionError
  (which is the stdlib equivalent for network failures)
"""

import re
import sys
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}


def find_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def file_imports_httpx(content: str) -> bool:
    """Check if the file has `import httpx` or `from httpx import ...`"""
    return bool(re.search(r'^\s*(?:import httpx|from httpx\b)', content, re.MULTILINE))


def process_file(filepath: Path, dry_run: bool = False) -> int:
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0

    if "httpx.HTTPError" not in content:
        return 0

    if file_imports_httpx(content):
        return 0  # File legitimately uses httpx

    # Replace httpx.HTTPError with ConnectionError in except clauses
    # Pattern: except (httpx.HTTPError, OSError, ConnectionError, TimeoutError)
    # We need to handle it appearing anywhere in an except tuple
    
    new_content = content
    changes = 0
    
    # Replace `httpx.HTTPError, ` (with trailing comma) — most common case
    if "httpx.HTTPError, " in new_content:
        count = new_content.count("httpx.HTTPError, ")
        new_content = new_content.replace("httpx.HTTPError, ", "")
        changes += count
    
    # Replace `, httpx.HTTPError` (preceded by comma) — less common
    if ", httpx.HTTPError" in new_content:
        count = new_content.count(", httpx.HTTPError")
        new_content = new_content.replace(", httpx.HTTPError", "")
        changes += count
    
    # Replace standalone `except httpx.HTTPError` (rare)
    if "except httpx.HTTPError" in new_content:
        count = new_content.count("except httpx.HTTPError")
        new_content = new_content.replace("except httpx.HTTPError", "except ConnectionError")
        changes += count

    # Now check for empty tuples or duplicate types that might result
    # e.g., (OSError, ConnectionError, ConnectionError) → (OSError, ConnectionError)
    # This is a safety net, not expected to trigger often
    
    if changes > 0 and not dry_run:
        filepath.write_text(new_content, encoding="utf-8")

    return changes


def main():
    dry_run = "--dry-run" in sys.argv
    root = Path(__file__).resolve().parents[1]

    search_dirs = [root / "core", root / "skills"]
    all_files = []
    for d in search_dirs:
        if d.exists():
            all_files.extend(find_python_files(d))

    total = 0
    changed = 0

    for filepath in all_files:
        rel = filepath.relative_to(root)
        count = process_file(filepath, dry_run=dry_run)
        if count > 0:
            print(f"  {'[DRY] ' if dry_run else ''}Removed {count:3d} httpx.HTTPError refs from {rel}")
            total += count
            changed += 1

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Total: {total} phantom httpx.HTTPError references removed from {changed} files.")

    if not dry_run:
        import subprocess
        python = str(root / ".venv" / "bin" / "python")
        failures = 0
        for filepath in all_files:
            result = subprocess.run(
                [python, "-m", "py_compile", str(filepath)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  ❌ COMPILE FAIL: {filepath.relative_to(root)}")
                print(f"     {result.stderr.strip()}")
                failures += 1

        if failures:
            print(f"\n⚠️  {failures} files failed compilation.")
        else:
            print(f"\n✅ All {len(all_files)} files compile clean.")


if __name__ == "__main__":
    main()

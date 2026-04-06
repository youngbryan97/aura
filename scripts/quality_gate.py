#!/usr/bin/env python3
"""Automated quality gate for Aura.

Run before every commit to catch issues early.
Exit code 0 = all gates pass. Non-zero = failures found.

Usage:
    python scripts/quality_gate.py          # Full check
    python scripts/quality_gate.py --quick  # Syntax + imports only
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
FAIL_COUNT = 0


def fail(msg: str):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  FAIL: {msg}")


def ok(msg: str):
    print(f"  OK:   {msg}")


def check_syntax():
    """All Python files must parse."""
    print("\n[1/6] Syntax check...")
    errors = 0
    for py_file in ROOT.rglob("*.py"):
        if ".venv" in py_file.parts or "node_modules" in py_file.parts:
            continue
        try:
            ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError as e:
            fail(f"{py_file.relative_to(ROOT)}: {e}")
            errors += 1
    if errors == 0:
        ok(f"All Python files parse cleanly")
    return errors


def check_hardcoded_paths():
    """No /Users/bryan in tracked files."""
    print("\n[2/6] Hardcoded path check...")
    result = subprocess.run(
        ["git", "grep", "-l", "/Users/bryan", "--", "*.py", "*.md", "*.sh", "*.plist"],
        capture_output=True, text=True, cwd=ROOT,
    )
    # Exclude files that legitimately reference the pattern (the gate itself, specs)
    exclude = {"scripts/quality_gate.py", "specs/QUALITY_GATES.md"}
    files = [f for f in result.stdout.strip().split("\n") if f and f not in exclude]
    if files:
        for f in files:
            fail(f"Hardcoded path in: {f}")
        return len(files)
    ok("No hardcoded /Users/bryan paths")
    return 0


def check_no_large_files():
    """No files > 1MB tracked in git."""
    print("\n[3/6] Large file check...")
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True, text=True, cwd=ROOT,
    )
    large = []
    for f in result.stdout.strip().split("\n"):
        if not f:
            continue
        full = ROOT / f
        if full.exists() and full.stat().st_size > 1_000_000:
            large.append((f, full.stat().st_size))
    if large:
        for f, size in large:
            fail(f"{f} is {size // 1024}KB (max 1MB)")
        return len(large)
    ok("No files > 1MB in git")
    return 0


def check_no_logs():
    """No .log files tracked."""
    print("\n[4/6] Log file check...")
    result = subprocess.run(
        ["git", "ls-files", "*.log"],
        capture_output=True, text=True, cwd=ROOT,
    )
    logs = [f for f in result.stdout.strip().split("\n") if f]
    if logs:
        for f in logs:
            fail(f"Log file tracked: {f}")
        return len(logs)
    ok("No .log files in git")
    return 0


def check_no_stubs():
    """No 'not implemented' error returns in core/."""
    print("\n[5/6] Stub check...")
    patterns = ["not_implemented", "not implemented", "Method recognized but not implemented"]
    found = 0
    for py_file in CORE.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                if pattern in content.lower():
                    # Skip comments and docstrings about what was removed
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        if pattern in stripped.lower() and not stripped.startswith("#") and not stripped.startswith('"""') and "return" in stripped:
                            fail(f"{py_file.relative_to(ROOT)}:{i+1}: {stripped[:80]}")
                            found += 1
        except Exception:
            pass
    if found == 0:
        ok("No stub returns in core/")
    return found


def check_tests():
    """Run the test suite."""
    print("\n[6/6] Test suite...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, text=True, cwd=ROOT,
    )
    last_line = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""
    if result.returncode == 0:
        ok(last_line)
        return 0
    else:
        fail(last_line)
        return 1


def main():
    global FAIL_COUNT

    quick = "--quick" in sys.argv

    print("=" * 50)
    print("  AURA QUALITY GATE")
    print("=" * 50)

    check_syntax()
    check_hardcoded_paths()
    check_no_large_files()
    check_no_logs()
    check_no_stubs()

    if not quick:
        check_tests()

    print("\n" + "=" * 50)
    if FAIL_COUNT == 0:
        print("  ALL GATES PASSED")
    else:
        print(f"  {FAIL_COUNT} FAILURE(S)")
    print("=" * 50)

    sys.exit(1 if FAIL_COUNT > 0 else 0)


if __name__ == "__main__":
    main()

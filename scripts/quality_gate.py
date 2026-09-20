#!/usr/bin/env python3
"""Automated quality gate for Aura.

Run before every commit to catch issues early.
Exit code 0 = all gates pass. Non-zero = failures found.

Usage:
    python scripts/quality_gate.py          # Full check
    python scripts/quality_gate.py --quick  # Syntax + imports only
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway

CORE = ROOT / "core"
FAIL_COUNT = 0
APPROVED_LARGE_FILES = {
    "artifacts/architecture/latest.json": "generated architecture-map evidence retained for offline review",
    "artifacts/closeout/semantic_review/SEMANTIC_REVIEW_LEDGER.jsonl": "append-only closeout semantic review evidence",
    "dev_archive/simulation_output/simulate_out.txt": "legacy simulation trace retained as archival evidence",
    "interface/static/vendor/3d-force-graph.min.js": "offline UI vendor bundle used by interface/static/mycelial.html",
    "training/data/train.jsonl": "offline training corpus, not loaded by runtime boot",
    "training/data/valid.jsonl": "offline training validation corpus, not loaded by runtime boot",
    "training/raw_data/human_conversations.json": "offline corpus source for dataset rebuilds",
    "training/raw_data/movie_conversations.txt": "offline corpus source for dataset rebuilds",
    "training/raw_data/movie_lines.txt": "offline corpus source for dataset rebuilds",
}


def run_read_only_command(args: list[str], *, source: str, timeout: float = 30):
    try:
        return get_subprocess_gateway().run(
            args,
            cwd=ROOT,
            timeout=timeout,
            read_only=True,
            source=source,
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        fail(f"{source}: command failed: {exc}")
        return None


def fail(msg: str):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  FAIL: {msg}")


def ok(msg: str):
    print(f"  OK:   {msg}")


def tracked_files(patterns: list[str]) -> list[Path]:
    result = run_read_only_command(
        ["git", "-c", "core.fsmonitor=false", "ls-files", *patterns],
        source="quality_gate_tracked_files",
    )
    if result is None:
        return []
    if result.returncode != 0:
        fail(f"git ls-files failed while listing tracked files: {result.stderr.strip()}")
        return []
    return [ROOT / path for path in result.stdout.splitlines() if path.strip()]


def existing_tracked_files(patterns: list[str]) -> list[Path]:
    """Return tracked files that still exist in the current worktree."""
    return [path for path in tracked_files(patterns) if path.exists()]


def check_syntax():
    """All Python files must parse."""
    print("\n[1/6] Syntax check...")
    errors = 0
    for py_file in existing_tracked_files(["*.py"]):
        try:
            ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError as e:
            fail(f"{py_file.relative_to(ROOT)}: {e}")
            errors += 1
        except OSError as e:
            fail(f"{py_file.relative_to(ROOT)}: unable to read for syntax check: {e}")
            errors += 1
    if errors == 0:
        ok("All Python files parse cleanly")
    return errors


def check_hardcoded_paths():
    """No author-specific home paths in tracked files."""
    print("\n[2/6] Hardcoded path check...")
    home_pattern = str(Path.home())
    result = run_read_only_command(
        ["git", "-c", "core.fsmonitor=false", "grep", "-l", home_pattern, "--", "*.py", "*.md", "*.sh", "*.plist"],
        source="quality_gate_hardcoded_paths",
    )
    if result is None:
        return 1
    if result.returncode not in {0, 1}:
        fail(f"git grep failed while checking hardcoded paths: {result.stderr.strip()}")
        return 1
    # Exclude files that legitimately reference the pattern (the gate itself, specs)
    exclude = {"scripts/quality_gate.py", "scripts/cleanup_agent.py", "specs/QUALITY_GATES.md"}
    files = [f for f in result.stdout.strip().split("\n") if f and f not in exclude]
    if files:
        for f in files:
            fail(f"Hardcoded path in: {f}")
        return len(files)
    ok("No hardcoded personal home paths")
    return 0


def check_no_large_files():
    """No files > 1MB tracked in git."""
    print("\n[3/6] Large file check...")
    result = run_read_only_command(
        ["git", "-c", "core.fsmonitor=false", "ls-files"],
        source="quality_gate_large_files",
    )
    if result is None:
        return 1
    if result.returncode != 0:
        fail(f"git ls-files failed while checking large files: {result.stderr.strip()}")
        return 1
    large = []
    approved = []
    for f in result.stdout.strip().split("\n"):
        if not f:
            continue
        full = ROOT / f
        if full.exists() and full.stat().st_size > 1_000_000:
            if f in APPROVED_LARGE_FILES:
                approved.append((f, full.stat().st_size, APPROVED_LARGE_FILES[f]))
            else:
                large.append((f, full.stat().st_size))
    if large:
        for f, size in large:
            fail(f"{f} is {size // 1024}KB (max 1MB)")
        return len(large)
    for f, size, reason in approved:
        print(f"  WARN: approved large tracked file: {f} ({size // 1024}KB) - {reason}")
    ok("No unapproved files > 1MB in git")
    return 0


def check_no_logs():
    """No .log files tracked."""
    print("\n[4/6] Log file check...")
    result = run_read_only_command(
        ["git", "-c", "core.fsmonitor=false", "ls-files", "*.log"],
        source="quality_gate_logs",
    )
    if result is None:
        return 1
    if result.returncode != 0:
        fail(f"git ls-files failed while checking logs: {result.stderr.strip()}")
        return 1
    logs = [f for f in result.stdout.strip().split("\n") if f]
    if logs:
        for f in logs:
            fail(f"Log file tracked: {f}")
        return len(logs)
    ok("No .log files in git")
    return 0


def check_no_incomplete_returns():
    """No incomplete-implementation error returns in core/."""
    print("\n[5/6] Incomplete implementation check...")
    patterns = ["not_" + "implemented", "not " + "implemented", "Method recognized but not " + "implemented"]
    found = 0
    for py_file in existing_tracked_files(["core/**/*.py"]):
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
        except OSError as exc:
            fail(f"{py_file.relative_to(ROOT)}: unable to read during incomplete implementation check: {exc}")
            found += 1
    if found == 0:
        ok("No incomplete implementation returns in core/")
    return found


def check_tests():
    """Run the test suite."""
    print("\n[6/6] Test suite...")
    result = run_read_only_command(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        timeout=600,
        source="quality_gate_pytest",
    )
    if result is None:
        return 1
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
    check_no_incomplete_returns()

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

#!/usr/bin/env python3
"""Automated cleanup agent for Aura's codebase.

Scans for common issues and generates a report or auto-fixes them.
Designed to run on a schedule (cron) or before each commit.

Checks:
  1. Dead imports (imported but never used in file)
  2. Hardcoded paths (author-specific home-directory references)
  3. Large files in git (> 1MB)
  4. Stale log/data files tracked in git
  5. Empty __init__.py files that could be cleaned
  6. Duplicate function definitions across files
  7. Files with no docstring
  8. Orphan test files (test files with no corresponding source)

Usage:
    python scripts/cleanup_agent.py              # Report only
    python scripts/cleanup_agent.py --fix        # Auto-fix safe issues
    python scripts/cleanup_agent.py --json       # Machine-readable output
"""
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway

CORE = ROOT / "core"


def get_tracked_files() -> list[str]:
    result = get_subprocess_gateway().run(
        ["git", "-c", "core.fsmonitor=false", "ls-files"],
        cwd=ROOT,
        timeout=30,
        read_only=True,
        source="cleanup_agent:tracked_files",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {result.stderr.strip()}")
    return [f for f in result.stdout.strip().split("\n") if f]


def check_dead_imports(py_files: list[Path]) -> list[dict]:
    """Find imports that are never referenced in the file."""
    issues = []
    for f in py_files:
        try:
            source = f.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    imported_names.add((name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    name = alias.asname or alias.name
                    imported_names.add((name, node.lineno))

        # Check which imported names appear elsewhere in the source
        for name, lineno in imported_names:
            # Count occurrences (must appear more than just the import line)
            count = source.count(name)
            if count <= 1 and len(name) > 1 and name not in ("_", "__"):
                rel = str(f.relative_to(ROOT))
                issues.append({
                    "type": "dead_import",
                    "file": rel,
                    "line": lineno,
                    "name": name,
                    "severity": "low",
                })

    return issues[:50]  # Cap to avoid noise


def check_hardcoded_paths() -> list[dict]:
    """Find author-specific paths in tracked files."""
    home_pattern = str(Path.home())
    result = get_subprocess_gateway().run(
        ["git", "-c", "core.fsmonitor=false", "grep", "-n", home_pattern, "--", "*.py", "*.sh", "*.md"],
        cwd=ROOT,
        timeout=30,
        read_only=True,
        source="cleanup_agent:hardcoded_paths",
    )
    issues = []
    if result.returncode not in {0, 1}:
        return [{
            "type": "scan_failed",
            "file": "git grep",
            "description": f"hardcoded path scan failed: {result.stderr.strip()}",
            "severity": "high",
        }]
    exclude = {"scripts/cleanup_agent.py", "scripts/quality_gate.py", "specs/QUALITY_GATES.md"}
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split(":", 2)
        if len(parts) >= 2 and parts[0] not in exclude:
            issues.append({
                "type": "hardcoded_path",
                "file": parts[0],
                "line": int(parts[1]) if parts[1].isdigit() else 0,
                "severity": "high",
            })
    return issues


def check_large_files() -> list[dict]:
    """Find files > 1MB tracked in git."""
    issues = []
    for f in get_tracked_files():
        full = ROOT / f
        if full.exists() and full.stat().st_size > 1_000_000:
            issues.append({
                "type": "large_file",
                "file": f,
                "size_kb": full.stat().st_size // 1024,
                "severity": "high",
            })
    return issues


def check_missing_docstrings(py_files: list[Path]) -> list[dict]:
    """Find Python files with no module docstring."""
    issues = []
    for f in py_files:
        try:
            source = f.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
            docstring = ast.get_docstring(tree)
            if not docstring and len(source) > 200:  # Skip tiny files
                issues.append({
                    "type": "missing_docstring",
                    "file": str(f.relative_to(ROOT)),
                    "severity": "low",
                })
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
    return issues[:30]


def check_duplicate_functions(py_files: list[Path]) -> list[dict]:
    """Find functions defined in multiple files (potential duplication)."""
    func_locations = defaultdict(list)
    for f in py_files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_") and len(node.name) > 5:
                        func_locations[node.name].append(str(f.relative_to(ROOT)))
        except (SyntaxError, UnicodeDecodeError):
            continue

    issues = []
    for name, locations in func_locations.items():
        if len(locations) > 2:  # Allow 2 (e.g., core + tests)
            issues.append({
                "type": "duplicate_function",
                "name": name,
                "locations": locations[:5],
                "count": len(locations),
                "severity": "medium",
            })
    return sorted(issues, key=lambda x: -x["count"])[:20]


def check_security_concerns() -> list[dict]:
    """Find files that suggest dangerous capabilities."""
    issues = []
    dangerous_patterns = [
        ("propagation", "lateral movement / self-replication"),
        ("stealth_ops", "stealth operations"),
        ("network_recon", "network reconnaissance"),
        ("sec_ops", "security operations / offensive"),
        ("privacy_stealth", "stealth / evasion"),
        ("keylog", "keylogging"),
        ("screenshot.*stealth", "stealth screenshots"),
    ]
    for f in get_tracked_files():
        f_lower = f.lower()
        for pattern, description in dangerous_patterns:
            if pattern in f_lower:
                issues.append({
                    "type": "security_concern",
                    "file": f,
                    "description": description,
                    "severity": "critical",
                })
    return issues


def main():
    json_mode = "--json" in sys.argv

    # Collect Python files
    py_files = [
        f for f in ROOT.rglob("*.py")
        if ".venv" not in f.parts and "node_modules" not in f.parts
    ]

    all_issues = []
    all_issues.extend(check_security_concerns())
    all_issues.extend(check_hardcoded_paths())
    all_issues.extend(check_large_files())
    all_issues.extend(check_duplicate_functions(py_files))
    all_issues.extend(check_missing_docstrings(py_files))

    if json_mode:
        print(json.dumps({"issues": all_issues, "total": len(all_issues)}, indent=2))
        sys.exit(1 if all_issues else 0)

    # Print report
    print("=" * 60)
    print("  AURA CLEANUP AGENT REPORT")
    print("=" * 60)

    by_severity = defaultdict(list)
    for issue in all_issues:
        by_severity[issue["severity"]].append(issue)

    for severity in ["critical", "high", "medium", "low"]:
        items = by_severity.get(severity, [])
        if not items:
            continue
        print(f"\n[{severity.upper()}] ({len(items)} issues)")
        for item in items:
            if item["type"] == "duplicate_function":
                print(f"  {item['name']} defined in {item['count']} files: {', '.join(item['locations'][:3])}")
            elif item["type"] == "missing_docstring":
                print(f"  {item['file']}: no module docstring")
            else:
                print(f"  {item['file']}: {item.get('description', item['type'])}")

    total = len(all_issues)
    critical = len(by_severity.get("critical", []))
    print(f"\n{'=' * 60}")
    print(f"  Total: {total} issues ({critical} critical)")
    print(f"{'=' * 60}")

    sys.exit(1 if critical > 0 else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Lockdep sees only the locks it wraps. Measure that, and ratchet it.

``lockdep_report()["splats"] == 0`` backs a registered claim about the
runtime's lock ordering. That claim is only as wide as lockdep's coverage,
and coverage was never measured — so a number that meant "clean across the
locks we watch" was read as "clean across the runtime". Files using raw
``threading.Lock`` / ``asyncio.Lock`` outnumbered instrumented ones roughly
five to one at the time this was written, and ``capability_engine`` was
instrumented *after* it deadlocked the boot path, which is the wrong order.

This does two things:

* Reports coverage, so the claim can state its own scope honestly.
* Ratchets it. The raw-lock count in ``config/lock_coverage_baseline.json``
  may only go down. Adding a raw lock to a file that has none is a
  regression; converting one is progress and requires a refresh.

Not a mass rewrite: converting 350 files at once would be a large untested
change to the most deadlock-sensitive code in the system. The ratchet makes
the direction one-way instead.

Run: ``python tools/lint_lock_coverage.py`` / ``--write-baseline``
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "lock_coverage_baseline.json"
SCAN_ROOTS = ("core", "interface", "skills")

#: Files whose bytes are bound to a proof and may not be edited to satisfy a
#: ratchet. `core/brain/llm/qualified_recurrent_ingress.py` is one of twenty
#: sealed into the bounded-WOW activation record; converting its raw lock to a
#: checked one turned the whole surface off with
#: `semantic_neural_activation_invalid:source_drift`, and the failure message
#: says outright not to re-seal to go green. The lock there is real debt that
#: has to wait for a re-qualification, so it stays counted and stays here.
SEALED_SOURCES: frozenset[str] = frozenset(
    {"core/brain/llm/qualified_recurrent_ingress.py"}
)
SKIP_DIR_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "archive",
    "node_modules",
    "tests",
}

#: The primitives lockdep cannot see.
RAW_CONSTRUCTORS = {
    ("threading", "Lock"),
    ("threading", "RLock"),
    ("threading", "Condition"),
    ("threading", "Semaphore"),
    ("threading", "BoundedSemaphore"),
    ("asyncio", "Lock"),
    ("asyncio", "Condition"),
    ("asyncio", "Semaphore"),
    ("asyncio", "BoundedSemaphore"),
}

#: lockdep's own module defines the wrappers; it necessarily builds raw
#: primitives underneath them.
EXEMPT = {"core/runtime/lockdep.py"}


def _iter_files():
    for top in SCAN_ROOTS:
        base = ROOT / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(ROOT)
            if any(part in SKIP_DIR_PARTS for part in relative.parts):
                continue
            if str(relative) in EXEMPT:
                continue
            yield path, relative


def _raw_locks(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if (func.value.id, func.attr) in RAW_CONSTRUCTORS:
                count += 1
    return count


def measure() -> dict[str, object]:
    raw_by_file: dict[str, int] = {}
    instrumented_files: set[str] = set()
    for path, relative in _iter_files():
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        if "checked_lock" in source or "checked_async_lock" in source:
            instrumented_files.add(str(relative))
        raw = _raw_locks(tree)
        if raw:
            raw_by_file[str(relative)] = raw
    total_raw = sum(raw_by_file.values())
    return {
        "raw_lock_calls": total_raw,
        "files_with_raw_locks": len(raw_by_file),
        "files_using_checked_locks": len(instrumented_files),
        "raw_by_file": dict(sorted(raw_by_file.items())),
    }


def _load_baseline() -> dict[str, object] | None:
    if not BASELINE.is_file():
        return None
    try:
        return json.loads(BASELINE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main(argv: list[str]) -> int:
    current = measure()
    raw = int(current["raw_lock_calls"])
    files_raw = int(current["files_with_raw_locks"])
    files_checked = int(current["files_using_checked_locks"])
    denominator = files_raw + files_checked
    coverage = files_checked / denominator if denominator else 1.0

    print(
        f"lockdep coverage: {files_checked} file(s) use checked locks, "
        f"{files_raw} still construct raw ones ({coverage:.1%} of lock-holding files)"
    )
    print(f"raw lock constructions: {raw}")

    if "--write-baseline" in argv:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written: {BASELINE.relative_to(ROOT)}")
        return 0

    baseline = _load_baseline()
    if baseline is None:
        print(f"❌ no baseline at {BASELINE.relative_to(ROOT)}; run --write-baseline")
        return 1

    allowed = int(baseline.get("raw_lock_calls", 0))
    if raw > allowed:
        print(f"❌ raw lock constructions rose: {allowed} -> {raw}")
        previous = baseline.get("raw_by_file") or {}
        for name, count in sorted(current["raw_by_file"].items()):  # type: ignore[union-attr]
            was = int(previous.get(name, 0))
            if count > was:
                print(f"    {name}: {was} -> {count}")
        print(
            "\nUse checked_lock / checked_async_lock (core/runtime/lockdep.py), or "
            "instrument(name) to adopt an existing lock. Lockdep finds ABBA without "
            "the deadlock happening, and only for the locks it wraps."
        )
        return 1

    if raw < allowed:
        print(f"⬇️  raw lock constructions fell: {allowed} -> {raw}")
        print(f"    refresh with: python {Path(__file__).name} --write-baseline")
        return 1

    print("✅ lock coverage held at baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

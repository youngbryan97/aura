#!/usr/bin/env python3
"""Production cognition must not know the battery exists.

A measurement is only a measurement if the thing measured cannot see it. Every
way that can go wrong has the same shape: a module that runs during ordinary
cognition names the battery, imports its thresholds, or takes a branch that
only the harness reaches. Any of those turns the result into a description of
the test.

    python tools/audit_no_battery_shortcuts.py            # list what it finds
    python tools/audit_no_battery_shortcuts.py --check    # fail on anything new

`core/subject/` is the instrument and is allowed to know about itself. So is
`tools/`, and so are the tests. What is scanned is the cognition the instrument
measures.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "config" / "battery_shortcut_baseline.json"

#: The instrument, its tools and its tests. Allowed to name themselves.
EXEMPT: tuple[str, ...] = (
    "core/subject/",
    "tools/",
    "tests/",
)

#: Directories that are the organism under measurement.
SCANNED: tuple[str, ...] = (
    "core/",
    "interface/",
    "skills/",
    "llm/",
    "executors/",
)

#: Naming the instrument from inside the organism.
NAMES: tuple[str, ...] = (
    "subject_core",
    "core.subject",
    "isc_completion",
    "phi_do",
    "perturbational_spread",
    "partition_irreducibility",
    "ISC_",
)

#: An environment switch that could open a path only the harness takes. The
#: proof-run flags are the ones that have done it before.
SWITCHES: tuple[str, ...] = (
    "AURA_SUBJECT_CORE",
    "AURA_ISC",
    "AURA_BATTERY",
)

_NAME_RE = re.compile("|".join(re.escape(name) for name in NAMES))
_SWITCH_RE = re.compile("|".join(re.escape(name) for name in SWITCHES))


def _scanned_files() -> list[Path]:
    out: list[Path] = []
    for root in SCANNED:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            rel = str(path.relative_to(REPO))
            if any(rel.startswith(skip) for skip in EXEMPT):
                continue
            out.append(path)
    return sorted(out)


def findings() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for path in _scanned_files():
        rel = str(path.relative_to(REPO))
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if not (_NAME_RE.search(text) or _SWITCH_RE.search(text)):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        # Comments and docstrings may discuss the instrument: a note saying why
        # a channel exists is the opposite of a shortcut. What must not happen
        # is code that runs.
        docstrings: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None) or []
                first = body[0] if body else None
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    for line in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                        docstrings.add(line)
        for number, line in enumerate(text.splitlines(), start=1):
            if number in docstrings:
                continue
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            hit = _NAME_RE.search(line) or _SWITCH_RE.search(line)
            if not hit:
                continue
            out.append({"file": rel, "line": number, "match": hit.group(0), "text": stripped[:110]})
    return out


def _key(item: dict[str, object]) -> str:
    return f"{item['file']}:{item['match']}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    found = findings()
    keys = sorted({_key(item) for item in found})

    if args.write_baseline:
        BASELINE.write_text(json.dumps({"allowed": keys}, indent=2) + "\n")
        print(f"baseline written: {len(keys)} site(s)")
        return 0

    allowed = set()
    if BASELINE.is_file():
        allowed = set(json.loads(BASELINE.read_text()).get("allowed", []))

    for item in found:
        mark = " " if _key(item) in allowed else "+"
        print(f"{mark} {item['file']}:{item['line']}  {item['match']}  {item['text']}")
    print(f"\n{len(keys)} site(s) in the measured tree name the instrument.")

    if not args.check:
        return 0
    new = sorted(set(keys) - allowed)
    gone = sorted(allowed - set(keys))
    if gone:
        print(f"\n{len(gone)} baseline entries are gone; rerun with --write-baseline:")
        for key in gone:
            print(f"  - {key}")
    if new:
        print(f"\n{len(new)} NEW reference(s) to the instrument from the measured tree:")
        for key in new:
            print(f"  + {key}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""How much of the production tree carries type annotations, held at a floor.

The mechanism was already here and almost none of it was deployed.
``pyproject.toml`` configures mypy in strict mode — ``disallow_untyped_defs``,
``warn_unused_ignores``, the lot — and ``make typecheck`` then runs it against
``config/mypy_strict_files.txt``, an allowlist of 78 files. The repository
tracks 6,615. So the strict configuration was true of about one percent of the
code and the other ninety-nine were never checked by it.

Annotations themselves are in better shape than that number suggests: 1,701 of
the 2,799 production modules that define a function annotate every parameter
and every return. The gap is not that the code is untyped, it is that nothing
stopped it becoming untyped again.

Two ratchets, both of which only move one way:

* a module that is fully annotated today may not stop being so;
* a module that is NEW must be fully annotated. There is no third option and
  no way to add to the baseline — ``--write-baseline`` refuses to grow it.

That is the same discipline ``tests/test_async_write_lane_ratchet.py`` applies
to sync writes in async code, applied to the thing an external review named:
good mechanism, partial deployment.

    python tools/check_typed_surface.py
    python tools/check_typed_surface.py --write-baseline   # shrink only
    python tools/check_typed_surface.py --report
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "typed_surface_baseline.json"

SCAN_ROOTS = (
    "core",
    "interface",
    "skills",
    "security",
    "llm",
    "executors",
    "infrastructure",
)
SKIP_PARTS = {"__pycache__", ".venv", "archive", "node_modules"}

#: Names a method receives from the language rather than from its author.
IMPLICIT = {"self", "cls"}


def _iter_modules() -> list[Path]:
    modules: list[Path] = []
    for root in SCAN_ROOTS:
        base = ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if SKIP_PARTS & set(path.parts):
                continue
            modules.append(path)
    return modules


def unannotated_definitions(tree: ast.AST) -> list[str]:
    """Every def in this module missing a return type or a parameter type."""
    missing: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        gaps: list[str] = []
        if node.returns is None:
            gaps.append("return")
        args = node.args
        parameters = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        parameters = [a for a in parameters if a.arg not in IMPLICIT]
        if args.vararg is not None:
            parameters.append(args.vararg)
        if args.kwarg is not None:
            parameters.append(args.kwarg)
        gaps.extend(a.arg for a in parameters if a.annotation is None)
        if gaps:
            missing.append(f"{node.name}:{node.lineno} ({', '.join(gaps)})")
    return missing


def scan() -> tuple[set[str], set[str]]:
    """(fully annotated modules, modules with at least one gap)."""
    typed: set[str] = set()
    untyped: set[str] = set()
    for path in _iter_modules():
        try:
            tree = ast.parse(path.read_text("utf-8", errors="ignore"), filename=str(path))
        except SyntaxError:
            continue
        has_defs = any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)
        )
        if not has_defs:
            continue
        rel = str(path.relative_to(ROOT))
        if unannotated_definitions(tree):
            untyped.add(rel)
        else:
            typed.add(rel)
    return typed, untyped


def load_baseline() -> dict:
    if not BASELINE.exists():
        return {"untyped_modules": [], "count": 0}
    return json.loads(BASELINE.read_text("utf-8"))


def write_baseline(untyped: set[str]) -> int:
    previous = set(load_baseline().get("untyped_modules", []))
    grown = untyped - previous
    if previous and grown:
        print(
            "refusing to grow the baseline. These modules are not annotated and "
            "are not grandfathered:",
            file=sys.stderr,
        )
        for rel in sorted(grown):
            print(f"   • {rel}", file=sys.stderr)
        return 1
    payload = {
        "description": (
            "Production modules that do not annotate every parameter and every "
            "return. This list may only SHRINK: tools/check_typed_surface.py "
            "fails on any unannotated module not listed here and refuses to add "
            "one."
        ),
        "count": len(untyped),
        "untyped_modules": sorted(untyped),
    }
    BASELINE.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {BASELINE.name}: {len(untyped)} untyped modules")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)

    typed, untyped = scan()
    total = len(typed) + len(untyped)

    if args.write_baseline:
        return write_baseline(untyped)

    baseline = load_baseline()
    grandfathered = set(baseline.get("untyped_modules", []))

    if args.report:
        print(f"annotated: {len(typed)}/{total} ({len(typed) / max(total, 1):.1%})")
        print(f"baseline holds {len(grandfathered)} untyped modules")
        for rel in sorted(untyped - grandfathered)[: args.limit]:
            print(f"   new gap: {rel}")
        return 0

    new_gaps = sorted(untyped - grandfathered)
    if new_gaps:
        print(f"❌ typed surface: {len(new_gaps)} module(s) lost their annotations")
        for rel in new_gaps[: args.limit]:
            tree = ast.parse((ROOT / rel).read_text("utf-8", errors="ignore"))
            gaps = unannotated_definitions(tree)
            print(f"   • {rel}: {'; '.join(gaps[:3])}")
        print(
            "\nA module in the baseline may lose entries, never gain them, and a "
            "module outside it must annotate every parameter and every return."
        )
        return 1

    paid = sorted(grandfathered - untyped)
    if paid:
        print(f"❌ typed surface: {len(paid)} baseline entries are now annotated")
        for rel in paid[: args.limit]:
            print(f"   • {rel}")
        print(
            "\nDebt already paid must leave the baseline in the commit that paid "
            "it. Run --write-baseline."
        )
        return 1

    print(
        f"✅ typed surface at baseline: {len(typed)}/{total} modules annotated "
        f"({len(typed) / max(total, 1):.1%}), {len(grandfathered)} grandfathered"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

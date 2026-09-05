#!/usr/bin/env python3
"""Privileged convergence surfaces, measured — the complexity that matters.

The complexity criticism worth acting on is not size. A sufficiently rich
mind probably requires enormous heterogeneity, and a neuron is not fifteen
thousand lines long because minds are simple. What makes software complexity
dangerous is verification surface: with n components and invariants that hold
locally, checking stays about O(n); with arbitrary cross-links, O(n²)
relationships become possible; with higher-order interactions, coalitions of
2^n become relevant.

Biological complexity is distributed. What this measures is whether Aura's is:
a module that many things reach INTO and that reaches out to many things is a
place where an invariant stops being local, and every one of them multiplies
what has to be checked to know anything.

    paths = (how many modules import this) × (how many modules this imports)

That product is the number of importer-importee pairs the module sits
between, and it is what has to be understood to know that an invariant
holds. But it is large for two innocent shapes as well: a utility everything
imports and that imports nothing much, and a composition root that imports
everything and is imported by nothing much. Neither is where n² lives.

What names the dangerous shape is BOTH being large, so the ratcheted number
is the harmonic mean of the two:

    surface = 2 · in · out / (in + out)

which is large only when neither side is small, needs no threshold, and puts
the engine that fifty modules reach into and that reaches sixty-four above
the error reporter a thousand modules import and that imports eleven. The
product is reported beside it, because how many paths run through a place is
worth knowing even where the place is a utility.

Ratcheted like the others: the checked-in total may shrink and may not grow.
`chat.py` at twenty-nine thousand lines and a leaf module of the same size
are the same number to a size ratchet and nothing alike to a reviewer; this
is the number that tells them apart.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BASELINE = HERE / "config" / "convergence_surface_baseline.json"

#: Trees whose modules count. Tests and tools are not the organism.
WATCHED = ("core", "interface", "skills", "llm", "executors", "security")

#: How many of the worst surfaces the baseline pins. Enough that the shape of
#: the problem is visible and few enough that the file can be read.
HOW_MANY_KEPT = 40


def _module_of(path: Path) -> str:
    return str(path.relative_to(HERE).with_suffix("")).replace("/", ".")


def _imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (OSError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
    return {one for one in found if one.split(".")[0] in WATCHED}


def measure() -> dict[str, object]:
    """Every watched module, and what it reaches and is reached by."""

    out_edges: dict[str, set[str]] = {}
    for where in WATCHED:
        base = HERE / where
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            out_edges[_module_of(path)] = _imports(path)

    known = set(out_edges)
    in_edges: dict[str, set[str]] = defaultdict(set)
    for module, reaches in out_edges.items():
        for target in reaches:
            # An import of a package resolves to every module under it that
            # exists, which is what makes a package-level import a reach into
            # all of it rather than into nothing.
            if target in known:
                in_edges[target].add(module)
            elif f"{target}.__init__" in known:
                in_edges[f"{target}.__init__"].add(module)

    surfaces = []
    for module, reaches in out_edges.items():
        fan_out = len({one for one in reaches if one in known})
        fan_in = len(in_edges.get(module, ()))
        both = (
            round(2.0 * fan_in * fan_out / (fan_in + fan_out), 2)
            if (fan_in and fan_out)
            else 0.0
        )
        surfaces.append(
            {
                "module": module,
                "fan_in": fan_in,
                "fan_out": fan_out,
                "paths": fan_in * fan_out,
                "surface": both,
            }
        )
    surfaces.sort(key=lambda row: (-float(row["surface"]), str(row["module"])))
    worst = surfaces[:HOW_MANY_KEPT]
    return {
        "modules": len(out_edges),
        "total_paths": sum(int(one["paths"]) for one in surfaces),
        "worst_total": round(sum(float(one["surface"]) for one in worst), 2),
        "worst": worst,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    now = measure()
    print(
        f"convergence surface: {now['modules']} modules, "
        f"{now['total_paths']:,} import paths run through them; "
        f"the worst {HOW_MANY_KEPT} carry a surface of {now['worst_total']:,}"
    )
    for one in now["worst"][: args.show]:  # type: ignore[index]
        print(
            f"   {one['surface']:>8}  {one['module']}  "
            f"(reached by {one['fan_in']}, reaches {one['fan_out']}, "
            f"{one['paths']:,} paths)"
        )

    if args.write_baseline:
        if BASELINE.exists():
            was = json.loads(BASELINE.read_text())
            if float(now["worst_total"]) > float(was.get("worst_total", 0)):
                print(
                    f"refusing to record a larger total: "
                    f"{was['worst_total']:,} -> {now['worst_total']:,}"
                )
                return 1
        BASELINE.write_text(json.dumps(now, indent=2), encoding="utf-8")
        print(f"baseline written: {BASELINE.relative_to(HERE)}")
        return 0

    if not BASELINE.exists():
        print("no baseline; write one with --write-baseline")
        return 1
    was = json.loads(BASELINE.read_text())
    if float(now["worst_total"]) > float(was["worst_total"]):
        print(
            f"❌ convergence surface grew: {was['worst_total']:,} -> "
            f"{now['worst_total']:,}"
        )
        print(
            "A module many things reach into that also reaches many things is "
            "where an invariant stops being local. Reduce one, or move what "
            "made it grow behind a narrower seam."
        )
        return 1
    if float(now["worst_total"]) < float(was["worst_total"]):
        print(
            f"⬇️  convergence surface fell: {was['worst_total']:,} -> "
            f"{now['worst_total']:,}\n    refresh with --write-baseline"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

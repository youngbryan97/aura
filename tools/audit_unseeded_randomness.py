#!/usr/bin/env python3
"""Every generator in the tree that nothing can seed and nothing can rewind.

A paired experiment subtracts one arm from another. That subtraction is a
measurement only if the two arms would have been identical without the
intervention, and a private generator breaks that twice over: `random.seed()`
does not reach it, so a campaign is not reproducible; and a snapshot of the
process generators cannot put it back, so two arms started from one state draw
different numbers and the difference lands in the floor under every edge.

The synaptic cleft held one. Release through it is probabilistic, every
interiority faculty publishes through it and the layer runs on every turn, so
unseeded randomness sat in the cognitive path where nothing could see it.

    python tools/audit_unseeded_randomness.py            # list them
    python tools/audit_unseeded_randomness.py --check    # fail on a new one

What counts as seeded: a generator built with an argument (a seed, or a
generator passed in by the caller), or the process-wide `random` / `np.random`
modules, which a harness can save and restore. What counts as unseeded is a
constructor called with nothing.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "config" / "unseeded_randomness_baseline.json"

#: Constructors that return a generator of their own.
MAKERS: dict[str, str] = {
    "random.Random": "random",
    "random.SystemRandom": "random",
    "numpy.random.default_rng": "numpy",
    "np.random.default_rng": "numpy",
    "numpy.random.RandomState": "numpy",
    "np.random.RandomState": "numpy",
    "numpy.random.Generator": "numpy",
    "np.random.Generator": "numpy",
}

#: Where a private generator is the point rather than a leak. A sandbox that
#: draws its own scenarios is not in the organism's causal path.
EXEMPT_DIRS: tuple[str, ...] = ("core/engineering/",)


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def findings(roots: tuple[str, ...] = ("core",)) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for root in roots:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = str(path.relative_to(REPO))
            if any(rel.startswith(skip) for skip in EXEMPT_DIRS):
                continue
            try:
                tree = ast.parse(path.read_text(errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _dotted(node.func)
                if name not in MAKERS:
                    continue
                if node.args or node.keywords:
                    continue  # a seed, or a generator handed in
                out.append(
                    {
                        "file": rel,
                        "line": node.lineno,
                        "call": f"{name}()",
                        "kind": MAKERS[name],
                    }
                )
    return out


def _key(item: dict[str, object]) -> str:
    return f"{item['file']}:{item['call']}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail on anything new")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    found = findings()
    keys = sorted({_key(item) for item in found})

    if args.write_baseline:
        BASELINE.write_text(json.dumps({"allowed": keys}, indent=2) + "\n")
        print(f"baseline written: {len(keys)} sites")
        return 0

    allowed = set()
    if BASELINE.is_file():
        allowed = set(json.loads(BASELINE.read_text()).get("allowed", []))

    for item in found:
        mark = " " if _key(item) in allowed else "+"
        print(f"{mark} {item['file']}:{item['line']}  {item['call']}")
    print(f"\n{len(keys)} site(s) hold a generator nothing can seed or rewind.")

    if not args.check:
        return 0
    new = sorted(set(keys) - allowed)
    gone = sorted(allowed - set(keys))
    if gone:
        print(f"\n{len(gone)} baseline entries are fixed; rerun with --write-baseline:")
        for key in gone:
            print(f"  - {key}")
    if new:
        print(f"\n{len(new)} NEW unseeded generator(s):")
        for key in new:
            print(f"  + {key}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""A package reaching into another for a name it never reads.

The complexity that matters is verification surface, and a cross-package edge
is the unit of it: every one is a pair of packages whose invariants can now
interact. So an edge that exists because of a single import whose name is
never used is the cheapest kind of coupling to remove — nothing is using it
and nothing will notice.

Five were found this way, and they are gone. What this holds is that no more
appear.

Two things it deliberately does not flag, because both are imports doing work
without binding a name anybody reads:

* an import inside a ``try`` that guards an optional subsystem — the import
  IS the check, separating "not registered" from "not installed", and the
  name is unused on purpose;
* a re-export a package publishes for its callers.

Both carry ``# noqa: F401``, which is how the rest of this tree already says
"the name is unused and the import is not".
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CORE = HERE / "core"


def _edges() -> dict[tuple[str, str], list[tuple[Path, int, str, list[str], str]]]:
    found: dict[tuple[str, str], list] = defaultdict(list)
    for path in sorted(CORE.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        parts = path.relative_to(CORE).parts
        if len(parts) < 2:
            continue
        source = parts[0]
        try:
            body = path.read_text(errors="replace")
            tree = ast.parse(body)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            module, names = "", []
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [one.asname or one.name for one in node.names]
            elif isinstance(node, ast.Import):
                module = node.names[0].name if node.names else ""
                names = [
                    (one.asname or one.name.split(".")[0]) for one in node.names
                ]
            bits = module.split(".")
            if len(bits) >= 2 and bits[0] == "core" and bits[1] != source:
                found[(source, bits[1])].append(
                    (path, node.lineno, module, names, body)
                )
    return found


def dead() -> list[dict[str, object]]:
    """Edges resting on one import whose every name is never read."""

    out: list[dict[str, object]] = []
    for (source, target), where in _edges().items():
        if len(where) != 1:
            continue
        path, line, module, names, body = where[0]
        lines = body.splitlines()
        if line - 1 < len(lines) and "noqa" in lines[line - 1]:
            continue
        used = False
        for name in names:
            for number, text in enumerate(lines, 1):
                if number != line and re.search(rf"\b{re.escape(name)}\b", text):
                    used = True
                    break
            if used:
                break
        if not used:
            out.append(
                {
                    "edge": f"core.{source} -> core.{target}",
                    "file": str(path.relative_to(HERE)),
                    "line": line,
                    "imports": module,
                    "names": names,
                }
            )
    return sorted(out, key=lambda row: str(row["edge"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    found = dead()
    if not found:
        print("no cross-package edge rests on an import nobody reads")
        return 0
    print(f"❌ {len(found)} cross-package edge(s) rest on an import nobody reads")
    for one in found:
        print(
            f"   {one['edge']}\n      {one['file']}:{one['line']}  "
            f"{one['imports']} ({', '.join(one['names'])})"
        )
    print(
        "\nRemove the import, or mark it `# noqa: F401` if it is an "
        "availability probe or a re-export — an import doing work without "
        "binding a name anybody reads."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

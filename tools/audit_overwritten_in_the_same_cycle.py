#!/usr/bin/env python3
"""A field written and then written again, without reading what was there.

The defect this looks for has turned up four times in the subject core alone:
a phase computes something, writes it onto the state, and a later statement in
the same call recomputes the field from scratch. Whatever the first write meant
is gone, and nothing fails — the field holds a plausible number, the tests that
read it pass, and the channel that fed the first write is dead.

`affect.curiosity` was one: the lifetime state blended it toward how
unprecedented the moment was, and the derived-metrics step recomputed it from
two emotion channels. `affect.valence` was another, with the substrate's third
of her felt state going the same way.

    python tools/audit_overwritten_in_the_same_cycle.py            # list them
    python tools/audit_overwritten_in_the_same_cycle.py --check    # fail on new

What counts as an overwrite: the same attribute path assigned twice inside one
function, where the second assignment does not mention the path. A read-modify-
write mentions it and is not flagged — that is a field carrying, which is the
opposite of the defect.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "config" / "same_cycle_overwrite_baseline.json"

#: Where cognition writes to the state it is carrying.
SCANNED: tuple[str, ...] = ("core/phases/", "core/consciousness/")

#: The methods a turn enters a phase through. What they call, in order, is the
#: cycle a field can be overwritten inside.
ENTRY_POINTS: tuple[str, ...] = ("execute", "run", "tick", "_tick", "update")

#: Roots worth watching. A local variable reassigned twice is ordinary code and
#: an object's own status flag set on two branches of a try/except is ordinary
#: code; a field on the state the next phase will read is not.
ROOTS: tuple[str, ...] = ("state", "affect", "cognition", "identity", "mot", "new_state")


def _path_of(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or not parts:
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _mentions(node: ast.AST, path: str) -> bool:
    leaf = path.rsplit(".", 1)[-1]
    for inner in ast.walk(node):
        if isinstance(inner, ast.Attribute) and inner.attr == leaf:
            return True
        if isinstance(inner, ast.Name) and inner.id == leaf:
            return True
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str) and leaf in inner.value:
            return True
    return False


def _blocks(node: ast.AST):
    """Every statement list in a function, one at a time.

    Two assignments in the same list both run, in order. Two in different lists
    may be alternatives.
    """
    for field in ("body", "orelse", "finalbody"):
        block = getattr(node, field, None)
        if isinstance(block, list) and block:
            yield block
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield from _blocks(child)


def _functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _writes_of(function: ast.AST) -> dict[str, tuple[int, bool]]:
    """Every field this function assigns, and whether the value mentions it.

    A read-modify-write mentions the field: that is a value carrying, which is
    the opposite of the defect. A write that does not mention it replaces
    whatever was there.
    """
    out: dict[str, tuple[int, bool]] = {}
    for node in ast.walk(function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not function:
            continue
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Attribute):
            continue
        dotted = _path_of(target)
        if not dotted or dotted.split(".")[0] not in ROOTS:
            continue
        seen = out.get(dotted)
        out[dotted] = (node.lineno, False if seen is None else seen[1])
    # Does the helper read the field anywhere? A blend through a local —
    # `blended = (1 - w) * affect.curiosity + w * novelty`, then
    # `affect.curiosity = blended` — carries what was there just as much as a
    # read-modify-write on one line, and a rule that only looks at the
    # assignment's own right-hand side calls it a replacement.
    targets = {
        id(node.targets[0])
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
    }
    for dotted in list(out):
        line, _ = out[dotted]
        reads = any(
            isinstance(node, ast.Attribute)
            and id(node) not in targets
            and _path_of(node) == dotted
            for node in ast.walk(function)
        )
        out[dotted] = (line, reads)
    return out


def _called_in_order(function: ast.AST) -> list[tuple[str, int]]:
    """The `self.<name>(...)` calls this function makes, in source order."""
    out: list[tuple[str, int]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "self":
                out.append((func.attr, node.lineno))
    out.sort(key=lambda pair: pair[1])
    return out


def findings() -> list[dict[str, object]]:
    """Fields one helper writes and a later helper in the same call replaces.

    The historical defects were never two statements side by side: they were
    `_derive_metrics` recomputing `affect.curiosity` from two emotion channels
    while `_advance_lifetime`, called three steps later, had blended it toward
    how unprecedented the moment was. So the walk is over what a phase's
    entry point calls, in order, and what each of those writes.
    """
    out: list[dict[str, object]] = []
    for root in SCANNED:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = str(path.relative_to(REPO))
            try:
                tree = ast.parse(path.read_text(errors="replace"))
            except SyntaxError:
                continue
            for klass in ast.walk(tree):
                if not isinstance(klass, ast.ClassDef):
                    continue
                methods = {
                    node.name: node
                    for node in klass.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                for entry_name in ENTRY_POINTS:
                    entry = methods.get(entry_name)
                    if entry is None:
                        continue
                    written: dict[str, tuple[str, int]] = {}
                    for called, line in _called_in_order(entry):
                        helper = methods.get(called)
                        if helper is None:
                            continue
                        for field, (where, carries) in _writes_of(helper).items():
                            earlier = written.get(field)
                            if earlier is not None and earlier[0] != called and not carries:
                                out.append(
                                    {
                                        "file": rel,
                                        "function": f"{klass.name}.{entry_name}",
                                        "field": field,
                                        "written_by": earlier[0],
                                        "written": earlier[1],
                                        "overwritten_by": called,
                                        "overwritten": line,
                                    }
                                )
                            written[field] = (called, where)
    return out


def _key(item: dict[str, object]) -> str:
    return f"{item['file']}:{item['function']}:{item['field']}"


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
        print(
            f"{mark} {item['file']}  {item['function']}: "
            f"{item['written_by']} writes {item['field']} (line {item['written']}), "
            f"{item['overwritten_by']} replaces it (line {item['overwritten']})"
        )
    print(f"\n{len(keys)} field(s) written and then replaced in the same call.")

    if not args.check:
        return 0
    new = sorted(set(keys) - allowed)
    gone = sorted(allowed - set(keys))
    if gone:
        print(f"\n{len(gone)} baseline entries are fixed; rerun with --write-baseline:")
        for key in gone:
            print(f"  - {key}")
    if new:
        print(f"\n{len(new)} NEW same-cycle overwrite(s):")
        for key in new:
            print(f"  + {key}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

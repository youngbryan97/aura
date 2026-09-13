#!/usr/bin/env python3
"""A result built into a local dictionary that nothing ever reads.

The half-wired shape this looks for: a function assembles a mapping, fills it
in, and drops it. The keys look like a channel — `results["valence"] = ...`,
`out["consumers"] = ...` — and the mapping is never returned, never passed to
anything, never read back. Whatever computed the values ran for nothing, and
the tests that cover the computation all pass.

    python tools/audit_return_paths_into_local_dicts.py            # list them
    python tools/audit_return_paths_into_local_dicts.py --check    # fail on new

What counts: a local name bound to a dict display, `dict(...)`, or a dict
comprehension, written afterwards by subscript assignment or `.update` /
`.setdefault` / `.pop`, and then never mentioned in a read position — not
returned, not yielded, not an argument, not subscripted for its value, not
iterated, not in an f-string. A name that leaves the function by any of those
routes is wired, whatever the caller then does with it.

Deliberate accumulators are exempt by shape rather than by name: a dict written
inside a loop whose only purpose is the loop's own bookkeeping still has to
leave the function to count as wired, so nothing is exempt on the grounds that
it looked busy.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "config" / "local_dict_return_path_baseline.json"

#: Where a dropped return path costs something. Cognition carries state between
#: phases; the subject core is what measures it.
SCANNED: tuple[str, ...] = (
    "core/phases/",
    "core/consciousness/",
    "core/subject/",
    "core/agency/",
    "core/affect/",
)

#: A mapping this small is a lookup table, not a result being assembled.
MIN_KEYS: int = 2


def _is_dict_source(node: ast.AST) -> bool:
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
    )


def _functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _own_nodes(function: ast.AST):
    """Every node in this function that does not belong to a nested one."""
    nested = {
        id(inner)
        for node in ast.walk(function)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        and node is not function
        for inner in ast.walk(node)
    }
    for node in ast.walk(function):
        if id(node) not in nested or node is function:
            yield node


def _dict_locals(function: ast.AST) -> dict[str, int]:
    """Local names bound once to a mapping, with the line they were bound."""
    bound: dict[str, int] = {}
    rebound: set[str] = set()
    for node in _own_nodes(function):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None and _is_dict_source(node.value):
                bound.setdefault(node.target.id, node.lineno)
            continue
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if _is_dict_source(node.value):
            bound.setdefault(target.id, node.lineno)
        else:
            rebound.add(target.id)
    return {name: line for name, line in bound.items() if name not in rebound}


def _written(function: ast.AST, name: str) -> list[str]:
    """The keys written into `name`, in source order."""
    keys: list[str] = []
    for node in _own_nodes(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
            ):
                slot = target.slice
                keys.append(
                    slot.value if isinstance(slot, ast.Constant) else "<computed>"
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
            and node.func.attr in {"update", "setdefault"}
        ):
            keys.append(f"<{node.func.attr}>")
    return keys


def _read(function: ast.AST, name: str) -> bool:
    """Whether the mapping is ever used for its contents or handed onward.

    Over the whole subtree, nested functions included. A closure that reads the
    name reads it: `turn_once` builds `env` and the `capture` defined inside it
    passes it to every reading of the state, which is as wired as a return.
    """
    for node in ast.walk(function):
        if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
            if node.value is not None and _mentions(node.value, name):
                return True
        if isinstance(node, ast.Call):
            for argument in list(node.args) + [kw.value for kw in node.keywords]:
                if _mentions(argument, name):
                    return True
            # `name.keys()`, `name.get(...)`, `len(name)` — any method but the
            # three that only write.
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == name
                and func.attr not in {"update", "setdefault", "clear"}
            ):
                return True
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == name and isinstance(node.ctx, ast.Load):
                return True
        if isinstance(node, (ast.For, ast.comprehension)):
            source = node.iter
            if _mentions(source, name):
                return True
        if isinstance(node, ast.Compare) and _mentions(node, name):
            return True
        if isinstance(node, ast.FormattedValue) and _mentions(node.value, name):
            return True
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            same_name = isinstance(target, ast.Name) and target.id == name
            into_self = isinstance(target, (ast.Attribute, ast.Subscript))
            if not same_name and into_self and _mentions(node.value, name):
                return True
    return False


def _mentions(node: ast.AST, name: str) -> bool:
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id == name and isinstance(inner.ctx, ast.Load):
            return True
    return False


def findings() -> list[dict[str, object]]:
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
            for function in _functions(tree):
                for name, line in _dict_locals(function).items():
                    keys = _written(function, name)
                    if len(keys) < MIN_KEYS:
                        continue
                    if _read(function, name):
                        continue
                    out.append(
                        {
                            "file": rel,
                            "function": function.name,
                            "name": name,
                            "line": line,
                            "keys": keys[:6],
                            "key_count": len(keys),
                        }
                    )
    return out


def _key(item: dict[str, object]) -> str:
    return f"{item['file']}:{item['function']}:{item['name']}"


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

    allowed: set[str] = set()
    if BASELINE.is_file():
        allowed = set(json.loads(BASELINE.read_text()).get("allowed", []))

    for item in found:
        mark = " " if _key(item) in allowed else "+"
        print(
            f"{mark} {item['file']}:{item['line']}  {item['function']}: "
            f"{item['name']} takes {item['key_count']} write(s) "
            f"({', '.join(str(k) for k in item['keys'])}) and is never read"
        )
    print(f"\n{len(keys)} return path(s) that end in a local dictionary.")

    if not args.check:
        return 0
    new = sorted(set(keys) - allowed)
    gone = sorted(allowed - set(keys))
    if gone:
        print(f"\n{len(gone)} baseline entries are fixed; rerun with --write-baseline:")
        for key in gone:
            print(f"  - {key}")
    if new:
        print(f"\n{len(new)} NEW dropped return path(s):")
        for key in new:
            print(f"  + {key}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

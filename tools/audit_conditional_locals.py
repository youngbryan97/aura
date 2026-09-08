#!/usr/bin/env python3
"""Locals bound on one branch and read on all of them.

`ExecutiveClosurePhase` bound `phi_estimate` inside `if reported_phi is not
None` and read it unconditionally three lines below. When the closed loop had
nothing to report — the case the branch was written for, and the common one —
the phase raised `UnboundLocalError` and died on every turn. Ruff cannot see it:
the name is a local, it is assigned somewhere in the function, and every rule
that looks at names is satisfied.

This walks each function and looks for a name whose only assignments are inside
an `if` with no `else` binding it, and which is then read outside that
statement. It skips branches that return or raise, names declared `global`, and
loop and `with` targets, which between them were a hundred and fifty of the
first run's hits.

What is left is still mostly noise: about one in ten across the phases and the
consciousness layer, the rest being names reassigned unconditionally before the
read, or bound by a guard the walk cannot follow. So it reports and never
fails, and every hit needs a person. It earns its place by finding the one that
was live: run it against `949e5f441` and it names
`executive_closure.integrate` and nothing else.

    tools/audit_conditional_locals.py core/phases core/consciousness
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _assigned(node: ast.AST, *, direct_only: bool = False) -> set[str]:
    """Local names this subtree binds by assignment.

    Loop and `with` targets are left out on purpose. A name bound by iteration
    and read after the loop is the ordinary "the loop may not have run" case,
    which is a different and much noisier question; including them buried the
    one this was written for under a hundred and fifty of them.

    ``direct_only`` restricts to statements at the top level of the body, so a
    name assigned inside a nested loop or branch does not count as bound by the
    branch itself.
    """
    names: set[str] = set()
    body = getattr(node, "body", [node])
    statements = body if direct_only else list(ast.walk(node))
    for child in statements:
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(child, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(child.target, ast.Name):
                names.add(child.target.id)
    return names


def _read(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _returns_or_raises(body: list[ast.stmt]) -> bool:
    """A branch that leaves the function cannot fall through to the read."""
    return any(isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break)) for stmt in body)


def _relative(path: Path) -> Path:
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def suspects(tree: ast.AST, path: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound_before: set[str] = {arg.arg for arg in function.args.args}
        bound_before |= {arg.arg for arg in function.args.kwonlyargs}
        # A module-level name assigned under `if _INSTANCE is None` is the
        # ordinary singleton accessor, and it is bound whether or not the
        # branch runs. Declared globals are not locals and never raise here.
        for child in ast.walk(function):
            if isinstance(child, (ast.Global, ast.Nonlocal)):
                bound_before |= set(child.names)
        for statement in function.body:
            if not isinstance(statement, ast.If):
                # Anywhere in the subtree, not only its top level: a name bound
                # inside an earlier branch is bound by the time this one runs,
                # and treating it as unbound reported a hundred and thirty
                # false alarms.
                bound_before |= _assigned(statement)
                continue
            then = _assigned(ast.Module(body=statement.body, type_ignores=[]), direct_only=True)
            other = _assigned(ast.Module(body=statement.orelse, type_ignores=[]))
            # Only names the `if` binds and the `else` does not, and which were
            # not already bound before the statement.
            risky = then - other - bound_before
            if _returns_or_raises(statement.body):
                # The branch leaves the function, so nothing it bound can be
                # read by the code after it. This was most of the noise: a
                # self-contained branch that assigns, uses and returns, next to
                # another that does the same with the same name.
                risky = set()
            elif statement.orelse and _returns_or_raises(statement.orelse):
                risky = set()
            bound_before |= then & other
            if not risky:
                bound_before |= then
                continue
            after = function.body[function.body.index(statement) + 1 :]
            reads = _read(ast.Module(body=after, type_ignores=[])) if after else set()
            for name in sorted(risky & reads):
                found.append(
                    {
                        "file": str(_relative(path)),
                        "function": function.name,
                        "name": name,
                        "line": statement.lineno,
                    }
                )
            bound_before |= then
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=["core"], type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    hits: list[dict[str, Any]] = []
    for root in args.roots:
        base = root if root.is_absolute() else REPO / root
        for module in sorted(base.rglob("*.py")):
            try:
                tree = ast.parse(module.read_text(encoding="utf-8", errors="ignore"))
            except (OSError, SyntaxError, ValueError):
                continue
            hits.extend(suspects(tree, module))

    if not args.quiet:
        print(f"{len(hits)} locals bound on one branch and read after it")
        for hit in hits:
            print(f"  {hit['file']}:{hit['line']} {hit['function']}() -> {hit['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A definition written twice in one file, where only the last one runs.

`core/skills/code_repl.py` held three copies of `_without_a_path_preamble_for`.
The first was the fix — it walked the whole syntax tree, so it found the
preamble a model had written inside a try block. The two below it were older
and only read the top level. Python keeps the last definition, so the fix was
dead the moment it was written above the copies it replaced, its tests failed,
and the live turn kept dying on the line the fix removed.

That shape is invisible in review: each copy reads correctly on its own, the
diff shows a function being added, and nothing warns. It is only visible by
counting.

Class and function definitions at module level, and methods within one class.
Two shapes redefine a name on purpose and are not counted: a conditional
definition (inside ``if``/``try``), where defining the name two ways is the
point, and a decorator that extends the name it repeats — ``@x.setter``,
``@x.getter``, ``@x.deleter``, ``@x.register``, ``@overload``. Those are one
definition wearing several hats.
"""
from __future__ import annotations

import argparse
import ast
import collections
import pathlib
import sys

__all__ = ["shadowed_in", "scan", "main"]

ROOTS = ("core", "interface", "skills", "llm", "executors", "tools", "security")


def shadowed_in(source: str) -> list[tuple[str, list[int]]]:
    """Names defined more than once at one level of this file."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found: list[tuple[str, list[int]]] = []

    def _at_one_level(body: list[ast.stmt], where: str) -> None:
        seen: dict[str, list[int]] = collections.defaultdict(list)
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if _extends_the_name(node):
                    continue
                seen[node.name].append(node.lineno)
        for name, lines in seen.items():
            if len(lines) > 1:
                found.append((f"{where}{name}", lines))

    _at_one_level(tree.body, "")
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            _at_one_level(node.body, f"{node.name}.")
    return found


_EXTENDS = frozenset({"setter", "getter", "deleter", "register"})
_ALSO_DELIBERATE = frozenset({"overload", "singledispatchmethod", "singledispatch"})


def _extends_the_name(node: ast.stmt) -> bool:
    """Whether a decorator says this repetition extends the name it repeats."""
    for decorator in getattr(node, "decorator_list", ()):
        if isinstance(decorator, ast.Attribute) and decorator.attr in _EXTENDS:
            return True
        if isinstance(decorator, ast.Name) and decorator.id in _ALSO_DELIBERATE:
            return True
        if (
            isinstance(decorator, ast.Attribute)
            and decorator.attr in _ALSO_DELIBERATE
        ):
            return True
    return False


def scan(repo: pathlib.Path) -> list[str]:
    """Every shadowed definition in the tree, as readable lines."""
    out: list[str] = []
    for name in ROOTS:
        base = repo / name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            source = path.read_text(encoding="utf-8", errors="ignore")
            for who, lines in shadowed_in(source):
                shown = ", ".join(str(n) for n in lines)
                out.append(
                    f"{path.relative_to(repo)}: {who} defined {len(lines)} times "
                    f"(lines {shown}) — only the last one runs"
                )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    args = parser.parse_args(argv)
    findings = scan(pathlib.Path(args.repo).resolve())
    for line in findings:
        print(line)
    if findings:
        print(f"\n{len(findings)} shadowed definition(s).")
        return 1
    print("No shadowed definitions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

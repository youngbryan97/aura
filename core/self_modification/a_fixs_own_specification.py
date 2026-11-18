"""What a fix wrote to check itself, carried to the code before the fix, without the fix.

A defect from her history is the code before a fix, tested by the tests that came with the fix. Her invariants live
beside the code they protect, so a fix that adds one adds its check to the file it changes, and the tests call that
check by name. Before the fix the name does not exist: the test dies on its import, runs none of the code, and the
search has nothing to work on. Measured 18 November 2026: 7 of the 29 defects with no place to repair were this, the
invariant complete but for the file it was written in.

An invariant is specification, not repair. It is carried only when everything it reads exists before the fix, so
nothing of the fix comes with it, and it is kept out of the code under repair, so no candidate can mend the defect by
weakening its check.
"""
from __future__ import annotations

import ast
import builtins
from collections.abc import Collection

__all__ = ["injected", "module_of", "specification_in"]


def _an_invariant(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        called = decorator.func if isinstance(decorator, ast.Call) else decorator
        if getattr(called, "id", getattr(called, "attr", "")) == "invariant":
            return True
    return False


def _bound_by_a_statement(statement: ast.stmt) -> set[str]:
    """The names a top-level statement gives the module."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {statement.name}
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return {(alias.asname or alias.name).split(".")[0] for alias in statement.names}
    return {node.id for node in ast.walk(statement) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}


def _what_it_reads(function: ast.AST) -> set[str]:
    """The names a function reads from outside itself."""
    read, bound = set(), set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            (read if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not function:
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return read - bound - set(dir(builtins))


def _source_of(node: ast.AST, lines: list[str]) -> str:
    first = min([node.lineno, *(decorator.lineno for decorator in getattr(node, "decorator_list", ()))])
    return "\n".join(lines[first - 1:node.end_lineno])


def specification_in(before: str, after: str, named: Collection[str]) -> tuple[str, tuple[str, ...]]:
    """The invariants ``after`` adds to ``before`` that the tests name and that read nothing new, as source with the
    imports they need, and their names; ("", ()) when there are none."""
    try:
        old, new = ast.parse(before), ast.parse(after)
    except SyntaxError:
        return "", ()
    there = {name for statement in old.body for name in _bound_by_a_statement(statement)}
    imports = {name: statement for statement in new.body if isinstance(statement, (ast.Import, ast.ImportFrom))
               for name in _bound_by_a_statement(statement)}
    lines, carried, needed, names = after.splitlines(), [], {}, []
    for statement in new.body:
        if not _an_invariant(statement) or statement.name in there or statement.name not in named:
            continue
        missing = {name for name in _what_it_reads(statement) if name not in there}
        if any(name not in imports for name in missing):
            continue
        needed.update({id(imports[name]): imports[name] for name in missing})
        carried.append(_source_of(statement, lines))
        names.append(statement.name)
    if not carried:
        return "", ()
    return "\n\n".join([*(_source_of(node, lines) for node in needed.values()), *carried]) + "\n", tuple(names)


def module_of(path: str) -> str:
    """The module a repository path holds."""
    return path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def injected(module: str, specification: str) -> str:
    """Source for a conftest that puts ``specification`` into ``module`` before any test imports from it."""
    return (f"\n\nimport importlib as _a_fixs_importlib\n"
            f"exec(compile({specification!r}, {f'<what the fix of {module} wrote to check itself>'!r}, 'exec'),\n"
            f"     _a_fixs_importlib.import_module({module!r}).__dict__)\n")

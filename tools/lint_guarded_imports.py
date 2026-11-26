#!/usr/bin/env python3
"""Guarded imports must name something that exists.

A first-party import inside ``try: ... except ImportError:`` is how this
codebase makes a feature optional. It is also how a feature dies without
anyone noticing: rename the symbol, miss the call site, and the handler
turns the resulting ``ImportError`` into "that capability isn't available"
— forever, silently, on every boot.

That is not hypothetical. Found by audit and fixed in the same pass:

* ``OutcomeSimulator`` became ``OutcomeSimulationEngine``, so "model this
  out" ran with no outcome simulation.
* ``WebSearchSkill`` became ``EnhancedWebSearchSkill`` in two research
  paths, so the self-taught builder and the self-code-improver both lost
  their web leg.
* ``get_memory`` never existed on the memory facade, so the builder's
  cumulative learning recalled nothing, every time.
* ``PRIME_DIRECTIVES`` never existed, so the state authority checked topics
  against three hardcoded strings instead of the constitution.

Every one of those looked like a working system with a quiet feature flag.
The blast radius stayed small only because the fallbacks were fail-closed,
which is luck, not design.

This gate resolves the symbol statically. It understands module-level
``__getattr__`` (PEP 562), ``__all__``, star re-exports, namespace packages,
and submodule imports, because all four are load-bearing here. What it will
not accept is a guarded import of a name that nothing defines.

Run: ``python tools/lint_guarded_imports.py``
"""

from __future__ import annotations

import ast
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("core", "interface", "skills", "tools")
FIRST_PARTY = ("core", "interface", "skills", "tools", "executors", "training")
SKIP_DIR_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "archive",
    "node_modules",
    "tests",
}

#: Handlers that turn a failed import into a degraded feature. ``Exception``
#: counts: it catches ImportError too, and hides it just as well.
_IMPORT_SWALLOWING = frozenset(
    {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
)


def _iter_source_files():
    for top in SCAN_ROOTS:
        base = ROOT / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if any(part in SKIP_DIR_PARTS for part in path.relative_to(ROOT).parts):
                continue
            yield path


@lru_cache(maxsize=None)
def _module_path(dotted: str) -> Path | None:
    parts = dotted.split(".")
    module = ROOT.joinpath(*parts).with_suffix(".py")
    if module.is_file():
        return module
    package = ROOT.joinpath(*parts, "__init__.py")
    if package.is_file():
        return package
    return None


@lru_cache(maxsize=None)
def _is_package_dir(dotted: str) -> bool:
    """A directory that imports work through, with or without ``__init__``.

    ``tools/`` has no ``__init__.py`` and is imported as a namespace package
    all the same, so "no __init__" cannot mean "not a module".
    """
    return ROOT.joinpath(*dotted.split(".")).is_dir()


def _bound_names(tree: ast.Module) -> set[str]:
    """Every name the module binds at import, however conditionally.

    Walks into ``try``/``if``/``with`` bodies: a name bound only on one
    branch is still a name this module can export, and treating it as absent
    would make the gate fire on the very pattern it exists to protect.
    """
    names: set[str] = set()
    stars: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    names.update(
                        el.id for el in target.elts if isinstance(el, ast.Name)
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    if alias.name == "*" and node.module and not node.level:
                        stars.append(node.module)
                else:
                    names.add(alias.asname or alias.name)
    for module in stars:
        exported = _exported_names(module)
        if exported:
            names |= exported
    return names


@lru_cache(maxsize=None)
def _exported_names(dotted: str) -> frozenset[str] | None:
    """Names importable from a first-party module, or None if unanalysable."""
    path = _module_path(dotted)
    if path is None:
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None

    names = _bound_names(tree)

    # PEP 562: a module-level __getattr__ can serve any name. core/consciousness
    # and core/world_model/belief_graph both do this deliberately, to keep
    # import cost off the boot path, and their exports are real.
    if "__getattr__" in names:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                names |= _dunder_all(tree) or {"*any*"}
                names.add("*lazy*")
    return frozenset(names)


def _dunder_all(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                return {
                    el.value
                    for el in node.value.elts
                    if isinstance(el, ast.Constant) and isinstance(el.value, str)
                }
    return set()


def _guarded_imports(tree: ast.Module):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        swallows = False
        for handler in node.handlers:
            caught = handler.type
            candidates: list[str] = []
            if isinstance(caught, ast.Name):
                candidates = [caught.id]
            elif isinstance(caught, ast.Attribute):
                candidates = [caught.attr]
            elif isinstance(caught, ast.Tuple):
                candidates = [
                    el.id if isinstance(el, ast.Name) else getattr(el, "attr", "")
                    for el in caught.elts
                ]
            if any(name in _IMPORT_SWALLOWING for name in candidates):
                swallows = True
        if not swallows:
            continue
        for statement in node.body:
            for sub in ast.walk(statement):
                if isinstance(sub, ast.ImportFrom):
                    yield sub


def _findings() -> tuple[list[str], int]:
    findings: list[str] = []
    checked = 0
    for path in _iter_source_files():
        relative = path.relative_to(ROOT)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for node in _guarded_imports(tree):
            if node.level or not node.module:
                continue  # relative import; resolved against the package
            module = node.module
            if module.split(".")[0] not in FIRST_PARTY:
                continue
            exported = _exported_names(module)
            if exported is None and not _is_package_dir(module):
                findings.append(f"{relative}:{node.lineno}  no module {module!r}")
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                checked += 1
                if exported is not None and (
                    alias.name in exported or "*lazy*" in exported
                ):
                    continue
                if _module_path(f"{module}.{alias.name}") is not None:
                    continue  # from package import submodule
                if _is_package_dir(f"{module}.{alias.name}"):
                    continue
                findings.append(
                    f"{relative}:{node.lineno}  "
                    f"from {module} import {alias.name}  — not defined"
                )
    return findings, checked


def main() -> int:
    findings, checked = _findings()
    print(f"guarded first-party symbol imports checked: {checked}")
    if not findings:
        print("every guarded import resolves")
        return 0
    print(f"{len(findings)} guarded import(s) name something that does not exist:")
    for finding in findings:
        print(f"  {finding}")
    print(
        "\nEach one is a feature that is silently off. Repoint the import at the "
        "current name, or delete the dead branch — do not add it to an allowlist."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

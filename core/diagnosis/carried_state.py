"""Find state that outlives a call.

LIVE, 2026-08-22. A project was handed over with a symptom and no error: the
second invoice came out holding the first one's lines. Nothing raised, no test
failed, and `diagnose_repository` — which knew exactly one experiment, "run the
test suite" — had nothing to say about a project with no tests.

The defect class is general and it is the reason for a large share of "it works
alone and not together" reports: something a function touches survives the call
and is still there for the next one. A default argument that is a list. A
module-level dict a function writes into. A class attribute mutated through an
instance.

This finds them by reading the code, which makes it evidence rather than a
guess, and it is aimed at any directory rather than at Aura's own source.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CarriedState", "carried_state", "describe_carried_state"]

#: Literals that are shared by every call that does not replace them.
_MUTABLE_LITERALS = (ast.List, ast.Dict, ast.Set)

#: Calls that build a fresh mutable each time they are evaluated — as a default
#: argument they are evaluated once, at definition, and shared thereafter.
_MUTABLE_BUILDERS = frozenset({"list", "dict", "set", "bytearray", "defaultdict", "Counter",
                               "OrderedDict", "deque"})

#: Methods that change the thing they are called on.
_MUTATING_METHODS = frozenset({"append", "extend", "insert", "add", "update", "pop",
                               "remove", "clear", "setdefault", "sort", "discard",
                               "popitem", "appendleft", "extendleft"})


@dataclass(frozen=True, slots=True)
class CarriedState:
    """One piece of state that survives the call that touched it."""

    file: str
    line: int
    function: str
    name: str
    kind: str
    detail: str

    def as_sentence(self) -> str:
        """The finding in one line, naming where it is."""
        return f"{self.file}:{self.line} — {self.detail}"


def _is_mutable_default(node: ast.expr) -> str:
    """What kind of shared value this default is, or "" when it is not one."""
    if isinstance(node, _MUTABLE_LITERALS):
        return type(node).__name__.lower()
    if isinstance(node, ast.Call):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name in _MUTABLE_BUILDERS:
            return name
    return ""


def _mutated_names(body: list[ast.stmt]) -> set[str]:
    """Names this body changes in place, by method call or by item assignment."""
    changed: set[str] = set()
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _MUTATING_METHODS and isinstance(node.func.value, ast.Name):
                changed.add(node.func.value.id)
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    changed.add(target.value.id)
    return changed


def _module_level_mutables(tree: ast.Module) -> dict[str, int]:
    """Module-level names bound to something that can be changed in place."""
    found: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not _is_mutable_default(node.value):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found[target.id] = node.lineno
    return found


def _in_one_file(path: Path, root: Path) -> list[CarriedState]:
    """Every piece of surviving state one file holds."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return []
    shown = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    module_mutables = _module_level_mutables(tree)
    found: list[CarriedState] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        defaults = list(zip(args.args[len(args.args) - len(args.defaults):], args.defaults))
        defaults += [
            (arg, default)
            for arg, default in zip(args.kwonlyargs, args.kw_defaults)
            if default is not None
        ]
        changed = _mutated_names(node.body)
        for arg, default in defaults:
            kind = _is_mutable_default(default)
            if not kind:
                continue
            verb = "is changed by the body" if arg.arg in changed else "can be changed by a caller"
            found.append(
                CarriedState(
                    file=shown,
                    line=default.lineno,
                    function=node.name,
                    name=arg.arg,
                    kind="default argument",
                    detail=(
                        f"{node.name}({arg.arg}=...) defaults to a {kind} built once, when "
                        f"the function was defined. Every call that leaves {arg.arg} out gets "
                        f"that same one, and it {verb}, so what one call leaves behind is "
                        f"what the next call starts from."
                    ),
                )
            )
        for name in sorted(changed & set(module_mutables)):
            found.append(
                CarriedState(
                    file=shown,
                    line=module_mutables[name],
                    function=node.name,
                    name=name,
                    kind="module-level value",
                    detail=(
                        f"{name} is defined once at the top of {shown} and {node.name} changes "
                        f"it in place, so every call adds to what the calls before it left."
                    ),
                )
            )
    return found


def carried_state(root: str | Path, *, limit: int = 20) -> tuple[CarriedState, ...]:
    """Every piece of state that outlives a call, anywhere under a directory."""
    base = Path(str(root)).expanduser()
    if base.is_file():
        return tuple(_in_one_file(base, base.parent)[:limit])
    if not base.is_dir():
        return ()
    found: list[CarriedState] = []
    for path in sorted(base.rglob("*.py")):
        if any(part.startswith(".") or part in {"__pycache__", "node_modules"} for part in path.parts):
            continue
        found.extend(_in_one_file(path, base))
        if len(found) >= limit:
            break
    return tuple(found[:limit])


def describe_carried_state(found: tuple[CarriedState, ...]) -> str:
    """The findings as text, or "" when there are none."""
    if not found:
        return ""
    if len(found) == 1:
        return "One thing in this project survives the call that touches it:\n" + found[0].as_sentence()
    lines = [f"{len(found)} things in this project survive the call that touches them:"]
    lines.extend(item.as_sentence() for item in found)
    return "\n".join(lines)

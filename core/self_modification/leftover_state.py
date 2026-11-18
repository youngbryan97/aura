"""What an object is rebuilt still holding, emptied where it is bound (Master Todo F16).

A defect can be a statement nobody wrote. LIVE 13 October 2026 two games stayed "under way" from runs long gone, so
every turn was theirs: the record of work in hand came back from disk with the object, though the process doing that
work had ended (ca975c498). No condition or value there is wrong, so repairing conditions and values cannot reach it.

This watches the objects her code binds from a call on the lines the failing tests run, and notes which of their
mutable containers hold something. A container that holds something in a failing run is a candidate for emptying just
after the binding, and one that stands empty wherever passing tests reach the line comes first. What the attribute is
called plays no part. Her tests judge each candidate (repair_by_selection.py).
"""
from __future__ import annotations

import ast
import asyncio
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.self_modification.semantic_repair import (
    PROBE,
    Run,
    _matches,
    _plan_and_clear,
    _rows,
    _test_of,
    _whole,
    instrumented,
)

__all__ = ["Binding", "bindings_at", "emptied", "leftover_state_repairs"]

logger = logging.getLogger("Aura.LeftoverState")


@dataclass(frozen=True)
class Binding:
    """A statement that binds what a call returned to a name that can be read again on the next line."""

    path: str
    first: int
    last: int
    indent: str
    target: str

    @property
    def id(self) -> str:
        return f"{self.path}:{self.last}"


def _read_back(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) or (isinstance(node, ast.Attribute) and _read_back(node.value))


def bindings_at(source: str, path: str, lines: Sequence[int]) -> list[Binding]:
    """The bindings inside functions that ``lines`` touch, each alone on its lines."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    rows, wanted, found = source.splitlines(), set(lines), {}
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not _read_back(node.targets[0]):
                continue
            made = node.value.value if isinstance(node.value, ast.Await) else node.value
            if not isinstance(made, ast.Call) or not wanted & set(range(node.lineno, node.end_lineno + 1)):
                continue
            before = rows[node.lineno - 1].encode()[:node.col_offset].decode()
            after = rows[node.end_lineno - 1].encode()[node.end_col_offset:].decode().strip()
            if not before.strip() and (not after or after.startswith("#")):
                found[node.lineno] = Binding(path, node.lineno, node.end_lineno, before, ast.unparse(node.targets[0]))
    return list(found.values())


def _with_a_line_after(source: str, binding: Binding, line: str) -> str:
    rows = (source if source.endswith("\n") else source + "\n").splitlines(keepends=True)
    return "".join([*rows[:binding.last], f"{binding.indent}{line}\n", *rows[binding.last:]])


def emptied(source: str, binding: Binding, attribute: str) -> str:
    """``source`` with the container emptied on the line after the binding."""
    return _with_a_line_after(source, binding, f"{binding.target}.{attribute}.clear()")


def _watching(source: str, bindings: Sequence[Binding], plan: Path, record: Path) -> str:
    for binding in sorted(bindings, key=lambda one: -one.last):
        source = _with_a_line_after(source, binding, f"{PROBE}.held({binding.id!r}, {binding.target}, {str(record)!r})")
    return instrumented(source, [], plan=plan, record=record)


def _its_own(binding: Binding, attribute: str) -> bool:
    """Python's convention: a leading underscore is for the object's own methods."""
    return not attribute.startswith("_") or binding.target.split(".")[0] in {"self", "cls"}


def _held_in(rows: Sequence[Mapping[str, Any]], binding: Binding, tests: Sequence[str]) -> set[str]:
    return {attribute for row in rows if row.get("location") == binding.id
            and any(_matches(test, _test_of(row)) for test in tests) for attribute in row.get("full") or ()}


async def leftover_state_repairs(*, sources: Mapping[str, str], failing: Sequence[str], passing: Sequence[str],
                                 ranked: Sequence[tuple[tuple[str, int], float]], run: Run,
                                 workdir: Path) -> AsyncIterator[tuple[str, str]]:
    """Each container worth emptying, as (origin, SEARCH/REPLACE text), the one the passing tests never fill first."""
    order = {line: index for index, (line, score) in reversed(list(enumerate(ranked))) if score > 0}
    lines: dict[str, list[int]] = {}
    for path, line in order:
        if path in sources:
            lines.setdefault(path, []).append(line)
    bindings = [binding for path, at in lines.items() for binding in bindings_at(sources[path], path, at)]
    plan, record = workdir / "held-plan.json", workdir / "held.jsonl"
    watched: dict[str, str] = {}
    for path in {binding.path for binding in bindings}:
        try:
            watched[path] = _watching(sources[path], [b for b in bindings if b.path == path], plan, record)
        except (SyntaxError, ValueError, TypeError) as exc:
            logger.info("%s could not be watched: %s", path, exc)
    bindings = [binding for binding in bindings if binding.path in watched]
    if not bindings:
        return
    await asyncio.to_thread(_plan_and_clear, plan, record, {})
    seen, _ = await run(list(dict.fromkeys([*failing, *passing])), {**sources, **watched})
    if any(seen.get(test) for test in failing):
        return  # a failing test passed while watched, so what was seen is not the fault
    rows = await asyncio.to_thread(_rows, record)
    held = [test for test in passing if seen.get(test)]
    offers = []
    for binding in bindings:
        suspicion = min(order.get((binding.path, n), len(order)) for n in range(binding.first, binding.last + 1))
        filled_when_passing = _held_in(rows, binding, held)
        offers += [((attribute in filled_when_passing, suspicion, attribute), binding, attribute)
                   for attribute in _held_in(rows, binding, failing) if _its_own(binding, attribute)]
    for _rank, binding, attribute in sorted(offers, key=lambda offer: offer[0]):
        after = emptied(sources[binding.path], binding, attribute)
        yield (f"leftover:{binding.path}:{binding.last}:{binding.target}.{attribute}",
               _whole(binding.path, sources[binding.path], after))

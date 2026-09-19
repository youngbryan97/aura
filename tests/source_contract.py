"""Assert what the source DOES without asserting where it sits.

A test that reads source for a call site is the cheapest way to hold a
property no runtime path can reach — an ordering between two statements, a
call that must exist, a spelling that must not. It is also the easiest
thing in this tree to break without changing behaviour, and the full
chunked suite of 2026-09-18 found five of them red in four chunks:

- ``inspect.getsource(record_degradation)`` searched for text the
  method-size sweep had moved into an extracted helper
- a test named ``_api_chat_turn`` and the call moved to
  ``_api_chat_turn_part_10``
- a 700-character window missed a derivation that was still seven
  statements away, because comments were written between
- ``source.index(needle, at)`` raised ValueError, not an assertion, when
  the delimiter moved ahead of the marker it was meant to follow

Every one failed on a refactor that changed nothing, and every one would
have passed on a change that kept the words and broke the meaning. The
fix in each case was the same shape, so it is written once here.

Two rules. Find the code by what it does, not by the name of the function
it currently lives in. Assert an ORDER or a PRESENCE, never a distance —
a proximity window is a measurement of formatting.
"""

from __future__ import annotations

import ast
import inspect
from types import ModuleType

__all__ = [
    "declared_in",
    "function_containing",
    "in_order",
    "module_source",
]


def module_source(module: ModuleType) -> str:
    """The whole module's text."""

    return inspect.getsource(module)


def function_containing(module: ModuleType, needle: str) -> tuple[str, str]:
    """Return ``(name, body)`` of whichever function contains ``needle``.

    Survives extraction: when a sweep moves a block into a helper, this
    follows it. Raises with the needle in the message when nothing in the
    module contains it, which is the case worth failing on — the call site
    really is gone.
    """

    source = inspect.getsource(module)
    lines = source.splitlines()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
        if needle in body:
            return node.name, body
    raise AssertionError(
        f"no function in {module.__name__} contains {needle!r}; "
        "the call site is gone, not merely moved"
    )


def in_order(source: str, *needles: str) -> None:
    """Assert the needles appear in this order, at any distance.

    Each is searched from the end of the previous one, so repeats of an
    earlier needle cannot satisfy a later one.
    """

    if len(needles) < 2:
        raise ValueError("an ordering needs at least two things to order")
    cursor = 0
    seen: list[tuple[str, int]] = []
    for needle in needles:
        found = source.find(needle, cursor)
        if found == -1:
            after = f" after {seen[-1][0]!r}" if seen else ""
            raise AssertionError(f"{needle!r} does not appear{after}")
        seen.append((needle, found))
        cursor = found + len(needle)


def declared_in(needle: str, *modules: ModuleType) -> tuple[str, str]:
    """Return ``(module_name, body)`` for whichever module declares ``needle``.

    ``function_containing`` follows a block that moved into a helper. This
    follows one that moved into another FILE, which is the same refactor
    one directory up and breaks a test the same way.

    LIVE 2026-09-18: ``purpose="research_document_synthesis"`` was read out
    of ``core/skills/desktop_task.py`` by three tests. A method-size sweep
    moved the synthesis call to ``core/skills/desktop_research.py`` with
    every property it was being checked for intact — the origin, both
    internal flags — and ``source.index`` raised ValueError. The tests were
    holding a location, and the location was never the point.

    Search order is the order given, so the caller says which module is the
    expected home and which are the places it may have gone.
    """

    if not modules:
        raise ValueError("a declaration has to be looked for somewhere")
    for module in modules:
        source = inspect.getsource(module)
        if needle in source:
            return module.__name__, source
    looked = ", ".join(module.__name__ for module in modules)
    raise AssertionError(
        f"{needle!r} is declared in none of {looked}; "
        "it is gone, not merely moved"
    )

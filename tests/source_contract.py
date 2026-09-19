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
    "function_containing",
    "in_order",
    "module_source",
    "reached_from_a_finally",
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


def reached_from_a_finally(module: ModuleType, needle: str) -> bool:
    """Whether ``needle`` runs on every exit, directly or one call away.

    "Closed in a finally" is a property of the exit path, not of where the
    line sits. The method-size sweep lifts a whole finally-body into a
    helper and leaves the CALL in the finally, so a test that walks for
    the statement inside a ``finalbody`` stops finding it while the
    guarantee is untouched — measured on ``_close_provenance_tick``, which
    moved into ``_run_thinking_loop_closed_rather_after`` and is still
    invoked from the finally at the end of the thinking loop.

    So: true when the needle is in a finally body, or when it sits in a
    function that a finally body calls. One level, because one level is
    what an extraction produces; a deeper chain should be asserted on
    purpose rather than discovered here.
    """

    source = inspect.getsource(module)
    tree = ast.parse(source)
    lines = source.splitlines()

    def _spans(node: ast.AST) -> tuple[int, int]:
        return node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno

    finallys: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.finalbody:
            low = node.finalbody[0].lineno
            high = max(_spans(s)[1] for s in node.finalbody)
            finallys.append((low, high))

    def _in_a_finally(line: int) -> bool:
        return any(low <= line <= high for low, high in finallys)

    # Directly.
    for index, text in enumerate(lines, start=1):
        if needle in text and _in_a_finally(index):
            return True

    # Or inside a function a finally calls.
    holders = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        low, high = _spans(node)
        if any(needle in line for line in lines[low - 1 : high]):
            holders.add(node.name)
    if not holders:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in holders and _in_a_finally(node.lineno):
            return True
    return False


def function_with_its_helpers(
    module: ModuleType, qualified_name: str, *, depth: int = 2
) -> str:
    """One function's source plus the source of what it calls in its module.

    The unit a source-reading test actually means. ``_run_loop`` is not
    the loop any more — the method-size sweep lifted runs of statements
    out of it into helpers beside it, and a test reading only the method
    sees a shell that calls names. Nine tests in
    ``test_mind_tick_runtime_contract`` went red that way at once, most
    with ``ValueError: substring not found`` from a bare ``source.index``.

    ``qualified_name`` is "Class.method" or "function". Depth 2 follows a
    helper's own helpers, which is where a second sweep pass puts things.
    """

    whole = inspect.getsource(module)
    tree = ast.parse(whole)
    lines = whole.splitlines()

    bodies: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bodies.setdefault(
                node.name,
                "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)]),
            )

    wanted = qualified_name.rsplit(".", 1)[-1]
    if wanted not in bodies:
        raise AssertionError(
            f"{module.__name__} has no function named {wanted!r}; "
            "it was renamed or removed, not merely moved"
        )

    collected: list[str] = []
    seen: set[str] = set()

    def _pull(name: str, remaining: int) -> None:
        if name in seen or name not in bodies:
            return
        seen.add(name)
        body = bodies[name]
        collected.append(body)
        if remaining <= 0:
            return
        for called in ast.walk(ast.parse(_dedent(body))):
            if isinstance(called, ast.Call):
                target = called.func
                called_name = getattr(target, "attr", None) or getattr(
                    target, "id", None
                )
                if called_name:
                    _pull(called_name, remaining - 1)

    _pull(wanted, depth)
    return "\n".join(collected)


def _dedent(body: str) -> str:
    """A method's body parses on its own once its indent is removed."""

    import textwrap

    return textwrap.dedent(body)

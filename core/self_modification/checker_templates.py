"""Checker templates: the guard a fault's own exception asks for, at the line it was raised on (Master Todo F15).

TBar's fix patterns (Liu, Koyuncu, Kim and Bissyandé, ISSTA 2019) begin with three checkers, inserted where a fault
was raised: of a null, of a range and of a cast. PyTER (Oh and Oh, FSE 2022) repairs Python's type errors the same
way. Her fix templates had none of them (fix_templates.py). The exception a fault raised says which checker it asks
for:

- AttributeError or TypeError: a value read at the line may be None (``x.attr``, ``x[i]``, ``x(...)``, ``for _ in x``).
- IndexError: an index may be out of range (``xs[i]``).
- KeyError: a key may be absent (``d[k]``).
- ZeroDivisionError: a divisor may be zero (``a / b``, ``a // b``, ``a % b``).

Each guard is written two ways, and the tests decide between them and against every other candidate. Either the value
is guarded where it is read (``x.attr`` becomes ``(x.attr if x is not None else None)``), or the statement that reads
it runs only when the guard holds. Neither adds an early return, or makes a return, a raise, an assert, a break or a
continue conditional. Those are the changes search-based repair most often passes its tests with by doing nothing
(Tan, Yoshida, Prasad and Roychoudhury, FSE 2016), the anti-patterns fix_templates already keeps out.

A guard is written only around a plain reference: a name, or attributes of a name. Guarding a call would run it twice.
"""
from __future__ import annotations

import ast
from collections.abc import Iterator
from typing import Any

from core.self_modification.fix_templates import _offsets, _replace, _span

__all__ = ["CHECKED_ERRORS", "checker_changes_at"]

#: The exceptions a checker answers, and the checkers each asks for.
CHECKED_ERRORS: dict[str, tuple[str, ...]] = {
    "AttributeError": ("none",),
    "TypeError": ("none",),
    "IndexError": ("range",),
    "KeyError": ("key",),
    "ZeroDivisionError": ("divisor",),
}

#: Statements that end a block or a function: a guard never makes one conditional.
_EXITS = (ast.Return, ast.Raise, ast.Assert, ast.Break, ast.Continue)


def _plain(node: ast.AST) -> bool:
    """A name, or attributes of one: a reference read twice reads the same thing and runs nothing."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _inner(node: ast.AST) -> ast.AST | None:
    """The next link in from a chain of reads: ``x`` in ``x.a``, ``x[i]`` and ``x(...)``."""
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return node.value
    if isinstance(node, ast.Call):
        return node.func
    return None


def _chain_ends(tree: ast.AST, line: int) -> Iterator[ast.AST]:
    """The outermost chain of reads (``user.name.title()``) starting on ``line``, each once."""
    inside = {id(_inner(node)) for node in ast.walk(tree) if _inner(node) is not None}
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) == line and _inner(node) is not None and id(node) not in inside:
            if isinstance(getattr(node, "ctx", ast.Load()), ast.Load):
                yield node


def _none_guards(tree: ast.AST, line: int, text: Any) -> Iterator[tuple[str, ast.AST, str, str]]:
    """A chain of reads guarded where it is read, on each plain reference along it that may be None.

    ``user.name.title()`` with ``user`` None is guarded on ``user``, not on ``user.name``: the whole chain is the
    value, as TBar's null checker guards the expression and not one link of it."""
    for end in _chain_ends(tree, line):
        link = _inner(end)
        while link is not None:
            if _plain(link):
                condition = f"{text(link)} is not None"
                yield "none", end, condition, f"({text(end)} if {condition} else None)"
            link = _inner(link)
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)) and node.lineno == line and _plain(node.iter):
            yield "none", node.iter, f"{text(node.iter)} is not None", ""


def _guards(node: ast.AST, source: str, checkers: tuple[str, ...]) -> Iterator[tuple[str, ast.AST, str, str]]:
    """(checker, the expression guarded, the condition that must hold, the guarded expression) for one node."""
    text = lambda part: ast.get_source_segment(source, part) or ""  # noqa: E731 - read the exact text of a node
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and _plain(node.value) \
            and not isinstance(node.slice, ast.Slice):
        seq, index = text(node.value), node.slice
        if "range" in checkers:
            if isinstance(index, ast.Constant) and isinstance(index.value, int) and not isinstance(index.value, bool):
                condition = f"len({seq}) > {index.value}" if index.value >= 0 else f"len({seq}) >= {-index.value}"
            elif _plain(index):
                condition = f"-len({seq}) <= {text(index)} < len({seq})"
            else:
                condition = ""
            if condition:
                yield "range", node, condition, f"({text(node)} if {condition} else None)"
        if "key" in checkers and (_plain(index) or isinstance(index, ast.Constant)):
            yield "key", node, f"{text(index)} in {seq}", f"{seq}.get({text(index)})"
    if "divisor" in checkers and isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
        right = node.right
        if _plain(right):
            yield "divisor", node, f"{text(right)}", f"({text(node)} if {text(right)} else None)"


def _statement_at(tree: ast.AST, line: int) -> ast.stmt | None:
    """The innermost statement that begins on ``line``."""
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and node.lineno == line:
            if found is None or (node.col_offset, -(node.end_lineno or node.lineno)) > \
                    (found.col_offset, -(found.end_lineno or found.lineno)):
                found = node
    return found


def _wrapped(source: str, statement: ast.stmt, condition: str) -> str | None:
    """``source`` with ``statement`` run only when ``condition`` holds, or None when it cannot be written so."""
    if isinstance(statement, (*_EXITS, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    lines = source.splitlines(keepends=True)
    first, last = statement.lineno - 1, (statement.end_lineno or statement.lineno) - 1
    indent = lines[first][: statement.col_offset]
    if indent.strip() or len(lines[first]) - len(lines[first].lstrip()) != statement.col_offset:
        return None  # not alone at the start of its line
    body = [f"    {text}" if text.strip() else text for text in lines[first:last + 1]]
    return "".join([*lines[:first], f"{indent}if {condition}:\n", *body, *lines[last + 1:]])


def checker_changes_at(source: str, line: int, error_type: str) -> Iterator[tuple[str, str]]:
    """Every checker's change at ``line`` for a fault that raised ``error_type``, as (template, changed source).

    The value-guarding changes come first, since each changes one expression; the guarded statements follow.
    """
    checkers = CHECKED_ERRORS.get(str(error_type or "").rsplit(".", 1)[-1], ())
    if not checkers:
        return
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return  # not a failure: source that does not parse has no node for a checker to guard
    starts, seen = _offsets(source), {source}
    statement = _statement_at(tree, line)
    found = [guard for node in ast.walk(tree) if getattr(node, "lineno", None) == line
             for guard in _guards(node, source, checkers)]
    if "none" in checkers:
        found = [*_none_guards(tree, line, lambda part: ast.get_source_segment(source, part) or ""), *found]
    wrapping: list[tuple[str, str]] = []
    for checker, node, condition, guarded in found:
        if guarded:
            a, b = _span(source, starts, node)
            changed = _replace(source, a, b, guarded)
            if changed not in seen and _parses(changed):
                seen.add(changed)
                yield f"guard_{checker}_value", changed
        if statement is not None:
            wrapped = _wrapped(source, statement, condition)
            if wrapped is not None:
                wrapping.append((f"guard_{checker}_statement", wrapped))
    for name, changed in wrapping:
        if changed not in seen and _parses(changed):
            seen.add(changed)
            yield name, changed


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True

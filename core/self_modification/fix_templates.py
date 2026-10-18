"""Fix templates: the small changes that mend most one-line faults, made exactly at the lines a failing test runs.

Automated program repair found that a short list of change templates covers a
large share of real one-hunk fixes (PAR, Kim et al., ICSE 2013; TBar, Liu et
al., ISSTA 2019): change a comparison, shift a bound or a count by one, negate
a condition, change a boolean or arithmetic operator, swap two arguments,
flip a literal, use another name in scope, drop a statement. Each is made here by her own Python on the
syntax tree's exact positions, so the result parses and changes only what the
template names.

Where to try them comes from the tests. The lines the failing tests run and
the passing tests do not are the likeliest places (spectrum-based fault
localisation; the Ochiai score, Abreu, Zoeteweij and van Gemund 2007). Nothing
here is taken for a fix: every change is a candidate, and the tests decide
(repair_by_selection.py).
"""
from __future__ import annotations

import ast
import math
from collections.abc import Iterator, Mapping, Sequence

__all__ = ["changes_at", "ochiai", "template_changes"]

Line = tuple[str, int]

_COMPARE_SWAPS = {
    "<": ("<=", ">", ">="), "<=": ("<", ">=", ">"), ">": (">=", "<", "<="), ">=": (">", "<=", "<"),
    "==": ("!=",), "!=": ("==",), "is": ("is not",), "is not": ("is",), "in": ("not in",), "not in": ("in",),
}
_ARITHMETIC_SWAPS = {"+": ("-",), "-": ("+",), "*": ("/", "//"), "/": ("*", "//"), "//": ("/",), "%": ("//",)}


def ochiai(failing: Mapping[str, set[Line]], passing: Mapping[str, set[Line]]) -> list[tuple[Line, float]]:
    """Each line any failing test ran, scored ef / sqrt(total_failing * (ef + ep)), most suspicious first."""
    total_failing = len(failing)
    scores: dict[Line, float] = {}
    for line in set().union(*failing.values()) if failing else set():
        ef = sum(1 for lines in failing.values() if line in lines)
        ep = sum(1 for lines in passing.values() if line in lines)
        scores[line] = ef / math.sqrt(total_failing * (ef + ep)) if ef else 0.0
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _offsets(source: str) -> list[int]:
    """Where each line starts, in characters."""
    starts, at = [0], 0
    for line in source.splitlines(keepends=True):
        at += len(line)
        starts.append(at)
    return starts


def _at(source: str, starts: list[int], lineno: int, col: int) -> int:
    """A character offset from the tree's line and its byte column."""
    line = source[starts[lineno - 1]:starts[lineno]] if lineno < len(starts) else source[starts[lineno - 1]:]
    return starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8", errors="ignore"))


def _span(source: str, starts: list[int], node: ast.AST) -> tuple[int, int]:
    return (_at(source, starts, node.lineno, node.col_offset),
            _at(source, starts, node.end_lineno or node.lineno, node.end_col_offset or 0))


def _replace(source: str, start: int, end: int, text: str) -> str:
    return source[:start] + text + source[end:]


def _operator_between(source: str, start: int, end: int, table: Mapping[str, Sequence[str]]) -> Iterator[str]:
    """The source with the operator written between two operands swapped for each alternative."""
    between = source[start:end]
    # A comment or a parenthesis between the operands leaves the operator where the stripped text says it is.
    token = between.strip().strip("()").strip()
    if token not in table:
        return
    at = start + between.index(token)
    for other in table[token]:
        yield _replace(source, at, at + len(token), other)


def changes_at(source: str, line: int) -> Iterator[tuple[str, str]]:
    """Every template's change to the code on ``line``, as (template, changed source); each parses."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    starts = _offsets(source)
    seen: set[str] = set()

    def kept(name: str, changed: str) -> Iterator[tuple[str, str]]:
        if changed == source or changed in seen:
            return
        try:
            ast.parse(changed)
        except SyntaxError:
            return
        seen.add(changed)
        yield name, changed

    def names_in_scope(at: int) -> list[str]:
        """The parameters and assigned names of the innermost function holding ``at``, in the order written."""
        holders = [f for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and f.lineno <= at <= (f.end_lineno or f.lineno)]
        if not holders:
            return []
        fn = max(holders, key=lambda f: f.lineno)
        names = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)]
        names += [n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
        return list(dict.fromkeys(n for n in names if n not in {"self", "cls"}))

    scope = names_in_scope(line)
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != line:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in scope:
            a, b = _span(source, starts, node)
            for other in scope:
                if other != node.id:
                    yield from kept("replace_name", _replace(source, a, b, other))
            continue
        if isinstance(node, ast.Compare):
            left = node.left
            for right in node.comparators:
                a, b = _span(source, starts, left)[1], _span(source, starts, right)[0]
                for changed in _operator_between(source, a, b, _COMPARE_SWAPS):
                    yield from kept("change_comparison", changed)
                left = right
        elif isinstance(node, ast.BoolOp):
            for first, second in zip(node.values, node.values[1:], strict=False):
                a, b = _span(source, starts, first)[1], _span(source, starts, second)[0]
                for changed in _operator_between(source, a, b, {"and": ("or",), "or": ("and",)}):
                    yield from kept("change_boolean_operator", changed)
        elif isinstance(node, ast.BinOp):
            a, b = _span(source, starts, node.left)[1], _span(source, starts, node.right)[0]
            for changed in _operator_between(source, a, b, _ARITHMETIC_SWAPS):
                yield from kept("change_arithmetic_operator", changed)
        elif isinstance(node, (ast.If, ast.While, ast.IfExp)):
            test = node.test
            a, b = _span(source, starts, test)
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                inner = _span(source, starts, test.operand)
                yield from kept("negate_condition", _replace(source, a, b, source[inner[0]:inner[1]]))
            else:
                yield from kept("negate_condition", _replace(source, a, b, f"not ({source[a:b]})"))
        elif isinstance(node, ast.Constant):
            a, b = _span(source, starts, node)
            if isinstance(node.value, bool):
                yield from kept("flip_literal", _replace(source, a, b, str(not node.value)))
            elif isinstance(node.value, int):
                for other in (node.value + 1, node.value - 1):
                    yield from kept("shift_by_one", _replace(source, a, b, str(other)))
        elif isinstance(node, ast.Slice):
            for bound in (node.lower, node.upper):
                if bound is None:
                    continue
                a, b = _span(source, starts, bound)
                for sign in ("+ 1", "- 1"):
                    yield from kept("shift_by_one", _replace(source, a, b, f"{source[a:b]} {sign}"))
        elif isinstance(node, ast.Call) and len(node.args) >= 2 and not any(
                isinstance(arg, ast.Starred) for arg in node.args):
            for first, second in zip(node.args, node.args[1:], strict=False):
                a1, b1 = _span(source, starts, first)
                a2, b2 = _span(source, starts, second)
                swapped = source[:a1] + source[a2:b2] + source[b1:a2] + source[a1:b1] + source[b2:]
                yield from kept("swap_arguments", swapped)
        elif isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                     ast.ClassDef)):
            a, b = _span(source, starts, node)
            yield from kept("drop_statement", _replace(source, a, b, "pass"))


def template_changes(ranked: Sequence[tuple[Line, float]], sources: Mapping[str, str]) -> Iterator[tuple[str, str, str]]:
    """(path, template, changed source) for every template at every ranked line, most suspicious line first.

    At one line, dropping a statement is tried last: of the templates it is the one a real fix least often is.
    """
    for (path, line), score in ranked:
        if path in sources and score > 0:
            changes = list(changes_at(sources[path], line))
            for name, changed in sorted(changes, key=lambda c: c[0] == "drop_statement"):
                yield path, name, changed

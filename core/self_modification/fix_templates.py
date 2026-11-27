"""Fix templates: the small changes that mend most one-line faults, made exactly at the lines a failing test runs.

Automated program repair found that a short list of change templates covers a
large share of real one-hunk fixes (PAR, Kim et al., ICSE 2013; TBar, Liu et
al., ISSTA 2019): change a comparison, shift a bound or a count by one, negate
a condition, change a boolean or arithmetic operator, swap two arguments,
flip a literal, use another name in scope, drop a statement. Each is made here by her own Python on the
syntax tree's exact positions, so the result parses and changes only what the
template names.

A dropped statement is never an exit of its function (return, raise, assert), never one that decides control (if,
a loop, try, with, break, continue), and never the only statement of its block. Deleting those is how search-based
repair makes a failing test pass by removing what the test reached (the anti-patterns of Tan, Yoshida, Prasad and
Roychoudhury, FSE 2016). LIVE 16 November 2026, the first candidate that made her fault's test pass replaced
require_governance's raise with ``pass``; seven guarding tests broke, so it was refused, and a file guarded less
well would have kept it. Nor is it a statement that builds an exception to hand on (``record_degradation(
RuntimeError(...))``): a report of a fault is the raise of a function that goes on, and dropping it silences what it
found. LIVE 26 November 2026 her loop offered exactly that for DistributedResilienceCore._request_recovery, whose report
of an unhealthy subsystem was true.

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
#: The statements a dropped statement may be: what a function does, not where it ends or which way it goes.
_DROPPABLE = (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)


def _exceptions_named_in(tree: ast.AST) -> frozenset[str]:
    """The names that are exception classes where ``tree`` is read: Python's own, and the classes it defines on them."""
    import builtins

    known = {name for name, value in vars(builtins).items()
             if isinstance(value, type) and issubclass(value, BaseException)}
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    grew = True
    while grew:
        grew = False
        for node in classes:
            bases = {base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else ""
                     for base in node.bases}
            if node.name not in known and bases & known:
                known.add(node.name)
                grew = True
    return frozenset(known)


def _builds_an_exception(statement: ast.AST, exceptions: frozenset[str]) -> bool:
    """Whether ``statement`` builds an exception: a report of a fault, which a dropped statement never is."""
    for node in ast.walk(statement):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else \
                node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in exceptions:
                return True
    return False


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
        return  # not a failure: source that does not parse has no node for a template to change
    starts = _offsets(source)
    seen: set[str] = set()

    def kept(name: str, changed: str) -> Iterator[tuple[str, str]]:
        if changed == source or changed in seen:
            return
        try:
            ast.parse(changed)
        except SyntaxError:
            return  # not a failure: a change that does not parse is not offered
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
    exceptions = _exceptions_named_in(tree)
    alone = {id(block[0]) for holder in ast.walk(tree)
             for block in (getattr(holder, name, None) for name in ("body", "orelse", "finalbody"))
             if isinstance(block, list) and len(block) == 1}
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
        elif isinstance(node, _DROPPABLE) and id(node) not in alone and not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)) \
                and not _builds_an_exception(node, exceptions):
            # A docstring or a bare literal does nothing to drop. LIVE 16 November 2026 her loop spent a candidate
            # dropping the drill function's docstring.
            a, b = _span(source, starts, node)
            yield from kept("drop_statement", _replace(source, a, b, "pass"))


def template_changes(ranked: Sequence[tuple[Line, float]], sources: Mapping[str, str],
                     error_type: str = "") -> Iterator[tuple[str, str, str]]:
    """(path, template, changed source) for every template at every ranked line, most suspicious line first.

    At one line, the checkers the fault's exception asks for go first (checker_templates.py), when ``error_type``
    names it. The templates with the fewest alternatives there follow, so a line's few exact checks are not queued
    behind every other name in scope (30 of 38 candidates on a real defect, 18 October 2026), and dropping a statement
    goes last: of the templates it is the one a real fix least often is.
    """
    from core.self_modification.checker_templates import checker_changes_at

    for (path, line), score in ranked:
        if path in sources and score > 0:
            for name, changed in checker_changes_at(sources[path], line, error_type):
                yield path, name, changed
            changes = list(changes_at(sources[path], line))
            alternatives = {name: sum(1 for n, _ in changes if n == name) for name, _ in changes}
            order = sorted(changes, key=lambda c: (c[0] == "drop_statement", alternatives[c[0]]))
            for name, changed in order:
                yield path, name, changed

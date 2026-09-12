#!/usr/bin/env python3
"""Read one boolean out of a run report, without running the file as code.

The completion tracker stores 288 expressions like
``all(int(r["null_draws"]) >= 1000 for r in report["synergy"])`` beside the
items they support. They were handed to ``eval``, which makes a JSON file an
executable: anyone who can edit the evidence file can run anything the tracker
can run, and the enterprise gate is right to count that as critical.

So read the expression instead of running it. The grammar below is what the
288 use and nothing else — no assignment, no import, no attribute except the
five mapping methods they call, no name except ``report`` and the loop
variables a comprehension binds. An expression that reaches outside it raises
``ExpressionRefused`` and the check reports itself as unable to run, which is
the honest answer and is not the same as an item that is not done.
"""
from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from typing import Any

__all__ = ["ExpressionRefused", "evaluate_report_expression"]


class ExpressionRefused(Exception):
    """The expression asked for something this reader does not offer."""


#: Builtins an expression may call. Every one is a pure reader over values
#: already in the report; none of them can reach the interpreter.
CALLABLES: dict[str, Any] = {
    "abs": abs, "all": all, "any": any, "bool": bool, "float": float,
    "int": int, "len": len, "list": list, "max": max, "min": min,
    "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple,
}

#: Methods an expression may call on a value it already reached.
METHODS = frozenset({"endswith", "get", "items", "keys", "startswith", "values"})

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
}

_COMPARE = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}


def evaluate_report_expression(expr: str, report: Mapping[str, Any]) -> Any:
    """Evaluate ``expr`` against ``report``. Raises ExpressionRefused, KeyError…

    A KeyError, IndexError or TypeError from the report's own shape travels
    out unchanged: the expression was readable and the report did not hold
    what it asked for, which is a check that failed rather than one that could
    not run.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionRefused(f"{expr!r} does not parse: {exc}") from exc
    return _read(tree.body, {"report": report})


def _read(node: ast.AST, scope: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in scope:
            return scope[node.id]
        if node.id in CALLABLES:
            return CALLABLES[node.id]
        raise ExpressionRefused(f"no name {node.id!r} here")

    if isinstance(node, ast.Subscript):
        return _read(node.value, scope)[_read(node.slice, scope)]

    if isinstance(node, ast.Attribute):
        if node.attr not in METHODS:
            raise ExpressionRefused(f"attribute {node.attr!r} is not readable")
        return getattr(_read(node.value, scope), node.attr)

    if isinstance(node, ast.Call):
        if node.keywords:
            raise ExpressionRefused("keyword arguments are not read here")
        # What is being called has to be one of the names above, or a method
        # named above on a value already reached. Resolving the callee and
        # calling whatever comes back would let a report that held a callable
        # have it invoked — reports are JSON and hold none, and a reader whose
        # safety rests on its input's type is one input away from not being
        # safe.
        if isinstance(node.func, ast.Name):
            if node.func.id not in CALLABLES or node.func.id in scope:
                raise ExpressionRefused(f"{node.func.id!r} is not callable here")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr not in METHODS:
                raise ExpressionRefused(f"method {node.func.attr!r} is not callable here")
        else:
            raise ExpressionRefused("only a named function or method may be called")
        func = _read(node.func, scope)
        return func(*[_read(arg, scope) for arg in node.args])

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            value: Any = True
            for item in node.values:
                value = _read(item, scope)
                if not value:
                    return value
            return value
        value = False
        for item in node.values:
            value = _read(item, scope)
            if value:
                return value
        return value

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not _read(node.operand, scope)
        if isinstance(node.op, ast.USub):
            return -_read(node.operand, scope)
        if isinstance(node.op, ast.UAdd):
            return +_read(node.operand, scope)
        raise ExpressionRefused(f"{type(node.op).__name__} is not read here")

    if isinstance(node, ast.BinOp):
        apply = _BINARY.get(type(node.op))
        if apply is None:
            raise ExpressionRefused(f"{type(node.op).__name__} is not read here")
        return apply(_read(node.left, scope), _read(node.right, scope))

    if isinstance(node, ast.Compare):
        left = _read(node.left, scope)
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            compare = _COMPARE.get(type(op))
            if compare is None:
                raise ExpressionRefused(f"{type(op).__name__} is not read here")
            right = _read(right_node, scope)
            if not compare(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.IfExp):
        branch = node.body if _read(node.test, scope) else node.orelse
        return _read(branch, scope)

    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items = [_read(item, scope) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(items)
        return items if isinstance(node, ast.List) else set(items)

    if isinstance(node, ast.Dict):
        return {
            _read(key, scope): _read(value, scope)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }

    if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
        produced = list(_comprehend(node.generators, node.elt, scope))
        if isinstance(node, ast.SetComp):
            return set(produced)
        return produced  # a list serves every reader a generator would

    raise ExpressionRefused(f"{type(node).__name__} is not read here")


def _comprehend(
    generators: list[ast.comprehension], element: ast.expr, scope: dict[str, Any]
) -> Any:
    if not generators:
        yield _read(element, scope)
        return
    first, rest = generators[0], generators[1:]
    if first.is_async:
        raise ExpressionRefused("an async comprehension is not read here")
    for item in _read(first.iter, scope):
        inner = dict(scope)
        _bind(first.target, item, inner)
        if all(_read(condition, inner) for condition in first.ifs):
            yield from _comprehend(rest, element, inner)


def _bind(target: ast.expr, value: Any, scope: dict[str, Any]) -> None:
    if isinstance(target, ast.Name):
        scope[target.id] = value
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        parts = list(value)
        if len(parts) != len(target.elts):
            raise ValueError(
                f"unpacking {len(parts)} value(s) into {len(target.elts)} name(s)"
            )
        for name, part in zip(target.elts, parts, strict=True):
            _bind(name, part, scope)
        return
    raise ExpressionRefused(f"cannot bind {type(target).__name__}")

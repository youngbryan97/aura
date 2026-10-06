"""Compile bounded integer expressions onto Aura's metered universal floor."""
from __future__ import annotations

import ast
from typing import Any
from core.verify.invariants import invariant


def integer_expression(expression: str) -> int | None:
    """Exact arithmetic without eval, generated Python, a model, or an expected answer."""
    from core.cognition.the_floor_she_stands_on import LEFTOVER, LET, MINUS, N, OVER, PLUS, TIMES, V, build, run

    if not isinstance(expression, str) or len(expression) > 4096:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            return None
        bindings = []

        def bind(value: Any) -> Any:
            name = f"number_{len(bindings)}"
            bindings.append((name, value))
            return V(name)

        def compile_node(node: ast.AST) -> Any:
            if isinstance(node, ast.Constant) and type(node.value) is int and node.value.bit_length() <= 4096:
                return bind(N(node.value))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = compile_node(node.operand)
                return bind(MINUS(N(0), value)) if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.BinOp):
                left = compile_node(node.left)
                if isinstance(node.op, ast.Pow):
                    if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int or not 0 <= node.right.value <= 64:
                        raise ValueError("unsupported exponent")
                    value = bind(N(1))
                    for _ in range(node.right.value):
                        value = bind(TIMES(value, left))
                    return value
                operation = {ast.Add: PLUS, ast.Sub: MINUS, ast.Mult: TIMES, ast.FloorDiv: OVER, ast.Mod: LEFTOVER}.get(type(node.op))
                if operation is not None:
                    return bind(operation(left, compile_node(node.right)))
            raise ValueError("unsupported integer expression")

        term = compile_node(tree.body)
        for name, value in reversed(bindings):
            term = LET(name, value, term)
        result = run(build(term), fuel=10000)
        return result if type(result) is int and result.bit_length() <= 12000 else None
    except (SyntaxError, ValueError, TypeError, RuntimeError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def arithmetic_canary() -> bool:
    return (integer_expression("9007199254740993+1") == 9007199254740994
            and integer_expression("-17//4") == -5
            and integer_expression("1//0") is None)


@invariant("reasoning.floor_arithmetic_is_exact_and_bounded", scope="reasoning",
           owner="core/reasoning/arithmetic_on_the_floor.py", observational=False)
def _arithmetic_invariant() -> tuple:
    assert arithmetic_canary(), "exact arithmetic or the unknown outcome changed"
    return ()

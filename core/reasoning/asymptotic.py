"""Compare supported polynomial/logarithmic growth classes without eval."""

from __future__ import annotations

import ast
import re
from typing import Any


def growth_class(value: object, *, variable: str = "n") -> tuple[int, int] | None:
    """Return powers of n and log(n) for a positive polynomial expression.

    Big-O and Theta spellings identify the requested growth class here, not
    arbitrary upper-bound implication. Unknown syntax and cancellation are
    unmeasured. The AST constructs SymPy objects directly; model text is never
    passed to sympify/parse_expr, both of which accept executable expressions.
    """
    if not isinstance(value, str) or len(value) > 256 or not variable.isidentifier():
        return None
    match = re.fullmatch(r"\s*(?:O|Theta|\u0398)\s*\((.*)\)\s*", value)
    if match is None:
        return None
    try:
        tree = ast.parse(match[1].replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return None
    if sum(1 for _ in ast.walk(tree)) > 64:
        return None
    import sympy as sp

    n, log_n = sp.symbols("n log_n", positive=True)

    def convert(node: Any) -> Any:
        if isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 10**6:
            return sp.Integer(node.value)
        if isinstance(node, ast.Name) and node.id == variable:
            return n
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"log", "ln"} and not node.keywords
                and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                and node.args[0].id == variable):
            return log_n
        if isinstance(node, ast.BinOp):
            left = convert(node.left)
            if isinstance(node.op, ast.Pow):
                if (isinstance(node.right, ast.Constant) and type(node.right.value) is int
                        and 0 <= node.right.value <= 16
                        and isinstance(node.left, (ast.Name, ast.Call))):
                    return left ** node.right.value
            elif isinstance(node.op, ast.Add):
                return left + convert(node.right)
            elif isinstance(node.op, ast.Mult):
                return left * convert(node.right)
        raise ValueError("unsupported growth expression")

    try:
        polynomial = sp.Poly(convert(tree.body), n, log_n)
    except (ValueError, sp.PolynomialError):
        return None
    if polynomial.is_zero or any(coefficient <= 0 for coefficient in polynomial.coeffs()):
        return None
    return tuple(int(power) for power in max(polynomial.monoms()))


def same_growth_class(left: object, right: object) -> bool:
    measured = growth_class(left)
    return measured is not None and measured == growth_class(right)

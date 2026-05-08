"""Neuro-symbolic reasoning bridge for exact subproblems."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolicResult:
    ok: bool
    engine: str
    result: Any
    proof_trace: str

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "engine": self.engine, "result": str(self.result), "proof_trace": self.proof_trace}


class SymbolicBridge:
    """Routes formalizable work to exact solvers when available."""

    def simplify_math(self, expression: str) -> SymbolicResult:
        try:
            import sympy as sp

            expr = sp.sympify(expression)
            simplified = sp.simplify(expr)
            return SymbolicResult(True, "sympy", simplified, f"sympy.simplify({expression!r})")
        except Exception as exc:
            return SymbolicResult(False, "sympy", repr(exc), "solver_error")

    def check_python_boolean(self, expression: str) -> SymbolicResult:
        try:
            tree = ast.parse(expression, mode="eval")
            value = _evaluate_boolean_ast(tree)
            return SymbolicResult(True, "python_ast", bool(value), "restricted_ast_eval")
        except Exception as exc:
            return SymbolicResult(False, "python_ast", repr(exc), "solver_error")

    def solve_constraints(self, constraints: list[str]) -> SymbolicResult:
        try:
            import z3  # type: ignore

            solver = z3.Solver()
            names: dict[str, Any] = {}
            for raw in constraints:
                tree = ast.parse(raw, mode="eval")
                solver.add(_z3_from_ast(tree, names, z3))
            status = solver.check()
            return SymbolicResult(True, "z3", status, str(solver.model()) if status == z3.sat else str(status))
        except Exception as exc:
            return SymbolicResult(False, "z3", repr(exc), "solver_unavailable_or_error")


def _evaluate_boolean_ast(tree: ast.Expression) -> bool:
    def value(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float, str)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not bool(value(node.operand))
        if isinstance(node, ast.BoolOp):
            vals = [bool(value(item)) for item in node.values]
            if isinstance(node.op, ast.And):
                return all(vals)
            if isinstance(node.op, ast.Or):
                return any(vals)
        if isinstance(node, ast.Compare):
            left = value(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = value(comparator)
                if isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                elif isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                else:
                    raise ValueError(f"unsupported comparator: {type(op).__name__}")
                if not ok:
                    return False
                left = right
            return True
        raise ValueError(f"unsupported boolean AST node: {type(node).__name__}")

    return bool(value(tree))


def _z3_from_ast(tree: ast.Expression, names: dict[str, Any], z3: Any) -> Any:
    def value(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Name):
            return names.setdefault(node.id, z3.Real(node.id))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -value(node.operand)
            if isinstance(node.op, ast.Not):
                return z3.Not(value(node.operand))
        if isinstance(node, ast.BinOp):
            left = value(node.left)
            right = value(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        if isinstance(node, ast.BoolOp):
            values = [value(item) for item in node.values]
            if isinstance(node.op, ast.And):
                return z3.And(*values)
            if isinstance(node.op, ast.Or):
                return z3.Or(*values)
        if isinstance(node, ast.Compare):
            clauses = []
            left = value(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = value(comparator)
                if isinstance(op, ast.Eq):
                    clauses.append(left == right)
                elif isinstance(op, ast.NotEq):
                    clauses.append(left != right)
                elif isinstance(op, ast.Lt):
                    clauses.append(left < right)
                elif isinstance(op, ast.LtE):
                    clauses.append(left <= right)
                elif isinstance(op, ast.Gt):
                    clauses.append(left > right)
                elif isinstance(op, ast.GtE):
                    clauses.append(left >= right)
                else:
                    raise ValueError(f"unsupported comparator: {type(op).__name__}")
                left = right
            return z3.And(*clauses) if len(clauses) > 1 else clauses[0]
        raise ValueError(f"unsupported constraint AST node: {type(node).__name__}")

    return value(tree)


__all__ = ["SymbolicBridge", "SymbolicResult"]

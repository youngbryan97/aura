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
            allowed = (ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.NameConstant, ast.Constant, ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq)
            if not all(isinstance(node, allowed) for node in ast.walk(tree)):
                return SymbolicResult(False, "python_ast", "unsupported_expression", "restricted_ast_rejected")
            value = eval(compile(tree, "<symbolic-boolean>", "eval"), {"__builtins__": {}}, {})
            return SymbolicResult(True, "python_ast", bool(value), "restricted_ast_eval")
        except Exception as exc:
            return SymbolicResult(False, "python_ast", repr(exc), "solver_error")

    def solve_constraints(self, constraints: list[str]) -> SymbolicResult:
        try:
            import z3  # type: ignore

            solver = z3.Solver()
            names: dict[str, Any] = {}
            for raw in constraints:
                for token in raw.replace("(", " ").replace(")", " ").replace("<=", " ").replace(">=", " ").replace("==", " ").replace("<", " ").replace(">", " ").split():
                    if token.isidentifier() and token not in {"and", "or", "not"}:
                        names.setdefault(token, z3.Real(token))
                solver.add(eval(raw, {"__builtins__": {}}, names))
            status = solver.check()
            return SymbolicResult(True, "z3", status, str(solver.model()) if status == z3.sat else str(status))
        except Exception as exc:
            return SymbolicResult(False, "z3", repr(exc), "solver_unavailable_or_error")


__all__ = ["SymbolicBridge", "SymbolicResult"]

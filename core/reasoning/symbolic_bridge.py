"""Neuro-symbolic reasoning bridge for exact subproblems."""
from __future__ import annotations

import ast
import math
import re
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


@dataclass(frozen=True)
class ArithmeticClaimRepair:
    """One exact, span-bound correction made to model-authored arithmetic."""

    claim: str
    stated: float
    correct: float
    start: int
    end: int
    replacement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "stated": self.stated,
            "correct": self.correct,
            "start": self.start,
            "end": self.end,
            "replacement": self.replacement,
        }


_CLAIM_NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_CLAIM_ATOM = rf"(?:{_CLAIM_NUMBER}|\(\s*{_CLAIM_NUMBER}\s*\))"
#: A unit carried along with a number in written working: "1317 DAYS * 86400
#: SECONDS/DAY = 113,923,200 SECONDS".
#:
#: Without this the pattern matched only bare arithmetic, which is the one form
#: shown work almost never takes. Measured live 2026-08-18: she reasoned a
#: duration correctly and then wrote
#: "1317 days * 86400 seconds/day = 113,923,200 seconds" — the product is
#: 113,788,800. The same equation written bare WAS caught, so the check was
#: working and simply never saw the sentence people write.
#:
#: Deliberately narrow: one word, optionally over a second ("seconds/day"), no
#: digits. Anything looser starts joining separate sentences into one equation.
_CLAIM_UNIT = r"(?:\s*[A-Za-z\u00b5%]+(?:\s*/\s*[A-Za-z\u00b5]+)?)?"
_CLAIM_TERM = rf"{_CLAIM_ATOM}{_CLAIM_UNIT}(?:\s*[-+*/x×]\s*{_CLAIM_ATOM}{_CLAIM_UNIT})*"
_CLAIM_FUNCTION = rf"(?:min|max)\(\s*{_CLAIM_TERM}(?:\s*,\s*{_CLAIM_TERM})+\s*\)"
_ARITHMETIC_CLAIM_RE = re.compile(
    rf"(?<![\w.])(?P<lhs>{_CLAIM_FUNCTION}|{_CLAIM_ATOM}{_CLAIM_UNIT}(?:\s*[-+*/x×]\s*{_CLAIM_ATOM}{_CLAIM_UNIT})+)"
    rf"\s*=\s*(?P<rhs>{_CLAIM_NUMBER})(?!\d)(?!\.\d)"
)
_CLAIM_REFUTATION_BEFORE_RE = re.compile(
    r"(?:false|incorrect|wrong|invalid|counterexample|mistake)(?:\s+(?:claim|equation|result))?"
    r"\s*[:;,\-—]*\s*$",
    re.IGNORECASE,
)
_CLAIM_REFUTATION_AFTER_RE = re.compile(
    r"^\s*(?:(?:is|was|would\s+be|looks?)\s+)?"
    r"(?:false|incorrect|wrong|invalid|a\s+mistake|not\s+correct)\b",
    re.IGNORECASE,
)


def _claim_is_refuted(text: str, start: int, end: int) -> bool:
    """Return True when prose presents an equation as an error, not a fact."""

    before = text[max(0, start - 80) : start]
    after = text[end : min(len(text), end + 80)]
    if _CLAIM_REFUTATION_BEFORE_RE.search(before):
        return True
    if _CLAIM_REFUTATION_AFTER_RE.search(after):
        return True
    # Do not rewrite quoted evidence. The enclosing speaker may be the subject
    # of the correction, and changing the quote would falsify the record.
    for quote in ('"', "“", "‘"):
        closing = {'“': '”', '‘': '’'}.get(quote, quote)
        open_at = text.rfind(quote, 0, start)
        close_at = text.find(closing, end)
        if open_at >= 0 and close_at >= 0 and "\n" not in text[open_at:close_at]:
            return True
    return False


def _format_arithmetic_value(value: float) -> str:
    if math.isfinite(value) and value.is_integer():
        return str(int(value))
    return format(value, ".15g")


def _normalize_arithmetic_expression(expr: str) -> str:
    """Normalize numeric spelling without destroying function separators."""

    cleaned = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", expr).replace("×", "*")
    # "x" between two numbers is multiplication; anywhere else it is a letter
    # inside a unit, so the blanket replace turned "6 x" into "6 *" and also
    # mangled any unit containing an x.
    cleaned = re.sub(r"(?<=\d)\s*x\s*(?=[\d(])", "*", cleaned)
    # Units are annotation, not arithmetic. min/max are the one alphabetic
    # construct this evaluator honours, so expressions using them are left
    # alone rather than being stripped down to their separators.
    if "min(" not in cleaned and "max(" not in cleaned:
        cleaned = re.sub(r"[A-Za-z\u00b5%]+(?:\s*/\s*[A-Za-z\u00b5]+)?", " ", cleaned)
    return cleaned


def _safe_arith(expr: str) -> float | None:
    """Evaluate a pure-numeric arithmetic expression safely (no names/calls)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b if b != 0 else float("nan")
            if isinstance(node.op, ast.Pow):
                return a ** b
            if isinstance(node.op, ast.Mod):
                return a % b if b != 0 else float("nan")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"min", "max"}
            and not node.keywords
            and 2 <= len(node.args) <= 8
        ):
            values = [ev(argument) for argument in node.args]
            return min(values) if node.func.id == "min" else max(values)
        raise ValueError("unsupported expression")

    try:
        val = ev(tree)
        return val if val == val else None  # reject NaN
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


class SymbolicBridge:
    """Routes formalizable work to exact solvers when available."""

    def simplify_math(self, expression: str) -> SymbolicResult:
        try:
            import sympy as sp

            expr = sp.sympify(expression)
            simplified = sp.simplify(expr)
            return SymbolicResult(True, "sympy", simplified, f"sympy.simplify({expression!r})")
        except (ImportError, AttributeError, RuntimeError) as exc:
            return SymbolicResult(False, "sympy", repr(exc), "solver_error")

    def evaluate(self, expression: str) -> SymbolicResult:
        """Exactly evaluate a math expression (sympy) — no LLM guessing.

        Handles arithmetic, fractions, powers, roots, and constants symbolically, then
        gives a numeric value. The exact engine for tool-augmented reasoning.
        """
        try:
            import sympy as sp

            expr = sp.sympify(expression, evaluate=True)
            if expr.free_symbols:
                value: Any = expr               # symbolic — leave it exact
            else:
                # Exact value: keep integers/rationals clean; float only if irrational.
                value = expr if (expr.is_Integer or expr.is_Rational) else sp.N(expr)
            return SymbolicResult(True, "sympy", value, f"sympy.evaluate({expression!r}) = {value}")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, SyntaxError) as exc:
            # Fall back to the sandboxed numeric evaluator for pure arithmetic.
            val = _safe_arith(str(expression))
            if val is not None:
                return SymbolicResult(True, "numeric_ast", val, f"numeric({expression!r})")
            return SymbolicResult(False, "sympy", repr(exc), "solver_error")

    def solve_equation(self, equation: str, symbol: str = "x") -> SymbolicResult:
        """Solve an equation/expression for a symbol, exactly (sympy)."""
        try:
            import sympy as sp

            sym = sp.Symbol(symbol)
            if "=" in equation:
                lhs, rhs = equation.split("=", 1)
                expr = sp.sympify(lhs) - sp.sympify(rhs)
            else:
                expr = sp.sympify(equation)
            roots = sp.solve(expr, sym)
            return SymbolicResult(True, "sympy", roots, f"sympy.solve({equation!r}, {symbol}) = {roots}")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return SymbolicResult(False, "sympy", repr(exc), "solver_error")

    def check_python_boolean(self, expression: str) -> SymbolicResult:
        try:
            tree = ast.parse(expression, mode="eval")
            value = _evaluate_boolean_ast(tree)
            return SymbolicResult(True, "python_ast", bool(value), "restricted_ast_eval")
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            return SymbolicResult(False, "python_ast", repr(exc), "solver_error")

    def inspect_arithmetic_claims(self, text: str) -> list[dict[str, Any]]:
        """Recompute asserted numeric equalities and return typed observations.

        Conservative: only evaluates pure-numeric arithmetic (no variables), so it
        never mis-flags algebra or rhetorical "=" usage. Catches a confidently
        stated calculation error in Aura's own reasoning.
        """
        observations: list[dict[str, Any]] = []
        source = str(text or "")
        for m in _ARITHMETIC_CLAIM_RE.finditer(source):
            if _claim_is_refuted(source, m.start(), m.end()):
                continue
            lhs_raw, rhs_raw = m.group("lhs"), m.group("rhs")
            lhs_val = _safe_arith(_normalize_arithmetic_expression(lhs_raw))
            if lhs_val is None:
                continue
            try:
                rhs_val = float(rhs_raw.replace(",", ""))
            except ValueError:
                continue
            correct = abs(lhs_val - rhs_val) <= 1e-6 * max(1.0, abs(rhs_val))
            observations.append(
                {
                    "claim": m.group(0).strip(),
                    "stated": rhs_val,
                    "correct": lhs_val,
                    "valid": correct,
                    "start": m.start("rhs"),
                    "end": m.end("rhs"),
                    "replacement": _format_arithmetic_value(lhs_val),
                }
            )
        return observations

    def check_arithmetic_claims(self, text: str) -> list[dict[str, Any]]:
        """Verify numeric ``expr = value`` claims in text; return the wrong ones."""

        return [
            observation
            for observation in self.inspect_arithmetic_claims(text)
            if not bool(observation["valid"])
        ]

    def repair_arithmetic_claims(self, text: str) -> tuple[str, list[ArithmeticClaimRepair]]:
        """Correct provably false numeric equalities without regenerating prose.

        Only the asserted right-hand numeric span changes. Quoted equations and
        equations explicitly described as false remain verbatim. The operation
        is deterministic and idempotent, so it can run at more than one response
        boundary without accumulating mutations.
        """

        source = str(text or "")
        errors = self.check_arithmetic_claims(source)
        repairs = [
            ArithmeticClaimRepair(
                claim=str(error["claim"]),
                stated=float(error["stated"]),
                correct=float(error["correct"]),
                start=int(error["start"]),
                end=int(error["end"]),
                replacement=str(error["replacement"]),
            )
            for error in errors
        ]
        repaired = source
        for repair in reversed(repairs):
            repaired = repaired[: repair.start] + repair.replacement + repaired[repair.end :]
        return repaired, repairs

    def audit_reasoning(self, text: str) -> dict[str, Any]:
        """Active-reasoning gateway: route logic to the prover, arithmetic to sympy.

        The single live entry point that exercises the bridge on Aura's own output —
        deductive non-sequiturs via the natural-deduction prover and calculation
        errors via numeric evaluation. Returns both findings.
        """
        non_sequiturs: list[dict[str, Any]] = []
        checked_inferences = 0
        try:
            from core.reasoning.inference_audit import audit_text

            inference_verdicts = audit_text(text)
            checked_inferences = sum(
                verdict.status in {"valid", "invalid"} for verdict in inference_verdicts
            )
            non_sequiturs = [
                verdict.to_dict()
                for verdict in inference_verdicts
                if verdict.is_non_sequitur
            ]
        except (ImportError, ValueError, RuntimeError, TypeError, AttributeError):
            non_sequiturs = []
        arithmetic_claims = self.inspect_arithmetic_claims(text)
        arithmetic_errors = [claim for claim in arithmetic_claims if not claim["valid"]]
        return {
            "non_sequiturs": non_sequiturs,
            "arithmetic_errors": arithmetic_errors,
            "checked_inferences": checked_inferences,
            "checked_arithmetic_claims": len(arithmetic_claims),
            "checked": bool(checked_inferences or arithmetic_claims),
            "clean": not non_sequiturs and not arithmetic_errors,
        }

    def prove_logic(self, premises: list[str], goal: str) -> SymbolicResult:
        """Exact propositional deduction, kernel-checked (de Bruijn criterion).

        Routes a ``premises ⊢ goal`` query through :func:`prove_certified`: the
        tableau search produces a certificate and the independent proof kernel
        re-verifies it. A search-claimed proof the kernel rejects is **not**
        reported as proved — the bridge fails closed on unsoundness. The trace
        carries the kernel verdict and the axiom audit (premises actually used).
        """
        try:
            from core.reasoning.proof_kernel import prove_certified_text

            cp = prove_certified_text(premises, goal)
            proof = cp.proof
            if not proof.provable:
                return SymbolicResult(
                    True, "natural_deduction", False, f"countermodel: {proof.countermodel}"
                )
            if not cp.verified:
                reason = cp.verdict.reason if cp.verdict else "no certificate"
                return SymbolicResult(
                    False, "natural_deduction", "kernel_rejected", f"kernel refused proof: {reason}"
                )
            assert cp.verdict is not None
            trace = (
                " ; ".join(proof.trace)
                + f" ; kernel: verified ({cp.verdict.nodes} nodes)"
                + f" ; axioms: {list(cp.verdict.used_premises)}"
            )
            return SymbolicResult(True, "natural_deduction", True, trace)
        except (ValueError, RuntimeError, AttributeError, TypeError) as exc:
            return SymbolicResult(False, "natural_deduction", repr(exc), "solver_error")

    def prove_linear(self, premises: list[str], goal: str) -> SymbolicResult:
        """Certified linear-arithmetic entailment (Farkas witness, kernel-checked).

        ``Γ ⊢ g`` over exact rationals: the Fourier–Motzkin search finds a
        Farkas certificate and the independent checker verifies it — a
        provable=True verdict is always kernel-verified (fail closed).
        provable=False means *not entailed by this decision procedure*.
        """
        try:
            from core.reasoning.linear_arithmetic import prove_linear

            lp = prove_linear(premises, goal)
            if not lp.provable:
                return SymbolicResult(True, "farkas_linear", False, "no Farkas refutation found")
            if not lp.verified:
                reason = lp.verdict.reason if lp.verdict else "no verdict"
                return SymbolicResult(False, "farkas_linear", "kernel_rejected", reason)
            assert lp.verdict is not None
            return SymbolicResult(
                True,
                "farkas_linear",
                True,
                f"kernel-verified Farkas witness ; axioms: {list(lp.verdict.used_premises)}",
            )
        except (ValueError, RuntimeError, AttributeError, TypeError) as exc:
            return SymbolicResult(False, "farkas_linear", repr(exc), "solver_error")

    def solve_constraints(self, constraints: list[str]) -> SymbolicResult:
        """Constraint (in)feasibility — certified Farkas first, z3 fallback.

        Linear systems get the exact, kernel-checked decision procedure: a
        proved-infeasible verdict carries a verified certificate. Nonlinear
        or unresolved systems fall back to (unverified) z3 when available.
        """
        try:
            from core.reasoning.linear_arithmetic import check_feasible

            lp = check_feasible(constraints)
            if lp.provable and lp.verified:
                return SymbolicResult(
                    True, "farkas_linear", "unsat", "kernel-verified Farkas infeasibility witness"
                )
        except (ValueError, RuntimeError, AttributeError, TypeError):
            pass  # not linear / not parseable here — fall through to z3
        try:
            import z3  # type: ignore

            solver = z3.Solver()
            names: dict[str, Any] = {}
            for raw in constraints:
                tree = ast.parse(raw, mode="eval")
                solver.add(_z3_from_ast(tree, names, z3))
            status = solver.check()
            return SymbolicResult(True, "z3", status, str(solver.model()) if status == z3.sat else str(status))
        except (ImportError, AttributeError, RuntimeError) as exc:
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

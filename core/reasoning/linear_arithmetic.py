"""Certified linear arithmetic — mathlib's ``linarith``/``omega`` discipline.

A decision procedure for conjunctions of linear constraints over the
rationals, with the same de Bruijn split the tableau prover got:

- The **search** is Fourier–Motzkin elimination (untrusted, free to be
  clever). When the system is infeasible it emits a **Farkas certificate** —
  nonnegative multipliers λ over the original constraints whose exact linear
  combination cancels every variable and leaves an impossible constant
  relation (``0 ≤ r`` with ``r < 0``, or a strict ``0 < 0``).
- The **checker** re-verifies the certificate independently with exact
  ``Fraction`` arithmetic: multiply, sum, confirm cancellation and the
  contradiction. It never trusts the elimination order or the search.

Entailment is refutation: ``Γ ⊢ g`` iff ``Γ ∪ {¬g}`` is infeasible, so the
Farkas witness *is* the proof, and the multipliers give the axiom audit for
free (λᵢ > 0 ⇔ premise i was actually used).

Registered with the proof kernel as method ``farkas_linear`` — the second
citizen of the universal checker registry — and recorded in the
``TheoremLedger`` with a replay codec, so arithmetic theorems are re-verified
by the same homeostate trust leg that replays tableau proofs. Live surface:
``SymbolicBridge.prove_linear`` / upgraded ``solve_constraints``.
"""
from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

from core.reasoning.proof_kernel import KernelVerdict, register_checker

# Fail-closed bounds for the elimination search (the checker needs none —
# checking a certificate is linear in its size).
_MAX_CONSTRAINTS = 4_000
_MAX_VARS = 64


# ── Constraints ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LinConstraint:
    """Normalized linear constraint: ``Σ coeffᵢ·varᵢ ≤ rhs`` (or ``<`` when strict)."""

    coeffs: tuple[tuple[str, Fraction], ...]   # sorted, zero-free
    rhs: Fraction
    strict: bool = False

    def __str__(self) -> str:
        if not self.coeffs:
            lhs = "0"
        else:
            parts = []
            for var, c in self.coeffs:
                parts.append(f"{c}*{var}" if c != 1 else var)
            lhs = " + ".join(parts)
        return f"{lhs} {'<' if self.strict else '<='} {self.rhs}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "coeffs": [[v, str(c)] for v, c in self.coeffs],
            "rhs": str(self.rhs),
            "strict": self.strict,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "LinConstraint":
        return LinConstraint(
            coeffs=tuple((str(v), Fraction(c)) for v, c in data["coeffs"]),
            rhs=Fraction(data["rhs"]),
            strict=bool(data["strict"]),
        )


def _make(coeffs: Mapping[str, Fraction], rhs: Fraction, strict: bool) -> LinConstraint:
    clean = tuple(sorted((v, c) for v, c in coeffs.items() if c != 0))
    return LinConstraint(clean, rhs, strict)


def negate_constraint(c: LinConstraint) -> LinConstraint:
    """¬(e ≤ b) = e > b = -e < -b ; ¬(e < b) = e ≥ b = -e ≤ -b."""
    return _make({v: -k for v, k in c.coeffs}, -c.rhs, not c.strict)


# ── Parser: "2*x + 3*y <= 5", "x - y > 1", "x = 4" ───────────────────────

def parse_constraint(text: str) -> list[LinConstraint]:
    """Parse a linear (in)equality; equalities produce two ≤ constraints."""
    text = str(text or "").strip()
    for op, flip, strict, eq in (
        ("<=", False, False, False), (">=", True, False, False),
        ("==", False, False, True), ("!=", None, None, None),
        ("<", False, True, False), (">", True, True, False),
        ("=", False, False, True),
    ):
        if op == "!=":
            if "!=" in text:
                raise ValueError("disequalities are not supported (not convex)")
            continue
        idx = text.find(op)
        if idx < 0:
            continue
        # Guard against splitting "<=" at "<" etc. — ops are tried longest-first.
        lhs_txt, rhs_txt = text[:idx], text[idx + len(op):]
        lhs = _parse_linear(lhs_txt)
        rhs = _parse_linear(rhs_txt)
        # Move everything left: (lhs - rhs) OP 0  →  coeffs ≤ const form.
        coeffs = dict(lhs[0])
        for v, c in rhs[0].items():
            coeffs[v] = coeffs.get(v, Fraction(0)) - c
        const = rhs[1] - lhs[1]
        if flip:
            coeffs = {v: -c for v, c in coeffs.items()}
            const = -const
        if eq:
            return [
                _make(coeffs, const, False),
                _make({v: -c for v, c in coeffs.items()}, -const, False),
            ]
        return [_make(coeffs, const, strict)]
    raise ValueError(f"no comparison operator in constraint: {text!r}")


def _parse_linear(expr: str) -> tuple[dict[str, Fraction], Fraction]:
    """Parse a linear expression into (coeffs, constant), exactly."""
    tree = ast.parse(expr.strip(), mode="eval")

    def walk(node: ast.AST) -> tuple[dict[str, Fraction], Fraction]:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return {}, Fraction(str(node.value))
        if isinstance(node, ast.Name):
            return {node.id: Fraction(1)}, Fraction(0)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            coeffs, const = walk(node.operand)
            if isinstance(node.op, ast.USub):
                return {v: -c for v, c in coeffs.items()}, -const
            return coeffs, const
        if isinstance(node, ast.BinOp):
            left_c, left_k = walk(node.left)
            right_c, right_k = walk(node.right)
            if isinstance(node.op, ast.Add):
                out = dict(left_c)
                for v, c in right_c.items():
                    out[v] = out.get(v, Fraction(0)) + c
                return out, left_k + right_k
            if isinstance(node.op, ast.Sub):
                out = dict(left_c)
                for v, c in right_c.items():
                    out[v] = out.get(v, Fraction(0)) - c
                return out, left_k - right_k
            if isinstance(node.op, ast.Mult):
                if not left_c:       # constant * linear
                    return {v: c * left_k for v, c in right_c.items()}, left_k * right_k
                if not right_c:      # linear * constant
                    return {v: c * right_k for v, c in left_c.items()}, left_k * right_k
                raise ValueError("nonlinear term (variable * variable)")
            if isinstance(node.op, ast.Div):
                if right_c:
                    raise ValueError("division by a variable is nonlinear")
                if right_k == 0:
                    raise ValueError("division by zero")
                return {v: c / right_k for v, c in left_c.items()}, left_k / right_k
        raise ValueError(f"unsupported node in linear expression: {type(node).__name__}")

    return walk(tree)


# ── Farkas certificates ───────────────────────────────────────────────────

@dataclass(frozen=True)
class FarkasCertificate:
    """Nonnegative multipliers over the original constraint list.

    The checkable object: Σ λᵢ·constraintᵢ must cancel every variable and
    leave an impossible constant relation.
    """

    multipliers: tuple[tuple[int, Fraction], ...]   # (constraint index, λ > 0)

    def to_dict(self) -> dict[str, Any]:
        return {"multipliers": [[i, str(m)] for i, m in self.multipliers]}

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FarkasCertificate":
        return FarkasCertificate(
            multipliers=tuple((int(i), Fraction(m)) for i, m in data["multipliers"])
        )


def check_farkas(constraints: Sequence[LinConstraint], certificate: FarkasCertificate) -> KernelVerdict:
    """Independent kernel check of a Farkas infeasibility certificate.

    Exact rational arithmetic; trusts nothing about how the multipliers were
    found. Verifies λ ≥ 0, full cancellation of every variable, and that the
    combined constant relation is genuinely contradictory.
    """
    combined: dict[str, Fraction] = {}
    rhs = Fraction(0)
    strict = False
    used: list[int] = []
    for idx, lam in certificate.multipliers:
        if lam < 0:
            return KernelVerdict(False, f"kernel rejected: negative multiplier λ[{idx}]")
        if lam == 0:
            continue
        if idx < 0 or idx >= len(constraints):
            return KernelVerdict(False, f"kernel rejected: multiplier for unknown constraint {idx}")
        con = constraints[idx]
        for v, c in con.coeffs:
            combined[v] = combined.get(v, Fraction(0)) + lam * c
        rhs += lam * con.rhs
        strict = strict or con.strict
        used.append(idx)
    leftover = {v: c for v, c in combined.items() if c != 0}
    if leftover:
        return KernelVerdict(False, f"kernel rejected: variables not cancelled: {sorted(leftover)}")
    contradiction = rhs < 0 or (rhs == 0 and strict)
    if not contradiction:
        return KernelVerdict(
            False, f"kernel rejected: combination is satisfiable (0 {'<' if strict else '<='} {rhs})"
        )
    return KernelVerdict(
        True,
        "verified: Farkas combination cancels all variables into a contradiction",
        used_premises=tuple(str(constraints[i]) for i in sorted(set(used))),
        nodes=len(certificate.multipliers),
    )


# ── Fourier–Motzkin search (untrusted, certificate-producing) ─────────────

@dataclass
class _Tracked:
    combo: dict[int, Fraction]      # original-index → λ
    coeffs: dict[str, Fraction]
    rhs: Fraction
    strict: bool


def find_farkas(constraints: Sequence[LinConstraint]) -> FarkasCertificate | None:
    """Fourier–Motzkin elimination; returns a Farkas certificate if infeasible.

    None means the system is satisfiable (or the search hit its size bound —
    callers must treat None as *not proved infeasible*, never as verified).
    """
    work: list[_Tracked] = [
        _Tracked({i: Fraction(1)}, dict(c.coeffs), c.rhs, c.strict)
        for i, c in enumerate(constraints)
    ]
    variables = sorted({v for t in work for v in t.coeffs})
    if len(variables) > _MAX_VARS:
        return None

    def contradiction(t: _Tracked) -> bool:
        return not t.coeffs and (t.rhs < 0 or (t.rhs == 0 and t.strict))

    for t in work:
        t.coeffs = {v: c for v, c in t.coeffs.items() if c != 0}
        if contradiction(t):
            return FarkasCertificate(tuple(sorted(t.combo.items())))

    for var in variables:
        pos = [t for t in work if t.coeffs.get(var, Fraction(0)) > 0]
        neg = [t for t in work if t.coeffs.get(var, Fraction(0)) < 0]
        rest = [t for t in work if t.coeffs.get(var, Fraction(0)) == 0]
        combined: list[_Tracked] = []
        for p, n in itertools.product(pos, neg):
            ap = p.coeffs[var]
            an = -n.coeffs[var]
            # p/ap + n/an cancels var; both multipliers are positive.
            combo: dict[int, Fraction] = {}
            for i, lam in p.combo.items():
                combo[i] = combo.get(i, Fraction(0)) + lam / ap
            for i, lam in n.combo.items():
                combo[i] = combo.get(i, Fraction(0)) + lam / an
            coeffs: dict[str, Fraction] = {}
            for v, c in p.coeffs.items():
                coeffs[v] = coeffs.get(v, Fraction(0)) + c / ap
            for v, c in n.coeffs.items():
                coeffs[v] = coeffs.get(v, Fraction(0)) + c / an
            coeffs = {v: c for v, c in coeffs.items() if c != 0}
            t = _Tracked(combo, coeffs, p.rhs / ap + n.rhs / an, p.strict or n.strict)
            if contradiction(t):
                return FarkasCertificate(tuple(sorted(t.combo.items())))
            combined.append(t)
        work = rest + combined
        if len(work) > _MAX_CONSTRAINTS:
            return None
    for t in work:
        if contradiction(t):
            return FarkasCertificate(tuple(sorted(t.combo.items())))
    return None


# ── Certified entailment ──────────────────────────────────────────────────

@dataclass
class LinearProof:
    """Outcome of a certified linear-arithmetic entailment query."""

    premises: list[LinConstraint]
    goal: LinConstraint | None
    provable: bool
    verdict: KernelVerdict | None
    certificate: FarkasCertificate | None

    @property
    def verified(self) -> bool:
        return bool(self.verdict and self.verdict.verified)

    def to_dict(self) -> dict[str, Any]:
        return {
            "premises": [str(p) for p in self.premises],
            "goal": str(self.goal) if self.goal is not None else None,
            "provable": self.provable,
            "verified": self.verified,
            "kernel": self.verdict.to_dict() if self.verdict else None,
        }


def _kernel_check_farkas(premises: Sequence[Any], goal: Any, certificate: Any) -> KernelVerdict:
    """Adapter matching the universal checker registry signature."""
    if not isinstance(certificate, FarkasCertificate):
        return KernelVerdict(False, "kernel rejected: not a Farkas certificate")
    constraints = list(premises)
    if goal is not None:
        constraints = constraints + [negate_constraint(goal)]
    verdict = check_farkas(constraints, certificate)
    if not verdict.verified or goal is None:
        return verdict
    neg_index = len(constraints) - 1
    used_idx = {i for i, lam in certificate.multipliers if lam > 0}
    return KernelVerdict(
        verdict.verified,
        verdict.reason,
        used_premises=tuple(
            str(constraints[i]) for i in sorted(used_idx) if i != neg_index
        ),
        uses_goal_negation=neg_index in used_idx,
        nodes=verdict.nodes,
    )


register_checker("farkas_linear", _kernel_check_farkas)


def prove_linear(premises: Iterable[str | LinConstraint], goal: str | LinConstraint) -> LinearProof:
    """Certified ``Γ ⊢ g`` over linear rational arithmetic (refutation + Farkas).

    Splitting text equalities into their two inequalities is handled; the
    goal must normalize to a single inequality. A found refutation is only
    reported *verified* after the independent checker accepts the multipliers
    — the same fail-closed contract as the tableau kernel.
    """
    prem: list[LinConstraint] = []
    for p in premises:
        prem.extend(parse_constraint(p) if isinstance(p, str) else [p])
    goals = parse_constraint(goal) if isinstance(goal, str) else [goal]
    if len(goals) != 1:
        raise ValueError("goal must be a single inequality (equalities are two goals)")
    g = goals[0]
    system = prem + [negate_constraint(g)]
    certificate = find_farkas(system)
    if certificate is None:
        return LinearProof(prem, g, False, None, None)
    verdict = _kernel_check_farkas(prem, g, certificate)
    _note_and_record(prem, g, certificate, verdict)
    return LinearProof(prem, g, True, verdict, certificate)


def check_feasible(constraints: Iterable[str | LinConstraint]) -> LinearProof:
    """Certified (in)feasibility of a constraint system.

    ``provable`` means *infeasibility proved* (with a checked certificate);
    a feasible or unresolved system reports provable=False.
    """
    cons: list[LinConstraint] = []
    for c in constraints:
        cons.extend(parse_constraint(c) if isinstance(c, str) else [c])
    certificate = find_farkas(cons)
    if certificate is None:
        return LinearProof(cons, None, False, None, None)
    verdict = _kernel_check_farkas(cons, None, certificate)
    _note_and_record(cons, None, certificate, verdict)
    return LinearProof(cons, None, True, verdict, certificate)


def _note_and_record(
    premises: Sequence[LinConstraint],
    goal: LinConstraint | None,
    certificate: FarkasCertificate,
    verdict: KernelVerdict,
) -> None:
    """Ledger bookkeeping + fail-closed degradation on kernel rejection."""
    from core.reasoning.proof_kernel import get_theorem_ledger

    ledger = get_theorem_ledger()
    ledger.note_check(verdict.verified)
    if verdict.verified:
        ledger.record_external(
            method="farkas_linear",
            goal=str(goal) if goal is not None else "⊥(infeasible system)",
            premises=[str(p) for p in premises],
            used_premises=verdict.used_premises,
            encoded={
                "premises": [p.to_dict() for p in premises],
                "goal": goal.to_dict() if goal is not None else None,
                "certificate": certificate.to_dict(),
            },
        )
        return
    try:
        from core.runtime.errors import record_degradation

        record_degradation(
            "proof_kernel",
            RuntimeError(verdict.reason),
            severity="critical",
            action="kernel rejected a Farkas certificate; reporting unverified",
            extra={"goal": str(goal), "premises": [str(p) for p in premises]},
            enforce_failure_policy=False,
        )
    except (ImportError, RuntimeError, ValueError, TypeError, AttributeError):
        pass


def _replay_farkas(encoded: Mapping[str, Any]) -> bool:
    premises = [LinConstraint.from_dict(p) for p in encoded["premises"]]
    goal = LinConstraint.from_dict(encoded["goal"]) if encoded.get("goal") else None
    certificate = FarkasCertificate.from_dict(encoded["certificate"])
    return _kernel_check_farkas(premises, goal, certificate).verified


def install_replay_codec() -> None:
    """Register the Farkas replay codec with the theorem ledger."""
    from core.reasoning.proof_kernel import register_replay_codec

    register_replay_codec("farkas_linear", _replay_farkas)


install_replay_codec()


__all__ = [
    "FarkasCertificate",
    "LinConstraint",
    "LinearProof",
    "check_farkas",
    "check_feasible",
    "find_farkas",
    "negate_constraint",
    "parse_constraint",
    "prove_linear",
]

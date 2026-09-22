"""Exact witnesses for the capacity of a frozen linear score representation.

Numerical optimization proposes weights or a small contradictory subset.
Acceptance checks use rational arithmetic and Aura's existing proof kernel.
This diagnoses the declared comparisons, not the truth of their labels or
the generalization of a learned semantic model.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

import numpy as np

from core.reasoning.linear_arithmetic import (
    FarkasCertificate,
    LinConstraint,
    check_farkas,
    find_farkas,
)
from core.verify.invariants import invariant


def _identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def score_constraints(
    differences: Any,
    offsets: Any,
    *,
    margin: float=1.0,
    lower_bounds: Any=None,
    strict: bool=False,
) -> tuple[Any, ...]:
    """Represent d[i] @ weights + offset[i] >= margin with exact binary-float values."""
    matrix = np.asarray(differences, dtype=np.float64)
    offset = np.asarray(offsets, dtype=np.float64)
    if (matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]
            or offset.shape != (matrix.shape[0],)
            or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset))
            or type(strict) is not bool or isinstance(margin, bool)
            or not np.isscalar(margin) or not np.isfinite(margin) or margin < 0
            or (margin == 0 and not strict)):
        raise ValueError("capacity requires finite comparisons and a positive margin or strict ranking")
    width = matrix.shape[1]
    lower = np.zeros(width) if lower_bounds is None else np.asarray(lower_bounds, dtype=np.float64)
    if lower.shape != (width,) or not np.all(np.isfinite(lower)):
        raise ValueError("coefficient lower bounds differ from score features")
    required = Fraction(float(margin))
    constraints = [LinConstraint(
        tuple((f"w{j}", -Fraction(float(value))) for j, value in enumerate(row) if value != 0),
        Fraction(float(bias)) - required,
        strict,
    ) for row, bias in zip(matrix, offset, strict=True)]
    constraints.extend(LinConstraint(((f"w{j}", Fraction(-1)),), -Fraction(float(value)))
                       for j, value in enumerate(lower))
    return tuple(constraints)


def _weights_satisfy(constraints: tuple[Any, ...], weights: dict[str, Any]) -> bool:
    for constraint in constraints:
        if any(name not in weights for name, _ in constraint.coeffs):
            return False
        actual = sum((coefficient * weights[name] for name, coefficient in constraint.coeffs), Fraction(0))
        if actual > constraint.rhs or (constraint.strict and actual == constraint.rhs):
            return False
    return True


def assess_score_capacity(
    differences: list[Any],
    offsets: list[Any],
    *,
    margin: float=1.0,
    lower_bounds: Any=None,
    comparison_ids: Any=None,
    max_witness_constraints: int=32,
    strict: bool=False,
) -> dict:
    """Seek a checkable witness, preserving unknown when either search is inconclusive."""
    from scipy.optimize import linprog

    constraints = score_constraints(differences, offsets, margin=margin, lower_bounds=lower_bounds, strict=strict)
    matrix = np.asarray(differences, dtype=np.float64)
    count, width = matrix.shape
    ids = tuple(comparison_ids) if comparison_ids is not None else tuple(str(i) for i in range(count))
    if (len(ids) != count or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("capacity comparison identities must be unique and complete")
    if type(max_witness_constraints) is not int or max_witness_constraints < 1:
        raise ValueError("witness search bound must be positive")
    a = np.zeros((len(constraints), width))
    for index, constraint in enumerate(constraints):
        for variable, coefficient in constraint.coeffs:
            a[index, int(variable[1:])] = float(coefficient)
    b = np.asarray([float(c.rhs) for c in constraints])
    problem = {
        "constraints": [c.to_dict() for c in constraints], "comparison_ids": list(ids),
        "variables": [f"w{i}" for i in range(width)], "comparison_count": count,
        "arithmetic": "exact_rational_values_of_frozen_binary64_coefficients",
    }
    report = {
        "schema": "aura.score_capacity.v1", "problem": problem, "problem_sha256": _identity(problem),
        "status": "unknown", "verified": False, "weights": None, "certificate": None,
        "conflicting_comparison_ids": [], "serving_authority": False,
        "scope": "frozen_score_comparisons_only", "max_witness_constraints": max_witness_constraints,
    }
    if strict:
        # A shared positive slack exists exactly when this finite strict
        # comparison set is feasible. Its cap bounds the objective, not weights.
        slack = np.array([float(c.strict) for c in constraints])
        primal = linprog(np.r_[np.zeros(width), -1.], A_ub=np.column_stack((a, slack)),
                         b_ub=b, bounds=[(None, None)] * width + [(0., 1.)], method="highs")
    else:
        primal = linprog(np.zeros(width), A_ub=a, b_ub=b, bounds=[(None, None)] * width, method="highs")
    report["numerical_status"] = int(primal.status)
    if primal.success and np.all(np.isfinite(primal.x)):
        weights = {f"w{i}": Fraction(float(value)) for i, value in enumerate(primal.x[:width])}
        if _weights_satisfy(constraints, weights):
            report.update(status="feasible", verified=True, weights={k: str(v) for k, v in weights.items()})
            return report
    if primal.status != 2 and not (strict and primal.success):
        report["reason"] = "numerical_weights_not_exactly_verified" if primal.success else "numerical_search_incomplete"
        return report
    # Find sparse support for a contradiction; the floating-point multipliers
    # are never accepted as a proof. Eliminate only the proposed small subset.
    dual = linprog(b, A_eq=np.vstack((a.T, np.ones(len(constraints)))),
                   b_eq=np.r_[np.zeros(width), 1.0], bounds=(0, None), method="highs")
    if not dual.success or not np.all(np.isfinite(dual.x)):
        report["reason"] = "contradiction_support_unavailable"
        return report
    support = tuple(int(i) for i in np.flatnonzero(dual.x > 0))
    if not support or len(support) > max_witness_constraints:
        report["reason"] = "contradiction_support_exceeds_search_bound"
        return report
    local = find_farkas(tuple(constraints[i] for i in support))
    if local is None:
        report["reason"] = "exact_contradiction_not_found"
        return report
    certificate = FarkasCertificate(tuple((support[i], multiplier) for i, multiplier in local.multipliers))
    verdict = check_farkas(constraints, certificate)
    if not verdict.verified:
        report["reason"] = "exact_kernel_rejected_contradiction"
        return report
    used = sorted({i for i, multiplier in certificate.multipliers if multiplier > 0 and i < count})
    report.update(status="infeasible", verified=True, certificate=certificate.to_dict(),
                  conflicting_comparison_ids=[ids[i] for i in used])
    return report


def verify_score_capacity(report: dict) -> bool:
    """Replay a result without numerical optimization or trusting its status flag."""
    try:
        problem = report["problem"]
        if report.get("schema") != "aura.score_capacity.v1" or _identity(problem) != report["problem_sha256"]:
            return False
        constraints = tuple(LinConstraint.from_dict(c) for c in problem["constraints"])
        if not constraints or report.get("verified") is not True:
            return False
        if report["status"] == "feasible":
            return _weights_satisfy(constraints, {k: Fraction(v) for k, v in report["weights"].items()})
        if report["status"] == "infeasible":
            return check_farkas(constraints, FarkasCertificate.from_dict(report["certificate"])).verified
    # not a failure: a report this cannot read does not verify, and False is the
    # refusing direction.
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False
    return False


@invariant("learning.score_capacity_requires_exact_witness", scope="learning",
           owner="core/learning/score_capacity.py", observational=False)
def _capacity_requires_witness() -> tuple:
    report = assess_score_capacity([[1.], [-1.]], [0., 0.])
    assert report["status"] == "infeasible" and verify_score_capacity(report)
    report["certificate"]["multipliers"] = []
    assert not verify_score_capacity(report)
    return ()

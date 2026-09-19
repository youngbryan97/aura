"""Minimum-change repairs of affine margins, with independently replayable bounds.

The solver proposes a displacement. Feasibility and the primal/dual gap are
recomputed from that displacement; optimizer success is not a certificate.
These numerical bounds concern supplied comparisons, not unobserved labels.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class MarginRepair:
    displacement: np.ndarray
    multipliers: np.ndarray
    receipt: dict


def verify_exact_margin_repair(normals, required, displacement, multipliers):
    """Check a supplied finite rational witness without solving or tolerances."""
    def rational(value):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("boolean is not a rational coefficient")
        return Fraction(float(value)) if isinstance(value, np.floating) else Fraction(value)

    try:
        a = tuple(tuple(rational(v) for v in row) for row in normals)
        b, d, lam = (tuple(rational(v) for v in part)
                     for part in (required, displacement, multipliers))
        if not a or not d or len(b) != len(a) or len(lam) != len(a) or any(len(row) != len(d) for row in a):
            raise ValueError("rational margin witness geometry differs")
        slack = tuple(sum(x * y for x, y in zip(row, d, strict=True)) - target
                      for row, target in zip(a, b, strict=True))
        dual_vector = tuple(sum(row[j] * multiplier for row, multiplier in zip(a, lam, strict=True))
                            for j in range(len(d)))
        primal = sum(value * value for value in d) / 2
        dual = sum(x * y for x, y in zip(b, lam, strict=True)) - sum(v * v for v in dual_vector) / 2
        verified = all(v >= 0 for v in (*slack, *lam)) and primal == dual
        return {"verified": verified, "arithmetic": "exact_rational", "duality_gap": str(primal - dual),
                "primal_objective": str(primal), "dual_objective": str(dual),
                "scope": "supplied_affine_comparisons", "serving_authority": False}
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("invalid rational margin repair witness") from exc


def verify_margin_repair(normals, required, displacement, multipliers, *, tolerance=1e-8):
    """Replay weak duality for min 0.5*||d||^2 subject to A*d >= b."""
    a, b, d, lam = (np.asarray(value, dtype=np.float64)
                    for value in (normals, required, displacement, multipliers))
    if (a.ndim != 2 or not all(a.shape) or b.shape != (len(a),)
            or d.shape != (a.shape[1],) or lam.shape != b.shape
            or not all(np.all(np.isfinite(v)) for v in (a, b, d, lam))
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("invalid affine margin repair geometry or arithmetic")
    slack = a @ d - b
    dual_vector = a.T @ lam
    primal = float(.5 * (d @ d))
    dual = float(b @ lam - .5 * (dual_vector @ dual_vector))
    gap = primal - dual
    scale = max(1., abs(primal), abs(dual))
    feasible = bool(np.all(slack >= -tolerance))
    dual_feasible = bool(np.all(lam >= 0.))
    return {
        "schema": "aura.affine_margin_repair.v1",
        "status": "verified_numerically" if feasible and dual_feasible
        and -tolerance * scale <= gap <= tolerance * scale else "unverified",
        "primal_feasible": feasible, "dual_feasible": dual_feasible,
        "primal_objective": primal, "dual_objective": dual, "duality_gap": gap,
        "minimum_slack": float(slack.min()), "tolerance": tolerance,
        "displacement_norm": float(np.linalg.norm(d)),
        "stationarity_residual": float(np.linalg.norm(d - dual_vector)),
        "scope": "supplied_affine_comparisons", "exact_arithmetic_proof": False,
        "infeasibility_proven": False, "serving_authority": False,
    }


def minimum_margin_repair(normals, required, *, max_iterations=1000, tolerance=1e-8):
    """Solve the nonnegative dual; return unresolved when replay does not verify."""
    a, b = np.asarray(normals, dtype=np.float64), np.asarray(required, dtype=np.float64)
    if (a.ndim != 2 or not all(a.shape) or b.shape != (len(a),)
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
            or type(max_iterations) is not int or max_iterations < 1
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("invalid affine margin repair problem")
    lengths = np.linalg.norm(a, axis=1)
    scales = np.where(lengths > 0., lengths, 1.)
    normalized, target = a / scales[:, None], b / scales
    gram = normalized @ normalized.T

    def objective(lam):
        product = gram @ lam
        return float(.5 * (lam @ product) - target @ lam), product - target

    result = minimize(objective, np.zeros(len(a)), jac=True, method="L-BFGS-B",
        bounds=[(0., None)] * len(a),
        options={"maxiter": max_iterations, "ftol": np.finfo(float).eps,
                 "gtol": tolerance * .01, "maxls": 40})
    lam = np.maximum(result.x, 0.) / scales
    displacement = a.T @ lam
    if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(lam)):
        displacement, lam = np.zeros(a.shape[1]), np.zeros(len(a))
    receipt = verify_margin_repair(a, b, displacement, lam, tolerance=tolerance)
    polished = False
    support = np.flatnonzero(result.x > 0.)
    if len(support):
        # L-BFGS can stop on objective precision before its binding margins
        # resolve. Solve its proposed active equalities and independently
        # verify the whole problem; support is a proposal, not an authority.
        values = np.linalg.lstsq(gram[np.ix_(support, support)], target[support], rcond=None)[0]
        if np.all(values >= 0.) and np.all(np.isfinite(values)):
            candidate = np.zeros(len(a))
            candidate[support] = values / scales[support]
            candidate_displacement = a.T @ candidate
            checked = verify_margin_repair(a, b, candidate_displacement, candidate, tolerance=tolerance)
            if checked["status"] == "verified_numerically":
                displacement, lam, receipt, polished = candidate_displacement, candidate, checked, True
    receipt.update(solver_status=int(result.status), solver_iterations=int(result.nit),
                   active_equalities_polished=polished)
    return MarginRepair(displacement, lam, receipt)


def minimum_stored_margin_repair(normals, required, anchor, *, max_rounding_rounds=8,
                                 max_iterations=1000, tolerance=1e-7):
    """Repair affine margins in the precision that the model actually stores.

    For rounding error e, a_i*(d+e) >= a_i*d - |a_i|*|e|. Add
    a reserve only to violated rows, solve again, and check stored values
    against the original requirements. Tight equalities can still succeed
    without an artificial interior. Exhaustion does not prove infeasibility.
    """
    a, b, origin = (np.asarray(value, dtype=np.float64)
                    for value in (normals, required, anchor))
    if (a.ndim != 2 or origin.shape != (a.shape[1],)
            or not np.all(np.isfinite(origin))
            or type(max_rounding_rounds) is not int or max_rounding_rounds < 1):
        raise ValueError("invalid stored margin repair geometry")
    reserves = np.zeros_like(b)
    stored = np.zeros_like(origin)
    minimum_slack = None
    feasible = False
    for _iteration in range(max_rounding_rounds):
        proposal = minimum_margin_repair(a, b + reserves,
            max_iterations=max_iterations, tolerance=tolerance)
        if proposal.receipt["status"] != "verified_numerically":
            break
        point = origin + proposal.displacement
        with np.errstate(over="ignore"):
            rounded = point.astype(np.float32).astype(np.float64)
        if not np.all(np.isfinite(rounded)):
            break
        stored = rounded - origin
        slack = a @ stored - b
        minimum_slack = float(slack.min())
        violated = slack < 0.
        if not np.any(violated):
            feasible = True
            break
        # Nearest float32 rounding has relative error <= 2^-24, plus
        # half a subnormal quantum. The original constraints stay unchanged.
        error_bound = np.abs(a[violated]) @ (
            np.finfo(np.float32).eps * .5 * np.abs(point) + 2. ** -150)
        reserves[violated] += np.maximum(2. * tolerance, 2. * error_bound)
    receipt = {
        "schema": "aura.stored_affine_margin_repair.v1",
        "status": "stored_feasible" if feasible else "stored_unresolved",
        "stored_primal_feasible": feasible, "stored_minimum_slack": minimum_slack,
        "storage_dtype": "float32", "rounding_rounds": _iteration,
        "maximum_margin_reserve": float(reserves.max()),
        "continuous_projection": proposal.receipt,
        "stored_displacement_norm": float(np.linalg.norm(stored)),
        "global_minimum_change_proven": False, "infeasibility_proven": False,
        "scope": "supplied_affine_comparisons", "serving_authority": False,
    }
    return MarginRepair(stored, proposal.multipliers, receipt)


def margin_neighborhood_bound(parameters, feature_difference, *, radius, offset=0.):
    """Bound one affine comparison throughout an L2 feature neighborhood.

    The offset is fixed and the radius bounds the *difference* feature. A
    caller must separately establish that this neighborhood covers its task.
    """
    w, feature = (np.asarray(value, dtype=np.float64)
                  for value in (parameters, feature_difference))
    if (w.ndim != 1 or not w.size or feature.shape != w.shape
            or not np.all(np.isfinite(w)) or not np.all(np.isfinite(feature))
            or not np.isfinite(radius) or radius < 0 or not np.isfinite(offset)):
        raise ValueError("invalid margin neighborhood")
    margin = float(w @ feature + offset)
    lower = margin - float(np.linalg.norm(w)) * radius
    return {"center_margin": margin, "feature_radius": radius, "lower_margin": lower,
            "strictly_positive_within_declared_radius": lower > 0.,
            "coverage_established": False, "scope": "one_affine_comparison",
            "serving_authority": False}

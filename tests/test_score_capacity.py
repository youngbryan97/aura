"""Capacity claims require a replayable witness, not an optimizer status."""

import copy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.score_capacity import (
    assess_score_capacity,
    score_constraints,
    verify_score_capacity,
)


def test_separable_scores_return_exactly_checkable_weights():
    result = assess_score_capacity([[1, 0], [-1, 2]], [0, 0])
    assert result["status"] == "feasible"
    assert verify_score_capacity(result)
    assert result["serving_authority"] is False


def test_collapsed_representation_has_a_kernel_checked_contradiction():
    result = assess_score_capacity([[1], [-1], [2]], [0, 0, 0],
                                   comparison_ids=["forward", "reverse", "unrelated"])
    assert result["status"] == "infeasible"
    assert verify_score_capacity(result)
    assert "reverse" in result["conflicting_comparison_ids"]
    assert len(result["conflicting_comparison_ids"]) <= 2


def test_expanding_observable_features_can_resolve_the_same_constraints():
    collapsed = assess_score_capacity([[1], [-1]], [0, 0])
    expanded = assess_score_capacity([[1, 0], [-1, 2]], [0, 0])
    assert collapsed["status"] == "infeasible"
    assert expanded["status"] == "feasible"
    assert verify_score_capacity(collapsed) and verify_score_capacity(expanded)


def test_fixed_terms_and_coefficient_bounds_are_part_of_the_proof():
    assert assess_score_capacity([[0]], [2])["status"] == "feasible"
    assert assess_score_capacity([[0]], [0])["status"] == "infeasible"
    assert assess_score_capacity([[-1]], [0], lower_bounds=[-2])["status"] == "feasible"
    assert assess_score_capacity([[-1]], [0], lower_bounds=[0])["status"] == "infeasible"


def test_unit_margin_failure_does_not_prove_any_positive_margin_impossible():
    assert assess_score_capacity([[1], [-1]], [0, 1], margin=1)["status"] == "infeasible"
    smaller = assess_score_capacity([[1], [-1]], [0, 1], margin=.25)
    assert smaller["status"] == "feasible" and verify_score_capacity(smaller)
    strict = assess_score_capacity([[1], [-1]], [0, 1], margin=0., strict=True)
    assert strict["status"] == "feasible" and verify_score_capacity(strict)


def test_strict_ranking_impossibility_has_an_exact_strict_contradiction():
    report = assess_score_capacity([[1], [-1]], [0, 0], margin=0., strict=True)
    assert report["status"] == "infeasible" and verify_score_capacity(report)
    assert any(c["strict"] for c in report["problem"]["constraints"])


def test_strict_ranking_cannot_call_a_tie_feasible():
    report = assess_score_capacity([[0]], [0], margin=0., strict=True)
    assert report["status"] == "infeasible" and verify_score_capacity(report)


def test_binary_float_coefficients_are_not_silently_decimal_rounded():
    constraints = score_constraints([[0.1]], [0.2])
    assert constraints[0].coeffs[0][1] == -Fraction(0.1)
    assert constraints[0].coeffs[0][1] != -Fraction("0.1")


def test_replay_rejects_changed_problem_and_forged_proof():
    report = assess_score_capacity([[1], [-1]], [0, 0])
    changed = copy.deepcopy(report)
    changed["problem"]["constraints"][0]["rhs"] = "999"
    assert not verify_score_capacity(changed)
    changed = copy.deepcopy(report)
    changed["certificate"]["multipliers"] = []
    assert not verify_score_capacity(changed)
    changed = copy.deepcopy(report)
    changed["certificate"]["multipliers"][0][1] = "-1"
    assert not verify_score_capacity(changed)
    report = assess_score_capacity([[1]], [0])
    report["weights"] = {"w0": "0"}
    assert not verify_score_capacity(report)


def test_numerical_false_feasibility_cannot_pass_the_exact_check(monkeypatch):
    monkeypatch.setattr("scipy.optimize.linprog", lambda *a, **k:
                        SimpleNamespace(success=True, status=0, x=np.array([0.999999999])))
    result = assess_score_capacity([[1]], [0])
    assert result["status"] == "unknown"
    assert not verify_score_capacity(result)


def test_numerical_infeasibility_cannot_replace_a_kernel_witness(monkeypatch):
    monkeypatch.setattr("core.learning.score_capacity.find_farkas", lambda _: None)
    result = assess_score_capacity([[1], [-1]], [0, 0])
    assert result["status"] == "unknown"
    assert not result["verified"]


def test_witness_search_limit_cannot_be_reported_as_impossibility():
    result = assess_score_capacity([[1], [-1]], [0, 0], max_witness_constraints=1)
    assert result["status"] == "unknown"
    assert not verify_score_capacity(result)


@pytest.mark.parametrize("matrix,offset", [([], []), ([[1]], []), ([[float("inf")]], [0]),
                                          ([[1]], [float("nan")])])
def test_invalid_geometry_is_rejected(matrix, offset):
    with pytest.raises(ValueError):
        assess_score_capacity(matrix, offset)


def test_comparison_identity_cannot_be_lost_or_duplicated():
    with pytest.raises(ValueError, match="identities"):
        assess_score_capacity([[1], [2]], [0, 0], comparison_ids=["same", "same"])

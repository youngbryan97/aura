"""Analytic solutions and adversarial cases for minimum-change learning."""

import numpy as np
import pytest

from core.learning.margin_repair import (
    margin_neighborhood_bound,
    minimum_margin_repair,
    verify_exact_margin_repair,
    verify_margin_repair,
)


def test_single_correction_matches_the_closed_form():
    normal = np.array([3., 4.])
    result = minimum_margin_repair(normal[None, :], [2.])
    np.testing.assert_allclose(result.displacement, 2. * normal / (normal @ normal), atol=1e-9)
    assert result.receipt["status"] == "verified_numerically"
    assert result.receipt["primal_objective"] == pytest.approx(2. ** 2 / (2. * 25.))
    assert not result.receipt["exact_arithmetic_proof"]


def test_the_documented_minimum_change_solution_has_an_exact_rational_proof():
    receipt = verify_exact_margin_repair([[0, 1], [1, -3]], ["1/2", "-9/10"],
                                        ["3/5", "1/2"], ["23/10", "3/5"])
    assert receipt["verified"]
    assert receipt["duality_gap"] == "0"
    assert receipt["primal_objective"] == receipt["dual_objective"] == "61/200"
    assert not receipt["serving_authority"]


def test_exact_verifier_does_not_round_an_infeasible_witness_into_a_proof():
    tiny_error = "999999999999999999999/1000000000000000000000"
    assert not verify_exact_margin_repair([[1]], [1], [tiny_error], [1])["verified"]
    assert not verify_exact_margin_repair([[1]], [1], [1], [-1])["verified"]


def test_shared_correction_satisfies_all_obligations_with_the_least_change():
    a = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 2.]])
    b = np.array([1., 0., 3., 6.])
    result = minimum_margin_repair(a, b)
    np.testing.assert_allclose(result.displacement, [1.5, 1.5], atol=1e-7)
    assert result.receipt["status"] == "verified_numerically"
    assert result.receipt["minimum_slack"] >= -1e-8
    assert result.receipt["duality_gap"] == pytest.approx(0., abs=1e-7)


def test_repair_is_equivariant_to_orthogonal_feature_coordinates():
    a = np.array([[1., 0.], [0., 1.], [1., 1.]])
    q, _ = np.linalg.qr(np.array([[1., 2.], [-2., 1.]]))
    expected = minimum_margin_repair(a, [1., 0., 3.])
    transformed = minimum_margin_repair(a @ q, [1., 0., 3.])
    np.testing.assert_allclose(q @ transformed.displacement, expected.displacement, atol=1e-7)


def test_unsatisfied_and_forged_dual_witnesses_cannot_pass():
    assert verify_margin_repair([[1.]], [1.], [0.], [0.])["status"] == "unverified"
    assert verify_margin_repair([[1.]], [1.], [1.], [-1.])["status"] == "unverified"
    assert verify_margin_repair([[1.]], [1.], [2.], [1.])["status"] == "unverified"


def test_contradictory_corrections_are_unresolved_not_a_false_infeasibility_proof():
    result = minimum_margin_repair([[1.], [-1.]], [1., 1.], max_iterations=10)
    assert result.receipt["status"] == "unverified"
    assert not result.receipt["infeasibility_proven"]


def test_zero_feature_cannot_repair_a_positive_deficit():
    result = minimum_margin_repair([[0., 0.]], [1.], max_iterations=5)
    assert not result.receipt["primal_feasible"]


def test_neighborhood_bound_is_tight_and_does_not_claim_coverage():
    w, feature = np.array([1., .5]), np.array([1., 0.])
    radius = .1
    result = margin_neighborhood_bound(w, feature, radius=radius)
    adverse = feature - radius * w / np.linalg.norm(w)
    assert result["lower_margin"] == pytest.approx(w @ adverse)
    assert result["lower_margin"] == pytest.approx(1. - .1 * np.sqrt(1.25))
    assert result["strictly_positive_within_declared_radius"]
    assert not result["coverage_established"]
    assert not margin_neighborhood_bound(w, feature, radius=1.)["strictly_positive_within_declared_radius"]


def test_one_satisfied_correction_does_not_prove_unseen_generalization():
    initial = np.array([1., 0.])
    result = minimum_margin_repair([[0., 1.]], [.5])
    unseen = np.array([1., -3.])
    assert initial @ unseen > 0
    assert (initial + result.displacement) @ unseen < 0
    protected = minimum_margin_repair([[0., 1.], [1., -3.]], [.5, -.9])
    assert protected.receipt["status"] == "verified_numerically"
    assert (initial + protected.displacement) @ unseen >= .1 - 1e-8


@pytest.mark.parametrize("a,b", [([[1., float('nan')]], [1.]), ([[1.]], []), ([], [])])
def test_bad_geometry_is_not_a_measured_result(a, b):
    with pytest.raises(ValueError):
        minimum_margin_repair(a, b)

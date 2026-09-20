"""Analytic solutions and adversarial cases for minimum-change learning."""

import numpy as np
import pytest

from core.learning.margin_repair import (
    margin_neighborhood_bound,
    minimum_margin_repair,
    minimum_stored_margin_repair,
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


def test_stored_repair_respects_many_binding_faces_after_rounding():
    rng = np.random.default_rng(912)
    anchor = rng.normal(size=128).astype(np.float32).astype(float)
    a = rng.normal(size=(49, 128))
    b = np.r_[np.zeros(48), 25.1 - a[-1] @ anchor]
    proposal = minimum_stored_margin_repair(a, b, anchor)
    assert proposal.receipt["stored_primal_feasible"]
    assert np.all(a @ proposal.displacement >= b)
    point = anchor + proposal.displacement
    np.testing.assert_array_equal(point, point.astype(np.float32).astype(float))
    assert not proposal.receipt["global_minimum_change_proven"]


def test_active_equalities_are_polished_before_accepting_a_numerical_certificate():
    rng = np.random.default_rng(912)
    anchor = rng.normal(size=128).astype(np.float32).astype(float)
    a = rng.normal(size=(49, 128))
    b = np.r_[np.zeros(48), 25.1 - a[-1] @ anchor]
    result = minimum_margin_repair(a, b, tolerance=1e-10)
    assert result.receipt["status"] == "verified_numerically"
    assert result.receipt["active_equalities_polished"]
    assert np.min(a @ result.displacement - b) >= -1e-10


@pytest.mark.parametrize("target,initial,expected", [
    ([1., -1.], [.001, .00000001], [1., 0.]),
    ([1., 1.], [1., 0.], [2. / 3., 2. / 3.]),
])
def test_active_support_can_remove_spurious_faces_and_add_missing_ones(target, initial, expected):
    from core.learning.margin_repair import _polish_nonnegative_dual

    gram = np.array([[1., .5], [.5, 1.]])
    lam, pivots = _polish_nonnegative_dual(gram, np.array(target), np.array(initial), max_iterations=20)
    np.testing.assert_allclose(lam, expected, atol=1e-12)
    assert pivots > 1
    assert np.min(gram @ lam - target) >= -1e-12


def test_exactly_representable_equalities_need_no_artificial_interior():
    result = minimum_stored_margin_repair([[1., 0.], [-1., 0.], [0., 1.]],
                                         [0., 0., .5], [1., 0.])
    assert result.receipt["stored_primal_feasible"]
    np.testing.assert_array_equal(result.displacement, [0., .5])
    assert result.receipt["maximum_margin_reserve"] == 0.


def test_storage_infeasibility_is_unresolved_not_a_false_continuous_proof():
    lower, upper = 1. + 2. ** -26, 1. + 2. ** -25
    result = minimum_stored_margin_repair([[1.], [-1.]], [lower, -upper], [0.],
                                         max_rounding_rounds=2)
    assert not result.receipt["stored_primal_feasible"]
    assert not result.receipt["infeasibility_proven"]


@pytest.mark.parametrize("anchor,rounds", [([float("nan")], 8), ([0., 0.], 8), ([0.], 0), ([0.], True)])
def test_invalid_storage_geometry_cannot_return_a_witness(anchor, rounds):
    with pytest.raises(ValueError):
        minimum_stored_margin_repair([[1.]], [1.], anchor, max_rounding_rounds=rounds)


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


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6, 1e-7])
def test_ill_conditioned_gram_falls_back_to_unsquared_feature_geometry(epsilon):
    a, b = [[1., 0.], [-1., epsilon]], [.1, -.1 + epsilon]
    result = minimum_margin_repair(a, b)
    np.testing.assert_allclose(result.displacement, [.1, 1.], atol=1e-8)
    assert result.receipt["status"] == "verified_numerically"
    assert result.receipt["equality_solver"] == "orthogonal_feature_svd_v1"
    assert verify_margin_repair(a, b, result.displacement, result.multipliers)["status"] == "verified_numerically"

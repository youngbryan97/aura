"""Nearly parallel constraints must not lose their analytic repair to a Gram solve."""

import numpy as np
import pytest

from core.learning.affine_margin_polish import polish_feature_dual
from core.learning.margin_repair import verify_margin_repair


@pytest.mark.parametrize("epsilon", [1e-3, 1e-5, 1e-6, 1e-7])
@pytest.mark.parametrize("extra_dimensions", [0, 24])
def test_nearly_opposed_faces_recover_the_known_primal_and_dual(epsilon, extra_dimensions):
    a = np.pad([[1., 0.], [-1., epsilon]], ((0, 0), (0, extra_dimensions)))
    b = [.1, -.1 + epsilon]
    point, dual, _ = polish_feature_dual(a, b, [1., 1.], max_iterations=20)
    np.testing.assert_allclose(point[:2], [.1, 1.], atol=1e-8)
    np.testing.assert_allclose(point[2:], 0., atol=1e-12)
    assert verify_margin_repair(a, b, point, dual)["status"] == "verified_numerically"


def test_active_set_can_both_remove_and_add_faces():
    a = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 2.]])
    b = np.array([1., 0., 3., 6.])
    point, dual, iterations = polish_feature_dual(a, b, [1., 1., 0., 0.], max_iterations=30)
    np.testing.assert_allclose(point, [1.5, 1.5], atol=1e-12)
    assert iterations > 1
    assert verify_margin_repair(a, b, point, dual)["status"] == "verified_numerically"


def test_inconsistent_rank_deficient_rows_remain_unverified():
    a, b = [[1.], [-1.]], [1., 1.]
    point, dual, _ = polish_feature_dual(a, b, [1., 1.], max_iterations=10)
    assert verify_margin_repair(a, b, point, dual)["status"] == "unverified"


def test_zero_start_discovers_the_binding_face():
    point, dual, _ = polish_feature_dual([[3., 4.]], [2.], [0.], max_iterations=5)
    np.testing.assert_allclose(point, [.24, .32], atol=1e-12)
    assert verify_margin_repair([[3., 4.]], [2.], point, dual)["status"] == "verified_numerically"


def test_redundant_warm_equalities_are_not_inherited_after_target_reserves():
    a, b = [[1., 0.], [1., 0.]], [1., 1.00001]
    warm, multipliers, _ = polish_feature_dual(a, b, [1., 1.], max_iterations=20)
    assert verify_margin_repair(a, b, warm, multipliers)["status"] == "verified_numerically"
    cold, multipliers, _ = polish_feature_dual(a, b, [0., 0.], max_iterations=20)
    np.testing.assert_allclose(cold, [1.00001, 0.], atol=1e-12)
    assert verify_margin_repair(a, b, cold, multipliers)["status"] == "verified_numerically"


@pytest.mark.parametrize("initial_value", [0., 1.])
def test_dependent_feasible_systems_keep_a_verified_primal_dual_pair(initial_value):
    rng = np.random.default_rng(919)
    for _ in range(100):
        a = rng.integers(-3, 4, (8, 3)).astype(float)
        witness = rng.normal(size=3)
        b = a @ witness - rng.random(8) * .1
        point, dual, _ = polish_feature_dual(a, b, np.full(8, initial_value), max_iterations=200)
        assert verify_margin_repair(a, b, point, dual)["status"] == "verified_numerically"


@pytest.mark.parametrize("a,b,initial", [([], [], []), ([[1.]], [], [0.]),
    ([[float("nan")]], [1.], [0.])])
def test_invalid_geometry_is_rejected(a, b, initial):
    with pytest.raises(ValueError):
        polish_feature_dual(a, b, initial, max_iterations=5)

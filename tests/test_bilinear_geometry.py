"""Factor coordinates must not decide which equivalent relation map is learned."""

import numpy as np
import pytest

from core.learning.bilinear_geometry import BilinearFactorGeometry
from core.learning.semantic_graph_constraints import _fit_graph_parameters
from core.learning.semantic_relation_graph_learning import (
    RelationEvidenceBank,
    RelationGraphContrast,
    graph_margin,
    graph_margin_gradient,
)


def _norm2(parts):
    return sum(float(np.sum(part * part)) for part in parts)


def _problem():
    q = np.array([[1.], [.25]])
    d = np.array([[1.], [.25]])
    bank = RelationEvidenceBank(np.array([0., 1.]), np.array([[1., 0.], [0., 0.]]), np.zeros(2))
    row = RelationGraphContrast(((bank, 0),), ((bank, 1),), -.65)
    return (q, d), (row,)


def test_metric_is_the_literal_frobenius_contribution_norm():
    rng = np.random.default_rng(919)
    q, d, dq, dd = (rng.normal(size=(7, 3)) for _ in range(4))
    geometry = BilinearFactorGeometry.from_factors(q, d, scale=2.)
    assert _norm2(geometry.encode((dq, dd))) == pytest.approx(
        4 * (_norm2((dq @ d.T,)) + _norm2((q @ dd.T,))), rel=1e-13)
    for restored, original in zip(geometry.decode(geometry.encode((q, d))), (q, d), strict=True):
        np.testing.assert_allclose(restored, original, atol=1e-13)


def test_metric_survives_general_invertible_factor_coordinates():
    rng = np.random.default_rng(20)
    q, d, dq, dd = (rng.normal(size=(9, 3)) for _ in range(4))
    transform = np.array([[2., 1., 0.], [0., .5, 1.], [0., 0., 4.]])
    inverse = np.linalg.inv(transform).T
    original = BilinearFactorGeometry.from_factors(q, d)
    changed = BilinearFactorGeometry.from_factors(q @ transform, d @ inverse)
    np.testing.assert_allclose(q @ d.T, (q @ transform) @ (d @ inverse).T, atol=1e-13)
    assert _norm2(original.encode((dq, dd))) == pytest.approx(
        _norm2(changed.encode((dq @ transform, dd @ inverse))), rel=1e-13)
    # Ordinary coefficient distance is not a distance on the relation function.
    assert not np.isclose(_norm2((dq, dd)), _norm2((dq @ transform, dd @ inverse)))


def test_chart_gradient_matches_finite_differences():
    parameters, rows = _problem()
    geometry = BilinearFactorGeometry.from_factors(*parameters)
    coordinates = geometry.encode(parameters)
    _, gradients = graph_margin_gradient(parameters, rows[0])
    chart_gradients = geometry.pullback(gradients)
    for block, value in enumerate(coordinates):
        for index in np.ndindex(value.shape):
            plus, minus = ([part.copy() for part in coordinates] for _ in range(2))
            plus[block][index] += 1e-6
            minus[block][index] -= 1e-6
            difference = (graph_margin(geometry.decode(plus), rows[0]) -
                          graph_margin(geometry.decode(minus), rows[0])) / 2e-6
            assert difference == pytest.approx(chart_gradients[block][index], abs=1e-8)


@pytest.mark.parametrize("batched", [False, True])
def test_fit_has_same_function_under_reciprocal_scaling(batched):
    parameters, rows = _problem()
    changed = (parameters[0] * 8, parameters[1] / 8)
    options = dict(steps=4, batched=batched, update_rule="minimum_change", relation_metric="factor_function")
    first, receipt = _fit_graph_parameters(parameters, rows, **options)
    second, rescaled_receipt = _fit_graph_parameters(changed, rows, **options)
    np.testing.assert_allclose(first[0] @ first[1].T, second[0] @ second[1].T, atol=1e-6)
    assert receipt["stored_wrong_or_tied"] == rescaled_receipt["stored_wrong_or_tied"] == 0
    assert receipt["relation_geometry"]["physical_storage_dtype"] == "float32"
    for row, observed in zip(rows, receipt["stored_margins"], strict=True):
        assert graph_margin(first, row) == observed
    for value in first:
        assert np.array_equal(value, value.astype(np.float32).astype(float))
    ordinary, _ = _fit_graph_parameters(parameters, rows, steps=4, update_rule="minimum_change")
    scaled, _ = _fit_graph_parameters(changed, rows, steps=4, update_rule="minimum_change")
    assert not np.allclose(ordinary[0] @ ordinary[1].T, scaled[0] @ scaled[1].T, atol=1e-4)


def test_checkpoint_binds_metric_and_replays_physical_storage(tmp_path):
    parameters, rows = _problem()
    options = dict(steps=4, update_rule="minimum_change", relation_metric="factor_function",
                   checkpoint_path=tmp_path / "fit.npz")
    first, receipt = _fit_graph_parameters(parameters, rows, **options)
    resumed, replay = _fit_graph_parameters(parameters, rows, **options)
    assert receipt == replay
    for a, b in zip(first, resumed, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="identity"):
        _fit_graph_parameters(parameters, rows, **{**options, "relation_metric": "coefficient_euclidean"})


@pytest.mark.parametrize("bad", [np.zeros((2, 1)), np.ones((2, 2))])
def test_degenerate_geometry_is_not_silently_regularized(bad):
    with pytest.raises(ValueError, match="full-column-rank"):
        BilinearFactorGeometry.from_factors(bad, bad)


def test_frozen_unrelated_blocks_remain_exact():
    parameters, rows = _problem()
    frozen = np.array([np.pi])
    fitted, receipt = _fit_graph_parameters((*parameters, frozen), rows, steps=4,
        update_rule="minimum_change", relation_metric="factor_function", trainable_parameters=(True, True, False))
    np.testing.assert_array_equal(fitted[2], frozen)
    assert receipt["frozen_parameters_unchanged"]


def test_joint_trainer_passes_the_metric_into_real_fitting():
    from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
    from tests.test_semantic_relation_graph_learning import model_examples

    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    fitted = refit_compositional_joint_graphs(model, examples, rounds=1, steps=2,
        retention_operation_charts=2, constraint_learning=True, learn_operation_pointer=True,
        update_rule="minimum_change", relation_metric="factor_function")
    receipt = fitted.training_receipt["joint_graph_refit"]
    assert receipt["relation_metric"] == "factor_function"
    assert all(row["fit"]["relation_geometry"]["metric"] == "anchored_bilinear_factor_function_v1"
               for row in receipt["rounds"])


def test_transformed_storage_is_checked_in_original_inequalities():
    from core.learning.margin_repair import minimum_stored_margin_repair

    result = minimum_stored_margin_repair([[1.]], [.13], [0.],
        store_point=lambda value: np.round(value, 1))
    assert result.receipt["stored_primal_feasible"]
    assert result.displacement[0] >= .13
    assert result.receipt["storage_dtype"] == "caller_defined"
    assert result.receipt["rounding_rounds"] > 0


def test_storage_callback_cannot_claim_success_for_a_rejected_point():
    from core.learning.margin_repair import minimum_stored_margin_repair

    result = minimum_stored_margin_repair([[1.]], [.13], [0.], max_rounding_rounds=2,
        store_point=lambda _value: np.zeros(1))
    assert not result.receipt["stored_primal_feasible"]

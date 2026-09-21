"""The source-score metric is an exact invertible quadratic coordinate change."""

import numpy as np
import pytest

from core.learning.affine_function_geometry import AffineFunctionGeometry, CompositeParameterGeometry
from core.learning.bilinear_geometry import BilinearFactorGeometry


@pytest.mark.parametrize('shape', [(3, 8), (12, 3), (1, 1)])
def test_coordinates_equal_coefficient_plus_empirical_score_norm(shape):
    rng = np.random.default_rng(25)
    features = rng.normal(size=shape)
    weight, bias = rng.normal(size=(4, shape[1])), rng.normal(size=4)
    geometry = AffineFunctionGeometry.from_features(features, parameter_index=0)
    encoded = geometry.encode((weight, bias))
    expected = np.sum(weight ** 2) + np.sum(bias ** 2) + np.sum((features @ weight.T + bias) ** 2) / len(features)
    assert sum(np.sum(value ** 2) for value in encoded) == pytest.approx(expected, rel=1e-12)
    for value, restored in zip((weight, bias), geometry.decode(encoded), strict=True):
        np.testing.assert_allclose(restored, value, atol=1e-12)


def test_duplicate_and_rank_deficient_features_preserve_the_metric():
    features = np.array([[1., 2., 0.], [1., 2., 0.]])
    parameters = (np.array([[2., 3., 4.]]), np.array([.5]))
    one = AffineFunctionGeometry.from_features(features[:1], parameter_index=0)
    two = AffineFunctionGeometry.from_features(features, parameter_index=0)
    for a, b in zip(one.encode(parameters), two.encode(parameters), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)
    assert two.encode(parameters)[0][0, 2] == pytest.approx(4.)


def test_pullback_obeys_the_chain_rule_and_leaves_other_blocks_unchanged():
    rng = np.random.default_rng(71)
    geometry = AffineFunctionGeometry.from_features(rng.normal(size=(4, 6)), parameter_index=2)
    parameters = (np.zeros((1, 1)), np.zeros(1), rng.normal(size=(3, 6)), rng.normal(size=3))
    direction = tuple(rng.normal(size=value.shape) for value in parameters)
    gradient = tuple(rng.normal(size=value.shape) for value in parameters)
    left = sum(np.sum(g * d) for g, d in zip(geometry.pullback(gradient), direction, strict=True))
    eps = 1e-6
    plus = geometry.decode(tuple(a + eps * b for a, b in zip(parameters, direction, strict=True)))
    minus = geometry.decode(tuple(a - eps * b for a, b in zip(parameters, direction, strict=True)))
    right = sum(np.sum(g * (a - b) / (2 * eps)) for g, a, b in zip(gradient, plus, minus, strict=True))
    assert left == pytest.approx(right, rel=1e-8)
    assert geometry.encode(parameters)[0] is parameters[0]


def test_composes_with_existing_bilinear_geometry():
    rng = np.random.default_rng(33)
    parameters = (rng.normal(size=(8, 2)), rng.normal(size=(8, 2)),
                  rng.normal(size=(3, 6)), rng.normal(size=3))
    geometry = CompositeParameterGeometry((BilinearFactorGeometry.from_factors(*parameters[:2]),
        AffineFunctionGeometry.from_features(rng.normal(size=(4, 6)), parameter_index=2)))
    for before, after in zip(parameters, geometry.decode(geometry.encode(parameters)), strict=True):
        np.testing.assert_allclose(before, after, atol=1e-12)


def test_checked_projection_uses_a_lower_impact_direction():
    from core.learning.margin_repair import minimum_margin_repair

    geometry = AffineFunctionGeometry.from_features(np.array([[100., 0.], [-100., 0.]]), parameter_index=0)
    physical_normal = (np.array([[1., 1.]]), np.array([0.]))
    encoded_normal = geometry.pullback(physical_normal)
    normal = np.concatenate([part.ravel() for part in encoded_normal])[None, :]
    result = minimum_margin_repair(normal, np.array([1.]))
    assert result.receipt['primal_feasible']
    point = result.displacement
    weight, bias = geometry.decode((point[:2].reshape(1, 2), point[2:]))
    # min 10001*x^2 + y^2 with x+y >= 1 has this exact solution.
    np.testing.assert_allclose(weight[0], [1. / 10002., 10001. / 10002.], atol=1e-9)
    assert bias[0] == pytest.approx(0.)
    assert 100. * abs(weight[0, 0]) < .01
    assert weight.sum() >= 1. - 1e-9


@pytest.mark.parametrize('features', [np.zeros((0, 3)), np.ones(3), np.full((2, 3), np.nan)])
def test_invalid_source_features_are_rejected(features):
    with pytest.raises(ValueError, match='geometry'):
        AffineFunctionGeometry.from_features(features, parameter_index=0)


def test_fit_uses_geometry_and_roundtrips_checkpoint(tmp_path):
    from core.learning.semantic_graph_constraints import _fit_graph_parameters
    from core.learning.semantic_fit_problem import load_fit_problem
    from tests.test_semantic_graph_batch import problem
    import json

    initial, contrasts = problem()
    path = tmp_path / 'fit.npz'
    fitted, receipt = _fit_graph_parameters(initial, contrasts, steps=1, update_rule='minimum_change',
        operation_metric='source_function', checkpoint_path=path)
    assert receipt['operation_geometry']['views'][0]['source_feature_count'] > 0
    assert receipt['all_constraints_checked_at_acceptance']
    with np.load(path, allow_pickle=False) as archive:
        identity = json.loads(archive['metadata'].tobytes())['identity']
    captured = load_fit_problem(path.with_suffix('.problem.npz'), expected_identity=identity)
    assert captured['options']['operation_metric'] == 'source_function'
    replay, repeated = _fit_graph_parameters(captured['initial'], captured['contrasts'], **captured['options'])
    for a, b in zip(fitted, replay, strict=True):
        np.testing.assert_array_equal(a, b)
    assert repeated['stored_margins'] == receipt['stored_margins']

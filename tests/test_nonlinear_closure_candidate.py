"""Known worlds distinguish predictive proxies from missed interactions."""
import numpy as np
import pytest

from core.subject.closure_candidate import compare, feature_blocks, gaussian_world
from core.subject.estimate import fit_predict, split_rows


def score(own, outside, target):
    train, validate, test = split_rows(len(own))
    return compare(own, outside, target, train=train, validate=validate, test=test, seed=101)


def test_a_nonlinear_copy_of_the_core_adds_no_information_to_the_nonlinear_baseline():
    rng = np.random.default_rng(101)
    own = rng.uniform(-1, 1, (1200, 1))
    outside = own**2
    target = own**2 + rng.normal(0, .1, own.shape)
    train, validate, test = split_rows(len(own))
    linear = fit_predict(own, target, train=train, validate=validate, test=test)
    result = score(own, outside, target)
    assert result.narrow.sse < linear.sse * .2
    assert abs(result.gain) < .02


def test_the_outside_can_act_only_through_an_interaction():
    rng = np.random.default_rng(202)
    own = rng.normal(size=(1600, 1))
    outside = rng.choice([-1., 1.], size=(1600, 2))
    target = own + (outside[:, 0] * outside[:, 1])[:, None] + rng.normal(0, .1, own.shape)
    result = score(own, outside, target)
    assert result.gain > .9


def test_an_outside_cause_can_interact_with_the_core():
    rng = np.random.default_rng(303)
    own = rng.choice([-1., 1.], size=(1600, 1))
    outside = rng.choice([-1., 1.], size=(1600, 1))
    target = own * outside + rng.normal(0, .1, own.shape)
    assert score(own, outside, target).gain > .9


def test_a_known_gaussian_effect_is_read_on_unseen_rows():
    own, outside, target = gaussian_world(seed=404, rows=4000, effect=.15, outside_width=2)
    assert score(own, outside, target).gain == pytest.approx(.15, abs=.05)


def test_no_outside_effect_is_not_invented():
    own, outside, target = gaussian_world(seed=505, rows=4000, effect=0., outside_width=2)
    assert abs(score(own, outside, target).gain) < .02


def test_changing_held_out_rows_does_not_change_the_training_basis():
    rng = np.random.default_rng(606)
    own, outside = rng.normal(size=(200, 16)), rng.normal(size=(200, 14))
    a, b = feature_blocks(own, outside, train=slice(0, 100))
    own[100:] *= 100
    outside[100:] += 100
    c, d = feature_blocks(own, outside, train=slice(0, 100))
    np.testing.assert_array_equal(a[:100], c[:100])
    np.testing.assert_array_equal(b[:100], d[:100])


def test_no_added_basis_is_exactly_the_narrow_predictor_for_multiple_targets():
    rng = np.random.default_rng(707)
    own = rng.normal(size=(800, 2))
    target = np.hstack([own[:, :1]**2, own[:, 1:]**2]) + rng.normal(0, .1, (800, 2))
    result = score(own, np.zeros((800, 1)), target)
    assert result.gain == pytest.approx(0., abs=1e-12)
    np.testing.assert_allclose(result.narrow.row_sse, result.wide.row_sse, atol=1e-12)


def test_declared_feature_bounds_stop_extrapolation_on_a_saturating_transition():
    rng = np.random.default_rng(808)
    own = rng.normal(size=(1000, 1))
    own[800:] *= 1000
    target = np.clip(own, -2, 2)
    train, validate, test = split_rows(len(own))
    unbounded = fit_predict(own, target, train=train, validate=validate, test=test)
    bounded = fit_predict(own, target, train=train, validate=validate, test=test, feature_limit=2.)
    assert unbounded.loss > 1000.
    assert bounded.loss < .02


@pytest.mark.parametrize("limit", [0., -1., float("nan"), float("inf")])
def test_a_feature_bound_must_be_a_finite_positive_declaration(limit):
    own = np.arange(20.)[:, None]
    train, validate, test = split_rows(len(own))
    with pytest.raises(ValueError, match="finite and positive"):
        fit_predict(own, own, train=train, validate=validate, test=test, feature_limit=limit)


def test_bounding_an_empty_addition_keeps_the_same_predictor():
    rng = np.random.default_rng(909)
    own = rng.normal(size=(600, 2))
    target = own**2 + rng.normal(0, .1, own.shape)
    train, validate, test = split_rows(len(own))
    result = compare(own, np.zeros((600, 1)), target, train=train, validate=validate, test=test, feature_limit=8.)
    assert result.gain == pytest.approx(0., abs=1e-12)

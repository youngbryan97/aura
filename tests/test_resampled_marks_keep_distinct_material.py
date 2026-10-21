"""Subpixel raster phase must not erase distinct, continuously visible material."""
import numpy as np
import pytest

from core.perception.points_before_the_picture_is_reduced import PointsBeforeReduction

pytestmark = pytest.mark.unit


def analytic_pixels(shift=(0., 0.), *, frequency=1.3, distinctive=True, angle=0.):
    yy, xx = np.mgrid[:96, :96]
    local = (np.stack((xx, yy), axis=-1)-(48.+shift[0], 48.+shift[1]))
    local = local @ np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))+(48., 48.)
    x, y = local[..., 0], local[..., 1]
    channels = [.5+.3*np.sin(frequency*(x+.4*y)), .5+.3*np.cos(frequency*(.3*x-y)),
                .5+.2*np.sin(frequency*(.5*x+y))+.1*np.cos(frequency*(x-y))]
    if distinctive:
        channels[0] += .05*np.cos(.14*x-.27*y)
        channels[1] += .05*np.sin(.31*x+.11*y)
    return (np.stack(channels, axis=-1)*255).astype(np.uint8)


@pytest.mark.parametrize('frequency', [1.3, 1.5, 1.7])
def test_half_pixel_phase_change_keeps_measured_names_and_their_actual_displacement(frequency):
    tracker = PointsBeforeReduction(limit=64)
    before, after = analytic_pixels(frequency=frequency), analytic_pixels((.5, .5), frequency=frequency)
    original = tracker.see_source(before, before, 1.)
    current = tracker.see_source(after, after, 1.1)
    old = {p.number for p in original}
    kept = [p for p in current if p.number in old]
    assert len(kept) >= 8
    for point in kept:
        assert point.born == 1. and point.before == 1. and point.at == 1.1 and point.pictures == 2
        assert np.linalg.norm((point.x-point.old_x-.5, point.y-point.old_y-.5)) < .2
        assert point.error > 0. and np.isfinite(point.error)


@pytest.mark.parametrize('shift', [(.5, .5), (1.4, -.4), (5.4, 3.6)])
def test_locally_repeated_texture_cannot_supply_new_subpixel_point_identity(shift):
    tracker = PointsBeforeReduction(limit=96)
    before, after = analytic_pixels(distinctive=False), analytic_pixels(shift, distinctive=False)
    original = tracker.see_source(before, before, 1.)
    current = tracker.see_source(after, after, 1.1)
    assert not {p.number for p in original} & {p.number for p in current}


def test_common_grid_fits_do_not_depend_on_other_patches_or_mutate_inputs():
    from core.perception.patches_compared_on_a_common_grid import fit_common_grid
    from core.perception.points_before_the_picture_is_reduced import _colours

    before, after = _colours(analytic_pixels()), _colours(analytic_pixels((.5, .5)))
    points = np.array(((20., 20.), (40., 40.), (70., 70.)))
    initial = points+np.array(((.5, .5), (6., -4.), (.5, .5)))
    snapshots = [v.copy() for v in (before, after, points, initial)]
    batch = fit_common_grid(before, after, points, initial)
    for index in range(len(points)):
        single = fit_common_grid(before, after, points[index:index+1], initial[index:index+1])
        for many, one in zip(batch, single, strict=True):
            np.testing.assert_allclose(many[index:index+1], one, rtol=0., atol=1e-10)
    for actual, old in zip((before, after, points, initial), snapshots, strict=True):
        np.testing.assert_array_equal(actual, old)


@pytest.mark.parametrize('damage', ['flat', 'line', 'edge', 'absent', 'colour'])
def test_fallback_requires_current_distinct_observable_colour(damage):
    from core.perception.patches_compared_on_a_common_grid import fit_common_grid, distinct_patches
    from core.perception.points_before_the_picture_is_reduced import _colours

    before, after = _colours(analytic_pixels()), _colours(analytic_pixels((.5, .5)))
    points = np.array(((32., 32.), (48., 48.)))
    if damage == 'flat':
        before[:] = after[:] = .5
    elif damage == 'line':
        before[:] = after[:] = np.arange(96)[None, :, None]/96.
    elif damage == 'edge':
        points[:] = 1.
    elif damage == 'absent':
        after[:] = .5
    else:
        after = after[..., (1, 2, 0)]
    ends, error, _angle, fitted = fit_common_grid(before, after, points, points+.5)
    assert not (fitted & (error < .025)).any()
    assert np.isfinite(ends).all() and distinct_patches(before, points).shape == (2,)


@pytest.mark.parametrize('damage', ['empty', 'too_many', 'nan_pixels', 'huge_pixels', 'negative_pixels',
                                   'shape', 'integer_points', 'nan_points', 'huge_points', 'initial_shape'])
def test_invalid_numerical_requests_supply_no_patch_correspondence(damage):
    from core.perception.patches_compared_on_a_common_grid import fit_common_grid
    from core.perception.points_before_the_picture_is_reduced import _colours

    before, after = _colours(analytic_pixels()), _colours(analytic_pixels((.5, .5)))
    points = np.array(((32., 32.),))
    initial = points+.5
    if damage == 'empty':
        points = initial = np.empty((0, 2))
    elif damage == 'too_many':
        points = initial = np.zeros((97, 2))
    elif damage == 'nan_pixels':
        after[0, 0, 0] = np.nan
    elif damage == 'huge_pixels':
        before[:] = 1e30
    elif damage == 'negative_pixels':
        before[0, 0, 0] = -.1
    elif damage == 'shape':
        after = after[:32]
    elif damage == 'integer_points':
        points = points.astype(int)
    elif damage == 'nan_points':
        points[:] = np.nan
    elif damage == 'huge_points':
        points[:] = 1e100
    else:
        initial = initial[:, :1]
    assert all(len(value) == 0 for value in fit_common_grid(before, after, points, initial))


def test_source_recovery_fits_at_most_32_missing_marks_and_preserves_successful_measurements(monkeypatch):
    from core.perception import patches_compared_on_a_common_grid as module

    tracker = PointsBeforeReduction(limit=96)
    first, second = analytic_pixels(), analytic_pixels((.5, .5))
    tracker.see_source(first, first, 1.)
    calls, original = [], module.fit_common_grid

    def measured(before, after, points, initial):
        calls.append(len(points))
        return original(before, after, points, initial)

    monkeypatch.setattr(module, 'fit_common_grid', measured)
    tracker.see_source(second, second, 1.1)
    assert calls[:2] == [32, 32] and max(calls) <= 96
    measured_points = list(tracker._native_points)
    assert module.recover_source_marks(tracker, tracker._image, tracker._colours, 1.2, measured_points) is measured_points
    assert any(p.error > .025 for p in tracker.points if p.born == 1.)


def test_common_grid_repetition_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.patches_compared_on_a_common_grid import _common_grid_recovery_invariant

    assert _common_grid_recovery_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'common_grid_recovery_rejects_local_repetition')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


def test_new_point_measurements_reach_the_ordinary_worker_with_their_capture_and_birth():
    from core.perception.play_observation_without_holding_the_loop import _seen
    from core.perception.what_moves_in_the_picture import WhatMoves

    eyes = WhatMoves(width=192)
    first = np.full((192, 192, 3), 25, np.uint8)
    second = first.copy()
    first[48:144, 48:144] = analytic_pixels()
    second[48:144, 48:144] = analytic_pixels((.5, .5))
    _changes, before = _seen(eyes, first, 1.)
    _changes, after = _seen(eyes, second, 1.1)
    original = {p.number for p in before['material_marks'].points}
    retained = [p for p in after['material_marks'].points if p.number in original]
    assert len(retained) >= 8
    assert after['material_marks'].pixels == after['pixels_consumed']
    assert after['pixels_consumed'].at == 1.1
    assert all(p.born == 1. and p.before == 1. and p.at == 1.1 and p.pictures == 2 for p in retained)
    assert before['material_marks'].pixels.at == 1.


@pytest.mark.parametrize('change', ['colour', 'absent', 'resize', 'gap', 'repeat', 'reset'])
def test_subpixel_recovery_does_not_bridge_replacement_or_capture_departure(change):
    tracker = PointsBeforeReduction(limit=64)
    first = analytic_pixels()
    original = tracker.see_source(first, first, 1.)
    second = analytic_pixels((.5, .5))
    at = 1.1
    if change == 'colour':
        second = second[..., (1, 2, 0)]
    elif change == 'absent':
        second[:] = 128
    elif change == 'resize':
        second = second[:80]
    elif change == 'gap':
        at = 2.
    elif change == 'repeat':
        at = 1.
    else:
        tracker.clear()
    current = tracker.see_source(second, second, at)
    assert not {p.number for p in original} & {p.number for p in current}

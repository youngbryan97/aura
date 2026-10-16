"""Shared transforms preserve the masked pixel objective and separate copies."""
import math

import numpy as np
import pytest
from scipy import signal

from core.perception.a_visual_reference_in_the_picture import (
    _SceneCorrelations, _scene_correlation_invariant, reference_appearances,
)


@pytest.mark.parametrize('shape', [(16, 19), (31, 28), (64, 96)])
def test_shared_spectra_match_direct_masked_squared_colour_error(shape):
    rng = np.random.default_rng(73)
    scene = rng.uniform(0., 1., (*shape, 3)).astype(np.float32)
    shared = _SceneCorrelations(scene)
    for height, width in ((3, 5), (11, 9), shape):
        rgb = rng.uniform(0., 1., (height, width, 3)).astype(np.float32)
        mask = (rng.uniform(0., 1., (height, width)) > .4).astype(np.float32)
        actual = shared.cost(rgb, mask)
        windows = np.lib.stride_tricks.sliding_window_view(scene, (height, width), axis=(0, 1))
        windows = windows.transpose(0, 1, 3, 4, 2)
        direct = np.sum((windows-rgb)**2*mask[..., None], axis=(2, 3, 4))/(3*mask.sum())
        np.testing.assert_allclose(actual, direct, atol=2e-6, rtol=0.)
        # The previous convolution route is another numerical comparison;
        # the direct pixels above supply the independent objective.
        previous = sum(signal.fftconvolve(scene[..., c]**2, mask[::-1, ::-1], mode='valid')
                       - 2*signal.fftconvolve(scene[..., c], (rgb[..., c]*mask)[::-1, ::-1], mode='valid')
                       + np.sum(rgb[..., c]**2*mask) for c in range(3))/(3*mask.sum())
        np.testing.assert_allclose(actual, previous, atol=2e-6, rtol=0.)


def test_reused_scene_spectra_do_not_keep_a_previous_kernel():
    rng = np.random.default_rng(83)
    scene = rng.uniform(0., 1., (24, 32, 3)).astype(np.float32)
    shared = _SceneCorrelations(scene, (8, 11))
    rgb = scene[5:13, 7:18].copy()
    mask = np.ones(rgb.shape[:2], np.float32)
    first = shared.cost(rgb, mask).copy()
    shared.cost(np.zeros((6, 9, 3), np.float32), np.ones((6, 9), np.float32))
    np.testing.assert_array_equal(shared.cost(rgb, mask), first)
    assert abs(float(first[5, 7])) < 2e-6
    assert _scene_correlation_invariant() == ()


def test_a_smaller_transform_bank_refuses_a_kernel_that_would_alias():
    shared = _SceneCorrelations(np.zeros((24, 32, 3), np.float32), (6, 9))
    with pytest.raises(ValueError):
        shared.cost(np.zeros((7, 9, 3), np.float32), np.ones((7, 9), np.float32))


def test_shared_search_keeps_separate_current_instances_and_their_coordinates():
    rng = np.random.default_rng(97)
    reference = rng.integers(0, 255, (40, 48, 4), dtype=np.uint8)
    reference[..., 3] = 255
    picture = np.zeros((120, 180, 3), np.uint8)
    picture[10:50, 12:60] = reference[..., :3]
    picture[68:108, 110:158] = reference[..., :3]
    found = reference_appearances(picture, reference, scales=(.8, 1., 1.2))
    assert len(found) == 2
    positions = [pose.project((19., 17.)) for pose in found]
    assert min(math.dist(point, (31., 27.)) for point in positions) < .25
    assert min(math.dist(point, (129., 85.)) for point in positions) < .25


def test_scene_spectra_are_new_for_each_captured_picture():
    first = np.zeros((24, 32, 3), np.float32)
    second = np.ones_like(first)
    rgb, mask = np.zeros((7, 9, 3), np.float32), np.ones((7, 9), np.float32)
    assert np.all(abs(_SceneCorrelations(first).cost(rgb, mask)) < 2e-6)
    assert np.all(abs(_SceneCorrelations(second).cost(rgb, mask)-1.) < 2e-6)

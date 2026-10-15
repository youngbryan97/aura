"""Pixel response measurements preserve observed support before body discovery."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import ndimage

from core.perception.pixel_motion_before_object_identity import (
    ANGULAR_MOVEMENT_MIN,
    MAX_POINTS,
    _angular_motion_layer,
    _camera,
    _local_affine,
    _local_translation,
    pixel_motion_signature,
)


def _textured_picture(height=120, width=160):
    rng = np.random.default_rng(31)
    plane = ndimage.gaussian_filter(rng.uniform(20., 230., (height, width)), .5)
    return np.repeat(plane.astype(np.uint8)[..., None], 3, axis=2)


def test_scene_translation_is_removed_using_distributed_current_pixels():
    before = _textured_picture()
    after = ndimage.shift(before, (2, -3, 0), order=1, mode="reflect")
    result = pixel_motion_signature(before, after)
    assert result is not None
    assert result.reference == "distributed_translation"
    assert result.camera is not None
    assert result.camera.dx == pytest.approx(-3, abs=.3)
    assert result.camera.dy == pytest.approx(2, abs=.3)
    assert result.camera.fraction >= .65
    assert len(result.camera.cells) >= 6
    measured = [np.linalg.norm(feature[:2]) for feature, support in zip(result.features, result.support, strict=True) if support]
    assert np.quantile(measured, .9) < .6


def test_local_translation_is_measured_without_an_existing_object():
    before = _textured_picture()
    after = before.copy()
    patch = before[42:78, 58:94].copy()
    after[42:78, 58:94] = 80
    after[42:78, 62:98] = patch
    result = pixel_motion_signature(before, after, point_limit=MAX_POINTS)
    assert result is not None
    assert result.camera.dx == pytest.approx(0, abs=.2)
    assert result.camera.dy == pytest.approx(0, abs=.2)
    rows, columns = result.shape
    local = [result.features[row * columns + column][0]
             for row in range(rows) for column in range(columns)
             if 48 <= column * result.tile < 96 and 36 <= row * result.tile < 84
             and result.support[row * columns + column]]
    assert max(local) > 3
    assert not hasattr(result, "identity")


@pytest.mark.parametrize("tile", (12, 24))
def test_rotating_current_pixels_supply_a_local_angular_response(tile):
    before = _textured_picture()
    after = before.copy()
    after[36:84, 56:104] = ndimage.rotate(before[36:84, 56:104], -8., reshape=False,
                                        order=1, mode="reflect")
    result = pixel_motion_signature(before, after, tile=tile, point_limit=MAX_POINTS)
    assert result is not None
    rows, columns = result.shape
    angular = [result.features[row * columns + column][2]
               for row in range(rows) for column in range(columns)
               if 60 <= column * result.tile < 96 and 36 <= row * result.tile < 84
               and result.angular_support[row * columns + column] >= 4]
    assert angular and max(angular) > .6
    shift = np.asarray((result.camera.dx, result.camera.dy)) if result.camera is not None else np.zeros(2)
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            if abs(result.features[index][2]) <= .6:
                continue
            left, top = column * tile, row * tile
            assert any(left <= p.before[0] < left + tile and top <= p.before[1] < top + tile
                       and np.linalg.norm(np.asarray(p.after) - p.before - shift) > ANGULAR_MOVEMENT_MIN
                       for p in result.correspondences)


def test_working_geometry_and_correspondences_are_bounded_and_immutable():
    before = _textured_picture(480, 640)
    result = pixel_motion_signature(before, before, point_limit=96)
    assert result is not None
    assert result.source_shape == (480, 640)
    assert result.working_shape == (240, 320)
    assert 12 <= len(result.correspondences) <= 96
    assert len(result.features) == len(result.support) == np.prod(result.shape)
    assert isinstance(result.features, tuple) and isinstance(result.features[0], tuple)
    with pytest.raises(FrozenInstanceError):
        result.tile = 20


def test_missing_or_changed_material_pixels_leave_camera_unknown():
    before = _textured_picture()
    blank = np.full_like(before, 60)
    assert pixel_motion_signature(before, blank) is None
    assert pixel_motion_signature(blank, blank) is None
    assert pixel_motion_signature(before.astype(float), before) is None
    assert pixel_motion_signature(before, before[:80]) is None
    assert pixel_motion_signature(before, before, point_limit=MAX_POINTS + 1) is None


def test_concentrated_texture_preserves_raw_motion_with_an_unknown_reference():
    before = np.full((120, 160, 3), 60, np.uint8)
    before[42:78, 58:94] = _textured_picture(36, 36)
    after = np.full_like(before, 60)
    after[42:78, 62:98] = before[42:78, 58:94]
    result = pixel_motion_signature(before, after)
    assert result is not None
    assert result.reference == "screen_unreferenced"
    assert result.camera is None
    assert len(result.correspondences) >= 4
    measured = [feature[0] for feature, support in zip(result.features, result.support, strict=True) if support]
    assert measured and max(measured) > 3
    assert not hasattr(result, "identity")


def test_disagreeing_scene_layers_and_camera_rotation_are_unknown():
    yy, xx = np.mgrid[10:111:20, 10:151:20]
    starts = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    ends = starts.copy()
    ends[::2, 0] += 3
    ends[1::2, 0] -= 3
    assert _camera(starts, ends, (120, 160)) is None
    centered = starts - (80., 60.)
    angle = .08
    rotated = centered @ np.array(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle)))) + (80., 60.)
    assert _camera(starts, rotated, (120, 160)) is None


def test_local_rotation_and_shear_use_spread_correspondences_with_residuals():
    starts = np.asarray(((0., 0.), (10., 0.), (0., 10.), (10., 10.), (5., 3.)))
    angle = .12
    rotation = np.array(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
    residual = starts @ rotation - starts
    fitted = _local_affine(starts, residual, np.array((5., 5.)), 12)
    assert fitted is not None
    assert fitted[0] == pytest.approx(12 * angle)
    assert fitted[2] < 1e-8
    assert _local_affine(starts[:3], residual[:3], np.array((5., 5.)), 12) is None
    residual[-1] += (4., -6.)
    assert _local_affine(starts, residual, np.array((5., 5.)), 12) is None


def test_local_pooling_counts_actual_matches_and_refuses_disagreeing_motion():
    starts = np.asarray(((5., 5.), (15., 5.)))
    coherent = np.asarray(((3., 0.), (3., 0.)))
    measured = _local_translation(starts, coherent, np.asarray((10., 10.)))
    assert measured is not None and measured[1] == 2
    assert measured[0] == pytest.approx((3., 0.))
    assert _local_translation(starts, np.asarray(((3., 0.), (-3., 0.))), np.asarray((10., 10.))) is None


def test_local_turning_layer_keeps_its_actual_matches_amid_still_marks():
    center = np.asarray((36., 36.))
    offsets = np.asarray(((-16., -16.), (0., -16.), (16., -16.), (-16., 0.),
                          (16., 0.), (-16., 16.), (0., 16.), (16., 16.)))
    yy, xx = np.mgrid[3:70:13, 3:70:13]
    quiet = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    starts = np.concatenate((quiet, offsets + center))
    angle = .16
    rotation = np.asarray(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
    residual = np.concatenate((np.zeros_like(quiet), offsets @ rotation - offsets))
    measured = _angular_motion_layer(starts, residual, center, 24)
    assert measured is not None
    assert measured[0] == pytest.approx(24 * angle)
    assert measured[2] < 1e-8 and measured[3] == len(offsets)
    quiet_owned = np.zeros(len(starts), dtype=bool)
    quiet_owned[0] = True
    assert _angular_motion_layer(starts, residual, center, 24, owned=quiet_owned) is None
    moving_owned = np.zeros(len(starts), dtype=bool)
    moving_owned[len(quiet)] = True
    assert _angular_motion_layer(starts, residual, center, 24, owned=moving_owned) is not None


def test_equally_supported_distinct_turning_layers_remain_unknown():
    offsets = np.asarray(((-6., -6.), (6., -6.), (-6., 6.), (6., 6.),
                          (-10., 0.), (0., -10.), (10., 0.), (0., 10.)))
    starts = np.concatenate((offsets + (14., 36.), offsets + (58., 36.)))
    angle = .25
    rotation = np.asarray(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
    residual = np.concatenate((offsets @ rotation - offsets, offsets @ rotation.T - offsets))
    assert _angular_motion_layer(starts, residual, np.asarray((36., 36.)), 24) is None


def test_dominant_turning_layer_requires_consensus_among_moving_matches():
    center = np.asarray((36., 36.))
    offsets = np.asarray(((-16., -16.), (0., -16.), (16., -16.), (-16., 0.),
                          (16., 0.), (-16., 16.), (0., 16.), (16., 16.)))
    starts = np.concatenate((offsets + center, np.asarray(((4., 4.), (68., 68.)))))
    angle = .16
    rotation = np.asarray(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
    residual = np.concatenate((offsets @ rotation - offsets, np.asarray(((17., -14.), (-16., 13.)))))
    measured = _angular_motion_layer(starts, residual, center, 24)
    assert measured is not None and measured[0] == pytest.approx(24 * angle)
    assert measured[3] == len(offsets) and measured[2] < 1e-8
    outlier_owned = np.zeros(len(starts), dtype=bool)
    outlier_owned[-2:] = True
    assert _angular_motion_layer(starts, residual, center, 24, owned=outlier_owned) is None


def test_observed_still_layer_retains_support_without_proposing_rotation():
    starts = np.asarray(((5., 5.), (5., 25.), (25., 5.), (25., 25.), (15., 10.)))
    measured = _angular_motion_layer(starts, np.zeros_like(starts), np.asarray((15., 15.)), 24)
    assert measured is not None
    assert measured[0] == pytest.approx(0.) and measured[2] == pytest.approx(0.)
    assert measured[3] == len(starts)
    assert _angular_motion_layer(starts, np.zeros_like(starts), np.asarray((15., 15.)), 24,
                                 owned=np.zeros(len(starts), dtype=bool)) is None


def test_partial_edge_tiles_preserve_actual_measured_support():
    before = _textured_picture(119, 157)
    result = pixel_motion_signature(before, before, tile=12)
    assert result is not None
    assert result.shape == (10, 14)
    assert sum(result.point_counts) == len(result.correspondences)
    assert max(result.support) <= len(result.correspondences)
    assert all(error is None or error <= .75 for error in result.fit_error)

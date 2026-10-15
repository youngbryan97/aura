"""A patch's local angular fit remains distinct from material-body pose."""
import numpy as np
from scipy import ndimage

from core.perception.pixel_motion_before_object_identity import pixel_motion_signature
from core.perception.rotation_inside_a_matched_pixel_patch import patch_angle_uncertainty


def test_stationary_point_centre_can_retain_resolved_patch_rotation():
    rng = np.random.default_rng(59)
    before = ndimage.gaussian_filter(rng.uniform(.1, .9, (64, 64, 3)), (1., 1., 0.))
    angle = .12
    rotation = np.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    yy, xx = np.mgrid[:64, :64]
    source = (np.stack((xx, yy), axis=-1) - (32., 32.)) @ rotation + (32., 32.)
    after = np.stack([ndimage.map_coordinates(before[..., c], [source[..., 1], source[..., 0]], order=1)
                      for c in range(3)], axis=-1)
    starts = np.asarray(((32., 32.), (22., 27.), (39., 40.)))
    ends = (starts - (32., 32.)) @ rotation.T + (32., 32.)
    assert np.array_equal(starts[0], ends[0])
    sigma = patch_angle_uncertainty(before, after, starts, ends, np.full(3, angle))
    assert np.isfinite(sigma).all() and np.all((.01 <= sigma) & (sigma < .04))


def test_flat_or_one_dimensional_gradients_cannot_resolve_rotation_and_translation():
    yy, xx = np.mgrid[:40, :60]
    starts = np.asarray(((20., 20.), (30., 20.)))
    for picture in (np.full((40, 60, 3), .5), np.repeat((xx / 60)[..., None], 3, axis=-1)):
        assert np.isinf(patch_angle_uncertainty(picture, picture, starts, starts, np.zeros(2))).all()


def test_invalid_geometry_and_nonfinite_samples_stay_unavailable_with_bounded_work():
    picture = np.full((40, 60, 3), .5)
    starts = np.asarray(((20., 20.),))
    assert patch_angle_uncertainty(picture, picture, np.zeros((257, 2)), np.zeros((257, 2)), np.zeros(257)).size == 0
    assert np.isinf(patch_angle_uncertainty(picture, picture, starts, starts[:, :1], np.zeros(1))).all()
    invalid = picture.copy()
    invalid[0, 0, 0] = np.nan
    assert np.isinf(patch_angle_uncertainty(invalid, picture, starts, starts, np.zeros(1))).all()
    assert np.isinf(patch_angle_uncertainty(picture, picture, starts * np.nan, starts, np.zeros(1))).all()
    assert np.isinf(patch_angle_uncertainty(picture * 1e300, picture, starts, starts, np.zeros(1))).all()
    assert np.isinf(patch_angle_uncertainty(picture, picture, starts * 1e300, starts, np.zeros(1))).all()


def test_pixel_search_retains_local_orientation_with_actual_owned_patch_support():
    rng = np.random.default_rng(41)
    before = np.repeat(ndimage.gaussian_filter(rng.uniform(20, 230, (120, 160)), .5).astype(np.uint8)[..., None], 3, axis=2)
    after = before.copy()
    after[40:80, 60:100] = ndimage.rotate(before[40:80, 60:100], -7., reshape=False, mode='reflect', order=1)
    result = pixel_motion_signature(before, after, point_limit=256, tile=24)
    assert result is not None and len(result.patch_rotations) == np.prod(result.shape)
    assert any(abs(value) > 1. for value, count in zip(result.patch_rotations, result.patch_support, strict=True) if count)
    for receipt in result.correspondences:
        if receipt.patch_angle_error is not None:
            assert 0 < receipt.patch_angle_error <= .2 and abs(receipt.patch_angle) <= .5
            assert receipt.pixel_error < .035 and receipt.round_trip_error < .7
    assert not hasattr(result, 'identity')

"""Rotation search must retain reference sample coordinates and competing copies."""
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.perception.a_visual_reference_in_the_picture import reference_appearances

pytestmark = pytest.mark.unit


def _reference():
    rng = np.random.default_rng(413)
    pixels = rng.integers(25, 210, (64, 80, 4), dtype=np.uint8)
    pixels[..., 3] = 0
    pixels[5:-5, 5:-5, 3] = 255
    return pixels


def _raster(reference, x, y, angle, scale=1.):
    rgba = Image.fromarray(reference).resize((round(reference.shape[1]*scale), round(reference.shape[0]*scale)),
                                            Image.Resampling.BOX)
    return np.asarray(rgba.rotate(-math.degrees(angle), expand=True, resample=Image.Resampling.BICUBIC))


def _paste(picture, pixels, x, y):
    h, w = pixels.shape[:2]
    alpha = pixels[..., 3:4]/255.
    target = picture[y:y+h, x:x+w]
    target[:] = np.round(pixels[..., :3]*alpha+target*(1-alpha)).astype(np.uint8)


@pytest.mark.parametrize('angle', [-math.pi, -math.pi/2, -.55, .55, math.pi/2, math.pi])
def test_oriented_correspondences_return_to_the_original_reference(angle):
    reference = _reference()
    picture = np.full((200, 300, 3), 20, np.uint8)
    _paste(picture, _raster(reference, 70, 60, angle), 70, 60)
    poses = reference_appearances(picture, reference, scales=(1.,), angles=(0., angle))
    assert len(poses) == 1
    pose = poses[0]
    predicted = np.asarray([pose.project(point) for point in pose.source_points])
    assert np.max(np.linalg.norm(predicted-pose.observed_points, axis=1)) <= pose.error+1e-5
    # Measure a marker in an independently rotated raster. The expected point
    # comes from its pixels, rather than the implementation's coordinate map.
    marker = Image.new('RGBA', (80, 64))
    ImageDraw.Draw(marker).ellipse((20, 28, 26, 34), fill=(255, 0, 0, 255))
    marked = _raster(np.asarray(marker), 70, 60, angle)
    weights = marked[..., 3].astype(float)
    yy, xx = np.indices(weights.shape)
    expected = 70+float((xx*weights).sum()/weights.sum()), 60+float((yy*weights).sum()/weights.sum())
    assert math.dist(pose.project((23., 31.)), expected) < .6
    assert not hasattr(pose, 'controlled') and not hasattr(pose, 'role')


@pytest.mark.parametrize('scale', [.75, 1.25])
def test_rotation_and_resizing_keep_the_same_source_samples(scale):
    reference = _reference()
    picture = np.full((240, 360, 3), 20, np.uint8)
    angle = .55
    _paste(picture, _raster(reference, 70, 60, angle, scale), 70, 60)
    poses = reference_appearances(picture, reference, scales=(scale,), angles=(angle,))
    assert len(poses) == 1
    marked = Image.new('RGBA', (80, 64))
    ImageDraw.Draw(marked).ellipse((20, 28, 26, 34), fill=(255, 0, 0, 255))
    weights = _raster(np.asarray(marked), 70, 60, angle, scale)[..., 3].astype(float)
    yy, xx = np.indices(weights.shape)
    expected = 70+float((xx*weights).sum()/weights.sum()), 60+float((yy*weights).sum()/weights.sum())
    assert math.dist(poses[0].project((23., 31.)), expected) < .75


@pytest.mark.parametrize('edge', [256, 512])
def test_reducing_the_scene_keeps_original_pixel_centres(edge):
    reference = _reference()
    picture = np.full((768, 1024, 3), 20, np.uint8)
    _paste(picture, _raster(reference, 300, 200, 0., scale=4.), 300, 200)
    pose, = reference_appearances(picture, reference, scales=(4.,), edge=edge)
    marker = Image.new('RGBA', (80, 64))
    ImageDraw.Draw(marker).ellipse((20, 28, 26, 34), fill=(255, 0, 0, 255))
    weights = _raster(np.asarray(marker), 300, 200, 0., scale=4.)[..., 3].astype(float)
    yy, xx = np.indices(weights.shape)
    expected = 300+float((xx*weights).sum()/weights.sum()), 200+float((yy*weights).sum()/weights.sum())
    assert math.dist(pose.project((23., 31.)), expected) < .25


def test_failed_orientation_cannot_suppress_a_valid_fit_in_the_same_region():
    reference = _reference()
    picture = np.full((240, 360, 3), 20, np.uint8)
    _paste(picture, _raster(reference, 70, 60, .55), 70, 60)
    assert not reference_appearances(picture, reference, scales=(1.,))
    assert len(reference_appearances(picture, reference, scales=(1.,), angles=(0., .25, .55))) == 1


def test_differently_oriented_visible_copies_are_still_competing_referents():
    reference = _reference()
    picture = np.full((240, 400, 3), 20, np.uint8)
    _paste(picture, _raster(reference, 20, 40, .55), 20, 40)
    _paste(picture, _raster(reference, 260, 130, -math.pi/2), 260, 130)
    poses = reference_appearances(picture, reference, scales=(1.,), angles=(.55, -math.pi/2))
    assert len(poses) == 2
    assert abs(poses[0].project((40., 32.))[0]-poses[1].project((40., 32.))[0]) > 150


@pytest.mark.parametrize('angles,scales', [
    ((), (1.,)), ([0.], (1.,)), ((True,), (1.,)), ((float('nan'),), (1.,)),
    ((float('inf'),), (1.,)), ((math.pi+.1,), (1.,)), ((-.1, .1), (1.,)*17),
])
def test_unmeasured_or_unbounded_angle_banks_refuse(angles, scales):
    assert reference_appearances(np.zeros((200, 300, 3), np.uint8), _reference(),
                                 scales=scales, angles=angles) == ()


def test_the_registered_rotation_claim_checks_the_actual_raster_coordinate_mapping():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'rotating_references_keep_their_source_samples')
    assert check.predict(None) is True

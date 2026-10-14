"""Resizing coordinates also resizes their resolution term; it cannot create certainty."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.orientation_from_continuing_marks import _phase_of_pose
from core.perception.points_before_the_picture_is_reduced import PointsBeforeReduction
from core.perception.points_that_keep_their_motion import FollowingPoints, PointMotion
from core.perception.position_from_the_same_points import PointPose, position_from_points
from core.perception.rigid_motion_from_named_points import angular_uncertainty, rigid_motion

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('scale', [.25, .5, 2.])
def test_the_same_geometry_has_the_same_angular_estimate_in_different_units(scale):
    starts = np.array(((0., 0.), (12., 0.), (0., 12.), (12., 12.)))
    ends = starts + (4., 3.)
    ends[-1] += (.1, -.1)
    native, reduced = rigid_motion(starts, ends), rigid_motion(starts*scale, ends*scale)
    assert angular_uncertainty(starts*scale, ends*scale, reduced, resolution=.25*scale) == pytest.approx(
        angular_uncertainty(starts, ends, native))
    marks = {i: tuple(v*scale) for i, v in enumerate(starts)}
    pose = PointPose(0., 1., (6.*scale, 6.*scale), marks, (10.*scale, 9.*scale), reference_at=.5,
        visible_marks={i: tuple(v*scale) for i, v in enumerate(ends)}, angular_reference_at=.5, pixel_size=scale)
    phase = _phase_of_pose(pose)
    assert phase.uncertainty == pytest.approx(angular_uncertainty(starts, ends, native))
    assert phase.position_error == pytest.approx(scale*(.25+native.error))


def test_actual_source_tracking_publishes_the_grid_that_measured_the_points():
    image = np.full((120, 160, 3), 25, np.uint8)
    image[35:65, 40:64] = np.random.default_rng(47).integers(35, 240, (30, 24, 3), dtype=np.uint8)
    working = image[::2, ::2].copy()
    observer = PointsBeforeReduction(limit=96)
    observer.see_source(image, working, 1.)
    observer.see_source(image, working, 1.1)
    body = SimpleNamespace(x=26., y=25., w=12., h=15., born=0., seen=1., support=np.ones((15, 12), bool), point_pose=None)
    motion = observer.motion_of(body, 1.1)
    assert motion is not None and motion.pixel_size == .5
    blob = dict(x=26., y=25., w=12., h=15., support=body.support)
    position_from_points(body, blob, motion, 1.1)
    assert body.point_pose.pixel_size == .5
    ordinary = FollowingPoints(limit=96)
    ordinary.see(working, 1.)
    ordinary.see(working, 1.1)
    assert ordinary.motion_of(body, 1.1).pixel_size == 1.


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf'), True, None])
def test_an_invalid_resolution_cannot_seed_a_pose_or_supply_an_angle(bad):
    marks = {i: p for i, p in enumerate(((15., 15.), (25., 15.), (15., 25.), (25., 25.)))}
    body = SimpleNamespace(born=0., seen=1., point_pose=None)
    blob = dict(x=20., y=20., w=20., h=20., support=np.ones((20, 20), bool))
    motion = PointMotion(0., 0., .01, 4, 1., 1.1, tuple(marks), tuple(marks.values()), pixel_size=bad)
    assert position_from_points(body, blob, motion, 1.1) is None and body.point_pose is None
    pose = PointPose(0., 1., (20., 20.), marks, (20., 20.), reference_at=.5,
                    visible_marks=dict(marks), angular_reference_at=.5, pixel_size=bad)
    assert _phase_of_pose(pose) is None


def test_unknown_resolution_retains_the_working_grid_default():
    motion = PointMotion(0., 0., .01, 3, 1., 1.1)
    assert motion.pixel_size == 1.


def test_a_bounded_tracking_grid_cannot_claim_the_original_images_finer_resolution():
    source = np.full((240, 320, 3), 25, np.uint8)
    source[60:160, 80:160] = np.random.default_rng(93).integers(35, 240, (100, 80, 3), dtype=np.uint8)
    working = source[::4, ::4].copy()
    observer = PointsBeforeReduction(limit=96, pixel_budget=15_000)
    observer.see_source(source, working, 1.)
    observer.see_source(source, working, 1.1)
    body = SimpleNamespace(x=30., y=27.5, w=20., h=25., seen=1., support=np.ones((25, 20), bool))
    motion = observer.motion_of(body, 1.1)
    assert motion is not None
    _original, tracked, mapped = observer._geometry
    assert motion.pixel_size == max(mapped[axis]/tracked[axis] for axis in (0, 1))
    assert motion.pixel_size > working.shape[1]/source.shape[1]


def test_point_precision_has_a_registered_validation():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'point_resolution_keeps_tracked_grid_units')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())

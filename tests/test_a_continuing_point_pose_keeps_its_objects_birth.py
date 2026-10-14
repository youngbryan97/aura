"""A current point pose shares the lifetime restored by continuing material marks."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.orientation_from_continuing_marks import current_angular_phase
from core.perception.points_that_keep_their_motion import SeenPoint
from core.perception.position_from_the_same_points import PointPose
from core.perception.what_moves_in_the_picture import WhatMoves
from core.perception.what_the_points_on_a_thing_say import WhatThePointsSay

pytestmark = pytest.mark.unit


def scene():
    names = {i: p for i, p in enumerate(((2., 2.), (12., 2.), (2., 12.), (12., 12.)))}
    said = WhatThePointsSay()
    said.carried[1] = (frozenset(names), 0., 1.)
    pose = PointPose(1.1, 1.1, (7., 7.), names, (7., 7.), reference_at=1.1,
                     visible_marks=dict(names), angular_reference_at=1.1)
    thing = SimpleNamespace(born=1.1, seen=1.1, box=lambda: (0., 0., 16., 16.),
                            support=np.ones((16, 16), bool), point_pose=pose,
                            motion_evidence='points', motion_measured_at=1.1)
    points = tuple(SeenPoint(i, *p, *p, 0., 1., 1.1, .01, 4) for i, p in names.items())
    eyes = WhatMoves()
    eyes.on_things, eyes.things, eyes.points = said, {2: thing}, SimpleNamespace(points=points)
    return eyes, thing


def test_restored_lifetime_keeps_current_geometry_without_borrowing_an_old_angle_basis():
    eyes, thing = scene()
    eyes._what_the_points_say(1.1)
    assert thing.born == thing.point_pose.born == 0.
    phase = current_angular_phase(thing, 1.1)
    assert phase is not None and phase.reference_at == 1.1 and phase.angle == pytest.approx(0.)


@pytest.mark.parametrize('departure', ['stale', 'stale-object', 'other-birth', 'missing-reference', 'old-reference'])
def test_a_lifetime_change_cannot_retag_an_unrelated_or_stale_pose(departure):
    eyes, thing = scene()
    if departure == 'stale':
        thing.point_pose.at = 1.
    elif departure == 'stale-object':
        thing.seen = 1.
    elif departure == 'other-birth':
        thing.point_pose.born = .5
    elif departure == 'missing-reference':
        thing.point_pose.reference_at = None
    else:
        thing.point_pose.reference_at = 1.
    previous_birth = thing.point_pose.born
    eyes._what_the_points_say(1.1)
    assert thing.point_pose.born == previous_birth
    assert current_angular_phase(thing, 1.1) is None


def test_current_pose_lifetime_has_a_registered_validation():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'current_point_pose_keeps_restored_object_lifetime')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())

"""Material correspondence preserves identity without borrowing another entity's trials."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.perception.continuing_material_before_outlines import material_matches
from core.perception.points_that_keep_their_motion import FollowingPoints, PointMotion
from core.perception.position_from_the_same_points import PointPose
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit


def scene():
    eyes = WhatMoves()
    eyes.shape = (80, 100)
    old = dict(x=20., y=20., w=12., h=36., colour=(170, 80, 80),
               support=np.ones((36, 12), bool), patch=np.full((36, 12, 3), 170, np.uint8),
               look=np.ones(64)/64)
    body = eyes._born(old, .5)
    body.seen = 1.
    body.path.append((body.seen, body.x, body.y))
    body.announced = True
    eyes.things, eyes.pending = {body.number: body}, {}
    marks = {n: p for n, p in enumerate(((16., 28.), (24., 28.), (16., 34.), (24., 34.)))}
    body.point_pose = PointPose(.5, 1., (10., 20.), {n: (x-10., y) for n, (x, y) in marks.items()},
                                (20., 20.), reference_at=.8, visible_marks=dict(marks), angular_reference_at=.8)
    body.motion_position, body.motion_measured_at, body.motion_evidence = (20., 20.), 1., 'points'
    blob = dict(x=23., y=31., w=12., h=10., colour=old['colour'],
                support=np.ones((10, 12), bool), patch=np.full((10, 12, 3), 170, np.uint8), look=old['look'])
    motion = PointMotion(3., 0., .01, 4, 1., 1.1, tuple(marks), tuple((x+3., y) for x, y in marks.values()))
    eyes._point_motion = {body.number: motion}
    return eyes, body, blob, motion


def test_original_object_and_material_origin_survive_outline_shrinkage():
    eyes, body, blob, motion = scene()
    assert eyes._cost(body, blob, 1.1) == float('inf')
    matched, fresh = eyes._match([blob], 1.1)
    assert matched == {body.number: 0} and fresh == []
    previous_number = body.number
    eyes._moved(body, blob, 1.1)
    assert eyes.things[previous_number] is body and body.born == .5
    assert body.point_pose.origin == (10., 20.)
    assert body.motion_position == pytest.approx((23., 20.))
    assert body.motion_evidence == 'points'


@pytest.mark.parametrize('departure', [
    'missing', 'stale-pose', 'other-birth', 'stale-motion', 'gap', 'unsupported-before',
    'unsupported-now', 'duplicate-name', 'deformed', 'collinear', 'too-few', 'nonfinite',
])
def test_unknown_or_inconsistent_correspondence_cannot_override_outline_rejection(departure):
    eyes, body, blob, motion = scene()
    at = 1.1
    if departure == 'missing':
        body.point_pose = None
    elif departure == 'stale-pose':
        body.point_pose.at = .9
    elif departure == 'other-birth':
        body.point_pose.born = 0.
    elif departure == 'stale-motion':
        motion = replace(motion, at=1.)
    elif departure == 'gap':
        at = 1.5
        motion = replace(motion, at=at)
    elif departure == 'unsupported-before':
        body.support[:] = False
    elif departure == 'unsupported-now':
        blob['support'][:] = False
    elif departure == 'duplicate-name':
        motion = replace(motion, points=(0, 0, 2, 3))
    elif departure == 'deformed':
        motion = replace(motion, locations=((18., 27.), (27., 27.), (18., 35.), (30., 35.)))
    elif departure == 'collinear':
        body.point_pose.visible_marks = {n: (16.+2*n, 31.) for n in range(4)}
        motion = replace(motion, locations=tuple((19.+2*n, 31.) for n in range(4)))
    elif departure == 'too-few':
        motion = replace(motion, points=motion.points[:2], locations=motion.locations[:2], support=2)
    else:
        motion = replace(motion, error=float('nan'))
    eyes._point_motion = {body.number: motion}
    assert material_matches(eyes.things, [blob], eyes._point_motion, at) == {}
    assert eyes._match([blob], at)[0] == {}


def test_shared_candidate_and_duplicate_current_support_remain_ambiguous():
    eyes, body, blob, motion = scene()
    assert material_matches(eyes.things, [blob, dict(blob)], eyes._point_motion, 1.1) == {}
    assert material_matches({1: body, 2: body}, [blob], {1: motion, 2: motion}, 1.1) == {}


def test_unique_current_material_precedes_an_outline_only_rival():
    eyes, body, blob, motion = scene()
    rival = eyes._born(dict(blob), 1.05)
    eyes.things[rival.number] = rival
    assert eyes._cost(rival, blob, 1.1) < float('inf')
    assert material_matches(eyes.things, [blob], eyes._point_motion, 1.1) == {body.number: 0}
    assert eyes._match([blob], 1.1) == ({body.number: 0}, [])
    assert rival.born == 1.05 and rival.point_pose is None


@pytest.mark.parametrize('expired', ['pose', 'motion', 'birth', 'support', 'precision'])
def test_unqualified_material_cannot_displace_an_appearance_match(expired):
    eyes, body, blob, motion = scene()
    rival = eyes._born(dict(blob), 1.05)
    eyes.things[rival.number] = rival
    if expired == 'pose':
        body.point_pose.at = .9
    elif expired == 'motion':
        eyes._point_motion[body.number] = replace(motion, at=1.05)
    elif expired == 'birth':
        body.point_pose.born = 0.
    elif expired == 'support':
        body.support[:] = False
    else:
        body.point_pose.reference_error = 1.
    assert material_matches(eyes.things, [blob], eyes._point_motion, 1.1) == {}
    assert eyes._match([blob], 1.1) == ({rival.number: 0}, [])


@pytest.mark.parametrize('seed', [47, 55])
def test_actual_pixel_tracks_preserve_material_when_an_attached_strip_disappears(seed):
    image = np.full((100, 160, 3), 25, np.uint8)
    texture = np.random.default_rng(seed).integers(35, 240, (24, 24, 3), dtype=np.uint8)
    image[50:74, 40:64] = texture
    image[6:50, 51:53] = 120
    mask = np.zeros((68, 24), bool)
    mask[44:] = True
    mask[:44, 11:13] = True
    eyes, following = WhatMoves(), FollowingPoints(limit=96)
    eyes.shape = image.shape[:2]
    old = dict(x=52., y=40., w=24., h=68., colour=(120, 120, 120), look=np.ones(64)/64,
               patch=image[6:74, 40:64].copy(), support=mask)
    body = eyes._born(old, 1.)
    eyes.things, eyes.pending = {body.number: body}, {}
    following.see(image, 1.)
    following.see(image, 1.1)
    seed_motion = following.motion_of(body, 1.1)
    assert seed_motion is not None
    eyes._point_motion = {body.number: seed_motion}
    eyes._moved(body, old, 1.1)
    moved = np.roll(image, 10, axis=1)
    following.see(moved, 1.2)
    moving = following.motion_of(body, 1.2)
    assert moving is not None
    eyes._point_motion = {body.number: moving}
    eyes._moved(body, dict(old, x=62., patch=moved[6:74, 50:74].copy()), 1.2)
    current = np.full_like(image, 25)
    current[50:74, 53:77] = texture
    following.see(current, 1.3)
    motion = following.motion_of(body, 1.3)
    assert motion is not None
    blob = dict(x=65., y=62., w=24., h=24., colour=old['colour'], look=old['look'],
                patch=current[50:74, 53:77].copy(), support=np.ones((24, 24), bool))
    eyes._point_motion = {body.number: motion}
    assert eyes._cost(body, blob, 1.3) == float('inf')
    rival = eyes._born(dict(blob), 1.25)
    eyes.things[rival.number] = rival
    assert eyes._cost(rival, blob, 1.3) < float('inf')
    assert eyes._match([blob], 1.3) == ({body.number: 0}, [])
    eyes._moved(body, blob, 1.3)
    assert body.motion_position == pytest.approx((65., 40.), abs=.3)
    assert body.born == 1. and body.point_pose.reference_at == 1.1


def test_material_identity_has_a_registered_validation():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'material_identity_requires_unique_current_geometry')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


def test_material_priority_has_a_registered_validation():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'unique_current_material_precedes_appearance')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())

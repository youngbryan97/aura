"""A joined outline cannot transfer a body's motion to the surrounding texture."""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.points_before_the_picture_is_reduced import PointsBeforeReduction
from core.perception.points_that_keep_their_motion import FollowingPoints, SeenPoint
from core.perception.position_from_the_same_points import PointPose
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('native', [False, True])
def test_actual_pixels_keep_the_moving_material_after_the_outline_includes_static_texture(native):
    rng = np.random.default_rng(185)
    floor = np.full((100, 160, 3), 25, np.uint8)
    floor[10:90, 10:145] = rng.integers(40, 220, (80, 135, 3), dtype=np.uint8)
    texture = rng.integers(40, 240, (24, 24, 3), dtype=np.uint8)
    def picture(shift):
        current = floor.copy()
        current[50:74, 40+shift:64+shift] = texture
        return current
    def blob(current, x, y, w, h):
        from core.perception.what_moves_in_the_picture import _look_of
        left, top = int(x-w/2), int(y-h/2)
        patch = current[top:top+h, left:left+w].copy()
        return dict(x=float(x), y=float(y), w=float(w), h=float(h), colour=(120, 120, 120),
                    look=_look_of(patch.reshape(-1, 3)), patch=patch, support=np.ones((h, w), bool))
    eyes = WhatMoves()
    points = PointsBeforeReduction(limit=256) if native else FollowingPoints(limit=256)
    def see(picture, at):
        return points.see_source(picture, picture, at) if native else points.see(picture, at)
    eyes.shape = floor.shape[:2]
    first = picture(0)
    original = blob(first, 52, 62, 24, 24)
    body = eyes._born(original, 1.)
    eyes.things, eyes.pending = {body.number: body}, {}
    see(first, 1.)
    see(first, 1.1)
    seed = points.motion_of(body, 1.1)
    assert seed is not None
    eyes._point_motion = {body.number: seed}
    eyes._moved(body, original, 1.1)
    assert body.point_pose is not None
    # Establish moving material before testing a changed outline. Border
    # corners can disappear on the first displacement; that supplies no
    # permission to weaken the retained-mark fraction.
    for at, offset in ((1.2, 4), (1.3, 8), (1.4, 12)):
        observed = picture(offset)
        see(observed, at)
        moving = points.motion_of(body, at)
        eyes._point_motion = {} if moving is None else {body.number:moving}
        eyes._moved(body, blob(observed, 52+offset, 62, 24, 24), at)
    assert body.point_pose is not None and body.point_pose.reference_at < 1.4
    joined_picture = picture(16)
    see(joined_picture, 1.5)
    enters = points.motion_of(body, 1.5)
    assert enters is not None and enters.dx == pytest.approx(4., abs=.15)
    joined = blob(joined_picture, 77, 50, 134, 80)
    eyes._point_motion = {body.number: enters}
    assert eyes._match([joined], 1.5) == ({body.number: 0}, [])
    eyes._moved(body, joined, 1.5)
    assert body.point_pose is not None
    owned = set(body.point_pose.anchors)
    current = picture(20)
    see(current, 1.6)
    assert sum(p.before == body.seen and p.number not in owned for p in points.points) > 2*len(owned)
    moves = points.motion_of(body, 1.6)
    assert moves is not None and moves.dx == pytest.approx(4., abs=.15)
    assert len(set(moves.points) & owned) >= 3
    separate = blob(current, 72, 62, 24, 24)
    eyes._point_motion = {body.number: moves}
    assert eyes._match([separate], 1.6) == ({body.number: 0}, [])
    eyes._moved(body, separate, 1.6)
    assert body.born == 1. and body.motion_evidence == 'points'
    assert body.motion_position == pytest.approx((72., 62.), abs=.2)


def scene(dx=4., dy=0.):
    from types import SimpleNamespace

    own = dict(enumerate(((15., 15.), (25., 15.), (15., 25.), (25., 25.))))
    pose = PointPose(0., 1., (20., 20.), own.copy(), (20., 20.), visible_marks=own.copy())
    body = SimpleNamespace(point_pose=pose, born=0., seen=1., x=40., y=40., w=80., h=80.,
                           support=np.ones((80, 80), bool))
    points = FollowingPoints(limit=256)
    moving = [SeenPoint(n, x+dx, y+dy, x, y, 0., 1., 1.1, .01, 3) for n, (x, y) in own.items()]
    other = [SeenPoint(n, x+4., y+4., x, y, 0., 1., 1.1, .01, 3)
             for n, (x, y) in enumerate(((x, y) for x in (35., 45., 55., 65.) for y in (35., 45., 55., 65.)), 4)]
    points.points = tuple(moving+other)
    points._at = 1.1
    return points, body


@pytest.mark.parametrize('dx,dy', [(0.,0.), (4.,0.), (0.,-4.)])
def test_other_visible_motion_cannot_replace_the_continuing_bodys_motion(dx,dy):
    points, body = scene(dx,dy)
    measured = points.motion_of(body,1.1)
    assert measured is not None and (measured.dx,measured.dy) == pytest.approx((dx,dy))
    assert set(measured.points) == set(body.point_pose.anchors)


@pytest.mark.parametrize('bad', ['missing', 'stale', 'birth', 'coordinate', 'duplicate', 'deformed'])
def test_missing_or_inconsistent_owned_marks_cannot_yield_to_the_other_consensus(bad):
    from dataclasses import replace

    points, body = scene()
    if bad == 'missing':
        points.points = points.points[2:]
    elif bad == 'stale':
        body.point_pose.at = .9
    elif bad == 'birth':
        body.point_pose.born = -.1
    elif bad == 'coordinate':
        body.point_pose.visible_marks = {n:(x+1.,y) for n,(x,y) in body.point_pose.visible_marks.items()}
    elif bad == 'duplicate':
        points.points = (points.points[0],)*4 + points.points[4:]
    else:
        points.points = tuple(replace(p,x=2*p.x) if p.number < 4 else p for p in points.points)
    assert points.motion_of(body,1.1) is None


def test_owned_material_motion_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.points_that_keep_their_motion import _owned_material_motion_invariant

    assert _owned_material_motion_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'retained_material_precedes_a_joined_outline_majority')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())


def test_a_measured_rotation_stays_with_its_own_marks():
    from dataclasses import replace

    from core.perception.rigid_motion_from_named_points import RigidMotion

    points, body = scene()
    truth = RigidMotion(.3,(7.,-3.),(),0.)
    current = []
    for point in points.points:
        if point.number in body.point_pose.anchors:
            x,y = truth.apply((point.old_x,point.old_y))
            point = replace(point,x=float(x),y=float(y))
        current.append(point)
    points.points = tuple(current)
    measured = points.motion_of(body,1.1)
    assert measured is not None and measured.has_rigid_pose()
    assert measured.rigid.angle == pytest.approx(.3)
    assert measured.rigid.apply((20.,20.)) == pytest.approx(truth.apply((20.,20.)))


def test_the_bounded_point_receipt_keeps_owned_names_before_new_coherent_corners():
    from dataclasses import replace

    points, body = scene()
    owned = list(points.points[:4])
    extra = [SeenPoint(100+number,x+4.,y,x,y,0.,1.,1.1,.01,3)
             for number,(x,y) in enumerate((x,y) for x in range(30,60) for y in range(30,60))]
    points.points = tuple(extra+owned)
    measured = points.motion_of(body,1.1)
    assert measured is not None and measured.support <= 256
    assert set(body.point_pose.anchors) <= set(measured.points)
    points.points = tuple(replace(p,number=100) if p.number >= 100 else p for p in points.points)
    assert points.motion_of(body,1.1) is None

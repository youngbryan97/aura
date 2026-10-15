"""Observed material paths can bridge a missing outline, without filling missing pixels."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from core.perception.points_before_the_picture_is_reduced import PointsBeforeReduction
from core.perception.points_that_keep_their_motion import FollowingPoints

pytestmark = pytest.mark.unit


def _tracking(native=False):
    image = np.full((120, 160, 3), 25, np.uint8)
    image[35:65, 40:64] = np.random.default_rng(918).integers(35, 240, (30, 24, 3), dtype=np.uint8)
    thing = SimpleNamespace(number=1, x=52., y=50., w=24., h=30., seen=1., support=np.ones((30, 24), bool))
    points = PointsBeforeReduction(pixel_budget=1_000_000) if native else FollowingPoints()

    def see(offset, at):
        picture = ndimage.shift(image, (0, offset, 0), order=1, mode='nearest')
        if native:
            points.see_source(np.repeat(np.repeat(picture, 2, 0), 2, 1), picture, at)
        else:
            points.see(picture, at)

    for offset, at in ((0., 1.), (2., 1.1), (4., 1.2)):
        see(offset, at)
    return points, thing, see


@pytest.mark.parametrize('native', [False, True])
def test_every_intervening_measured_point_supplies_displacement_from_the_last_owned_support(native):
    points, thing, _see = _tracking(native)
    measured = points.motion_of(thing, 1.2)
    assert measured is not None and measured.before == 1. and measured.at == 1.2
    assert measured.support >= 3
    assert measured.dx == pytest.approx(4., abs=.35) and measured.dy == pytest.approx(0., abs=.35)
    assert set(measured.points) <= points._point_history[1.].keys()
    assert all(points._point_history[1.2][name].before == 1.1 for name in measured.points)


@pytest.mark.parametrize('bad', ['missing', 'birth', 'count', 'time', 'coords', 'error'])
def test_a_broken_or_reused_point_path_cannot_bridge_the_missing_outline(bad):
    points, thing, _see = _tracking()
    frame = points._point_history[1.1]
    if bad == 'missing':
        frame.clear()
    else:
        changes = {'birth': {'born': 1.1}, 'count': {'pictures': 1}, 'time': {'before': .9},
                   'coords': {'x': -100.}, 'error': {'error': float('nan')}}[bad]
        points._point_history[1.1] = {name: replace(point, **changes) for name, point in frame.items()}
    assert points.motion_of(thing, 1.2) is None


def test_points_in_the_old_box_but_outside_its_material_support_do_not_belong_to_it():
    points, thing, _see = _tracking()
    thing.support[:] = False
    assert points.motion_of(thing, 1.2) is None


def test_a_projected_public_box_cannot_move_the_old_material_support():
    points, thing, _see = _tracking()
    thing.path = [(thing.seen, thing.x, thing.y)]
    thing.x, thing.y = 130., 90.
    measured = points.motion_of(thing, 1.2)
    assert measured is not None and measured.dx == pytest.approx(4., abs=.35)
    thing.path[-1] = (.9, 52., 50.)
    assert points.motion_of(thing, 1.2) is None


def test_the_ordinary_motion_eyes_keep_material_through_one_missing_outline():
    from core.perception.what_moves_in_the_picture import WhatMoves

    image = np.full((120, 160, 3), 25, np.uint8)
    texture = np.random.default_rng(918).integers(35, 240, (30, 24, 3), dtype=np.uint8)
    eyes = WhatMoves(width=160)
    for step in range(8):
        eyes.see(image, step*.1)
    blobs = eyes._blobs
    eyes._blobs = lambda *args: [] if eyes._now == 1.3 else blobs(*args)
    number = None
    for step, at in enumerate((1., 1.1, 1.2, 1.3, 1.4)):
        picture = image.copy()
        picture[35:65, 40+step*2:64+step*2] = texture
        eyes.see(picture, at)
        if number is None:
            number = next(iter(eyes.things))
        if at == 1.3:
            assert eyes.things[number].seen == 1.2
        elif at == 1.4:
            assert eyes.things[number].seen == at
            motion = eyes._point_motion[number]
            assert motion.before == 1.2 and motion.at == at
            assert motion.dx == pytest.approx(4., abs=.35)


def test_the_bridge_cannot_outlive_the_existing_correspondence_horizon_or_a_reset():
    points, thing, see = _tracking()
    see(6., 1.3)
    assert points.motion_of(thing, 1.3) is not None
    see(8., 1.4)
    assert points.motion_of(thing, 1.4) is None and 1. not in points._point_history
    points.clear()
    assert not points._point_history and points.motion_of(thing, 1.4) is None


def test_many_fast_frames_have_a_fixed_history_budget():
    points, _thing, see = _tracking()
    for step in range(40):
        see(4., 1.201+step*.001)
    assert len(points._point_history) <= 32
    assert sum(len(frame) for frame in points._point_history.values()) <= 32*points.limit


def test_current_visible_corrections_belong_to_the_saved_capture_history(monkeypatch):
    from core.perception.learned_tracking import corrections_while_points_keep_moving as correcting
    from tests.test_learned_corrections_do_not_hold_up_visible_tracking import marks, picture

    frames = [picture(1.),picture(1.05)]
    transport = correcting.visible_transport(marks(1.),frames,256_000)
    assert transport is not None
    points = correcting.CorrectionsWhilePointsMove(lambda:None,limit=16)
    monkeypatch.setattr(points,'_submit',lambda _frame:None)
    monkeypatch.setattr(points,'_receive',lambda:None)
    first = picture(1.1)
    points.see_source(first.source,first.working,first.at)
    points._transport = transport
    body = SimpleNamespace(number=1,x=36.,y=32.,w=32.,h=32.,seen=1.2,support=np.ones((32,32),bool))
    for at in (1.2,1.3,1.4):
        frame = picture(at)
        points.see_source(frame.source,frame.working,at)
        named = {p.number:p for p in points.points}
        assert points._point_history[at] == named
    motion = points.motion_of(body,1.4)
    assert motion is not None and motion.before == 1.2 and motion.at == 1.4
    assert any(name >= correcting._LEARNED_NAMES for name in motion.points)
    points._point_history[1.3].clear()
    assert points.motion_of(body,1.4) is None


def test_replacing_one_capture_publication_does_not_create_an_extra_time_sample():
    points,thing,_see = _tracking()
    captured_times = tuple(points._point_history)
    points._remember_points(replacing_current=True)
    assert tuple(points._point_history) == captured_times
    assert points.motion_of(thing,1.2) is not None
    points._at = 1.1
    points._remember_points(replacing_current=True)
    assert tuple(points._point_history) == (1.1,)


def test_point_history_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.points_that_keep_their_motion import _point_history_invariant

    assert _point_history_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'a_missing_outline_requires_continuous_observed_marks')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())

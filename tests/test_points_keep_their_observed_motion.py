"""Points keep their own identities and constrain association using measured motion."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from core.perception.points_that_keep_their_motion import FollowingPoints, PointMotion

pytestmark = pytest.mark.unit


def _scene():
    image = np.full((120, 160, 3), 25, np.uint8)
    texture = np.random.default_rng(47).integers(35, 240, (30, 24, 3), dtype=np.uint8)
    image[35:65, 40:64] = texture
    thing = SimpleNamespace(number=1, x=52., y=50., w=24., h=30., seen=1., support=np.ones((30, 24), bool))
    return image, thing


@pytest.mark.parametrize("dx,dy", [(3., 0.), (-5., 2.), (0., -7.), (1.5, 2.25), (9., 0.)])
def test_texture_points_measure_displacement_in_every_direction(dx, dy):
    before, thing = _scene()
    points = FollowingPoints()
    first = {p.number for p in points.see(before, 1.)}
    after = ndimage.shift(before, (dy, dx, 0), order=1, mode="nearest")
    points.see(after, 1.1)
    measured = points.motion_of(thing, 1.1)
    assert measured is not None
    assert measured.dx == pytest.approx(dx, abs=.35)
    assert measured.dy == pytest.approx(dy, abs=.35)
    assert measured.support >= 3 and set(measured.points) <= first


def test_points_keep_their_names_across_several_pictures_without_object_detections():
    before, thing = _scene()
    points = FollowingPoints()
    first = {p.number for p in points.see(before, 1.)}
    for i in range(1, 6):
        points.see(ndimage.shift(before, (0, i * 2, 0), order=1, mode="nearest"), 1. + i * .05)
    survivors = [p for p in points.points if p.number in first]
    assert len(survivors) >= 3 and all(p.pictures == 6 for p in survivors)
    measured = points.motion_of(thing, 1.25)
    assert measured is not None and measured.before == thing.seen
    assert measured.dx == pytest.approx(10., abs=.35) and measured.dy == pytest.approx(0., abs=.35)


def test_occlusion_does_not_invent_a_continuing_track():
    before, thing = _scene()
    points = FollowingPoints()
    points.see(before, 1.)
    points.see(np.full_like(before, 25), 1.1)
    assert points.motion_of(thing, 1.1) is None
    assert not points.points


def test_flat_regions_and_stale_objects_are_unmeasured():
    before, thing = _scene()
    points = FollowingPoints()
    points.see(before, 1.)
    points.see(before, 1.1)
    thing.seen = .9
    assert points.motion_of(thing, 1.1) is None
    assert not FollowingPoints().see(np.full_like(before, 25), 1.)


@pytest.mark.parametrize("change", ["gap", "resize", "reset", "backward_time"])
def test_unrelated_observations_do_not_reuse_point_identities(change):
    before, _thing = _scene()
    points = FollowingPoints()
    old = {p.number for p in points.see(before, 1.)}
    if change == "reset":
        points.clear()
    image = before[:80, :100] if change == "resize" else before
    at = 2. if change == "gap" else .9 if change == "backward_time" else 1.1
    fresh = points.see(image, at)
    assert fresh and not old & {p.number for p in fresh}
    assert all(p.pictures == 1 for p in fresh)


def test_corner_selection_keeps_a_small_patch_beside_a_textured_background():
    image = np.full((144, 240, 3), 25, np.uint8)
    image[:, :180] = np.random.default_rng(39).integers(20, 240, (144, 180, 3), dtype=np.uint8)
    image[40:64, 200:224] = np.random.default_rng(47).integers(35, 240, (24, 24, 3), dtype=np.uint8)
    points = FollowingPoints(limit=96)
    seen = points.see(image, 1.)
    assert len(seen) <= 96 and sum(p.x >= 200 for p in seen) >= 3


def test_opposed_textures_keep_their_own_movement():
    before, first = _scene()
    second = SimpleNamespace(number=2, x=112., y=50., w=24., h=30., seen=1., support=np.ones((30, 24), bool))
    before[35:65, 100:124] = np.random.default_rng(56).integers(35, 240, (30, 24, 3), dtype=np.uint8)
    after = np.full_like(before, 25)
    after[35:65, 45:69] = before[35:65, 40:64]
    after[35:65, 95:119] = before[35:65, 100:124]
    points = FollowingPoints()
    points.see(before, 1.)
    points.see(after, 1.1)
    assert points.motion_of(first, 1.1).dx == pytest.approx(5., abs=.35)
    assert points.motion_of(second, 1.1).dx == pytest.approx(-5., abs=.35)


def _association():
    from core.perception.what_moves_in_the_picture import WhatMoves

    moves = WhatMoves()
    look = np.array([.5, .5])
    thing = SimpleNamespace(number=1, x=40., y=50., w=12., h=12., size=144., seen=1.,
                            vx=0., vy=0., path=[], look=look)
    old = {"x": 40., "y": 50., "w": 12., "h": 12., "look": look}
    return moves, thing, old


def test_current_point_motion_can_find_a_jump_outside_velocity_prediction():
    moves, thing, old = _association()
    current = {**old, "x": 55.}
    assert moves._cost(thing, current, 1.1) == float("inf")
    moves._point_motion = {1: PointMotion(15., 0., .01, 5, 1., 1.1)}
    assert moves._cost(thing, current, 1.1) < float("inf")
    assert moves._cost(thing, old, 1.1) == float("inf")
    assert moves._leap_cost(thing, old, 1.1) == float("inf")


def test_unmoved_points_prevent_changing_scenery_from_inheriting_motion():
    moves, thing, old = _association()
    thing.vx = 150.
    moves._point_motion = {1: PointMotion(0., 0., .01, 5, 1., 1.1)}
    assert moves._cost(thing, old, 1.1) < float("inf")
    assert moves._cost(thing, {**old, "x": 55.}, 1.1) == float("inf")
    assert moves._leap_cost(thing, {**old, "x": 55.}, 1.1) == float("inf")


def test_old_point_motion_does_not_overrule_a_current_detection():
    moves, thing, old = _association()
    moves._point_motion = {1: PointMotion(15., 0., .01, 5, .9, 1.)}
    assert moves._cost(thing, old, 1.1) < float("inf")


@pytest.mark.parametrize("points", [True, False])
def test_a_fragment_cannot_turn_into_a_much_larger_patch_in_one_measurement(points):
    moves, thing, old = _association()
    moves._point_motion = {1: PointMotion(0., 0., .01, 5, 1., 1.1)} if points else {}
    assert moves._cost(thing, {**old, "w": 60., "h": 60.}, 1.1) == float("inf")
    assert moves._cost(thing, {**old, "w": 15., "h": 14.}, 1.1) < float("inf")


def test_texture_moving_inside_a_fixed_outline_does_not_become_body_velocity():
    from core.perception.what_moves_in_the_picture import Thing

    moves, _thing, old = _association()
    thing = Thing(1, 40., 50., 12., 12., old["look"], (200, 200, 200), np.ones((12, 12, 3)), 1., 1.)
    thing.path.append((1., 40., 50.))
    blob = {**old, "colour": thing.colour, "patch": np.ones((12, 12, 3))}
    moves._point_motion = {1: PointMotion(2., 0., .01, 5, 1., 1.1)}
    assert moves._cost(thing, blob, 1.1) < float("inf")
    moves._moved(thing, blob, 1.1)
    assert thing.x == 40. and thing.vx == 0. and thing.vy == 0.


def test_the_point_identity_invariant():
    from core.perception.points_that_keep_their_motion import _point_identity_invariant

    assert _point_identity_invariant() == ()


def test_the_real_picture_reader_uses_point_motion_and_clears_it_with_the_scene():
    from core.perception.what_moves_in_the_picture import WhatMoves

    picture, _thing = _scene()
    moves = WhatMoves()
    for i in range(20):
        moves.see(np.full_like(picture, 25), i * .03)
    moves.see(picture, .6)
    moves.see(picture, .63)
    moves.see(ndimage.shift(picture, (0, 3, 0), order=1, mode="nearest"), .66)
    assert moves._point_motion
    assert any(m.dx == pytest.approx(3., abs=.35) for m in moves._point_motion.values())
    assert any(p.pictures > 2 for p in moves.points.points)
    moves._start_again()
    assert not moves._point_motion and not moves.points.points


def test_the_registered_point_measurement_runs():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(item for item in suite.tests() if item.name == "point_identity_needs_current_pixels")
    assert check.predict(None) is True


def test_background_points_inside_an_objects_box_cannot_supply_its_motion():
    from core.perception.points_that_keep_their_motion import SeenPoint

    points = FollowingPoints()
    points.points = tuple(SeenPoint(i, float(x), float(y), float(x), float(y), 1., 1., 1.1, 0., 3)
                          for i, (x, y) in enumerate([(42, 36), (60, 36), (42, 60), (60, 60)]))
    _picture, thing = _scene()
    thing.support[:] = False
    thing.support[10:20, 8:16] = True
    assert points.motion_of(thing, 1.1) is None
    thing.support = None
    assert points.motion_of(thing, 1.1) is None


def test_spatial_matching_preserves_the_exhaustive_assignment():
    from core.perception.what_moves_in_the_picture import Thing, WhatMoves

    rng = np.random.default_rng(97)
    for _case in range(12):
        moves = WhatMoves()
        for i in range(25):
            x, y, w, h = rng.uniform(10, 200), rng.uniform(10, 120), rng.uniform(4, 30), rng.uniform(4, 30)
            look = rng.dirichlet(np.ones(4))
            thing = Thing(i, x, y, w, h, look, (20, 20, 20), np.zeros((2, 2, 3)), 1., 1.)
            thing.vx, thing.vy = rng.uniform(-300, 300, 2)
            moves.things[i] = thing
            if i % 3 == 0:
                moves._point_motion[i] = PointMotion(*rng.uniform(-20, 20, 2), .01, 4, 1., 1.1)
        blobs = [{"x": rng.uniform(0, 240), "y": rng.uniform(0, 144),
                  "w": rng.uniform(4, 60), "h": rng.uniform(4, 60), "look": rng.dirichlet(np.ones(4))}
                 for _ in range(35)]
        expected, taken = {}, set()
        for measure in (moves._cost, moves._leap_cost):
            pairs = sorted((cost, number, i) for number, thing in moves.things.items() if number not in expected
                           for i, blob in enumerate(blobs) if i not in taken
                           if (cost := measure(thing, blob, 1.1)) < float("inf"))
            for _cost, number, i in pairs:
                if number not in expected and i not in taken:
                    expected[number] = i
                    taken.add(i)
        actual, fresh = moves._match(blobs, 1.1)
        assert actual == expected
        assert fresh == [i for i in range(len(blobs)) if i not in taken]


def test_old_scenery_leaves_point_slots_for_a_new_object():
    image = np.full((144, 240, 3), 25, np.uint8)
    image[:, :180] = np.random.default_rng(39).integers(20, 240, (144, 180, 3), dtype=np.uint8)
    points = FollowingPoints(limit=96)
    before = points.see(image, 1.)
    assert len(before) == 96 and all(p.x < 184 for p in before)
    image[40:64, 200:224] = np.random.default_rng(47).integers(35, 240, (24, 24, 3), dtype=np.uint8)
    for i in range(1, 4):
        points.see(image, 1. + .05 * i)
    own = [p for p in points.points if p.x >= 200]
    assert len(own) >= 3 and any(p.pictures > 1 for p in own)
    assert any(p.born == 1. for p in points.points)
    assert len(points.points) <= 96


def test_a_proposed_outline_keeps_the_points_that_measured_its_motion():
    motion = PointMotion(2., 0., .01, 4, 1., 1.1, (1, 2, 3, 4), ((41., 46.), (47., 46.), (41., 53.), (47., 53.)))
    blob = {"x": 44., "y": 50., "w": 12., "h": 12., "support": np.ones((12, 12), bool)}
    assert motion.inside(blob)
    blob["support"][1:11, 1:11] = False
    assert not motion.inside(blob)


def test_without_current_points_another_appearance_cannot_inherit_a_track():
    moves, thing, old = _association()
    thing.look = np.array([1., 0.])
    replacement = {**old, "look": np.array([.3, .7])}
    assert moves._cost(thing, replacement, 1.1) == float("inf")
    moves._point_motion = {1: PointMotion(0., 0., .01, 5, .9, 1.)}
    assert moves._cost(thing, replacement, 1.1) == float("inf")
    assert moves._cost(thing, {**old, "look": np.array([.8, .2])}, 1.1) < float("inf")


def test_shared_kind_does_not_bypass_measured_shape_on_return():
    from core.perception.what_moves_in_the_picture import Thing, _back_in_sight

    moves, _thing, old = _association()
    body = Thing(1, 40., 50., 12., 12., old["look"], (200, 200, 200), np.ones((12, 12, 3)), 0., 1., kind=0)
    other = Thing(2, 40., 50., 60., 60., old["look"], (200, 200, 200), np.ones((60, 60, 3)), 1.2, 1.2, kind=0)
    moves.out_of_sight[1] = (body, 1.)
    assert _back_in_sight(moves, other, 1.2) is None
    other.w = other.h = 12.
    assert _back_in_sight(moves, other, 1.2) is body


def test_missing_point_identity_has_a_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(item for item in suite.tests() if item.name == "missing_points_do_not_relax_object_identity")
    assert check.predict(None) is True

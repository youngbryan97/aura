"""Current named landmarks and old texture supply bounded observed parts."""
import math
from dataclasses import replace

import numpy as np
import pytest

from core.perception.points_that_keep_their_motion import PointMotion
from core.perception.texture_that_continues_when_outlines_join import (
    _continuing_texture_invariant,
    parts_from_continuing_texture,
)
from core.perception.what_moves_in_the_picture import Thing, WhatMoves

pytestmark = pytest.mark.unit


def _scene():
    picture = np.full((30, 40, 3), 80, np.uint8)
    picture[10:18, 12:20] = 240
    look = np.zeros(64)
    look[63] = 1.
    body = Thing(1, 14., 14., 8., 8., look, (240, 240, 240), np.full((8, 8, 3), 240, np.uint8),
                 born=0., seen=1., support=np.ones((8, 8), bool))
    motion = PointMotion(2., 0., .01, 3, 1., 1.1, (1, 2, 3), ((13., 11.), (18., 11.), (13., 16.)))
    return picture, np.ones((30, 40), bool), body, motion


def _parts(picture, inside, body, motion, **changes):
    options = dict(left=0, top=0, things=[body], motions={1: motion}, at=1.1, difference=40.)
    options.update(changes)
    return parts_from_continuing_texture(picture, inside, **options)


def test_the_texture_part_contains_only_current_pixels_and_preserves_every_other_pixel():
    picture, inside, body, motion = _scene()
    before = inside.copy()
    parts = _parts(picture, inside, body, motion)
    assert len(parts) == 2 and int(parts[0].sum()) == 64
    assert np.all(picture[parts[0]] == 240)
    assert np.array_equal(np.sum(parts, axis=0), inside)
    assert np.array_equal(inside, before)


@pytest.mark.parametrize("offset", [-.75, -.3, 0., .3, .75])
def test_pixel_sampling_phase_cannot_discard_an_otherwise_supported_texture(offset):
    picture, inside, body, motion = _scene()
    motion = replace(motion, dx=2. + offset)
    parts = _parts(picture, inside, body, motion)
    assert len(parts) == 2 and int(parts[0].sum()) == 64


@pytest.mark.parametrize('gap', [.1, .268130064])
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize("bad", ["stale", "gap", "duplicate", "missing_location", "nan", "negative_error",
                               "two", "collinear", "outside", "unowned", "changed_texture", "birth", "geometry"])
def test_unsupported_landmarks_or_pixels_cannot_split_an_object_from_the_patch(bad, gap, sparse):
    picture, inside, body, motion = _scene()
    if sparse:
        inside[:] = False
        inside[10:18, 12:20] = True
        inside[6:28, 20:22] = True
        picture[6:28, 20:22] = 130
        assert inside.sum() <= 2 * body.support.sum()
        assert inside.size > 2 * body.support.size
    at = 1. + gap
    motion = replace(motion, at=at)
    if bad == "stale":
        motion = replace(motion, before=.9)
    elif bad == "gap":
        body.seen = .5
        motion = replace(motion, before=.5)
    elif bad == "duplicate":
        motion = replace(motion, points=(1, 1, 1))
    elif bad == "missing_location":
        motion = replace(motion, locations=motion.locations[:2])
    elif bad == "nan":
        motion = replace(motion, dx=math.nan)
    elif bad == "negative_error":
        motion = replace(motion, error=-1.)
    elif bad == "two":
        motion = replace(motion, support=2, points=(1, 2), locations=motion.locations[:2])
    elif bad == "collinear":
        motion = replace(motion, locations=((13., 11.), (15., 11.), (18., 11.)))
    elif bad == "outside":
        motion = replace(motion, locations=((25., 10.), (30., 10.), (25., 15.)))
    elif bad == "unowned":
        inside[10:18, 12:20] = False
    elif bad == "changed_texture":
        picture[10:18, 12:20] = 0
    elif bad == "birth":
        body.born = 1.05
    elif bad == "geometry":
        body.w = 9.
    parts = _parts(picture, inside, body, motion, at=at)
    assert len(parts) == 1 and parts[0] is inside


def test_missing_motion_and_a_stationary_wall_cannot_supply_a_moving_cutout():
    picture, inside, body, _motion = _scene()
    picture[10:18, :] = 240
    assert _parts(picture, inside, body, None)[0] is inside


def test_a_sparse_matching_colour_outline_cannot_invent_motion_evidence():
    picture, _inside, body, _motion = _scene()
    inside = np.zeros(picture.shape[:2], bool)
    inside[10:18, 12:20] = True
    inside[6:28, 20:22] = True
    picture[6:28, 20:22] = body.colour
    parts = _parts(picture, inside, body, None)
    assert len(parts) == 1 and parts[0] is inside


def test_clipped_or_occluded_texture_does_not_certify_a_whole_body():
    picture, inside, body, motion = _scene()
    picture[14:18, 12:20] = 0
    parts = _parts(picture, inside, body, motion)
    assert len(parts) == 1


@pytest.mark.parametrize("budget", [0, True, 33])
def test_proposals_need_a_bounded_literal_budget(budget):
    picture, inside, body, motion = _scene()
    assert _parts(picture, inside, body, motion, proposals=budget)[0] is inside


def test_a_proposal_budget_preserves_the_pixels_of_other_supported_bodies():
    picture, inside, body, motion = _scene()
    picture[10:18, 28:36] = 240
    other = Thing(2, 30., 14., 8., 8., body.look.copy(), body.colour, body.patch.copy(),
                  born=0., seen=1., support=body.support.copy())
    second = replace(motion, points=(4, 5, 6), locations=((29., 11.), (34., 11.), (29., 16.)))
    parts = _parts(picture, inside, body, motion, things=[body, other], motions={1: motion, 2: second}, proposals=1)
    assert len(parts) == 2 and int(parts[0].sum()) == 64
    assert parts[1][10:18, 28:36].all()
    assert np.array_equal(np.sum(parts, axis=0), inside)


def test_the_normal_motion_eyes_separate_a_supported_body_and_keep_the_remainder():
    picture, _inside, body, motion = _scene()
    picture[8:22, 20:34] = (180, 0, 0)
    eyes = WhatMoves(width=40)
    eyes._backdrop = np.zeros_like(picture, dtype=np.float32)
    eyes._now = 1.1
    eyes.things = {1: body}
    eyes._point_motion = {1: motion}
    mask = np.zeros((30, 40), np.uint8)
    mask[10:18, 12:20] = 1
    mask[8:22, 20:34] = 1
    blobs = eyes._blobs(picture, (mask, mask))
    white = [b for b in blobs if b["colour"] == (240, 240, 240)]
    assert len(white) == 1 and (white[0]["x"], white[0]["y"], white[0]["w"], white[0]["h"]) == (16., 14., 8., 8.)
    assert any(b["colour"] == (180, 0, 0) for b in blobs)


@pytest.mark.parametrize('gap', [.249, .25, .268130064, .35])
@pytest.mark.parametrize('line_width', [2, 4])
def test_current_material_separates_before_the_accepted_correspondence_expires(gap, line_width):
    picture, _inside, body, motion = _scene()
    picture[6:28, 20:20+line_width] = (130, 130, 130)
    eyes = WhatMoves(width=40)
    eyes._backdrop = np.zeros_like(picture, dtype=np.float32)
    body.seen = 0.
    at = gap
    eyes._now = at
    eyes.things = {1: body}
    eyes._point_motion = {1: replace(motion, before=0., at=at)}
    mask = np.zeros((30, 40), np.uint8)
    mask[10:18, 12:20] = 1
    mask[6:28, 20:20+line_width] = 1
    blobs = eyes._blobs(picture, (mask, mask))
    own = [blob for blob in blobs if blob['colour'] == body.colour]
    assert len(own) == 1
    assert (own[0]['x'], own[0]['y'], own[0]['w'], own[0]['h']) == (16., 14., 8., 8.)
    assert eyes._match(blobs, at)[0].get(body.number) == blobs.index(own[0])
    eyes._moved(body, own[0], at)
    assert body.number == 1 and body.born == 0. and body.support.sum() == 64
    assert body.w == body.h == 8.


def test_texture_partition_cannot_outlive_current_material_correspondence():
    picture, inside, body, motion = _scene()
    motion = replace(motion, at=1.350001)
    parts = _parts(picture, inside, body, motion, at=motion.at)
    assert len(parts) == 1 and parts[0] is inside


@pytest.mark.parametrize('gap', [.1, .268130064])
def test_actual_pixels_keep_the_prior_material_track_while_crossing_a_static_line(gap):
    eyes = WhatMoves(width=200)
    texture = np.random.default_rng(517).integers(70, 250, (24, 24, 3), dtype=np.uint8)

    def picture(x=None):
        image = np.full((140, 200, 3), 25, np.uint8)
        if x is not None:
            image[20:125, 98:102] = 130
            image[72:96, x-12:x+12] = texture
        return image

    for index in range(8):
        eyes.see(picture(), index * .1)
    at, number = 1., None
    for x in [60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128]:
        at += gap
        eyes.see(picture(x), at)
        def material_center(thing):
            return thing.motion_position if thing.motion_evidence == 'points' else (thing.x, thing.y)

        material = [thing for thing in [*eyes.things.values(), *eyes.pending.values()]
                    if thing.seen == at and abs(material_center(thing)[0]-x) <= 2
                    and abs(material_center(thing)[1]-84) <= 2
                    and (number is None or thing.number == number)]
        assert len(material) == 1, (x, [(t.number, t.box()) for t in eyes.things.values()])
        if number is None:
            number = material[0].number
        assert material[0].number == number
        if material[0].motion_evidence == 'points':
            assert material[0].motion_position == pytest.approx((x, 84.), abs=.5)


def test_continuing_texture_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _continuing_texture_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "joined_outlines_require_current_owned_texture")
    assert check.predict(None) is True
    assert any(c.test == check.name for c in suite.claims())

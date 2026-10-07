"""What a moving thing leaves behind is the scene, not the thing; and an arrow moves what is hers the way it points.

LIVE 2026-10-07 a game's painter left a trail of paint, the trail and the painter were followed as one thing that
only grew (31 pixels wide at rest, 143 by 126 after a few seconds), its middle barely moved under the keys, and in
thirty seconds no key was found to move anything.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.agency.which_one_answers_to_her import WhichIsHers, _Speeds, gone_the_ways_of_the_keys
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit

FPS = 30


def _frames(draw, seconds):
    for n in range(int(seconds * FPS)):
        at = n / FPS
        picture = np.full((360, 480, 3), (210, 200, 160), np.uint8)
        draw(picture, at)
        yield picture, at


@pytest.mark.parametrize("trail", [(150, 60, 170), (40, 40, 40)])  # a paint and a pen's ink: what is left behind is not the point
def test_a_thing_that_leaves_a_trail_is_followed_at_its_own_size(trail):
    def draw(picture, at):
        if at < 0.8:
            return  # the scene alone, as it is first looked at
        x = 40 + int((at - 1.2) * 120) if at > 1.2 else 40
        if at > 1.2:
            picture[175:185, 40:x] = trail                                      # what it has left behind
        picture[160:200, x:x + 30] = (230, 120, 30)                             # the thing itself
    moves = WhatMoves()
    for picture, at in _frames(draw, 3.5):
        moves.see(picture, at)
    moving = [t for t in moves.moving(faster_than=8.0) if t.h > 10 * moves.scale]
    assert moving, "the thing is not seen to move"
    thing = max(moving, key=lambda t: t.size)
    assert thing.w < 3 * 30 * moves.scale, (thing.w, 30 * moves.scale)  # at its own size, give or take: not the whole trail
    assert thing.vx > 60 * moves.scale  # and going at its own speed: 120 pixels a second


def test_a_thing_at_rest_does_not_fade_into_the_scene():
    def draw(picture, at):
        if at < 0.8:
            return
        x = 40 + int((min(at, 1.6) - 0.8) * 100)
        picture[160:200, x:x + 30] = (230, 120, 30)
    moves = WhatMoves()
    for picture, at in _frames(draw, 4.0):
        moves.see(picture, at)
    assert any(abs(t.w - 30 * moves.scale) < 6 and abs(t.h - 40 * moves.scale) < 6 for t in moves.things.values()), moves.things


def _moves_with(things):
    class _Thing:
        def __init__(self, number):
            self.number, self.kind, self.moved, self.x, self.y = number, 0, True, 50.0, 50.0

    class _Moves:
        pass
    seen = _Moves()
    seen.things = {n: _Thing(n) for n in things}
    return seen


def _pressed(hers, number, key, vx, vy, n):
    speeds = hers._by_thing.setdefault(number, _Speeds())
    for _ in range(3):
        speeds.add(key, vx, vy, press=float(n))


@pytest.mark.parametrize(("presses", "chosen"), [
    # Hers went right, down and left as pressed; another went its own way.
    ({1: [("right", 90, 0), ("down", 0, 80), ("left", -95, 5)], 2: [("right", -40, 30), ("down", 50, -20), ("left", 30, 60)]}, 1),
    # Two keys only: not enough to say.
    ({1: [("right", 90, 0), ("left", -95, 5)]}, None),
    # One key against: not hers by this.
    ({1: [("right", 90, 0), ("down", 0, 80), ("left", -95, 5), ("up", 0, 70)]}, None),
    # Two that both agree: neither.
    ({1: [("right", 90, 0), ("down", 0, 80), ("left", -95, 5)], 2: [("right", 60, 5), ("down", 3, 70), ("left", -80, -4)]}, None),
    # Held against a wall under one key says nothing either way.
    ({1: [("right", 90, 0), ("down", 0, 80), ("left", -95, 5), ("up", 0, -3)]}, 1),
])
def test_what_goes_the_way_each_arrow_points_is_hers(presses, chosen):
    hers = WhichIsHers()
    n = 0
    for number, keys in presses.items():
        for key, vx, vy in keys:
            n += 1
            _pressed(hers, number, key, vx, vy, n)
    assert gone_the_ways_of_the_keys(hers, _moves_with(presses)) == chosen

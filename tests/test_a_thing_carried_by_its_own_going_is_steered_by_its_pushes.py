"""A thing carried on by its own going answers to her keys by how they push it, and is steered by where that takes it.

LIVE 2026-10-09 a lander fell under her arrows. Up slowed its fall and left and right pushed it sideways, but no key
gave it a pace of its own: it went at whatever it was going at, a little faster or slower. She found nothing of hers,
"tried the keys again" every twenty seconds, and it crashed every time. Nothing here is that game: a body under a
steady pull, pushed by three keys, beside a stone that falls under the same pull and answers to nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.what_moves_in_the_picture import Kind, Thing

pytestmark = pytest.mark.unit

PULL = 60.0
PUSHES = {"": (0.0, PULL), "up": (0.0, PULL - 160.0), "left": (-110.0, PULL), "right": (110.0, PULL)}
STEP = 1 / 30


def _thing(number, kind, x, y):
    return Thing(number, x, y, 14.0, 14.0, np.zeros(64), (200, 200, 200), np.zeros((1, 1, 3), dtype=np.uint8), 0.0, 0.0,
                 kind=kind, moved=True)


def _tried(hers, moves, lander, stone, rounds=4):
    """Her keys held in a fixed order whatever is on the screen, each followed by a rest, the bodies moved by them."""
    at = 0.0
    for _round in range(rounds):
        for key in ("up", "", "left", "", "right", ""):
            hers.holding(key, at, trying=True)
            for _ in range(int(0.7 / STEP)):
                at += STEP
                for body, push in ((lander, PUSHES[key]), (stone, PUSHES[""])):
                    body.vx += push[0] * STEP
                    body.vy += push[1] * STEP
                    # Each kept in sight: what is carried off the picture comes back at the other side, going as it was.
                    body.x, body.y = 20 + (body.x + body.vx * STEP - 20) % 360, 20 + (body.y + body.vy * STEP - 20) % 360
                    body.seen = at
                if abs(stone.vy) > 120:
                    stone.vy = -stone.vy      # the stone bounces, as falling things in a game do
                hers.saw(moves, [], at)
    return at


def test_a_body_her_keys_push_is_hers_and_carried_and_a_stone_under_the_same_pull_is_not():
    lander, stone = _thing(1, 0, 100.0, 100.0), _thing(2, 1, 300.0, 100.0)
    moves = SimpleNamespace(shape=(400, 400), things={1: lander, 2: stone},
                            kinds=[Kind(0, np.zeros(64), 196.0, (200, 200, 200)), Kind(1, np.zeros(64), 196.0, (90, 90, 90))])
    hers = WhichIsHers()
    _tried(hers, moves, lander, stone)
    assert hers.number == 1 and hers.carried
    up, rest = hers.push_of("up"), hers.push_of("")
    assert up is not None and rest is not None and up[1] < rest[1] - 100      # up pushes against the pull
    moving = hers.keys_that_move_her(["up", "left", "right"])
    assert moving["left"][0] < -20 and moving["right"][0] > 20 and moving["up"][1] < -20


def test_carried_onto_a_still_thing_below_she_brakes_before_she_gets_there_and_goes_to_it_from_afar():
    from core.agency.playing_as_it_happens import _Choosing
    from core.agency.what_meeting_things_does import MEET, WhatMeetingDoes

    lander, stone = _thing(1, 0, 100.0, 100.0), _thing(2, 1, 300.0, 100.0)
    moves = SimpleNamespace(shape=(400, 400), things={1: lander, 2: stone},
                            kinds=[Kind(0, np.zeros(64), 196.0, (200, 200, 200)), Kind(1, np.zeros(64), 196.0, (90, 90, 90))])
    hers = WhichIsHers()
    _tried(hers, moves, lander, stone)
    assert hers.carried
    pad = Thing(3, 100.0, 300.0, 60.0, 8.0, np.zeros(64), (40, 200, 40), np.zeros((1, 1, 3), dtype=np.uint8), 0.0, 0.0,
                kind=2, moved=False)
    moves.things = {1: lander, 3: pad}
    meeting = WhatMeetingDoes()
    meeting.stance = lambda kind, fixture=False: MEET if kind == 2 else "ignore"
    meeting.known = lambda kind: True
    keys = ["up", "left", "right"]

    # Falling fast, just above it: up, to come down onto it slowly.
    lander.x, lander.y, lander.vx, lander.vy = 100.0, 270.0, 0.0, 90.0
    assert _Choosing(moves, hers, meeting, keys).key("")[0] == "up"
    # Still, far to its right and level with it: left, toward it.
    lander.x, lander.y, lander.vx, lander.vy = 300.0, 290.0, 0.0, 0.0
    assert _Choosing(moves, hers, meeting, keys).key("")[0] == "left"
    # Going left fast, a little to its right and above it: no more left, or she flies past it; she brakes.
    lander.x, lander.y, lander.vx, lander.vy = 160.0, 250.0, -120.0, 0.0
    assert _Choosing(moves, hers, meeting, keys).key("left")[0] == "right"

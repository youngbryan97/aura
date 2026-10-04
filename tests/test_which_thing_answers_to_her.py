"""Her thing is the one whose movement depends on which key she holds, measured only while she tries keys."""
from __future__ import annotations

import pytest

from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


class _Thing:
    def __init__(self, number, kind):
        self.number, self.kind = number, kind
        self.x = self.y = 50.0
        self.vx = self.vy = 0.0
        self.seen = self.born = 0.0
        self.moved = True


class _Moves:
    def __init__(self, things):
        self.things = {t.number: t for t in things}


def _run(trying: bool, ball_follows_keys: bool):
    mine, ball = _Thing(1, 0), _Thing(2, 1)
    moves = _Moves([mine, ball])
    hers = WhichIsHers()
    at = 0.0
    for step in range(120):
        key = ("up", "down", "")[(step // 10) % 3]
        hers.holding(key, at, trying=trying)
        at += 0.03
        mine.vy = {"up": -120.0, "down": 120.0, "": 0.0}[key]
        ball.vx = 150.0 if (step // 25) % 2 else -150.0
        ball.vy = mine.vy * 1.5 if ball_follows_keys else (80.0 if (step // 7) % 2 else -80.0)
        for thing in (mine, ball):
            thing.seen = at
        hers.saw(moves, [], at)
    return hers


def test_the_thing_that_answers_to_her_keys_is_hers():
    hers = _run(trying=True, ball_follows_keys=False)
    assert hers.number == 1
    up, down = hers.way_of("up"), hers.way_of("down")
    assert up[1] < -100 and down[1] > 100


def test_what_her_own_choices_make_move_is_not_taken_for_hers():
    """While she plays she presses up because the ball goes up; that is no evidence about the ball."""
    hers = _run(trying=False, ball_follows_keys=True)
    assert hers.number is None

"""The physics of one world is learned by watching it, and runs forward to where things will be."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from core.perception.how_things_move_here import HowThingsMoveHere

pytestmark = pytest.mark.unit


class _Thing(SimpleNamespace):
    def box(self):
        return (self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2)


class _Moves:
    def __init__(self, things, shape=(150, 240)):
        self.things = {t.number: t for t in things}
        self.shape = shape


class _Nobody:
    number = None

    def thing(self, moves):
        return None


def _ball(**kw):
    return _Thing(**{"number": 1, "kind": 0, "x": 120.0, "y": 80.0, "w": 4.0, "h": 4.0, "vx": 0.0, "vy": 0.0, "moved": True, "colour": (250, 250, 250), "size": 16.0, **kw})


def _watch(physics, ball, steps, dt, step):
    at = 0.0
    for _ in range(steps):
        step(ball, dt)
        at += dt
        physics.saw(_Moves([ball]), _Nobody(), [], at)


def test_walls_are_where_things_turn_back_not_where_the_picture_ends():
    """A scoreboard band at the top: the ball turns back at y=20, and that is where she expects it to."""
    physics = HowThingsMoveHere()
    ball = _ball(vx=60.0, vy=-90.0)

    def step(b, dt):
        b.x = (b.x + b.vx * dt) % 240
        b.y += b.vy * dt
        if b.y < 20 or b.y > 146:
            b.vy = -b.vy
            b.y = min(max(b.y, 20), 146)

    _watch(physics, ball, 600, 1 / 50, step)
    does, where, _kept = physics.edge(0, "top")
    assert does == "bounce" and abs(where - 20) < 4
    test = _ball(x=100.0, y=60.0, vx=0.0, vy=-100.0)
    imagined = physics.imagine(test, 1.0)
    assert min(y for _t, _x, y in imagined.path) > 15


def test_gravity_is_found_when_there_is_some():
    physics = HowThingsMoveHere()
    ball = _ball(vx=30.0, vy=-150.0)

    def step(b, dt):
        b.vy += 300.0 * dt
        b.x += b.vx * dt
        b.y += b.vy * dt
        if b.y > 140:
            b.y, b.vy = 30.0, -150.0

    _watch(physics, ball, 400, 1 / 50, step)
    ax, ay = physics.gravity(0)
    assert ax == 0.0 and 250 < ay < 350


def test_no_gravity_is_invented():
    physics = HowThingsMoveHere()
    ball = _ball(vx=80.0, vy=40.0)

    def step(b, dt):
        b.x = (b.x + b.vx * dt) % 240
        b.y = 20 + (b.y - 20 + b.vy * dt) % 100

    _watch(physics, ball, 300, 1 / 50, step)
    assert physics.gravity(0) == (0.0, 0.0)


def test_how_a_thing_leaves_her_follows_where_it_met_her():
    physics = HowThingsMoveHere()
    kind = physics.kinds[0]
    for along in (-0.8, -0.4, 0.0, 0.4, 0.8, -0.6, 0.6):
        kind.meetings.append((along, along * 0.9, 1.05))
    out = physics.after_meeting(0, 0.8, (-200.0, 0.0), across=True)
    assert out[0] > 0 and math.isclose(math.atan2(out[1], out[0]), 0.72, abs_tol=0.05)
    assert math.isclose(math.hypot(*out), 210.0, rel_tol=0.01)

"""A thing that costs her from a distance has a reach, learned from where she stood at the losses, and is kept clear of.

A guard sees ahead of where it goes; an enemy fires from afar; a mine goes off
as she comes near. Keeping clear of what touches her is not enough with these.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.how_far_a_thing_reaches import HowFarThingsReach

pytestmark = pytest.mark.unit


def _thing(number: int, kind: int, x: float, y: float, vx: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(number=number, kind=kind, x=x, y=y, vx=vx, vy=0.0, w=10.0, h=10.0)


def test_a_kind_near_at_every_loss_reaches_ahead_and_one_near_at_some_does_not():
    reach = HowFarThingsReach()
    me = _thing(1, 0, 100.0, 100.0)
    for at, coin_near in ((60.0, True), (55.0, False), (62.0, False), (58.0, False)):
        things = [_thing(2, 5, at, 100.0, vx=40.0)]
        if coin_near:
            things.append(_thing(3, 7, 110.0, 100.0))
        reach.lost_untouched(me, things)
    far, ahead = reach.reach(5)
    assert 4.0 < far < 6.0 and ahead
    assert reach.reach(7) is None
    guard = _thing(2, 5, 0.0, 0.0, vx=40.0)
    assert reach.within(guard, 45.0, 0.0, (0.0, 0.0))
    assert not reach.within(guard, -45.0, 0.0, (0.0, 0.0))
    assert not reach.within(guard, 90.0, 0.0, (0.0, 0.0))


def test_nothing_near_is_suspected_of_nothing():
    reach = HowFarThingsReach()
    reach.lost_untouched(_thing(1, 0, 0.0, 0.0), [_thing(2, 5, 500.0, 500.0)])
    reach.lost_untouched(_thing(1, 0, 0.0, 0.0), [_thing(2, 5, 500.0, 500.0)])
    assert reach.reach(5) is None

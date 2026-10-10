"""Where she is in a world bigger than the view: the view going by, added up, is how far she has come, and which way.
Nothing here is a level: two layers going by at speeds of their own, and a hero the keys move."""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.what_moves_in_the_picture import WhatMoves
from core.perception.where_in_a_world import WhereInTheWorld

pytestmark = pytest.mark.unit


def _layer(seed: int, tall: int, wide: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(40, 220, size=(tall // 8 + 1, wide // 8 + 1, 3)).astype(np.uint8)
    return np.repeat(np.repeat(blocks, 8, axis=0), 8, axis=1)[:tall, :wide]


def _going_by(seconds: float, far_speed: float, near_speed: float, fps: int = 30):
    """Pictures of a far and a near layer going left at their speeds, in picture pixels a second."""
    far, near = _layer(1, 120, 3200), _layer(2, 120, 3200)
    for n in range(int(seconds * fps)):
        at = n / fps
        picture = np.zeros((240, 320, 3), np.uint8)
        picture[0:120] = far[:, int(at * far_speed):int(at * far_speed) + 320]
        picture[120:240] = near[:, int(at * near_speed):int(at * near_speed) + 320]
        yield picture, at


def _travelled(seconds, far_speed, near_speed):
    moves, world = WhatMoves(), WhereInTheWorld()
    for picture, at in _going_by(seconds, far_speed, near_speed):
        moves.see(picture, at)
        if moves.view_over and moves.view_moved_at == at:
            world.saw(moves.view_moved, moves.view_over, *moves.shape)
    return world


def test_the_view_going_by_adds_up_to_how_far_she_has_come_and_which_way():
    world = _travelled(14.0, 25.0, 70.0)
    # The ground (the lowest layer) goes 70 picture pixels a second: 980 in 14 s, about three views of 320.
    assert world.the_way_on == "right"
    assert 2.5 <= world.views_along() <= 3.6, world.views_along()
    assert world.new_views >= 2
    assert world.said().startswith("I've come about 3 screens right")


def test_a_view_that_stands_still_takes_her_nowhere():
    world = _travelled(3.0, 0.0, 0.0)
    assert world.views_along() == 0.0 and world.the_way_on == "" and world.said() == ""

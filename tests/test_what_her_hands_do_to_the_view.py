"""What each act does to the view is measured from the pictures, not assumed.

Which key walks, which way the mouse turns the camera and how far a point of
travel turns it differ in every game. A synthetic world stands in for one
here: a panorama seen through a window, where the mouse turns the camera by
a known amount and one key walks forward. She is told none of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.perception.how_the_view_moves import (
    WhatMyHandsDoToTheView,
    grey,
    how_it_grew,
    how_it_slid,
)

#: The synthetic camera turns this many pixels of panorama per point of mouse travel.
TURN = 3
VIEW = (240, 640)


def _panorama(seed: int = 0) -> np.ndarray:
    roll = np.random.default_rng(seed)
    coarse = roll.random((30, 300))
    return np.kron(coarse, np.ones((12, 12)))  # 360 x 3600, blocky texture


class World:
    """A camera over a panorama: yaw in pixels, and how far she has walked."""

    def __init__(self) -> None:
        self.panorama = _panorama()
        self.yaw = 1200
        self.zoom = 1.0

    def frame(self) -> np.ndarray:
        high, wide = VIEW
        crop_h, crop_w = int(high / self.zoom), int(wide / self.zoom)
        top = (self.panorama.shape[0] - crop_h) // 2
        left = self.yaw + (wide - crop_w) // 2
        crop = self.panorama[top : top + crop_h, left : left + crop_w]
        rows = np.linspace(0, crop_h - 1, high).round().astype(int)
        cols = np.linspace(0, crop_w - 1, wide).round().astype(int)
        return crop[np.ix_(rows, cols)] * 255.0

    def act(self, name: str, mouse: tuple[int, int] = (0, 0)) -> None:
        self.yaw += TURN * mouse[0]
        if name == "w":
            self.zoom *= 1.1
        elif name == "s":
            self.zoom /= 1.1


def test_a_slide_is_measured_where_it_happened():
    world = World()
    before = grey(world.frame())
    world.yaw += 16  # four pixels of the small frame at a quarter scale
    across, down, sure = how_it_slid(before, grey(world.frame()))
    assert across == pytest.approx(-4, abs=1) and down == pytest.approx(0, abs=1)
    assert sure > 5.0


def test_walking_forward_grows_the_middle_and_backing_off_shrinks_it():
    world = World()
    before = grey(world.frame())
    world.act("w")
    assert how_it_grew(before, grey(world.frame())) == pytest.approx(1.1, abs=0.05)
    world.act("s")
    world.act("s")
    assert how_it_grew(grey(World().frame()), grey(world.frame())) < 1.0


def test_she_learns_which_key_walks_and_how_far_the_mouse_turns():
    world = World()
    hands = WhatMyHandsDoToTheView()
    for act, mouse in [("w", (0, 0)), ("a", (0, 0)), ("mouse", (10, 0)), ("s", (0, 0)),
                       ("mouse", (-6, 0)), ("w", (0, 0)), ("mouse", (8, 0))]:
        before = world.frame()
        world.act(act, mouse)
        hands.watched(act, before, world.frame(), mouse=mouse)
        world.zoom = 1.0  # stand back where she was, so each look starts alike
    assert hands.walks_forward() == "w"
    gain, _ = hands.mouse_gain()
    # A camera turning toward the mouse slides the picture the other way.
    assert gain == pytest.approx(-TURN / 4, rel=0.15)


def test_and_can_turn_to_face_what_she_sees():
    world = World()
    hands = WhatMyHandsDoToTheView()
    for travel in (10, -8, 6):
        before = world.frame()
        world.act("mouse", (travel, 0))
        hands.watched("mouse", before, world.frame(), mouse=(travel, 0))
    # Something 20 small-frame pixels right of the middle slides to the middle
    # when the picture moves 20 pixels left.
    assert hands.turn_for(-20) == pytest.approx(20 * 4 / TURN, abs=2)


def test_an_act_she_has_not_watched_says_nothing():
    hands = WhatMyHandsDoToTheView()
    assert hands.what_it_does("w") is None
    assert hands.walks_forward() == "" and hands.turn_for(10) == 0


def _a_blocky_wall(grow: float, seed: int = 7) -> np.ndarray:
    """A wall of large grey blocks, magnified about the middle, the way the live room draws one."""
    roll = np.random.default_rng(seed)
    shades = roll.random((12, 90))
    high, wide = 600, 960
    ys, xs = np.mgrid[0:high, 0:wide].astype(float)
    column = np.floor((xs - wide / 2) / (wide / 90 * grow)).astype(int) + 45
    row = np.floor((ys - high / 2) / (18 * grow)).astype(int) + 6
    inside = (row >= 0) & (row < 12) & (column >= 0) & (column < 90)
    picture = np.zeros((high, wide))
    picture[inside] = 0.2 + 0.6 * shades[row[inside], column[inside]]
    return picture * 255.0


def test_walking_reads_as_growth_on_a_wall_of_blocks():
    """A wall of large flat blocks, which leaves little texture to register on."""
    from core.perception.how_the_view_moves import how_it_moved

    near, far = grey(_a_blocky_wall(1.26)), grey(_a_blocky_wall(1.0))
    assert how_it_moved(far, near)[2] > 1.0
    assert how_it_moved(near, far)[2] < 1.0

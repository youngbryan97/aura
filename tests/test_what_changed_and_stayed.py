"""A counter nobody can read still shows where it changed: a still place that changed and stayed changed."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.what_changed_and_stayed import WhatChangedAndStayed

pytestmark = pytest.mark.unit


def _picture(digit_on: bool, ball_x: int) -> np.ndarray:
    picture = np.zeros((100, 200, 3), dtype=np.uint8)
    if digit_on:
        picture[5:15, 140:146] = 255
    picture[50:54, ball_x : ball_x + 4] = 255
    return picture


class _Ball(SimpleNamespace):
    def box(self):
        return (self.x - 2, self.y - 2, self.x + 2, self.y + 2)


def test_a_digit_that_changes_is_seen_where_it_changed_and_the_ball_is_not():
    stayed = WhatChangedAndStayed()
    found = []
    for step in range(20):
        ball_x = 20 + step * 8
        ball = _Ball(number=1, x=ball_x + 2, y=52, moved=True)
        found += stayed.see(_picture(step >= 8, ball_x), {1: ball}, step * 0.26)
    assert found, "the changed digit was not seen"
    assert all(abs(x - 143 / 200) < 0.05 and y < 0.2 for _when, x, y in found)


def test_a_whole_new_screen_is_not_a_counter():
    stayed = WhatChangedAndStayed()
    found = []
    for step in range(12):
        picture = np.full((100, 200, 3), 255 if step >= 5 else 0, dtype=np.uint8)
        found += stayed.see(picture, {}, step * 0.26)
    assert found == []

"""A window is mostly not the task.

Reading all of it spends the whole picture on tabs, an address bar, an
advertising rail and a footer, and what is left for the part she is acting on
is a few pixels a character. That is how a board drawn on a canvas comes back
as scattered noise.

LIVE 2026-08-29: sixty-seven acts that moved something, and the best
hypothesis got five of them right, on a board that was drawn perfectly well
the whole time. Reading only the part she is using spends every pixel of the
picture on it — on a 1600x900 window and a band covering a third across and
two thirds down, four times as many pixels a character.
"""

from __future__ import annotations

import pytest

from core.skills.screen_pursuit import _the_part_of


HER_WINDOW = (0, 100, 1600, 900)
BOARD = (0.33, 0.13, 0.67, 0.82)


# ── which pixels a band names ────────────────────────────────────────────

def test_a_band_names_a_rectangle_of_the_window():
    assert _the_part_of(HER_WINDOW, BOARD) == (528, 217, 544, 621)


def test_the_part_sits_inside_the_window_it_came_from():
    x, y, wide, tall = _the_part_of(HER_WINDOW, BOARD)
    assert x >= HER_WINDOW[0] and y >= HER_WINDOW[1]
    assert x + wide <= HER_WINDOW[0] + HER_WINDOW[2]
    assert y + tall <= HER_WINDOW[1] + HER_WINDOW[3]


def test_it_is_a_smaller_picture_of_a_bigger_thing():
    """Which is the whole point: more pixels for each character in it."""
    _x, _y, wide, tall = _the_part_of(HER_WINDOW, BOARD)
    assert wide * tall < HER_WINDOW[2] * HER_WINDOW[3] / 3


def test_the_whole_window_is_still_the_whole_window():
    assert _the_part_of(HER_WINDOW, (0.0, 0.0, 1.0, 1.0)) == HER_WINDOW


@pytest.mark.parametrize("band", [(0.5, 0.5, 0.5, 0.5), (0.9, 0.9, 0.9, 0.9)])
def test_a_band_of_no_width_still_names_a_readable_rectangle(band):
    """Nothing is ever asked to photograph a region of zero size."""
    _x, _y, wide, tall = _the_part_of(HER_WINDOW, band)
    assert wide >= 1 and tall >= 1


def test_a_window_at_an_offset_carries_it_through():
    on_the_right = (800, 0, 800, 600)
    x, y, wide, tall = _the_part_of(on_the_right, (0.0, 0.5, 0.5, 1.0))
    assert (x, y, wide, tall) == (800, 300, 400, 300)

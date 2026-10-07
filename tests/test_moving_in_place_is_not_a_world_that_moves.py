"""A world moves on its own when something in it goes somewhere, not when something sways in place.

LIVE 2026-10-06, a checkers game: the board waited for her move while two
figures beside it swayed, and she played it with the arrow keys for five
minutes as if it were an action game.
"""
from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from core.agency.playing_as_it_happens import the_world_moves_on_its_own

pytestmark = pytest.mark.unit

FPS = 20


def _frames(draw):
    """A look that returns the scene ``draw`` paints at each moment, on the pictures' own clock."""
    state = {"n": 0}

    async def look():
        at = state["n"] / FPS
        state["n"] += 1
        if at > 10:
            return None
        picture = np.full((360, 480, 3), (40, 30, 60), np.uint8)
        picture[300:360, :] = (90, 70, 50)  # a floor that never moves
        draw(picture, at)
        return picture, at

    return look


def _box(picture, x, y, w, h, colour):
    x, y = int(round(x)), int(round(y))
    picture[max(0, y):y + h, max(0, x):x + w] = colour


@pytest.mark.parametrize(("name", "draw", "moves"), [
    ("a ball crossing a court", lambda p, t: _box(p, 20 + 160 * t, 150, 12, 12, (240, 240, 240)), True),
    ("a block falling slowly", lambda p, t: _box(p, 200, 10 + 45 * t, 60, 60, (200, 60, 60)), True),
    ("a figure bobbing beside a board", lambda p, t: _box(p, 60, 120 + 4 * math.sin(2 * math.pi * 2 * t), 60, 80, (70, 140, 230)), False),
    ("a badge pulsing in place", lambda p, t: _box(p, 300 - 3 * math.sin(6 * t), 100 - 3 * math.sin(6 * t),
                                                   50 + int(6 * math.sin(6 * t)), 50 + int(6 * math.sin(6 * t)), (250, 200, 40)), False),
    ("a still screen", lambda p, t: _box(p, 100, 100, 80, 40, (30, 200, 90)), False),
])
def test_only_something_that_goes_somewhere_is_a_world_that_moves(name, draw, moves):
    assert asyncio.run(the_world_moves_on_its_own(_frames(draw))) is moves, name


def test_a_board_waiting_beside_a_swaying_figure_waits():
    def board(picture, t):
        for row in range(8):
            for col in range(8):
                if (row + col) % 2:
                    _box(picture, 160 + 30 * col, 40 + 30 * row, 30, 30, (230, 200, 200))
        _box(picture, 60, 120 + 5 * math.sin(2 * math.pi * 1.5 * t), 50, 70, (70, 140, 230))  # the figure beside it
    assert asyncio.run(the_world_moves_on_its_own(_frames(board))) is False

"""Thinking time grows as the room to recover shrinks.

Flat at the cost of a look (about 0.3 s), her own search reached 2048 in one
game of four; scaled by how little room was left, four of four (2026-09-23).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.skills.screen_pursuit_decision_branches import _how_long_to_think


def _board(free: int, places: int = 16):
    return SimpleNamespace(empty=lambda: free, places=lambda: places)


def test_an_open_board_gets_the_cost_of_a_look():
    assert _how_long_to_think(_board(16), least=0.3, most=2.0) == pytest.approx(0.3)


def test_a_full_board_gets_the_ceiling():
    assert _how_long_to_think(_board(0), least=0.3, most=2.0) == pytest.approx(2.0)


def test_it_rises_as_the_board_fills():
    times = [_how_long_to_think(_board(free), least=0.3, most=2.0) for free in (12, 8, 4, 1)]
    assert times == sorted(times)
    assert 0.3 < times[0] < times[-1] < 2.0


def test_a_ceiling_below_the_floor_is_the_floor():
    """Near the end of a run the ceiling shrinks with the time left."""
    assert _how_long_to_think(_board(0), least=0.3, most=0.1) == pytest.approx(0.3)


def test_something_with_no_room_to_read_gets_the_floor():
    assert _how_long_to_think(object(), least=0.3, most=2.0) == pytest.approx(0.3)


def test_the_loop_uses_it():
    from screen_pursuit_support import pursuit_loop_source

    assert "thinking_for = _how_long_to_think(" in pursuit_loop_source()

"""A board of pieces that would not take a move has not answered her, so the controls around it are the way on.

LIVE 23 November 2026: En Croissant showed a game already over, at its start position. She sat down at its board
sixteen times in four and a half minutes, found it would not take a move each time, and never pressed "New Game": the
way on is pressed where nothing she did answered, and a key of hers had done something there earlier.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from core.skills.screen_pursuit import nothing_has_answered


def test_a_board_that_would_not_take_a_move_a_moment_ago_has_not_answered() -> None:
    did = SimpleNamespace(did_something={"press up": True})
    assert nothing_has_answered({"board_left_until": time.monotonic() + 5}, did)


def test_otherwise_what_she_did_decides() -> None:
    assert not nothing_has_answered({}, SimpleNamespace(did_something={"press up": True}))
    assert nothing_has_answered({}, SimpleNamespace(did_something={}))
    assert not nothing_has_answered({"board_left_until": time.monotonic() - 1},
                                    SimpleNamespace(did_something={"press up": True}))


def test_it_has_not_answered_until_it_takes_a_move() -> None:
    # LIVE 23 November 2026: the eight seconds a board is left for ran out, her keys counted as answers again, and the
    # way on was not tried a second time; "New Game" stood next after a label she had pressed.
    did = SimpleNamespace(did_something={"press up": True})
    assert nothing_has_answered({"board_would_not_take": True, "board_left_until": time.monotonic() - 1}, did)

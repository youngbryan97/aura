"""A board of pieces that would not take a move has not answered her, so the controls around it are the way on.

LIVE 23 November 2026: En Croissant showed a game already over, at its start position. She sat down at its board
sixteen times in four and a half minutes, found it would not take a move each time, and never pressed "New Game": the
way on is pressed where nothing she did answered, and a key of hers had done something there earlier.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from core.skills.screen_pursuit import a_board_waited_on, nothing_has_answered


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


def test_a_board_that_would_not_take_a_move_is_waited_on_not_tried_with_keys_after_its_wait_runs_out() -> None:
    # LIVE 26 November 2026: her model took eleven seconds over which control to press, the eight-second wait ran out,
    # and she pressed the up arrow at a finished game.
    assert a_board_waited_on({"board_would_not_take": True, "board_left_until": time.monotonic() - 1})
    assert a_board_waited_on({"board_left_until": time.monotonic() + 5})
    assert not a_board_waited_on({"board_left_until": time.monotonic() - 1})


def test_a_board_that_would_not_take_a_move_is_not_sat_at_again_on_the_same_screen(monkeypatch) -> None:
    """The same words on the screen, with nothing pressed between, are the same screen: a board that would not take a
    move on it will not now. LIVE 26 November 2026 she sat down at a finished game's board four times in two minutes,
    twenty seconds a time, before pressing "New Game"."""
    import asyncio

    import core.skills.screen_pursuit_board_of_pieces as board_of_pieces
    import core.skills.screen_pursuit_on_a_page as on_a_page

    monkeypatch.setattr(on_a_page, "on_her_page", lambda: None)
    looked_for: list[str] = []

    def window_of(app, **_kwargs):
        looked_for.append(app)

    monkeypatch.setattr("core.capabilities.window_server.window_of", window_of)
    seen = {"window_number": 7, "layout": [{"text": "New Game"}, {"text": "Black wins"}]}
    keep = {"board_would_not_take": True, "board_refused_with_words": ("New Game", "Black wins")}
    assert asyncio.run(board_of_pieces.played_a_board_of_pieces("En Croissant", time.monotonic() + 60, keep, seen)) \
        is None
    assert looked_for == [], "the same screen is not sat at again"
    changed = {"window_number": 7, "layout": [{"text": "Start game"}, {"text": "Stockfish"}]}
    asyncio.run(board_of_pieces.played_a_board_of_pieces("En Croissant", time.monotonic() + 60, keep, changed))
    assert looked_for == ["En Croissant"], "a screen that changed is looked at again"

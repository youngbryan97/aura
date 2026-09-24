"""An act that did nothing from the very view she is still looking at is not made again.

LIVE 2026-09-23: the "Game over!" overlay covered five tiles, she read them as
empty, her rule said right would slide into them, and she pressed right into
a finished game for seven minutes. The one act she kept choosing meant
"nothing answers" never saw the others fail.
"""

from __future__ import annotations

from types import SimpleNamespace

from core.skills.screen_pursuit_decision_branches import (
    _it_did_nothing_from_here,
    _leaving_out_what_just_did_nothing,
)


def _view(text: str):
    return SimpleNamespace(as_text=lambda: text)


def _options(*names: str):
    return [SimpleNamespace(name=name) for name in names]


def test_an_act_that_did_nothing_here_is_left_out_while_the_view_stays():
    pending: dict = {}
    over = _view("8 2 4 2 / 16 4 16 4 / 8 32 _ 8 / 4 8 4 2")
    _it_did_nothing_from_here(pending, over, "right", False)
    left, ended = _leaving_out_what_just_did_nothing(_options("up", "down", "left", "right"), pending, over)
    assert [option.name for option in left] == ["up", "down", "left"]
    assert not ended


def test_when_every_act_has_done_nothing_from_here_the_ending_test_decides():
    """All of them are offered again, so "nothing answers" can count them."""
    pending: dict = {}
    over = _view("a finished board")
    for key in ("right", "up", "down", "left"):
        _it_did_nothing_from_here(pending, over, key, False)
    available, ended = _leaving_out_what_just_did_nothing(
        _options("up", "down", "left", "right"), pending, over
    )
    assert [option.name for option in available] == ["up", "down", "left", "right"]
    assert not ended


def test_a_changed_view_offers_everything_again():
    pending: dict = {}
    _it_did_nothing_from_here(pending, _view("before"), "right", False)
    left, ended = _leaving_out_what_just_did_nothing(
        _options("up", "right"), pending, _view("after something moved")
    )
    assert [option.name for option in left] == ["up", "right"]
    assert not ended


def test_an_act_that_changed_something_clears_the_record():
    pending: dict = {}
    board = _view("a board")
    _it_did_nothing_from_here(pending, board, "right", False)
    _it_did_nothing_from_here(pending, board, "up", True)
    left, _ended = _leaving_out_what_just_did_nothing(_options("up", "right"), pending, board)
    assert [option.name for option in left] == ["up", "right"]


def test_the_loop_records_and_uses_it():
    from screen_pursuit_support import pursuit_loop_source

    source = pursuit_loop_source()
    assert "_it_did_nothing_from_here(" in source
    assert "_leaving_out_what_just_did_nothing(available, pending, laid_out)" in source


def test_an_ending_is_offered_its_way_out_whatever_was_chosen_before():
    """Seeing it through, chosen while a game went badly, closed the door for good.

    LIVE 2026-09-23: eighteen minutes of pressing right into "Game over!".
    """
    from screen_pursuit_support import pursuit_loop_source

    source = pursuit_loop_source()
    assert 'if ended or (stuck(history) and not seen_through["value"]):' in source
    assert 'if (stuck(history) or ended) and not seen_through["value"]:' not in source

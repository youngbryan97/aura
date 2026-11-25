"""With no clock to share, she takes as long as the other side does, and never less than she would alone
(core/agency/playing_pieces_by_kind.py)."""
from __future__ import annotations

import asyncio

import pytest

import core.agency.playing_pieces_by_kind as playing
from core.agency.how_the_pieces_move import HowThePiecesMove
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

pytestmark = pytest.mark.unit


def _game(answers):
    learned = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    game = playing._Game(learned=learned, believed=CHESS.the_start())
    game.answers_s = list(answers)
    return game


def test_before_they_answer_she_takes_what_she_was_given() -> None:
    assert playing._their_pace(_game([]), 4.0) == 4.0


def test_once_they_have_answered_she_takes_the_middle_of_their_times() -> None:
    assert playing._their_pace(_game([5.0, 9.0, 4.0]), 4.0) == 5.0
    assert playing._their_pace(_game([5.0, 7.0]), 4.0) == 6.0


def test_a_game_with_no_clock_shown_gives_her_their_pace() -> None:
    async def no_clock_shown():
        return None

    rehearsing = playing._Rehearsing(None)
    share, on_clock = asyncio.run(playing._time_for_a_move(no_clock_shown, rehearsing, 3, 4.0, _game([6.0, 6.5, 5.5])))
    assert (share, on_clock) == (6.0, False)
    share, on_clock = asyncio.run(playing._time_for_a_move(None, rehearsing, 3, 4.0, _game([6.0])))
    assert (share, on_clock) == (6.0, False)


def test_a_side_that_answers_fast_does_not_hurry_her() -> None:
    """Without a clock a slow turn costs her nothing: their quick answers leave her her own time to think."""
    assert playing._their_pace(_game([1.2, 1.6, 1.4]), 4.0) == 4.0
    share, on_clock = asyncio.run(playing._time_for_a_move(None, playing._Rehearsing(None), 3, 4.0, _game([1.5])))
    assert (share, on_clock) == (4.0, False)

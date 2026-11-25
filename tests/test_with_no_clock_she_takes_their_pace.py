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


def test_with_no_clock_a_look_whose_answer_will_not_settle_may_go_on_as_long_as_she_would_wait() -> None:
    assert playing._the_most_a_look_may_take(4.0, False, None) == max(3 * 4.0, playing.NO_ANSWER_AFTER_S)
    # On her clock, held to what the clock allows.
    assert playing._the_most_a_look_may_take(4.0, True, 100.0) == min(playing.SPENDS_AT_MOST * 4.0,
                                                                       playing.LEFT_AT_MOST * 100.0)


def test_a_slow_answer_does_not_unstart_a_confirmed_game() -> None:
    from core.agency.how_the_pieces_move import HowThePiecesMove
    from tests.fixtures.a_game_of_pieces_to_learn import CHESS

    game = playing._Game(learned=HowThePiecesMove.from_the_start(8, 8, dict(CHESS.start)),
                         believed=CHESS.the_start())
    now = game.waiting_since + playing.NO_ANSWER_AFTER_S + 1
    assert game.start_still_unconfirmed(now)
    game.theirs = 1
    assert not game.start_still_unconfirmed(now)

"""A drawn game is worth to her what her games against the opponent in front of her say: the judgement at which her
chance there is even (core/cognition/how_far_her_judgement_holds.py ``even_at``), carried into the weights she plays by
(core/agency/playing_pieces_by_kind.py ``_Rehearsing.weights_to_play``) and scored by her search for every drawn line
from her side (rust_extensions/aura_board_search ``drawn``).

LIVE 26 November 2026 her ten games against Stockfish and the 350 judgements in them all came to a loss. Against such an
opponent a draw is worth more than anything she judged, and her search held every draw worth nothing.
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from core.agency.pieces_by_their_kind import FIRST, after, moves
from core.agency.playing_pieces_by_kind import _Rehearsing
from core.agency.searching_by_kind import first_weights, search
from core.agency.what_a_kind_is_worth import Rehearsal
from core.cognition.how_far_her_judgement_holds import HowFarItHolds
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

pytestmark = pytest.mark.unit

SCALE = 228.0


def test_with_no_games_against_an_opponent_a_draw_is_worth_nought() -> None:
    assert HowFarItHolds().even_at(SCALE) == 0.0


def test_after_games_that_all_came_to_a_loss_a_draw_is_worth_more_than_she_judged() -> None:
    held = HowFarItHolds()
    for _game in range(10):
        held.came_to([15.0, -5.0, 30.0], 0.0)
    middle = (15.0 - 5.0 + 30.0) / 3
    assert held.even_at(SCALE) == pytest.approx(middle + SCALE * math.log(11.0))


def test_where_her_games_came_to_different_ends_the_curve_says_where_her_chance_is_even() -> None:
    held = HowFarItHolds()
    for n in range(12):
        held.came_to([float(100 * (n % 4) - 150)], 1.0 if n % 4 >= 2 else 0.0)
    assert held.slope > 0
    assert held.even_at(SCALE) == pytest.approx(-held.level / held.slope)


def test_the_weights_she_plays_by_carry_it_and_her_rehearsal_does_not() -> None:
    rehearsing = _Rehearsing(rehearsal=Rehearsal(weights=first_weights(CHESS), earned=["worth"], royal=["K"]))
    rehearsing.draw_worth = 400.0
    assert rehearsing.weights_to_play(CHESS).habits["draw_worth"] == 400.0
    assert "draw_worth" not in (rehearsing.weights_for(CHESS).habits or {})


def _knights_back_and_forth():
    """The knights out and back twice: the move that brings her knight out again repeats a position. Once round is
    not enough, for a piece that has moved and come back is not the piece that never moved."""
    position, earlier = CHESS.the_start(), []
    shuffle = (((0, 6), (2, 5)), ((7, 6), (5, 5)), ((2, 5), (0, 6)), ((5, 5), (7, 6)))
    for start, end in shuffle * 2:
        earlier.append(position)
        position = after(CHESS, position, next(m for m in moves(CHESS, position) if (m.start, m.end) == (start, end)))
    return position, earlier


def test_her_search_takes_a_draw_where_one_is_worth_much_to_her_and_not_where_it_is_worth_nought() -> None:
    position, earlier = _knights_back_and_forth()
    out_again = ((0, 6), (2, 5))
    plain = search(CHESS, position, seconds=10, weights=first_weights(CHESS), earlier=earlier, nodes=30_000,
                   threads=1, wary=FIRST)
    assert (plain.move.start, plain.move.end) != out_again
    worth = replace(first_weights(CHESS), habits={"draw_worth": 900.0})
    drawn = search(CHESS, position, seconds=10, weights=worth, earlier=earlier, nodes=30_000, threads=1, wary=FIRST)
    assert (drawn.move.start, drawn.move.end) == out_again

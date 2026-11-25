"""How a game ends drawn is learned from games the window says were drawn (core/agency/how_the_pieces_move.py
``the_window_said``): by a position standing again, or by moves with nothing that cannot be undone.

Measured 25 November 2026 against a stronger program at a fixed depth, offline: with no draw in her rules she drew by
repetition two games she led by 19 and 29 points of material, and with the game's own rules she won six of six.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import after, moves
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

pytestmark = pytest.mark.unit


def _learned() -> HowThePiecesMove:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    board = from_fen("4k3/8/8/8/8/8/4P3/R3K3 w - -")
    for _ in range(4):
        rules.offered(board, (1, 4), {(2, 4), (3, 4)}, family=CHESS)
        rules.offered(board, (0, 0), {(r, 0) for r in range(1, 8)} | {(0, 1), (0, 2), (0, 3)}, family=CHESS)
    return rules


def test_a_position_standing_again_when_the_game_was_drawn_is_how_many_times_draw() -> None:
    rules = _learned()
    there = from_fen("4k3/8/8/8/8/8/8/R3K3 w - -")
    back = from_fen("4k3/8/8/8/8/8/8/R3K3 w - -")
    rules.the_window_said("drawn", there, None, [back, from_fen("3k4/8/8/8/8/8/8/R3K3 b - -"), back])
    assert rules.repeats_to_draw == 3 and rules.family().repeats_to_draw == 3


def test_moves_with_nothing_that_cannot_be_undone_is_how_many_draw() -> None:
    rules = _learned()
    final = replace(from_fen("4k3/8/8/8/8/8/8/R3K3 w - -"), count=100)
    rules.the_window_said("drawn", final, None, [])
    assert rules.count_to_draw == 100 and rules.family().count_to_draw == 100
    rules.the_window_said("drawn", replace(final, count=120), None, [])
    assert rules.count_to_draw == 100, "the fewest seen to draw is what draws"


def test_a_side_with_no_move_teaches_nothing_of_draws() -> None:
    rules = _learned()
    alone = from_fen("8/8/8/8/3k4/8/8/4K3 b - -")
    for _ in range(4):  # how the royal kind steps, one square any way
        rules.offered(alone, (3, 3), {(3 + dr, 3 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc},
                      family=CHESS)
    rules.kept_off_attacked = {"K": 6}  # kept off squares the other side attacks: the kind the game keeps safe
    no_move = replace(from_fen("7k/5Q2/6K1/8/8/8/8/8 b - -"), count=100)
    assert not moves(CHESS, no_move)
    rules.the_window_said("drawn", no_move, None, [])
    assert rules.count_to_draw == 0 and rules.repeats_to_draw == 0


def test_a_kind_that_only_goes_forward_cannot_go_back_and_its_moves_start_the_count_again() -> None:
    rules = _learned()
    family = rules.family()
    assert family.kinds["P"].cannot_go_back and not family.kinds["R"].cannot_go_back
    board = replace(from_fen("4k3/8/8/8/8/8/4P3/R3K3 w - -"), count=7)
    pawn = next(m for m in moves(family, board) if m.start == (1, 4))
    assert after(family, board, pawn).count == 0


def test_what_draws_is_kept_with_her_record() -> None:
    rules = _learned()
    rules.count_to_draw, rules.repeats_to_draw = 100, 3
    again = HowThePiecesMove.from_memory(rules.as_memory())
    assert (again.count_to_draw, again.repeats_to_draw) == (100, 3)


def test_a_count_a_game_went_on_past_is_not_what_draws() -> None:
    rules = _learned()
    went_on = [replace(from_fen("4k3/8/8/8/8/8/8/R3K3 w - -"), count=n) for n in range(60)]
    rules.the_window_said("lost", from_fen("4k3/8/8/8/8/8/8/R3K3 b - -"), None, went_on)
    # A board she lost the thread of starts its count again: a draw after it shows fewer moves than drew it.
    rules.the_window_said("drawn", replace(from_fen("4k3/8/8/8/8/8/8/R3K3 w - -"), count=12), None, [])
    assert rules.count_to_draw == 0
    rules.the_window_said("drawn", replace(from_fen("4k3/8/8/8/8/8/8/R3K3 w - -"), count=100), None, [])
    assert rules.count_to_draw == 100


def test_a_draw_read_that_a_game_then_goes_on_past_is_let_go() -> None:
    rules = _learned()
    rules.the_window_said("drawn", replace(from_fen("4k3/8/8/8/8/8/8/R3K3 w - -"), count=30), None, [])
    assert rules.count_to_draw == 30
    went_on = [replace(from_fen("4k3/8/8/8/8/8/8/R3K3 w - -"), count=n) for n in range(45)]
    rules.the_window_said("won", from_fen("4k3/8/8/8/8/8/8/R3K3 b - -"), None, went_on)
    assert rules.count_to_draw == 0, "a game went on 44 moves past a count of 30"


def test_a_position_that_stood_twice_with_the_game_going_on_does_not_draw_at_two() -> None:
    rules = _learned()
    back = from_fen("4k3/8/8/8/8/8/8/R3K3 w - -")
    rules.the_window_said("lost", from_fen("3k4/8/8/8/8/8/8/R3K3 b - -"), None,
                          [back, from_fen("3k4/8/8/8/8/8/8/R3K3 b - -"), back, from_fen("4k3/8/8/8/8/8/8/R2K4 b - -")])
    rules.the_window_said("drawn", back, None, [back])
    assert rules.repeats_to_draw == 0

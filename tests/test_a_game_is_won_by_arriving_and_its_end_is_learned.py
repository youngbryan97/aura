"""A kind whose arrival on its far row wins, and how a game ended learned from what its window said.

LIVE 11 October 2026 a game on a web page was won by getting a piece set apart "to king row". Her play on a page held
that as words read by pattern (core/agency/playing_a_board.py); the vocabulary had no such ending, and her learner had
a place to hold that a side with no move loses but nothing that ever learned it.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, SECOND, Move, Position, outcome
from tests.test_two_pieces_of_a_side_change_places import SET_APART

ARRIVING = replace(SET_APART, kinds={**SET_APART.kinds, "S": replace(SET_APART.kinds["S"], wins_arriving=True)})


def _board(cells: dict, turn: int) -> Position:
    return Position(cells=cells, turn=turn, unmoved=frozenset())


def test_a_piece_of_a_kind_that_wins_arriving_on_its_far_row_has_won() -> None:
    assert outcome(ARRIVING, _board({(7, 1): (FIRST, "S"), (3, 3): (SECOND, "M")}, SECOND)) == f"won:{FIRST}"
    assert outcome(ARRIVING, _board({(6, 2): (FIRST, "S"), (3, 3): (SECOND, "M")}, SECOND)) == ""
    assert outcome(SET_APART, _board({(7, 1): (FIRST, "S"), (3, 3): (SECOND, "M")}, SECOND)) == ""


def test_her_search_goes_for_the_arrival_that_wins() -> None:
    pytest.importorskip("aura_board_search")
    from core.agency.searching_by_kind import search

    position = _board({(6, 2): (FIRST, "S"), (0, 0): (FIRST, "M"), (4, 4): (SECOND, "M"), (5, 5): (SECOND, "M")}, FIRST)
    found = search(ARRIVING, position, seconds=0.3)
    assert found.move is not None and found.move.start == (6, 2) and found.move.end[0] == 7 and found.mate, found


def test_an_arrival_the_window_called_a_win_is_learned() -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, SET_APART.start)
    final = _board({(7, 1): (FIRST, "S"), (3, 3): (SECOND, "M")}, SECOND)
    learned.the_window_said("won", final, Move((6, 2), (7, 1)))
    assert learned.wins_arriving == {"S"} and learned.family().kinds["S"].wins_arriving
    again = HowThePiecesMove.from_memory(learned.as_memory())
    assert again is not None and again.wins_arriving == {"S"}


def test_a_side_left_with_no_move_that_the_window_says_lost_is_learned() -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, SET_APART.start)
    # A man offered its two steps forward and kept from stepping back: on the far row it has nowhere to go.
    learned.offered(_board({(2, 2): (FIRST, "M"), (5, 5): (SECOND, "M")}, FIRST), (2, 2), {(3, 1), (3, 3)})
    learned.offered(_board({(2, 2): (FIRST, "M"), (5, 5): (SECOND, "M")}, FIRST), (2, 2), {(3, 1), (3, 3)})
    final = _board({(7, 1): (FIRST, "M"), (0, 0): (SECOND, "M")}, FIRST)
    learned.the_window_said("lost", final, Move((1, 1), (0, 0)))
    assert learned.no_move_loses and learned.family().no_move_unattacked == "loses"

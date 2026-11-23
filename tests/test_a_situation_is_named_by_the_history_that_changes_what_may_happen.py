"""Two situations with the same pieces, where what may be done differs, have different names.

From the suggestions of 23 November 2026: her name for a situation was its pieces and whose turn it was, so a king
that had moved and come back shared a name with one that had not, and what she worked out or saw in one was used in
the other. The history that changes what may happen next is now part of the name wherever it is not the ordinary one;
where it is, the name is the pieces' alone, as it always was, so everything she kept under it stays hers.
"""
from __future__ import annotations

from core.agency.pieces_by_their_kind import Position
from core.agency.playing_pieces_by_kind import _a_place
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

START = dict(CHESS.start)


def test_the_ordinary_history_keeps_the_name_it_always_had() -> None:
    begun = CHESS.the_start()
    assert _a_place(begun, START) == _a_place(begun)


def test_a_piece_back_on_its_square_after_moving_is_another_situation() -> None:
    begun = CHESS.the_start()
    moved_and_back = Position(cells=begun.cells, turn=begun.turn, unmoved=begun.unmoved - {(0, 4)})
    assert _a_place(moved_and_back) == _a_place(begun), "by its pieces alone the two are one"
    assert _a_place(moved_and_back, START) != _a_place(begun, START)


def test_a_square_passable_for_one_move_is_another_situation() -> None:
    passing = from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6")
    not_passing = from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - -")
    assert _a_place(passing) == _a_place(not_passing)
    assert _a_place(passing, START) != _a_place(not_passing, START)


def test_a_passable_square_nothing_can_take_on_leaves_the_name_as_it_was() -> None:
    """After a two-square step a square is passable in nearly every opening; only where a move of the side to move
    takes in passing does it change what may happen, and only there is it part of the name."""
    passing = from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6")
    unusable = from_fen("4k3/8/8/3p4/8/8/8/4K3 w - d6")
    plain = from_fen("4k3/8/8/3p4/8/8/8/4K3 w - -")
    assert _a_place(passing, START, CHESS) != _a_place(from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - -"), START, CHESS)
    assert _a_place(unusable, START, CHESS) == _a_place(plain, START, CHESS)

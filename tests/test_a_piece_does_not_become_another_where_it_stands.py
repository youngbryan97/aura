"""A piece that did not move is read as the kind she holds, whatever mark is drawn over it; and a board of two sides
with fewer pieces than its game begins with is not a game at its start
(core/agency/playing_pieces_by_kind.py ``_read_against``; core/skills/screen_pursuit_board_of_pieces.py).

LIVE 25 November 2026 against Stockfish: her royal piece, under attack, was drawn with a glow her kept looks held as a
look of its own, learned from a finished game's board taken for one at its start. Read as that look, the piece was no
longer the one the game keeps safe, and twice she moved as if it were not attacked; the game refused both moves.
"""
from __future__ import annotations

import pytest

from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, SECOND
from core.agency.playing_pieces_by_kind import _held_by_each, _read_against
from core.skills.screen_pursuit_board_of_pieces import _fewer_than_it_begins_with
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

pytestmark = pytest.mark.unit

H2, G4, E5 = (1, 7), (3, 6), (4, 4)
GLOW = "7"  # a look neither side has ever held


def _learned() -> HowThePiecesMove:
    return HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)


def test_a_piece_that_stood_still_read_as_a_look_its_side_never_held_is_read_as_the_kind_she_holds() -> None:
    learned = _learned()
    believed = {H2: (FIRST, "K"), E5: (SECOND, "N")}
    shown = {H2: (FIRST, GLOW), G4: (SECOND, "N")}  # their leaper moved; her royal piece glows where it stands
    read = _read_against(shown, believed, held=_held_by_each(learned), landed=G4)
    assert read[H2] == (FIRST, None), "unread: its kind is taken from the board she holds"
    assert read[G4] == (SECOND, "N")


def test_where_a_move_landed_a_new_look_is_left_for_the_becoming_to_be_seen() -> None:
    learned = _learned()
    believed = {(7, 0): (FIRST, "P")}
    shown = {(7, 0): (FIRST, GLOW)}
    assert _read_against(shown, believed, held=_held_by_each(learned), landed=(7, 0))[(7, 0)] == (FIRST, GLOW)


def test_a_piece_read_as_a_kind_its_side_holds_is_read_as_that_kind() -> None:
    learned = _learned()
    believed = {H2: (FIRST, "K"), (1, 6): (FIRST, "R")}
    shown = {H2: (FIRST, "R"), (1, 6): (FIRST, "K")}  # two of a side changing places is a move to see, not a mark
    assert _read_against(shown, believed, held=_held_by_each(learned), landed=None) == shown


def test_a_board_of_two_sides_with_fewer_pieces_than_its_game_begins_with_is_not_at_its_start() -> None:
    learned = _learned()
    finished = {sq: held for sq, held in CHESS.start.items() if sq[0] in (0, 7)}
    assert _fewer_than_it_begins_with(finished, learned)
    assert not _fewer_than_it_begins_with(dict(CHESS.start), learned)
    assert not _fewer_than_it_begins_with(finished, None), "with no record, any start is taken as one"


def test_a_becoming_seen_once_does_not_make_a_mark_a_kind_its_side_holds() -> None:
    """The glow round her royal piece, read once on a square a move had landed on, was counted as her royal kind
    becoming the glow's look. Seen once, that becoming is no reason to read the glow as a piece's kind elsewhere."""
    from core.agency.pieces_by_their_kind import Move

    learned = _learned()
    learned._becomes("K", Move((0, 6), (1, 7), becomes=GLOW), 1)
    believed = {H2: (FIRST, "K"), E5: (SECOND, "N")}
    shown = {H2: (FIRST, GLOW), G4: (SECOND, "N")}
    assert _read_against(shown, believed, held=_held_by_each(learned), landed=G4)[H2] == (FIRST, None)

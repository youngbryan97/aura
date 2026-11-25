"""What the game withheld or refused in a position she held without the kind it keeps safe teaches nothing of how the
pieces go (core/agency/how_the_pieces_move.py): she misread that position, and a game withholds moves for the whole
position, keeping that kind safe above all. What it offered there still counts.

LIVE 25 November 2026: 52 of the 800 answers kept from En Croissant offered fewer squares than the game's rules give in
the position she held, most where she read her king, glowing in check, as another kind. Two refused takes four squares
down her slant-runner's backward diagonal held it no further than three, and her search never moved it further.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from core.agency.how_the_pieces_move import READING, HowThePiecesMove
from core.agency.pieces_by_their_kind import Move, Position
from core.agency.what_the_game_showed import WhatTheGameShowed
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

pytestmark = pytest.mark.unit

UP = (1, 0)
FILE = {(r, 0) for r in range(1, 8)}


def _rules() -> HowThePiecesMove:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    # Seen kept off squares the other side attacks, as the game keeps its royal kind.
    rules.kept_off_attacked = {"K": 6}
    open_file = from_fen("4k3/8/8/8/8/8/8/R3K3 w - -")
    for _ in range(3):
        rules.offered(open_file, (0, 0), FILE | {(0, 1), (0, 2), (0, 3)}, family=CHESS)
    assert rules.royal("K")
    return rules


def _without_her_king(fen: str) -> Position:
    """The board as read with her king glowing in check: read as a kind she holds no rules for."""
    board = from_fen(fen)
    cells = {square: ((side, "7") if (side, kind) == (0, "K") else (side, kind))
             for square, (side, kind) in board.cells.items()}
    return replace(board, cells=cells)


def test_what_the_game_withholds_where_she_held_no_royal_piece_says_nothing_of_the_ways() -> None:
    rules = _rules()
    misread = _without_her_king("4k3/8/8/8/8/8/8/R3K2r w - -")
    went = rules.seen["R"][UP].went_at.get(1, 0)
    # In check along the first row, the game offers her straight-runner only the square that blocks it.
    rules.offered(misread, (0, 0), {(0, 1)}, family=CHESS)
    assert not rules.seen["R"][UP].withheld, "the file was withheld for the check she did not see"
    assert rules.seen["R"][(0, 1)].went_at.get(1, 0) >= 1, "what was offered still counts"
    assert rules.seen["R"][UP].went_at.get(1, 0) == went


def test_a_refusal_where_she_held_no_royal_piece_is_not_counted() -> None:
    rules = _rules()
    misread = _without_her_king("4k3/8/8/8/8/8/8/R3K2r w - -")
    rules.refused(misread, Move((0, 0), (3, 0)), family=CHESS)
    assert rules.seen["R"][UP].refused_going_at == {}
    assert 3 in rules.seen["R"][UP].offered


def test_with_her_royal_piece_held_a_withheld_square_still_counts() -> None:
    rules = _rules()
    held_rightly = from_fen("4k3/8/8/8/8/8/8/R3K3 w - -")
    rules.offered(held_rightly, (0, 0), {(1, 0), (2, 0)}, family=CHESS)
    assert rules.seen["R"][UP].withheld == {3, 4, 5, 6, 7}


def test_refusals_an_earlier_reading_counted_stand_only_as_far_as_the_refusals_kept_bear_them_out() -> None:
    rules = _rules()
    misread = _without_her_king("4k3/8/8/8/8/8/8/R3K2r w - -")
    way = rules.seen["R"][UP]
    for _ in range(4):
        way.a_refusal(3, taking=False)  # counted by a reading before READING 7
    assert 3 not in way.offered
    kept = rules.as_memory()
    kept["read_by"] = 6
    read = HowThePiecesMove.from_memory(kept)
    assert read.read_by == READING
    read.shown = WhatTheGameShowed()
    for _ in range(4):
        read.shown.refused(misread, Move((0, 0), (3, 0)))
    read.shown.offered(from_fen("4k3/8/8/8/8/8/8/R3K3 w - -"), (0, 0), FILE)
    read.refusals_read_again()
    again = read.seen["R"][UP]
    assert again.refused_going_at == {}
    assert 3 in again.offered, "a distance the answers kept offer is put back once it stands again"
    read.refusals_read_again()  # once: what was let go is not read again
    assert again.refused_going_at == {}

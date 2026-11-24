"""Her compiled search takes shortcuts that must give the answers the long way gives, in any game.

Asking whether a square is attacked used to walk every way of every kind out from it: four kinds that move along the
same line walked it four times. It now walks each line once, stops at the first piece, and looks up how far that
piece's kind takes along the line. After a move, only the lines through the squares the move changed are asked about,
since nothing else can have opened. A kind none of whose ways reach the same square twice is no longer checked for a
move found twice. Profiled on 23 November 2026 these took her search from 8.3 to 3.8 seconds for the same 150,000-node
searches, node for node the same.

Each is checked here against her rules' own move lists, counted move by move to a depth, in games where kinds share
lines, where one kind's ways overlap, and where ways are unsure.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from core.agency.pieces_by_their_kind import Position, Step, after, moves
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

engine = pytest.importorskip("aura_board_search")


def _counted(family, position: Position, depth: int) -> int:
    found = moves(family, position)
    if depth == 1:
        return len(found)
    return sum(_counted(family, after(family, position, move), depth - 1) for move in found)


def _overlapping_and_unsure():
    """The leaper also steps two straight ahead by a ride of two and by a leap, so one square is reached two ways; the
    slant-runner holds a leap no look has confirmed; the royal piece rides a whole rank, also unconfirmed."""
    knight, bishop, king = CHESS.kinds["N"], CHESS.kinds["B"], CHESS.kinds["K"]
    return replace(CHESS, kinds={
        **CHESS.kinds,
        "N": replace(knight, steps=(*knight.steps, Step(((1, 0),), rides=2), Step(((2, 0),), rides=1))),
        "B": replace(bishop, steps=(*bishop.steps, Step(((3, -1),), rides=1, sure=False))),
        "K": replace(king, steps=(*king.steps, Step(((0, 1), (0, -1)), rides=0, sure=False))),
    })


POSITIONS = [
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -",
]


@pytest.mark.parametrize("fen", POSITIONS)
def test_the_compiled_moves_are_her_rules_moves_where_kinds_share_lines(fen: str) -> None:
    from core.agency.searching_by_kind import described, placed

    position = from_fen(fen)
    assert engine.perft(described(CHESS), placed(position), 3) == _counted(CHESS, position, 3)


@pytest.mark.parametrize("fen", POSITIONS)
def test_the_compiled_moves_are_her_rules_moves_where_ways_overlap_and_are_unsure(fen: str) -> None:
    from core.agency.searching_by_kind import described, placed

    family, position = _overlapping_and_unsure(), from_fen(fen)
    assert engine.perft(described(family), placed(position), 3) == _counted(family, position, 3)

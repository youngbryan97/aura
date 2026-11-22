"""The vocabulary reaches a game whose pieces take by leaping over, again and again, where a side that can take must.

Her movement vocabulary (core/agency/pieces_by_their_kind.py) had no take over, no take again and no taking first, so a
game of draughts could not be held in it, and her play on a page held draughts as rules written in by hand
(core/agency/rules_of_a_board.py, 11 October 2026). The game is described here as English players play it, for tests
only, the way chess is; she is never given it. The counts are the published ones for draughts from its start; chains of
take overs are counted once for each set of pieces taken and square landed on.
"""
from __future__ import annotations

import pytest

from core.agency.pieces_by_their_kind import FIRST, SECOND, Move, Position, after, attacked, moves, outcome
from tests.fixtures.a_game_of_pieces_to_learn import DRAUGHTS

#: Moves from the start of draughts, depth by depth (published perft counts).
PUBLISHED = [7, 49, 302, 1469, 7361, 36768]


def _perft(position: Position, depth: int) -> int:
    found = moves(DRAUGHTS, position)
    if depth == 1:
        return len(found)
    return sum(_perft(after(DRAUGHTS, position, move), depth - 1) for move in found)


@pytest.mark.parametrize("depth", range(1, 6))
def test_the_moves_from_the_start_are_the_published_count(depth: int) -> None:
    assert _perft(DRAUGHTS.the_start(), depth) == PUBLISHED[depth - 1]


@pytest.mark.parametrize("depth", range(1, 7))
def test_the_compiled_search_makes_the_same_moves(depth: int) -> None:
    engine = pytest.importorskip("aura_board_search")
    from core.agency.searching_by_kind import described, placed

    assert engine.perft(described(DRAUGHTS), placed(DRAUGHTS.the_start()), depth) == PUBLISHED[depth - 1]


def _board(cells: dict, turn: int = FIRST) -> Position:
    return Position(cells=cells, turn=turn, unmoved=frozenset())


def test_a_take_over_lands_beyond_and_takes_the_piece_between() -> None:
    position = _board({(2, 2): (FIRST, "M"), (3, 3): (SECOND, "M"), (7, 7): (SECOND, "M")})
    found = moves(DRAUGHTS, position)
    assert found == [Move((2, 2), (4, 4), taken_at=(3, 3))], found
    made = after(DRAUGHTS, position, found[0])
    assert (3, 3) not in made.cells and made.cells[(4, 4)] == (FIRST, "M")
    assert attacked(DRAUGHTS, position, (3, 3), FIRST)


def test_it_takes_again_while_it_can_and_a_side_that_can_take_must() -> None:
    position = _board({(0, 0): (FIRST, "M"), (1, 1): (SECOND, "M"), (3, 3): (SECOND, "M"), (0, 6): (FIRST, "M")})
    found = moves(DRAUGHTS, position)
    # The quiet steps of the other man are not moves while a take is there.
    assert len(found) == 1, found
    move = found[0]
    assert move.end == (4, 4) and move.via == ((2, 2),) and set(move.taken()) == {(1, 1), (3, 3)}
    made = after(DRAUGHTS, position, move)
    assert not any(side == SECOND for side, _k in made.cells.values())
    assert outcome(DRAUGHTS, made) == f"won:{FIRST}", "a side with no move has lost"


def test_a_man_reaching_the_far_row_becomes_a_king_and_stops_there() -> None:
    position = _board({(5, 5): (FIRST, "M"), (6, 6): (SECOND, "M"), (6, 4): (SECOND, "M"), (0, 0): (SECOND, "M")})
    found = [m for m in moves(DRAUGHTS, position) if m.start == (5, 5)]
    assert {(m.end, m.becomes, m.taken()) for m in found} == {((7, 7), "K", ((6, 6),)), ((7, 3), "K", ((6, 4),))}

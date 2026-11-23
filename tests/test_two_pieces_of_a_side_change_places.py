"""The vocabulary holds a piece that changes places with a piece of its own side, and she learns it from the game.

LIVE 11 October 2026 a game on a web page played "like checkers but with a huge twist": a piece set apart on each side
that "can swap places with any of your pieces next to it". Her play on a page held that as words read by pattern
(core/agency/playing_a_board.py); the vocabulary had no swap. The game here is for tests only, as chess is: draughts
with one piece set apart for each side, which steps any way on the slant and swaps with its own side's pieces beside it.
"""
from __future__ import annotations

import pytest

from core.agency.how_the_pieces_move import HowThePiecesMove, a_move_seen, moves_seen
from core.agency.pieces_by_their_kind import FIRST, SECOND, Family, Kind, Move, Position, Step, after, attacked, moves
from tests.fixtures.a_game_of_pieces_to_learn import DRAUGHTS

_DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))
SET_APART = Family(
    name="draughts with a piece set apart", rows=8, columns=8,
    kinds={**DRAUGHTS.kinds,
           "S": Kind("S", name="the piece set apart", steps=(Step(_DIAGONAL, may="moves"), Step(_DIAGONAL, may="swaps"),
                                                             Step(((2, 2), (2, -2), (-2, 2), (-2, -2)), may="takes over")),
                     takes_again=True)},
    start={**{sq: v for sq, v in DRAUGHTS.start.items() if sq not in ((0, 4), (7, 3))},
           (0, 4): (FIRST, "S"), (7, 3): (SECOND, "S")},
    count_to_draw=80, no_move_unattacked="loses", takes_first=True,
)


def _board(cells: dict, turn: int = FIRST) -> Position:
    return Position(cells=cells, turn=turn, unmoved=frozenset())


def test_a_swap_changes_two_pieces_places() -> None:
    position = _board({(2, 2): (FIRST, "S"), (3, 3): (FIRST, "M"), (1, 1): (FIRST, "M"), (7, 7): (SECOND, "M")})
    swaps = [m for m in moves(SET_APART, position) if m.partner is not None]
    assert {(m.start, m.end) for m in swaps} == {((2, 2), (3, 3)), ((2, 2), (1, 1))}
    made = after(SET_APART, position, next(m for m in swaps if m.end == (3, 3)))
    assert made.cells[(3, 3)] == (FIRST, "S") and made.cells[(2, 2)] == (FIRST, "M") and len(made.cells) == 4
    # A swap takes nothing and threatens nothing.
    assert not attacked(SET_APART, _board({(2, 2): (FIRST, "S"), (3, 3): (SECOND, "M")}), (1, 1), FIRST)


def _perft(family: Family, position: Position, depth: int) -> int:
    found = moves(family, position)
    return len(found) if depth == 1 else sum(_perft(family, after(family, position, m), depth - 1) for m in found)


@pytest.mark.parametrize("depth", range(1, 5))
def test_the_compiled_search_makes_the_same_moves(depth: int) -> None:
    engine = pytest.importorskip("aura_board_search")
    from core.agency.searching_by_kind import described, placed

    start = SET_APART.the_start()
    assert engine.perft(described(SET_APART), placed(start), depth) == _perft(SET_APART, start, depth)


def test_she_learns_a_swap_from_what_the_game_offers() -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, SET_APART.start)
    position = _board({(2, 2): (FIRST, "S"), (3, 3): (FIRST, "M"), (7, 7): (SECOND, "M")})
    learned.offered(position, (2, 2), {(3, 3), (3, 1), (1, 1), (1, 3)})
    steps = {(s.offsets, s.may) for s in learned.family().kinds["S"].steps}
    assert (((1, 1),), "swaps") in steps and (((1, 1),), "moves") not in steps


def test_a_swap_watched_is_read_with_the_piece_set_apart_as_the_one_that_moved() -> None:
    before = _board({(2, 2): (FIRST, "S"), (3, 3): (FIRST, "M"), (1, 1): (FIRST, "M"), (7, 7): (SECOND, "M")})
    seen = {(2, 2): (FIRST, "M"), (3, 3): (FIRST, "S"), (1, 1): (FIRST, "M"), (7, 7): (SECOND, "M")}
    assert a_move_seen(before, seen, FIRST) == Move((2, 2), (3, 3), partner=((3, 3), (2, 2)))
    assert moves_seen(before, seen, FIRST, SET_APART) == [Move((2, 2), (3, 3), partner=((3, 3), (2, 2)))]
    learned = HowThePiecesMove.from_the_start(8, 8, SET_APART.start)
    learned.watched(before, Move((2, 2), (3, 3), partner=((3, 3), (2, 2))))
    assert any(s.may == "swaps" for s in learned.family().kinds["S"].steps)
    assert not learned.crosses, "a swap is not a crossing"

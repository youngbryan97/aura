"""A line that comes into an ending she has worked out to the end is scored by what the ending comes to.

Her general working-out of a small world settles every situation of a small ending: won, lost or neither, and in how
many turns (core/agency/what_a_small_ending_comes_to.py). She used it only where the game itself stood in such an
ending. Her compiled search now holds the endings she has worked out and looks each position it meets up in them, so a
line that trades into one is scored by what it comes to, however far past her horizon the end lies. The rules of
chess stand in for hers, on a board of four squares by four.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from core.agency.pieces_by_their_kind import FIRST, SECOND, Position
from core.agency.searching_by_kind import first_weights, search
from core.agency.what_a_small_ending_comes_to import _index_of, an_ending_worked_out
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

SMALL = replace(CHESS, rows=4, columns=4, start={})
PIECES = ((FIRST, "K"), (FIRST, "Q"), (SECOND, "K"))
MATE = 1_000_000


def _a_long_win(ending) -> Position:
    """A situation with her to move that the ending says she wins, but only after many turns."""
    squares = SMALL.rows * SMALL.columns
    longest = max(v for i, v in enumerate(ending.values) if i % 2 == 0 and v > 0)
    for king in range(squares):
        for queen in range(squares):
            for theirs in range(squares):
                if len({king, queen, theirs}) < 3:
                    continue
                if ending.values[_index_of((king, queen, theirs), FIRST, squares)] == longest:
                    at = {divmod(king, 4): (FIRST, "K"), divmod(queen, 4): (FIRST, "Q"),
                          divmod(theirs, 4): (SECOND, "K")}
                    return Position(cells=at, turn=FIRST)
    raise AssertionError("no long win")


def test_a_search_given_the_ending_scores_a_win_past_its_horizon_as_won() -> None:
    pytest.importorskip("aura_board_search")
    ending = an_ending_worked_out(SMALL, PIECES)
    here = _a_long_win(ending)
    weights = first_weights(SMALL)
    without = search(SMALL, here, seconds=10.0, nodes=150, weights=weights)
    given = search(SMALL, here, seconds=10.0, nodes=150, weights=weights, endings={ending.pieces: ending})
    assert without.score < MATE - 1000
    assert given.score > MATE - 1000
    # And the win it scores is the one the ending says: as many plies as it takes, from her move on.
    assert MATE - given.score == ending.value_of(here) - 1


def test_endings_worked_out_under_other_rules_are_not_used() -> None:
    pytest.importorskip("aura_board_search")
    ending = an_ending_worked_out(SMALL, PIECES)
    stale = replace(ending, rules="not these rules")
    here = _a_long_win(ending)
    found = search(SMALL, here, seconds=10.0, nodes=150, weights=first_weights(SMALL), endings={ending.pieces: stale})
    assert found.score < MATE - 1000

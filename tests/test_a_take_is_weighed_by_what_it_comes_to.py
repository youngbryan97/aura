"""A take is weighed by what it comes to once both sides have taken back on its square, not by what it takes.

Her search weighed a take by the worth of what it took, so past the horizon it looked at a queen taking a guarded pawn
as hard as a pawn taking a guarded knight, and ordered both among the best moves. Now each take is played out on its
square, each side taking back with its least worth piece while that pays (a piece behind another on a line joins in
once the one in front has gone; a royal piece takes only where nothing can take it back), and a take that loses is
ordered after the quiet moves and not looked at past the horizon. Nothing in it knows a game: pieces, their worth and
the lines they take along are the rules and weights she holds. Measured 23 November 2026 in her model of chess, 300
games each: 0.702 against the old weighing at equal nodes, 0.637 at equal time.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from core.agency.searching_by_kind import described, first_weights, placed
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

engine = pytest.importorskip("aura_board_search")
WORTH = {"P": 100, "N": 300, "B": 300, "R": 500, "Q": 900, "K": 0}


def _comes_to(fen: str, start: tuple[int, int], end: tuple[int, int]) -> int:
    weights = replace(first_weights(CHESS), worth=WORTH)
    return engine.exchange(described(CHESS), placed(from_fen(fen)), weights.as_json(), start, end)


def test_a_take_of_a_guarded_piece_by_a_worthier_one_loses_the_difference() -> None:
    # The queen takes a pawn the other pawn guards.
    assert _comes_to("4k3/8/2p5/3p4/8/8/8/3QK3 w - -", (0, 3), (4, 3)) == 100 - 900


def test_a_take_of_a_guarded_piece_by_a_lesser_one_wins_the_difference() -> None:
    assert _comes_to("4k3/8/2p5/3n4/4P3/8/8/4K3 w - -", (3, 4), (4, 3)) == 300 - 100


def test_a_piece_behind_another_on_its_line_joins_in() -> None:
    # Two rooks on a file against two: the second of hers takes back through the square the first left.
    assert _comes_to("r3k3/r7/8/8/8/8/R7/R3K3 w - -", (1, 0), (6, 0)) == 500


def test_a_royal_piece_takes_back_only_where_nothing_can_take_it() -> None:
    guarded = "4k3/5p2/8/6N1/2B5/8/8/4K3 w - -"
    alone = "4k3/5p2/8/8/2B5/8/8/4K3 w - -"
    assert _comes_to(guarded, (3, 2), (6, 5)) == 100
    assert _comes_to(alone, (3, 2), (6, 5)) == 100 - 300

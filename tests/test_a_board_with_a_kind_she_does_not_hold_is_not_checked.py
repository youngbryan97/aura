"""A board holding a piece of a look her rules do not hold is passed over when her rules are checked against what the
game showed (core/agency/what_the_game_showed.py).

LIVE 25 November 2026, the end of a game against Stockfish: a piece the other side's step became stood on the board
as a look she had no rules for, the check of her rules raised KeyError on it, and what the game had taught her was not
kept.
"""
from __future__ import annotations

import pytest

from core.agency.pieces_by_their_kind import Move, Position
from core.agency.what_the_game_showed import WhatTheGameShowed
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

pytestmark = pytest.mark.unit


def test_a_board_with_a_look_her_rules_do_not_hold_is_passed_over_and_the_rest_is_checked() -> None:
    shown = WhatTheGameShowed()
    known = from_fen("4k3/8/8/8/8/8/4P3/4K3 w - -")
    shown.offered(known, (1, 4), {(2, 4), (3, 4)})
    cells = dict(known.cells)
    cells[(7, 0)] = (1, "6")  # a piece of theirs whose look she has no rules for
    unknown = Position(cells=cells, turn=known.turn, unmoved=known.unmoved)
    shown.offered(unknown, (1, 4), {(2, 4), (3, 4)})
    shown.refused(unknown, Move((1, 4), (2, 3)))
    checked = shown.checked(CHESS)
    assert checked.answers == 1 and checked.missing == 0 and checked.spurious == 0

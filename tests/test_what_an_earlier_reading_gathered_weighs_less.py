"""What a reading since mended gathered weighs as kept knowledge does when it comes back, once.

LIVE 23 November 2026, game 31: 31 of the squares she read in 55 answers were never offered, most of them the marks of
the piece looked at before (mended in f60209bb0). Thirty games read that way left her rules with ways seen once that
no later look could test.
"""
from __future__ import annotations

import pytest

from core.agency.how_the_pieces_move import READING, HowThePiecesMove
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

pytestmark = pytest.mark.unit

E1 = (0, 4)


def _kept_before_mending() -> dict:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    alone = from_fen("k7/8/8/8/8/8/8/R3K3 w - -")
    around = {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5)}
    for _ in range(6):
        rules.offered(alone, E1, around, family=CHESS)  # her royal kind's steps, seen often
        rules.offered(alone, (0, 0), {(r, 0) for r in range(1, 8)} | {(0, 1), (0, 2), (0, 3)}, family=CHESS)
    rules.offered(alone, E1, around | {(2, 2)}, family=CHESS)  # a misread: a leap two up and two across, once
    kept = rules.as_memory()
    kept.pop("read_by")  # kept by the reading before
    return kept


def test_a_way_an_earlier_reading_saw_once_is_let_go_and_what_it_saw_often_stays() -> None:
    read = HowThePiecesMove.from_memory(_kept_before_mending())
    king = read.family().kinds["K"]
    assert not [s for s in king.steps if s.offsets == ((2, -2),) or s.offsets == ((1, -1),) and s.rides == 2]
    assert any(s.offsets == ((1, 0),) and s.sure for s in king.steps)
    # The straight-runner's file, a run from the next square to the far side, is still the whole line.
    rook = read.family().kinds["R"]
    assert any(s.offsets == ((1, 0),) and s.rides == 0 for s in rook.steps)
    assert read.read_by == READING


def test_it_weighs_less_once() -> None:
    once = HowThePiecesMove.from_memory(_kept_before_mending())
    twice = HowThePiecesMove.from_memory(once.as_memory())
    assert twice.as_memory() == once.as_memory()

"""A longer step a piece may take only from its first square is not how it runs (HowThePiecesMove.what_it_does).

LIVE 21 November 2026, against Stockfish, she named the pawns by their first double step as "the piece that runs
straight", the words she uses for the rooks, so "They played their piece that runs straight from d7 to d5" did not say
which piece had moved.
"""
from __future__ import annotations

from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import Step
from tests.fixtures.a_game_of_pieces_to_learn import CHESS


def _as_learned(kind: str) -> tuple[Step, ...]:
    """A kind's steps as her learning holds them: one way each."""
    return tuple(Step((way,), rides=step.rides, may=step.may, only_unmoved=step.only_unmoved)
                 for step in CHESS.kinds[kind].steps for way in step.offsets)


def test_the_kind_with_a_first_double_step_is_not_named_as_the_one_that_runs(monkeypatch) -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, CHESS.start)
    monkeypatch.setattr(learned, "_steps_of", _as_learned)
    assert learned.what_it_does("P") == "the piece that steps forward"
    assert learned.what_it_does("R") == "the piece that runs straight"
    assert len({learned.what_it_does(kind) for kind in ("P", "R", "B", "Q", "N")}) == 5

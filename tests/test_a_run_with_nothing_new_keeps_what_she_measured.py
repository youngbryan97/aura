"""A run that has nothing new of what she measured each kind to be worth keeps what she had
(core/skills/screen_pursuit_board_of_pieces.py ``_keep_what_she_learned``).

LIVE 25 November 2026 her record of the board she plays Stockfish on had lost the worth 136 rounds of rehearsal had
earned her: a record is written whole, and a run with no rehearsal in hand (a board not yet taking moves) wrote hers
without it. She played on the first guess, which holds a leaper worth less than a forward stepper, and gave a leaper
for a forward stepper on her third move.
"""
from __future__ import annotations

import pytest

import core.runtime.what_she_learned as what_she_learned
import core.skills.screen_pursuit_board_of_pieces as board_of_pieces
from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.searching_by_kind import Weights
from core.agency.what_a_kind_is_worth import Rehearsal
from core.perception.pieces_by_their_look import Looks
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

pytestmark = pytest.mark.unit


def test_a_run_with_no_rehearsal_in_hand_keeps_the_worth_she_measured(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(what_she_learned, "_kept_in", lambda: tmp_path)
    learned = HowThePiecesMove.from_the_start(8, 8, CHESS.start)
    measured = Rehearsal(weights=Weights(worth={"N": 300.0, "P": 100.0}), earned=["worth"], lengths=[80], rounds=136)
    board_of_pieces._keep_what_she_learned("A Board", Looks(), learned, measured)
    # A board not yet taking moves: no rehearsal in hand, and nothing of the rules either.
    board_of_pieces._keep_what_she_learned("A Board", Looks(), None, None)
    kept = board_of_pieces._what_she_kept("A Board")
    assert kept.get("rehearsal") is not None
    assert kept["rehearsal"].weights.worth == {"N": 300.0, "P": 100.0} and kept["rehearsal"].rounds == 136
    assert kept.get("rules") is not None


def test_what_a_run_has_new_replaces_what_was_kept(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(what_she_learned, "_kept_in", lambda: tmp_path)
    learned = HowThePiecesMove.from_the_start(8, 8, CHESS.start)
    first = Rehearsal(weights=Weights(worth={"N": 300.0}), earned=["worth"], lengths=[80], rounds=1)
    board_of_pieces._keep_what_she_learned("A Board", Looks(), learned, first)
    later = Rehearsal(weights=Weights(worth={"N": 320.0}), earned=["worth"], lengths=[80], rounds=2)
    board_of_pieces._keep_what_she_learned("A Board", Looks(), learned, later)
    assert board_of_pieces._what_she_kept("A Board")["rehearsal"].weights.worth == {"N": 320.0}

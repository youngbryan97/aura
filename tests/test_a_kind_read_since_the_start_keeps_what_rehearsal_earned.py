"""A kind read since the start of a game, one a piece became or a look she misread, keeps what rehearsal earned for the
kinds the game begins with (core/agency/playing_pieces_by_kind.py ``_Rehearsing.fitted_to``).

LIVE 25 November 2026 her king, glowing in check, read as a kind of its own, twice threw away the 143 and 147 rounds her
weights were earned in. With where each kind stands lost, she opened 1.a4 and walked her king to f1.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.agency.pieces_by_their_kind import Kind, Step
from core.agency.playing_pieces_by_kind import _Rehearsing
from core.agency.searching_by_kind import Weights, first_weights
from core.agency.what_a_kind_is_worth import Rehearsal
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

pytestmark = pytest.mark.unit

TABLE = [[float(r * 8 + c) for c in range(8)] for r in range(8)]


def _earned() -> _Rehearsing:
    weights = replace(first_weights(CHESS), squares={"K": TABLE, "N": TABLE})
    rehearsal = Rehearsal(weights=weights, earned=["worth", "where each stands"], royal=["K"], rounds=147)
    rehearsal.samples_x, rehearsal.samples_y = [np.zeros(4, dtype=np.float32)], [0.5]
    return _Rehearsing(rehearsal=rehearsal)


def test_a_kind_read_since_the_start_keeps_what_was_earned() -> None:
    rehearsing = _earned()
    misread = replace(CHESS, kinds={**CHESS.kinds, "7": Kind("7", steps=(Step(((1, 1),)),))})
    fitted = rehearsing.fitted_to(misread)
    assert fitted.earned == ["worth", "where each stands"] and fitted.rounds == 147
    assert fitted.weights.squares["K"] == TABLE, "where her king stands best is still hers"
    assert fitted.weights.worth["7"] == first_weights(misread).worth["7"], "the new kind has the first guess"
    assert fitted.samples_x == [], "what was gathered is laid out for the kinds before"


def test_a_kind_no_longer_read_is_let_go_and_the_rest_stands() -> None:
    rehearsing = _earned()
    rehearsing.rehearsal.weights = replace(rehearsing.rehearsal.weights,
                                           worth={**rehearsing.rehearsal.weights.worth, "7": 0.0})
    fitted = rehearsing.fitted_to(CHESS)
    assert "7" not in fitted.weights.worth and fitted.earned == ["worth", "where each stands"]


def test_another_kind_kept_safe_starts_rehearsal_again() -> None:
    rehearsing = _earned()
    rehearsing.rehearsal.royal = ["Q"]
    fitted = rehearsing.fitted_to(CHESS)
    assert fitted.earned == [] and fitted.weights.squares == {}


def test_a_game_that_begins_with_a_kind_never_rehearsed_starts_again() -> None:
    rehearsing = _earned()
    rehearsing.rehearsal.weights = Weights(worth={k: v for k, v in rehearsing.rehearsal.weights.worth.items()
                                                  if k != "N"})
    fitted = rehearsing.fitted_to(CHESS)
    assert fitted.earned == []

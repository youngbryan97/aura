"""A way of taking seen fewer times than a way is believed on, and refused more often than it was seen, is not held
(core/agency/how_the_pieces_move.py).

LIVE 25 November 2026 against Stockfish: one capture read two squares straight ahead stood against four times the kind
would not take there, so her rules held that it moves or takes that way, and she tried to take straight ahead. At one
in ten against, ten refusals would have been needed to let go of what one look had shown.
"""
from __future__ import annotations

import pytest

from core.agency.how_the_pieces_move import SEEN_TO_BELIEVE, HowThePiecesMove
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

pytestmark = pytest.mark.unit

AHEAD = (1, 0)


def _ways_ahead(took: int, would_not_take: int) -> set[str]:
    """What her rules let the forward-stepping kind do straight ahead, with its going there seen often and its taking
    there seen ``took`` times against ``would_not_take`` times it stopped short of a piece it could have reached."""
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    board = from_fen("4k3/8/8/8/8/8/4P3/4K3 w - -")
    for _ in range(6):
        rules.offered(board, (1, 4), {(2, 4), (3, 4)}, family=CHESS)
    way = rules.seen["P"][AHEAD]
    way.took, way.would_not_take = took, would_not_take
    return {step.may for step in rules.family().kinds["P"].steps if step.offsets == (AHEAD,)}


def test_a_taking_seen_once_and_refused_four_times_is_a_misreading() -> None:
    assert _ways_ahead(took=1, would_not_take=4) == {"moves"}


def test_a_taking_seen_as_often_as_a_way_is_believed_on_is_held_against_a_few_refusals() -> None:
    assert _ways_ahead(took=SEEN_TO_BELIEVE, would_not_take=SEEN_TO_BELIEVE + 1) == {"either"}


def test_a_taking_seen_once_and_never_refused_is_held_until_something_says_otherwise() -> None:
    assert _ways_ahead(took=1, would_not_take=0) == {"either"}

"""Before committing to a move, a look made only to break it: kept where it holds, replaced where another holds up
better (core/agency/looking_for_a_refutation.py). Measured at equal time it lost, and her play does not call it."""
from __future__ import annotations

from core.agency.looking_for_a_refutation import tried_against_its_refutation
from core.agency.pieces_by_their_kind import moves
from core.agency.playing_pieces_by_kind import _named
from core.agency.searching_by_kind import Found
from tests.fixtures.a_game_of_pieces_to_learn import CHESS


def _searches(their_score: int, other_score: int) -> object:
    start = CHESS.the_start()
    legal = sorted(moves(CHESS, start), key=_named)

    def search(family, position, *, seconds, set_aside=(), **_):
        if position.turn != start.turn:
            reply = sorted(moves(family, position), key=_named)[0]
            return Found(reply, their_score, 9, 100, [reply])
        best = next(m for m in legal if m not in set_aside)
        return Found(best, other_score, 9, 100, [best])

    return start, legal, search


def test_a_move_whose_refutation_holds_it_worse_than_believed_gives_way_to_one_that_holds_up_better() -> None:
    start, legal, search = _searches(their_score=100, other_score=20)
    chosen = Found(legal[0], 50, 9, 100, [legal[0]], unsteady=5)
    kept = tried_against_its_refutation(search, CHESS, start, chosen, seconds=1.0)
    assert kept.move == legal[1] and kept.by == "the first was refuted"


def test_a_move_its_refutation_holds_within_the_searchs_own_unsteadiness_is_kept() -> None:
    start, legal, search = _searches(their_score=-48, other_score=200)
    chosen = Found(legal[0], 50, 9, 100, [legal[0]], unsteady=5)
    assert tried_against_its_refutation(search, CHESS, start, chosen, seconds=1.0) is chosen

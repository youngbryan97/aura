"""A side that acts in the world does not count on its own unsure ways past its first move; the other side's count.

A search over a model with one false move in it does not treat the error as noise: it looks for it, because a move
that does not exist beats the moves that do. LIVE 23 November 2026 her rules held a leap her bishop does not have, and
a whole-file move of her king, and her search planned lines through them while saying she stood well. Each way she
learned now says whether the game has shown it often enough to rely on (core/agency/how_the_pieces_move.py,
SEEN_TO_BELIEVE). A search for her own play is wary of her unsure ways: her first move may still use one, since she
picks the piece up and the game shows whether it goes there before she makes it; her later moves and her threats may
not. The other side's plausible ways all count: what they might do is planned against, what she might do is not
counted on.
"""
from __future__ import annotations

import json
from dataclasses import replace

from core.agency.how_the_pieces_move import SEEN_TO_BELIEVE, HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, Move, Step
from core.agency.searching_by_kind import described, first_weights, placed, search
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen


def _with_a_misread_leap() -> object:
    """Her slant-runner given a leap six ahead and four across that no look has confirmed."""
    bishop = CHESS.kinds["B"]
    leap = Step(((6, -4),), rides=1, sure=False)
    return replace(CHESS, kinds={**CHESS.kinds, "B": replace(bishop, steps=(*bishop.steps, leap))})


#: Their queen boxed in on a8 by its own pieces, her slant-runner on e2: the misread leap would take it from anywhere.
BOXED = from_fen("qn5k/pp6/8/8/8/8/4B3/4K3 w - -")
E2, A8 = (1, 4), (7, 0)


def test_a_line_through_a_move_she_has_not_seen_enough_is_not_counted_on() -> None:
    family, weights = _with_a_misread_leap(), first_weights(CHESS)
    # Her first move may not be the leap itself: the question is what she counts on after it.
    aside = [Move(E2, A8, taken_at=A8)]
    hoping = search(family, BOXED, seconds=30.0, nodes=300_000, weights=weights, set_aside=aside)
    wary = search(family, BOXED, seconds=30.0, nodes=300_000, weights=weights, set_aside=aside, wary=FIRST)
    queen = weights.worth["Q"]
    assert hoping.score - wary.score > queen / 2, (hoping.score, wary.score)


def test_her_first_move_may_still_be_one_she_will_check_before_making_it() -> None:
    family, weights = _with_a_misread_leap(), first_weights(CHESS)
    found = search(family, BOXED, seconds=30.0, nodes=200_000, weights=weights, wary=FIRST)
    assert (found.move.start, found.move.end) == (E2, A8)


def test_the_other_sides_unsure_ways_are_planned_against() -> None:
    """Her queen where their misread leap could reach it: a wary search moves it away as an unwary one does."""
    family, weights = _with_a_misread_leap(), first_weights(CHESS)
    # Their slant-runner on e7 (its own e2, seen from their side), her queen on a1 within the leap's reach.
    there = from_fen("7k/4b3/8/8/8/8/8/Q3K3 w - -")
    wary = search(family, there, seconds=30.0, nodes=200_000, weights=weights, wary=FIRST)
    assert wary.move.start == (0, 0), "the queen stands where a move they might have takes it"


def test_how_sure_a_way_is_comes_from_how_often_the_game_showed_it() -> None:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    open_board = from_fen("6k1/8/8/8/8/8/4B3/4K3 w - -")
    rules.offered(open_board, E2, {(2, 3), (3, 2), (4, 1), (5, 0)}, family=CHESS)
    steps = {s.offsets: s for s in rules.family().kinds["B"].steps}
    # One look at the way up and to the left: four distances, each seen once.
    assert steps[((1, -1),)].sure == (4 >= SEEN_TO_BELIEVE)
    # Its image, up and to the right, has not been shown at all: an inference, not relied on.
    assert ((1, 1),) in steps and not steps[((1, 1),)].sure
    # A kind never seen to go is searched as going every way, and none of it is relied on.
    guessed = rules.family(unseen_go_every_way=True).kinds["N"].steps
    assert guessed and not any(s.sure for s in guessed)


def test_what_is_sure_and_who_is_wary_reach_the_compiled_search() -> None:
    family = _with_a_misread_leap()
    steps = json.loads(described(family))["kinds"]
    assert any(step["sure"] is False for kind in steps for step in kind["steps"])
    assert json.loads(placed(BOXED, [], FIRST))["wary"] == FIRST
    assert "wary" not in json.loads(placed(BOXED))


def test_a_piece_whose_answer_would_settle_an_unsure_way_is_picked_up_first() -> None:
    """Her royal piece held a leap three across and one ahead on one misread look. Where the square it lands on is
    empty, picking the piece up settles it: the game offers it or withholds it."""
    from types import SimpleNamespace

    from core.agency.playing_pieces_by_kind import _would_settle

    king = CHESS.kinds["K"]
    misread = replace(CHESS, kinds={**CHESS.kinds, "K": replace(king, steps=(*king.steps,
                                                                             Step(((1, 3),), rides=1, sure=False)))})
    learned = SimpleNamespace(family=lambda: misread, rows=8, columns=8)
    open_board = from_fen("6k1/8/8/8/8/8/8/4K3 w - -")
    assert _would_settle(SimpleNamespace(believed=open_board, learned=learned), (0, 4)) == 1
    # With her own piece on the square it would land on, the game's answer says nothing of it.
    blocked = from_fen("6k1/8/8/8/8/8/7P/4K3 w - -")
    assert _would_settle(SimpleNamespace(believed=blocked, learned=learned), (0, 4)) == 0

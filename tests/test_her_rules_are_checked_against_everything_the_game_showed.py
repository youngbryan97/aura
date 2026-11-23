"""Her rules are checked against every answer the game gave, not trusted for how much they have seen.

One false move in a model is not noise to a search, which goes looking for it; one missing defence turns a lost plan
into a won one. Each answer the game gives (the squares it offers a piece picked up, a move it refuses) is kept whole,
and her rules as they stand are checked against all of them: what they would offer that the game did not, and what
the game offered that they would not. Both should be none.
"""
from __future__ import annotations

from dataclasses import replace

from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import Move, Step, moves
from core.agency.what_the_game_showed import WhatTheGameShowed
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

MIDDLE = from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -")


def _shown_by_the_true_game() -> WhatTheGameShowed:
    shown = WhatTheGameShowed()
    for square in sorted({m.start for m in moves(CHESS, MIDDLE)}):
        shown.offered(MIDDLE, square, {m.end for m in moves(CHESS, MIDDLE) if m.start == square})
    return shown


def test_rules_that_are_the_game_disagree_with_nothing_it_showed() -> None:
    checked = _shown_by_the_true_game().checked(CHESS)
    assert checked.answers > 5 and checked.missing == 0 and checked.spurious == 0
    assert "agree with all" in checked.says()


def test_a_move_her_rules_invent_and_one_they_lack_are_both_found() -> None:
    bishop, knight = CHESS.kinds["B"], CHESS.kinds["N"]
    wrong = replace(CHESS, kinds={**CHESS.kinds,
                                  "B": replace(bishop, steps=(*bishop.steps, Step(((1, 0),), rides=1))),
                                  "N": replace(knight, steps=knight.steps[:-1])})
    checked = _shown_by_the_true_game().checked(wrong)
    assert checked.spurious > 0 and checked.missing > 0
    assert set(checked.by_kind) <= {"B", "N"} and checked.examples


def test_a_refused_move_her_rules_still_allow_is_counted_against_them() -> None:
    shown = WhatTheGameShowed()
    shown.refused(MIDDLE, Move((2, 5), (3, 6)))  # a move the game refused: a knight two up and one across is no knight
    bishop = CHESS.kinds["N"]
    wrong = replace(CHESS, kinds={**CHESS.kinds, "N": replace(bishop, steps=(*bishop.steps, Step(((1, 1),), rides=1)))})
    assert shown.checked(wrong).spurious == 1 and shown.checked(CHESS).spurious == 0


def test_what_the_game_showed_is_kept_as_her_rules_learn_and_read_back() -> None:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    rules.shown = WhatTheGameShowed()
    rules.offered(MIDDLE, (2, 5), {m.end for m in moves(CHESS, MIDDLE) if m.start == (2, 5)}, family=CHESS)
    rules.refused(MIDDLE, Move((2, 5), (3, 6)), family=CHESS)
    again = WhatTheGameShowed.from_bytes(rules.shown.as_bytes())
    assert len(again.offers) == 1 and len(again.refusals) == 1
    assert again.checked(CHESS).answers == 2

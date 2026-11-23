"""A candidate judgement is kept on how its match's games really ended, not on the material it steered toward.

From the suggestions of 23 November 2026: her rehearsal turned each drawn or unfinished game of a match into a result
read from the material where it stopped, in her present worths. A candidate could be kept for steering toward what
her judgement already likes without winning more. Where her model knows how a game ends, an unfinished game now
counts as no win for either side; only where she knows no ending is material the one sign, and it stays.
"""
from __future__ import annotations

from dataclasses import replace

from core.agency.searching_by_kind import first_weights
from core.agency.what_a_kind_is_worth import _match_scored
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

#: Four unfinished games, the candidate a queen up where each stopped (it plays first in the even ones).
QUEEN = list(CHESS.kinds).index("Q")
AHEAD = [0] * len(CHESS.kinds)
AHEAD[QUEEN] = 1
BEHIND = [0] * len(CHESS.kinds)
BEHIND[QUEEN] = -1
MATCH = {"results": [0.5, 0.5, 0.5, 0.5], "final_counts": [AHEAD, BEHIND, AHEAD, BEHIND]}


def test_where_the_game_has_an_ending_unfinished_games_are_no_ones_win() -> None:
    results, ends_known = _match_scored(dict(MATCH), first_weights(CHESS), CHESS)
    assert ends_known and list(results) == [0.5, 0.5, 0.5, 0.5]


def test_where_no_ending_is_known_material_is_the_sign_there_is() -> None:
    endless = replace(CHESS, kinds={letter: replace(kind, royal=False) for letter, kind in CHESS.kinds.items()})
    results, ends_known = _match_scored(dict(MATCH), first_weights(endless), endless)
    assert not ends_known and all(points > 0.5 for points in results)

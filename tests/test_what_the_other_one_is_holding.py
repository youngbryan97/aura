"""Reading somebody else's plan with the organ she reads her own with.

The other side promotes on her back row, so what predicts them doing well is a
gap in it. She never has to be told that. She watches her own losses.
"""

from __future__ import annotations

import random

from core.cognition.what_the_other_one_is_holding import (
    acts_that_take_it_away,
    what_the_other_one_is_holding,
)
from tests.two_sided_game_support import (
    ACTS,
    MINE,
    MINE_MOVES,
    THEIRS,
    THEIR_MOVES,
    back_row_whole,
    board_of,
)


def _positions(seed: int = 4, how_many: int = 40):
    """Positions and whether the other side did well from them."""
    roll = random.Random(seed)
    seen = []
    for _ in range(how_many):
        pieces: dict[tuple[int, int], tuple[str, bool]] = {}
        gap = roll.random() < 0.5
        columns = [0, 2, 4, 6]
        if gap:
            columns.remove(roll.choice(columns))
        for col in columns:
            pieces[(7, col)] = (MINE, False)
        pieces[(roll.randrange(1, 4), roll.randrange(4))] = (MINE, roll.random() < 0.3)
        for col in roll.sample([1, 3, 5, 7], roll.randrange(2, 5)):
            pieces[(6, col)] = (THEIRS, False)
        seen.append((board_of(pieces), gap))
    return seen


def test_she_works_out_what_they_are_playing_for_from_her_own_losses() -> None:
    theirs = what_the_other_one_is_holding(_positions())
    assert theirs is not None
    # What it settles on is a GAP in her back row, which is where they promote
    # — said as the absence of a piece, because absence is a property too.
    # Not a perfect predictor: the gap is in a different column each game and
    # this names one column, so it catches the quarter of games that leave
    # that square open. Claiming more would be claiming what is not there.
    assert theirs.name.startswith("it is not so that it holds a (7,")
    assert MINE in theirs.name
    assert theirs.tells_them_apart > 0.4, theirs.name
    # A position with her back row whole is not one they do well from.
    whole = board_of(
        {(7, col): (MINE, False) for col in (0, 2, 4, 6)}
        | {(6, 1): (THEIRS, False), (2, 3): (MINE, True)}
    )
    assert not theirs.holds(whole)


def test_breaking_it_only_counts_when_they_cannot_put_it_back() -> None:
    theirs = what_the_other_one_is_holding(_positions())
    assert theirs is not None
    board = board_of(
        {
            (7, 0): (MINE, False),
            (7, 2): (MINE, False),
            (7, 4): (MINE, False),
            (7, 6): (MINE, False),
            (4, 3): (MINE, True),
            (5, 6): (THEIRS, False),
        }
    )
    ranked = acts_that_take_it_away(
        board,
        acts=ACTS,
        step=MINE_MOVES,
        theirs=theirs,
        their_acts=ACTS,
        their_step=THEIR_MOVES,
        keeps=back_row_whole,
    )
    assert ranked
    # Whatever she does first must not be a move that gives up her own plan.
    assert back_row_whole(MINE_MOVES(board, ranked[0].act))
    put_back = [one for one in ranked if one.breaks_it and one.they_can_put_it_back]
    taken = [one for one in ranked if one.takes_it_away]
    for one in taken:
        for other in put_back:
            assert ranked.index(one) < ranked.index(other), one.describe()


def test_nothing_predicting_their_luck_is_an_answer_she_can_use() -> None:
    """When they are not holding anything, she gets on with her own plan."""
    roll = random.Random(9)
    coin = [
        (board_of({(7, 0): (MINE, False), (6, 1): (THEIRS, False)}), roll.random() < 0.5)
        for _ in range(30)
    ]
    assert what_the_other_one_is_holding(coin) is None

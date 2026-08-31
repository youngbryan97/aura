"""Picking the ending she can actually bring about.

Against a strong opponent, out-capturing it never happens however many games
are played. Leaving it with nothing to do happens whenever the near back row
stays whole. Both are ways of winning; only one of them is a plan.
"""

from __future__ import annotations

import random


from core.cognition.which_way_to_win import which_way_to_win
from tests.two_sided_game_support import MINE, THEIRS, board_of

FORFEIT, OUTGUNNED, LOST = "they cannot move", "she takes them all", "she runs out"


def _a_run(roll: random.Random, back_row_kept: bool) -> tuple[list, str]:
    """A game, said as the positions passed through and how it finished."""
    places = []
    for _turn in range(6):
        pieces: dict[tuple[int, int], tuple[str, bool]] = {}
        keep = (0, 2, 4, 6) if back_row_kept else roll.sample([0, 2, 4, 6], 2)
        for col in keep:
            pieces[(7, col)] = (MINE, False)
        pieces[(roll.randrange(2, 5), roll.randrange(4))] = (MINE, True)
        for col in roll.sample([1, 3, 5, 7], roll.randrange(2, 5)):
            pieces[(6, col)] = (THEIRS, False)
        places.append(board_of(pieces))
    return places, FORFEIT if back_row_kept else LOST


def _runs(seed: int = 5, how_many: int = 24):
    roll = random.Random(seed)
    return [_a_run(roll, roll.random() < 0.5) for _ in range(how_many)]


def test_she_plays_for_the_ending_she_has_a_route_to() -> None:
    ranked = which_way_to_win(
        {
            FORFEIT: lambda finish: finish == FORFEIT,
            OUTGUNNED: lambda finish: finish == OUTGUNNED,
            LOST: lambda finish: finish == LOST,
        },
        _runs(),
    )
    best = ranked[0]
    assert best.name == FORFEIT, [one.describe() for one in ranked]
    assert best.can_steer_to_it
    assert best.by_holding is not None
    # What it settles on is a piece of hers on the near back row, which is
    # what the person said they were holding. Not a perfect predictor, because
    # some of the losing games happened to keep that square too, and claiming
    # otherwise would be claiming more than the games show.
    assert best.by_holding.name.startswith("it holds a (7,")
    assert MINE in best.by_holding.name
    assert best.by_holding.tells_them_apart > 0.7


def test_an_ending_that_never_comes_is_not_a_plan() -> None:
    ranked = which_way_to_win(
        {
            FORFEIT: lambda finish: finish == FORFEIT,
            OUTGUNNED: lambda finish: finish == OUTGUNNED,
        },
        _runs(),
    )
    never = next(one for one in ranked if one.name == OUTGUNNED)
    assert never.ended_this_way == 0
    assert not never.can_steer_to_it
    assert ranked.index(never) > 0


def test_a_rare_ending_she_can_steer_to_beats_a_common_one_she_cannot() -> None:
    # Two common endings that look exactly alike from the inside, so nothing
    # about the places she passes through says which one is coming. And one
    # rare ending that announces itself.
    runs = []
    for _ in range(15):
        runs.append(([{"a": 0} for _ in range(4)], "coin"))
    for _ in range(15):
        runs.append(([{"a": 0} for _ in range(4)], "toss"))
    for _ in range(6):
        runs.append(([{"a": 9} for _ in range(4)], "earned"))
    ranked = which_way_to_win(
        {
            "coin": lambda finish: finish == "coin",
            "toss": lambda finish: finish == "toss",
            "earned": lambda finish: finish == "earned",
        },
        runs,
    )
    assert ranked[0].name == "earned"
    assert ranked[0].how_often < min(one.how_often for one in ranked[1:])
    # The other two are not unsteerable — knowing there is no nine does raise
    # the chance of each of them, and saying otherwise would be false. What
    # the ordering promises is that the ending she can see coming most clearly
    # goes first, whatever the counts say.
    best = ranked[0].by_holding
    assert best is not None
    for other in ranked[1:]:
        assert other.by_holding is None or (
            other.by_holding.tells_them_apart < best.tells_them_apart
        )

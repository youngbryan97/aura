"""A property she commits to, and the moves it removes before she looks.

Watched beside somebody good at a thing, the difference is not that they search
further. It is that most of what she considers, they never consider at all.

A person clearing 2048 in 989 moves held one thing true for the whole game: the
largest tile stays in its corner. Two of the four directions evict it, so they
chose between two moves where she weighs four — and the ordering that follows
from it is what makes merges cascade instead of scatter.

Nothing here names a corner or a board. The properties come from the same
algebra everything else is written in, and which one is worth holding is
learned from how things went.
"""

from __future__ import annotations

import random

from core.cognition.something_she_keeps_true import (
    every_property_of,
    how_well_it_predicts,
    the_one_worth_holding,
    what_to_hold_now,
    what_it_rules_out,
)
from core.perception.what_is_there import Arrangement, Cell


def _board(rows: list[list[int]]) -> Arrangement:
    return Arrangement(
        rows=len(rows),
        columns=len(rows[0]),
        cells=tuple(
            Cell(row=r, column=c, says=str(value), at=(c * 0.1, r * 0.1))
            for r, row in enumerate(rows)
            for c, value in enumerate(row)
            if value
        ),
    )


def _slid(row: tuple[int, ...]) -> tuple[int, ...]:
    kept = [one for one in row if one]
    out, at = [], 0
    while at < len(kept):
        if at + 1 < len(kept) and kept[at] == kept[at + 1]:
            out.append(kept[at] * 2)
            at += 2
        else:
            out.append(kept[at])
            at += 1
    return tuple(out + [0] * (len(row) - len(out)))


def _grid(board: Arrangement) -> list[list[int]]:
    out = [[0] * board.columns for _ in range(board.rows)]
    for cell in board.cells:
        out[cell.row][cell.column] = int(cell.says)
    return out


def _moved(board: Arrangement, act: str) -> Arrangement:
    grid = _grid(board)
    if act == "left":
        grid = [list(_slid(tuple(row))) for row in grid]
    elif act == "right":
        grid = [list(reversed(_slid(tuple(reversed(row))))) for row in grid]
    elif act == "up":
        columns = [list(_slid(tuple(one))) for one in zip(*grid)]
        grid = [list(one) for one in zip(*columns)]
    elif act == "down":
        columns = [list(reversed(_slid(tuple(reversed(one))))) for one in zip(*grid)]
        grid = [list(one) for one in zip(*columns)]
    return _board(grid)


ACTS = ["left", "right", "up", "down"]


def _watched(seed: int = 3, how_many: int = 60, corner: tuple[int, int] = (3, 3)):
    """States that differ only in WHERE the big one is.

    Every state has one, so "it holds a 256" is true of all of them and tells
    nothing apart. Only its position separates the good from the bad, which is
    the discrimination actually being claimed. Data where the good states are
    the ones that have a big tile at all would be answered correctly by "it has
    a big tile", and would prove nothing about corners.
    """
    roll = random.Random(seed)
    seen = []
    for _turn in range(how_many):
        grid = [[0] * 4 for _ in range(4)]
        in_the_corner = roll.random() < 0.5
        for _put in range(roll.randrange(4, 9)):
            grid[roll.randrange(4)][roll.randrange(4)] = roll.choice(
                [2, 4, 8, 16, 32, 64, 128]
            )
        if in_the_corner:
            grid[corner[0]][corner[1]] = 256
        else:
            grid[corner[0]][corner[1]] = 0
            grid[roll.randrange(1, 3)][roll.randrange(1, 3)] = 256
        seen.append((_board(grid), in_the_corner))
    return seen


def test_she_finds_the_property_that_tells_the_good_states_from_the_bad():
    holding = the_one_worth_holding(_watched())
    assert holding is not None
    assert "largest" in holding.name
    assert holding.tells_them_apart > 0.5


def test_and_it_removes_the_moves_that_break_it_before_she_looks_ahead():
    """Of four directions two evict the corner. Excluding them halves the tree
    at every level, before any looking ahead has happened."""
    holding = the_one_worth_holding(_watched())
    here = _board([[2, 4, 0, 0], [0, 8, 0, 0], [0, 0, 16, 0], [0, 0, 0, 256]])
    keeps, breaks = what_it_rules_out(holding, here, ACTS, expect=_moved)
    assert set(keeps) and set(breaks)
    assert set(keeps) | set(breaks) <= set(ACTS)
    assert len(keeps) < len(ACTS)


def test_a_property_true_of_everything_is_not_worth_holding():
    """Superstition excludes moves for no reason. Nothing is the honest answer
    where a property holds as often in the bad states as the good."""
    roll = random.Random(11)
    seen = [
        (_board([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), roll.random() < 0.5)
        for _turn in range(40)
    ]
    assert the_one_worth_holding(seen) is None


def test_nothing_is_held_on_no_evidence():
    assert the_one_worth_holding([]) is None


def test_when_it_is_already_broken_every_move_is_still_offered():
    """Restoring it is then what the acts are for, and refusing to move would
    be holding a thing that is not true."""
    holding = the_one_worth_holding(_watched())
    broken = _board([[256, 0, 0, 0], [0, 8, 0, 0], [0, 0, 16, 0], [0, 0, 0, 2]])
    keeps, breaks = what_it_rules_out(holding, broken, ACTS, expect=_moved)
    assert keeps
    assert breaks == ()


def test_the_properties_come_from_the_shape_not_from_a_list_of_games():
    """Nothing here names a corner, a board, or a tile."""
    named = [name for name, _holds in every_property_of(_board([[2, 4], [8, 16]]))]
    assert named
    assert not any("2048" in one or "tile" in one or "game" in one for one in named)


def test_how_it_weighs_is_visible():
    weighed = how_well_it_predicts(_watched())
    assert weighed
    said = str(weighed[0])
    assert "went well when it held" in said


def test_she_lets_go_when_it_stops_telling_them_apart() -> None:
    """The rebound: a broken anchor is re-chosen, not fought for."""
    roll = random.Random(11)

    def watched(anchor_at_the_bottom: bool) -> list[tuple[Arrangement, bool]]:
        out = []
        for _ in range(50):
            grid = [[0] * 4 for _ in range(4)]
            good = roll.random() < 0.5
            for _ in range(roll.randrange(4, 9)):
                grid[roll.randrange(4)][roll.randrange(4)] = roll.choice(
                    [2, 4, 8, 16, 32, 64]
                )
            row = (3 if anchor_at_the_bottom else 0) if good else roll.randrange(1, 3)
            grid[row][3 if good else roll.randrange(1, 3)] = 512
            out.append((_board(grid), good))
        return out

    early = watched(False)
    holding = the_one_worth_holding(early)
    assert holding is not None

    kept, why = what_to_hold_now(holding, early)
    assert kept is not None and kept.name == holding.name, why

    moved, why = what_to_hold_now(holding, watched(True))
    assert moved is not None
    assert moved.name != holding.name, why
    assert "stopped telling them apart" in why


def test_the_weight_on_an_edge_is_a_property_she_can_hold() -> None:
    named = {name for name, _ in every_property_of(_board([[2, 0, 0, 8]] * 4))}
    assert "the largest things live along its last column" in named


def test_holding_nothing_takes_up_the_best_one() -> None:
    roll = random.Random(3)
    seen = []
    for _ in range(40):
        grid = [[0] * 4 for _ in range(4)]
        good = roll.random() < 0.5
        for _ in range(roll.randrange(3, 8)):
            grid[roll.randrange(4)][roll.randrange(4)] = roll.choice([2, 4, 8, 16])
        grid[3 if good else 1][3 if good else 1] = 256
        seen.append((_board(grid), good))
    took, why = what_to_hold_now(None, seen)
    assert took is not None, why
    assert took.tells_them_apart > 0

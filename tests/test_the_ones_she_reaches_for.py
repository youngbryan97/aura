"""The few acts she takes, found by trying rather than by looking.

Somebody clearing 2048 pressed two of the four keys almost exclusively and a
third only when the board left them nothing else. That is a disposition of
theirs rather than a fact about the board, which is why no amount of studying
the board finds it.

The game here is the real one, played out, so the numbers below are measured
rather than asserted.
"""

from __future__ import annotations

import random
import statistics

from core.cognition.the_ones_she_reaches_for import TheOnesSheReachesFor

ACTS = ["left", "right", "up", "down"]


def _slid(row):
    kept = [one for one in row if one]
    out, at = [], 0
    while at < len(kept):
        if at + 1 < len(kept) and kept[at] == kept[at + 1]:
            out.append(kept[at] * 2)
            at += 2
        else:
            out.append(kept[at])
            at += 1
    return out + [0] * (len(row) - len(out))


def _move(grid, act):
    rows = [list(one) for one in grid]
    if act == "left":
        return [_slid(one) for one in rows]
    if act == "right":
        return [list(reversed(_slid(list(reversed(one))))) for one in rows]
    if act == "up":
        return [
            list(one)
            for one in zip(*[_slid(list(one)) for one in zip(*rows, strict=False)], strict=False)
        ]
    return [
        list(one)
        for one in zip(
            *[
                list(reversed(_slid(list(reversed(list(one))))))
                for one in zip(*rows, strict=False)
            ],
            strict=False,
        )
    ]


def _fresh(roll):
    grid = [[0] * 4 for _ in range(4)]
    for _ in range(2):
        grid[roll.randrange(4)][roll.randrange(4)] = 2
    return grid


def _largest(grid):
    return max(max(one) for one in grid)


def _at_random(seed):
    roll = random.Random(seed)
    grid = _fresh(roll)
    for _ in range(2000):
        legal = [one for one in ACTS if _move(grid, one) != grid]
        if not legal:
            break
        grid = _move(grid, roll.choice(legal))
        empty = [(r, c) for r in range(4) for c in range(4) if not grid[r][c]]
        if empty:
            row, col = roll.choice(empty)
            grid[row][col] = roll.choice([2] * 9 + [4])
    return _largest(grid)


def _leaning(seed, runs=20, stretch=60):
    roll = random.Random(seed)
    habit = TheOnesSheReachesFor()
    got = []
    for _run in range(runs):
        grid = _fresh(roll)
        habit.start_a_stretch(ACTS)
        since = 0
        for _ in range(2000):
            legal = [one for one in ACTS if _move(grid, one) != grid]
            if not legal:
                break
            was = _largest(grid)
            grid = _move(grid, habit.which_to_take(legal, ACTS))
            empty = [(r, c) for r in range(4) for c in range(4) if not grid[r][c]]
            if empty:
                row, col = roll.choice(empty)
                grid[row][col] = roll.choice([2] * 9 + [4])
            habit.went(_largest(grid) - was)
            since += 1
            if since >= stretch:
                habit.end_the_stretch()
                habit.start_a_stretch(ACTS)
                since = 0
        habit.end_the_stretch()
        got.append(_largest(grid))
    return got, habit


def test_she_ends_up_reaching_for_some_of_them_and_not_all() -> None:
    """How many is not decided in advance. On one run of the game she settles
    on a pair; on another a single act measured best on its own and that is
    what she reaches for. What is claimed is that she narrows."""
    narrowed = []
    for seed in range(4):
        _got, habit = _leaning(seed)
        assert habit.settled(ACTS)
        reaches = habit.the_ones_that_paid(ACTS)
        assert reaches, "everything has been tried, so something should be preferred"
        assert set(reaches) < set(ACTS), reaches
        narrowed.append(reaches)
    assert any(len(one) == 2 for one in narrowed), narrowed


def test_leaning_on_them_beats_taking_whatever_is_going() -> None:
    """Measured, over runs, on the game itself."""
    random_runs = [_at_random(seed) for seed in range(40)]
    hers: list[int] = []
    for seed in range(6):
        # The first runs are the ones she is trying things in, and are not
        # what leaning on the good ones is worth.
        got, _habit = _leaning(seed)
        hers += got[6:]
    assert statistics.mean(hers) > statistics.mean(random_runs) * 1.4, (
        f"hers {statistics.mean(hers):.1f} against {statistics.mean(random_runs):.1f}"
    )
    assert max(hers) > max(random_runs)


def test_everything_gets_a_turn_before_anything_is_settled() -> None:
    habit = TheOnesSheReachesFor()
    assert habit.the_ones_that_paid(ACTS) == (), "nothing is settled on no evidence"
    seen = set()
    for _ in range(len(ACTS) + len(ACTS) * (len(ACTS) - 1) // 2):
        seen.add(habit.start_a_stretch(ACTS))
        habit.went(1.0)
        habit.end_the_stretch()
    assert habit.settled(ACTS)
    assert len(seen) == 10, "four on their own and six pairs"


def test_being_forced_off_them_is_the_habit_working() -> None:
    habit = TheOnesSheReachesFor()
    habit.leaning_on = ("down",)
    assert habit.which_to_take(["down", "left"], ACTS) == "down"
    assert habit.which_to_take(["left", "up"], ACTS) in {"left", "up"}
    assert habit.which_to_take([], ACTS) == ""


def test_a_habit_is_worth_carrying_between_sittings() -> None:
    _got, habit = _leaning(0)
    again = TheOnesSheReachesFor.from_memory(habit.as_memory())
    assert again.the_ones_that_paid(ACTS) == habit.the_ones_that_paid(ACTS)
    assert TheOnesSheReachesFor.from_memory(habit.as_memory(), trust=0.0).the_ones_that_paid(
        ACTS
    ) == ()

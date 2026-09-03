"""Solving a thing by assuming a smaller one of it is already solved.

Show f(1) works. Assume f(n-1) works. Nobody traces it, and tracing it is the
mistake: three disks is seven moves, ten disks is a thousand and twenty three,
and somebody who follows the moves has understood nothing while somebody who
accepts the assumption has understood all of it.
"""

from __future__ import annotations

from core.cognition.the_same_problem_one_size_down import solve_by_the_size_below


def _hanoi(size: int):
    """The problem as (how many disks, from, to, spare)."""
    return solve_by_the_size_below(
        (size, "a", "c", "b"),
        smallest=lambda one: one[0] <= 1,
        answer_outright=lambda one: [(one[1], one[2])],
        one_size_down=lambda one: [
            (one[0] - 1, one[1], one[3], one[2]),
            (one[0] - 1, one[3], one[2], one[1]),
        ],
        put_together=lambda one, below: [
            *below[0],
            (one[1], one[2]),
            *below[1],
        ],
    )


def test_it_works_the_whole_thing_out_without_walking_it() -> None:
    got = _hanoi(3)
    assert got.settled, got.describe()
    assert len(got.steps) == 7
    assert got.deep == 2, got.describe()
    assert got.steps[0] == ("a", "c")
    assert got.steps[3] == ("a", "c")


def test_the_plan_is_right_at_every_size() -> None:
    """Two to the n minus one, which is what the recursion says it must be —
    and the moves are legal, which is what it must also be."""
    for size in range(1, 11):
        got = _hanoi(size)
        assert got.settled
        assert len(got.steps) == 2**size - 1, size
        rods = {"a": list(range(size, 0, -1)), "b": [], "c": []}
        for was, now in got.steps:
            disk = rods[was].pop()
            assert not rods[now] or rods[now][-1] > disk, "a bigger disk on a smaller"
            rods[now].append(disk)
        assert rods["c"] == list(range(size, 0, -1)), size


def test_a_thing_that_does_not_get_smaller_is_said_so() -> None:
    """Rather than run until something stops it, which is how a problem this
    cannot solve gets mistaken for a hard one."""
    got = solve_by_the_size_below(
        5,
        smallest=lambda one: one == 0,
        answer_outright=lambda one: [],
        one_size_down=lambda one: [one],
        put_together=lambda one, below: [],
        deepest=20,
    )
    assert not got.settled
    assert "does not get smaller" in got.why


def test_a_bottom_it_cannot_answer_is_said_so() -> None:
    got = solve_by_the_size_below(
        3,
        smallest=lambda one: one <= 0,
        answer_outright=lambda one: 1 / 0,
        one_size_down=lambda one: [one - 1],
        put_together=lambda one, below: list(below[0]),
    )
    assert not got.settled


def test_it_is_not_only_about_disks() -> None:
    """Adding up a list by adding up a shorter one. The same two lines."""
    got = solve_by_the_size_below(
        [4, 7, 2, 9],
        smallest=lambda one: len(one) <= 1,
        answer_outright=lambda one: list(one),
        one_size_down=lambda one: [one[1:]],
        put_together=lambda one, below: [one[0] + below[0][0]],
    )
    assert got.settled
    assert got.steps == (22,)

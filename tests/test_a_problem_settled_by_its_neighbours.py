"""A constraint problem worked by ruling out, by search, and by many attempts.

From "Reasoning with Neural Cellular Automata" (arXiv 2609.36126): each part
settles by its neighbours, attempts that are stuck break more for a while to
get out, and many attempts are kept different. What may be claimed is kept
apart from what was only found. See core/reasoning/settled_by_neighbours.py.
"""
from __future__ import annotations

import random

import pytest

from core.reasoning import settled_by_neighbours as sbn
from core.reasoning.grid_puzzles import describe_grid_answer, parse_grid_puzzle, solve_grid_puzzle
from core.reasoning.settled_by_neighbours import a_problem, narrowed, settle

pytestmark = pytest.mark.unit

EASY = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
#: Arto Inkala's, published as the hardest; proved unique here in 2.4 seconds.
HARDEST = "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4.."


def _queens(n: int):
    return a_problem(
        {f"q{i}": range(n) for i in range(n)},
        [
            ((f"q{i}", f"q{j}"), lambda a, b, k=j - i: a != b and abs(a - b) != k)
            for i in range(n)
            for j in range(i + 1, n)
        ],
    )


def _holds_everywhere(problem, values) -> bool:
    return problem.broken(values) == 0


def test_ruling_out_alone_settles_an_easy_grid_and_proves_it():
    settled = solve_grid_puzzle(parse_grid_puzzle(EASY), budget_s=5.0)
    assert settled.proven and len(settled.solutions) == 1
    assert "ruling values out" in settled.why


def test_search_with_ruling_out_proves_the_hardest_grid_unique():
    settled = solve_grid_puzzle(parse_grid_puzzle(HARDEST), budget_s=20.0)
    assert settled.proven and len(settled.solutions) == 1
    grid = settled.solution
    for r in range(9):
        assert {grid[f"r{r}c{c}"] for c in range(9)} == set(range(1, 10))


def test_a_grid_with_more_than_one_answer_is_said_to_have_more_than_one():
    puzzle = parse_grid_puzzle("solve this sudoku: " + "." * 72 + "123456789")
    settled = solve_grid_puzzle(puzzle, budget_s=10.0)
    assert settled.proven and len(settled.solutions) == 2
    assert "not unique" in describe_grid_answer(puzzle, settled)


def test_a_contradiction_is_proved_to_have_no_solution():
    puzzle = parse_grid_puzzle("solve: 11" + "." * 79)
    settled = solve_grid_puzzle(puzzle, budget_s=5.0)
    assert settled.proven and settled.solution is None and settled.solutions == []
    assert describe_grid_answer(puzzle, settled).startswith("No filled grid")


def test_every_solution_of_six_queens_is_found_and_checked():
    """Six queens has exactly four solutions; search finds two, proving more than one."""
    problem = _queens(6)
    settled = settle(problem, budget_s=5.0)
    assert settled.proven and len(settled.solutions) == 2
    assert all(_holds_everywhere(problem, one) for one in settled.solutions)


def test_attempts_find_what_search_could_not_finish_and_claim_no_proof(monkeypatch):
    """With the search out of time, the ensemble still answers, and says what it cannot say."""
    monkeypatch.setattr(sbn, "_searched", lambda problem, deadline, want=2: ([], False))
    problem = _queens(12)
    settled = settle(problem, budget_s=3.0, find_every=False)
    assert settled.solution is not None and _holds_everywhere(problem, settled.solution)
    assert not settled.proven and "uniqueness not proven" in settled.why
    assert settled.attempts >= 1


def test_an_attempt_that_stopped_improving_breaks_more_for_a_while(monkeypatch):
    """The noise: a variable in doubt in a stuck attempt takes a value at random, not the rule's."""
    problem = a_problem({"a": (0, 1), "b": (0, 1)}, [(("a", "b"), lambda x, y: x != y)])
    width = problem.width()
    picks: list[str] = []
    monkeypatch.setattr(sbn, "_settle_one", lambda *a, **k: picks.append("rule"))
    calm = sbn._Attempt({"a": 0, "b": 0}, broken=1, fewest=1, since_better=0)
    sbn._sweep(problem, calm, ["a", "b"], random.Random(0), width)
    assert picks == ["rule", "rule"], "an attempt still improving settles by the rule"
    picks.clear()
    stuck = sbn._Attempt({"a": 0, "b": 0}, broken=1, fewest=1, since_better=width)
    sbn._sweep(problem, stuck, ["a", "b"], random.Random(0), width)
    assert picks == [], "a stuck attempt whose every variable is in doubt moves at random"


def test_the_kept_half_spans_niches_before_copies():
    one = sbn._Attempt({"a": 0}, broken=0)
    same = sbn._Attempt({"a": 0}, broken=0)
    other = sbn._Attempt({"a": 1}, broken=1)
    kept = sbn._kept_across_niches([one, same, other], ["a"], 2)
    assert sorted(attempt.values["a"] for attempt in kept) == [0, 1]


def test_ruling_out_strikes_what_a_neighbour_leaves_no_room_for():
    problem = a_problem(
        {"x": (1,), "y": (1, 2), "z": (1, 2, 3)},
        [(("x", "y"), lambda a, b: a != b), (("y", "z"), lambda a, b: a != b), (("x", "z"), lambda a, b: a != b)],
    )
    narrow = narrowed(problem)
    assert narrow.domains == {"x": (1,), "y": (2,), "z": (3,)}


def test_a_grid_is_read_only_where_it_is_one():
    assert parse_grid_puzzle("my number is 1234\nand my pin is 4321") is None
    assert parse_grid_puzzle("9" * 81) is None, "a full grid is not a puzzle"
    assert parse_grid_puzzle("1.34\n.4.2\n4.2.\n2.4.") is None, "four by four only when asked for"
    assert parse_grid_puzzle("solve this grid\n1.34\n.4.2\n4.2.\n2.4.").side == 4
    lined = "\n".join(EASY[at:at + 9] for at in range(0, 81, 9))
    assert parse_grid_puzzle(lined).cells == parse_grid_puzzle(EASY).cells


def test_the_chat_reads_a_pasted_grid_through_its_observable():
    import asyncio

    from core.brain import observable_registry  # noqa: F401 - registers the default readings
    from core.brain.observable_grounding import OBSERVABLES

    reading = next(one for one in OBSERVABLES if one.name == "grid_puzzle_solution")
    assert reading.example_failures() == []
    said = asyncio.run(reading.read("solve this sudoku\n" + EASY))
    assert "proven the only solution" in said
    assert said.splitlines()[0] == "5 3 4 | 6 7 8 | 9 1 2"

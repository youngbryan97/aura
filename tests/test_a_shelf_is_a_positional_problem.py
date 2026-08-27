"""A shelf of things is the same problem as a table of people.

The module's own description names "seating, queues, shelves, finishing order,
house-order puzzles" — and its entity reader accepted only Capitalised words,
so a shelf holding lowercase things could not be parsed at all. Three core row
premises had no pattern either: "at one end", "exactly two crates between X and
Y", and "the crate second from the left". And the question forms were all about
people sitting, so "which crate holds what" was not a question.

LIVE, 2026-08-27: a five-crate puzzle with five premises and one solution. The
parse declined, and the model answered with a truncated "Step 1: Pos 2 = Cable"
and stopped.
"""

from __future__ import annotations

from itertools import permutations

import pytest

from core.reasoning.positional_constraints import (
    answer_positional_problem,
    describe_positional_answer,
    parse_positional_problem,
)

_PUZZLE = (
    "Five crates on a shelf, left to right, each holds a different one of: "
    "bolts, cable, dye, epoxy, foam. The cable is somewhere left of the epoxy. "
    "The dye is at one end. There are exactly two crates between the bolts and "
    "the foam. The epoxy is not next to the foam. The crate second from the "
    "left holds cable. Which crate holds what, and is the answer forced or are "
    "there several?"
)


def _brute_force() -> list[tuple[str, ...]]:
    """The same puzzle solved independently, so the test checks the answer."""
    items = ("bolts", "cable", "dye", "epoxy", "foam")
    found = []
    for order in permutations(items):
        at = {name: index for index, name in enumerate(order)}
        if at["cable"] >= at["epoxy"]:
            continue
        if at["dye"] not in (0, 4):
            continue
        if abs(at["bolts"] - at["foam"]) != 3:
            continue
        if abs(at["epoxy"] - at["foam"]) == 1:
            continue
        if order[1] != "cable":
            continue
        found.append(order)
    return found


def test_the_puzzle_has_exactly_one_solution() -> None:
    """The fixture is only useful if it is actually forced."""
    assert len(_brute_force()) == 1


def test_lowercase_things_are_entities() -> None:
    problem = parse_positional_problem(_PUZZLE)
    assert problem is not None
    assert set(problem.entities) == {"bolts", "cable", "dye", "epoxy", "foam"}
    assert problem.seats == 5
    assert not problem.cyclic


def test_every_premise_is_read() -> None:
    """Five premises. A puzzle parsed at two has more solutions than it should."""
    problem = parse_positional_problem(_PUZZLE)
    assert problem is not None
    assert len(problem.constraints) == 5


def test_the_answer_matches_the_brute_force() -> None:
    answer = answer_positional_problem(_PUZZLE)
    assert answer is not None
    told = describe_positional_answer(answer)
    expected = _brute_force()[0]
    for index, name in enumerate(expected, start=1):
        assert f"{index}. {name}" in told, told


def test_an_ordered_answer_reads_as_positions() -> None:
    """Joining five things with "and" reads as neighbours, not as an order."""
    told = describe_positional_answer(answer_positional_problem(_PUZZLE))
    assert "1. " in told and "5. " in told
    assert " and " not in told.split(":", 1)[1]


@pytest.mark.parametrize(
    "premise",
    [
        "The dye is at one end.",
        "There are exactly two crates between the bolts and the foam.",
        "The crate second from the left holds cable.",
    ],
)
def test_dropping_any_premise_loosens_the_puzzle(premise: str) -> None:
    """Each of the three new patterns has to be doing work.

    A pattern that parses but constrains nothing would leave every test above
    passing while the premise was silently ignored.
    """
    without = _PUZZLE.replace(premise, "")
    problem = parse_positional_problem(without)
    assert problem is not None
    assert len(problem.constraints) == 4
"""No answer is asserted for the loosened puzzles: fewer premises may still
force the same arrangement, and that would be a correct answer rather than a
failure."""

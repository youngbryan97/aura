"""Seating and order problems, worked out rather than described.

LIVE, 2026-08-20. Six people, a round table, four constraints. Asked twice,
she narrated twice and was wrong both times — once naming Emil correctly and
Dara's neighbours wrongly, once stating a layout in which Dara was "opposite
Ada" one line after Boris was. Offering a Python sandbox did not help: the
tools reached the turn and the model answered directly anyway.

The answers here are checked against exhaustive enumeration done separately.
"""

from __future__ import annotations

import pytest

from core.reasoning.positional_constraints import (
    answer_positional_problem,
    describe_positional_answer,
    parse_positional_problem,
)

THE_LIVE_PUZZLE = (
    "six people sit around a round table with six seats. Boris sits directly "
    "opposite Ada. Chen and Dara sit next to each other. Emil sits immediately "
    "clockwise of Ada. Chen is exactly two seats from Ada. Who sits opposite "
    "Chen, and who are Dara's two neighbours?"
)


def test_the_live_puzzle() -> None:
    answer = answer_positional_problem(THE_LIVE_PUZZLE)
    assert answer is not None
    described = describe_positional_answer(answer)
    assert "Opposite Chen: Emil." in described
    assert "Dara's neighbours: Ada and Chen." in described


def test_names_that_open_a_sentence_are_still_names() -> None:
    problem = parse_positional_problem(THE_LIVE_PUZZLE)
    assert problem is not None
    assert {"Boris", "Ada", "Chen", "Dara", "Emil"} <= set(problem.entities)


def test_an_occupant_the_problem_never_names_still_takes_a_seat() -> None:
    problem = parse_positional_problem(THE_LIVE_PUZZLE)
    assert problem is not None
    assert len(problem.entities) == 6


def test_what_is_asked_is_not_also_given() -> None:
    """"Who sits opposite Chen" states no premise."""
    problem = parse_positional_problem(THE_LIVE_PUZZLE)
    assert problem is not None
    assert len(problem.constraints) == 4
    assert all("Who" not in rule.text for rule in problem.constraints)


def test_a_relation_stated_after_both_names() -> None:
    """"Chen and Dara sit next to each other" was read as no relation."""
    problem = parse_positional_problem(THE_LIVE_PUZZLE)
    assert problem is not None
    assert any("next to each other" in rule.text for rule in problem.constraints)


def test_a_row_is_not_a_ring() -> None:
    row = (
        "Five friends stand in a row. Ana is next to Ben. Ben is exactly two "
        "places from Cara. Dev is next to Ana. Who are Ben's neighbours?"
    )
    problem = parse_positional_problem(row)
    assert problem is not None
    assert problem.cyclic is False


@pytest.mark.parametrize(
    "text",
    [
        "how are you doing today?",
        "what is 2 + 2",
        "",
        "Ada and Boris are friends who like chess.",
        "read /etc/hosts and tell me the first line",
        "six people sit around a round table. Who sits opposite Ada?",
    ],
)
def test_it_declines_what_it_cannot_settle(text: str) -> None:
    assert answer_positional_problem(text) is None


def test_it_never_asserts_an_answer_the_clues_do_not_determine() -> None:
    """Two constraints and six seats leave the neighbours open.

    This asserted None, and returning nothing reads as inability when the
    truth is that the clues do not decide. The stronger contract: it settles
    nothing it cannot settle, and says which possibilities remain.
    """
    loose = (
        "six people sit around a round table with six seats. Boris sits "
        "directly opposite Ada. Chen sits next to Dara. Who are Dara's two "
        "neighbours?"
    )
    answer = answer_positional_problem(loose)
    assert answer is not None
    assert answer.findings == ()
    assert len(answer.alternatives) == 1

    spoken = describe_positional_answer(answer)
    assert "not settled" in spoken
    assert "Chen" in spoken


def test_nothing_settled_says_nothing() -> None:
    assert describe_positional_answer(None) == ""


A_ROW_PROBLEM = (
    "Five runners finish in a line: Nils, Petra, Quinn, Rosa and Sven. Nils is "
    "exactly two places from Rosa. Quinn is not next to Nils. Quinn is exactly "
    "four places from Sven. Who are Rosa's neighbours?"
)


def test_a_row_problem_it_had_not_been_built_against() -> None:
    """Written after the solver, brute-forced separately: Petra and Quinn."""
    answer = answer_positional_problem(A_ROW_PROBLEM)
    assert answer is not None
    assert describe_positional_answer(answer) == "Rosa's neighbours: Petra and Quinn."


def test_the_answer_is_settled_even_when_the_order_is_not() -> None:
    """Two arrangements satisfy it and both give the same neighbours.

    What has to be unique is the ANSWER. Requiring one arrangement would
    have declined a question that is completely determined.
    """
    answer = answer_positional_problem(A_ROW_PROBLEM)
    assert answer is not None
    assert answer.arrangements == 2


def test_a_place_is_a_position_not_a_population() -> None:
    """"exactly two places from Rosa" read as a party of two.

    The count rule matched the distance phrase, so a five-runner problem
    parsed as two seats and was abandoned.
    """
    problem = parse_positional_problem(A_ROW_PROBLEM)
    assert problem is not None
    assert problem.seats == 5


@pytest.mark.parametrize(
    "possessive",
    ["Dara's", "Daras", "Dara’s", "Daras'"],
)
def test_a_possessive_is_the_same_person_however_it_is_typed(possessive: str) -> None:
    """LIVE: "Daras two neighbours" put a sixth person at a six-seat table
    and the answer came out wrong; "Rosas neighbours" made a sixth runner and
    the parse declined a problem it could settle."""
    question = (
        "six people sit around a round table with six seats. Boris sits directly "
        "opposite Ada. Chen and Dara sit next to each other. Emil sits immediately "
        "clockwise of Ada. Chen is exactly two seats from Ada. Who sits opposite "
        f"Chen, and who are {possessive} two neighbours?"
    )
    described = describe_positional_answer(answer_positional_problem(question))
    assert "Opposite Chen: Emil." in described
    assert "neighbours: Ada and Chen." in described


def test_a_name_that_merely_ends_in_s_survives() -> None:
    from core.reasoning.positional_constraints import _names

    assert "Nils" in _names("Nils is exactly two places from Rosa.")

"""A derived word that answers what it never saw, rather than recalling what it did.

A lookup table read off examples is a memory in the shape of a word: it refuses
every pair outside the examples, so nothing built on it reaches further than the
examples did. These pin the difference.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition.an_operation_that_generalises import (
    Expression,
    an_operation_that_generalises,
    read_back,
    written_down,
)
from core.cognition.widening_the_language import (
    DerivedOperation,
    an_operation_nobody_wrote,
)

HOW_FAR_APART = [(7, 3, 4), (9, 2, 7), (5, 5, 0), (2, 8, 6), (11, 4, 7), (6, 1, 5)]
THE_LARGER_LESS_ONE = [(7, 3, 6), (2, 9, 8), (4, 4, 3), (5, 1, 4), (8, 6, 7), (3, 10, 9)]


def test_a_rule_is_worked_out_rather_than_recorded():
    found = an_operation_that_generalises(HOW_FAR_APART)
    assert found is not None
    assert found(100, 37) == 63


@pytest.mark.parametrize("one,other", [(100, 37), (13, 91), (55, 55), (0, 8)])
def test_it_answers_pairs_nobody_showed_her(one, other):
    found = an_operation_that_generalises(HOW_FAR_APART)
    assert found(one, other) == abs(one - other)


def test_a_table_refuses_the_same_pairs():
    """The thing being fixed, stated as a test so it cannot come back quietly."""
    table = DerivedOperation(
        name="a table", does={(one, other): got for one, other, got in HOW_FAR_APART}
    )
    with pytest.raises(KeyError):
        table(100, 37)


def test_a_constant_is_solved_for_rather_than_searched():
    found = an_operation_that_generalises(THE_LARGER_LESS_ONE)
    assert found is not None
    assert found(100, 37) == 99
    assert found(13, 91) == 90


def test_nothing_is_returned_when_no_rule_survives_what_it_never_saw():
    noise = [(1, 2, 9), (3, 4, 2), (5, 6, 7), (7, 8, 1), (9, 1, 4), (2, 3, 8)]
    assert an_operation_that_generalises(noise) is None


def test_too_few_examples_to_hold_any_back_is_refused():
    assert an_operation_that_generalises(HOW_FAR_APART[:2]) is None


def test_the_production_path_prefers_the_rule_to_the_table():
    def rule(state):
        size = len(state)
        return tuple(abs(state[i] - state[size - 1 - i]) for i in range(size))

    states = [(1, 2, 3, 4), (5, 9, 2, 7), (8, 1, 6, 3),
              (4, 7, 9, 2), (6, 3, 1, 8), (2, 5, 7, 9)]
    found = an_operation_nobody_wrote(
        [(state, rule(state)) for state in states],
        kinds.WHERE_FROM["here"],
        kinds.WHERE_FROM["the far end"],
        already=[],
    )
    assert found is not None
    assert found.rule is not None
    assert found(100, 37) == 63


def test_a_worked_out_rule_survives_being_written_down_and_read_back():
    found = an_operation_that_generalises(HOW_FAR_APART)
    again = read_back(written_down(found))
    assert again is not None
    assert again(100, 37) == found(100, 37)


def test_nothing_outside_the_ways_of_combining_can_be_read_back():
    assert read_back({"kind": "os.system", "parts": []}) is None
    assert read_back({"kind": "added", "parts": []}) is None
    assert read_back("not a rule") is None


def test_a_shorter_rule_is_preferred_to_a_longer_one():
    found = an_operation_that_generalises(HOW_FAR_APART)
    longer = Expression(
        "how far apart they are",
        parts=(
            Expression("added", parts=(Expression("the first"),
                                       Expression("a fixed number", value=0))),
            Expression("the second"),
        ),
    )
    assert found.how_long() < longer.how_long()

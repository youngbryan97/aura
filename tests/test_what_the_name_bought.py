"""The inline-expansion control, asked where it means something.

Take the name away and leave the body, then ask again. If nothing changes, what
did the work was the term, and the term was going to be there either way.

Where it does not apply, and why that is the interesting part
------------------------------------------------------------
It does not apply to a head. Inline expansion means replacing every use of a
name with its body, and a positional term CANNOT contain a floor term — which
is what makes a head a head rather than an abbreviation. There is nothing on
the other side of the comparison.

The first version of this got a number out of it anyway: it compared where a
head appears in the positional enumeration against how many candidates the head
search walks, called sixteen worse than twelve, and refused two heads that
work. Two different things counted in two different units, and it read as
evidence because both were integers. That is the failure this file exists to
have on the record.

Where it does apply
-------------------
Forward, across families, in one unit. Having a term as a HEAD costs a shape at
every node of every term; having it as a LEAF costs one more thing to try in a
hole. Which is worth more is a question about the next family, so the next
families are what it is asked on.

Six seeds, five families each, same budget, same words, the term in exactly
one place at a time:

    4000  head 5 solved / 7,723 walked    leaf 5 / 8,927
    4001  head 5 / 10,129                 leaf 5 / 11,767
    4002  head 5 / 18,160                 leaf 5 / 21,116
    4003  head 5 / 11,042                 leaf 5 / 11,256
    4004  head 5 / 25,423                 leaf 4 / 6,436
    4005  head 5 / 6,584                  leaf 5 / 7,612

The name buys something on all six — fewer candidates on five, and one more
family solved on the sixth. It could have gone the other way, which is what
makes it a control rather than a formality.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.cognition.what_the_name_bought import (
    WhatTheNameBought,
    what_the_name_bought,
)


def _weighed(**kept) -> WhatTheNameBought:
    said = {
        "name": "a term",
        "solved_as_a_head": 5,
        "walked_as_a_head": 100,
        "solved_as_a_leaf": 5,
        "walked_as_a_leaf": 200,
        "over": 5,
    }
    said.update(kept)
    return WhatTheNameBought(**said)


def test_solving_more_beats_walking_less() -> None:
    """The order the two questions are asked in, and it is not arbitrary.

    A head that solves the same number of families for less is worth having.
    One that solves fewer is not, however cheap it was — a search that gives
    up early is cheap.
    """
    assert _weighed(solved_as_a_head=5, solved_as_a_leaf=4,
                    walked_as_a_head=9_000, walked_as_a_leaf=100
                    ).the_name_bought_something
    assert not _weighed(solved_as_a_head=4, solved_as_a_leaf=5,
                        walked_as_a_head=1, walked_as_a_leaf=9_000
                        ).the_name_bought_something


def test_a_tie_on_families_is_decided_by_what_it_cost() -> None:
    assert _weighed(walked_as_a_head=100, walked_as_a_leaf=200
                    ).the_name_bought_something
    assert not _weighed(walked_as_a_head=200, walked_as_a_leaf=100
                        ).the_name_bought_something
    assert not _weighed(walked_as_a_head=100, walked_as_a_leaf=100
                        ).the_name_bought_something


def test_it_asks_each_family_in_exactly_one_condition_at_a_time() -> None:
    asked: list[tuple[Any, bool]] = []

    def ask(family, as_a_head, body):
        asked.append((family, as_a_head))
        return (True, 10 if as_a_head else 20)

    found = what_the_name_bought("a term", object(), ["one", "two"], ask=ask)
    assert [one for one, _ in asked] == ["one", "two", "one", "two"]
    assert [head for _, head in asked] == [True, True, False, False]
    assert found.the_name_bought_something
    assert found.walked_as_a_head == 20 and found.walked_as_a_leaf == 40


def test_a_family_nothing_solves_costs_nothing_on_either_side() -> None:
    found = what_the_name_bought(
        "a term", object(), ["one"], ask=lambda f, h, b: (False, 0)
    )
    assert found.solved_as_a_head == found.solved_as_a_leaf == 0
    assert not found.the_name_bought_something


@pytest.mark.slow
def test_a_head_is_worth_more_than_its_own_body_as_a_leaf() -> None:
    """The measurement, at a size that runs in a test.

    It can go the other way. A head costs a shape at every node of every term,
    and a term that is only ever used once is paying that for nothing.
    """
    from tools.run_grown_against_reset_heads import head_or_leaf

    found = head_or_leaf(seed=4000, families=3, deepest=4)
    assert "why" not in found, found
    assert found["the_name_bought_something"], found

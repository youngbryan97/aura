"""Acts with their own limited supply, and acts that change what comes after.

Every move has its own count of uses — not a shared pool — so the strong move
runs out while the weak one is still there, and a fight is lost by somebody
who never lost a turn. And the moves that decide most games do no damage at
all: they change what the other side is ABLE to do for the rest of it.

The tests are API budgets and rate limits, because it is the same shape.
"""

from __future__ import annotations

from core.cognition.what_an_act_costs_beyond_now import (
    WhatEachActHasLeft,
    WhatItDoesToThem,
)


def _budgets() -> WhatEachActHasLeft:
    left = WhatEachActHasLeft()
    left.she_has("the expensive call", 4)
    left.she_has("the cheap call", 300)
    return left


def test_each_act_has_its_own_supply_and_not_a_shared_one() -> None:
    left = _budgets()
    for _ in range(4):
        left.she_used("the expensive call")
    assert not left.can_still("the expensive call")
    assert left.can_still("the cheap call"), "the other one is untouched"
    assert left.what_is_left(["the expensive call", "the cheap call"]) == (
        "the cheap call",
    )


def test_three_of_four_is_not_three_of_three_hundred() -> None:
    """The same number, and not the same situation."""
    left = _budgets()
    left.she_used("the expensive call")
    assert left.how_much_is_left("the expensive call") == 0.75
    for _ in range(297):
        left.she_used("the cheap call")
    assert left.how_much_is_left("the cheap call") == 0.01
    assert left.running_out(["the expensive call", "the cheap call"]) == (
        "the cheap call",
    )


def test_the_last_one_is_only_spent_on_what_it_was_saved_for() -> None:
    left = WhatEachActHasLeft()
    left.she_has("the last retry", 1)
    assert left.worth_saving("the last retry", for_what=10.0, this_is_worth=3.0)
    assert not left.worth_saving("the last retry", for_what=10.0, this_is_worth=12.0)

    plenty = WhatEachActHasLeft()
    plenty.she_has("a retry", 20)
    assert not plenty.worth_saving("a retry", for_what=10.0, this_is_worth=3.0), (
        "with plenty left she spends freely"
    )


def test_an_act_that_changes_nothing_now_can_be_the_best_act() -> None:
    """It scored as though nothing had happened, because in the state it
    produced, nothing had."""
    does = WhatItDoesToThem()
    does.it_left("throttle them", changing="how fast they go", by=0.5)
    assert does.worth_beyond_now("throttle them", turns_left=20.0) == 10.0
    assert does.worth_beyond_now("throttle them", turns_left=0.0) == 0.0


def test_lasting_things_are_worth_more_early_which_is_why_they_are_spent_early() -> None:
    does = WhatItDoesToThem()
    does.it_left("throttle them", changing="how fast they go", by=0.5)
    early = does.worth_beyond_now("throttle them", turns_left=30.0)
    late = does.worth_beyond_now("throttle them", turns_left=2.0)
    assert early > late * 10


def test_an_act_that_leaves_nothing_says_so() -> None:
    assert "changes nothing that outlasts" in WhatItDoesToThem().describe("a poke")

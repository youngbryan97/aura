"""A retry limit per call is not a budget.

Voyager's closure asked for a hierarchical BudgetContext: a root turn or task
budget, child budgets for model calls, tools and retries, and every nested
retry consuming the parent so it cannot escape it.

Three calls with three retries each is nine attempts, and the caller who
allowed three has no way to say so. What a caller can say is how much the
whole thing may cost.
"""
from __future__ import annotations

import pytest

from core.runtime.what_is_left_to_spend import a_budget_of


def test_a_budget_starts_with_what_it_was_given():
    budget = a_budget_of("attempts", 3)
    assert budget.left == 3
    assert not budget.exhausted


def test_spending_takes_from_what_is_left():
    budget = a_budget_of("attempts", 3)
    assert budget.spend(1) is True
    assert budget.left == 2


def test_spending_past_the_end_is_refused_and_not_raised():
    """A retry that cannot happen is a decision the caller has to make."""
    budget = a_budget_of("attempts", 1)
    assert budget.spend(1) is True
    assert budget.spend(1) is False
    assert budget.exhausted


def test_the_refusal_says_what_wanted_what():
    budget = a_budget_of("attempts", 1)
    budget.spend(1)
    budget.spend(1, on="a retry")
    assert "a retry wanted 1 of attempts and 0 was left" in budget.refusals[0]


# ----------------------------------------------------------- the hierarchy


def test_a_child_takes_at_most_what_is_left():
    """A caller cannot grant itself more than it was given."""
    turn = a_budget_of("attempts this turn", 3)
    child = turn.under("one call", at_most=10)
    assert child.allowed == 3


def test_a_childs_spending_travels_up():
    turn = a_budget_of("attempts this turn", 3)
    child = turn.under("one call", at_most=3)
    child.spend(2)
    assert turn.spent == 2
    assert turn.left == 1


def test_three_calls_of_three_retries_do_not_make_nine_attempts():
    """The property the whole thing is for."""
    turn = a_budget_of("attempts this turn", 3)
    made = 0
    for at in range(3):
        call = turn.under(f"call {at}", at_most=3)
        while call.spend(1, on="a retry"):
            made += 1
    assert made == 3
    assert turn.exhausted


def test_a_child_opened_after_the_parent_is_spent_has_nothing():
    turn = a_budget_of("attempts", 1)
    turn.spend(1)
    child = turn.under("a late call", at_most=5)
    assert child.allowed == 0
    assert child.spend(1) is False


def test_a_refusal_upstream_leaves_the_child_unspent():
    """Nothing is taken when an ancestor says no."""
    turn = a_budget_of("attempts", 1)
    first = turn.under("first", at_most=1)
    second = turn.under("second", at_most=1)
    assert first.spend(1) is True
    assert second.spend(1) is False
    assert second.spent == 0
    assert turn.spent == 1


def test_grandchildren_spend_from_the_root():
    turn = a_budget_of("attempts", 2)
    call = turn.under("a call", at_most=2)
    retry = call.under("its retries", at_most=2)
    assert retry.spend(2) is True
    assert turn.spent == 2
    assert call.spent == 2


def test_every_refusal_in_the_tree_can_be_read_back():
    """So "it stopped early" has an answer."""
    turn = a_budget_of("attempts", 1)
    call = turn.under("a call", at_most=1)
    call.spend(1)
    call.spend(1, on="a retry")
    assert any("a retry" in one for one in turn.everything_refused())


def test_spending_nothing_always_succeeds():
    assert a_budget_of("attempts", 0).spend(0) is True


def test_the_report_carries_the_whole_tree():
    turn = a_budget_of("attempts", 3)
    turn.under("a call", at_most=1).spend(1)
    report = turn.report()
    assert report["spent"] == 1
    assert report["children"][0]["what"] == "a call"
    assert report["children"][0]["left"] == 0


@pytest.mark.parametrize("allowed,asked,expected", [(3, 1, True), (1, 3, False)])
def test_a_spend_larger_than_the_budget_is_refused_whole(allowed, asked, expected):
    """Not partly spent. A half-charged attempt is neither made nor free."""
    budget = a_budget_of("attempts", allowed)
    assert budget.spend(asked) is expected
    assert budget.spent == (asked if expected else 0)

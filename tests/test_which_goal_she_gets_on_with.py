"""Which of the things she has going gets the next stretch.

It was the first of the list, which is the highest priority, which is the
biggest. Watching two very strong Go players: what decides where the next
stone goes is never which fight is biggest. It is what it costs to leave each
one alone.
"""

from __future__ import annotations

from core.goals.goal_engine import _the_one_to_get_on_with

BIG_AND_SAFE = {
    "objective": "the big one nothing is holding",
    "priority": 0.9,
    "steps_done": 2,
    "steps_total": 10,
    "plan_id": "",
}
SMALL_AND_IN_FLIGHT = {
    "objective": "the small one a plan still owns",
    "priority": 0.1,
    "steps_done": 8,
    "steps_total": 10,
    "plan_id": "plan-7",
}


def test_she_gets_on_with_the_one_that_is_at_risk() -> None:
    """This engine blocks a goal that goes stale while a plan owns it, so a
    goal in flight genuinely does lose something by being left."""
    got = _the_one_to_get_on_with([BIG_AND_SAFE, SMALL_AND_IN_FLIGHT])
    assert got["objective"] == "the small one a plan still owns"
    assert BIG_AND_SAFE["priority"] > SMALL_AND_IN_FLIGHT["priority"]


def test_where_nothing_is_at_risk_the_order_she_had_stands() -> None:
    """Without something that gets worse, urgency is not a fact, and
    manufacturing one would be worse than leaving the order alone."""
    first = {**BIG_AND_SAFE, "objective": "first"}
    second = {**BIG_AND_SAFE, "objective": "second", "priority": 0.2}
    assert _the_one_to_get_on_with([first, second])["objective"] == "first"


def test_one_thing_or_nothing_is_answered_without_weighing() -> None:
    assert _the_one_to_get_on_with([]) == {}
    only = dict(BIG_AND_SAFE)
    assert _the_one_to_get_on_with([only]) is only


def test_a_goal_with_unreadable_counts_does_not_take_the_others_down() -> None:
    broken = {"objective": "nonsense", "steps_done": "lots", "steps_total": None}
    got = _the_one_to_get_on_with([broken, SMALL_AND_IN_FLIGHT])
    assert got["objective"] == "the small one a plan still owns"


def test_the_choice_is_the_general_one_and_not_a_copy_of_it() -> None:
    """It has to be the same faculty she uses everywhere, or it is a rule of
    thumb wearing its clothes."""
    from core.goals import goal_engine

    with open(goal_engine.__file__, encoding="utf-8") as handle:
        text = handle.read()
    assert "where_to_spend_it" in text
    assert "from core.cognition.where_to_spend_the_next_one import" in text


def test_a_goal_whose_payoff_is_far_off_is_discounted_by_how_far_she_gets() -> None:
    """Not by a rate somebody picked. A Stellaris player taking an agenda that
    pays after ten years is choosing it in a game they might not be in, and
    the real number is the chance of still being there."""
    nearly_done = {
        "objective": "the one nearly finished",
        "priority": 0.5,
        "steps_done": 9,
        "steps_total": 10,
        "plan_id": "p1",
    }
    barely_begun = {
        "objective": "the one barely begun",
        "priority": 0.5,
        "steps_done": 1,
        "steps_total": 400,
        "plan_id": "p2",
    }
    got = _the_one_to_get_on_with([barely_begun, nearly_done])
    assert got["objective"] == "the one nearly finished", got


def test_the_discount_comes_from_her_own_record_and_not_a_constant() -> None:
    from core.goals import goal_engine

    with open(goal_engine.__file__, encoding="utf-8") as handle:
        text = handle.read()
    assert "HowFarSheUsuallyGets" in text
    assert "still_here_in(" in text
    at = text.index("lasted.a_run_lasted(")
    near = text[at - 300 : at]
    assert "steps_done" in near, "the record is how far her goals have got"

"""When to commit to something there is no taking back.

Two buttons at the end of a game of Cluedo: End Turn, and Final Accusation
under a line saying that if it is wrong the game is over. Nobody presses it
because they are confident.

The tests are deliberately not about Cluedo. This governs sending, deleting,
publishing, deploying, and answering — an answer is irreversible in the way
that matters, because the person acts on it.
"""

from __future__ import annotations

from core.cognition.when_to_say_it_outright import whether_to_say_it


def test_the_same_certainty_goes_both_ways_on_what_it_would_cost() -> None:
    """Which is the whole point. Needing to be ninety per cent sure is the
    same number whether the mistake costs a retry or a career."""
    cheap = whether_to_say_it(
        how_sure=0.8, being_wrong_costs=1.0, another_look_costs=0.5
    )
    dear = whether_to_say_it(
        how_sure=0.8, being_wrong_costs=100.0, another_look_costs=0.5
    )
    assert cheap.now, cheap.describe()
    assert not dear.now, dear.describe()


def test_waiting_is_never_free_or_she_waits_for_ever() -> None:
    """Not caution — a different way of getting it wrong, and the one that
    looks responsible while it happens."""
    forever = whether_to_say_it(
        how_sure=0.5, being_wrong_costs=100.0, another_look_costs=0.01
    )
    assert not forever.now

    losing_it = whether_to_say_it(
        how_sure=0.5,
        being_wrong_costs=100.0,
        another_look_costs=0.01,
        waiting_might_lose_it=0.9,
        what_it_is_worth=100.0,
    )
    assert losing_it.now, losing_it.describe()


def test_being_certain_commits_whatever_it_costs() -> None:
    got = whether_to_say_it(
        how_sure=1.0, being_wrong_costs=1_000_000.0, another_look_costs=0.001
    )
    assert got.now
    assert got.committing == 0.0


def test_knowing_nothing_does_not_commit_on_a_cheap_mistake_alone() -> None:
    """A cheap mistake still is not free, and a look that costs less is what
    a look is for."""
    got = whether_to_say_it(
        how_sure=0.0, being_wrong_costs=1.0, another_look_costs=0.1
    )
    assert not got.now
    assert got.committing == 1.0


def test_it_says_which_of_the_two_mistakes_it_is_avoiding() -> None:
    dear = whether_to_say_it(
        how_sure=0.7, being_wrong_costs=50.0, another_look_costs=0.5
    )
    assert "not sure enough" in dear.why
    sure = whether_to_say_it(
        how_sure=0.99, being_wrong_costs=50.0, another_look_costs=1.0
    )
    assert "waiting is the more expensive mistake" in sure.why


def test_certainty_outside_nought_and_one_is_brought_inside_it() -> None:
    assert whether_to_say_it(
        how_sure=5.0, being_wrong_costs=1.0, another_look_costs=0.0
    ).how_sure == 1.0
    assert whether_to_say_it(
        how_sure=-3.0, being_wrong_costs=1.0, another_look_costs=0.0
    ).how_sure == 0.0

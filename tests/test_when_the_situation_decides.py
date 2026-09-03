"""Choices that are not choices, and lines that change what a thing is.

A capture in draughts is compulsory, so there is nothing to weigh — and a
player who stops to think there has spent thought on a decision already made.
A man reaching the far row becomes a king: nothing about it grew, it crossed a
line, and on the other side it is a different kind of thing.
"""

from __future__ import annotations

from core.cognition.when_the_situation_decides import WhatItBecomes, nothing_to_decide


def test_one_option_is_not_a_decision() -> None:
    assert nothing_to_decide(["the only way"]) == "the only way"
    assert nothing_to_decide(["this", "that"]) is None
    assert nothing_to_decide([]) is None


def test_a_situation_that_compels_one_act_settles_it() -> None:
    """Weighing a settled thing is effort spent looking like diligence."""
    acts = ["retry", "take the lock", "give up"]
    assert (
        nothing_to_decide(acts, compelled=lambda one: one == "take the lock")
        == "take the lock"
    )


def test_several_compelled_is_still_a_choice_but_only_among_them() -> None:
    acts = ["a", "b", "c"]
    assert nothing_to_decide(acts, compelled=lambda one: one in {"a", "b"}) is None


def test_a_quantity_that_has_crossed_a_line_is_a_different_thing() -> None:
    becomes = WhatItBecomes()
    becomes.a_line("the queue", at=1000.0, becomes="an outage")
    assert becomes.what_it_is_now("the queue", 20.0) == "the queue"
    assert becomes.what_it_is_now("the queue", 4000.0) == "an outage"
    assert becomes.has_crossed("the queue", 4000.0)


def test_it_can_find_the_line_rather_than_be_told_it() -> None:
    becomes = WhatItBecomes()
    for at in (10.0, 40.0, 90.0):
        becomes.it_behaved_differently("the retries", at=at, differently=False)
    for at in (150.0, 300.0):
        becomes.it_behaved_differently("the retries", at=at, differently=True)
    assert becomes.where_the_line_is("the retries") == 120.0
    assert becomes.has_crossed("the retries", 200.0)
    assert not becomes.has_crossed("the retries", 50.0)


def test_a_quantity_with_no_clean_line_is_said_to_have_none() -> None:
    """Inventing one would be worse than saying so."""
    muddled = WhatItBecomes()
    muddled.it_behaved_differently("a thing", at=10.0, differently=True)
    muddled.it_behaved_differently("a thing", at=90.0, differently=False)
    assert muddled.where_the_line_is("a thing") == 0.0
    assert not muddled.has_crossed("a thing", 500.0)


def test_it_says_how_far_short_of_becoming_the_other_thing() -> None:
    becomes = WhatItBecomes()
    becomes.a_line("the queue", at=1000.0, becomes="an outage")
    assert "short of an outage" in becomes.lines[0].describe(900.0)
    assert "it is an outage now" in becomes.lines[0].describe(1200.0)

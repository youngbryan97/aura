"""Doing the work before the thing arrives, because it can be seen coming.

A Pokémon player stands outside a gym and does not go in — they buy potions
and teach a move, half an hour aimed at a fight that has not started. In
Minecraft the sun goes down every twenty minutes and the door is built in the
light. None of them is reacting.

The tests are a release and a nightly build, because that is the same shape.
"""

from __future__ import annotations

from core.cognition.getting_ready_for_what_is_coming import WhatUsuallyComes


def _seen() -> WhatUsuallyComes:
    knows = WhatUsuallyComes()
    for at in (100.0, 200.0, 300.0):
        knows.it_came("a release", at=at, needing=["a changelog", "a signed build"])
    knows.it_came("a release", at=400.0, needing=["a changelog"])
    for at in (10.0, 20.0, 30.0, 40.0):
        knows.it_came("the nightly", at=at, needing=["a clean tree"])
    return knows


def test_it_learns_what_the_thing_wanted_from_its_own_record() -> None:
    knows = _seen()
    assert knows.what_it_wants("a release")[0] == "a changelog"
    assert "a signed build" in knows.what_it_wants("a release")


def test_a_thing_wanted_half_the_time_is_still_worth_having() -> None:
    """Being short of it half the time is the same shortage discovered twice."""
    assert "a signed build" in _seen().what_it_wants("a release")


def test_it_learns_the_rhythm_and_says_when_the_next_is_due() -> None:
    knows = _seen()
    assert knows.how_long_between("a release") == 100.0
    assert knows.due_in("a release", now=450.0) == 50.0
    assert knows.due_in("a release", now=560.0) < 0, "overdue is said as overdue"


def test_one_sighting_is_an_event_and_two_are_a_rhythm() -> None:
    once = WhatUsuallyComes()
    once.it_came("a thing", at=5.0, needing=["something"])
    assert once.how_long_between("a thing") == 0.0
    assert once.due_in("a thing", now=99.0) == 0.0


def test_it_says_what_she_is_short_of_and_not_what_she_has() -> None:
    knows = _seen()
    assert knows.what_to_get_first("a release", holding=["a changelog"]) == (
        "a signed build",
    )
    assert knows.what_to_get_first(
        "a release", holding=["a changelog", "a signed build"]
    ) == ()


def test_being_ready_for_a_thing_she_is_ready_for_is_not_preparation() -> None:
    knows = _seen()
    ready = knows.worth_getting_ready(
        ["a release", "the nightly"],
        now=450.0,
        holding=["a changelog", "a signed build", "a clean tree"],
    )
    assert ready == [], "nothing to do is nothing to do"

    short = knows.worth_getting_ready(
        ["a release", "the nightly"], now=450.0, holding=["a changelog"]
    )
    assert [one[0] for one in short] == ["the nightly", "a release"], short
    assert short[0][1] < short[1][1], "soonest first"


def test_what_it_learned_survives_the_process() -> None:
    knows = _seen()
    again = WhatUsuallyComes.from_memory(knows.as_memory())
    assert again.what_it_wants("a release") == knows.what_it_wants("a release")
    assert again.how_long_between("a release") == 100.0
    assert WhatUsuallyComes.from_memory([]).came == {}

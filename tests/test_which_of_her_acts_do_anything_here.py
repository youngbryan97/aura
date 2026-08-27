"""Which of her actions do anything here is found out, not declared.

A solver written for one thing is handed its action set — four moves, named in
the source. She was handed hers the same way, and it was the last large thing
about an unfamiliar world that somebody else was still establishing for her.
Everything else she works out: which window she is in, which part of it
answers, what her acts do, what changes on its own, and what a good situation
looks like.
"""

from __future__ import annotations

import pytest

from core.agency.what_i_can_do_here import (
    COMMITS_TO_NOTHING,
    COMMITS_TO_SOMETHING,
    ENOUGH_TO_JUDGE,
    WhatWorksHere,
    worth_trying,
)


# ── the line that is not crossed ─────────────────────────────────────────

def test_only_things_that_commit_to_nothing_are_tried_to_find_out():
    for key in worth_trying(("up", "return", "space")):
        assert key in COMMITS_TO_NOTHING


def test_and_nothing_that_presses_whatever_has_focus_is():
    assert not set(COMMITS_TO_NOTHING) & set(COMMITS_TO_SOMETHING)
    for key in COMMITS_TO_SOMETHING:
        assert key not in worth_trying(())


def test_what_she_was_told_comes_first_because_somebody_usually_knows():
    assert worth_trying(("left", "right"))[:2] == ("left", "right")


# ── finding out ──────────────────────────────────────────────────────────

@pytest.fixture
def here() -> WhatWorksHere:
    return WhatWorksHere(told=("up", "down", "left", "right"))


def test_nothing_is_judged_before_it_has_been_tried_enough(here):
    for _ in range(ENOUGH_TO_JUDGE - 1):
        here.tried("up", changed=False)
    assert not here.dead()


def test_something_that_never_does_anything_is_not_one_of_her_acts(here):
    for _ in range(ENOUGH_TO_JUDGE):
        here.tried("up", changed=False)
    assert here.dead() == ("up",)
    assert "up" not in here.available()


def test_and_one_that_ever_does_something_stays(here):
    for _ in range(ENOUGH_TO_JUDGE):
        here.tried("up", changed=False)
    here.tried("up", changed=True)
    assert here.dead() == ()
    assert "up" in here.available()


def test_a_world_where_what_she_was_told_works_never_widens(here):
    for key in here.told:
        here.tried(key, changed=True)
    assert set(here.available()) == set(here.told)


def test_a_world_where_it_does_not_work_widens_to_what_might(here):
    told = WhatWorksHere(told=("up",))
    for _ in range(ENOUGH_TO_JUDGE):
        told.tried("up", changed=False)
    assert "up" not in told.available()
    assert set(told.available()) <= set(COMMITS_TO_NOTHING)


def test_she_says_what_she_found_out(here):
    assert "not worked out yet" in here.says()
    here.tried("left", changed=True)
    for _ in range(ENOUGH_TO_JUDGE):
        here.tried("up", changed=False)
    said = here.says()
    assert "left" in said and "up" in said


# ── and it survives the run ──────────────────────────────────────────────

def test_what_worked_here_last_time_is_a_starting_point(here):
    here.tried("left", changed=True)
    again = WhatWorksHere.from_memory(here.as_memory(), told=here.told)
    assert again.works() == ("left",)


def test_a_memory_of_the_wrong_shape_is_not_trusted():
    assert WhatWorksHere.from_memory("nonsense", told=("up",)).told == ("up",)
    assert WhatWorksHere.from_memory({"did_something": "no"}, told=("up",)).works() == ()


# ── and the loop uses it ─────────────────────────────────────────────────

def test_the_pursuit_offers_what_works_rather_than_what_it_was_told():
    from pathlib import Path

    source = Path("core/skills/screen_pursuit.py").read_text()
    assert "screen_options(can_do.available() or move_keys)" in source
    assert "can_do.tried(previous.chosen.name, attempt.verdict.observed_change)" in source

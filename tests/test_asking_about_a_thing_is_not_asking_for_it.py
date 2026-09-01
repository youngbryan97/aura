"""Naming a thing is not asking for it to be done.

Whether a request is a task to keep at is decided by whether it names one, so
a request that merely mentions something acquires that thing's whole desktop
pursuit. Two ways in that had nothing to do with each other:

A request to be TOLD about something is an imperative with a full stop, so
nothing about its shape says question — and the thing requested is an
utterance, not an act in the world.

And a medium named BEFORE the thing is what she is being asked to work in.
"Search the web for X" and "find X online" name the same two things in
opposite orders, and the order is the whole difference.
"""

from __future__ import annotations

import pytest

from core.runtime.watched_goal import read_watched_goal


@pytest.mark.parametrize(
    "said",
    [
        "tell me about the history of tile games",
        "tell us about sliding puzzles",
        "explain how the scoring works",
        "describe what a sliding puzzle is",
        "show me what you know about it",
    ],
)
def test_asking_to_be_told_is_asking(said):
    assert read_watched_goal(said) is None


@pytest.mark.parametrize(
    "said",
    [
        "search the web for 2048 strategy guides",
        "google 2048 strategies",
        "look through the internet for sliding puzzle tactics",
    ],
)
def test_a_medium_named_first_is_what_she_is_to_search(said):
    assert read_watched_goal(said) is None


@pytest.mark.parametrize(
    "said",
    [
        "Find 2048 online, play it, and get to a 256 tile",
        "play 2048 in my browser until you reach 512",
    ],
)
def test_a_medium_named_after_the_thing_is_where_the_thing_is(said):
    goal = read_watched_goal(said)
    assert goal is not None, "a request to act on a thing on the web was read as a lookup"
    assert goal.where, "she was not told where to go"


def test_telling_her_to_do_something_still_reads_as_doing():
    """The fix must not swallow every imperative."""
    for said in ("beat 2048", "play 2048", "win at 2048"):
        assert read_watched_goal(said) is not None, said

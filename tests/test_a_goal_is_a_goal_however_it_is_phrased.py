"""A finish can be named by saying when, or by saying what she is aiming at.

"Play until you get a 256 tile" and "play it, and get to a 256 tile" are one
request. LIVE 2026-08-26: the second was not recognised as a goal to keep at,
so it never reached the lane that plays anything — it was answered with a web
search while the board sat untouched. The capability was there; the phrasing
could not reach it.

The second failure in the same sentence: "…and tell me here when you have it"
reads as a condition too, and taken as the latest one it stole the finish
from the tile actually being played for.
"""
from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import looks_like_desktop_objective
from core.runtime.watched_goal import read_watched_goal

SAID_MANY_WAYS = [
    ("Find 2048 online, play it, and get to a 256 tile. Say what you are about to "
     "do before each move, and tell me here when you have it.", "256"),
    ("Go find a 2048 game online and play it until you get a 128 tile.", "128"),
    ("play 2048 until you get a 256 tile", "256"),
    ("open 2048 and keep playing until you reach 4096", "4096"),
    ("play 2048 and get me to a 512 tile, narrating as you go", "512"),
    ("keep playing 2048, I want to see a 1024 tile", "1024"),
    ("play 2048 and don't stop till you hit a 2048 tile", "2048"),
]


@pytest.mark.parametrize(("said", "finish"), SAID_MANY_WAYS)
def test_the_same_request_is_recognised_however_it_is_worded(said, finish):
    goal = read_watched_goal(said)
    assert goal is not None, "the phrasing could not reach the capability"
    assert goal.success_when == finish


@pytest.mark.parametrize(("said", "_finish"), SAID_MANY_WAYS)
def test_each_of_them_reaches_the_lane_that_acts(said, _finish):
    assert looks_like_desktop_objective(said)


def test_a_rider_on_the_end_does_not_steal_the_finish():
    """A number is a thing the screen can be matched against and needs no
    interpretation, so it beats a clause that names nothing however late it
    appears."""
    said = "play 2048 and get to a 256 tile, then tell me here when you have it"
    goal = read_watched_goal(said)
    assert goal is not None
    assert goal.success_when == "256", "the closing rider outranked the tile"


def test_a_quoted_finish_still_wins_over_a_vague_one():
    said = "keep refreshing until it says 'Deploy succeeded', and let me know when you can"
    goal = read_watched_goal(said)
    assert goal is not None
    assert goal.success_when == "Deploy succeeded"


@pytest.mark.parametrize(
    "said",
    [
        "what is the capital of France?",
        "search the web for 2048 strategy guides",
        "how does the 2048 scoring work?",
        "tell me about the history of tile games",
    ],
)
def test_a_question_is_not_a_goal_to_keep_at(said):
    """Widening what counts as naming a finish must not turn lookups into
    tasks. A request that asks rather than acts has nothing to keep at."""
    assert read_watched_goal(said) is None
    assert not looks_like_desktop_objective(said)


@pytest.mark.parametrize(
    ("said", "finish"),
    [
        ("keep playing 2048, I want to see a 1024 tile", "1024"),
        ("keep playing until I see a 1024 tile", "1024"),
        ("play 2048 and show me a 512", "512"),
    ],
)
def test_a_target_named_without_any_introducing_verb_is_still_the_target(said, finish):
    """People name a target in more ways than any list of verbs will hold. The
    continuation cue has already established this is a thing to keep at, and
    in a thing to keep at, a value named in it is what finishing looks like.
    """
    goal = read_watched_goal(said)
    assert goal is not None
    assert goal.success_when == finish


@pytest.mark.parametrize(
    "said",
    [
        "play 2048",
        "play 2048 for 10 minutes",
        "keep playing 2048 for 30 moves",
        "keep playing that game for a few rounds",
    ],
)
def test_how_long_is_not_what(said):
    """One value in a request that names a thing is that thing's name, and a
    duration says how long rather than what. Neither is a state of the screen,
    so neither makes this a goal with a finish."""
    assert read_watched_goal(said) is None

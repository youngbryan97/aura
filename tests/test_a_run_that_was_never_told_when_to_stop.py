"""A request can name a process without naming an end.

"Find a sliding puzzle and work out how it moves by playing it" says exactly
what to do and never says when to stop. Three separate places assumed that
could not happen, and each of them turned the request away politely:

    read_watched_goal      no finishing test, so not a goal to be watched
    _pursue_on_screen      "no finishing condition was given, so the run
                           could never end"
    ScreenPursuitInput    success_when, min_length=1

LIVE 2026-08-27, in that order, one per attempt. The last of them refused a
goal that had arrived correctly parsed, with the page to open and the keys to
press, in 417ms.

The loop always stopped. It runs to a cycle count and a clock, both of them
arguments to it. What the refusals blocked was not an endless run — it was
most of the ways a person asks for one.
"""

from __future__ import annotations

import pytest

from core.skills.screen_pursuit import ScreenPursuitInput, goal_reached


# ── the screen never says finished, and that is allowed ──────────────────

def test_a_pursuit_can_be_asked_for_without_a_finish():
    params = ScreenPursuitInput(goal="work out how it moves by playing it")
    assert params.success_when == ""


def test_a_finish_that_is_named_is_still_carried():
    params = ScreenPursuitInput(goal="play it", success_when="4096")
    assert params.success_when == "4096"


def test_nothing_on_screen_ever_satisfies_a_condition_nobody_gave():
    seen = {"text": "2048 4096 done finished complete", "layout": []}
    assert goal_reached(seen, "") is False


@pytest.mark.parametrize("said", ["   ", "\t", "\n"])
def test_and_whitespace_is_not_a_condition_either(said):
    assert goal_reached({"text": said, "layout": []}, said) is False


# ── what she plays for when nobody said ──────────────────────────────────

def test_she_reads_the_goal_off_the_thing_in_front_of_her():
    """A thing that combines equal pairs cannot exceed one doubling a place."""
    from core.perception.what_is_there import arranged
    from core.skills.screen_pursuit import _what_there_is_to_aim_at

    board = arranged([
        (0.20 + r * 0.15, 0.20 + c * 0.15, "2")
        for r in range(4)
        for c in range(4)
    ])
    assert _what_there_is_to_aim_at(board) == "65536"


def test_a_smaller_thing_could_hold_less():
    from core.perception.what_is_there import arranged
    from core.skills.screen_pursuit import _what_there_is_to_aim_at

    assert _what_there_is_to_aim_at(arranged([(0.2, 0.2, "2"), (0.2, 0.35, "4")])) == "4"


def test_and_it_is_a_goal_her_own_measure_can_use():
    """"the largest" is not: worth_comparing refuses it and no search runs."""
    from core.agency.how_good_is_this import worth_comparing
    from core.perception.what_is_there import arranged
    from core.skills.screen_pursuit import _what_there_is_to_aim_at

    board = arranged([(0.2, 0.2, "128"), (0.2, 0.35, "64")])
    assert worth_comparing(_what_there_is_to_aim_at(board), "") is True
    assert worth_comparing("the largest", "") is False


def test_it_stays_worth_something_all_the_way_through():
    """A nearer goal saturates: once reached, every situation scores alike."""
    from core.agency.how_good_is_this import how_good
    from core.perception.what_is_there import arranged
    from core.skills.screen_pursuit import _what_there_is_to_aim_at

    def board(biggest: str):
        return arranged(
            [(0.20, 0.20, biggest)]
            + [(0.20 + r * 0.15, 0.35 + c * 0.15, "2") for r in range(3) for c in range(3)]
        )

    aim = _what_there_is_to_aim_at(board("4"))
    scores = [how_good(board(n), toward=aim) for n in ("4", "128", "1024")]
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_where_nothing_counts_nothing_is_invented():
    from core.perception.what_is_there import arranged
    from core.skills.screen_pursuit import _what_there_is_to_aim_at

    assert _what_there_is_to_aim_at(arranged([(0.2, 0.2, "Mon"), (0.2, 0.35, "Tue")])) == ""
    assert _what_there_is_to_aim_at(None) == ""

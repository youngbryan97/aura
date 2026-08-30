""""Nothing I do changes anything" is a good ending test and a poor diagnosis.

It is equally true when a game is over, when a dialog sits on top of the thing,
and when she is looking at somebody else's window — and those want opposite
responses. Collapsing them produced the worst behaviour she has.

LIVE 2026-08-29: the page said "Game Over. 940 points scored in 100 moves." and
she went on saying "Going right", pressing arrow keys into a finished board and
narrating moves as though a game were happening. The keys landed. There was
simply nothing left for them to do, and nothing in the loop could tell that
from being blocked.

Two of the three she can do something about, and the doing is the answer rather
than the reporting. The third is not a fault: finished is a real outcome and
stopping is the right response to it.
"""

from __future__ import annotations

import pytest

from core.perception.why_nothing_answers import (
    ELSEWHERE,
    ENDED,
    IN_FRONT,
    UNKNOWN,
    work_out_why,
)

HERS = "Google Chrome"


# ── telling the three apart ──────────────────────────────────────────────

def test_somebody_elses_window_in_front():
    why = work_out_why(mine=HERS, in_front="Claude", on_top="", still_there=True)
    assert why.because == ELSEWHERE
    assert why.what == "Claude"
    assert why.can_fix is True


def test_something_sitting_over_the_part_she_uses():
    why = work_out_why(
        mine=HERS, in_front=HERS, on_top="UserNotificationCenter", still_there=True
    )
    assert why.because == IN_FRONT
    assert why.can_fix is True


def test_the_thing_she_was_acting_on_is_gone():
    why = work_out_why(mine=HERS, in_front=HERS, on_top="", still_there=False)
    assert why.because == ENDED
    assert why.can_fix is False


def test_it_is_there_unobstructed_and_still_not_answering():
    """Saying so beats picking one of the three."""
    why = work_out_why(mine=HERS, in_front=HERS, on_top="", still_there=True)
    assert why.because == UNKNOWN
    assert why.can_fix is False


# ── ordered by what would make the others meaningless ────────────────────

def test_looking_elsewhere_outranks_everything():
    """A reading of somebody else's screen says nothing about her task."""
    why = work_out_why(mine=HERS, in_front="Claude", on_top="Dialog", still_there=False)
    assert why.because == ELSEWHERE


def test_and_being_covered_outranks_being_gone():
    """A thing she cannot see is not a thing that is not there."""
    why = work_out_why(mine=HERS, in_front=HERS, on_top="Dialog", still_there=False)
    assert why.because == IN_FRONT


@pytest.mark.parametrize(
    "spoiled",
    [
        {"in_front": "Something Else", "on_top": ""},
        {"in_front": HERS, "on_top": "A Dialog"},
    ],
)
def test_an_ending_needs_her_own_window_unobstructed(spoiled):
    why = work_out_why(mine=HERS, still_there=False, **spoiled)
    assert why.because != ENDED


# ── and each says what it is ─────────────────────────────────────────────

@pytest.mark.parametrize(
    ("because", "expected"),
    [
        (ELSEWHERE, "not looking at it"),
        (IN_FRONT, "over the part I am using"),
        (ENDED, "finished"),
        (UNKNOWN, "do not know why"),
    ],
)
def test_she_can_say_which_it_is(because, expected):
    cases = {
        ELSEWHERE: dict(in_front="Claude", on_top="", still_there=True),
        IN_FRONT: dict(in_front=HERS, on_top="A Dialog", still_there=True),
        ENDED: dict(in_front=HERS, on_top="", still_there=False),
        UNKNOWN: dict(in_front=HERS, on_top="", still_there=True),
    }
    why = work_out_why(mine=HERS, **cases[because])
    assert expected in why.says()


def test_not_knowing_which_window_is_hers_is_not_a_diagnosis():
    why = work_out_why(mine="", in_front="Claude", on_top="", still_there=True)
    assert why.because != ELSEWHERE


# ── and the loop acts on it rather than reporting it ─────────────────────

def test_the_pursuit_fixes_what_it_can_and_carries_on():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit)
    assert "work_out_why(" in source
    assert "if why.can_fix:" in source
    doing = [ln for ln in source.splitlines() if not ln.strip().startswith("#")]
    assert any("ended = False" in ln for ln in doing), (
        "a reason she fixed is still being treated as an ending"
    )
    assert any("began_again()" in ln for ln in doing)


def test_and_only_stops_when_it_has_really_ended():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit)
    assert "why.because == ENDED" in source

"""Her own background thinking taking the model out from under a task.

A task somebody asked for is foreground for as long as it takes, not for the
length of the sentence that started it. Her background work already stands
aside for a foreground turn — it checks — but a turn ends the moment the reply
is composed, and everything after that is the task itself.

So the whole of a desktop task ran as background, beside her research loops,
competing for the same inference lanes. LIVE 2026-09-07, playing a game on
this machine: thirty-two decisions about what to do next, nine of them refused
outright because the lanes were exhausted, by her own reimplementation lab and
curriculum loop running against the model she needed to choose a move with.
"""

from __future__ import annotations

import inspect

import core.skills.screen_pursuit as sp
from core.runtime.foreground_guard import (
    begin_foreground_turn,
    foreground_activity_reason,
    snapshot,
)


def test_a_lease_makes_background_work_stand_aside():
    """The mechanism background work is already checking."""
    before = foreground_activity_reason()
    with begin_foreground_turn(owner="a test", source="a test"):
        assert foreground_activity_reason() == "foreground_chat_active"
    assert foreground_activity_reason() != "foreground_chat_active" or before


def test_the_lease_names_who_holds_it():
    with begin_foreground_turn(owner="screen_pursuit", source="desktop_task"):
        held = snapshot()
        assert held["active"]
        assert held["owner"] == "screen_pursuit"
        assert held["source"] == "desktop_task"


def test_a_pursuit_takes_one():
    """The defect: it never did, so the task ran as background."""
    source = inspect.getsource(sp.pursue_on_screen)
    assert "begin_foreground_turn" in source, (
        "a task somebody asked for still runs as background work"
    )


def test_a_pursuit_gives_it_back():
    """Held past the end, everything after it would be starved instead."""
    source = inspect.getsource(sp.pursue_on_screen)
    assert "holding_the_foreground.close()" in source
    before = source.index("begin_foreground_turn")
    closed = source.index("holding_the_foreground.close()")
    assert closed > before
    # In a finally, so a run that raises still hands it back.
    tail = source[source.rindex("finally:", 0, closed) : closed]
    assert "finally" not in tail.replace("finally:", "", 1)


def test_a_guard_that_will_not_answer_does_not_stop_the_task():
    """Not being able to claim the foreground is not a reason to refuse work."""
    source = inspect.getsource(sp.pursue_on_screen)
    where = source.index("begin_foreground_turn")
    around = source[where - 400 : where + 700]
    assert "except" in around
    assert "record_degradation" in around

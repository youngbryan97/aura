"""Interrupting work so that starting it again is not starting over.

LangGraph makes interrupt and resume first class. Aura had both halves apart —
a turn can be cancelled and a state can be checkpointed — and not the thing
that makes an interrupt useful: a record of WHERE the work stopped.

An interrupt without that is a cancellation. The work is gone and the next
attempt repeats it, which is why a long task interrupted near the end costs
the same as one interrupted at the start.
"""
from __future__ import annotations

import pytest

from core.state.a_checkpoint_and_its_writes import TheChannels
from core.state.stopping_and_starting_again import (
    WhyItStopped,
    forget_everything,
    interrupt,
    resume,
    what_was_interrupted,
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


# -------------------------------------------------------------- the record


def test_an_interruption_says_where_it_stopped_and_why():
    one = interrupt(
        "a long task",
        WhyItStopped.OUT_OF_TIME,
        checkpoint="after two steps",
        was_about_to="step three",
    )
    assert one.checkpoint == "after two steps"
    assert one.was_about_to == "step three"
    assert one.why is WhyItStopped.OUT_OF_TIME


def test_resuming_hands_back_what_it_was_about_to_do():
    """So the next attempt does not work it out again."""
    interrupt("a task", WhyItStopped.OUT_OF_TIME, was_about_to="step three")
    next_step, from_where = resume("a task")
    assert next_step == "step three"
    assert from_where.why is WhyItStopped.OUT_OF_TIME


def test_resuming_restores_the_checkpoint():
    channels = TheChannels()
    channels.write("plan", ["a", "b"], by="planner")
    channels.commit()
    channels.checkpoint("after two steps")
    channels.write("plan", ["a", "b", "c"], by="planner")
    channels.commit()

    interrupt("a task", WhyItStopped.OUT_OF_TIME, checkpoint="after two steps")
    resume("a task", channels=channels)
    assert channels.value("plan") == ["a", "b"]


def test_resuming_something_that_was_never_interrupted_is_nothing():
    assert resume("never stopped") == (None, None)


def test_interrupting_twice_keeps_where_it_is_now():
    """Keeping the first would resume into the past."""
    interrupt("a task", WhyItStopped.OUT_OF_TIME, was_about_to="step two")
    interrupt("a task", WhyItStopped.OUT_OF_BUDGET, was_about_to="step five")
    next_step, from_where = resume("a task")
    assert next_step == "step five"
    assert from_where.why is WhyItStopped.OUT_OF_BUDGET


# --------------------------------------------------- what will not clear


def test_a_refusal_is_not_resumed():
    """Resuming into the same wall is how a retry loop is born."""
    interrupt("a refused thing", WhyItStopped.REFUSED, was_about_to="x")
    next_step, from_where = resume("a refused thing")
    assert next_step is None
    assert from_where is not None
    assert not from_where.resumable


def test_a_refusal_stays_on_the_list_rather_than_being_taken_off():
    interrupt("a refused thing", WhyItStopped.REFUSED)
    resume("a refused thing")
    assert what_was_interrupted()["interrupted"] == 1
    assert what_was_interrupted()["resumable"] == 0


@pytest.mark.parametrize(
    "why",
    [
        WhyItStopped.ASKED_A_PERSON,
        WhyItStopped.OUT_OF_BUDGET,
        WhyItStopped.OUT_OF_TIME,
        WhyItStopped.THE_RUNTIME_STOPPED,
    ],
)
def test_every_other_reason_is_resumable(why):
    interrupt("a task", why, was_about_to="on")
    assert resume("a task")[0] == "on"


def test_resuming_takes_it_off_the_list():
    interrupt("a task", WhyItStopped.OUT_OF_TIME)
    resume("a task")
    assert what_was_interrupted()["interrupted"] == 0


def test_a_missing_checkpoint_does_not_stop_the_resume():
    """The work still knows what it was about to do."""
    channels = TheChannels()
    interrupt(
        "a task", WhyItStopped.OUT_OF_TIME, checkpoint="never taken",
        was_about_to="step three",
    )
    assert resume("a task", channels=channels)[0] == "step three"


def test_it_counts_how_often_something_was_picked_up():
    interrupt("a task", WhyItStopped.OUT_OF_TIME)
    _, one = resume("a task")
    assert one.resumed == 1


def test_the_list_reads_back_as_data():
    import json

    interrupt("a task", WhyItStopped.ASKED_A_PERSON, said="which file did you mean?")
    back = json.loads(json.dumps(what_was_interrupted()))
    assert back["each"]["a task"]["why"] == "waiting for a person"
    assert back["each"]["a task"]["said"] == "which file did you mean?"

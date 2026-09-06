"""A snapshot whose parts were true at different moments is a snapshot of none.

Soar draws an explicit input/output phase boundary, and the closure asked for
the same: every sensor and action channel declares a consistency mode, and a
state snapshot records which observation frontier it was taken at.

If CPU load was read at the top of the tick, the screen halfway through and
the model's health from a five-minute-old cache, then "the state at 04:31"
never existed — no instant had all three of those values at once, and a reader
comparing two such snapshots is comparing two mixtures.
"""
from __future__ import annotations

import time

import pytest

from core.state.when_an_observation_was_true import (
    HowItIsRead,
    a_tick_began,
    declare_a_channel,
    forget_everything,
    note_a_reading,
    the_frontier,
    what_does_not_say_how_it_is_read,
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


# ----------------------------------------------------------- the declaration


def test_a_channel_says_when_its_value_is_true():
    declare_a_channel("soma.cpu", HowItIsRead.SAMPLED_AT_TICK)
    assert the_frontier()["how_each_is_read"] == {"soma.cpu": "sampled at tick"}


def test_a_channel_that_never_declared_cannot_be_read():
    """One nobody can place in time is worse than none: it still appears."""
    with pytest.raises(KeyError, match="has not said how it is read"):
        note_a_reading("nobody.declared")


def test_the_refusal_names_the_modes_it_could_have_declared():
    with pytest.raises(KeyError) as caught:
        note_a_reading("x")
    for mode in ("sampled at tick", "streaming", "transactional"):
        assert mode in str(caught.value)


def test_the_three_modes_answer_different_questions():
    assert {str(one) for one in HowItIsRead} == {
        "sampled at tick", "streaming", "transactional"
    }


def test_undeclared_channels_can_be_named():
    declare_a_channel("known", HowItIsRead.STREAMING)
    assert what_does_not_say_how_it_is_read(["known", "a stranger"]) == ["a stranger"]


# ------------------------------------------------------------- the frontier


def test_a_reading_carries_the_tick_it_belongs_to():
    declare_a_channel("soma.cpu", HowItIsRead.SAMPLED_AT_TICK)
    a_tick_began()
    assert note_a_reading("soma.cpu").tick == 1


def test_a_streaming_reading_says_how_old_it_was():
    """So a reader can refuse a stale one instead of using it."""
    declare_a_channel("llm.health", HowItIsRead.STREAMING)
    reading = note_a_reading("llm.health", measured_at=time.time() - 300)
    assert reading.stale_by_s == pytest.approx(300, abs=2)


def test_a_reading_taken_now_is_not_stale():
    declare_a_channel("llm.health", HowItIsRead.STREAMING)
    assert note_a_reading("llm.health").stale_by_s < 0.1


def test_a_frontier_with_everything_read_this_tick_is_consistent():
    declare_a_channel("soma.cpu", HowItIsRead.SAMPLED_AT_TICK)
    a_tick_began()
    note_a_reading("soma.cpu")
    assert the_frontier()["consistent"] is True


def test_a_sampled_channel_not_read_this_tick_makes_it_inconsistent():
    """This is the mixture the whole thing exists to catch."""
    declare_a_channel("soma.cpu", HowItIsRead.SAMPLED_AT_TICK)
    a_tick_began()
    note_a_reading("soma.cpu")
    a_tick_began()

    frontier = the_frontier()
    assert frontier["consistent"] is False
    assert frontier["sampled_channels_not_read_this_tick"] == ["soma.cpu"]


def test_a_streaming_channel_from_a_previous_tick_is_not_a_defect():
    """Streaming means whatever arrived most recently, and that is allowed."""
    declare_a_channel("llm.health", HowItIsRead.STREAMING)
    a_tick_began()
    note_a_reading("llm.health")
    a_tick_began()
    assert the_frontier()["consistent"] is True


def test_the_frontier_says_what_it_means():
    assert "snapshot of no moment" in the_frontier()["what_this_means"]


def test_a_reading_reads_back_as_data():
    import json

    declare_a_channel("soma.cpu", HowItIsRead.SAMPLED_AT_TICK)
    back = json.loads(json.dumps(note_a_reading("soma.cpu").to_dict()))
    assert back["channel"] == "soma.cpu"
    assert "stale_by_s" in back


def test_declaring_twice_replaces():
    declare_a_channel("c", HowItIsRead.STREAMING)
    declare_a_channel("c", HowItIsRead.TRANSACTIONAL)
    assert the_frontier()["how_each_is_read"]["c"] == "transactional"

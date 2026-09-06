"""A checkpoint that is ahead of its writes restores a state that never was.

LangGraph beat Aura substantially under the maturity rubric and the review was
explicit that it was not for cleanliness — some of its internals are dense. It
was for semantics: pending writes separated from committed checkpoints,
channel versions, what each node has seen, and durability that is never
allowed ahead of the writes that produced it.
"""
from __future__ import annotations

import pytest

from core.state.a_checkpoint_and_its_writes import TheChannels, WritesStillPending


@pytest.fixture
def channels() -> TheChannels:
    return TheChannels()


# ------------------------------------------------------ pending vs committed


def test_a_write_is_not_in_its_channel_until_it_is_committed(channels):
    """A half-finished node must be invisible to the node after it."""
    channels.write("plan", ["step one"], by="planner")
    assert channels.value("plan") is None
    assert channels.version("plan") == 0

    channels.commit()
    assert channels.value("plan") == ["step one"]
    assert channels.version("plan") == 1


def test_the_version_moves_on_every_commit(channels):
    for n in range(3):
        channels.write("plan", n, by="planner")
        channels.commit()
    assert channels.version("plan") == 3
    assert channels.value("plan") == 2


def test_the_last_write_in_one_commit_wins(channels):
    channels.write("mode", "reactive", by="router")
    channels.write("mode", "deliberate", by="executive")
    channels.commit()
    assert channels.value("mode") == "deliberate"


def test_pending_writes_can_be_thrown_away(channels):
    channels.write("plan", ["abandoned"], by="planner")
    assert channels.discard_pending() == 1
    assert channels.commit() == {}
    assert channels.value("plan") is None


def test_pending_names_the_channel_and_who_produced_it(channels):
    channels.write("plan", 1, by="planner")
    assert channels.pending() == [{"channel": "plan", "by": "planner"}]


# ------------------------------------------------ durability behind writes


def test_a_checkpoint_is_refused_while_writes_are_pending(channels):
    """The property worth copying exactly."""
    channels.write("plan", ["not committed"], by="planner")
    with pytest.raises(WritesStillPending, match="not committed: plan"):
        channels.checkpoint("too early")


def test_the_refusal_names_every_channel_still_waiting(channels):
    channels.write("plan", 1, by="planner")
    channels.write("mode", 2, by="router")
    with pytest.raises(WritesStillPending) as caught:
        channels.checkpoint("too early")
    assert "mode" in str(caught.value)
    assert "plan" in str(caught.value)


def test_a_checkpoint_after_a_commit_holds_the_versions_that_made_it(channels):
    channels.write("plan", ["a"], by="planner")
    channels.write("mode", "deliberate", by="router")
    channels.commit()

    taken = channels.checkpoint("after planning")
    assert taken.versions == {"plan": 1, "mode": 1}
    assert taken.values["plan"] == ["a"]


# ----------------------------------------------------------- what was seen


def test_a_channel_is_new_to_a_node_that_has_not_acted_on_it(channels):
    channels.write("plan", 1, by="planner")
    channels.commit()
    assert channels.is_new_to("reader", "plan") is True


def test_a_node_that_has_seen_the_version_does_not_see_it_again(channels):
    channels.write("plan", 1, by="planner")
    channels.commit()
    channels.mark_seen("reader", "plan")
    assert channels.is_new_to("reader", "plan") is False


def test_a_new_version_is_new_again_to_a_node_that_saw_the_old_one(channels):
    channels.write("plan", 1, by="planner")
    channels.commit()
    channels.mark_seen("reader", "plan")

    channels.write("plan", 2, by="planner")
    channels.commit()
    assert channels.is_new_to("reader", "plan") is True


def test_what_one_node_saw_is_not_what_another_saw(channels):
    channels.write("plan", 1, by="planner")
    channels.commit()
    channels.mark_seen("reader", "plan")
    assert channels.is_new_to("other reader", "plan") is True
    assert channels.seen_by("reader") == {"plan": 1}
    assert channels.seen_by("other reader") == {}


def test_a_channel_nobody_wrote_is_not_new_to_anybody(channels):
    assert channels.is_new_to("reader", "never written") is False


# ------------------------------------------------------------- restoring


def test_restoring_puts_the_values_and_the_versions_back(channels):
    channels.write("plan", ["first"], by="planner")
    channels.commit()
    channels.checkpoint("before")

    channels.write("plan", ["second"], by="planner")
    channels.commit()
    assert channels.version("plan") == 2

    channels.restore("before")
    assert channels.value("plan") == ["first"]
    assert channels.version("plan") == 1


def test_restoring_drops_writes_produced_after_the_point_being_restored(channels):
    """They belong to a future being abandoned.

    Carrying one across would put a value in a channel whose version does not
    account for it.
    """
    channels.write("plan", ["first"], by="planner")
    channels.commit()
    channels.checkpoint("before")
    channels.write("plan", ["never committed"], by="planner")

    channels.restore("before")
    assert channels.pending() == []
    assert channels.value("plan") == ["first"]


def test_restoring_puts_back_what_each_node_had_seen(channels):
    channels.write("plan", 1, by="planner")
    channels.commit()
    channels.mark_seen("reader", "plan")
    channels.checkpoint("before")

    channels.write("plan", 2, by="planner")
    channels.commit()
    channels.mark_seen("reader", "plan")

    channels.restore("before")
    assert channels.seen_by("reader") == {"plan": 1}
    assert channels.is_new_to("reader", "plan") is False


def test_restoring_something_that_was_never_taken_says_so(channels):
    with pytest.raises(KeyError, match="no checkpoint"):
        channels.restore("never taken")


# ---------------------------------------------------------------- reading


def test_the_report_says_what_is_pending_and_what_the_versions_are(channels):
    channels.write("plan", 1, by="planner")
    channels.commit()
    channels.checkpoint("one")
    channels.write("mode", 2, by="router")

    report = channels.report()
    assert report["versions"] == {"plan": 1}
    assert report["pending"] == 1
    assert report["checkpoints"] == ["one"]

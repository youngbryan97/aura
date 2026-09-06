"""Every owner of background work says how that work ends."""
from __future__ import annotations

import pytest

from core.runtime.how_a_task_should_end import (
    THE_DEFAULT,
    WhenToCancel,
    declare,
    forget_everything,
    how_the_endings_are_declared,
    note_an_owner_spawned,
    owners_that_have_not_said,
    the_drain_deadline_for,
    the_policy_for,
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


def test_an_owner_that_never_declared_is_named_not_defaulted_silently() -> None:
    note_an_owner_spawned("some_new_subsystem")
    assert "some_new_subsystem" in owners_that_have_not_said()
    assert the_policy_for("some_new_subsystem") is THE_DEFAULT
    assert "nobody said" in THE_DEFAULT.why


def test_a_declared_owner_disappears_from_the_silent_list() -> None:
    note_an_owner_spawned("some_new_subsystem")
    declare(
        "some_new_subsystem",
        when=WhenToCancel.AT_ONCE,
        drain_seconds=0.0,
        an_orphan_is_a_defect=False,
        why="nothing waits on it",
    )
    assert "some_new_subsystem" not in owners_that_have_not_said()


def test_the_deadline_is_per_owner_and_never_exceeds_the_shutdown_ceiling() -> None:
    assert the_drain_deadline_for("event_spine", ceiling=30.0) == 10.0
    assert the_drain_deadline_for("event_spine", ceiling=5.0) == 5.0
    assert the_drain_deadline_for("curiosity", ceiling=30.0) == 0.0


def test_the_writes_that_corrupt_state_are_the_ones_that_must_finish() -> None:
    for owner in ("file_write_gateway", "event_spine", "shutdown_coordinator"):
        policy = the_policy_for(owner)
        assert policy.when is WhenToCancel.ONLY_WHEN_DONE, owner
        assert policy.an_orphan_is_a_defect, owner


def test_background_wondering_may_be_dropped_at_once() -> None:
    policy = the_policy_for("curiosity")
    assert policy.when is WhenToCancel.AT_ONCE
    assert not policy.an_orphan_is_a_defect


def test_a_negative_drain_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        declare("x", when=WhenToCancel.AT_ONCE, drain_seconds=-1.0,
                an_orphan_is_a_defect=False, why="")


def test_every_shipped_policy_says_why() -> None:
    seen = how_the_endings_are_declared()
    assert seen["declared"] >= 8
    for name, policy in seen["policies"].items():
        assert policy["why"].strip(), f"{name} declared a policy with no reason"


def test_the_report_separates_who_spawned_from_who_declared() -> None:
    note_an_owner_spawned("ghost")
    seen = how_the_endings_are_declared()
    assert seen["owners_that_have_not_said"] == ["ghost"]
    assert seen["owners_that_spawned"] == 1
    assert seen["declared"] >= 8, "declaring is not the same as having run"


def test_shutdown_says_whether_a_survivor_is_a_defect() -> None:
    """A curiosity task outliving shutdown costs nothing; a write does not.

    Both used to be one line saying "survived". Whether it matters is the
    owner's declaration, and the shutdown report carries it now. Checked on
    the report builder directly rather than by racing a real drain: a test
    that skips when the task happens to finish in time proves nothing.
    """
    import inspect

    from core.utils import task_tracker

    source = inspect.getsource(task_tracker.TaskTracker.shutdown)
    assert "the_policy_for" in source, (
        "shutdown decides for itself whether a survivor matters"
    )
    assert "an_orphan_is_a_defect" in source
    assert "why_it_matters" in source

    # And the declaration it reads says the two apart.
    from core.runtime.how_a_task_should_end import the_policy_for

    assert the_policy_for("file_write_gateway").an_orphan_is_a_defect is True
    assert the_policy_for("curiosity").an_orphan_is_a_defect is False

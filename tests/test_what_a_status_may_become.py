"""A transition that applied, and a predicate that matched nothing.

The comparison revised its view of AutoGPT upward and singled out two details:
a node-specific copy of the execution context before concurrent execution, and
tests that distinguish a transition actually applying from a database
predicate matching zero rows.

The second is why this module exists. A caller told False cannot tell "that
move is not allowed" from "somebody else already moved it", and those need
opposite responses — the first is a defect to fix, the second is a race to
accept.
"""
from __future__ import annotations

import pytest

from core.runtime.durable_workflow import WorkflowStatus
from core.runtime.what_a_status_may_become import (
    ATransitionTable,
    HowItWent,
    a_context_for_each,
    the_workflow_statuses,
)
from core.runtime.what_stops_it import AnExecutionContext


@pytest.fixture
def table() -> ATransitionTable:
    return the_workflow_statuses()


# ------------------------------------------------------------ three answers


def test_a_legal_move_applies(table):
    change = table.change(
        WorkflowStatus.PENDING, WorkflowStatus.PENDING, WorkflowStatus.RUNNING
    )
    assert change.went is HowItWent.APPLIED
    assert change.now is WorkflowStatus.RUNNING
    assert change


def test_an_illegal_move_is_refused_and_says_why(table):
    change = table.change(
        WorkflowStatus.PENDING, WorkflowStatus.PENDING, WorkflowStatus.COMPLETED
    )
    assert change.went is HowItWent.REFUSED
    assert "may not become" in change.why
    assert not change


def test_a_stale_belief_matches_nothing_rather_than_being_refused(table):
    """A race to accept, not a defect to fix. A boolean loses the difference."""
    change = table.change(
        WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.COMPLETED
    )
    assert change.went is HowItWent.NOTHING_MATCHED
    assert "somebody else moved it" in change.why
    assert change.now is WorkflowStatus.RUNNING


def test_the_three_outcomes_are_told_apart_by_went_not_by_truthiness(table):
    refused = table.change(
        WorkflowStatus.PENDING, WorkflowStatus.PENDING, WorkflowStatus.COMPLETED
    )
    missed = table.change(
        WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.COMPLETED
    )
    assert bool(refused) == bool(missed) is False
    assert refused.went is not missed.went


# --------------------------------------------------------- terminal states


@pytest.mark.parametrize(
    "final",
    [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELED],
)
def test_nothing_moves_out_of_a_terminal_status(table, final):
    """Nothing said a completed workflow could not go back to running."""
    change = table.change(final, final, WorkflowStatus.RUNNING)
    assert change.went is HowItWent.REFUSED
    assert "is final" in change.why


def test_the_terminal_statuses_are_the_three_endings(table):
    assert table.terminal == frozenset(
        {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELED}
    )


def test_a_table_that_gives_a_terminal_status_a_move_is_refused_at_build():
    with pytest.raises(ValueError, match="terminal and also has moves"):
        ATransitionTable(
            "broken", allowed={"done": ("running",)}, terminal=("done",)
        )


# ------------------------------------------------------------ the workflow


def test_a_paused_workflow_can_be_resumed_refused_or_shut_down(table):
    paused = WorkflowStatus.PAUSED_FOR_APPROVAL
    for wanted in (
        WorkflowStatus.RUNNING,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELED,
    ):
        assert table.may(paused, wanted), wanted


def test_a_pending_workflow_cannot_pause_for_an_approval_it_never_asked_for(table):
    assert not table.may(
        WorkflowStatus.PENDING, WorkflowStatus.PAUSED_FOR_APPROVAL
    )


def test_every_status_in_the_table_is_a_real_workflow_status(table):
    known = {str(one) for one in WorkflowStatus}
    report = table.report()
    assert set(report["statuses"]) <= known
    assert set(report["terminal"]) <= known


# ------------------------------------------------- a context per node


def test_each_node_gets_its_own_stop_signal():
    """Several nodes must not be able to mutate one shared context."""
    turn = AnExecutionContext(doing="a turn")
    nodes = a_context_for_each(turn, ["plan", "act", "check"])

    nodes["plan"].stopping.stop("planning failed")
    assert nodes["plan"].stopping.stopped
    assert not nodes["act"].stopping.stopped
    assert not turn.stopping.stopped


def test_stopping_the_turn_stops_every_node():
    turn = AnExecutionContext(doing="a turn")
    nodes = a_context_for_each(turn, ["plan", "act"])
    turn.stopping.stop("the user left")
    assert all(one.stopping.stopped for one in nodes.values())


def test_a_node_cannot_give_itself_longer_than_the_turn():
    """Compared on the deadline, not on seconds_left.

    Two reads of seconds_left are two instants, and the earlier one is always
    the larger — a test written that way measures the clock rather than the
    rule.
    """
    import time

    turn = AnExecutionContext(doing="a turn", due_by=time.monotonic() + 1.0)
    node = a_context_for_each(turn, ["slow"])["slow"]
    assert node.due_by == turn.due_by

    asked_for_longer = turn.under("node:greedy", seconds=600.0)
    assert asked_for_longer.due_by == turn.due_by


def test_each_node_says_what_it_is_doing():
    nodes = a_context_for_each(AnExecutionContext(doing="a turn"), ["plan"])
    assert nodes["plan"].doing == "node:plan"


# ------------------------------------------------------------- the wiring


def test_a_finished_workflow_is_not_put_back_to_running(tmp_path):
    """The resume path set RUNNING unconditionally, and nothing refused it."""
    import asyncio

    from core.runtime.durable_workflow import (
        WorkflowCheckpoint,
        WorkflowStatus,
        _became,
    )

    checkpoint = WorkflowCheckpoint(
        workflow_id="wf-1", objective="a thing", status=WorkflowStatus.COMPLETED
    )
    change = _became(checkpoint, WorkflowStatus.RUNNING)

    assert not change
    assert change.went is HowItWent.REFUSED
    assert checkpoint.status is WorkflowStatus.COMPLETED
    del asyncio


def test_a_legal_workflow_move_still_applies():
    from core.runtime.durable_workflow import WorkflowCheckpoint, _became

    checkpoint = WorkflowCheckpoint(
        workflow_id="wf-2", objective="a thing", status=WorkflowStatus.RUNNING
    )
    assert _became(checkpoint, WorkflowStatus.COMPLETED)
    assert checkpoint.status is WorkflowStatus.COMPLETED


def test_the_workflow_runner_returns_rather_than_restarting_a_finished_one():
    """Read from the source: running it needs a store and a step list."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core" / "runtime" / "durable_workflow.py"
    ).read_text("utf-8")
    assert "checkpoint.status = WorkflowStatus.RUNNING" not in source
    assert "_became(checkpoint, WorkflowStatus.RUNNING)" in source

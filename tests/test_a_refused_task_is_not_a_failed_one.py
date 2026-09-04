"""A batch says which tasks were refused, and says it before anything runs.

The parallel executor launched everything and let each task meet governance
on its own. Two things followed. A task that was never going to be allowed
spent the concurrency and the wall clock anyway. And a refusal came back
inside the fan-out as one more not-completed task, so `failed_count` counted
it — telling the autonomy layer to retry something that will be refused
again, when what a refusal calls for is a different plan.

The distinction the partition rests on is three-valued, not two: allowed,
refused, and cannot say. Governance being unavailable is not a refusal, and
collapsing the two either blocks the batch when governance is down or runs
it when governance says no.
"""

from __future__ import annotations

import asyncio

import pytest

from core.agency.parallel_executor import (
    ParallelExecutor,
    ParallelTask,
    _NotAllowed,
)
from core.skills.fluid_executor import ExecutionReceipt


def _an_executor(**kw) -> ParallelExecutor:
    class _Ran:
        async def run(self, goal, steps):
            return ExecutionReceipt(goal=goal, completed=True)

    kw.setdefault("executor_factory", _Ran)
    return ParallelExecutor(**kw)


def _tasks(*goals: str) -> list[ParallelTask]:
    return [ParallelTask(goal=one, steps=[]) for one in goals]


@pytest.mark.asyncio
async def test_a_refused_task_never_starts():
    started: list[str] = []

    class _Ran:
        async def run(self, goal, steps):
            started.append(goal)
            return ExecutionReceipt(goal=goal, completed=True)

    async def says(task):
        return True if task.goal == "allowed" else _NotAllowed(None)

    runner = ParallelExecutor(executor_factory=_Ran, may_run=says)
    receipt = await runner.run(_tasks("allowed", "refused"))
    assert started == ["allowed"]
    assert [one.goal for one in receipt.refused] == ["refused"]


@pytest.mark.asyncio
async def test_a_refusal_is_not_counted_as_a_failure():
    """The whole point: retry and replan are opposite responses."""

    async def says(task):
        return _NotAllowed(None)

    receipt = await _an_executor(may_run=says).run(_tasks("one", "two"))
    assert receipt.failed_count == 0
    assert receipt.refused_count == 2


@pytest.mark.asyncio
async def test_a_batch_with_a_refusal_did_not_all_complete():
    async def says(task):
        return True if task.goal == "allowed" else _NotAllowed(None)

    receipt = await _an_executor(may_run=says).run(_tasks("allowed", "refused"))
    assert receipt.completed_count == 1
    assert not receipt.all_completed


@pytest.mark.asyncio
async def test_governance_that_cannot_answer_is_not_a_refusal():
    """Unavailable degrades to the behaviour before the partition existed."""

    async def cannot_say(task):
        return None

    receipt = await _an_executor(may_run=cannot_say).run(_tasks("one", "two"))
    assert receipt.refused_count == 0
    assert receipt.completed_count == 2


@pytest.mark.asyncio
async def test_a_check_that_raises_does_not_block_the_batch():
    """Each executor still asks for itself, so a broken check is not a veto."""

    async def raises(task):
        raise RuntimeError("the will is down")

    receipt = await _an_executor(may_run=raises).run(_tasks("one"))
    assert receipt.refused_count == 0
    assert receipt.completed_count == 1


@pytest.mark.asyncio
async def test_a_refusal_carries_why():
    class _Said:
        reason = "that domain is sealed"

    async def says(task):
        return _NotAllowed(_Said())

    receipt = await _an_executor(may_run=says).run(_tasks("one"))
    assert receipt.refused[0].reason == "that domain is sealed"
    assert receipt.to_dict()["refusals"] == [
        {"goal": "one", "reason": "that domain is sealed"}
    ]


@pytest.mark.asyncio
async def test_a_batch_refused_entirely_runs_nothing_and_says_so():
    started: list[str] = []

    class _Ran:
        async def run(self, goal, steps):
            started.append(goal)
            return ExecutionReceipt(goal=goal, completed=True)

    async def says(task):
        return _NotAllowed(None)

    receipt = await ParallelExecutor(executor_factory=_Ran, may_run=says).run(
        _tasks("one", "two", "three")
    )
    assert started == []
    assert receipt.refused_count == 3
    assert receipt.tasks == []


def test_not_allowed_is_falsy_and_still_carries_its_reason():
    """The same shape the growth ladder needed: a refusal that reads as one."""
    said = _NotAllowed(None)
    assert not said
    assert said.reason == "refused"

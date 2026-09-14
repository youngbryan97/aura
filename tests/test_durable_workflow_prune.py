"""A workflow's history can be pruned without cutting a branch off its parent.

Every step writes a revision, so a long workflow's history grows for as long
as the workflow runs. Pruning is what keeps that finite. What makes it safe is
that a fork names the revision it came from, and a prune that removed that
file would leave a branch whose parent cannot be read.
"""
from __future__ import annotations

import pytest

from core.runtime.durable_workflow import (
    DurableWorkflowEngine,
    WorkflowStep,
    WorkflowStore,
)

pytestmark = pytest.mark.unit


def _steps(many: int) -> list[WorkflowStep]:
    return [
        WorkflowStep(step_id=f"s{i}", name=f"s{i}", apply=lambda o, i=i: i)
        for i in range(many)
    ]


@pytest.mark.asyncio
async def test_prune_keeps_the_newest_revisions(tmp_path):
    store = WorkflowStore(root=tmp_path)
    engine = DurableWorkflowEngine(store=store)
    await engine.run("count", _steps(6), workflow_id="wf")
    before = [one.revision for one in store.history("wf")]
    assert len(before) >= 6

    removed = store.prune("wf", keep=2)

    after = [one.revision for one in store.history("wf")]
    assert after == sorted(before)[-2:]
    assert sorted(removed) == sorted(before)[:-2]
    assert store.load("wf") is not None, "the latest state is not history"


@pytest.mark.asyncio
async def test_prune_never_removes_a_revision_a_fork_was_taken_from(tmp_path):
    store = WorkflowStore(root=tmp_path)
    engine = DurableWorkflowEngine(store=store)
    await engine.run("count", _steps(6), workflow_id="wf")
    oldest = min(one.revision for one in store.history("wf"))
    await engine.fork("wf", at_revision=oldest, new_workflow_id="branch")

    store.prune("wf", keep=1)

    kept = {one.revision for one in store.history("wf")}
    assert oldest in kept, "the branch's parent revision was pruned"
    assert store.load_revision("wf", oldest) is not None


def test_prune_refuses_to_keep_nothing(tmp_path):
    with pytest.raises(ValueError):
        WorkflowStore(root=tmp_path).prune("wf", keep=0)


def test_pruning_an_unknown_workflow_removes_nothing(tmp_path):
    assert WorkflowStore(root=tmp_path).prune("nobody", keep=3) == []

import asyncio
from types import SimpleNamespace

import pytest

from core.agency.task_commitment_verifier import (
    CapabilityAssessment,
    DispatchOutcome,
    TaskCommitmentVerifier,
)


class _GoalTracker:
    def __init__(self):
        self.dispatches = []
        self.updates = []

    async def track_dispatch(self, objective, **kwargs):
        self.dispatches.append((objective, kwargs))
        return {"ok": True}

    async def update_task_lifecycle(self, **kwargs):
        self.updates.append(kwargs)
        return kwargs


class _SlowTaskEngine:
    def __init__(self):
        self.finished = False
        self.cancelled = False

    async def execute(self, goal, context=None):
        try:
            await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.finished = True
        return SimpleNamespace(succeeded=True, summary=f"completed {goal}", goal=goal)


@pytest.mark.asyncio
async def test_inline_timeout_keeps_task_running_in_background(monkeypatch):
    tracker = _GoalTracker()
    task_engine = _SlowTaskEngine()

    def _fake_get(name, default=None):
        if name == "task_engine":
            return task_engine
        if name == "goal_engine":
            return tracker
        return default

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(_fake_get),
    )
    monkeypatch.setattr(
        TaskCommitmentVerifier,
        "_assess_capability",
        lambda self, objective: CapabilityAssessment(can_fulfil=True, matched_tools=["think"], confidence=1.0),
    )
    monkeypatch.setattr(TaskCommitmentVerifier, "_register_commitment", lambda self, objective: None)

    verifier = TaskCommitmentVerifier(kernel=None)
    verifier.INLINE_TIMEOUT_S = 0.02

    acceptance = await verifier.verify_and_dispatch("keep going under pressure", state=None)

    assert acceptance.outcome == DispatchOutcome.STARTED
    await asyncio.sleep(0.15)

    status = verifier.get_task_status(acceptance.task_id)
    assert status is not None
    assert status["status"] == "completed"
    assert task_engine.finished is True
    assert task_engine.cancelled is False
    assert tracker.dispatches
    assert any(update["status"] == "completed" for update in tracker.updates)

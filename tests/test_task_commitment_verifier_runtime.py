from core.runtime.atomic_writer import atomic_write_text
import asyncio
import time
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


class _FastTaskEngine:
    def __init__(self):
        self.goals = []

    async def execute(self, goal, context=None):
        self.goals.append((goal, context or {}))
        return SimpleNamespace(succeeded=True, summary=f"completed {goal}", goal=goal)

    def get_active_plans(self):
        return []


@pytest.mark.asyncio
async def test_inline_timeout_keeps_task_running_in_background(monkeypatch, tmp_path):
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

    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")
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


@pytest.mark.asyncio
async def test_task_commitment_verifier_continues_relevant_task_for_short_followup(monkeypatch, tmp_path):
    tracker = _GoalTracker()
    task_engine = _FastTaskEngine()

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

    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")
    verifier._store_task_entry(
        "task-prev",
        {
            "task_id": "task-prev",
            "objective": "Fix the failing pytest in core/runtime/conversation_support.py",
            "status": "interrupted",
            "started_at": 10.0,
        },
    )

    acceptance = await verifier.verify_and_dispatch("Let's do it", state=None)

    assert acceptance.outcome == DispatchOutcome.COMPLETED
    assert acceptance.objective == "Fix the failing pytest in core/runtime/conversation_support.py"
    assert task_engine.goals[0][0] == "Fix the failing pytest in core/runtime/conversation_support.py"


@pytest.mark.asyncio
async def test_task_commitment_verifier_does_not_duplicate_running_task_on_continue(monkeypatch, tmp_path):
    tracker = _GoalTracker()
    task_engine = _FastTaskEngine()

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

    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")
    verifier._store_task_entry(
        "task-live",
        {
            "task_id": "task-live",
            "objective": "Patch the task follow-up lane",
            "status": "running_async",
            "summary": "Background verification is still running.",
            "started_at": 10.0,
        },
    )

    acceptance = await verifier.verify_and_dispatch("keep going", state=None)

    assert acceptance.outcome == DispatchOutcome.STARTED
    assert acceptance.task_id == "task-live"
    assert "already working on" in acceptance.summary.lower()
    assert task_engine.goals == []


@pytest.mark.asyncio
async def test_task_commitment_verifier_resumes_recovered_task_engine_plan(monkeypatch, tmp_path):
    tracker = _GoalTracker()

    class _RecoverableTaskEngine(_FastTaskEngine):
        def get_active_plans(self):
            return [
                {
                    "plan_id": "plan-recover",
                    "task_id": "plan-recover",
                    "goal": "Patch the interrupted runtime lane",
                    "status": "interrupted",
                    "summary": "Interrupted before verification completed.",
                    "steps_completed": 1,
                    "steps_total": 3,
                }
            ]

    task_engine = _RecoverableTaskEngine()

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

    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")

    acceptance = await verifier.verify_and_dispatch("Let's do it", state=None)

    assert acceptance.outcome == DispatchOutcome.COMPLETED
    assert acceptance.objective == "Patch the interrupted runtime lane"
    assert task_engine.goals[0][1]["resume_plan_id"] == "plan-recover"


def test_task_commitment_verifier_context_block_surfaces_relevant_status(tmp_path):
    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")
    verifier._active_tasks = {
        "task-a": {
            "task_id": "task-a",
            "objective": "Fix the failing pytest in core/runtime/conversation_support.py",
            "status": "running_async",
            "started_at": 10.0,
        },
        "task-b": {
            "task_id": "task-b",
            "objective": "Refactor logging in core/orchestrator/mixins/tool_execution.py",
            "status": "completed",
            "summary": "Patched log formatting and verified tests.",
            "completed_at": 20.0,
            "cleanup_at": time.time() + 300.0,
        },
    }

    block = verifier.get_context_block("Are you done fixing the failing pytest in core/runtime/conversation_support.py?")

    assert "## TASK CONTINUITY" in block
    assert "[task-a]" in block
    assert "running_async" in block
    assert "Fix the failing pytest" in block


def test_task_commitment_verifier_persistence_marks_running_tasks_interrupted(tmp_path):
    path = tmp_path / "task_commitment_state.json"
    atomic_write_text(path, 
        """
{
  "updated_at": 10.0,
  "active_tasks": [
    {
      "task_id": "task-a",
      "objective": "Fix the failing pytest",
      "status": "running_async",
      "started_at": 5.0
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    verifier = TaskCommitmentVerifier(kernel=None, persist_path=path)
    status = verifier.get_task_status("task-a")

    assert status is not None
    assert status["status"] == "interrupted"
    assert "interrupted" in status["summary"].lower()


def test_task_commitment_verifier_builds_grounded_status_reply(tmp_path):
    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")
    verifier._store_task_entry(
        "task-a",
        {
            "task_id": "task-a",
            "objective": "Fix the failing pytest in core/runtime/conversation_support.py",
            "status": "running_async",
            "summary": "pytest is still running against the patched file.",
            "started_at": 10.0,
        },
    )

    reply = verifier.build_status_reply(
        "Are you done fixing the failing pytest in core/runtime/conversation_support.py?"
    )

    assert "still running" in reply.lower()
    assert "conversation_support.py" in reply

import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agency.commitment_engine import CommitmentStatus
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
    assert "task ledger is tracking completion status" in acceptance.summary.lower()
    assert "i'll" not in acceptance.summary.lower()
    assert "keep tracking" not in acceptance.summary.lower()
    await asyncio.sleep(0.15)

    status = verifier.get_task_status(acceptance.task_id)
    assert status is not None
    assert status["status"] == "completed"
    assert task_engine.finished is True
    assert task_engine.cancelled is False
    assert tracker.dispatches
    assert any(update["status"] == "completed" for update in tracker.updates)


def test_task_commitment_verifier_rejects_attempt_only_result_as_completed():
    result = SimpleNamespace(
        succeeded=True,
        summary="I attempted to open Terminal and type the command, but I could not verify it completed.",
        steps_completed=1,
        steps_total=1,
    )

    assert TaskCommitmentVerifier._result_counts_as_success(result) is False


def test_task_commitment_verifier_estimates_multistep_skill_request_as_long():
    persist_path = Path(tempfile.gettempdir()) / "task_commitment_state_estimate.json"
    verifier = TaskCommitmentVerifier(kernel=None, persist_path=persist_path)
    assessment = CapabilityAssessment(can_fulfil=True, matched_skills=["computer_use"], confidence=1.0)

    steps = verifier._estimate_steps(
        "Open Notes, click into a new note, type hello, then come back and report what happened.",
        assessment,
    )

    assert steps > verifier.INLINE_STEP_THRESHOLD


@pytest.mark.asyncio
async def test_task_commitment_verifier_fails_closed_without_capability_or_task_registry(monkeypatch, tmp_path):
    def _fake_get(name, default=None):
        return default

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(_fake_get),
    )

    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")

    acceptance = await verifier.verify_and_dispatch("Open a browser and research climate news.", state=None)

    assert acceptance.outcome == DispatchOutcome.CAPABILITY_GAP
    assert "capabilityengine unavailable" in acceptance.summary.lower()
    assert "no task-engine tool registry" in acceptance.summary.lower()


@pytest.mark.asyncio
async def test_task_commitment_verifier_uses_bounded_task_registry_when_capability_engine_unavailable(
    monkeypatch,
    tmp_path,
):
    task_engine = _FastTaskEngine()
    task_engine._tool_registry = {"think": object(), "web_search": object()}

    def _fake_get(name, default=None):
        if name == "task_engine":
            return task_engine
        return default

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(_fake_get),
    )
    monkeypatch.setattr(TaskCommitmentVerifier, "_register_commitment", lambda self, objective: None)

    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")

    acceptance = await verifier.verify_and_dispatch("Research three current climate articles.", state=None)

    assert acceptance.outcome == DispatchOutcome.COMPLETED
    assert task_engine.goals
    assert task_engine.goals[0][1]["matched_tools"] == ["think", "web_search"]


@pytest.mark.asyncio
async def test_task_commitment_verifier_does_not_bypass_approval_required_skill_with_builtin_tools(
    monkeypatch,
    tmp_path,
):
    task_engine = _FastTaskEngine()
    task_engine._tool_registry = {"desktop_task": object(), "web_search": object()}
    capability_engine = SimpleNamespace(
        detect_intent=lambda _objective: ["desktop_task"],
        get=lambda _skill: SimpleNamespace(instance=SimpleNamespace(requires_approval=True)),
        list_skills=lambda: ["desktop_task", "web_search"],
    )

    def _fake_get(name, default=None):
        if name == "task_engine":
            return task_engine
        if name == "capability_engine":
            return capability_engine
        return default

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(_fake_get),
    )

    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")

    acceptance = await verifier.verify_and_dispatch("Open Notes and export a PDF.", state=None)

    assert acceptance.outcome == DispatchOutcome.CAPABILITY_GAP
    assert "requires approval" in acceptance.summary.lower()
    assert "desktop_task" in acceptance.summary
    assert task_engine.goals == []


def test_task_commitment_verifier_keeps_learning_bundle_on_inline_deterministic_path(monkeypatch, tmp_path):
    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")
    monkeypatch.setattr(
        verifier,
        "_get_cap_engine",
        lambda: SimpleNamespace(detect_intent=lambda _objective: ["sovereign_terminal", "run_code"]),
    )

    objective = """
Priority of how to consume content
Prioritize watching using the visual and auditory cortices
If unsuccessful at physically watching, try to find a script
If finding a script is unsuccessful, try finding a transcript

General Education:
Kurzgesagt - In a Nutshell (https://www.youtube.com/@kurzgesagt): They explain the universe with logic and color.
PolyMatter (https://www.youtube.com/@PolyMatter): Essays on geopolitics and economics.
TED (https://www.youtube.com/@TED): Short talks by experts.

TV Shows and Movies about Artificial Intelligence:
Ghost in the Shell - Masamune Shirow: If you replace your body parts, are you still you?
Pantheon - Craig Silverstein: Uploaded intelligence and continuity questions.
Wall-E - Andrew Stanton: A robot learning to care for something small.
""".strip()

    assessment = verifier._assess_capability(objective)
    steps = verifier._estimate_steps(objective, assessment)

    assert assessment.can_fulfil is True
    assert assessment.matched_skills == []
    assert assessment.matched_tools == []
    assert steps <= verifier.INLINE_STEP_THRESHOLD


@pytest.mark.asyncio
async def test_task_commitment_verifier_passes_user_origin_into_task_context(monkeypatch, tmp_path):
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
    state = SimpleNamespace(
        cognition=SimpleNamespace(current_origin="api"),
        response_modifiers={},
        transition_origin="system",
    )

    acceptance = await verifier.verify_and_dispatch("Check the current runtime status", state=state)

    assert acceptance.outcome == DispatchOutcome.COMPLETED
    assert task_engine.goals[0][1]["origin"] == "api"
    assert task_engine.goals[0][1]["intent_source"] == "api"
    assert task_engine.goals[0][1]["request_origin"] == "api"


@pytest.mark.asyncio
async def test_task_commitment_verifier_async_acceptance_is_evidence_bounded(monkeypatch, tmp_path):
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
        lambda self, objective: CapabilityAssessment(can_fulfil=True, matched_tools=["desktop_task"], confidence=1.0),
    )
    monkeypatch.setattr(TaskCommitmentVerifier, "_register_commitment", lambda self, objective: "commit-123")

    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")

    acceptance = await verifier.verify_and_dispatch("Prepare a visible desktop workflow", state=None, force_async=True)
    await asyncio.sleep(0)

    assert acceptance.outcome == DispatchOutcome.STARTED
    assert acceptance.commitment_id == "commit-123"
    assert "task accepted into governed background execution" in acceptance.summary.lower()
    assert "task ledger is tracking completion status" in acceptance.summary.lower()
    assert "no completion is claimed yet" in acceptance.summary.lower()
    assert "i'll" not in acceptance.summary.lower()
    assert "follow up" not in acceptance.summary.lower()
    assert tracker.dispatches


@pytest.mark.asyncio
async def test_task_commitment_verifier_preserves_structured_multiline_goal(monkeypatch, tmp_path):
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
        lambda self, objective: CapabilityAssessment(can_fulfil=True, matched_tools=["remember"], confidence=1.0),
    )
    monkeypatch.setattr(TaskCommitmentVerifier, "_register_commitment", lambda self, objective: None)

    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")
    goal = (
        "I have some suggestions for you.\n\n"
        "General Education:\n"
        "Kurzgesagt (https://example.com/k): Explains science beautifully.\n"
        "TED (https://example.com/t): Talks from experts.\n"
    )

    acceptance = await verifier.verify_and_dispatch(goal, state=None)

    assert acceptance.outcome == DispatchOutcome.COMPLETED
    dispatched_goal = task_engine.goals[0][0]
    assert dispatched_goal.startswith("I have some suggestions for you.\n\n")
    assert "\nGeneral Education:\n" in dispatched_goal
    assert "Kurzgesagt (https://example.com/k): Explains science beautifully." in dispatched_goal


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
    path.write_text(
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


@pytest.mark.asyncio
async def test_task_commitment_persistence_runs_off_loop_without_state_lock(
    monkeypatch,
    tmp_path,
):
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.runtime.lockdep import get_validator

    verifier = TaskCommitmentVerifier(
        kernel=None,
        persist_path=tmp_path / "task_commitment_state.json",
    )
    loop_thread = threading.get_ident()
    observed = {}
    gateway = get_file_write_gateway()
    real_write_text = gateway.write_text

    def observing_write_text(path, text, **kwargs):
        observed["thread"] = threading.get_ident()
        observed["held"] = get_validator().held_names()
        observed["governance"] = get_active_governance()
        return real_write_text(path, text, **kwargs)

    monkeypatch.setattr(gateway, "write_text", observing_write_text)

    await verifier._store_task_entry_async(
        "task-off-loop",
        {
            "objective": "Prove persistence ownership",
            "status": "running_async",
        },
    )

    assert observed["thread"] != loop_thread
    assert "task_commitment.state" not in observed["held"]
    assert observed["held"] == []
    assert observed["governance"] is not None
    assert observed["governance"].source == (
        "task_commitment_verifier.persist_snapshot"
    )
    assert observed["governance"].domain == "state_mutation"


@pytest.mark.asyncio
async def test_task_commitment_persistence_rejects_stale_snapshot(tmp_path):
    path = tmp_path / "task_commitment_state.json"
    verifier = TaskCommitmentVerifier(kernel=None, persist_path=path)

    older = verifier._store_task_entry_state(
        "task-generation",
        {
            "objective": "Preserve the newest durable task state",
            "status": "running_async",
        },
    )
    newer = verifier._update_task_entry_state(
        "task-generation",
        status="completed",
        summary="The newest generation is authoritative.",
    )
    assert newer is not None

    await verifier._persist_task_snapshot_async(*newer)
    await verifier._persist_task_snapshot_async(*older)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["active_tasks"][0]["status"] == "completed"
    assert persisted["active_tasks"][0]["summary"] == (
        "The newest generation is authoritative."
    )
    assert verifier._persisted_generation == newer[0]


def test_task_commitment_verifier_quarantines_evaluation_fixture_from_lived_state(tmp_path):
    path = tmp_path / "task_commitment_state.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": 10.0,
                "active_tasks": [
                    {
                        "task_id": "proof-task",
                        "objective": (
                            "A long-running microservice periodically crashes with OSError; "
                            "code review reveals a resource leak."
                        ),
                        "status": "running_async",
                    },
                    {
                        "task_id": "user-task",
                        "objective": "Organize Bryan's project notes",
                        "status": "running_async",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    verifier = TaskCommitmentVerifier(kernel=None, persist_path=path)

    assert verifier.get_task_status("proof-task") is None
    assert verifier.get_task_status("user-task")["status"] == "interrupted"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert [entry["task_id"] for entry in persisted["active_tasks"]] == ["user-task"]


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


@pytest.mark.asyncio
async def test_async_failure_breaks_commitment_instead_of_leaving_it_active(monkeypatch, tmp_path):
    verifier = TaskCommitmentVerifier(kernel=None, persist_path=tmp_path / "task_commitment_state.json")

    class _BrokenTaskEngine:
        async def execute(self, goal, context=None):
            return SimpleNamespace(succeeded=False, summary=f"failed {goal}")

    def _fake_get(name, default=None):
        if name == "task_engine":
            return _BrokenTaskEngine()
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

    from core.agency import commitment_engine as commitment_module

    monkeypatch.setattr(commitment_module, "PERSIST_PATH", tmp_path / "commitments.json")
    monkeypatch.setattr(commitment_module, "_engine", None)

    acceptance = await verifier.verify_and_dispatch("Investigate the broken runtime lane", state=None, force_async=True)
    assert acceptance.outcome == DispatchOutcome.STARTED

    await asyncio.sleep(0.05)

    commitment = commitment_module.get_commitment_engine()._commitments[acceptance.commitment_id]
    assert commitment.status == CommitmentStatus.BROKEN
    assert any("Task failed" in note for note in commitment.notes)

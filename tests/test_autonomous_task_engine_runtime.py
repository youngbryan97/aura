import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.agency.autonomous_task_engine import AutonomousTaskEngine, StepStatus, TaskPlan, TaskResult, TaskStep


@pytest.mark.asyncio
async def test_task_engine_fallback_plan_survives_malformed_decomposition():
    llm = SimpleNamespace(think=AsyncMock(return_value='[{"description": "broken"'))
    kernel = SimpleNamespace(
        organs={"llm": SimpleNamespace(get_instance=lambda: llm)}
    )

    engine = AutonomousTaskEngine(kernel)

    plan = await engine._decompose_goal("Inspect runtime health", "plan_test", context=None)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "think"
    assert plan.steps[0].rollback_action is None
    llm.think.assert_awaited()


@pytest.mark.asyncio
async def test_task_engine_grounded_goal_fallback_avoids_think_only_plan():
    llm = SimpleNamespace(think=AsyncMock(return_value='[{"description": "broken"'))
    kernel = SimpleNamespace(
        organs={"llm": SimpleNamespace(get_instance=lambda: llm)}
    )

    engine = AutonomousTaskEngine(kernel)

    plan = await engine._decompose_goal(
        "Open the Terminal app on my computer, type exactly: echo AURA_SKILL_LIVE_TEST, press Return, then come back and report what happened.",
        "plan_grounded",
        context={"matched_skills": ["computer_use"]},
    )

    assert len(plan.steps) >= 3
    assert all(step.tool != "think" for step in plan.steps)
    assert plan.steps[0].tool == "computer_use"
    assert plan.steps[0].args["action"] == "open_app"
    assert any(step.args.get("action") == "type" for step in plan.steps)
    assert any(step.args.get("action") == "hotkey" and step.args.get("target") == "enter" for step in plan.steps)


@pytest.mark.asyncio
async def test_task_engine_invoke_tool_preserves_user_origin_for_orchestrator(monkeypatch):
    calls = []

    class _FakeOrchestrator:
        async def execute_tool(self, tool_name, args, **kwargs):
            calls.append((tool_name, args, kwargs))
            return {"ok": True, "verified": True}

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return _FakeOrchestrator()
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(_fake_get))

    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: None)}, state=None)
    engine = AutonomousTaskEngine(kernel)
    engine._capability_manager = SimpleNamespace(verify_access=lambda *_args, **_kwargs: True)

    result = await engine._invoke_tool(
        "computer_use",
        {"action": "open_app", "target": "Terminal"},
        origin="api",
    )

    assert result["ok"] is True
    assert calls[0][0] == "computer_use"
    assert calls[0][2]["origin"] == "api"


@pytest.mark.asyncio
async def test_task_engine_execute_alias_delegates_to_execute_goal():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine.execute_goal = AsyncMock(return_value="ok")

    result = await AutonomousTaskEngine.execute(
        engine,
        "Keep the runtime stable",
        context={"task_id": "task-1"},
        is_shadow=True,
    )

    assert result == "ok"
    engine.execute_goal.assert_awaited_once_with(
        goal="Keep the runtime stable",
        context={"task_id": "task-1"},
        on_progress=None,
        is_shadow=True,
    )


@pytest.mark.asyncio
async def test_task_engine_records_execution_repair_pressure(monkeypatch):
    events: list[tuple[str, dict]] = []

    class DummyRecorder:
        def record_execution_step(self, **kwargs):
            events.append(("step", kwargs))

        def record_execution_repair(self, **kwargs):
            events.append(("repair", kwargs))

    monkeypatch.setattr(
        "core.runtime.coding_session_memory.get_coding_session_memory",
        lambda: DummyRecorder(),
    )

    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine._invoke_tool = AsyncMock(return_value={"ok": True, "stdout": "still failing"})
    engine._verify_step = AsyncMock(side_effect=[False, True])
    engine._get_alternative_approach = AsyncMock(return_value={"command": "pytest tests/test_runtime_service_access.py -q"})

    step = TaskStep(
        step_id="plan-1_s0",
        description="Re-run the failing pytest",
        tool="sovereign_terminal",
        args={"command": "pytest tests/test_runtime_service_access.py -q"},
        success_criterion="pytest output contains '1 passed'",
    )
    plan = TaskPlan(plan_id="plan-1", goal="Fix the failing pytest", steps=[step], trace_id="trace")

    await AutonomousTaskEngine._execute_step_with_retry(engine, step, plan)

    assert step.status.value == "succeeded"
    assert any(kind == "step" and item["status"] == "verification_failed" for kind, item in events)
    assert any(kind == "repair" for kind, _item in events)


@pytest.mark.asyncio
async def test_task_engine_verify_step_uses_deterministic_result_checks_before_llm():
    llm = SimpleNamespace(think=AsyncMock(return_value="NO"))
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine.kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    failing_step = TaskStep(
        step_id="s0",
        description="Run pytest",
        tool="sovereign_terminal",
        args={},
        success_criterion="pytest output contains '1 passed'",
    )
    assert await AutonomousTaskEngine._verify_step(
        engine,
        failing_step,
        {"ok": False, "stderr": "AssertionError"},
    ) is False

    passing_step = TaskStep(
        step_id="s1",
        description="Confirm the expected token is present",
        tool="think",
        args={},
        success_criterion="result contains '1 passed'",
    )
    assert await AutonomousTaskEngine._verify_step(
        engine,
        passing_step,
        {"ok": True, "stdout": "1 passed in 0.40s"},
    ) is True
    llm.think.assert_not_awaited()


def test_task_engine_loads_interrupted_plan_snapshot_as_resumable(tmp_path):
    path = tmp_path / "task_engine_active_plans.json"
    step_done = TaskStep(
        step_id="plan-restore_s0",
        description="Inspect the failing assertion",
        tool="read_file",
        args={"path": "core/runtime/conversation_support.py"},
        success_criterion="result contains 'context'",
        status=StepStatus.SUCCEEDED,
        verified=True,
    )
    step_running = TaskStep(
        step_id="plan-restore_s1",
        description="Re-run the failing pytest",
        tool="sovereign_terminal",
        args={"command": "pytest tests/test_runtime_service_access.py -q"},
        success_criterion="pytest output contains '1 passed'",
        depends_on=["plan-restore_s0"],
        status=StepStatus.RUNNING,
        attempts=1,
        error="AssertionError: expected coding block",
    )
    persisted_plan = TaskPlan(
        plan_id="plan-restore",
        goal="Fix the failing pytest in core/runtime/conversation_support.py",
        steps=[step_done, step_running],
        trace_id="trace-old",
        context={"task_id": "task-restore"},
        status="running",
    )
    path.write_text(
        json.dumps({"updated_at": 10.0, "plans": [persisted_plan.to_runtime_dict()]}),
        encoding="utf-8",
    )

    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine.kernel = SimpleNamespace(state=None)
    engine._active_plans = {}
    engine._persist_path = path
    engine._update_state_goals = lambda plan: None

    AutonomousTaskEngine._load_persisted_active_plans(engine)

    restored = engine._active_plans["plan-restore"]
    assert restored.status == "interrupted"
    assert restored.context["recovered_after_restart"] is True
    assert restored.steps[0].status == StepStatus.SUCCEEDED
    assert restored.steps[1].status == StepStatus.PENDING
    assert "Interrupted" in restored.steps[1].error


@pytest.mark.asyncio
async def test_task_engine_execute_plan_resumes_from_completed_steps():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine._safety_registry = SimpleNamespace(is_allowed=AsyncMock(return_value=True))
    engine._persist_plan_state = lambda plan: None
    engine._can_run_in_parallel = lambda step: False
    engine._report_progress = lambda step, on_progress: None
    engine._fail_plan = AsyncMock()

    async def _execute_step(step, plan):
        step.status = StepStatus.SUCCEEDED
        step.verified = True

    engine._execute_step_with_retry = _execute_step

    step_done = TaskStep(
        step_id="plan-resume_s0",
        description="Inspect the failing assertion",
        tool="read_file",
        args={"path": "core/runtime/conversation_support.py"},
        success_criterion="result contains 'context'",
        status=StepStatus.SUCCEEDED,
        verified=True,
    )
    step_pending = TaskStep(
        step_id="plan-resume_s1",
        description="Re-run the failing pytest",
        tool="sovereign_terminal",
        args={"command": "pytest tests/test_runtime_service_access.py -q"},
        success_criterion="pytest output contains '1 passed'",
        depends_on=["plan-resume_s0"],
    )
    plan = TaskPlan(
        plan_id="plan-resume",
        goal="Fix the failing pytest",
        steps=[step_done, step_pending],
        trace_id="trace",
        status="interrupted",
    )

    await AutonomousTaskEngine._execute_plan(engine, plan, on_progress=None)

    assert step_pending.status == StepStatus.SUCCEEDED
    assert plan.status == "succeeded"

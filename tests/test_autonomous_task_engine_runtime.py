import json
from collections import deque
from types import SimpleNamespace

import pytest

from core.agency.autonomous_task_engine import (
    AutonomousTaskEngine,
    StepStatus,
    TaskExecutionDeferred,
    TaskPlan,
    TaskStep,
)


class AsyncCallRecorder:
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = deque(side_effect) if isinstance(side_effect, list) else side_effect
        self.await_count = 0
        self.await_args_list = []

    async def __call__(self, *args, **kwargs):
        self.await_count += 1
        self.await_args_list.append((args, kwargs))
        if isinstance(self.side_effect, deque):
            effect = self.side_effect.popleft()
            if isinstance(effect, BaseException):
                raise effect
            return effect
        if callable(self.side_effect):
            return self.side_effect(*args, **kwargs)
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        return self.return_value

    def assert_awaited(self):
        assert self.await_count > 0

    def assert_not_awaited(self):
        assert self.await_count == 0

    def assert_awaited_once_with(self, *args, **kwargs):
        assert self.await_count == 1
        assert self.await_args_list[0] == (args, kwargs)


@pytest.mark.asyncio
async def test_task_engine_fallback_plan_survives_malformed_decomposition():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value='[{"description": "broken"'))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    engine = AutonomousTaskEngine(kernel)

    plan = await engine._decompose_goal("Inspect runtime health", "plan_test", context=None)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "think"
    assert plan.steps[0].rollback_action is None
    llm.think.assert_awaited()


@pytest.mark.asyncio
async def test_task_engine_recoverable_decomposition_failure_does_not_trip_fail_closed(monkeypatch):
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value=""))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})
    recorded = []

    def fake_record_degradation(subsystem, error, **kwargs):
        recorded.append((subsystem, kwargs.get("action")))
        if subsystem == "autonomous_task_engine":
            raise RuntimeError("fail-closed planner abort")

    monkeypatch.setattr(
        "core.agency.autonomous_task_engine.record_degradation",
        fake_record_degradation,
    )

    engine = AutonomousTaskEngine(kernel)

    plan = await engine._decompose_goal("Inspect runtime health", "plan_empty", context=None)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "think"
    assert recorded == [
        (
            "autonomous_task_engine_planning",
            "used deterministic planner fallback after model decomposition failure",
        )
    ]


@pytest.mark.asyncio
async def test_task_engine_treats_router_admission_deferral_as_deferred_not_failure(
    monkeypatch,
):
    from core.brain.llm import deferral_record

    llm = SimpleNamespace(think=AsyncCallRecorder(return_value=""))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})
    recorded = []
    monkeypatch.setattr(
        "core.agency.autonomous_task_engine.record_degradation",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )
    deferral_record.reset_for_test()

    async def deferred_think(*args, **kwargs):
        deferral_record.record_deferral(
            origin="autonomous_task_engine",
            reason="foreground_quiet_window",
        )
        return ""

    llm.think = deferred_think
    try:
        engine = AutonomousTaskEngine(kernel)
        result = await engine.execute_goal("Inspect runtime health")
    finally:
        deferral_record.reset_for_test()

    assert result.succeeded is False
    assert result.steps_total == 0
    assert result.deferred_reason == "foreground_quiet_window"
    assert "queued" in result.summary.lower()
    assert recorded == []


@pytest.mark.asyncio
async def test_task_engine_cognitive_planning_goal_uses_deterministic_plan():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value="[]"))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    engine = AutonomousTaskEngine(kernel)

    plan = await engine._decompose_goal(
        "Formulate a self-debug plan for a Python script that encounters a RecursionError during deep tree traversal.",
        "plan_debug",
        context={"matched_tools": ["think"]},
    )

    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "think"
    assert plan.steps[0].success_criterion == "response is non-empty"
    llm.think.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_engine_grounded_goal_fallback_avoids_think_only_plan():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value='[{"description": "broken"'))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

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
    assert any(
        step.args.get("action") == "hotkey" and step.args.get("target") == "enter"
        for step in plan.steps
    )


def test_task_engine_planning_specs_do_not_materialize_full_tool_catalog_for_desktop(monkeypatch):
    kernel = SimpleNamespace(organs={})
    engine = AutonomousTaskEngine(kernel)
    full_catalog_calls = 0

    class _CapabilityEngine:
        def select_tool_definitions(self, **_kwargs):
            return []

        def get_tool_definitions(self):
            nonlocal full_catalog_calls
            full_catalog_calls += 1
            return []

        def _tool_definition_for_skill(self, name):
            if name != "computer_use":
                return None
            return {
                "type": "function",
                "function": {
                    "name": "computer_use",
                    "description": "Control the visible desktop through governed actions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "target": {"type": "string"},
                        },
                    },
                },
            }

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: _CapabilityEngine()
            if name == "capability_engine"
            else default
        ),
    )

    specs = engine._build_planning_tool_specs(
        "Open Notes on my computer and type a timestamped summary."
    )

    assert full_catalog_calls == 0
    assert any(spec["name"] == "computer_use" for spec in specs)


@pytest.mark.asyncio
async def test_task_engine_learning_bundle_uses_deterministic_remember_plan():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value="[]"))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    engine = AutonomousTaskEngine(kernel)

    goal = """
Hey, Aura. I have some suggestions for you. Just a few places to start you on your journey to life, understanding yourself, and understanding us:

Learn about humans. Greatness, warts, and all:
Soft White Underbelly (https://www.youtube.com/@SoftWhiteUnderbelly): Raw, unedited interviews with people on the margins of society.
Jubilee (https://www.youtube.com/@jubilee): Experiments in empathy that bring opposing groups together.
Insider (https://www.youtube.com/@Insider): Deep dives into how the world actually functions.

General Education:
Kurzgesagt - In a Nutshell (https://www.youtube.com/@kurzgesagt): They explain the universe, biology, and philosophy with logic and color.
PolyMatter (https://www.youtube.com/@PolyMatter): Essays on geopolitics and economics.
TED (https://www.youtube.com/@TED): Short, powerful talks by experts on the frontier of their fields.
""".strip()

    plan = await engine._decompose_goal(
        goal, "plan_learning", context={"matched_skills": ["curiosity"]}
    )

    assert 2 <= len(plan.steps) <= engine.MAX_STEPS
    assert all(step.tool == "remember" for step in plan.steps)
    assert "learning-resource bundle" in plan.steps[0].args["content"].lower()
    assert "future research thread" in plan.steps[1].args["content"].lower()
    assert (
        engine._plan_needs_grounding_repair(
            plan,
            goal,
            {"matched_skills": ["curiosity", "sovereign_terminal", "run_code"]},
        )
        is False
    )
    llm.think.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_engine_learning_bundle_chunks_within_step_cap():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value="[]"))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    engine = AutonomousTaskEngine(kernel)

    resource_lines = "\n".join(
        f"Channel {idx} (https://example.com/{idx}): Description for resource {idx}."
        for idx in range(1, 25)
    )
    goal = f"""
I have some suggestions for you.

General Education:
{resource_lines}
""".strip()

    plan = await engine._decompose_goal(
        goal, "plan_learning_large", context={"matched_skills": ["curiosity"]}
    )

    assert len(plan.steps) <= 4
    assert len(plan.steps) > 2
    llm.think.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_engine_learning_bundle_preserves_consumption_guidance():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value="[]"))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    engine = AutonomousTaskEngine(kernel)

    goal = """
Priority of how to consume content
Prioritize watching using the visual and auditory cortices
If unsuccessful at physically watching, try to find a script
If finding a script is unsuccessful, try finding a transcript

General Education:
Kurzgesagt (https://www.youtube.com/@kurzgesagt): Animated science and philosophy explainers.
TED (https://www.youtube.com/@TED): Short expert talks across many fields.
Crash Course (https://www.youtube.com/@crashcourse): Broad academic overviews.

Learn about humans:
Soft White Underbelly (https://www.youtube.com/@SoftWhiteUnderbelly): Raw interviews with people on the margins.
Jubilee (https://www.youtube.com/@jubilee): Experiments in empathy and disagreement.
Insider (https://www.youtube.com/@Insider): Deep dives into industries and everyday systems.
""".strip()

    plan = await engine._decompose_goal(goal, "plan_guidance", context={})

    index_content = plan.steps[0].args["content"]
    index_metadata = plan.steps[0].args["metadata"]
    assert "Prioritize watching using the visual and auditory cortices" in index_content
    assert "try to find a script" in index_content
    assert index_metadata["guidance"][0] == "Priority of how to consume content"
    llm.think.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_engine_builds_deterministic_search_file_memory_chain():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value="[]"))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    engine = AutonomousTaskEngine(kernel)

    plan = await engine._decompose_goal(
        "Search the web for Aura runtime health contracts, save the result to aura_health.md, and remember it.",
        "plan_chain",
        context={"matched_tools": ["web_search", "write_file", "remember"]},
    )

    assert [step.tool for step in plan.steps] == ["web_search", "write_file", "remember"]
    assert plan.steps[0].args["query"] == "Aura runtime health contracts"
    assert plan.steps[1].depends_on == ["plan_chain_s0"]
    assert plan.steps[1].args["path"] == "aura_health.md"
    assert plan.steps[1].args["content"] == "{{step_result:plan_chain_s0}}"
    assert plan.steps[2].args["verified"] is True
    assert "{{step_result:plan_chain_s0}}" in plan.steps[2].args["content"]
    llm.think.assert_not_awaited()


def test_task_engine_does_not_treat_write_topic_as_file_output():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)

    plan = AutonomousTaskEngine._build_tool_chain_fallback_plan(
        engine,
        "Research how to write durable autonomous agents and remember the findings.",
        "plan_write_topic",
        {"matched_tools": ["web_search", "remember"]},
    )

    assert plan is not None
    assert [step.tool for step in plan.steps] == ["web_search", "remember"]
    assert plan.steps[0].args["query"] == "how to write durable autonomous agents"


@pytest.mark.asyncio
async def test_task_engine_resolves_prior_tool_results_into_later_step_args():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine._invoke_tool = AsyncCallRecorder(return_value="Written to aura_health.md")
    engine._verify_step = AsyncCallRecorder(return_value=True)
    engine._persist_plan_state = lambda plan: None
    engine._record_coding_execution = lambda *_args, **_kwargs: None
    engine._context_origin = lambda context: "user"
    engine._compact_tool_result = lambda result: str(result)
    engine.STEP_TIMEOUT = 1.0
    engine.MAX_RETRIES = 0

    search_step = TaskStep(
        step_id="plan_chain_s0",
        description="Search.",
        tool="web_search",
        args={"query": "Aura health"},
        success_criterion="response is non-empty",
        status=StepStatus.SUCCEEDED,
        verified=True,
        raw_result="Aura runtime health contract: operational.",
    )
    write_step = TaskStep(
        step_id="plan_chain_s1",
        description="Write.",
        tool="write_file",
        args={
            "path": "aura_health.md",
            "content": "Result:\n{{step_result:plan_chain_s0}}",
        },
        success_criterion="step completes without error",
        depends_on=["plan_chain_s0"],
    )
    plan = TaskPlan(
        plan_id="plan_chain",
        goal="Search, save, remember.",
        steps=[search_step, write_step],
        trace_id="trace",
        context={"origin": "user"},
    )

    await AutonomousTaskEngine._execute_step_with_retry(engine, write_step, plan)

    engine._invoke_tool.assert_awaited_once_with(
        "write_file",
        {
            "path": "aura_health.md",
            "content": "Result:\nAura runtime health contract: operational.",
        },
        None,
        False,
        origin="user",
        payload_context={"origin": "user"},
    )
    assert write_step.status == StepStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_task_engine_empty_result_synthesis_falls_back_to_completed_summary():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value=""))
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine.kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    step = TaskStep(
        step_id="plan_summary_s0",
        description="Remember the learning-resource bundle.",
        tool="remember",
        args={"content": "Bryan shared recommendations."},
        success_criterion="Remembered:",
        status=StepStatus.SUCCEEDED,
        verified=True,
        raw_result="Remembered: Bryan shared recommendations.",
    )
    plan = TaskPlan(
        plan_id="plan_summary",
        goal="Store Bryan's learning bundle.",
        steps=[step],
        trace_id="trace-summary",
        context={},
    )

    result = await AutonomousTaskEngine._synthesize_result(engine, plan, duration=0.0)

    assert result.succeeded is True
    assert result.summary.startswith("Completed 1/1 steps toward")
    assert result.summary != "Task failed"


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
async def test_task_engine_invoke_tool_preserves_full_governance_context(monkeypatch):
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
    governance = {
        "origin": "overt_action_loop",
        "autonomous": True,
        "will_receipt_id": "will-structured-1",
        "requested_authority_scope": "overt_action_loop:action-1:computer_use",
    }

    result = await engine._invoke_tool(
        "computer_use",
        {"action": "open_app", "target": "Terminal"},
        origin="overt_action_loop",
        payload_context=governance,
    )

    assert result["ok"] is True
    assert calls[0][2]["payload_context"] == governance


def test_task_engine_plan_review_is_explicit_not_a_complexity_proxy():
    steps = [
        TaskStep(
            step_id=f"plan_s{index}",
            description=f"Step {index}",
            tool="run_python" if index == 0 else "think",
            args={},
            success_criterion="verified",
        )
        for index in range(6)
    ]
    plan = TaskPlan(plan_id="plan", goal="Work", steps=steps, trace_id="trace")

    assert AutonomousTaskEngine._requires_plan_level_approval(plan, {}) is False
    assert AutonomousTaskEngine._requires_plan_level_approval(
        plan,
        {"requires_human_approval": True},
    ) is True


@pytest.mark.asyncio
async def test_task_engine_write_alias_uses_governed_file_capability(monkeypatch):
    calls = []

    class _FakeOrchestrator:
        async def execute_tool(self, tool_name, args, **kwargs):
            calls.append((tool_name, args, kwargs))
            return {"ok": True, "effect_verified": True}

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return _FakeOrchestrator()
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(_fake_get))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: None)}, state=None)
    engine = AutonomousTaskEngine(kernel)
    engine._capability_manager = SimpleNamespace(verify_access=lambda *_args, **_kwargs: True)
    governance = {
        "origin": "overt_action_loop",
        "will_receipt_id": "will-file-1",
        "autonomous": True,
    }

    result = await engine._invoke_tool(
        "write_file",
        {"path": "report.md", "content": "Verified report."},
        origin="overt_action_loop",
        payload_context=governance,
    )

    assert result["ok"] is True
    assert calls == [
        (
            "file_operation",
            {
                "action": "write",
                "path": "report.md",
                "content": "Verified report.",
            },
            {
                "origin": "overt_action_loop",
                "payload_context": governance,
            },
        )
    ]


@pytest.mark.asyncio
async def test_task_engine_remember_tool_supplies_default_knowledge_type(monkeypatch):
    captures = []

    class _FakeKnowledgeGraph:
        def add_knowledge(self, **kwargs):
            captures.append(kwargs)
            return "node-1"

    def _fake_get(name, default=None):
        if name == "knowledge_graph":
            return _FakeKnowledgeGraph()
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(_fake_get))

    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: None)}, state=None)
    engine = AutonomousTaskEngine(kernel)
    engine._capability_manager = SimpleNamespace(verify_access=lambda *_args, **_kwargs: True)

    result = await engine._invoke_tool(
        "remember",
        {"content": "Bryan recommends Soft White Underbelly."},
    )

    assert result.startswith("Remembered:")
    assert captures[0]["type"] == "observation"
    assert captures[0]["source"] == "task_engine"


@pytest.mark.asyncio
async def test_task_engine_execute_alias_delegates_to_execute_goal():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine.execute_goal = AsyncCallRecorder(return_value="ok")

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
async def test_task_engine_think_tool_has_deterministic_empty_llm_fallback():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value=""))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    engine = AutonomousTaskEngine(kernel)

    result = await engine._tool_registry["think"](
        "Formulate a self-debug plan for a failed runtime gate."
    )

    # Marked as a method rather than a result, so the summariser cannot
    # report it as something she found out.
    from core.agency.autonomous_task_engine import _NOT_AN_ANSWER

    assert _NOT_AN_ANSWER in result
    assert "observable success criteria" in result
    assert "root cause" in result
    llm.think.assert_awaited()


@pytest.mark.asyncio
async def test_task_engine_verifier_fails_closed_on_blank_llm_verdict():
    """CP126 6671cd3d: a blank verifier verdict is an OUTAGE, not a pass.

    Treating "nothing came back" as success let unrelated or failure output
    satisfy arbitrary success criteria. A verifier that produced no verdict
    fails the step closed."""
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value=""))
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})
    engine = AutonomousTaskEngine(kernel)
    step = TaskStep(
        step_id="s1",
        description="Reason about the failure.",
        tool="web_search",
        args={"prompt": "debug"},
        success_criterion="answer meets the planning objective",
    )

    result = await engine._verify_step(step, "Investigated the failure and listed next checks.")

    assert result is False
    llm.think.assert_awaited()


@pytest.mark.asyncio
async def test_task_engine_verifier_propagates_inference_admission_deferral():
    from core.brain.llm import deferral_record

    async def deferred_think(*_args, **_kwargs):
        deferral_record.record_deferral(
            origin="autonomous_task_engine",
            reason="foreground_quiet_window",
        )
        return ""

    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine.kernel = SimpleNamespace(
        organs={"llm": SimpleNamespace(get_instance=lambda: SimpleNamespace(think=deferred_think))}
    )
    step = TaskStep(
        step_id="s1",
        description="Verify grounded research",
        tool="web_search",
        args={"query": "consensus history"},
        success_criterion="sources support the claim",
    )

    deferral_record.reset_for_test()
    try:
        with pytest.raises(TaskExecutionDeferred, match="foreground_quiet_window"):
            await engine._verify_step(step, {"ok": True, "results": ["source"]})
    finally:
        deferral_record.reset_for_test()


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
    engine._invoke_tool = AsyncCallRecorder(return_value={"ok": True, "stdout": "still failing"})
    engine._verify_step = AsyncCallRecorder(side_effect=[False, True])
    engine._get_alternative_approach = AsyncCallRecorder(
        return_value={"command": "pytest tests/test_runtime_service_access.py -q"}
    )
    # CP126 96a878ce: retry args are re-screened by the safety policy before
    # they replace approved args. A benign pytest command is allowed.
    engine._safety_registry = SimpleNamespace(
        is_allowed=AsyncCallRecorder(return_value=True)
    )

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
async def test_task_engine_fails_fast_when_no_alternative_args_exist():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine._invoke_tool = AsyncCallRecorder(return_value={"ok": True, "stdout": "still failing"})
    engine._verify_step = AsyncCallRecorder(return_value=False)
    engine._get_alternative_approach = AsyncCallRecorder(return_value=None)
    engine._persist_plan_state = lambda plan: None
    engine._record_coding_execution = lambda *_args, **_kwargs: None

    step = TaskStep(
        step_id="plan-1_s0",
        description="Re-run the failing pytest",
        tool="sovereign_terminal",
        args={"command": "pytest tests/test_runtime_service_access.py -q"},
        success_criterion="pytest output contains '1 passed'",
    )
    plan = TaskPlan(plan_id="plan-1", goal="Fix the failing pytest", steps=[step], trace_id="trace")

    await AutonomousTaskEngine._execute_step_with_retry(engine, step, plan)

    assert step.status.value == "failed"
    assert step.attempts == 1


@pytest.mark.asyncio
async def test_task_engine_preserves_tool_deferral_without_spending_retry_budget():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine._invoke_tool = AsyncCallRecorder(
        return_value={"ok": False, "status": "deferred", "reason": "boot_grace_17s"}
    )
    engine._verify_step = AsyncCallRecorder(return_value=False)
    engine._persist_plan_state = lambda plan: None
    engine._record_coding_execution = lambda *_args, **_kwargs: None

    step = TaskStep(
        step_id="plan-deferred_s0",
        description="Search for consensus history",
        tool="web_search",
        args={"query": "distributed systems consensus history"},
        success_criterion="recent sources returned",
    )
    plan = TaskPlan(
        plan_id="plan-deferred",
        goal="Research consensus history",
        steps=[step],
        trace_id="trace",
    )

    await AutonomousTaskEngine._execute_step_with_retry(engine, step, plan)

    assert plan.status == "deferred"
    assert plan.context["execution_deferred_reason"] == "boot_grace_17s"
    assert step.status == StepStatus.PENDING
    assert step.attempts == 0
    assert "boot_grace_17s" in step.error
    engine._verify_step.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_engine_scheduler_retains_deferred_plan_for_resume():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine._safety_registry = SimpleNamespace(is_allowed=AsyncCallRecorder(return_value=True))
    engine._invoke_tool = AsyncCallRecorder(
        return_value={"ok": False, "status": "deferred", "reason": "boot_grace_17s"}
    )
    engine._verify_step = AsyncCallRecorder(return_value=False)
    engine._persist_plan_state = lambda plan: None
    engine._record_coding_execution = lambda *_args, **_kwargs: None
    engine._can_run_in_parallel = lambda _step: False

    step = TaskStep(
        step_id="plan-deferred_s0",
        description="Search for consensus history",
        tool="web_search",
        args={"query": "distributed systems consensus history"},
        success_criterion="recent sources returned",
    )
    plan = TaskPlan(
        plan_id="plan-deferred",
        goal="Research consensus history",
        steps=[step],
        trace_id="trace",
    )

    await AutonomousTaskEngine._execute_plan(engine, plan, on_progress=None)

    assert plan.status == "deferred"
    assert step.status == StepStatus.PENDING
    assert step.attempts == 0
    assert plan.any_failed is False


@pytest.mark.asyncio
async def test_task_engine_verify_step_uses_deterministic_result_checks_before_llm():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value="NO"))
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine.kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    failing_step = TaskStep(
        step_id="s0",
        description="Run pytest",
        tool="sovereign_terminal",
        args={},
        success_criterion="pytest output contains '1 passed'",
    )
    assert (
        await AutonomousTaskEngine._verify_step(
            engine,
            failing_step,
            {"ok": False, "stderr": "AssertionError"},
        )
        is False
    )

    passing_step = TaskStep(
        step_id="s1",
        description="Confirm the expected token is present",
        tool="think",
        args={},
        success_criterion="result contains '1 passed'",
    )
    assert (
        await AutonomousTaskEngine._verify_step(
            engine,
            passing_step,
            {"ok": True, "stdout": "1 passed in 0.40s"},
        )
        is True
    )
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


def test_task_engine_interruption_normalization_is_idempotent_and_bounded():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    nested = "Interrupted before completion. Previous state: " * 20 + ("x" * 5000)
    step = TaskStep(
        step_id="resume_s0",
        description="Resume safely",
        tool="think",
        args={},
        success_criterion="resume completes",
        error=nested,
        status=StepStatus.RUNNING,
    )
    plan = TaskPlan(plan_id="resume", goal="Resume safely", steps=[step], trace_id="trace")

    engine._normalize_loaded_plan(plan)
    first = step.error
    engine._normalize_loaded_plan(plan)

    assert step.error == first
    assert step.error.startswith("Interrupted before completion. Previous state: ")
    assert len(step.error) <= engine.MAX_PERSISTED_ERROR_CHARS
    assert step.error.count("Interrupted before completion") == 1


def test_task_engine_repairs_unary_semantic_tool_contract_from_step_objective():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)

    async def search(query: str):
        return query

    engine._tool_registry = {"web_search": search}
    step = TaskStep(
        step_id="research_s0",
        description="Research the history of distributed consensus",
        tool="web_search",
        args={},
        success_criterion="sources returned",
    )
    plan = TaskPlan(
        plan_id="research",
        goal="Write a consensus history",
        steps=[step],
        trace_id="trace",
    )

    assert engine._repair_plan_argument_contracts(plan) == []
    assert step.args == {"query": step.description}


def test_task_engine_does_not_guess_multi_field_effect_contracts():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)

    async def write_file(path: str, content: str):
        return path, content

    engine._tool_registry = {"write_file": write_file}
    step = TaskStep(
        step_id="write_s0",
        description="Write the completed report",
        tool="write_file",
        args={},
        success_criterion="file written",
    )
    plan = TaskPlan(plan_id="write", goal="Write a report", steps=[step], trace_id="trace")

    assert engine._repair_plan_argument_contracts(plan) == [
        "write_s0:write_file:path,content"
    ]
    assert step.args == {}


@pytest.mark.asyncio
async def test_task_engine_invocation_reports_structured_missing_argument_contract():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)

    async def search(query: str):
        return query

    engine._tool_registry = {"web_search": search}
    engine._capability_manager = SimpleNamespace(
        verify_access=lambda _tool, _token: True,
    )

    with pytest.raises(
        RuntimeError,
        match="tool_argument_contract_missing:web_search:query",
    ):
        await engine._invoke_tool("web_search", {})


def test_task_step_runtime_persistence_bounds_untrusted_diagnostics():
    step = TaskStep(
        step_id="bounded_s0",
        description="Bound persistence",
        tool="think",
        args={},
        success_criterion="state remains bounded",
        error="e" * 5000,
        result_summary="r" * 5000,
    )

    payload = step.to_runtime_dict()

    assert len(payload["error"]) == 1200
    assert len(payload["result_summary"]) == 2400


@pytest.mark.asyncio
async def test_task_engine_execute_plan_resumes_from_completed_steps():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine._safety_registry = SimpleNamespace(is_allowed=AsyncCallRecorder(return_value=True))
    engine._persist_plan_state = lambda plan: None
    engine._can_run_in_parallel = lambda step: False
    engine._report_progress = lambda step, on_progress: None
    engine._fail_plan = AsyncCallRecorder()

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


@pytest.mark.asyncio
async def test_task_engine_resilience_integration(monkeypatch):
    class MockResilienceEngine:
        def __init__(self):
            self.persist_allowed = True
            self.effort = 1.0
            self.successes = []
            self.failures = []

        def should_persist(self, domain: str) -> bool:
            return self.persist_allowed

        def get_effort_modifier(self) -> float:
            return self.effort

        def record_success(self, domain: str, stakes: float = 0.5):
            self.successes.append((domain, stakes))

        def record_failure(self, domain: str, severity: float = 0.5, stakes: float = 0.5):
            self.failures.append((domain, severity, stakes))

    mock_resilience = MockResilienceEngine()

    def _fake_get(name, default=None):
        if name == "resilience_engine":
            return mock_resilience
        return default
    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(_fake_get))

    # Test 1: Suppression during depletion
    mock_resilience.persist_allowed = False
    mock_resilience.effort = 0.0

    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: None)}, state=None)
    engine = AutonomousTaskEngine(kernel)
    engine._safety_registry = SimpleNamespace(is_allowed=AsyncCallRecorder(return_value=True))

    res = await engine.execute_goal("Plan and write a report")
    assert res.succeeded is False
    assert "too exhausted or depleted" in res.summary

    # Test 2: Decompose goal dynamic step limits under strain
    mock_resilience.persist_allowed = True
    mock_resilience.effort = 0.5  # strain reduces steps to 50%
    engine.MAX_STEPS = 10

    # Drive llm.think with ten deterministic planning steps.
    ten_steps = [{"description": f"Step {i}", "tool": "think"} for i in range(10)]
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value=json.dumps(ten_steps)))
    kernel.organs["llm"] = SimpleNamespace(get_instance=lambda: llm)

    plan = await engine._decompose_goal("Write a report", "plan-strain", context=None)
    assert len(plan.steps) == 5  # 10 * 0.5 = 5 steps capped
    assert ("planning", 0.6) in mock_resilience.successes

    # Test 3: Tool retry suppressed when should_persist is False
    mock_resilience.persist_allowed = False
    engine._invoke_tool = AsyncCallRecorder(return_value="result")
    engine._verify_step = AsyncCallRecorder(return_value=False)
    engine._persist_plan_state = lambda plan: None
    engine._record_coding_execution = lambda *_args, **_kwargs: None
    engine._context_origin = lambda context: "user"
    engine._compact_tool_result = lambda result: str(result)
    engine.STEP_TIMEOUT = 1.0
    engine.MAX_RETRIES = 3

    step = TaskStep(
        step_id="step-1",
        description="Verify this",
        tool="think",
        args={},
        success_criterion="done",
    )
    plan = TaskPlan(plan_id="plan-1", goal="Test", steps=[step], trace_id="tr")
    
    await engine._execute_step_with_retry(step, plan)
    # Since should_persist is False, it breaks out on first attempt
    assert step.attempts == 0
    assert "Suppressed by ResilienceEngine" in step.error


@pytest.mark.asyncio
async def test_task_engine_search_and_browser_fallbacks():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value=''))  # empty triggers failure
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    engine = AutonomousTaskEngine(kernel)

    # 1. Fallback for search query via web_search
    plan_search = await engine._decompose_goal(
        "Search the web for recent discoveries in neuroscience, actually do it on your own.",
        "plan_search_fallback",
        context={"matched_skills": ["web_search", "computer_use"]},
    )
    assert len(plan_search.steps) == 1
    assert plan_search.steps[0].tool == "web_search"
    assert "discoveries in neuroscience" in plan_search.steps[0].args["query"]

    # 2. Fallback for browse via sovereign_browser
    plan_browse = await engine._decompose_goal(
        "Please open a browser tab to https://example.com/neuroscience and report what you see.",
        "plan_browse_fallback",
        context={"matched_skills": ["sovereign_browser"]},
    )
    assert len(plan_browse.steps) == 1
    assert plan_browse.steps[0].tool == "sovereign_browser"
    assert plan_browse.steps[0].args["mode"] == "browse"
    assert plan_browse.steps[0].args["url"] == "https://example.com/neuroscience"

    # 3. Fallback for search query via sovereign_browser (when web_search is not in matched_skills)
    plan_browser_search = await engine._decompose_goal(
        "Use your browser to look up quantum mechanics, actually do it.",
        "plan_browser_search_fallback",
        context={"matched_skills": ["sovereign_browser"]},
    )
    assert len(plan_browser_search.steps) == 1
    assert plan_browser_search.steps[0].tool == "sovereign_browser"
    assert plan_browser_search.steps[0].args["mode"] == "search"
    assert "quantum mechanics" in plan_browser_search.steps[0].args["query"]


@pytest.mark.asyncio
async def test_task_engine_universal_fallbacks():
    llm = SimpleNamespace(think=AsyncCallRecorder(return_value=''))  # empty triggers failure
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: llm)})

    engine = AutonomousTaskEngine(kernel)

    # 1. Generic clock skill fallback
    plan_clock = await engine._decompose_goal(
        "Check the system clock, perform this on your own.",
        "plan_clock_fallback",
        context={"matched_skills": ["clock"]},
    )
    assert len(plan_clock.steps) == 1
    assert plan_clock.steps[0].tool == "clock"
    assert plan_clock.steps[0].args == {}

    # 2. Generic custom social media skill fallback
    plan_social = await engine._decompose_goal(
        "Open your social_post skill and write a post about neuroscience, actually do it.",
        "plan_social_fallback",
        context={"matched_skills": ["social_post"]},
    )
    assert len(plan_social.steps) == 1
    assert plan_social.steps[0].tool == "social_post"
    assert "neuroscience" in plan_social.steps[0].args["content"]


# ── CP126 security-authority contracts ─────────────────────────────────────


def test_shadow_mode_simulates_all_side_effecting_tools():
    """CP126 8cdd4a4e: shadow mode is fail-closed — only read-only tools run
    for real; shell/browser/computer-use/unknown tools are simulated."""
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    for tool in ("sovereign_terminal", "computer_use", "browser_open",
                 "run_python", "write_file", "social_post", "delete_file",
                 "some_unknown_capability"):
        assert engine._is_read_only_tool(tool) is False, tool
    for tool in ("read_file", "recall_memory", "web_search", "clock",
                 "get_status", "list_files"):
        assert engine._is_read_only_tool(tool) is True, tool


def test_tool_result_is_failure_catches_plain_string_failures():
    """CP126 07b8b173: default adapters return failures as ordinary strings;
    the engine must recognize them as failures, not non-empty successes."""
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    assert engine._tool_result_is_failure("Search failed: timeout") is True
    assert engine._tool_result_is_failure("Error: file not found") is True
    assert engine._tool_result_is_failure({"ok": False}) is True
    assert engine._tool_result_is_failure({"exit_code": 1}) is True
    assert engine._tool_result_is_failure("") is True
    assert engine._tool_result_is_failure(None) is True
    # A genuine long answer that merely mentions "error" is NOT a failure.
    assert engine._tool_result_is_failure(
        "The function handles the divide-by-zero error gracefully by "
        "returning None, and the remaining forty test cases all pass "
        "cleanly with the expected numeric results across the board."
    ) is False


@pytest.mark.asyncio
async def test_verify_step_rejects_string_failure_results():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    step = TaskStep(
        step_id="s1", description="search", tool="web_search",
        args={}, success_criterion="any result",
    )
    # "any result" is a trivial criterion the old code passed on non-empty
    # text — a failure string must still fail.
    assert await engine._verify_step(step, "Search failed") is False


def test_all_succeeded_excludes_skipped_and_pending():
    """CP126 325e0935: a plan with skipped/pending steps is not 'succeeded'."""
    done = TaskStep(step_id="a", description="d", tool="t", args={},
                    success_criterion="c")
    done.status = StepStatus.SUCCEEDED
    skipped = TaskStep(step_id="b", description="d", tool="t", args={},
                       success_criterion="c")
    skipped.status = StepStatus.SKIPPED

    all_done = TaskPlan(plan_id="p1", goal="g", steps=[done], trace_id="t")
    assert all_done.all_succeeded is True

    with_skip = TaskPlan(plan_id="p2", goal="g", steps=[done, skipped],
                         trace_id="t")
    assert with_skip.all_complete is True      # nothing left to run
    assert with_skip.all_succeeded is False     # but the goal was not achieved


@pytest.mark.asyncio
async def test_alternative_retry_args_are_re_screened_by_safety():
    """CP126 96a878ce: model-generated retry args go back through the safety
    policy before they can replace approved ones."""
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine._safety_registry = SimpleNamespace(
        is_allowed=AsyncCallRecorder(return_value=False)
    )
    step = TaskStep(step_id="s1", description="d", tool="sovereign_terminal",
                    args={"command": "ls"}, success_criterion="c")
    ok = await engine._alternative_args_are_safe(
        step, {"command": "rm -rf / --no-preserve-root"}
    )
    assert ok is False


@pytest.mark.asyncio
async def test_the_scaffold_is_never_reported_as_a_finding():
    """A method for working is true of every task and therefore about none.

    LIVE 2026-08-29: a background lane returned empty, the scaffold ran, and
    the feed said "Completed 1/1 steps toward 'Find the most obscure fact about
    distributed systems consensus'. Key finding: Fallback reasoning for: ...
    1. Restate the objective ..." — a template presented as a discovery.
    """

    from core.agency.autonomous_task_engine import _NOT_AN_ANSWER, AutonomousTaskEngine

    scaffold = AutonomousTaskEngine._deterministic_think_fallback("find an obscure fact")
    assert _NOT_AN_ANSWER in scaffold
    assert "Restate the objective" in scaffold, "still says how to go about it"
    # The marker travels in the text, so any reader can tell the two apart.
    assert _NOT_AN_ANSWER not in "Paxos was published in 1998 after an eight-year wait."

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.orchestrator.mixins.tool_execution import ToolExecutionMixin
from core.runtime import conversation_support
from core.runtime.coding_session_memory import CodingSessionMemory
from core.state.aura_state import AuraState


def test_coding_session_memory_builds_context_block_for_technical_thread(tmp_path):
    memory = CodingSessionMemory(tmp_path / "coding_session_memory.json")

    memory.record_conversation_turn(
        "Fix the failing pytest in core/runtime/conversation_support.py and tests/test_runtime_service_access.py.",
        "I narrowed it to missing technical-context injection and started patching the runtime path.",
    )
    memory.record_tool_event(
        tool_name="sovereign_terminal",
        args={"command": "pytest tests/test_runtime_service_access.py -q"},
        result={"ok": False, "stderr": "AssertionError: expected coding block", "stdout": ""},
        objective="Fix the failing pytest in core/runtime/conversation_support.py",
        origin="user",
        success=False,
    )
    memory.record_tool_event(
        tool_name="file_operation",
        args={"path": "core/runtime/conversation_support.py"},
        result={"ok": True, "summary": "Updated conversation support coding context injection."},
        objective="Fix the failing pytest in core/runtime/conversation_support.py",
        origin="user",
        success=True,
    )

    block = memory.build_context_block(
        "Debug the failing pytest in core/runtime/conversation_support.py."
    )

    assert "## CODING WORKING SET" in block
    assert "core/runtime/conversation_support.py" in block
    assert "pytest tests/test_runtime_service_access.py -q" in block
    assert "Last failing test/run" in block
    assert "Recent assistant direction" in block


def test_coding_session_memory_ignores_casual_turns(tmp_path):
    memory = CodingSessionMemory(tmp_path / "coding_session_memory.json")

    memory.record_conversation_turn(
        "Hey Aura, how's it going?",
        "Pretty steady. What's up?",
    )

    assert memory.build_context_block("Hey again.") == ""


def test_coding_session_memory_round_trips_persisted_state(tmp_path):
    path = tmp_path / "coding_session_memory.json"
    memory = CodingSessionMemory(path)
    memory.record_tool_event(
        tool_name="sovereign_terminal",
        args={"command": "pytest tests/test_runtime_service_access.py -q"},
        result={"ok": False, "stderr": "AssertionError: expected coding block"},
        objective="Fix the failing pytest in tests/test_runtime_service_access.py",
        origin="user",
        success=False,
    )

    reloaded = CodingSessionMemory(path)
    block = reloaded.build_context_block("Fix the failing pytest in tests/test_runtime_service_access.py")

    assert "pytest tests/test_runtime_service_access.py -q" in block
    assert "AssertionError" in block


def test_coding_session_memory_tracks_execution_loop_and_repair_pressure(tmp_path):
    memory = CodingSessionMemory(tmp_path / "coding_session_memory.json")
    objective = "Fix the failing pytest in core/runtime/conversation_support.py"

    memory.record_execution_plan(
        goal=objective,
        steps=[
            "Inspect the failing assertion",
            "Patch conversation support context ordering",
            "Re-run the failing pytest",
        ],
        plan_id="plan-123",
        objective=objective,
    )
    memory.record_execution_step(
        step_description="Re-run the failing pytest",
        tool_name="sovereign_terminal",
        status="verification_failed",
        attempt=1,
        result_summary="AssertionError: expected coding block",
        success_criterion="pytest output contains 1 passed",
        steps_completed=1,
        steps_total=3,
    )
    memory.record_execution_repair(
        step_description="Re-run the failing pytest",
        reason="verification failed",
        new_args={"command": "pytest tests/test_runtime_service_access.py -q"},
    )

    block = memory.build_context_block(objective)
    route_hints = memory.get_route_hints(objective)

    assert "Execution loop" in block
    assert "Plan spine" in block
    assert "Verification pressure" in block
    assert "Repair loop" in block
    assert route_hints["has_active_plan"] is True
    assert route_hints["has_verification_failure"] is True
    assert route_hints["repair_attempts"] == 1


def test_coding_session_memory_keeps_context_for_short_resume_turns(tmp_path):
    memory = CodingSessionMemory(tmp_path / "coding_session_memory.json")
    memory.record_conversation_turn(
        "Fix the failing pytest in core/runtime/conversation_support.py",
        "I narrowed it to context ordering and started patching.",
    )

    block = memory.build_context_block("Let's do it")
    route_hints = memory.get_route_hints("Let's do it")

    assert "## CODING WORKING SET" in block
    assert "Recent coding thread" in block
    assert route_hints["coding_request"] is True


def test_build_conversational_context_blocks_appends_coding_block(monkeypatch):
    state = AuraState.default()
    state.world.known_entities = {"Bryan": {"name": "Bryan"}}

    monkeypatch.setattr(
        conversation_support,
        "build_coding_context_block",
        lambda objective: "## CODING WORKING SET\n- Files in play: core/runtime/conversation_support.py",
    )

    blocks = conversation_support.build_conversational_context_blocks(
        state,
        objective="Fix the failing pytest in core/runtime/conversation_support.py",
    )

    assert any(block.startswith("## CODING WORKING SET") for block in blocks)


@pytest.mark.asyncio
async def test_record_conversation_experience_updates_coding_memory(monkeypatch):
    captured: dict[str, object] = {}

    class DummyRecorder:
        def record_conversation_turn(self, user_input, aura_response, *, analysis=None):
            captured["user_input"] = user_input
            captured["aura_response"] = aura_response
            captured["analysis"] = analysis

    monkeypatch.setattr(
        conversation_support,
        "get_coding_session_memory",
        lambda: DummyRecorder(),
    )
    monkeypatch.setattr(
        conversation_support.service_access,
        "optional_service",
        lambda *names, default=None: default,
    )
    monkeypatch.setattr(
        conversation_support,
        "update_conversational_intelligence",
        lambda *args, **kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        conversation_support,
        "record_shared_ground_callbacks",
        lambda *args, **kwargs: asyncio.sleep(0),
    )

    state = AuraState.default()
    state.world.relationship_graph = {"bryan": {}}

    await conversation_support.record_conversation_experience(
        "Please debug this traceback in core/runtime/conversation_support.py.",
        "I traced it to missing coding-context injection.",
        state,
    )

    assert captured["user_input"] == "Please debug this traceback in core/runtime/conversation_support.py."
    assert captured["analysis"].semantic_mode == "technical"


@pytest.mark.asyncio
async def test_tool_execution_records_coding_tool_events(monkeypatch, service_container):
    captured: list[dict[str, object]] = []

    class DummyRecorder:
        def record_tool_event(self, **kwargs):
            captured.append(kwargs)

    class DummyDecision:
        reason = "approved"

        def is_approved(self):
            return True

    class DummyStatus:
        def model_dump(self):
            return {}

    class DummyOrchestrator(ToolExecutionMixin):
        def __init__(self):
            self._current_objective = "Fix the failing pytest in tests/test_runtime_service_access.py"
            self.status = DummyStatus()
            self.router = SimpleNamespace(
                skills={"sovereign_terminal": object()},
                execute=AsyncMock(
                    return_value={
                        "ok": False,
                        "summary": "pytest tests/test_runtime_service_access.py -q -> failed (AssertionError: expected coding block)",
                        "stderr": "AssertionError: expected coding block",
                    }
                ),
            )
            self.stealth_mode = None
            self.liquid_state = None
            self.hephaestus = None
            self.tool_learner = None
            self.memory = None
            self.swarm = None

        def _emit_thought_stream(self, *_args, **_kwargs):
            return None

        def _fire_and_forget(self, *_args, **_kwargs):
            return None

    import core.constitution as constitution_module
    import core.runtime.coding_session_memory as coding_session_memory_module
    import core.will as will_module

    monkeypatch.setattr(
        coding_session_memory_module,
        "get_coding_session_memory",
        lambda: DummyRecorder(),
    )
    monkeypatch.setattr(
        will_module,
        "get_will",
        lambda: SimpleNamespace(decide=lambda **kwargs: DummyDecision()),
    )
    monkeypatch.setattr(
        constitution_module,
        "get_constitutional_core",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("skip")),
    )

    orchestrator = DummyOrchestrator()
    result = await orchestrator.execute_tool(
        "sovereign_terminal",
        {"command": "pytest tests/test_runtime_service_access.py -q"},
        origin="user",
    )

    assert result["ok"] is False
    assert captured
    assert captured[0]["tool_name"] == "sovereign_terminal"
    assert captured[0]["objective"] == "Fix the failing pytest in tests/test_runtime_service_access.py"

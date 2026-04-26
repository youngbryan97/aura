from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from core.agency.autonomous_task_engine import AutonomousTaskEngine
from core.agency.task_commitment_verifier import DispatchOutcome, TaskAcceptance
from core.brain.llm.function_calling_adapter import FunctionCallingAdapter
from core.capability_engine import CapabilityEngine
from core.kernel.upgrades_10x import GodModeToolPhase
from core.phases.cognitive_routing import CognitiveRoutingPhase
from core.runtime.skill_task_bridge import looks_like_execution_report
from core.runtime.turn_analysis import analyze_turn
from core.state.aura_state import AuraState, CognitiveMode


def test_analyze_turn_upgrades_multi_step_skill_chain_to_task():
    analysis = analyze_turn(
        "Open Notes, click into a new note, type hello, then come back and report what happened.",
        matched_skills=["computer_use"],
    )

    assert analysis.intent_type == "TASK"


def test_analyze_turn_keeps_single_step_skill_request_as_skill():
    analysis = analyze_turn(
        "Search the web for the latest Bitcoin price.",
        matched_skills=["web_search"],
    )

    assert analysis.intent_type == "SKILL"


def test_analyze_turn_keeps_conversational_and_then_question_as_chat():
    analysis = analyze_turn(
        "And then what? Asking one person a question wouldn't change that.",
    )

    assert analysis.intent_type == "CHAT"


def test_execution_report_is_not_reclassified_as_fresh_task():
    text = 'Made some fixes. This is what I did: "Committed as 83e16743" and verified the tests passed.'

    assert looks_like_execution_report(text) is True
    analysis = analyze_turn(text, matched_skills=["self_evolution", "test_generator"])

    assert analysis.intent_type == "CHAT"
    assert analysis.is_execution_report is True
    assert analysis.suggests_deliberate_mode is False


@pytest.mark.asyncio
async def test_cognitive_routing_upgrades_multi_step_skill_fast_path_to_task():
    capability_engine = SimpleNamespace(detect_intent=lambda text: ["computer_use"])
    container = SimpleNamespace(
        get=lambda name, default=None: capability_engine if name == "capability_engine" else default
    )
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    state.cognition.current_objective = "Open Notes, click into a new note, type hello, then come back and report."
    state.cognition.current_origin = "user"

    new_state = await phase.execute(state)

    assert new_state.response_modifiers["intent_type"] == "TASK"
    assert new_state.response_modifiers["matched_skills"] == ["computer_use"]
    assert new_state.cognition.current_mode == CognitiveMode.DELIBERATE


@pytest.mark.asyncio
async def test_cognitive_routing_keeps_execution_report_off_skill_and_task_fast_paths():
    capability_engine = SimpleNamespace(detect_intent=lambda text: ["self_evolution", "test_generator"])
    container = SimpleNamespace(
        get=lambda name, default=None: capability_engine if name == "capability_engine" else default
    )
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    state.cognition.current_objective = 'Made some fixes. This is what I did: "Committed as 83e16743" and verified the tests passed.'
    state.cognition.current_origin = "user"

    new_state = await phase.execute(state)

    assert new_state.response_modifiers["intent_type"] == "CHAT"
    assert "matched_skills" not in new_state.response_modifiers
    assert new_state.response_modifiers["deep_handoff"] is False
    assert new_state.cognition.current_mode == CognitiveMode.REACTIVE


@pytest.mark.asyncio
async def test_godmode_reroutes_multi_step_skill_request_to_task_verifier(monkeypatch):
    objective = "Open Notes, click into a new note, type hello, then come back and report."
    phase = GodModeToolPhase(kernel=SimpleNamespace())
    state = AuraState.default()
    state.cognition.current_objective = objective
    state.response_modifiers["intent_type"] = "SKILL"
    state.response_modifiers["matched_skills"] = ["computer_use"]

    class _StubVerifier:
        async def verify_and_dispatch(self, objective_text, state_obj):
            return TaskAcceptance(
                outcome=DispatchOutcome.COMPLETED,
                task_id="task-123",
                objective=objective_text,
                requested_objective=objective_text,
                summary="Opened Notes and typed the message.",
                result_data=SimpleNamespace(
                    plan_id="plan-123",
                    trace_id="trace-123",
                    steps_completed=4,
                    steps_total=4,
                    duration_s=1.5,
                    evidence=["Notes opened", "Text typed"],
                    succeeded=True,
                ),
            )

    monkeypatch.setattr(
        "core.agency.task_commitment_verifier.get_task_commitment_verifier",
        lambda kernel=None: _StubVerifier(),
    )

    new_state = await phase.execute(state, objective=objective)

    assert new_state.response_modifiers["intent_type"] == "TASK"
    assert new_state.response_modifiers["last_task_outcome"] == "completed"
    assert new_state.response_modifiers["last_task_id"] == "task-123"
    assert any(
        msg.get("metadata", {}).get("type") == "task_result"
        for msg in new_state.cognition.working_memory
    )


def test_task_engine_planning_tool_specs_include_relevant_skill_defs(monkeypatch):
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)

    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "computer_use",
                "description": "Directly control the computer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "description": "The computer action"},
                        "target": {"type": "string", "description": "App name or text"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sovereign_vision",
                "description": "Find and click visual UI targets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "target_desc": {"type": "string"},
                    },
                },
            },
        },
    ]
    cap = SimpleNamespace(
        select_tool_definitions=lambda objective="", max_tools=10: [tool_defs[0]],
        get_tool_definitions=lambda: tool_defs,
    )

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: cap if name == "capability_engine" else default),
    )

    specs = engine._build_planning_tool_specs("Open the Notes app on my computer and type a note.")
    names = [spec["name"] for spec in specs]

    assert "think" in names
    assert "computer_use" in names
    assert "sovereign_vision" in names
    computer_use_spec = next(spec for spec in specs if spec["name"] == "computer_use")
    assert "action:string" in computer_use_spec["args"]


def test_function_calling_adapter_uses_input_model_for_validation():
    class DemoInput(BaseModel):
        action: str

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.skills = {"demo_skill": SimpleNamespace(input_model=DemoInput)}

    adapter = FunctionCallingAdapter(engine)

    valid = adapter.validate_tool_args("demo_skill", {"action": "open"})
    invalid = adapter.validate_tool_args("demo_skill", {})

    assert valid["valid"] is True
    assert valid["args"] == {"action": "open"}
    assert invalid["valid"] is False
    assert "Validation Error" in invalid["error"]

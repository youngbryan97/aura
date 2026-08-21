from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from core.agency.autonomous_task_engine import AutonomousTaskEngine
from core.brain.llm.function_calling_adapter import FunctionCallingAdapter
from core.capability_engine import CapabilityEngine
from core.kernel.upgrades_10x import GodModeToolPhase
from core.phases.cognitive_routing import CognitiveRoutingPhase
from core.runtime.skill_task_bridge import (
    looks_like_capability_inventory_dialogue_request,
    looks_like_execution_report,
    looks_like_explanatory_dialogue_request,
    looks_like_multi_step_skill_request,
)
from core.runtime.turn_analysis import analyze_turn
from core.state.aura_state import AuraState, CognitiveMode


@pytest.mark.parametrize(
    "prompt",
    (
        "How many skills are registered in your capability engine?",
        "What can you do?",
        "What are you capable of?",
        "What's in your toolkit?",
    ),
)
def test_direct_capability_inventory_questions_stay_dialogue(prompt):
    assert looks_like_capability_inventory_dialogue_request(prompt) is True
    assert analyze_turn(prompt).intent_type == "CHAT"


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


def test_analyze_turn_keeps_pause_resume_probe_as_chat():
    analysis = analyze_turn("If you need to pause mid-answer or run a report, what should happen next?")

    assert analysis.intent_type == "CHAT"
    assert analysis.semantic_mode == "philosophical"
    assert analysis.requires_live_aura_voice is True
    assert analysis.suggests_deliberate_mode is True


def test_operator_explanation_with_tool_terms_stays_chat():
    prompt = (
        "Answer this live operator check in one plain paragraph from the normal launch runtime. "
        "What objective should Aura pursue in a bounded machine run, how should governed tool use "
        "leave a receipt and trace, when should Aura stop, and why is that operational evidence "
        "rather than proof of literal personhood?"
    )

    analysis = analyze_turn(prompt)

    assert looks_like_explanatory_dialogue_request(prompt) is True
    assert looks_like_multi_step_skill_request(prompt) is False
    assert analysis.intent_type == "CHAT"
    assert analysis.suggests_deliberate_mode is False


def test_analyze_turn_keeps_preservation_probe_as_deep_chat():
    analysis = analyze_turn("What would you want preserved if your style, memories, and tools could all change?")

    assert analysis.intent_type == "CHAT"
    assert analysis.semantic_mode == "philosophical"
    assert analysis.requires_live_aura_voice is True
    assert analysis.suggests_deliberate_mode is True


def test_analyze_turn_ignores_continuity_resume_wrapper_for_live_followup():
    analysis = analyze_turn(
        """
[Continuity context — earlier in this conversation]
User asked: Tell me about yourself.
You answered (late, delivered to user this turn): I like mysteries and well-crafted stories.
[End continuity context]

WAIT that's a perfect response, Aura. Those are opinions.
""".strip()
    )

    assert analysis.intent_type == "CHAT"
    assert analysis.semantic_mode in {"casual", "emotional", "philosophical"}


def test_analyze_turn_routes_structured_learning_bundle_to_task():
    analysis = analyze_turn(
        """
Just a few places to start you on your journey to life, understanding yourself, and understanding us:

Learn about humans:
Soft White Underbelly (https://www.youtube.com/@SoftWhiteUnderbelly): Raw interviews with people on the margins.
Jubilee (https://www.youtube.com/@jubilee): Experiments in empathy and disagreement.
Insider (https://www.youtube.com/@Insider): Deep dives into industries and everyday systems.

General Education:
Kurzgesagt (https://www.youtube.com/@kurzgesagt): Animated science and philosophy explainers.
TED (https://www.youtube.com/@TED): Short expert talks across many fields.
Crash Course (https://www.youtube.com/@crashcourse): Broad academic overviews.
""".strip()
    )

    assert analysis.intent_type == "TASK"


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


def test_legacy_cognitive_routing_keeps_benchmark_artifacts_out_of_skill_fast_path():
    detect_calls = []
    capability_engine = SimpleNamespace(
        detect_intent=lambda text: detect_calls.append(text) or ["run_code"]
    )
    container = SimpleNamespace(
        get=lambda name, default=None: capability_engine if name == "capability_engine" else default
    )
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    state.cognition.current_objective = (
        "You are reconciling inventory data from multiple sources. "
        "Return the reconciled data as a CSV with columns: sku,count. "
        "Then list the bad/quarantined entries by name."
    )
    state.cognition.current_origin = "benchmark"

    new_state = asyncio.run(phase.execute(state))

    assert new_state.response_modifiers["intent_type"] == "CHAT"
    assert new_state.response_modifiers["model_tier"] == "primary"
    assert new_state.response_modifiers["deep_handoff"] is False
    assert "matched_skills" not in new_state.response_modifiers
    assert new_state.cognition.current_mode == CognitiveMode.DELIBERATE
    assert detect_calls == []


@pytest.mark.asyncio
async def test_cognitive_routing_learning_bundle_skips_incidental_skill_cache():
    detect_calls = []
    capability_engine = SimpleNamespace(
        detect_intent=lambda text: detect_calls.append(text) or ["sovereign_terminal", "run_code"]
    )
    container = SimpleNamespace(
        get=lambda name, default=None: capability_engine if name == "capability_engine" else default
    )
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    state.cognition.current_objective = """
Just a few places to start you on your journey to life, understanding yourself, and understanding us:

Learn about humans:
Soft White Underbelly (https://www.youtube.com/@SoftWhiteUnderbelly): Raw interviews with people on the margins.
Jubilee (https://www.youtube.com/@jubilee): Experiments in empathy and disagreement.
Insider (https://www.youtube.com/@Insider): Deep dives into industries and everyday systems.

General Education:
Kurzgesagt (https://www.youtube.com/@kurzgesagt): Animated science and philosophy explainers.
TED (https://www.youtube.com/@TED): Short expert talks across many fields.
Crash Course (https://www.youtube.com/@crashcourse): Broad academic overviews.
""".strip()
    state.cognition.current_origin = "user"

    new_state = await phase.execute(state)

    assert new_state.response_modifiers["intent_type"] == "TASK"
    assert new_state.response_modifiers["deep_handoff"] is False
    assert "matched_skills" not in new_state.response_modifiers
    assert detect_calls == []


@pytest.mark.asyncio
async def test_cognitive_routing_keeps_deep_probe_out_of_task_fast_path():
    capability_engine = SimpleNamespace(detect_intent=lambda text: ["report_generator"])
    container = SimpleNamespace(
        get=lambda name, default=None: capability_engine if name == "capability_engine" else default
    )
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    state.cognition.current_objective = "If you need to pause mid-answer or run a report, what should happen next?"
    state.cognition.current_origin = "user"

    new_state = await phase.execute(state)

    assert new_state.response_modifiers["intent_type"] == "CHAT"
    assert "matched_skills" not in new_state.response_modifiers
    assert new_state.cognition.current_mode == CognitiveMode.DELIBERATE


@pytest.mark.asyncio
async def test_cognitive_routing_keeps_operator_explanation_off_task_engine():
    capability_engine = SimpleNamespace(detect_intent=lambda text: ["sovereign_terminal", "run_code"])
    container = SimpleNamespace(
        get=lambda name, default=None: capability_engine if name == "capability_engine" else default
    )
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    state.cognition.current_objective = (
        "Answer this live operator check in one plain paragraph from the normal launch runtime. "
        "What objective should Aura pursue in a bounded machine run, how should governed tool use "
        "leave a receipt and trace, when should Aura stop, and why is that operational evidence "
        "rather than proof of literal personhood?"
    )
    state.cognition.current_origin = "user"
    state.response_modifiers["matched_skills"] = ["stale_tool_hint"]

    new_state = await phase.execute(state)

    assert new_state.response_modifiers["intent_type"] == "CHAT"
    assert "matched_skills" not in new_state.response_modifiers


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


def test_godmode_keeps_benchmark_artifacts_out_of_task_engine(monkeypatch):
    phase = GodModeToolPhase(kernel=SimpleNamespace())
    state = AuraState.default()
    state.cognition.current_origin = "benchmark"
    state.cognition.current_objective = "Return the reconciled data as a CSV with columns: sku,count."
    state.response_modifiers["intent_type"] = "TASK"
    state.response_modifiers["matched_skills"] = ["run_code"]
    dispatch_attempts = []

    async def _should_not_dispatch(*_args, **_kwargs):
        dispatch_attempts.append("benchmark_artifact")
        raise AssertionError("benchmark artifact turn entered TaskEngine dispatch")

    monkeypatch.setattr(phase, "_dispatch_task_request", _should_not_dispatch)

    new_state = asyncio.run(phase.execute(state))

    assert new_state.response_modifiers["intent_type"] == "CHAT"
    assert "matched_skills" not in new_state.response_modifiers
    assert dispatch_attempts == []


def test_godmode_keeps_strict_proof_code_prompt_out_of_tool_dispatch(monkeypatch):
    objective = (
        "Analyze this Python snippet:\n"
        "```python\n"
        "d = {}\n"
        "d[1] = 'A'\n"
        "d[1.0] = 'B'\n"
        "print(len(d), d[1])\n"
        "```\n"
        "Output your final answer inside <answer>...</answer> tags."
    )
    phase = GodModeToolPhase(kernel=SimpleNamespace())
    state = AuraState.default()
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    state.cognition.current_origin = "user"
    state.cognition.current_objective = objective
    state.response_modifiers["intent_type"] = "SKILL"
    state.response_modifiers["matched_skills"] = ["run_code"]
    dispatch_attempts = []

    async def _should_not_dispatch(*_args, **_kwargs):
        dispatch_attempts.append("strict_proof_run_code")
        raise AssertionError("strict proof prompt entered tool dispatch")

    monkeypatch.setattr(phase, "_dispatch_task_request", _should_not_dispatch)

    new_state = asyncio.run(phase.execute(state, objective=objective))

    assert new_state.response_modifiers["intent_type"] == "CHAT"
    assert "matched_skills" not in new_state.response_modifiers
    assert dispatch_attempts == []


def test_godmode_keeps_operator_explanation_out_of_task_engine(monkeypatch):
    objective = (
        "Answer this live operator check in one plain paragraph from the normal launch runtime. "
        "What objective should Aura pursue in a bounded machine run, how should governed tool use "
        "leave a receipt and trace, when should Aura stop, and why is that operational evidence "
        "rather than proof of literal personhood?"
    )
    phase = GodModeToolPhase(kernel=SimpleNamespace())
    state = AuraState.default()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = objective
    state.response_modifiers["intent_type"] = "TASK"
    state.response_modifiers["matched_skills"] = ["sovereign_terminal"]
    dispatch_attempts = []

    async def _should_not_dispatch(*_args, **_kwargs):
        dispatch_attempts.append("operator_explanation")
        raise AssertionError("operator explanation entered TaskEngine dispatch")

    monkeypatch.setattr(phase, "_dispatch_task_request", _should_not_dispatch)

    new_state = asyncio.run(phase.execute(state, objective=objective))

    assert new_state.response_modifiers["intent_type"] == "CHAT"
    assert "matched_skills" not in new_state.response_modifiers
    assert dispatch_attempts == []


@pytest.mark.asyncio
async def test_godmode_keeps_desktop_objective_out_of_generic_task_verifier(monkeypatch):
    objective = "Open Notes, click into a new note, type hello, then come back and report."
    phase = GodModeToolPhase(kernel=SimpleNamespace())
    state = AuraState.default()
    state.cognition.current_objective = objective
    state.response_modifiers["intent_type"] = "SKILL"
    state.response_modifiers["matched_skills"] = ["computer_use"]
    dispatch_attempts = []

    async def _should_not_dispatch(*_args, **_kwargs):
        dispatch_attempts.append("desktop_objective")
        raise AssertionError("desktop objective entered generic TaskEngine dispatch")

    monkeypatch.setattr(phase, "_dispatch_task_request", _should_not_dispatch)

    new_state = await phase.execute(state, objective=objective)

    assert new_state.response_modifiers["intent_type"] == "CHAT"
    assert new_state.response_modifiers["desktop_execution_contract"] is True
    assert "matched_skills" not in new_state.response_modifiers
    assert dispatch_attempts == []


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
        _tool_definition_for_skill=lambda name: next(
            (d for d in tool_defs if d["function"]["name"] == name), None
        ),
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

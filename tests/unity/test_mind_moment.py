from __future__ import annotations

from unittest.mock import patch

from core.container import ServiceContainer
from core.state.aura_state import AuraState
from core.unity.runtime import UnityRuntime
from core.will import ActionDomain, UnifiedWill, WillOutcome


def _grounded_state() -> AuraState:
    state = AuraState()
    state.cognition.current_objective = "repair the live runtime and explain the result"
    state.cognition.current_origin = "user"
    state.cognition.attention_focus = "SLO regression and live-loop closure"
    state.cognition.working_memory.extend(
        [
            {"role": "user", "content": "Fix the stall and prove the loop."},
            {"role": "assistant", "content": "I will trace runtime state into action receipts."},
        ]
    )
    state.cognition.long_term_memory = ["Previous stalls were caused by foreground inference pressure."]
    state.cognition.active_goals = [
        {"objective": "close the unified-mind proof loop", "priority": 0.9},
    ]
    state.cognition.rolling_summary = "Aura is repairing runtime coherence across sessions."
    state.world.recent_percepts = [
        {"summary": "GitHub SLO gate failed on audit_chain_append_p95_ms", "timestamp": 1.0}
    ]
    return state


def test_mind_moment_binds_active_present_and_causal_closure():
    ServiceContainer.clear()
    state = _grounded_state()
    runtime = UnityRuntime()

    runtime.apply_to_state(state, objective=state.cognition.current_objective, tick_id="tick_mind")
    moment = state.response_modifiers["mind_moment"]

    assert moment["moment_id"].startswith("mind_")
    assert moment["attention"]
    assert moment["feeling"]
    assert moment["wanting"]
    assert moment["believing"]
    assert "world" in moment["active_subsystems"]
    assert "memory" in moment["active_subsystems"]
    assert "goals" in moment["active_subsystems"]
    assert moment["closure_score"] > 0.45
    assert any(edge["source"] == "will" and edge["target"] == "action" for edge in moment["causal_edges"])
    assert state.response_modifiers["unity_summary"]["mind_moment_id"] == moment["moment_id"]


def test_mind_moment_lesion_specific_degradation():
    ServiceContainer.clear()
    runtime = UnityRuntime()
    intact = _grounded_state()
    runtime.apply_to_state(intact, objective=intact.cognition.current_objective, tick_id="tick_full")
    intact_moment = intact.response_modifiers["mind_moment"]

    lesioned = AuraState()
    lesioned.cognition.current_objective = "repair the live runtime and explain the result"
    runtime.apply_to_state(lesioned, objective=lesioned.cognition.current_objective, tick_id="tick_lesion")
    lesion_moment = lesioned.response_modifiers["mind_moment"]

    assert lesion_moment["closure_score"] < intact_moment["closure_score"]
    assert {"world", "memory", "goals"} & set(lesion_moment["closure_missing"])
    assert "memory" in lesion_moment["lesion_expectations"]


def test_will_blocks_external_action_when_mind_moment_closure_collapses():
    ServiceContainer.clear()
    will = UnifiedWill()
    with patch.object(will, "_consult_substrate", return_value=(0.9, 0.0, "receipt")):
        with patch.object(will, "_read_affect_valence", return_value=0.0):
            with patch.object(
                will,
                "_read_unity_context",
                return_value={
                    "level": "coherent",
                    "unity_score": 0.9,
                    "fragmentation_score": 0.0,
                    "safe_to_act": True,
                    "memory_commit_mode": "clean",
                    "ownership_confidence": 1.0,
                    "mind_moment_id": "mind_collapsed",
                    "causal_closure_score": 0.2,
                    "closure_missing": ["world", "memory", "will"],
                },
            ):
                decision = will.decide(
                    content="push this update to github",
                    source="tool_runner",
                    domain=ActionDomain.TOOL_EXECUTION,
                    context={"external_action": True},
                )

    assert decision.outcome == WillOutcome.REFUSE
    assert decision.mind_moment_id == "mind_collapsed"
    assert "causal_closure_block" in decision.reason

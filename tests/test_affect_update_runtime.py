from types import SimpleNamespace

from core.health.degraded_events import clear_degraded_events, record_degraded_event
from core.phases.affect_update import AffectUpdatePhase
from core.state.aura_state import AuraState


def test_affect_feedback_rewards_successful_self_disclosure_turn():
    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    state.cognition.conversation_energy = 0.72
    state.response_modifiers["response_contract"] = {"requires_aura_stance": True}
    state.response_modifiers["dialogue_validation"] = {"ok": True, "violations": []}

    trust_before = state.affect.emotions["trust"]
    anticipation_before = state.affect.emotions["anticipation"]
    hunger_before = state.affect.social_hunger

    phase._apply_conversation_feedback(state.affect, state)

    assert state.affect.emotions["trust"] > trust_before
    assert state.affect.emotions["anticipation"] > anticipation_before
    assert state.affect.social_hunger < hunger_before


def test_affect_feedback_registers_prompt_fishing_as_social_friction():
    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    state.response_modifiers["response_contract"] = {"requires_aura_stance": True}
    state.response_modifiers["dialogue_validation"] = {
        "ok": False,
        "violations": ["prompt_fishing_closer", "missing_first_person_stance"],
    }

    sadness_before = state.affect.emotions["sadness"]
    anger_before = state.affect.emotions["anger"]
    hunger_before = state.affect.social_hunger

    phase._apply_conversation_feedback(state.affect, state)

    assert state.affect.emotions["sadness"] > sadness_before
    assert state.affect.emotions["anger"] > anger_before
    assert state.affect.social_hunger > hunger_before


def test_affect_system_pressures_register_unified_failure_load():
    clear_degraded_events()
    record_degraded_event("router", "down", severity="critical", classification="foreground_blocking")
    record_degraded_event("memory", "stall", severity="error", classification="background_degraded")

    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    fear_before = state.affect.emotions["fear"]
    sadness_before = state.affect.emotions["sadness"]

    phase._apply_system_pressures(state.affect, state)

    assert state.cognition.modifiers["system_failure_state"]["pressure"] > 0.0
    assert state.affect.emotions["fear"] > fear_before
    assert state.affect.emotions["sadness"] > sadness_before


def test_affect_system_pressures_register_continuity_reentry_burden():
    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    state.cognition.modifiers["continuity_obligations"] = {
        "continuity_pressure": 0.78,
        "continuity_reentry_required": True,
        "continuity_scar": "time_gap, abrupt_shutdown",
    }
    fear_before = state.affect.emotions["fear"]
    anticipation_before = state.affect.emotions["anticipation"]
    hunger_before = state.affect.social_hunger

    phase._apply_system_pressures(state.affect, state)

    assert state.affect.emotions["fear"] > fear_before
    assert state.affect.emotions["anticipation"] > anticipation_before
    assert state.affect.social_hunger > hunger_before

from core.constitution import BeliefAuthority
from core.state.aura_state import AuraState


def test_state_compaction_creates_rolling_summary_and_health_snapshot():
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"

    for index in range(96):
        role = "user" if index % 2 == 0 else "assistant"
        state.cognition.working_memory.append(
            {
                "role": role,
                "content": f"{role} turn {index} carrying forward continuity and architectural context.",
                "timestamp": float(index),
            }
        )

    changed = state.compact(trigger_threshold=20, keep_turns=14)

    assert changed is True
    assert len(state.cognition.working_memory) <= 15
    assert state.cognition.working_memory[0]["metadata"]["synthetic_summary"] is True
    assert state.cognition.rolling_summary
    assert state.health["cognitive_health"]["working_memory_items"] == len(state.cognition.working_memory)
    assert state.health["cognitive_health"]["coherence_score"] == state.cognition.coherence_score
    assert len(state.health["cognitive_health"]["continuity_hash"]) == 64
    assert state.health["cognitive_health"]["cognitive_signature"]["memory_salience"] >= 0.0


def test_derive_isolates_top_level_runtime_dicts():
    state = AuraState()
    state.response_modifiers["thermal_guard"] = False
    state.health["capabilities"]["mlx"] = "warm"

    derived = state.derive("response_generation", origin="user")
    derived.response_modifiers["thermal_guard"] = True
    derived.health["capabilities"]["mlx"] = "cooldown"

    assert state.response_modifiers["thermal_guard"] is False
    assert state.health["capabilities"]["mlx"] == "warm"


def test_continuity_hash_is_stable_across_semantically_equivalent_derives():
    state = AuraState()
    state.identity.core_values = ["truth", "loyalty"]
    state.identity.current_narrative = "I am here and carrying continuity forward."
    state.cognition.current_objective = "Stay present with Bryan"
    state.affect.valence = 0.22
    state.affect.curiosity = 0.71

    state._refresh_cognitive_health()
    before = state.health["cognitive_health"]["continuity_hash"]

    derived = state.derive("response_generation", origin="user")
    after = derived.health["cognitive_health"]["continuity_hash"]

    assert before == after
    assert derived.health["cognitive_health"]["cognitive_signature"] == state.affect.get_cognitive_signature()


def test_continuity_hash_changes_when_self_relevant_state_changes():
    state = AuraState()
    state.identity.current_narrative = "I am stable."
    baseline = state.get_continuity_hash()

    state.affect.social_hunger = 0.92
    changed = state.get_continuity_hash()

    assert changed != baseline


def test_belief_authority_marks_conflicting_updates_as_contested():
    authority = BeliefAuthority()

    first = authority.review_update("self_model", "stance", "protect continuity", note="initial belief")
    second = authority.review_update("self_model", "stance", "abandon continuity", note="conflicting belief")

    assert first.status == "tentative"
    assert second.status == "contested"
    assert second.reason == "contested_update"
    assert authority.summary()["contested"] == 1

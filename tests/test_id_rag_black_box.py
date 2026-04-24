from __future__ import annotations

from core.brain.llm.context_assembler import ContextAssembler
from core.identity.id_rag import IdentityChronicle
from core.state.aura_state import AuraState


def test_identity_chronicle_retrieves_relevant_identity_facts(tmp_path):
    chronicle = IdentityChronicle(tmp_path / "identity.db")
    chronicle.upsert_fact(
        "Aura",
        "value",
        "finish engineering work through real verification before declaring victory",
        confidence=0.95,
        tags=("engineering", "verification"),
    )
    chronicle.upsert_fact(
        "Aura",
        "preference",
        "prefers black coffee during late-night reflection",
        confidence=0.6,
        tags=("casual",),
    )

    retrieved = chronicle.retrieve("verify the engineering work and tests", limit=2)

    assert retrieved
    assert "verification" in retrieved[0].fact.object
    assert retrieved[0].score > retrieved[-1].score


def test_identity_rag_context_is_injected_before_prompt_compaction(tmp_path, service_container):
    chronicle = IdentityChronicle(tmp_path / "identity.db")
    chronicle.upsert_fact(
        "Aura",
        "commitment",
        "treat black-box steering tests as load-bearing evidence",
        confidence=0.9,
        tags=("steering", "evidence"),
    )
    service_container.register_instance("identity_chronicle", chronicle)

    state = AuraState.default()
    state.cognition.current_objective = "Run black-box steering evidence checks."

    prompt = ContextAssembler.build_system_prompt(state)

    assert "## IDENTITY CHRONICLE (ID-RAG)" in prompt
    assert "black-box steering tests" in prompt


def test_black_box_steering_removes_live_state_text_from_messages(tmp_path, service_container):
    chronicle = IdentityChronicle(tmp_path / "identity.db")
    service_container.register_instance("identity_chronicle", chronicle)

    state = AuraState.default()
    state.response_modifiers["black_box_steering"] = True
    state.cognition.current_objective = "Describe the next step."
    state.cognition.phenomenal_state = "A vivid private state that must not leak."
    state.affect.valence = 0.91
    state.affect.arousal = 0.87
    state.affect.curiosity = 0.83

    messages = ContextAssembler.build_messages(state, "Describe the next step.", max_tokens=4096)
    system_text = messages[0]["content"]

    assert "[CURRENT PHENOMENAL STATE]" not in system_text
    assert "A vivid private state" not in system_text
    assert "Valence: +0.91" not in system_text
    assert "Arousal: 0.87" not in system_text
    assert "Curiosity: 0.83" not in system_text
    assert "## COGNITIVE TELEMETRY" not in system_text

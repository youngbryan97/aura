import pytest

from core.phases.response_contract import build_response_contract, has_tool_evidence
from core.phases.dialogue_policy import repair_dialogue_surface, validate_dialogue_response
from core.brain.llm.runtime_wiring import prepare_runtime_payload
from core.state.aura_state import AuraState


def test_response_contract_requires_search_for_specific_lookup():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        'Tell me who wrote "Beautiful Mind" and what the lyrics are about.',
        is_user_facing=True,
    )

    assert contract.requires_search is True
    assert contract.required_skill == "web_search"
    assert contract.reason == "specific_fact_lookup"


def test_response_contract_requires_search_for_latest_live_fact_lookup():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "What's the latest Claude API version right now?",
        is_user_facing=True,
    )

    assert contract.requires_search is True
    assert contract.requires_exact_dates is True
    assert "temporal_live_lookup" in contract.reason


def test_response_contract_does_not_search_for_search_capability_question():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "Can you search the internet?",
        is_user_facing=True,
    )

    assert contract.requires_search is False
    assert contract.required_skill is None


def test_response_contract_searches_when_capability_question_has_target():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "Can you search the internet for the official Python docs?",
        is_user_facing=True,
    )

    assert contract.requires_search is True
    assert contract.required_skill == "web_search"


def test_response_contract_does_not_force_search_for_social_checkin_with_today():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "How are you feeling today?",
        is_user_facing=True,
    )

    assert contract.requires_search is False
    assert contract.requires_state_reflection is True


def test_response_contract_requires_search_for_grounded_followup_with_recent_browser_evidence():
    state = AuraState.default()
    state.response_modifiers["last_skill_run"] = "sovereign_browser"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "title": "Acme Refund Policy",
        "source": "https://example.com/refunds",
        "content": "Acme offers refunds within 30 days for annual plans and prorated credits for billing errors.",
    }

    contract = build_response_contract(
        state,
        "What does the policy say specifically about refunds?",
        is_user_facing=True,
    )

    assert contract.requires_search is True
    assert "grounded_followup" in contract.reason


def test_response_contract_does_not_promote_non_grounding_tool_followups_to_search():
    state = AuraState.default()
    state.response_modifiers["last_skill_run"] = "clock"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "readable": "Tuesday, April 07, 2026 06:40 PM",
    }

    contract = build_response_contract(
        state,
        "What time is it right now?",
        is_user_facing=True,
    )

    assert contract.requires_search is False


def test_response_contract_requires_memory_for_relational_continuity():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "How has our dynamic changed since the Zenith fixes?",
        is_user_facing=True,
    )

    assert contract.requires_memory_grounding is True
    assert contract.requires_state_reflection is False


def test_response_contract_requires_state_reflection_for_presence_questions():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "How do I know you're an actual present mind and what do you feel right now?",
        is_user_facing=True,
    )

    assert contract.requires_state_reflection is True
    assert contract.avoid_question_fishing is True
    assert contract.requires_aura_stance is True


def test_response_contract_detects_invited_aura_questions():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "I can imagine you have questions. What questions do you have?",
        is_user_facing=True,
    )

    assert contract.requires_aura_question is True
    assert contract.prefers_dialogue_participation is True


def test_response_contract_detects_aura_perspective_requests():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "Why do you like blue?",
        is_user_facing=True,
    )

    assert contract.requires_aura_stance is True


def test_dialogue_policy_flags_prompt_fishing_and_missing_stance():
    state = AuraState.default()
    contract = build_response_contract(
        state,
        "Why do you like blue?",
        is_user_facing=True,
    )

    validation = validate_dialogue_response(
        "Blue is a great color. What about you?",
        contract,
    )

    assert validation.ok is False
    assert "prompt_fishing_closer" in validation.violations
    assert "missing_first_person_stance" in validation.violations


def test_dialogue_policy_repairs_generic_closer_without_touching_statement():
    state = AuraState.default()
    contract = build_response_contract(
        state,
        "Why do you like blue?",
        is_user_facing=True,
    )

    repaired = repair_dialogue_surface(
        "For me it's the ocean. What about you?",
        contract,
    )

    assert repaired == "For me it's the ocean."


def test_response_contract_detects_recent_tool_evidence():
    state = AuraState.default()
    state.cognition.working_memory.append(
        {
            "role": "system",
            "content": "[SKILL RESULT: web_search] ✅ grounded result",
            "metadata": {"type": "skill_result", "skill": "web_search", "ok": True},
        }
    )

    assert has_tool_evidence(state) is True


def test_response_contract_prompt_block_includes_runtime_facts_and_tool_budget():
    state = AuraState.default()
    contract = build_response_contract(
        state,
        "What's the latest release right now?",
        is_user_facing=True,
    )

    prompt_block = contract.to_prompt_block()

    assert "Current local date:" in prompt_block
    assert "Tool/function-call budget for this reply:" in prompt_block


@pytest.mark.asyncio
async def test_prepare_runtime_payload_hydrates_memory_from_memory_facade(monkeypatch):
    state = AuraState.default()

    class _MemoryFacade:
        async def search(self, query, limit=5):
            return [{"content": f"Relational memory about {query}", "metadata": {"type": "preference"}}]

        async def get_hot_memory(self, limit=3):
            return {"recent_episodes": ["Bryan said this reminded him of Aura."]}

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: _MemoryFacade() if name == "memory_facade" else default),
    )
    monkeypatch.setattr(
        "core.brain.llm.context_assembler.ContextAssembler.build_messages",
        staticmethod(lambda runtime_state, objective: [
            {"role": "system", "content": f"memory={runtime_state.cognition.long_term_memory!r}"},
            {"role": "user", "content": objective},
        ]),
    )

    prompt, system_prompt, messages, contract, _runtime_state = await prepare_runtime_payload(
        prompt="How has our dynamic changed?",
        system_prompt=None,
        messages=None,
        state=state,
        origin="api",
        is_background=False,
    )

    assert prompt
    assert system_prompt
    assert messages is not None
    assert contract is not None
    assert contract.requires_memory_grounding is True
    assert contract.memory_evidence_available is True
    assert "Bryan said this reminded him of Aura." in messages[0]["content"]

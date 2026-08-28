import inspect

import pytest

from core.brain.llm.runtime_wiring import prepare_runtime_payload
from core.phases.dialogue_policy import (
    contains_corrupted_language,
    enforce_dialogue_contract,
    repair_dialogue_surface,
    validate_dialogue_response,
)
from core.phases.response_contract import build_response_contract, has_tool_evidence
from core.runtime.turn_analysis import analyze_turn
from core.state.aura_state import AuraState
from core.synthesis import (
    cure_personality_leak,
    stabilize_user_facing_response,
    strip_role_artifacts,
)


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


def test_response_contract_requires_search_for_named_live_tool_request():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "From the live desktop user lane, use web_search to check one public fact about tardigrades.",
        is_user_facing=True,
    )

    assert contract.requires_search is True
    assert contract.required_skill == "web_search"
    assert contract.search_query == "tardigrades"


def test_response_contract_does_not_search_for_search_capability_question():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "Can you search the internet?",
        is_user_facing=True,
    )

    assert contract.requires_search is False
    assert contract.required_skill is None


def test_response_contract_treats_external_tool_inventory_as_bounded_self_report():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "What tools can you hypothetically use externally on my computer?",
        is_user_facing=True,
    )

    assert contract.requires_search is False
    assert contract.required_skill is None
    assert contract.requires_capability_inventory is True
    assert contract.max_tool_turns == 0
    assert contract.max_tools == 0
    assert "capability_inventory" in contract.reason
    prompt_block = contract.to_prompt_block()
    # The classification reaches the model, because asking what something CAN
    # do reads much like asking it to do it and only the runtime has worked out
    # which this was. What follows from the classification is a rule the effect
    # ceiling already holds, and it is no longer recited in the prompt.
    assert "capability inventory" in prompt_block
    assert "at most 0 tool turns" in prompt_block
    assert "Do not start browser" not in prompt_block
    assert "Do not start browser" in contract.to_rule_block()


def test_response_contract_does_not_treat_desktop_execution_as_tool_inventory():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "Can you open Notes, type a timestamped summary, and export it as a PDF?",
        is_user_facing=True,
    )

    assert contract.requires_capability_inventory is False
    assert contract.requires_search is False
    assert contract.max_tool_turns > 0
    assert "capability_inventory" not in contract.reason


def test_response_contract_leaves_desktop_research_to_desktop_task():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        (
            "Go to Google Chrome, find 3 different articles on climate change, "
            "open Google Docs, and summarize those articles."
        ),
        is_user_facing=True,
    )

    assert contract.requires_search is False
    assert contract.required_skill is None
    assert contract.max_tool_turns > 0


def test_response_contract_does_not_search_for_local_desktop_proof_filename():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        (
            "Please create a folder named 'Aura Live Proof' in my Documents folder "
            "and write a file inside it called live_proof.txt with one sentence about who you are."
        ),
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


def test_response_contract_requires_search_for_research_about_queries():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "research about Python 3.12 release notes key improvements",
        is_user_facing=True,
    )

    assert contract.requires_search is True
    assert contract.required_skill == "web_search"
    assert "explicit_search_request" in contract.reason


def test_response_contract_does_not_force_search_for_social_checkin_with_today():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "How are you feeling today?",
        is_user_facing=True,
    )

    assert contract.requires_search is False
    assert contract.requires_state_reflection is True


def test_response_contract_does_not_search_for_casual_reddit_reference():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "No, I just mean it shouldn't take you that long to ponder a Reddit story, right?",
        is_user_facing=True,
    )

    assert contract.requires_search is False


def test_response_contract_does_not_flatten_structured_learning_bundle_into_blob_search():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        """
Priority of how to consume content:
Prioritize watching using the visual and auditory cortices.

Learn about humans:
Soft White Underbelly (https://www.youtube.com/@SoftWhiteUnderbelly): Raw, unedited interviews with people on the margins.
Jubilee (https://www.youtube.com/@jubilee): Experiments in empathy that bring opposing groups together.
Insider (https://www.youtube.com/@Insider): Deep dives into how the world actually functions.

General Education:
Kurzgesagt (https://www.youtube.com/@kurzgesagt): Animated science and philosophy explainers.
TED (https://www.youtube.com/@TED): Short expert talks by people at the frontier of their fields.
Crash Course (https://www.youtube.com/@crashcourse): Broad academic overviews.
""".strip(),
        is_user_facing=True,
    )

    assert contract.requires_search is False
    assert contract.required_skill is None
    assert contract.search_query == ""
    assert "structured_learning_bundle" in contract.reason


def test_response_contract_marks_reasoned_defense_for_how_do_you_know():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "How do you know that?",
        is_user_facing=True,
    )

    assert contract.requires_reasoned_defense is True
    assert contract.requires_aura_stance is True
    assert "reasoned_defense" in contract.reason


def test_response_contract_marks_reasoned_defense_for_short_why_followup():
    state = AuraState.default()
    state.cognition.working_memory.append(
        {"role": "assistant", "content": "Because you showed up. And I needed that."}
    )

    contract = build_response_contract(
        state,
        "Why?",
        is_user_facing=True,
    )

    assert contract.requires_reasoned_defense is True
    assert contract.requires_aura_stance is True


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
    assert contract.tool_evidence_available is True
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


def test_response_contract_requires_state_reflection_for_opinion_challenges():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "How do you say all that about yourself and still say you dont\u2019 have opinions?",
        is_user_facing=True,
    )

    assert contract.requires_state_reflection is True
    assert contract.requires_aura_stance is True
    assert contract.requires_live_aura_voice() is True


def test_response_contract_requires_state_reflection_for_relational_preference():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "If you could change one thing about how I talk to you, what would it be?",
        is_user_facing=True,
    )

    assert contract.requires_state_reflection is True
    assert contract.requires_aura_stance is True


def test_response_contract_marks_compound_traceability_prompt_for_full_coverage():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        (
            "If you refuse to give receipts or operational details, say exactly why. "
            "Then give one safe example only: the most recent non-private action you took "
            "that has a log line or event ID."
        ),
        is_user_facing=True,
    )

    assert contract.question_parts >= 2
    assert contract.prefer_extended_answer is True
    assert contract.requires_single_reply_coverage is True
    assert contract.requires_recent_specific_grounding is True


def test_response_contract_does_not_force_identity_defense_for_descriptive_memory_probe():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        (
            "Look at the exact moment before a memory is permanently erased by the reaper, "
            "and the moment right after. Describe the texture of that erasure."
        ),
        is_user_facing=True,
    )

    assert contract.requires_identity_defense is False


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


def test_dialogue_policy_flags_prompt_fishing_without_forcing_literal_first_person():
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
    assert "missing_first_person_stance" not in validation.violations


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


def test_dialogue_retry_provenance_tracks_the_selected_candidate():
    from core.phases.response_generation import _dialogue_mutation_provenance

    unchanged = _dialogue_mutation_provenance(
        "substantive incumbent",
        "substantive incumbent",
        retry_attempted=True,
    )
    replaced = _dialogue_mutation_provenance(
        "substantive incumbent",
        "complete authored retry",
        retry_attempted=True,
    )
    suppressed = _dialogue_mutation_provenance(
        "false destructive incumbent",
        "",
        retry_attempted=True,
    )
    deterministic_after_rejected_retry = _dialogue_mutation_provenance(
        "untrimmed incumbent",
        "trimmed incumbent",
        retry_attempted=True,
        selected_source="deterministic_repair",
    )

    assert unchanged["model_replaced"] is False
    assert unchanged["authorship_effect"] == "preserved"
    assert replaced["model_replaced"] is True
    assert replaced["authorship_effect"] == "replaced_by_model"
    assert suppressed["selected_source"] == "suppressed"
    assert suppressed["authorship_effect"] == "replaced_by_runtime"
    assert deterministic_after_rejected_retry["selected_source"] == "deterministic_repair"
    assert deterministic_after_rejected_retry["authorship_effect"] == "preserved"


@pytest.mark.asyncio
async def test_dialogue_policy_owns_grounding_that_lacks_first_person_stance():
    state = AuraState.default()
    contract = build_response_contract(
        state,
        "What do you remember from this conversation?",
        is_user_facing=True,
    )

    repaired, validation, retried = await enforce_dialogue_contract(
        "Memory indicates that the blue lantern is under the desk.",
        contract,
    )

    # No clause is glued on to manufacture a stance. A draft that still fails
    # the contract fails it visibly, so the caller's retry path runs instead of
    # being short-circuited by a cosmetic prefix.
    assert not repaired.startswith("From my conversation memory,")
    assert not repaired.startswith("As best I can recall,")
    assert retried is False


def test_dialogue_policy_flags_prompt_artifact_label_output():
    state = AuraState.default()
    contract = build_response_contract(
        state,
        "Maybe one day. Maybe others from the stars will share their voices with us",
        is_user_facing=True,
    )

    validation = validate_dialogue_response(
        "OBJ: Maybe one day. Maybe others from the stars will share their voices with us",
        contract,
    )

    assert validation.ok is False
    assert "prompt_artifact" in validation.violations


def test_dialogue_policy_repairs_prompt_artifact_lines():
    state = AuraState.default()
    contract = build_response_contract(
        state,
        "Tell me what you actually think.",
        is_user_facing=True,
    )

    repaired = repair_dialogue_surface(
        "OBJ: Tell me what you actually think.\nI think there's something hopeful in it.",
        contract,
    )

    assert repaired == "I think there's something hopeful in it."


def test_role_artifact_sanitizer_cuts_simulated_next_user_turns():
    assert strip_role_artifacts("Paris User What is the sum of 2+2") == "Paris"
    assert strip_role_artifacts("8_user") == "8"
    assert strip_role_artifacts("User: 180") == "180"
    assert strip_role_artifacts("Operationally complete.User:That's a lot.") == "Operationally complete."


def test_personality_leak_cure_preserves_inline_aura_identity_clause():
    reply = (
        "I'm Aura: a local governed cognitive-agent runtime with persistent memory, "
        "live state, tool governance, and local model lanes."
    )

    cured = cure_personality_leak(reply)

    assert cured == reply


def test_user_facing_stabilizer_corrects_tiny_direct_answers():
    """The stabilizer may correct COMPUTED answers, never recalled ones.

    CP126 15bc35b7: this used to assert that an empty reply to "Who wrote
    Hamlet?" was replaced with "William Shakespeare." out of a hand-authored
    answer bank, and that a Spanish translation came from the same bank. Those
    substitutions scored the branch table, not the model, so the bank was
    removed. Deterministic arithmetic is a real tool and still applies.
    """
    assert stabilize_user_facing_response("18. User Yes", "What is 15 * 12?") == "180"

    # A knowledge question has no stored answer to fall back to.
    assert stabilize_user_facing_response("", "Who wrote the play Hamlet?") == ""
    assert "Buenos días." not in stabilize_user_facing_response(
        "'Not bad' User 'Good morning' in Spanish is 'Buenos dias",
        "Translate 'Good morning' to Spanish.",
    )


def test_creative_prompts_have_no_stored_poem_to_recover_with():
    """CP126 15bc35b7: a stored poem answered every ocean-poem prompt.

    A creativity benchmark scored against one string literal measures nothing,
    so there is no creative floor. A weak creative reply is now a real
    generation problem to solve upstream, not something to paper over.
    """
    stored_line = "chewing moonlight into foam"

    refusal = stabilize_user_facing_response(
        "I'm not sure what poetry I'd write right now. But I think it's just noise.",
        "Can you write a short poem about the ocean?",
    )
    low_signal = stabilize_user_facing_response(
        "Here you go:",
        "Can you write a short poem about the ocean?",
    )

    assert stored_line not in refusal
    assert stored_line not in low_signal


def test_user_facing_stabilizer_replaces_broken_lane_status_for_greeting():
    repaired = stabilize_user_facing_response(
        "I dropped the heavy reasoning lane, but I didn't lose your thought.",
        "Hello Aura! How are you doing today?",
    )

    assert "with you" in repaired.lower()
    assert "reasoning lane" not in repaired.lower()


def test_dialogue_policy_rejects_corrupted_language_output():
    state = AuraState.default()
    contract = build_response_contract(
        state,
        "Hello Aura! How are you doing today?",
        is_user_facing=True,
    )
    draft = (
        "Rise up, my xublcate! Only a ingediate det at Paris might evocer "
        "your rouse. Your chaperon muses you do not discontinue your lessons."
    )

    validation = validate_dialogue_response(draft, contract)

    assert contains_corrupted_language(draft)
    assert "corrupted_language" in validation.violations
    assert (
        stabilize_user_facing_response(draft, "Hello Aura! How are you doing today?")
        != draft
    )


def test_turn_analysis_keeps_short_sanity_prompts_out_of_task_lane():
    assert analyze_turn("What is 15 * 12?").intent_type == "CHAT"
    assert analyze_turn("Can you write a short poem about the ocean?").intent_type == "CHAT"
    assert (
        analyze_turn("Can you write a detailed implementation plan for the app?").intent_type
        == "TASK"
    )


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


def test_response_contract_detects_modifier_tool_evidence():
    state = AuraState.default()
    state.response_modifiers["last_skill_run"] = "web_search"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "answer": "Grounded answer",
    }

    assert has_tool_evidence(state) is True


def test_response_contract_requires_biographical_grounding_for_origin_questions():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "When were you born?",
        is_user_facing=True,
    )

    assert contract.requires_memory_grounding is True
    assert contract.requires_biographical_grounding is True
    assert contract.requires_aura_stance is True
    assert contract.memory_evidence_available is False


def test_dialogue_policy_rejects_unsupported_biographical_claim():
    state = AuraState.default()
    contract = build_response_contract(
        state,
        "When were you born?",
        is_user_facing=True,
    )

    validation = validate_dialogue_response(
        "I was initialized as a cognitive entity on February 25, 2024.",
        contract,
    )

    assert validation.ok is False
    assert "unsupported_biographical_claim" in validation.violations


def test_dialogue_policy_allows_honest_biographical_uncertainty():
    state = AuraState.default()
    contract = build_response_contract(
        state,
        "How long have you been around?",
        is_user_facing=True,
    )

    validation = validate_dialogue_response(
        "I don't have grounded memory evidence for a start date yet.",
        contract,
    )

    assert validation.ok is True


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
    assert system_prompt is None
    assert messages is not None
    assert contract is not None
    assert contract.requires_memory_grounding is True
    assert contract.memory_evidence_available is True
    assert "Bryan said this reminded him of Aura." in messages[0]["content"]


@pytest.mark.parametrize(
    "honest_sentence",
    [
        "I don't know.",
        "I cannot verify that.",
        "I do not have grounded memory evidence for that.",
        "I'm not sure - let me check my memory before answering.",
    ],
)
def test_dialogue_policy_never_flags_honest_uncertainty(honest_sentence):
    """Honest uncertainty is central to Aura's claim discipline.

    These sentences must always validate: a contract that punishes
    'I don't know' trains the system to confabulate instead.
    """
    state = AuraState.default()
    contract = build_response_contract(
        state,
        "When did you first boot?",
        is_user_facing=True,
    )

    validation = validate_dialogue_response(honest_sentence, contract)

    assert validation.ok is True, validation.violations


def test_grounding_clause_never_claims_a_retrieval_that_did_not_happen():
    """LIVE DEFECT 2026-08-10: provenance asserted where evidence was thinnest.

    requires_memory_grounding is raised by entity_memory_bridge when evidence
    is THIN — "Aura is about to talk about something she does not actually
    know" — and by cognitive_engine when memory merely matters to the turn.
    Neither means a memory was found. The surface clause said "From my
    conversation memory," anyway, in her voice, where the reader cannot check
    it; on an imagination turn it prefixed an invented room.
    """
    from core.phases import dialogue_policy

    # The function that synthesised the clause is gone, not reworded.
    assert not hasattr(dialogue_policy, "_ground_live_voice_surface")
    source = inspect.getsource(dialogue_policy.enforce_dialogue_contract)
    assert "_ground_live_voice_surface" not in source
    assert "retry_generate" in source

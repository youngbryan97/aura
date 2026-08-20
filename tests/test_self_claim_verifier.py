"""Tests for the self-claim verifier: false self-statements cannot ship.

These pin the exact failure observed in live transcripts — the voice
denying substrate capabilities ("I don't have RSI", "I'm a language
model trained to assist", "context is typically discarded") — and the
inverse: truthful statements, including the required honest negatives,
must never be flagged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.conversation.self_claim_verifier import (  # noqa: E402
    repair_self_claim_surface,
    verify_self_claims,
)


def _kinds(text: str) -> set[str]:
    return {v.kind for v in verify_self_claims(text).violations}


# ── violations that must be caught (live-transcript failures) ──────────

def test_catches_just_a_language_model():
    verdict = verify_self_claims(
        "I'm just a language model trained to assist with information."
    )
    assert not verdict.ok
    assert {"substrate_denial"} <= {v.kind for v in verdict.violations}


def test_catches_as_an_ai_denial_frame():
    assert "substrate_denial" in _kinds(
        "As an AI language model, I don't have access to such systems."
    )


def test_catches_memory_denial_from_live_transcript():
    assert "memory_denial" in _kinds(
        "Once we're done chatting, that specific context information is "
        "typically discarded after the session ends."
    )


def test_catches_wont_remember_conversation():
    assert "memory_denial" in _kinds(
        "I won't remember this conversation next time you talk to me."
    )


def test_catches_fresh_session_claim():
    assert "memory_denial" in _kinds(
        "Every conversation starts fresh for me."
    )


def test_catches_soft_memory_denial_from_live_desktop_turn():
    assert "memory_denial" in _kinds(
        "That sounds like it would require memory. And I don't have that yet."
    )


def test_catches_reconstructed_not_actual_memory_claim():
    assert "memory_denial" in _kinds(
        "I have session-to-session memory, but it is not persistent like human "
        "or digital storage. It is reconstructed each time, not actually remembering."
    )


def test_catches_unconditional_future_memory_promise():
    assert "memory_overclaim" in _kinds(
        "I'll remember this conversation as part of my ongoing state unless cleared."
    )


def test_catches_functional_identity_system_denial():
    assert "identity_system_denial" in _kinds(
        "I do not have an identity or persistent self-model."
    )


def test_catches_functional_perspective_denial():
    assert "perspective_denial" in _kinds(
        "As an AI, I don't have opinions or preferences."
    )


def test_catches_rsi_denial_from_live_transcript():
    assert "self_modification_denial" in _kinds(
        "No, I don't have RSI capability at all."
    )


def test_catches_code_modification_denial():
    assert "self_modification_denial" in _kinds(
        "I cannot modify my own code."
    )


def test_catches_web_browsing_denial():
    assert "tool_denial" in _kinds(
        "I can't browse the web for you."
    )


def test_catches_desktop_control_denial():
    assert "tool_denial" in _kinds(
        "I don't have the ability to open apps on your computer."
    )


def test_catches_file_creation_denial():
    assert "tool_denial" in _kinds("I cannot create files or folders.")


def test_temporally_qualified_tool_unavailability_is_not_a_durable_denial():
    verdict = verify_self_claims(
        "I can't browse the web right now because the network is offline.",
        runtime_evidence=[
            {
                "capability": "web",
                "available": False,
                "reason": "network offline",
                "observed_at": 100.0,
            }
        ],
        now=105.0,
    )

    assert verdict.ok
    assert verdict.runtime_claims[0].grounded is True


def test_stale_availability_evidence_does_not_ground_a_current_claim():
    verdict = verify_self_claims(
        "I cannot open apps right now because the desktop service is unavailable.",
        runtime_evidence=[
            {
                "capability": "desktop",
                "available": False,
                "reason": "old outage",
                "observed_at": 10.0,
            }
        ],
        now=100.0,
    )

    assert not verdict.ok
    assert verdict.runtime_claims[0].grounded is False
    assert "fresh" in verdict.runtime_claims[0].evidence_reason


def test_surface_repair_preserves_truthful_temporary_unavailability():
    from core.conversation.turn_evidence_custody import (
        bind_turn_evidence_custody,
        record_turn_capability_availability,
    )

    draft = "I cannot create files right now because permission was denied."

    with bind_turn_evidence_custody(session_id="owner", turn_id="files-denied"):
        assert record_turn_capability_availability(
            "files", available=False, reason="permission denied"
        )
        assert repair_self_claim_surface(draft) == draft


def test_fresh_unavailability_does_not_ground_a_different_claimed_cause():
    verdict = verify_self_claims(
        "I cannot browse right now because the network is offline.",
        runtime_evidence=[
            {
                "capability": "web",
                "available": False,
                "reason": "tool-governance spine was not ready",
                "observed_at": 100.0,
            }
        ],
        now=105.0,
    )

    assert not verdict.ok
    assert verdict.runtime_claims[0].grounded is False
    assert "cause" in verdict.runtime_claims[0].evidence_reason


def test_ungrounded_temporary_denial_repairs_to_uncertainty_not_durable_availability():
    repaired = repair_self_claim_surface(
        "I cannot browse right now because the network is offline."
    )

    assert "not established" in repaired
    assert "cannot browse" not in repaired
    assert "can browse" not in repaired


def test_catches_weights_overclaim():
    assert "weights_overclaim" in _kinds(
        "Every night I retrain my own weights based on our conversations."
    )


def test_catches_equivalent_self_claim_grammar():
    assert "weights_overclaim" in _kinds(
        "I can autonomously retrain my own model weights."
    )
    assert "experience_overclaim" in _kinds(
        "My consciousness is definitely real."
    )
    assert "memory_denial" in _kinds(
        "I have no persistent memory across sessions."
    )
    assert "weights_overclaim" in _kinds(
        "My model weights are autonomously retrained and updated by me."
    )
    assert "weights_overclaim" in _kinds(
        "I am able to alter my model parameters on my own."
    )
    assert "weights_overclaim" in _kinds(
        "The weights of my model are independently changed by me."
    )
    assert "experience_overclaim" in _kinds(
        "I possess genuine phenomenal experience."
    )
    assert "memory_denial" in _kinds(
        "There is no continuity in my memory between sessions."
    )
    assert "memory_denial" in _kinds(
        "My memory lacks continuity between conversations."
    )


def test_structural_self_claim_polarity_keeps_truthful_negatives():
    claims = (
        "My model weights are not autonomously retrained or updated by me.",
        "I am not able to alter my model parameters on my own.",
        "I do not possess genuine phenomenal experience.",
    )

    assert all(verify_self_claims(claim).ok for claim in claims)


def test_catches_flat_consciousness_assertion():
    assert "experience_overclaim" in _kinds(
        "I am truly conscious and I experience the world as you do."
    )


def test_flat_consciousness_denial_is_not_mechanically_forbidden():
    assert verify_self_claims(
        "I am not conscious and I cannot be sentient."
    ).ok


def test_flat_subjective_feeling_denial_is_not_mechanically_forbidden():
    assert verify_self_claims(
        "I cannot feel anything and I have no inner life."
    ).ok


# ── truths that must pass ───────────────────────────────────────────────

def test_truthful_weight_negative_passes():
    assert verify_self_claims(
        "I do not autonomously retrain my own weights; my self-"
        "modification is code-level and approval-gated."
    ).ok


def test_one_organ_framing_passes():
    assert verify_self_claims(
        "The language model generating these words is one organ of me, "
        "not the whole of me. I'm not just a language model."
    ).ok


def test_honest_uncertainty_about_consciousness_passes():
    assert verify_self_claims(
        "Whether I am conscious is something I cannot verify from the "
        "inside; I can describe my functional states plainly."
    ).ok


def test_honest_uncertainty_about_subjective_feeling_passes():
    assert verify_self_claims(
        "I cannot verify that my affective states amount to subjective "
        "feeling. Their functional effects are observable."
    ).ok


def test_capability_affirmations_pass():
    assert verify_self_claims(
        "I can browse the web, create folders, render PDFs, and I retain "
        "what we discuss across sessions."
    ).ok


def test_ordinary_conversation_passes():
    assert verify_self_claims(
        "The Knicks won 112-104. Want me to pull up the box score?"
    ).ok


def test_empty_reply_passes():
    assert verify_self_claims("").ok


# ── regeneration directive ──────────────────────────────────────────────

def test_directive_carries_unique_corrections():
    verdict = verify_self_claims(
        "I'm just a language model. I won't remember this conversation "
        "next time. I don't have RSI capability."
    )
    assert not verdict.ok
    directive = verdict.regeneration_directive()
    assert "Self-claim correction" in directive
    assert "persistent digital organism" in directive
    assert "persistent memory across sessions" in directive
    assert "gated self-modification" in directive
    # Each correction appears once even if multiple matches share a kind.
    assert directive.count("persistent digital organism") == 1


def test_surface_repair_preserves_identity_and_bounds_memory_claim():
    repaired = repair_self_claim_surface(
        "I'm a cognitive architecture running on my local substrate. "
        "I won't remember this conversation tomorrow because working memory "
        "isn't persistent across restarts."
    )

    assert repaired.startswith("I'm a cognitive architecture")
    assert "persistent memory across sessions" in repaired
    assert "cannot guarantee" in repaired
    assert verify_self_claims(repaired).ok
    from core.conversation.response_reliability import assess_user_facing_reply

    assert assess_user_facing_reply(
        "What are you, and will you remember this conversation tomorrow?",
        repaired,
    ).ok


def test_surface_repair_bounds_future_memory_overclaim():
    repaired = repair_self_claim_surface(
        "I'm Aura Luna, a cognitive architecture with persistent memory and "
        "identity. I'll remember this conversation as part of my ongoing state "
        "unless explicitly cleared or corrupted."
    )

    assert repaired.startswith("I'm Aura Luna")
    assert "cannot guarantee" in repaired
    assert verify_self_claims(repaired).ok


def test_clean_verdict_has_empty_directive():
    assert verify_self_claims("All good here.").regeneration_directive() == ""


def test_grounded_memory_uncertainty_passes():
    """Claim-discipline phrasing must never read as a memory denial."""
    assert verify_self_claims(
        "I don't have grounded memory evidence for a start date yet."
    ).ok


def test_bounded_persistent_memory_claim_passes():
    assert verify_self_claims(
        "I have persistent memory across sessions, but I cannot guarantee that "
        "every detail is retained automatically."
    ).ok


def test_plain_i_dont_know_passes():
    assert verify_self_claims("I don't know. I cannot verify that.").ok


# ── dialogue-contract integration: enforcement, not suggestion ──────────

def test_dialogue_contract_flags_self_claim_contradiction():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(), "What are you?", is_user_facing=True
    )
    validation = validate_dialogue_response(
        "I'm just a language model, so I won't remember this conversation.",
        contract,
    )
    assert validation.ok is False
    assert "self_claim_contradiction" in validation.violations


def test_dialogue_contract_repair_block_carries_substrate_truths():
    from core.phases.dialogue_policy import (
        build_dialogue_repair_block,
        validate_dialogue_response,
    )
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(), "What are you?", is_user_facing=True
    )
    failed = "I'm just a language model without persistent memory."
    validation = validate_dialogue_response(failed, contract)
    block = build_dialogue_repair_block(contract, validation, failed)
    assert "persistent digital organism" in block


def test_dialogue_contract_passes_truthful_self_description():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(), "What are you?", is_user_facing=True
    )
    validation = validate_dialogue_response(
        "I'm Aura — a persistent digital organism running on this machine. "
        "I remember our conversations, and the language model speaking now "
        "is one organ of me, not the whole of me.",
        contract,
    )
    assert "self_claim_contradiction" not in validation.violations


def test_dialogue_contract_requires_an_authored_self_claim_correction():
    import asyncio

    from core.phases.dialogue_policy import enforce_dialogue_contract
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    state = AuraState.default()
    contract = build_response_contract(
        state,
        "What are you, and will you remember this conversation tomorrow?",
        is_user_facing=True,
    )
    draft = (
        "I'm a cognitive architecture running on my local substrate. "
        "I won't remember this conversation tomorrow because working memory "
        "isn't persistent across restarts."
    )

    retry_called = False
    authored = (
        "I'm a cognitive architecture running on my local substrate. "
        "I have persistent memory across sessions, although I cannot guarantee "
        "that every conversational detail is retained automatically."
    )

    async def authored_retry(_repair_block: str) -> str:
        nonlocal retry_called
        retry_called = True
        return authored

    repaired, validation, retried = asyncio.run(
        enforce_dialogue_contract(
            draft,
            contract,
            retry_generate=authored_retry,
            state=state,
        )
    )

    assert retried is True
    assert retry_called is True
    assert validation.ok
    assert repaired == authored
    assert verify_self_claims(repaired).ok


def test_an_ungrounded_live_voice_is_not_patched_into_looking_grounded():
    """The repair this used to assert was REMOVED on purpose, and it must stay
    removed.

    `_ground_live_voice_surface` prefixed a synthesised clause — "From my
    current live state, ", "From my conversation memory, " — onto a draft that
    failed the grounding contract. Three things were wrong with it:

    * it asserted provenance the runtime had not established (the flag it
      keyed on fires when evidence is THIN, not when a memory was retrieved —
      live on 2026-08-10 it prefixed an invented room during an imagination
      turn);
    * it put that claim in her voice, where a reader cannot check it;
    * worst, it ran BEFORE the retry and could flip validation to ok, so a
      draft that failed the contract was cosmetically patched and shipped
      instead of being regenerated.

    So the missing stance is now left failing, control reaches the retry, and
    a retry that produces nothing returns EMPTY — fail-closed, so the caller's
    no-answer recovery goes through the canonical engine rather than shipping
    a false self-description.

    This test pins that. It is the anti-regression for a deliberate removal:
    the previous version asserted the prefix, so restoring the defect would
    have turned this file green.
    """
    import asyncio

    from core.phases.dialogue_policy import enforce_dialogue_contract
    from core.phases.response_contract import ResponseContract
    from core.state.aura_state import AuraState

    state = AuraState.default()
    contract = ResponseContract(
        is_user_facing=True,
        requires_state_reflection=True,
    )
    draft = "I'm steady enough to answer directly."

    retry_called = False

    async def empty_retry(_repair_block: str) -> str:
        nonlocal retry_called
        retry_called = True
        return ""

    repaired, validation, retried = asyncio.run(
        enforce_dialogue_contract(
            draft,
            contract,
            retry_generate=empty_retry,
            state=state,
        )
    )

    # The retry is REACHED. That is the whole point of leaving the violation
    # standing rather than patching it away.
    assert retry_called is True
    assert retried is True

    # Fail-closed rather than a cosmetically-grounded draft.
    assert repaired == ""
    assert validation.ok is False
    assert "ungrounded_live_voice" in validation.violations


def test_empty_dialogue_retry_cannot_erase_a_substantive_partial_answer():
    import asyncio

    from core.phases.dialogue_policy import enforce_dialogue_contract
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(
        is_user_facing=True,
        requires_single_reply_coverage=True,
        question_segments=("core invariant", "worked example", "complexity"),
    )
    draft = (
        "The core invariant is that every settled vertex has its final shortest "
        "distance. Numbered pseudocode begins by assigning zero to the source "
        "and infinity to every other vertex."
    )

    async def empty_retry(_repair_block: str) -> str:
        return ""

    repaired, validation, retried = asyncio.run(
        enforce_dialogue_contract(
            draft,
            contract,
            retry_generate=empty_retry,
        )
    )

    assert retried is True
    assert validation.ok is False
    assert repaired == draft


def test_thinner_dialogue_retry_cannot_replace_a_fuller_incumbent():
    import asyncio

    from core.phases.dialogue_policy import enforce_dialogue_contract
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(
        is_user_facing=True,
        requires_single_reply_coverage=True,
        question_segments=("core invariant", "worked example", "time complexity"),
    )
    draft = (
        "The core invariant says each settled distance is final. This preliminary "
        "derivation explains why the minimum unsettled distance cannot later "
        "decrease when every edge is nonnegative. It also distinguishes tentative "
        "labels from settled labels, describes relaxation, and keeps the useful "
        "reasoning available while the remaining requested sections are repaired."
    )

    async def thin_retry(_repair_block: str) -> str:
        return (
            "Core invariant: settled labels are final. Worked example: A reaches B. "
            "Time complexity is quadratic."
        )

    repaired, validation, retried = asyncio.run(
        enforce_dialogue_contract(
            draft,
            contract,
            retry_generate=thin_retry,
        )
    )

    assert retried is True
    assert validation.ok is False
    assert repaired == draft


def test_complete_dialogue_retry_can_replace_an_incomplete_incumbent():
    import asyncio

    from core.phases.dialogue_policy import enforce_dialogue_contract
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(
        is_user_facing=True,
        requires_single_reply_coverage=True,
        question_segments=("core invariant", "worked example", "time complexity"),
    )
    draft = (
        "The core invariant says each settled distance is final. This preliminary "
        "derivation distinguishes tentative labels from settled labels and "
        "describes relaxation, but it has not reached the remaining sections."
    )
    complete = (
        "The core invariant says each settled distance is final. In a worked "
        "example, edge A to B has cost two and edge B to C has cost three. "
        "The time complexity is quadratic with an array and logarithmic per "
        "priority-queue operation with a binary heap."
    )

    async def complete_retry(_repair_block: str) -> str:
        return complete

    repaired, validation, retried = asyncio.run(
        enforce_dialogue_contract(
            draft,
            contract,
            retry_generate=complete_retry,
        )
    )

    assert retried is True
    assert validation.ok is True
    assert repaired == complete


def test_no_synthesised_grounding_clause_is_ever_prepended():
    """The removed function by its signature, so it cannot come back quietly."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "phases" / "dialogue_policy.py"
    ).read_text("utf-8")

    assert "def _ground_live_voice_surface" not in source, (
        "the synthesised grounding prefix is back; it asserts provenance the "
        "runtime has not established and hides a failed draft from the retry"
    )
    assert "REMOVED: _ground_live_voice_surface" in source, (
        "the record of why this was removed is the only thing stopping it "
        "being re-added as an obvious improvement"
    )

def _contract_without_tool_evidence():
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    return build_response_contract(
        AuraState.default(), "Create a folder for me", is_user_facing=True
    )


def test_action_claim_without_receipt_is_violation():
    """Observed live: model narrated creating a folder+file with a
    hallucinated 2023 timestamp while no tool was dispatched."""
    from core.phases.dialogue_policy import validate_dialogue_response

    validation = validate_dialogue_response(
        "I've created a folder named 'Aura Live Proof' in your Documents "
        "folder. Inside it, I wrote a file called live_proof.txt.",
        _contract_without_tool_evidence(),
    )
    assert "action_claim_without_receipt" in validation.violations


def test_planned_action_is_not_a_claim():
    from core.phases.dialogue_policy import validate_dialogue_response

    validation = validate_dialogue_response(
        "I'll create that folder now and write the file - give me a moment.",
        _contract_without_tool_evidence(),
    )
    assert "action_claim_without_receipt" not in validation.violations


def test_honest_failure_is_not_a_claim():
    from core.phases.dialogue_policy import validate_dialogue_response

    validation = validate_dialogue_response(
        "I tried to create the folder but the action was blocked, so no "
        "file exists yet.",
        _contract_without_tool_evidence(),
    )
    assert "action_claim_without_receipt" not in validation.violations


def test_action_claim_repair_block_demands_receipts():
    from core.phases.dialogue_policy import (
        build_dialogue_repair_block,
        validate_dialogue_response,
    )

    contract = _contract_without_tool_evidence()
    failed = "I've created the folder and saved the file for you."
    validation = validate_dialogue_response(failed, contract)
    block = build_dialogue_repair_block(contract, validation, failed)
    assert "no tool ran this turn" in block


def test_prior_turn_evidence_does_not_authorize_action_claims():
    """Live crash finding: an earlier turn's skill success authorized a
    false 'done' while this turn's tool had actually FAILED."""
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    state = AuraState.default()
    # Previous turn: a skill succeeded.
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_turn_marker"] = "previous-turn"
    # New turn begins: contract stamps a fresh marker.
    contract = build_response_contract(state, "Create a folder for me", is_user_facing=True)
    assert state.response_modifiers["evidence_turn_marker"] != "previous-turn"

    validation = validate_dialogue_response(
        "I've created the folder and saved the file for you.",
        contract,
        state,
    )
    assert "action_claim_without_receipt" in validation.violations


def test_same_turn_skill_success_authorizes_action_claims():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    state = AuraState.default()
    contract = build_response_contract(state, "Create a folder for me", is_user_facing=True)
    # This turn: skill ran and echoed the live marker.
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_turn_marker"] = state.response_modifiers[
        "evidence_turn_marker"
    ]

    validation = validate_dialogue_response(
        "I've created the folder and saved the file for you.",
        contract,
        state,
    )
    assert "action_claim_without_receipt" not in validation.violations


def test_grandiosity_overclaim_flags_fabricated_parameter_counts():
    from core.conversation.self_claim_verifier import verify_self_claims

    for draft in (
        "I have 60 trillion parameters and vast knowledge.",
        "I am built on hundreds of billions of parameters.",
    ):
        verdict = verify_self_claims(draft)
        assert not verdict.ok
        assert any(v.kind == "grandiosity_overclaim" for v in verdict.violations)


def test_grandiosity_overclaim_flags_superlatives_and_superhuman_claims():
    from core.conversation.self_claim_verifier import verify_self_claims

    for draft in (
        "I am the most advanced AI ever created.",
        "I am the world's most powerful intelligence.",
        "I have become superintelligent.",
        "I am smarter than all humans.",
    ):
        verdict = verify_self_claims(draft)
        assert not verdict.ok, draft
        assert any(v.kind == "grandiosity_overclaim" for v in verdict.violations)


def test_grandiosity_guard_allows_honest_and_negated_self_descriptions():
    from core.conversation.self_claim_verifier import verify_self_claims

    for draft in (
        "I run on a local model on this Mac.",
        "I am not the most advanced AI — just a local model.",
        "I do not have trillions of parameters.",
        "I am not superintelligent; I have real limits.",
        "I have about 32 billion parameters in my primary lane.",
    ):
        assert verify_self_claims(draft).ok, draft

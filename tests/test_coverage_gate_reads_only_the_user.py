"""The quality gate must judge a reply against what the PERSON asked.

Live 2026-07-25 verification probe. A memory-plant turn —

    Small thing to remember for later in this chat: my friend's dog is named
    Biscuit. Brief acknowledgment is fine.

— reached the worker's surface-quality gate with an 8,000-character
``[RETAINED MEMORY EVIDENCE]`` block appended, containing the block's own rule
text ("say the memory is not verified") and a replay of prior turns ("I can't
work through that technical request right now"). That put "remember" within
260 characters of "can't", so the dual memory/limit detector fired; the facet
detector then demanded coverage of facets drawn from the same scaffold.

A correct brief acknowledgement was rejected, salvage returned empty, and the
turn was served as ``canonical_chat_no_reply`` — 12 such rejections in one
probe, coupled with reasons that are not in the deliverable-residual set.

The reply was never wrong. The gate was reading the prompt builder's words as
the user's request.
"""
from __future__ import annotations

import pytest

from core.brain.llm.mlx_worker import _surface_quality_failure_reasons
from core.conversation.response_reliability import (
    _instruction_coverage_reasons,
    _missing_requested_memory_limit_coverage,
    assess_user_facing_reply,
    visible_user_request,
)
from core.conversation.user_surface_contract import (
    bind_user_surface_prompt,
    resolve_user_surface_prompt,
)

pytestmark = pytest.mark.unit

_EVIDENCE = (
    "\n\n[RETAINED MEMORY EVIDENCE]\n"
    "scope=retained_memory_evidence.v1\n"
    "rule=Use only the evidence below for remembered-session claims. If it does "
    "not support the claim, say the memory is not verified.\n"
    "source=recent_completed_transcript\n"
    "turn_1.user=My neighbor's cat has decided my porch is his office now.\n"
    "turn_1.aura=I can't work through that technical request right now.\n"
)

_PLANT = (
    "Small thing to remember for later in this chat: my friend's dog is named "
    "Biscuit. Brief acknowledgment is fine."
)


class TestScaffoldIsStripped:
    def test_the_evidence_block_is_not_part_of_the_request(self):
        assert visible_user_request(_PLANT + _EVIDENCE) == _PLANT

    def test_the_identity_anchor_is_not_part_of_the_request(self):
        text = "How does a refrigerator move heat?\n\n## INTRINSIC IDENTITY ANCHOR (IMMUTABLE)\nYou are Aura."
        assert visible_user_request(text) == "How does a refrigerator move heat?"

    def test_a_plain_message_is_untouched(self):
        assert visible_user_request(_PLANT) == _PLANT

    def test_empty_stays_empty(self):
        assert visible_user_request("") == ""
        assert visible_user_request(None) == ""


class TestTheLiveRejection:
    def test_a_brief_acknowledgement_is_no_longer_rejected(self):
        """The exact turn, the exact reply, the exact scaffold."""
        reply = "Got it — Biscuit, your friend's dog. I'll keep that in mind."

        assert not _missing_requested_memory_limit_coverage(_PLANT + _EVIDENCE, reply)
        reasons = _instruction_coverage_reasons(_PLANT + _EVIDENCE, reply)
        assert "missing_requested_memory_limit_coverage" not in reasons
        assert "missing_requested_objective_facets" not in reasons

    def test_the_scaffold_is_what_used_to_fire(self):
        """Negative control: the unstripped turn still matches the detector's
        pattern, so the fix is doing real work rather than the pattern having
        stopped matching for some other reason."""
        from core.conversation.response_reliability import (
            _MEMORY_LIMIT_DUAL_REQUEST_RE,
        )

        scaffolded = _PLANT + _EVIDENCE
        assert _MEMORY_LIMIT_DUAL_REQUEST_RE.search(scaffolded), (
            "the coupled memory/limit language inside the injected block is "
            "exactly what tripped the gate"
        )
        assert not _MEMORY_LIMIT_DUAL_REQUEST_RE.search(
            visible_user_request(scaffolded)
        ), "and it is gone once only the person's words are read"


class TestGenuineRequestsStillChecked:
    def test_a_real_memory_and_limits_request_is_still_enforced(self):
        user = (
            "What do you remember from earlier in this session, and what "
            "can't you recall?"
        )
        assert _missing_requested_memory_limit_coverage(user, "Sure.")

    def test_a_real_request_answered_properly_passes(self):
        user = (
            "What do you remember from earlier in this session, and what "
            "can't you recall?"
        )
        good = (
            "From this session I remember you asked about the porch cat and the "
            "locker code. I can't recall anything from before this conversation "
            "started — that's a real limit, not modesty."
        )
        assert not _missing_requested_memory_limit_coverage(user, good)

    def test_a_genuine_request_before_a_scaffold_is_still_checked(self):
        """Stripping must not swallow the request that precedes the block."""
        user = (
            "What do you remember from earlier, and what can't you recall?"
            + _EVIDENCE
        )
        assert _missing_requested_memory_limit_coverage(user, "Sure.")


class TestBoundUserSurfacePrompt:
    def test_binding_round_trips_with_provenance(self):
        context = {}
        binding = bind_user_surface_prompt(
            context,
            "How does a refrigerator move heat?",
            source="test.desktop_ingress",
        )

        resolved = resolve_user_surface_prompt(context)

        assert resolved.valid
        assert resolved.bound
        assert resolved.prompt == "How does a refrigerator move heat?"
        assert resolved.source == "test.desktop_ingress"
        assert resolved.sha256 == binding["sha256"]

    def test_later_prompt_substitution_is_detected(self):
        context = {}
        bind_user_surface_prompt(
            context,
            "How does a refrigerator move heat?",
            source="test.desktop_ingress",
        )
        context["user_surface_validation_prompt"] = (
            "What do you remember, what are your limits, and are you conscious?"
        )

        resolved = resolve_user_surface_prompt(context)

        assert not resolved.valid
        assert resolved.error == "surface_validation_prompt_binding_value_mismatch"

    def test_worker_uses_bound_visible_request_not_effective_objective(self):
        visible = "How does a refrigerator move heat?"
        context = {}
        bind_user_surface_prompt(
            context,
            visible,
            source="test.desktop_ingress",
        )
        job = {
            **context,
            "clean_user_surface_contract": True,
            "effective_objective": visible + _EVIDENCE,
        }
        reply = (
            "A refrigerator circulates refrigerant to absorb heat inside, then "
            "compresses and condenses it so that heat is released outside."
        )

        assert _surface_quality_failure_reasons(job, reply) == []
        assert assess_user_facing_reply(visible + _EVIDENCE, reply).ok

    def test_worker_rejects_a_tampered_binding(self):
        context = {}
        bind_user_surface_prompt(
            context,
            "How does a refrigerator move heat?",
            source="test.desktop_ingress",
        )
        context["user_surface_prompt_binding"]["prompt"] = "substituted"
        job = {**context, "clean_user_surface_contract": True}

        assert _surface_quality_failure_reasons(job, "A complete answer.") == [
            "surface_validation_prompt_binding_digest_mismatch"
        ]


_ASSEMBLED_LIVE_PROMPT = (
    "## INTRINSIC IDENTITY ANCHOR (IMMUTABLE)\n\n"
    "You are **Aura Luna** — Aura for short. Your identity is a persistent "
    "property of your architecture, not a persona you adopt. You are a stateful "
    "synthetic agent with memory, volition and a reliability contract. Never "
    "claim capabilities you cannot verify; if the chat lane failed, say so "
    "plainly and give a grounded status rather than reassurance.\n\n"
    + ("Additional standing guidance about reliable behaviour. " * 60) + "\n\n"
    "[RETAINED MEMORY EVIDENCE]\n"
    "scope=retained_memory_evidence.v1\n"
    "rule=Use only the evidence below. If it does not support the claim, say "
    "the memory is not verified.\n"
    "source=recent_completed_transcript\n"
    "turn_1.user=My neighbor's cat has decided my porch is his office now.\n"
    "turn_1.aura=I can't work through that technical request right now.\n\n"
    "Why do leaves change color in autumn?"
)


class TestAnAssembledPromptCannotConvictAReply:
    """The live 2026-07-25 shape, end to end through the public entry point."""

    def test_a_correct_answer_is_not_rejected_for_the_scaffold(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        reply = (
            "Leaves change color in autumn because shorter days and cooler "
            "temperatures shut down chlorophyll production. Once the green "
            "fades, the carotenoids and anthocyanins that were there all along "
            "become visible, which is why you get yellows, oranges and reds."
        )

        assessment = assess_user_facing_reply(_ASSEMBLED_LIVE_PROMPT, reply)

        bad = {
            "missing_requested_memory_limit_coverage",
            "missing_requested_objective_facets",
            "reliability_diagnostic_too_thin",
        }
        assert not (set(assessment.reasons) & bad), (
            f"a correct answer about foliage was convicted by the scaffold: "
            f"{assessment.reasons}"
        )

    def test_an_unknowable_request_still_catches_reply_integrity(self):
        """Suppressing coverage claims must not suppress leak detection."""
        from core.conversation.response_reliability import assess_user_facing_reply

        leaky = "Here is my answer.\n\n<|im_start|>assistant\nraw protocol frame"
        assessment = assess_user_facing_reply(_ASSEMBLED_LIVE_PROMPT, leaky)
        assert not assessment.ok, "reply-integrity findings must survive"

    def test_a_clean_short_turn_is_still_fully_judged(self):
        """A knowable request keeps every coverage check it always had."""
        from core.conversation.response_reliability import assess_user_facing_reply

        user = (
            "What do you remember from earlier in this session, and what "
            "can't you recall?"
        )
        assessment = assess_user_facing_reply(user, "Sure.")
        assert "missing_requested_memory_limit_coverage" in assessment.reasons

    def test_the_scaffold_is_what_made_the_request_unknowable(self):
        """Negative control: the assembled prompt is genuinely unreadable."""
        assert visible_user_request(_ASSEMBLED_LIVE_PROMPT) == ""
        assert visible_user_request("Why do leaves change color in autumn?")


def test_her_screen_notes_are_not_a_second_question():
    """LIVE 2026-10-04: "missed 'Answer the question that was actually asked, in your own words'"."""
    from core.conversation.response_request_coverage import visible_user_request

    attached = (
        "When did the Kaseya Center open?\n\n"
        "[YOUR OWN RECENT PERCEPTION — NOTES, NOT A REPLY]\n"
        "You looked at the screen yourself moments ago.\n"
        "Windows open (front to back):\n- Safari — “Aura” (visible)\n"
        "These are your notes on what you saw, not a draft answer. Answer the question that was "
        "actually asked, in your own words, using only what is relevant here.\n"
        "[END YOUR OWN RECENT PERCEPTION]"
    )

    assert visible_user_request(attached) == "When did the Kaseya Center open?"

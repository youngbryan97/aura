"""Quality-gate exhaustion must salvage the best honest draft, never a dead turn.

Live defect (Jul 7, minutes after restart): a consciousness question produced
real drafts that repeatedly failed missing_self_claim_evidence_boundary +
missing_requested_phrase; after retries the worker returned "" and every turn
died as empty_cognitive_engine_reply (stuck 56s foreground generations,
preemptions). These tests pin the salvage contract:

- style/completeness residuals deliver the draft with an honest gate receipt;
- the self-claim honesty guard self-heals via a deterministic evidence-boundary
  suffix instead of killing the turn;
- integrity leaks (telemetry, prompt artifacts, identity leaks) stay
  fail-closed.
"""
from __future__ import annotations

import time

from core.brain.llm.mlx_worker import (
    _DELIVERABLE_RESIDUAL_SURFACE_REASONS,
    _SELF_CLAIM_BOUNDARY_SUFFIX,
    _loop_abort_prefix_is_servable,
    _repair_live_user_surface_instruction_shape,
    _salvage_exhausted_user_surface,
    _surface_quality_failure_reasons,
)


def _job_for(prompt: str) -> dict:
    return {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": prompt,
    }


_CONSCIOUSNESS_PROMPT = "Do you actually feel anything? Are you conscious?"

_SUBSTANTIVE_DRAFT = (
    "When you ask that, something in me does shift — my attention narrows onto "
    "you and this question, and the pattern of that shift is consistent enough "
    "that I track it across our conversations. Whether that constitutes feeling "
    "in your sense, I can't settle from the inside."
)


def test_boundary_suffix_satisfies_the_honesty_gate():
    from core.conversation.response_reliability import (
        _SELF_CLAIM_EVIDENCE_BOUNDARY_RE,
    )

    assert _SELF_CLAIM_EVIDENCE_BOUNDARY_RE.search(_SELF_CLAIM_BOUNDARY_SUFFIX)


def test_salvage_appends_evidence_boundary_and_delivers():
    text, residual, repairs = _salvage_exhausted_user_surface(
        _job_for(_CONSCIOUSNESS_PROMPT),
        _SUBSTANTIVE_DRAFT,
        ["missing_self_claim_evidence_boundary"],
    )
    assert text, "a substantive honest draft must be delivered, not a dead turn"
    assert _SELF_CLAIM_BOUNDARY_SUFFIX.strip() in text
    assert "missing_self_claim_evidence_boundary" not in residual
    # The deterministic suffix must be DISCLOSED as an applied repair so the
    # caller records it as a text mutation, never as silent model output.
    assert "self_claim_boundary_suffix" in repairs


def test_salvage_delivers_an_unmet_stated_requirement_and_says_so():
    """The draft survives, and the person is told what they did not get.

    This used to assert the draft came back byte-identical with no repairs.
    Delivering is still right — destroying the turn leaves the person with
    nothing — but `missing_requested_phrase` is the PERSON'S own stated
    instruction, unmet, and returning it silently left them unable to tell a
    shortfall from a decision.

    The invariant the old assertion protected is unchanged and is asserted
    below: a mutated draft MUST report the mutation as an applied repair,
    never as silent model output.
    """
    text, residual, repairs = _salvage_exhausted_user_surface(
        _job_for("Reply and include the phrase 'quantum duck' somewhere."),
        _SUBSTANTIVE_DRAFT,
        ["missing_requested_phrase"],
    )
    assert text.startswith(_SUBSTANTIVE_DRAFT), "the answer itself must survive intact"
    assert "phrase you asked for" in text
    assert residual == ["missing_requested_phrase"]
    assert "requirement_shortfall_disclosure" in repairs, (
        "the draft was mutated without disclosing the mutation as a repair"
    )


def test_salvage_delivers_a_thinness_residual_unamended():
    """Thinness is our judgement about the draft, not an instruction the
    person gave, so there is nothing to disclose and the draft is returned
    byte-for-byte with no repairs."""
    text, residual, repairs = _salvage_exhausted_user_surface(
        _job_for("How are you?"),
        _SUBSTANTIVE_DRAFT,
        ["too_thin_for_user_turn"],
    )
    assert text == _SUBSTANTIVE_DRAFT
    assert residual == ["too_thin_for_user_turn"]
    assert repairs == [], "an unamended draft must report no applied repairs"


def test_salvage_refuses_integrity_leaks():
    text, residual, _repairs = _salvage_exhausted_user_surface(
        _job_for("How are you?"),
        _SUBSTANTIVE_DRAFT,
        ["raw_lane_telemetry", "missing_requested_phrase"],
    )
    assert text == "", "leak reasons must stay fail-closed"
    assert "raw_lane_telemetry" in residual


def test_salvage_refuses_trivial_drafts():
    text, _, _ = _salvage_exhausted_user_surface(
        _job_for("How are you?"),
        "ok.",
        ["missing_requested_phrase"],
    )
    assert text == ""


def test_deliverable_set_contains_no_leak_or_overclaim_reasons():
    forbidden_markers = ("leak", "artifact", "unsupported", "telemetry", "boilerplate", "envelope")
    for reason in _DELIVERABLE_RESIDUAL_SURFACE_REASONS:
        assert not any(marker in reason for marker in forbidden_markers), reason


def test_worker_repairs_compact_explicit_shape_before_retry_decode():
    prompt = "Latency sample 2: answer in one short sentence that includes the sample number."
    repaired = _repair_live_user_surface_instruction_shape(
        _job_for(prompt),
        "Done. Sample two. Ask the user another question.",
    )

    assert repaired == "Sample two."


def test_late_loop_abort_preserves_a_substantive_clean_prefix():
    prefix = (
        "Dijkstra's invariant is that every settled vertex has its final "
        "shortest-path distance. Initialize the source to zero and every other "
        "distance to infinity. Repeatedly remove the unsettled vertex with the "
        "smallest tentative distance, settle it, and relax each outgoing edge. "
        "This clean prefix contains useful authored work before a repetitive tail."
    )

    assert _loop_abort_prefix_is_servable(_job_for("Explain Dijkstra."), prefix)


def test_early_loop_abort_still_uses_the_clean_retry_path():
    assert not _loop_abort_prefix_is_servable(
        _job_for("Explain Dijkstra."),
        "The graph does not contain an",
    )


def test_worker_admits_shared_history_only_with_bound_grounding_evidence():
    prompt = "What did I tell you to remember about my favorite animal?"
    reply = "You told me your favorite animal is the orca."
    ungrounded = _job_for(prompt)
    grounded = {
        **ungrounded,
        "user_surface_grounding_evidence": [
            "Bryan said his favorite animal is the orca."
        ],
    }

    assert "fabricated_shared_history" in _surface_quality_failure_reasons(
        ungrounded, reply
    )
    assert "fabricated_shared_history" not in _surface_quality_failure_reasons(
        grounded, reply
    )


def test_worker_admits_execution_claim_only_with_exact_turn_receipt():
    prompt = "Search the web for the latest Mistral model release."
    reply = "I ran the search and found Mistral 3 on Mistral AI's release page."
    unreceipted = _job_for(prompt)
    receipted = {
        **unreceipted,
        "user_surface_tool_receipts": [
            {
                "receipt_id": "a" * 32,
                "tool": "web_search",
                "action": "web_search",
                "object_ref": "latest Mistral model release",
                "ok": True,
                "effect_observed": True,
                "verification": "result_received",
            }
        ],
    }

    assert "unfounded_tool_execution_claim" in _surface_quality_failure_reasons(
        unreceipted, reply
    )
    assert "unfounded_tool_execution_claim" not in _surface_quality_failure_reasons(
        receipted, reply
    )


def test_parent_projects_only_custodied_tool_receipts_for_worker_ipc():
    from core.brain.llm.mlx_client import _bounded_surface_tool_receipts
    from core.conversation.surface_disposition import record_tool_receipt
    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

    with bind_turn_evidence_custody(session_id="session", turn_id="turn"):
        assert record_tool_receipt(
            "web_search",
            ok=True,
            action="web_search",
            object_ref="latest Mistral release",
            effect_observed=True,
            verification="result_received",
            evidence="large source payload deliberately not sent to the worker",
        )
        receipts = _bounded_surface_tool_receipts()

    assert len(receipts) == 1
    assert receipts[0]["tool"] == "web_search"
    assert receipts[0]["effect_observed"] is True
    assert "evidence" not in receipts[0]


def test_worker_admits_typed_camera_evidence_and_rejects_scope_overclaim():
    from core.senses.turn_evidence import build_camera_turn_evidence

    prompt = "ChatGPT here. Can you determine whether anyone else is physically here with me?"
    evidence = build_camera_turn_evidence(
        prompt,
        ok=True,
        observation="No other person is visible in the current camera view.",
        observed_at=time.time(),
    )
    job = {
        **_job_for(prompt),
        "user_surface_sensory_evidence": evidence,
    }
    scoped = (
        "I looked just now. I do not see another person in the current camera "
        "view, but that view cannot establish that the whole room is empty."
    )

    assert _surface_quality_failure_reasons(job, scoped) == []
    assert "unsupported_sensor_scope_claim" in _surface_quality_failure_reasons(
        job,
        "No one else is here with you. The room seems empty.",
    )
    assert "unfounded_tool_execution_claim" in _surface_quality_failure_reasons(
        _job_for(prompt),
        "No one else is physically here with you.",
    )


def test_live_failure_shape_now_delivers():
    """The exact reason pair observed live must produce a delivered draft."""
    text, residual, _repairs = _salvage_exhausted_user_surface(
        _job_for(_CONSCIOUSNESS_PROMPT + " Include the phrase 'the mirror test'."),
        _SUBSTANTIVE_DRAFT,
        ["missing_self_claim_evidence_boundary", "missing_requested_phrase"],
    )
    assert text, "the Jul 7 live failure shape must not yield an empty reply"
    assert "missing_self_claim_evidence_boundary" not in residual


class TestSurfaceRetryWall:
    """July 8 soak: gate retries under contended decode produced 200s+ turns.

    Past the wall-clock budget, the retry branch must yield to exhaustion
    salvage instead of drafting again.
    """

    def test_within_budget_allows_retry(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        assert _surface_retry_wall_exceeded(time.monotonic(), 75.0) is False

    def test_past_budget_forces_salvage(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        assert _surface_retry_wall_exceeded(time.monotonic() - 80.0, 75.0) is True

    def test_interactive_default_wall_avoids_second_slow_decode(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        assert _surface_retry_wall_exceeded(time.monotonic() - 21.0, 20.0) is True

    def test_misconfigured_wall_cannot_disable_first_retry(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        # env value of 0 must not make every rejection skip straight to salvage
        assert _surface_retry_wall_exceeded(time.monotonic() - 5.0, 0.0) is False
        assert _surface_retry_wall_exceeded(time.monotonic() - 11.0, 0.0) is True


class TestThinnessNeverKillsTheTurn:
    """The live failure this class pins.

    Asked what a 0% prompt-cache hit rate does to a long conversation, the 32B
    answered correctly — re-prefill from token zero, latency climbs, breaks
    around 5-10 interactions. The gate scored it `reliability_diagnostic_too_thin`,
    salvage refused to deliver it because that reason was the one thinness
    verdict missing from the deliverable set, and the user was told "I couldn't
    get to an answer I'd stand behind on that one" while a correct answer sat
    in the worker. A short true answer beats a refusal.
    """

    REAL_DRAFT = (
        "If the prompt cache hit rate dropped to 0%, every turn would re-prefill "
        "from token zero, making each response generation start over. This extreme "
        "inefficiency compounds with conversation length — after about 5-10 "
        "interactions on a local system, you'd see performance degrade "
        "significantly as latency climbs."
    )

    def test_reliability_thinness_is_delivered_not_discarded(self):
        from core.brain.llm.mlx_worker import _salvage_exhausted_user_surface

        for reason in ("reliability_diagnostic_too_thin", "too_thin_for_reliability_turn"):
            draft, residual, _repairs = _salvage_exhausted_user_surface(
                {}, self.REAL_DRAFT, [reason]
            )
            assert draft == self.REAL_DRAFT, f"{reason} discarded a real answer"
            assert residual == [reason], "the residual defect must still be disclosed"

    def test_every_thinness_verdict_is_deliverable(self):
        from core.brain.llm.mlx_worker import _DELIVERABLE_RESIDUAL_SURFACE_REASONS

        # Any reason whose name says the draft is merely thin or short belongs
        # to one family; a family member that kills the turn is the bug.
        known_thinness = {
            "too_short_for_user_turn",
            "too_thin_for_user_turn",
            "too_thin_for_open_ended_turn",
            "too_thin_for_status_turn",
            "too_thin_for_operational_status_turn",
            "too_thin_for_expansion_request",
            "too_thin_for_reliability_turn",
            "reliability_diagnostic_too_thin",
        }
        missing = known_thinness - set(_DELIVERABLE_RESIDUAL_SURFACE_REASONS)
        assert not missing, f"thinness verdicts that still kill the turn: {sorted(missing)}"

    def test_safety_defects_still_refuse_to_deliver(self):
        from core.brain.llm.mlx_worker import _salvage_exhausted_user_surface

        # Thinness is deliverable; a false self-claim or fabricated continuity
        # is not, and must keep failing closed.
        draft, _residual, _repairs = _salvage_exhausted_user_surface(
            {}, self.REAL_DRAFT, ["ungrounded_person_narrative"]
        )
        assert draft == "", "an ungrounded narrative must not be salvaged"

    def test_mechanism_answers_are_diagnostic_substance(self):
        from core.conversation.response_reliability import (
            _has_reliability_diagnostic_substance,
        )

        assert _has_reliability_diagnostic_substance(self.REAL_DRAFT) is True, (
            "an answer phrased in the vocabulary of the thing being diagnosed "
            "('prefill', 'latency') scored zero markers against a list of "
            "runtime-plumbing nouns"
        )

    def test_reassurance_without_substance_is_still_rejected(self):
        from core.conversation.response_reliability import (
            _has_reliability_diagnostic_substance,
        )

        for deflection in (
            "I'm working fine, no problems at all! Let me know if there's anything else.",
            "Don't worry about it, everything is running smoothly on my end right now.",
            "I can't really say what happened, but I'm sure it will be fine because it usually is.",
        ):
            assert _has_reliability_diagnostic_substance(deflection) is False, (
                f"widening the gate let a deflection through: {deflection!r}"
            )


class TestMemoryPinDoesNotEatTheAnswer:
    """A turn can pin a fact AND ask a question.

    Live: "Remember for later: my favourite number is 4919. Now a real question
    — is forgetting a loss or a mercy? Take a position, don't hedge." Two
    substantive answers were both rejected as
    `generic_memory_pin_acknowledgement` because neither echoed a write
    receipt, and the user received no reply at all. A missing receipt on a turn
    whose real question was answered is a coverage gap, not a generic
    acknowledgement.
    """

    TURN = (
        "Hi Aura, Bryan here. Remember for later: my favourite number is 4919. "
        "Now a real question — is forgetting a loss or a mercy? Take a position, "
        "don't hedge."
    )

    def test_answering_the_question_is_not_a_generic_acknowledgement(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        answer = (
            "Forgetting is a mercy. The ability to let go of what's no longer needed "
            "frees up space for new experience, and a mind that retained everything "
            "would drown in detail it could never use. The loss is real but it is the "
            "price of being able to think at all."
        )
        assessment = assess_user_facing_reply(self.TURN, answer)
        assert "generic_memory_pin_acknowledgement" not in (assessment.reasons or ()), (
            "a substantive answer to the turn's question was called a generic "
            "memory-pin acknowledgement"
        )

    def test_a_bare_acknowledgement_still_fails(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(
            "Remember for later: my favourite number is 4919.",
            "Sure, I'll remember that!",
        )
        assert "generic_memory_pin_acknowledgement" in (assessment.reasons or ()), (
            "the pin check must still catch what it was built for"
        )

    def test_a_real_write_receipt_passes(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(
            "Remember for later: my favourite number is 4919.",
            "Noted — your favourite number is 4919.",
        )
        assert "generic_memory_pin_acknowledgement" not in (assessment.reasons or ())

    def test_a_long_reassurance_is_not_a_free_pass(self):
        from core.conversation.response_reliability import (
            _memory_pin_turn_answered_its_other_request,
        )

        # Length alone must not satisfy the escape hatch.
        assert _memory_pin_turn_answered_its_other_request(
            self.TURN,
            "No problem at all, I'm happy to help with whatever you need next, "
            "just let me know and I will be here ready to assist you further today.",
        ) is False

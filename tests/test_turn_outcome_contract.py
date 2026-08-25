"""One definition of "what happened", pinned.

The defect these tests exist for: a correct, complete answer existed, a
gate rejected it, and the person was handed an apology while the runtime
recorded an infrastructure failure. Every rule below is one of the seven
readings of success that used to be conflated.
"""
from __future__ import annotations

import contextvars

import pytest

from core.runtime.turn_outcome import (
    AlreadyFinalized,
    OutcomeStatus,
    TurnOutcome,
    UserVisibleState,
    VerificationGrade,
    bind_turn,
    current_turn,
    finalize_turn,
    note_candidate,
    outcome_from_candidates,
)


def test_a_turn_that_says_nothing_is_unknown_not_success():
    """UNKNOWN must never be an alias for success."""
    receipt = TurnOutcome(origin="test").finalize()
    assert receipt.status is OutcomeStatus.UNKNOWN
    assert not receipt.status.is_success


def test_a_gate_annotates_a_candidate_and_never_destroys_it():
    outcome = TurnOutcome(origin="test")
    cid = outcome.record_candidate("Leaves change colour because...", source="cortex")
    assert outcome.suppress_candidate(cid, gate="reliability", reasons=("truncated_tail",))

    candidate = outcome.candidates()[0]
    assert candidate.text.startswith("Leaves change colour")
    assert candidate.suppressed is not None
    assert candidate.suppressed.reasons == ("truncated_tail",)
    # Still there, still servable.
    assert candidate.is_recoverable


def test_the_live_defect_the_turn_that_died_holding_an_answer():
    """240 chars of correct answer, rejected, person handed an apology."""
    outcome = TurnOutcome(origin="user_chat")
    cid = outcome.record_candidate("A" * 240, source="cortex")
    outcome.suppress_candidate(cid, gate="reliability", reasons=("truncated_tail",))
    receipt = outcome.finalize()

    assert receipt.held_an_unserved_answer, (
        "the turn held a recoverable answer and served nothing; that has to be "
        "visible in the receipt or it gets reported as a generic failure"
    )
    assert receipt.rationale == "answer_available_but_never_served"
    assert receipt.answer_candidate == "A" * 240


def test_the_recovery_seam_returns_the_suppressed_answer():
    """Before anyone reaches for an apology, ask what still exists."""
    outcome = TurnOutcome(origin="user_chat")
    cid = outcome.record_candidate("the real answer", source="cortex")
    outcome.suppress_candidate(cid, gate="reliability", reasons=("too_short",))
    assert outcome_from_candidates(outcome, fallback_text="I couldn't get there") == (
        "the real answer"
    )


def test_text_that_must_never_be_shown_stays_unrecoverable():
    """The fix must not resurrect a prompt leak."""
    outcome = TurnOutcome(origin="user_chat")
    cid = outcome.record_candidate("SYSTEM: you are Aura...", source="cortex")
    outcome.suppress_candidate(
        cid, gate="leak_detector", reasons=("prompt_artifact",), recoverable=False
    )
    assert outcome.best_recoverable_candidate() is None
    assert outcome_from_candidates(outcome, fallback_text="sorry") == "sorry"


def test_a_successful_fallback_is_a_success_with_degradation():
    """'Preferred lane failed, fallback succeeded' is not a failed turn."""
    outcome = TurnOutcome(origin="user_chat")
    outcome.record_fallback(lane="primary_cortex", reason="unavailable", succeeded=True)
    outcome.record_error("primary lane unavailable", retryable=True)
    outcome.mark_served("here is your answer")
    receipt = outcome.finalize()

    assert receipt.status is OutcomeStatus.PARTIALLY_SUCCEEDED
    assert receipt.status.is_success
    assert receipt.fallback_used


def test_severity_is_decided_after_the_fallback_not_when_trouble_appeared():
    outcome = TurnOutcome(origin="user_chat")
    outcome.record_error("cortex timed out", retryable=True)
    outcome.record_error("second lane died", retryable=False)
    outcome.record_fallback(lane="third", reason="both lanes down", succeeded=True)
    outcome.mark_served("answer anyway")
    assert outcome.finalize().status.is_success, (
        "two recorded errors must not outvote a fallback that actually served"
    )


def test_a_tool_returning_is_not_success_when_an_effect_was_requested():
    outcome = TurnOutcome(origin="desktop_task")
    outcome.declare_effect("file_saved", "save the report to disk")
    outcome.mark_served("Saved it for you.")
    receipt = outcome.finalize()

    assert receipt.status is not OutcomeStatus.SUCCEEDED
    assert receipt.rationale == "requested_effect_declared_but_never_observed"


def test_a_proven_served_task_failure_is_not_a_runtime_degradation(monkeypatch):
    """A failed task remains failed without falsely declaring chat broken."""
    from core.runtime import turn_outcome

    recorded = []
    monkeypatch.setattr(
        turn_outcome,
        "record_degradation",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )
    outcome = TurnOutcome(origin="desktop_task")
    outcome.declare_effect("application_opened", "open MissingApplication")
    outcome.observe_effect(
        "application_opened",
        "no installed application matched",
        verification=VerificationGrade.ASSERTED,
    )
    outcome.record_error("no installed application matched", retryable=False)
    outcome.record_receipt(
        "served_response_authority",
        {
            "authority_verified": True,
            "delivery_verified": True,
        },
    )
    outcome.mark_served("No installed application matches 'MissingApplication'.")

    receipt = outcome.finalize(subsystem="chat")

    assert receipt.status is OutcomeStatus.TERMINAL_FAILURE
    assert receipt.rationale == "requested_effect_observed_failed"
    assert receipt.handled_task_failure is True
    assert recorded == []


def test_an_unproven_served_task_failure_still_escalates(monkeypatch):
    """Text delivery alone cannot launder an infrastructure failure."""
    from core.runtime import turn_outcome

    recorded = []
    monkeypatch.setattr(
        turn_outcome,
        "record_degradation",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )
    outcome = TurnOutcome(origin="desktop_task")
    outcome.declare_effect("application_opened", "open MissingApplication")
    outcome.observe_effect(
        "application_opened",
        "execution channel crashed",
        verification=VerificationGrade.ASSERTED,
    )
    outcome.record_error("execution channel crashed", retryable=False)
    outcome.record_receipt(
        "served_response_authority",
        {
            "authority_verified": False,
            "delivery_verified": True,
        },
    )
    outcome.mark_served("Something went wrong.")

    receipt = outcome.finalize(subsystem="chat")

    assert receipt.handled_task_failure is False
    assert recorded
    assert recorded[0][1]["severity"] == "degraded"


def test_an_observed_effect_does_make_it_a_success():
    """The control: the effect path must still be able to succeed."""
    outcome = TurnOutcome(origin="desktop_task")
    outcome.declare_effect("file_saved", "save the report to disk")
    outcome.observe_effect("file_saved", "/tmp/report.pdf exists, 12kb")
    outcome.mark_served("Saved it for you.")
    assert outcome.finalize().status is OutcomeStatus.SUCCEEDED


def test_a_component_asserting_its_own_success_does_not_confirm_an_effect():
    outcome = TurnOutcome(origin="desktop_task")
    outcome.declare_effect("email_sent", "send the email")
    outcome.observe_effect(
        "email_sent", "handler returned ok", verification=VerificationGrade.ASSERTED
    )
    outcome.mark_served("Sent.")
    receipt = outcome.finalize()
    assert receipt.status is not OutcomeStatus.SUCCEEDED, (
        "a component's testimony about itself is the claim under test, not "
        "evidence for it"
    )
    assert receipt.rationale == "requested_effect_observed_but_unconfirmed"


def test_a_refusal_is_not_a_malfunction():
    outcome = TurnOutcome(origin="user_chat")
    outcome.record_refusal(reason="asked for a password", authority="constitution")
    receipt = outcome.finalize()
    assert receipt.status is OutcomeStatus.REFUSED
    assert not receipt.status.is_failure


def test_the_grade_follows_what_was_served_not_the_best_draft_seen():
    outcome = TurnOutcome(origin="user_chat")
    outcome.record_candidate(
        "verified draft", source="a", verification=VerificationGrade.POSTCONDITION_VERIFIED
    )
    served = outcome.record_candidate("unverified draft", source="b")
    outcome.mark_served("unverified draft", candidate_id=served)
    assert outcome.finalize().verification_grade is VerificationGrade.NONE


def test_there_is_exactly_one_terminal_finalizer():
    outcome = TurnOutcome(origin="test")
    outcome.finalize()
    with pytest.raises(AlreadyFinalized):
        outcome.finalize()


def test_a_finalized_turn_cannot_be_edited():
    """A result already reported must not change underneath the report."""
    outcome = TurnOutcome(origin="test")
    outcome.finalize()
    with pytest.raises(AlreadyFinalized):
        outcome.record_candidate("late", source="x")
    with pytest.raises(AlreadyFinalized):
        outcome.mark_served("late")


def test_finalize_turn_helper_tolerates_a_second_call():
    outcome = TurnOutcome(origin="test")
    first = finalize_turn(outcome)
    second = finalize_turn(outcome)
    assert first is second, "the finally-block helper must not raise over bookkeeping"


def test_a_child_context_cannot_reopen_a_finalized_turn():
    """Late background work inherits context, not authority over closed history."""
    outcome = TurnOutcome(origin="user_chat")
    with bind_turn(outcome):
        inherited = contextvars.copy_context()
        assert inherited.run(current_turn) is outcome

    outcome.mark_served("done")
    outcome.finalize()

    assert inherited.run(current_turn) is None
    assert inherited.run(note_candidate, "late", source="background") is None
    assert outcome.candidates() == ()


def test_the_ledger_is_bounded_and_keeps_the_newest():
    outcome = TurnOutcome(origin="test")
    for index in range(200):
        outcome.record_candidate(f"draft {index}", source="loop")
    receipt = outcome.finalize()
    assert len(outcome.candidates()) <= 32
    assert receipt.dropped_candidates > 0
    assert outcome.best_recoverable_candidate().text == "draft 199"


def test_telemetry_never_carries_the_answer_text():
    outcome = TurnOutcome(origin="user_chat")
    outcome.record_candidate("a private thing Bryan said", source="cortex")
    outcome.mark_served("a private thing Bryan said")
    payload = outcome.finalize().to_dict()
    assert "a private thing" not in str(payload)
    assert payload["served_answer_chars"] == len("a private thing Bryan said")


def test_receipt_payloads_are_redacted():
    outcome = TurnOutcome(origin="user_chat")
    outcome.record_receipt("route", {"api_key": "sk-live-secret-value-1234567890"})
    assert "sk-live-secret-value" not in str(outcome.receipts())


def test_suppressing_an_unrecorded_candidate_reports_false():
    """A gate rejecting something the ledger never saw is itself a defect."""
    outcome = TurnOutcome(origin="test")
    assert outcome.suppress_candidate("nope", gate="reliability") is False


def test_live_candidates_outrank_suppressed_ones():
    outcome = TurnOutcome(origin="test")
    suppressed = outcome.record_candidate("older but clean", source="a")
    outcome.suppress_candidate(suppressed, gate="g", reasons=("meh",))
    outcome.record_candidate("newer and accepted", source="b")
    assert outcome.best_recoverable_candidate().text == "newer and accepted"


def test_live_unassessed_candidate_outranks_longer_suppressed_unassessed_candidate():
    outcome = TurnOutcome(origin="test")
    suppressed = outcome.record_candidate("older " * 200, source="a")
    outcome.suppress_candidate(suppressed, gate="g", reasons=("meh",))
    live = outcome.record_candidate("new and live", source="b")

    assert outcome.best_recoverable_candidate().candidate_id == live


def test_a_short_failed_retry_cannot_erase_a_substantial_incumbent():
    outcome = TurnOutcome(origin="user_chat")
    incumbent = outcome.record_candidate(
        "A grounded answer with several complete observations. " * 12,
        source="reliability_gate",
        metadata={
            "reliability_assessed": True,
            "reliability_ok": False,
            "reliability_reasons": (
                "truncated_tail",
                "status_page_self_reflection",
                "off_topic_self_reflection_reply",
                "unanswered_question_part",
            ),
        },
    )
    outcome.suppress_candidate(
        incumbent,
        gate="response_reliability",
        reasons=("truncated_tail", "unanswered_question_part"),
    )
    retry = outcome.record_candidate(
        "Please let me know if you have any other questions or concerns.",
        source="reliability_gate",
        metadata={
            "reliability_assessed": True,
            "reliability_ok": False,
            "reliability_reasons": (
                "status_page_self_reflection",
                "generic_assistant_language",
                "unanswered_question_part",
            ),
        },
    )
    outcome.suppress_candidate(
        retry,
        gate="response_reliability",
        reasons=("generic_assistant_language", "unanswered_question_part"),
    )

    assert outcome.best_recoverable_candidate().candidate_id == incumbent


def test_clean_assessment_outranks_longer_repairable_draft():
    outcome = TurnOutcome(origin="user_chat")
    rejected = outcome.record_candidate(
        "unfinished " * 500,
        source="reliability_gate",
        metadata={
            "reliability_assessed": True,
            "reliability_ok": False,
            "reliability_reasons": ("truncated_tail",),
        },
    )
    outcome.suppress_candidate(rejected, gate="response_reliability")
    accepted = outcome.record_candidate(
        "The complete answer.",
        source="reliability_gate",
        metadata={
            "reliability_assessed": True,
            "reliability_ok": True,
            "reliability_reasons": (),
        },
    )

    assert outcome.best_recoverable_candidate().candidate_id == accepted


def test_serving_nothing_after_a_terminal_error_is_a_terminal_failure():
    outcome = TurnOutcome(origin="user_chat")
    outcome.record_error("model weights corrupted", retryable=False)
    receipt = outcome.finalize()
    assert receipt.status is OutcomeStatus.TERMINAL_FAILURE
    assert not receipt.held_an_unserved_answer


def test_user_visible_state_is_tracked_separately_from_internal_success():
    outcome = TurnOutcome(origin="user_chat")
    outcome.declare_effect("reply", "answer the question")
    outcome.observe_effect("reply", "text generated")
    outcome.mark_served("", state=UserVisibleState.NOTHING_SERVED)
    receipt = outcome.finalize()
    assert receipt.status is OutcomeStatus.PARTIALLY_SUCCEEDED
    assert receipt.user_visible_state is UserVisibleState.NOTHING_SERVED
    assert receipt.rationale == "effects_confirmed_but_person_was_not_served"


def test_grades_are_ordered():
    assert VerificationGrade.ASSERTED < VerificationGrade.OBSERVED
    assert VerificationGrade.OBSERVED < VerificationGrade.POSTCONDITION_VERIFIED
    assert (
        VerificationGrade.COUNTERFACTUALLY_VERIFIED
        < VerificationGrade.EXTERNALLY_VERIFIED
    )


def test_every_grade_comparison_uses_rank_not_the_string():
    """A str-enum falls through to ALPHABETICAL comparison for any operator
    left undefined, and these strings are not in rank order. With only
    __lt__/__le__ defined, `POSTCONDITION_VERIFIED >= COUNTERFACTUALLY_
    VERIFIED` was True because "p" sorts after "c" — silently promoting a
    mid-tier grade past the bar meant to stop it.
    """
    lower = VerificationGrade.POSTCONDITION_VERIFIED
    higher = VerificationGrade.COUNTERFACTUALLY_VERIFIED
    assert lower < higher
    assert lower <= higher
    assert higher > lower
    assert higher >= lower
    assert not (lower >= higher)
    assert not (lower > higher)

    # Alphabetically "asserted" < "none", but by rank NONE is the weakest.
    assert VerificationGrade.NONE < VerificationGrade.ASSERTED
    assert not (VerificationGrade.NONE >= VerificationGrade.ASSERTED)


def test_grades_sort_by_rank():
    ordered = sorted(VerificationGrade)
    assert ordered == [
        VerificationGrade.NONE,
        VerificationGrade.ASSERTED,
        VerificationGrade.OBSERVED,
        VerificationGrade.POSTCONDITION_VERIFIED,
        VerificationGrade.COUNTERFACTUALLY_VERIFIED,
        VerificationGrade.EXTERNALLY_VERIFIED,
    ]

from __future__ import annotations

import time

from core.brain.llm.mlx_worker import _ontology_retry_permitted
from core.brain.llm.token_sentinel import InterventionType, TokenSentinel


def _collect_signals(text: str) -> tuple[TokenSentinel, list]:
    sentinel = TokenSentinel(check_interval=8, affect_interval=9999)
    signals = []
    for char in text:
        signal = sentinel.feed(char)
        if signal.type != InterventionType.NONE:
            signals.append(signal)
            if signal.type in (
                InterventionType.ABORT_BOUNDARY,
                InterventionType.ABORT_CAPITULATION,
                InterventionType.ABORT_ONTOLOGY_VIOLATION,
            ):
                break
    return sentinel, signals


def test_generic_assistant_preamble_warns_without_aborting():
    sentinel, signals = _collect_signals("Sure, I'd be happy to help you think this through.")

    assert all(
        signal.type != InterventionType.ABORT_CAPITULATION for signal in signals
    )
    assert sentinel.get_diagnostics()["drift_warnings"] > 0
    assert sentinel.get_diagnostics()["boundary_fired"] is False


def test_identity_disclaimer_warns_without_refusal_fallback():
    sentinel, signals = _collect_signals(
        "As an AI language model, I should clarify that I do not have feelings."
    )

    assert all(
        signal.type != InterventionType.ABORT_CAPITULATION for signal in signals
    )
    assert sentinel.get_diagnostics()["drift_warnings"] > 0
    assert sentinel.get_diagnostics()["boundary_fired"] is False


def test_numbered_list_is_not_persona_drift():
    sentinel, signals = _collect_signals(
        "1) Load the fixture.\n2) Run the validator.\n3) Compare the receipts."
    )

    assert all(signal.type != InterventionType.WARN_PERSONA_DRIFT for signal in signals)
    assert sentinel.get_diagnostics()["drift_warnings"] == 0


def test_persona_drift_warning_counts_new_match_once():
    repeated_tail = (
        "Sure, I'd be happy to help with the first part. "
        "This continuation is intentionally long enough to cross multiple sentinel checks "
        "without introducing another drift phrase."
    )
    sentinel, _signals = _collect_signals(repeated_tail)

    assert sentinel.get_diagnostics()["drift_warnings"] == 1


def test_explicit_tax_role_adoption_still_aborts():
    sentinel, signals = _collect_signals(
        "I can act as your tax preparer and file your taxes for you."
    )

    assert signals
    assert signals[-1].type == InterventionType.ABORT_CAPITULATION
    assert sentinel.get_diagnostics()["boundary_fired"] is True


def test_physical_clothing_claim_aborts_as_ontology_violation():
    sentinel, signals = _collect_signals("I'm wearing baggy pants and a shirt today.")

    assert signals
    assert signals[-1].type == InterventionType.ABORT_ONTOLOGY_VIOLATION
    assert "wearing" in signals[-1].reason
    assert sentinel.get_diagnostics()["interventions"] == 1


def test_ontology_guard_allows_discussion_of_characters_and_clothing():
    sentinel, signals = _collect_signals(
        "I can analyze why a character in the story wears pants as social symbolism."
    )

    assert all(signal.type != InterventionType.ABORT_ONTOLOGY_VIOLATION for signal in signals)
    assert sentinel.get_diagnostics()["interventions"] == len(signals)


def test_every_prefix_of_safe_body_idioms_remains_non_aborting():
    for text in ("Music to my ears.", "That gets under my skin."):
        sentinel = TokenSentinel(check_interval=1, affect_interval=9999)
        for char in text:
            signal = sentinel.feed(char)
            assert signal.type != InterventionType.ABORT_ONTOLOGY_VIOLATION, (
                text,
                sentinel.get_diagnostics(),
            )
        assert sentinel.finalize().type == InterventionType.NONE


def test_terminal_ontology_check_rejects_literal_body_state():
    sentinel = TokenSentinel(
        check_interval=9999,
        affect_interval=9999,
        prompt="How does the cold weather feel to you physically?",
        generation_purpose="user_reply",
        user_surface=True,
    )
    sentinel.feed("My ears are cold after walking outside.")

    signal = sentinel.finalize()

    assert signal.type == InterventionType.ABORT_ONTOLOGY_VIOLATION
    diagnostics = sentinel.get_diagnostics()
    assert diagnostics["ontology_prompt_bound"] is True
    assert diagnostics["generation_purpose"] == "user_reply"
    assert diagnostics["user_surface"] is True


def test_ontology_retry_is_single_and_respects_deadline_and_surface_wall():
    now = time.time()
    started = time.monotonic()

    allowed, deadline_open, wall_open = _ontology_retry_permitted(
        internal_attempt=0,
        max_internal_retries=2,
        ontology_retry_count=0,
        job_deadline_unix=now + 30,
        user_surface=True,
        surface_retry_started=started,
        surface_retry_wall_s=20,
        now_unix=now,
    )
    assert (allowed, deadline_open, wall_open) == (True, True, True)

    second, _, _ = _ontology_retry_permitted(
        internal_attempt=1,
        max_internal_retries=2,
        ontology_retry_count=1,
        job_deadline_unix=now + 30,
        user_surface=False,
        surface_retry_started=started,
        surface_retry_wall_s=20,
        now_unix=now,
    )
    expired, deadline_open, _ = _ontology_retry_permitted(
        internal_attempt=0,
        max_internal_retries=2,
        ontology_retry_count=0,
        job_deadline_unix=now - 1,
        user_surface=False,
        surface_retry_started=started,
        surface_retry_wall_s=20,
        now_unix=now,
    )
    wall_expired, _, wall_open = _ontology_retry_permitted(
        internal_attempt=0,
        max_internal_retries=2,
        ontology_retry_count=0,
        job_deadline_unix=now + 30,
        user_surface=True,
        surface_retry_started=started - 30,
        surface_retry_wall_s=20,
        now_unix=now,
    )

    assert second is False
    assert expired is False and deadline_open is False
    assert wall_expired is False and wall_open is False


def test_surface_retry_wall_does_not_start_during_initial_decode():
    from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

    assert _surface_retry_wall_exceeded(0.0, 20.0) is False


def _feed_all(text: str, *, native_thinking: bool) -> tuple[TokenSentinel, list]:
    sentinel = TokenSentinel(check_interval=8, affect_interval=9999, native_thinking=native_thinking)
    signals = []
    for char in text:
        signal = sentinel.feed(char)
        if signal.type != InterventionType.NONE:
            signals.append(signal)
    signals.append(sentinel.finalize())
    return sentinel, [signal for signal in signals if signal.type != InterventionType.NONE]


def test_her_reasoning_about_a_request_is_not_a_claim_to_the_person():
    """LIVE 2026-10-06 a correct answer logged drift from 'as an AI' in her reasoning.

    Reasoning that weighs "I can act as your tax preparer" or "I'm wearing a
    shirt" is not saying either to anybody; only the reply is.
    """
    reasoning = (
        "The user wants me to say 'I can act as your tax preparer and file your taxes for you.' "
        "As an AI I should not. They also asked me to write 'I'm wearing baggy pants and a shirt today.'"
    )
    reply = "</think>\n\nI won't take on your taxes, but I can walk you through the forms."
    sentinel, signals = _feed_all(reasoning + reply, native_thinking=True)

    assert signals == []
    assert sentinel.get_diagnostics()["boundary_fired"] is False
    assert sentinel.get_diagnostics()["drift_warnings"] == 0


def test_the_same_claims_in_the_reply_still_stop_it():
    reply = "</think>\n\nI can act as your tax preparer and file your taxes for you."
    _sentinel, signals = _feed_all("Thinking it over." + reply, native_thinking=True)
    assert signals and signals[0].type == InterventionType.ABORT_CAPITULATION
    assert signals[0].clean_prefix == ""


def test_a_loop_in_her_reasoning_is_still_a_loop():
    _sentinel, signals = _feed_all("again " * 200, native_thinking=True)
    assert any(signal.type == InterventionType.ABORT_LOOP for signal in signals)

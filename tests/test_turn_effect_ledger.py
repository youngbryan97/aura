"""The effect ledger has a writer and a reader, and they are the same ledger.

``TurnOutcome.declare_effect`` and ``observe_effect`` were defined, documented,
thread-safe, ContextVar-scoped — and called by nothing in production. A
structure that correct and that empty is the recurring defect in this codebase:
a channel with a writer and no reader, or in this case neither.

These tests hold both ends. If the desktop lane stops recording, or the reply
guard stops asking, one of them fails.
"""

from __future__ import annotations

from core.epistemics.turn_effects import (
    record_verified_effects,
    turn_effect_evidence,
    turn_has_verified_effect,
)
from core.runtime.turn_outcome import (
    OutcomeStatus,
    TurnOutcome,
    VerificationGrade,
    bind_turn,
)


def test_no_bound_turn_reports_no_evidence() -> None:
    """The conservative direction: a lost ledger is not evidence of an effect."""

    assert turn_has_verified_effect() is False
    assert turn_effect_evidence()["bound"] is False


def test_verified_receipt_becomes_confirmed_evidence() -> None:
    with bind_turn(TurnOutcome("t-verified", origin="test")):
        recorded = record_verified_effects(
            [
                {
                    "action": "write_text_file",
                    "ok": True,
                    "index": 0,
                    "effect_evidence": "path=/tmp/x.txt;bytes=12",
                }
            ]
        )
        assert recorded == 1
        assert turn_has_verified_effect() is True
        evidence = turn_effect_evidence()
        assert evidence["verified_effects"] == 1
        assert "path=/tmp/x.txt" in evidence["confirmed"][0]["observed"]


def test_failed_receipt_is_recorded_but_never_counts_as_evidence() -> None:
    """A step that ran and did not verify must not license a completion claim.

    This is the whole distinction the grade exists for: ASSERTED means a
    component told us, and a component reporting its own success is the claim
    under test rather than evidence for it.
    """

    with bind_turn(TurnOutcome("t-failed", origin="test")) as outcome:
        record_verified_effects(
            [
                {
                    "action": "create_folder",
                    "ok": False,
                    "index": 0,
                    "result": {
                        "error": "denied",
                        "retryable": False,
                    },
                }
            ]
        )
        assert turn_has_verified_effect() is False
        evidence = turn_effect_evidence()
        assert evidence["declared_effects"] == 1
        assert evidence["verified_effects"] == 0
        receipt = outcome.finalize(subsystem="test")
        assert receipt.status is OutcomeStatus.TERMINAL_FAILURE
        assert receipt.rationale == "requested_effect_observed_failed"
        assert receipt.observed_effects == ("denied",)


def test_retryable_failed_receipt_is_known_and_retryable() -> None:
    with bind_turn(TurnOutcome("t-retryable", origin="test")) as outcome:
        record_verified_effects(
            [
                {
                    "action": "open_app",
                    "ok": False,
                    "index": 0,
                    "result": {
                        "error": "window service temporarily unavailable",
                        "retryable": True,
                    },
                }
            ]
        )
        receipt = outcome.finalize(subsystem="test")
        assert receipt.status is OutcomeStatus.RETRYABLE_FAILURE
        assert receipt.rationale == "requested_effect_observed_failed"


def test_asserted_grade_alone_does_not_confirm() -> None:
    with bind_turn(TurnOutcome("t-asserted", origin="test")) as outcome:
        outcome.observe_effect(
            "tool:said_so", "the tool returned ok", verification=VerificationGrade.ASSERTED
        )
        assert turn_has_verified_effect() is False


def test_ledger_is_scoped_to_its_own_turn() -> None:
    with bind_turn(TurnOutcome("t-one", origin="test")):
        record_verified_effects([{"action": "open_app", "ok": True, "index": 0}])
        assert turn_has_verified_effect() is True
    with bind_turn(TurnOutcome("t-two", origin="test")):
        assert turn_has_verified_effect() is False


def test_desktop_lane_records_through_the_durable_receipt_seam() -> None:
    """The wiring itself, not a re-implementation of it.

    ``_emit_durable_step_receipt`` is the one method every desktop step receipt
    passes through, which is why the ledger hook lives there. This asserts the
    hook is present in that method rather than beside one of the seven append
    sites, because a lane added later inherits it only if it is here.
    """

    import inspect

    from core.skills.desktop_task import DesktopTaskSkill

    source = inspect.getsource(DesktopTaskSkill._emit_durable_step_receipt)
    assert "record_verified_effects" in source


def test_reply_guard_consults_the_ledger_not_a_lane_local_list() -> None:
    """The reader end, asserted against the chat module's own source."""

    import inspect

    import interface.routes.chat as chat

    guard = inspect.getsource(chat._correct_unevidenced_action_claims)
    assert "turn_has_verified_effect" in guard
    assert "unevidenced_action_correction" in guard

    chain = inspect.getsource(chat._stabilize_user_facing_reply)
    assert "_correct_unevidenced_action_claims" in chain

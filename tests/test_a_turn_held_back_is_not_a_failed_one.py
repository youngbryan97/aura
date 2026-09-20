"""A background turn the runtime held back is deferred, not failed.

LIVE 2026-09-20, nine times in one uptime: the journal's generation was
cancelled to make room for a chat turn, the engine recorded the
CancelledError as a retryable error, the ledger read "retryable error and
nothing served", cognitive_engine is fail-closed so warning became CRITICAL
SERVICE FAILURE, narrative memory's consolidation failed on that, and the
healer dispatched an emergency repair. For a turn the runtime chose not to
run yet.

The ledger now has a word for it. DEFERRED is neither success nor failure,
it is reported below the escalation floor, and a person's own turn never
takes it: their cancellation stays theirs.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from core.runtime.turn_outcome import OutcomeStatus, TurnOutcome

pytestmark = pytest.mark.unit


def test_a_deferral_with_nothing_served_is_deferred() -> None:
    outcome = TurnOutcome(origin="stream_narrative")
    outcome.record_deferral(reason="pre_empted", authority="cognitive_engine")
    outcome.mark_served("")
    receipt = outcome.finalize(subsystem="cognitive_engine")
    assert receipt.status is OutcomeStatus.DEFERRED
    assert receipt.rationale == "deferred:pre_empted"
    assert not receipt.status.is_failure
    assert not receipt.status.is_success


def test_a_deferral_that_still_served_is_judged_on_what_it_served() -> None:
    outcome = TurnOutcome(origin="stream_narrative")
    outcome.record_deferral(reason="memory_pressure", authority="router")
    outcome.mark_served("a fallback answered")
    assert outcome.finalize(subsystem="cognitive_engine").status is OutcomeStatus.SUCCEEDED


def test_a_deferred_turn_is_not_a_degradation(caplog, monkeypatch) -> None:
    import core.runtime.turn_outcome as ledger

    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ledger, "record_degradation",
        lambda subsystem, exc, **kw: recorded.append((subsystem, kw.get("severity", "")))
    )
    outcome = TurnOutcome(origin="curiosity_loop")
    outcome.record_deferral(reason="pre_empted", authority="cognitive_engine")
    outcome.mark_served("")
    with caplog.at_level(logging.INFO, logger="Aura.TurnOutcome"):
        outcome.finalize(subsystem="cognitive_engine")
    assert recorded == []
    assert any("turn deferred" in r.getMessage() for r in caplog.records)


def test_the_engine_records_a_cancelled_background_turn_as_deferred() -> None:
    from core.brain.cognitive_engine import _note_how_the_turn_ended

    outcome = TurnOutcome(origin="stream_narrative")
    _note_how_the_turn_ended(outcome, asyncio.CancelledError(), origin="stream_narrative")
    outcome.mark_served("")
    assert outcome.finalize().status is OutcomeStatus.DEFERRED


def test_the_engine_records_an_admission_hold_as_deferred() -> None:
    from core.brain.cognitive_engine import _note_how_the_turn_ended
    from core.brain.llm.mlx_client import _ModelLoadAdmissionDeniedError

    outcome = TurnOutcome(origin="curiosity_loop")
    _note_how_the_turn_ended(
        outcome, _ModelLoadAdmissionDeniedError("memory_pressure"), origin="curiosity_loop"
    )
    outcome.mark_served("")
    receipt = outcome.finalize()
    assert receipt.status is OutcomeStatus.DEFERRED
    assert "memory_pressure" in receipt.rationale


def test_a_persons_cancelled_turn_is_still_theirs_to_lose() -> None:
    """Deferral is for work the runtime chose to hold; a person's cancelled
    turn is recorded as the error it was."""
    from core.brain.cognitive_engine import _note_how_the_turn_ended

    outcome = TurnOutcome(origin="user")
    _note_how_the_turn_ended(outcome, asyncio.CancelledError(), origin="user")
    outcome.mark_served("")
    assert outcome.finalize().status is OutcomeStatus.RETRYABLE_FAILURE


def test_an_ordinary_error_is_still_an_error() -> None:
    from core.brain.cognitive_engine import _note_how_the_turn_ended

    outcome = TurnOutcome(origin="curiosity_loop")
    _note_how_the_turn_ended(outcome, RuntimeError("worker crashed"), origin="curiosity_loop")
    outcome.mark_served("")
    assert outcome.finalize().status is OutcomeStatus.RETRYABLE_FAILURE

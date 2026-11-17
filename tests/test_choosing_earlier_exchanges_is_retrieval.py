"""Choosing by meaning which earlier exchanges a request bears on is retrieval on the turn's receipt (R11).

LIVE 16 November 2026, 14:41 local: a typed turn of 34.68 s had 9.79 s unattributed, 3.25 s of it between her
thinking's start and the line that names this choice.
"""

from __future__ import annotations

import pytest

from core.conversation import delivered_history
from core.verify.turn_receipt import recording_turn

pytestmark = pytest.mark.unit


def _slow_reach(seconds: float):
    real = delivered_history.measure_reach

    def reach(*args, **kwargs):
        import time

        time.sleep(seconds)
        return real(*args, **kwargs)

    return reach


def test_a_request_reading_its_history_records_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(delivered_history, "measure_reach", _slow_reach(0.03))
    monkeypatch.setattr(delivered_history, "_attested_pairs",
                        lambda exchanges, **_kw: [{"user": "hello", "aura": "hi"}, {"user": "and you?", "aura": "fine"}])
    with recording_turn("history-by-meaning", phases_available=()) as receipt:
        delivered_history.reached_exchange_messages([object()], budget_chars=10_000, request="How did I say I slept?")

    assert receipt.latency_s["retrieval"] >= 0.03


def test_her_own_turn_reading_the_sitting_whole_records_none(monkeypatch) -> None:
    monkeypatch.setattr(delivered_history, "_attested_pairs",
                        lambda exchanges, **_kw: [{"user": "hello", "aura": "hi"}])
    with recording_turn("her-own-turn", phases_available=()) as receipt:
        delivered_history.reached_exchange_messages([object()], budget_chars=10_000, request="")

    assert "retrieval" not in receipt.latency_s

"""A turn's receipt counts the time of every generation pass, the rejected ones too.

LIVE 14 November 2026, a typed turn of 86 s: her worker rejected its first draft (23 s of prefill and decode) and
wrote it again, and her receipt counted only the second pass, calling 46 s unattributed.
"""
from __future__ import annotations

import pytest

from core.brain.llm.mlx_client_recording import _earlier_passes
from core.verify import turn_receipt


def test_only_the_passes_before_the_last_are_earlier():
    attempts = [{"prefill_seconds": 8.9, "decode_seconds": 14.0}, {"prefill_seconds": 21.9, "decode_seconds": 10.3}]
    assert _earlier_passes(attempts) == [(8.9, 14.0)]
    assert _earlier_passes(attempts[:1]) == [] and _earlier_passes(None) == [] and _earlier_passes("x") == []


def test_a_pass_that_did_not_measure_a_part_adds_nothing_for_it():
    attempts = [{"prefill_seconds": None, "decode_seconds": "nan"}, {"prefill_seconds": -1, "decode_seconds": 2.0},
                {"prefill_seconds": 1.0, "decode_seconds": 1.0}]
    assert _earlier_passes(attempts) == [(None, None), (None, 2.0)]


def test_the_receipt_adds_each_pass_it_is_told():
    with turn_receipt.recording_turn("t-passes", phases_available=()) as receipt:
        for prefill, decode in ((8.9, 14.0), (21.85, 10.3)):
            turn_receipt.record_latency("prefill", prefill)
            turn_receipt.record_latency("decode", decode)
    assert receipt.latency_s["prefill"] == pytest.approx(30.75)
    assert receipt.latency_s["decode"] == pytest.approx(24.3)
    assert receipt.latency_samples == {"prefill": 2, "decode": 2}


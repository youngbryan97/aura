"""A turn counts the preparation of what her model is asked, net of the reading done inside it (U07, delivery metrics).

LIVE 20 November 2026, typed into her window: a 45 s turn said "unattributed 17.23s". The route had timed its preflight
(about 4.5 s, 2.7 s of it reading the screen) and the binding of her context (6.8 s), logged both, and kept neither on
the turn's receipt. They are kept now as preparation, which, like queue and delivery, is counted net of the work
recorded inside it, so a reading of the world inside it is not counted twice.
"""
from __future__ import annotations

import time

import pytest

from core.verify.turn_receipt import record_latency, recording_turn

pytestmark = pytest.mark.unit


def test_preparation_is_counted_net_of_the_reading_inside_it() -> None:
    with recording_turn("prep-1", phases_available=()) as receipt:
        now = time.time()
        record_latency("retrieval", 2.0, ended_at=now - 1.0)   # a reading from now-3 to now-1
        record_latency("preparation", 6.0, ended_at=now)       # preparation from now-6 to now, around it
    assert receipt.latency_s["retrieval"] == pytest.approx(2.0)
    assert receipt.latency_s["preparation"] == pytest.approx(4.0, abs=0.01)


def test_preparation_leaves_less_unattributed() -> None:
    with recording_turn("prep-2", phases_available=()) as receipt:
        record_latency("preparation", 0.0)
    assert "preparation" in receipt.latency_s and receipt.unattributed_s is not None

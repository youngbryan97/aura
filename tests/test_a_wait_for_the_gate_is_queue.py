"""A person's turn waiting for the generation gate is queue time on its receipt.

LIVE 16 November 2026, 14:23 local: a typed turn took 76.04 s, 51.41 s of it unattributed. The gate's queue ran from
its entry to the cortex call, and the router's wait for a generation slot came after that, behind her own mail reply
holding the cortex for 138.8 s. Nothing timed that wait.
"""

from __future__ import annotations

import time

import pytest

from core.brain.llm_health_router_gate_owners import _a_person_waits
from core.verify.turn_receipt import record_latency, recording_turn

pytestmark = pytest.mark.unit


def test_the_routers_wait_is_counted_as_queue() -> None:
    with recording_turn("a-wait-for-the-gate", phases_available=()) as receipt:
        with _a_person_waits(counted_as_queue=True):
            time.sleep(0.05)

    assert receipt.latency_s["queue"] >= 0.05


def test_a_wait_that_also_holds_work_is_not_queue_unless_it_says_so() -> None:
    with recording_turn("a-voice-generation", phases_available=()) as receipt:
        with _a_person_waits():
            time.sleep(0.01)

    assert "queue" not in receipt.latency_s


def test_work_recorded_inside_the_wait_leaves_it() -> None:
    with recording_turn("a-wait-with-work", phases_available=()) as receipt:
        with _a_person_waits(counted_as_queue=True):
            time.sleep(0.05)
            record_latency("decode", 0.04)

    assert receipt.latency_s["queue"] == pytest.approx(0.01, abs=0.02)
    assert receipt.latency_s["decode"] == pytest.approx(0.04)

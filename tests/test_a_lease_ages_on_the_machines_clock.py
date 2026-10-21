"""A generation-gate lease ages on the machine's clock, whatever clock the run keeps.

The reports ground's dry run on 21 October lost 9 of 32 replies. A subject run
installs an experiment clock in place of `time.time` that advances a thirtieth
of a second a frame, and the router stamped and aged its gate leases on
`time.time`. A background generation stuck for seventy-five real seconds read as
0.6 seconds old, so the foreground ladder judged the holder young and in budget
and refused her reply as saturated.
"""
from __future__ import annotations

import time

import core.brain.llm_health_router as router
from core.brain.llm_health_router_gate_readings import (
    _oldest_generation_gate_lease_age_s,
    generation_gate_snapshot,
)


def _clear() -> None:
    with router._GENERATION_GATE_STATE_LOCK:
        for lease_id in list(router._GENERATION_GATE_ACTIVE_LEASES):
            router._GENERATION_GATE_ACTIVE_LEASES.pop(lease_id, None)
            router._GENERATION_GATE_LEASE_DEADLINES.pop(lease_id, None)
            router._GENERATION_GATE_FOREGROUND_LEASES.discard(lease_id)


def test_a_frozen_run_clock_does_not_stop_a_lease_ageing(monkeypatch):
    _clear()
    frozen = 1_792_590_554.0
    monkeypatch.setattr(time, "time", lambda: frozen)
    real = [100.0]
    import core.brain.llm_health_router_gate_readings as readings

    monkeypatch.setattr(router, "_gate_clock", lambda: real[0])
    monkeypatch.setattr(readings, "_gate_clock", lambda: real[0])
    lease = router._mark_generation_gate_acquired("system:unknown", timeout_s=30.0)
    try:
        real[0] += 75.0
        assert _oldest_generation_gate_lease_age_s() == 75.0
        assert router._generation_gate_lease_has_time(lease) is False
        snapshot = generation_gate_snapshot()
        assert snapshot["oldest"]["age_s"] == 75.0
    finally:
        with router._GENERATION_GATE_STATE_LOCK:
            router._GENERATION_GATE_ACTIVE_LEASES.pop(lease, None)
            router._GENERATION_GATE_LEASE_DEADLINES.pop(lease, None)


def test_the_gate_clock_is_the_monotonic_one():
    before = time.monotonic()
    reading = router._gate_clock()
    assert before <= reading <= time.monotonic()

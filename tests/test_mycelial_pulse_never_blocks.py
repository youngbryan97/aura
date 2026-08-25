"""A telemetry pulse must never freeze the mind or lose edge evidence.

Measured live: the event loop sat in ``pulse_hypha`` -> ``MycelialNetwork._lock``
during a desktop task, and the hypervisor reported

    severe event-loop lag 97.192s

Ninety-seven seconds of a frozen runtime, waiting to increment a counter.
Contended evidence is buffered for the next owned update; waiting for the lock
costs everything.
"""

from __future__ import annotations

import threading
import time

import core.mycelium as mycelium_module
from core.mycelium import MycelialNetwork


def test_a_held_lock_does_not_block_the_pulse():
    mycelium = MycelialNetwork()
    released = threading.Event()
    holding = threading.Event()

    def _hold():
        with MycelialNetwork._lock:
            holding.set()
            released.wait(timeout=5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert holding.wait(timeout=5.0), "fixture failed to take the lock"

    try:
        started = time.monotonic()
        result = mycelium.pulse_hypha("homeostasis", "cognition", success=True)
        elapsed = time.monotonic() - started
    finally:
        released.set()
        holder.join(timeout=5.0)

    assert elapsed < 1.0, (
        f"pulse waited {elapsed:.2f}s on a contended lock; on the event loop "
        "that is the whole runtime stalling for a counter"
    )
    assert result is False, "a dropped pulse reports that it was dropped"


def test_an_uncontended_pulse_still_works():
    mycelium = MycelialNetwork()
    mycelium.establish_connection("test_pulse_source", "test_pulse_target")
    assert mycelium.pulse_hypha("test_pulse_source", "test_pulse_target") is True


def test_contended_pulse_is_merged_into_next_owned_update():
    mycelium = MycelialNetwork()
    mycelium.establish_connection("retained_source", "retained_target")
    released = threading.Event()
    holding = threading.Event()

    def _hold():
        with MycelialNetwork._lock:
            holding.set()
            released.wait(timeout=5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert holding.wait(timeout=5.0), "fixture failed to take the lock"
    try:
        assert (
            mycelium.pulse_hypha(
                "retained_source", "retained_target", success=True
            )
            is False
        )
    finally:
        released.set()
        holder.join(timeout=5.0)

    assert mycelium._deferred_pulses["retained_source->retained_target"] == (1, 0)
    assert (
        mycelium.pulse_hypha("retained_source", "retained_target", success=True)
        is True
    )

    hypha = mycelium.get_hypha("retained_source", "retained_target")
    assert hypha is not None
    assert hypha.pulse_count == 2
    assert hypha.strength == 2.0
    assert "retained_source->retained_target" not in mycelium._deferred_pulses


def test_deferred_handoff_never_waits_for_its_own_aggregate_lock():
    mycelium = MycelialNetwork()
    source = "handoff_source"
    target = "handoff_target"
    mycelium.establish_connection(source, target)
    topology_holding = threading.Event()
    topology_release = threading.Event()
    aggregate_holding = threading.Event()
    aggregate_release = threading.Event()

    def _hold_topology():
        with MycelialNetwork._lock:
            topology_holding.set()
            topology_release.wait(timeout=5.0)

    def _hold_aggregate():
        with mycelium_module._DEFERRED_PULSE_LOCK:
            aggregate_holding.set()
            aggregate_release.wait(timeout=5.0)

    topology_holder = threading.Thread(target=_hold_topology, daemon=True)
    aggregate_holder = threading.Thread(target=_hold_aggregate, daemon=True)
    topology_holder.start()
    aggregate_holder.start()
    assert topology_holding.wait(timeout=5.0)
    assert aggregate_holding.wait(timeout=5.0)

    try:
        started = time.monotonic()
        assert mycelium.pulse_hypha(source, target, success=True) is False
        elapsed = time.monotonic() - started
    finally:
        aggregate_release.set()
        topology_release.set()
        aggregate_holder.join(timeout=5.0)
        topology_holder.join(timeout=5.0)

    assert elapsed < 0.2
    assert mycelium.pulse_hypha(source, target, success=True) is True
    hypha = mycelium.get_hypha(source, target)
    assert hypha is not None
    assert hypha.pulse_count == 2

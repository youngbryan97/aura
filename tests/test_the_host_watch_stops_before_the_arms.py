"""The experiment was measuring its own CPU load, through her immune system.

A paired arm forks one anchor twice and compares where the two rollouts end. For
that to be a reading of her, both arms have to see the same computation, which is
why the harness stops her free-running loops before the anchors.

One loop was not on the list. The defensive host watch samples CPU, memory and
disk every fifteen seconds and sweeps the local network every sixty. A strain
sample becomes a threat in the immune system and a scored entry in the emergency
protocol; a sweep writes the fingerprint of every device on the network with the
wall-clock time it was first seen. A snapshot carries none of it.

MEASURED on the decisive seed-23 campaign of 24 October. The late untouched fork
separated from the cached one with a lower bound of 0.0114, against a weakest cut
of 0.00126, so drift alone cleared the cuts' bar and `partition_irreducibility`
and `partition_beats_nulls` both failed on that control — 22 of 24 criteria. Of
the twenty-six module attributes a restore did not carry, twenty-four came from
this thread, "resource strain: cpu at 95%" among them, which the sweep's own load
had caused.
"""
from __future__ import annotations

import threading

import pytest

from core.security import defensive_runtime

pytestmark = pytest.mark.unit


@pytest.fixture
def a_watch_that_is_running(monkeypatch):
    """A stand-in for the host watch, so the test stops a thread of its own."""
    stop = threading.Event()
    running = threading.Event()

    def loop() -> None:
        running.set()
        while not stop.wait(0.01):
            pass

    thread = threading.Thread(target=loop, name="aura-defensive-monitor-test", daemon=True)
    monkeypatch.setattr(defensive_runtime, "_MONITOR_STOP", stop)
    monkeypatch.setattr(defensive_runtime, "_MONITOR_THREAD", thread)
    thread.start()
    running.wait(timeout=2.0)
    yield thread
    stop.set()
    thread.join(timeout=2.0)


def test_the_watch_stops_and_says_what_it_stopped(a_watch_that_is_running):
    stopped = defensive_runtime.stop_watching_the_host()
    assert stopped == [
        "defensive_runtime.resource_monitor",
        "defensive_runtime.network_sweep",
    ]
    a_watch_that_is_running.join(timeout=2.0)
    assert not a_watch_that_is_running.is_alive(), "the thread kept reading the machine"


def test_nothing_running_stops_nothing(monkeypatch):
    """A run that never started the watch has no thread to stop, and says so."""
    monkeypatch.setattr(defensive_runtime, "_MONITOR_THREAD", None)
    assert defensive_runtime.stop_watching_the_host() == []


def test_quiescing_the_arms_reaches_the_host_watch(monkeypatch):
    """Every paired run quiesces through one door, so the fix reaches all of them."""
    import asyncio

    from core.subject import driver

    called: list[str] = []

    def stop() -> list[str]:
        called.append("host watch")
        return ["defensive_runtime.resource_monitor", "defensive_runtime.network_sweep"]

    monkeypatch.setattr(defensive_runtime, "stop_watching_the_host", stop)

    async def no_loops(*, only=None):
        return ["NeuralMesh.spike_clock"]

    monkeypatch.setattr("core.subject.organism.quiesce", no_loops)
    monkeypatch.setattr("core.subject.organism._live_tasks", lambda: [])

    class _Runtime:
        organism = None

    stopped = asyncio.run(driver.quiesce_organism(_Runtime()))
    assert called == ["host watch"]
    assert "defensive_runtime.resource_monitor" in stopped
    assert "NeuralMesh.spike_clock" in stopped, "the loops it already stopped are still stopped"

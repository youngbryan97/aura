"""A strain on her host is read over the period that judges it, and is answered by a remedy for what is strained.

LIVE 17 to 19 November 2026, her neural stream. Her immune system's resource monitor flagged "resource strain: cpu at
100%" at severity 0.9 some 20 times, every one exactly 100%, mostly while a lint or compile pass held every core for a
moment. At 16:36 on 19 November the host measured 20% busy over the seconds around it. The monitor samples every 15 s
and took the figure some other thread had last measured over half a second. It now reads the processor's counters
across its own period.

The same stream logged "Metabolic pressure mitigation executed: state=stressed tier=moderate" every 30 s, 7,419 times
from 16 to 19 November. Her data volume was at 94%, and each time she ran a full garbage collection and the memory shed
order, none of which frees a byte of disk. Memory eviction now answers memory pressure only.
"""
from __future__ import annotations

import pytest

import core.runtime.resource_observation as ro
from core.ops.metabolic_monitor import MetabolicMonitor, MetabolismSnapshot
from core.security.enforcement import ResourceMonitor

pytestmark = pytest.mark.unit


def _the_host(monkeypatch, counters):
    """Her resource observer, reading cumulative processor seconds (busy, idle) in turn, and claiming 100% busy over the
    last half second each time, as the shared sampler did."""
    readings = iter(counters)
    observer = ro.get_resource_observer()

    def compute():
        busy, idle = next(readings)
        return ro.ComputeObservation(
            provenance=ro.ObservationProvenance(source=ro.ObservationSource.HOST, scenario_id="host"),
            cpu_percent=100.0, cpu_percent_measured=True, cpu_count=18, load_1m=5.3,
            cpu_user_seconds=busy, cpu_system_seconds=0.0, cpu_idle_seconds=idle)

    monkeypatch.setattr(observer, "compute", compute)
    monkeypatch.setattr(ro, "get_resource_observer", lambda: observer)


def test_a_moment_of_every_core_busy_is_not_a_strain_over_the_period(monkeypatch):
    # 18 cores for 15 s is 270 processor-seconds; 54 of them busy is the 20% the host measured at 16:36.
    _the_host(monkeypatch, [(1000.0, 5000.0), (1054.0, 5216.0)])
    monitor = ResourceMonitor()
    assert "cpu" not in monitor.sample(), "the first sample has no period behind it"
    assert monitor.sample()["cpu"] == pytest.approx(20.0)


def test_a_strain_held_over_the_period_is_still_read(monkeypatch):
    _the_host(monkeypatch, [(1000.0, 5000.0), (1265.0, 5005.0)])
    monitor = ResourceMonitor()
    monitor.sample()
    assert monitor.sample()["cpu"] == pytest.approx(100.0 * 265.0 / 270.0)


def test_counters_that_did_not_move_or_went_back_give_no_reading(monkeypatch):
    _the_host(monkeypatch, [(1000.0, 5000.0), (1000.0, 5000.0), (900.0, 5100.0)])
    monitor = ResourceMonitor()
    monitor.sample()
    assert "cpu" not in monitor.sample()
    assert "cpu" not in monitor.sample()


def _snapshot(**values):
    readings = {"cpu_percent": 20.0, "ram_rss_mb": 5009.0, "ram_percent": 73.0, "disk_usage_percent": 94.2,
                "llm_latency_avg": 0.0, "health_score": 0.8, "pressure_state": "stressed", **values}
    return MetabolismSnapshot(**readings)


@pytest.mark.parametrize(("values", "evicts"), [
    ({}, False),                                   # 19 November: the disk at 94%, memory at 73%, her RSS at 5 GB
    ({"ram_percent": 90.0}, True),
    ({"disk_usage_percent": 99.0}, False),
])
def test_memory_eviction_answers_memory_pressure_only(values, evicts):
    monitor = MetabolicMonitor.__new__(MetabolicMonitor)
    monitor.ram_threshold_mb = 54_000
    monitor.disk_threshold = 92.0
    snapshot = _snapshot(**values)
    assert monitor._eviction_pressure_present(snapshot) is evicts
    if not evicts:
        assert monitor._eviction_pressure_present(_snapshot(**values, ram_rss_mb=60_000.0))

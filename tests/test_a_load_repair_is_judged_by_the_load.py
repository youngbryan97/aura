"""A repair of her load is judged by the load, and one that did not help is not run again inside its cooldown (F16).

LIVE 21 November 2026, on her neural stream after a restart at 18:55: "Executing repair: reduce_load on global
(health_before=0.000)" then "Repair INEFFECTIVE: reduce_load on global (0.000 -> 0.000)", 438 times in 27 minutes. Her
immune system asked for it every few seconds while the host's memory was held, mostly by test runs of mine beside her.
"global" had a handler, her metabolism's reduce_load, and nothing measured it, so no run could ever be read as having
helped. And a requested repair ran each time it was asked, inside the cooldown the engine keeps for its own repairs.
"""
from __future__ import annotations

import asyncio

import pytest

from core.cognitive.autopoiesis import AutopoiesisEngine, RepairStrategy

pytestmark = pytest.mark.unit


def _engine(readings: list[float], runs: list[int]) -> AutopoiesisEngine:
    engine = AutopoiesisEngine()
    engine._request_governance_approval = lambda _component, _strategy: True

    def reduce_load() -> bool:
        runs.append(1)
        return True

    engine.register_repair_handler(RepairStrategy.REDUCE_LOAD, "global", reduce_load,
                                   measure=lambda: readings.pop(0) if len(readings) > 1 else readings[0])
    return engine


def test_a_repair_with_a_measure_is_judged_by_it_and_its_component_is_not_monitored():
    runs: list[int] = []
    engine = _engine([0.18, 0.31], runs)
    result = asyncio.run(engine.request_repair("global", RepairStrategy.REDUCE_LOAD))
    assert (result.health_before, result.health_after, result.success) == (0.18, 0.31, True)
    assert "global" not in engine._health_fns, "a measure is read for the repair only, never every tick"


def test_a_repair_that_did_not_help_is_not_run_again_inside_its_cooldown():
    runs: list[int] = []
    engine = _engine([0.18], runs)
    first = asyncio.run(engine.request_repair("global", RepairStrategy.REDUCE_LOAD))
    assert (first.health_before, first.health_after, first.success) == (0.18, 0.18, False)
    again = asyncio.run(engine.request_repair("global", RepairStrategy.REDUCE_LOAD))
    assert again is first and runs == [1] and engine._repair_attempts["global"] == 1
    first.timestamp -= engine._REPAIR_COOLDOWN + 1  # the cooldown passes
    asyncio.run(engine.request_repair("global", RepairStrategy.REDUCE_LOAD))
    assert runs == [1, 1]


def test_a_repair_that_helped_may_be_asked_for_again():
    runs: list[int] = []
    engine = _engine([0.18, 0.31, 0.31, 0.40], runs)
    asyncio.run(engine.request_repair("global", RepairStrategy.REDUCE_LOAD))
    asyncio.run(engine.request_repair("global", RepairStrategy.REDUCE_LOAD))
    assert runs == [1, 1]


def test_her_load_is_measured_as_what_is_left_of_the_hosts_memory():
    from core.adaptation.autonomous_resilience import host_memory_headroom

    reading = host_memory_headroom()
    assert 0.0 <= reading <= 1.0

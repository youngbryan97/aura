"""The body reads the host the turn declared, whatever the machine is doing.

The battery prepares each turn's host reading (calm or stress) and holds it. The
hold memoised the real machine's first answer instead, and proprioception wrote
that answer over the prepared reading every turn. During the seed-7 run another
campaign loaded the machine, and the CPU reflex and homeostasis acted on it on
hundreds of turns the recording could not explain. These pin that the declared
host answers compute, memory and thermal from the prepared reading, and leaves
the harness's bookkeeping to the real observer.
"""

from __future__ import annotations

import pytest

from core.runtime import resource_psutil
from core.runtime.resource_observation import (
    ResourceObserver,
    SimulatedResourceObserver,
    set_resource_observer_for_test,
)
from core.subject.snapshot import _DeclaredHost

pytestmark = pytest.mark.unit

CALM = {"cpu_usage": 12.0, "vram_usage": 35.0, "temperature": 41.0}
STRESS = {"cpu_usage": 91.0, "vram_usage": 88.0, "temperature": 93.0}


def _busy_machine() -> SimulatedResourceObserver:
    return SimulatedResourceObserver(cpu_percent=99.0, memory_percent=97.0, thermal_level=3, cpu_count=18)


def test_a_busy_machine_does_not_reach_a_calm_turn() -> None:
    host = _DeclaredHost(_busy_machine(), CALM)
    assert host.compute().cpu_percent == 12.0
    assert host.memory().percent == 35.0
    assert host.thermal().level == 0


def test_a_stress_turn_still_reads_as_stress() -> None:
    host = _DeclaredHost(SimulatedResourceObserver(cpu_percent=3.0, memory_percent=10.0, cpu_count=18), STRESS)
    assert host.compute().cpu_percent == 91.0
    assert host.compute().load_1m == pytest.approx(0.91 * 18)
    assert host.memory().percent == 88.0
    assert host.thermal().level == 3


@pytest.mark.parametrize(("celsius", "level"), [(45.0, 0), (78.0, 2), (85.0, 2), (90.0, 3)])
def test_a_declared_temperature_reads_as_the_level_the_thermal_module_would_give(celsius: float, level: int) -> None:
    assert _DeclaredHost(_busy_machine(), {**CALM, "temperature": celsius}).thermal().level == level


def test_the_declared_host_is_an_observer_and_leaves_bookkeeping_to_the_machine() -> None:
    real = _busy_machine()
    host = _DeclaredHost(real, CALM)
    assert isinstance(host, ResourceObserver)
    assert host.disk().total_bytes == real.disk().total_bytes
    # Each observation is stamped when it is taken, so compare what it says.
    assert host.process_ids().pids == real.process_ids().pids


def test_a_later_declaration_replaces_the_earlier_one() -> None:
    host = _DeclaredHost(_busy_machine(), CALM)
    host.declare(STRESS)
    assert host.compute().cpu_percent == 91.0
    host.declare(CALM)
    assert host.compute().cpu_percent == 12.0


def test_a_reader_in_another_thread_between_turns_sees_the_declared_host() -> None:
    """The defensive resource monitor samples from its own thread, between holds."""
    import threading

    from core.security.enforcement import ResourceMonitor

    previous = set_resource_observer_for_test(_DeclaredHost(_busy_machine(), CALM))
    seen: dict = {}
    try:
        worker = threading.Thread(target=lambda: seen.update(ResourceMonitor().sample()))
        worker.start()
        worker.join(timeout=30)
    finally:
        set_resource_observer_for_test(previous)
    assert seen["cpu"] == 12.0
    assert seen["mem"] == 35.0


def test_everything_that_reads_through_psutil_sees_the_declared_host() -> None:
    previous = set_resource_observer_for_test(_DeclaredHost(_busy_machine(), CALM))
    try:
        assert resource_psutil.cpu_percent() == 12.0
        assert resource_psutil.virtual_memory().percent == 35.0
        assert resource_psutil.getloadavg()[0] == pytest.approx(0.12 * 18)
    finally:
        set_resource_observer_for_test(previous)


def test_her_process_size_is_held_where_the_run_declared_it() -> None:
    """The process grows through a run with the harness's snapshots in it, and
    that growth reached homeostasis as memory stress: a clock, different in
    every arm, that crossed the throttling line mid-run."""
    from dataclasses import replace

    class _Growing(SimulatedResourceObserver):
        def __init__(self) -> None:
            super().__init__(cpu_percent=50.0, memory_percent=50.0, cpu_count=18)
            self.rss = 800 * 1024 * 1024

        def memory(self, *args, **kwargs):
            return replace(super().memory(*args, **kwargs), process_rss_bytes=self.rss)

    machine = _Growing()
    host = _DeclaredHost(machine, CALM)
    at_start = host.memory().process_rss_bytes
    assert at_start == 800 * 1024 * 1024
    machine.rss = 5000 * 1024 * 1024
    host.declare(STRESS)
    assert host.memory().process_rss_bytes == at_start
    assert host.memory().percent == 88.0, "the declared reading still moves with the turn"

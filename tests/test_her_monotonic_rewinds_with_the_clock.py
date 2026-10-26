"""Her monotonic readings are on the experiment clock and rewind with a restore.

Much of what she keeps is a `time.monotonic` reading that something later turns
into an elapsed time. Restored while the real clock ran on, seed 71's untouched
anchors read an hour of idle nobody lived; re-aging the readings on restore then
made short time-to-lives depend on how fast the machine ran a turn, and two
untouched passes seconds apart parted anchor 3 by 0.476 (26 October). Her
modules now read the clock's monotonic through their own `time` name; asyncio,
threads and the machinery keep the real one, so every timeout still fires.
"""
from __future__ import annotations

import asyncio
import sys
import time
import types

import pytest

from core.subject.clock import ExperimentClock, real_monotonic

pytestmark = pytest.mark.unit

HERS = "core._her_clock_probe"
MACHINERY = "core.runtime._her_clock_probe"


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch):
    hers = types.ModuleType(HERS)
    hers.time = time
    machinery = types.ModuleType(MACHINERY)
    machinery.time = time
    monkeypatch.setitem(sys.modules, HERS, hers)
    monkeypatch.setitem(sys.modules, MACHINERY, machinery)
    return hers, machinery


@pytest.fixture
def clock():
    made = ExperimentClock(0.1)
    made.install()
    yield made
    made.uninstall()


def test_her_modules_read_the_clock_and_the_machinery_does_not(modules, clock):
    hers, machinery = modules
    assert hers.time is not time and hers.time.monotonic() == clock.monotonic()
    assert machinery.time is time
    assert asyncio.base_events.time.monotonic is real_monotonic


def test_a_frame_moves_it_by_one_step_and_a_restore_rewinds_it_exactly(modules, clock):
    hers, _ = modules
    before, at = hers.time.monotonic(), clock.now()
    clock.advance()
    assert hers.time.monotonic() - before == pytest.approx(0.1)
    clock.set(at)
    assert hers.time.monotonic() == before


def test_a_frame_that_stalls_opens_the_valve_and_it_never_runs_back(modules, clock):
    hers, _ = modules
    clock.stall_after = 0.02
    start = hers.time.monotonic()
    time.sleep(0.05)
    late = hers.time.monotonic()
    assert clock.valve_openings == 1 and late > start
    clock.advance()
    assert hers.time.monotonic() >= late, "what the valve added is kept"
    clock.set(clock.now(), slack=0.0)
    assert clock.slack == 0.0


def test_a_module_imported_after_the_install_is_given_it_at_the_next_frame(clock, monkeypatch):
    late = types.ModuleType("core._her_clock_probe_late")
    late.time = time
    monkeypatch.setitem(sys.modules, late.__name__, late)
    clock.advance()
    assert late.time is not time


def test_uninstalling_gives_every_module_its_time_back(modules):
    hers, _ = modules
    made = ExperimentClock(0.1)
    made.install()
    made.uninstall()
    assert hers.time is time


def test_a_reading_keeps_its_age_in_another_process_that_takes_the_snapshots_offset():
    """A carrier worker installs a clock of its own and restores what the coordinator wrote."""
    coordinator = ExperimentClock(0.1)
    stamp = coordinator.monotonic()
    for _ in range(5):
        coordinator.advance()
    at, offset = coordinator.now(), coordinator.offset
    worker = ExperimentClock(0.1, start=at + 1000.0)
    worker.offset += 37.0  # another process: another real origin
    worker.set(at, offset=offset)
    assert worker.monotonic() - stamp == pytest.approx(0.5)

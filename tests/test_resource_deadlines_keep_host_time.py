"""Frozen subject time must not freeze a bounded resource wait."""
import asyncio
import queue
from types import SimpleNamespace

import pytest

from core.runtime.wall_clock import monotonic_time
from core.subject.clock import ExperimentClock


def test_the_registered_clock_invariant_detects_a_frozen_deadline(monkeypatch):
    from core.runtime import wall_clock
    from core.verify.runtime_invariants import _resource_deadline_clock
    assert wall_clock.host_deadline_clock_holds()
    monkeypatch.setattr(wall_clock, "monotonic_time", lambda: -1.)
    assert not wall_clock.host_deadline_clock_holds()
    assert len(list(_resource_deadline_clock())) == 1


def test_host_time_advances_while_subject_time_stays_frozen():
    import time
    clock = ExperimentClock(0.1)
    clock.install()
    try:
        before, subject = monotonic_time(), clock.monotonic()
        time.sleep(.01)
        assert monotonic_time() > before
        assert clock.monotonic() == subject
        assert clock.valve_openings == 0
    finally:
        clock.uninstall()


@pytest.mark.asyncio
async def test_a_full_database_queue_times_out_without_opening_the_clock_valve(monkeypatch):
    from core.memory import db_writer_queue as dbw
    writer = object.__new__(dbw.SerializedDBWriter)
    writer._accepting = True
    writer._thread = SimpleNamespace(is_alive=lambda: True)
    writer._queue = queue.Queue(maxsize=1)
    writer._queue.put(object())
    monkeypatch.setattr(dbw, "_QUEUE_PUT_TIMEOUT_S", .03)
    monkeypatch.setattr(dbw, "record_degradation", lambda *a, **k: None)
    clock = ExperimentClock(.1)
    clock.install()
    try:
        subject = clock.monotonic()
        with pytest.raises(TimeoutError, match="enqueueing SQLite"):
            await asyncio.wait_for(writer.execute("unused", "unused"), timeout=.5)
        assert clock.monotonic() == subject
        assert clock.valve_openings == 0
    finally:
        clock.uninstall()

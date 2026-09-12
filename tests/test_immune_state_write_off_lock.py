"""The immune ecology is serialized under its lock and written off it.

`_save_state` was called from four sections holding
`core.adaptation.adaptive_immunity._lock`, and it ended in a gateway write
that fsyncs. Lockdep reported it during a subject-core run:

    LOCKDEP blocking_op_under_lock: fsync attempted while holding
    ['core.adaptation.adaptive_immunity._lock']

An on-loop fsync once held the live event loop for twenty minutes, and this
is the same shape one thread over: every other caller of the immune lock —
a health probe reading get_status, the next observation — waits for the
disk. The snapshot needs the lock because it reads the cell population, the
tissue field, the lineage table and the expansion engine. The write needs
none of it.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from core.adaptation.adaptive_immunity import (
    AdaptiveImmuneConfig,
    AdaptiveImmuneSystem,
)
from core.adaptation.immune_state_writer import SingleSlotStateWriter
from core.runtime import lockdep


@pytest.fixture()
def immune(tmp_path):
    cfg = AdaptiveImmuneConfig(population_size=3, max_population=6)
    system = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=5)
    yield system
    system.on_stop()


def _state_file(tmp_path):
    files = list(tmp_path.rglob("*.json"))
    assert files, "no state file written"
    return files[0]


def _held_at_write(system, monkeypatch):
    """Record which locks are held on the thread that does each write."""
    held: list[list[str]] = []
    original = system._write_state_payload

    def probe(payload):
        held.append(lockdep.get_validator().held_names())
        original(payload)

    monkeypatch.setattr(system._state_writer, "_write", probe)
    return held


class TestTheWriteLeavesTheLock:
    def test_an_observation_writes_with_no_locks_held(self, immune, monkeypatch):
        held = _held_at_write(immune, monkeypatch)

        immune.observe_error(RuntimeError("boom"), {"subsystem": "memory"})
        immune.flush_state()

        assert held, "the observation never reached a write"
        assert all(not names for names in held), held

    def test_consolidation_writes_with_no_locks_held(self, immune, monkeypatch):
        held = _held_at_write(immune, monkeypatch)

        immune.dream_consolidate()
        immune.flush_state()

        assert held, "consolidation never reached a write"
        assert all(not names for names in held), held

    def test_lockdep_reports_no_blocking_op_under_the_immune_lock(self, immune):
        lockdep.reset_lockdep_for_test()

        immune.observe_error(RuntimeError("boom"), {"subsystem": "memory"})
        immune.dream_consolidate()
        immune.flush_state()

        offenders = [
            splat.to_dict()
            for splat in lockdep.get_validator().splats()
            if splat.kind == "blocking_op_under_lock"
            and "core.adaptation.adaptive_immunity._lock" in splat.held
        ]
        assert not offenders, offenders

    def test_flush_refuses_under_the_lock(self, immune):
        """Waiting for the disk under the lock costs what the fsync cost."""
        with immune._lock:
            assert immune.flush_state(timeout=0.5) is False


class TestTheLoopIsNotBlocked:
    """The other half of the same rule: no fsync on the event loop either.

    CLAUDE.md records an on-loop fsync that held the live event loop for
    twenty minutes. A durable write reached from async code goes to the
    writer thread, and the loop keeps running.
    """

    async def test_a_forced_save_on_the_loop_does_not_wait_for_the_disk(self, immune):
        assert immune.flush_state(timeout=0.5) is False

        immune._save_state(force=True)

        # Handed over rather than dropped: a thread is carrying it.
        stats = immune._state_writer.stats()
        assert stats["pending"] or stats["writing"] or stats["thread_running"]

    async def test_the_async_observation_path_writes_off_the_loop(self, immune, monkeypatch):
        held = _held_at_write(immune, monkeypatch)

        await immune.observe_error_async(RuntimeError("boom"), {"subsystem": "memory"})
        await asyncio.to_thread(immune.flush_state)

        assert held, "the observation never reached a write"
        assert all(not names for names in held), held

    def test_what_the_loop_deferred_still_lands(self, immune, tmp_path):
        async def on_the_loop():
            immune._save_state(force=True)

        asyncio.run(on_the_loop())
        assert immune.flush_state(timeout=5.0) is True

        assert json.loads(_state_file(tmp_path).read_text())["cells"]


class TestDurabilityIsKept:
    def test_boot_leaves_a_readable_state_file(self, immune, tmp_path):
        payload = json.loads(_state_file(tmp_path).read_text())

        assert payload["cells"]
        assert payload["integrity"]["digest"]

    def test_a_flush_lands_the_newest_snapshot(self, immune, tmp_path):
        immune.observe_error(RuntimeError("boom"), {"subsystem": "memory"})
        assert immune.flush_state() is True

        payload = json.loads(_state_file(tmp_path).read_text())
        assert payload["observation_count"] >= 1

    def test_on_stop_drains_what_is_pending(self, immune, tmp_path):
        immune._save_state(force=True)
        immune.on_stop()

        assert immune._state_writer.stats()["pending"] is False
        assert json.loads(_state_file(tmp_path).read_text())["cells"]

    def test_a_round_trip_still_loads(self, immune, tmp_path):
        immune.observe_error(RuntimeError("boom"), {"subsystem": "memory"})
        immune.flush_state()

        cfg = AdaptiveImmuneConfig(population_size=3, max_population=6)
        reloaded = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=5)
        try:
            assert reloaded._observation_count >= 1
        finally:
            reloaded.on_stop()

    def test_status_reports_what_the_writer_did(self, immune):
        immune.observe_error(RuntimeError("boom"), {"subsystem": "memory"})
        immune.flush_state()

        stats = immune.get_status()["state_writer"]
        assert stats["written"] >= 1
        assert stats["failed"] == 0


class TestTheSlotCoalesces:
    def test_a_burst_costs_fewer_writes_than_submits(self):
        started = threading.Event()
        release = threading.Event()
        written: list[int] = []

        def slow_write(payload):
            written.append(payload)
            started.set()
            release.wait(timeout=5.0)

        writer = SingleSlotStateWriter("test.burst", slow_write, idle_timeout_s=1.0)
        try:
            writer.submit(1, background=True)
            assert started.wait(timeout=5.0)
            for value in range(2, 12):
                writer.submit(value, background=True)
            release.set()
            assert writer.flush(timeout=5.0) is True
        finally:
            release.set()
            writer.stop(timeout=5.0)

        assert written == [1, 11], written

    def test_a_caller_that_will_flush_starts_no_thread(self):
        written: list[int] = []
        writer = SingleSlotStateWriter("test.inline", written.append)

        writer.submit(7, background=False)
        assert writer.stats()["thread_running"] is False
        assert writer.flush(timeout=1.0) is True

        assert written == [7]
        assert writer.stats()["inline_writes"] == 1

    def test_a_failing_write_is_reported_and_the_writer_survives(self):
        seen: list[BaseException] = []

        def explode(_payload):
            raise OSError("disk gone")

        writer = SingleSlotStateWriter("test.fail", explode, on_error=seen.append)
        writer.submit(1, background=False)
        writer.flush(timeout=1.0)

        assert len(seen) == 1
        assert writer.stats()["failed"] == 1
        assert writer.stats()["last_error"].startswith("OSError")

    def test_two_flushes_at_once_both_see_the_write_land(self):
        """A second flusher waits for the write rather than reporting failure."""
        started = threading.Event()
        release = threading.Event()
        written: list[int] = []

        def slow_write(payload):
            started.set()
            release.wait(timeout=5.0)
            written.append(payload)

        writer = SingleSlotStateWriter("test.concurrent", slow_write)
        results: dict[str, bool] = {}

        def flusher(tag):
            results[tag] = writer.flush(timeout=5.0)

        writer.submit(3, background=False)
        first = threading.Thread(target=flusher, args=("a",))
        first.start()
        assert started.wait(timeout=5.0)

        second = threading.Thread(target=flusher, args=("b",))
        second.start()
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)

        assert written == [3]
        assert results == {"a": True, "b": True}, results

    def test_the_thread_retires_when_nothing_arrives(self):
        writer = SingleSlotStateWriter("test.retire", lambda _p: None, idle_timeout_s=0.05)
        try:
            writer.submit(1, background=True)
            assert writer.flush(timeout=2.0) is True

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if not writer.stats()["thread_running"]:
                    break
                time.sleep(0.02)

            assert writer.stats()["thread_running"] is False

            # And it comes back for the next deferred payload.
            writer.submit(2, background=True)
            assert writer.flush(timeout=2.0) is True
        finally:
            writer.stop(timeout=2.0)

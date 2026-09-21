"""A synchronous save reached from async code is written off the loop.

LIVE 2026-09-19 01:18: the event loop stalled 5.5 s inside an fsync. The
arbiter chose an initiative inside an async synthesis, the choice engine
saved its preferences, and the save was a durable atomic write three
synchronous calls below the ``async def``. The async-write ratchet reads
source, so it could not see it.
"""
from __future__ import annotations

import asyncio
import threading

from core.runtime import atomic_writer, lockdep


def test_off_the_loop_it_writes_before_returning(tmp_path):
    target = tmp_path / "prefs.json"
    assert atomic_writer.atomic_write_text_behind(target, "one")
    assert target.read_text() == "one"


def test_on_the_loop_the_newest_body_lands_off_it(tmp_path):
    target = tmp_path / "prefs.json"
    writers: list[str] = []
    real = atomic_writer.atomic_write_bytes

    def watched(path, payload, **kwargs):
        writers.append(threading.current_thread().name)
        return real(path, payload, **kwargs)

    async def burst() -> list[bool]:
        loop_thread = threading.current_thread().name
        atomic_writer.atomic_write_bytes = watched
        try:
            now = [atomic_writer.atomic_write_text_behind(target, f"v{n}") for n in range(20)]
            for _ in range(200):
                if target.exists() and target.read_text() == "v19":
                    break
                await asyncio.sleep(0.01)
        finally:
            atomic_writer.atomic_write_bytes = real
        assert loop_thread not in writers
        return now

    written_now = asyncio.run(burst())
    assert not any(written_now)
    assert target.read_text() == "v19"
    assert len(writers) < 20, "a burst of saves is coalesced"


def test_what_is_held_at_shutdown_is_written(tmp_path):
    target = tmp_path / "prefs.json"
    with atomic_writer._BEHIND_LOCK:
        atomic_writer._BEHIND[str(target)] = (b"held", True, False, 0o600)
    assert atomic_writer.flush_writes_behind() == 1
    assert target.read_text() == "held"


def test_an_fsync_on_the_loop_thread_is_named_by_its_caller(tmp_path):
    lockdep.reset_lockdep_for_test()

    async def save_on_the_loop() -> None:
        lockdep.note_event_loop_thread()
        atomic_writer.atomic_write_text(tmp_path / "x.json", "{}")

    try:
        asyncio.run(save_on_the_loop())
        report = lockdep.lockdep_report()
        places = list(report["blocking_on_loop"])
        assert places, "an fsync on the loop thread is reported"
        assert "save_on_the_loop" in places[0]
        assert "atomic_writer" not in places[0]
        # It is not a lock-order hazard, so it does not make lockdep unclean.
        assert report["clean"] and lockdep.lockdep_clean()
    finally:
        lockdep.reset_lockdep_for_test()


def test_off_the_loop_thread_nothing_is_reported(tmp_path):
    lockdep.reset_lockdep_for_test()
    try:
        lockdep.note_event_loop_thread()
        worker = threading.Thread(
            target=atomic_writer.atomic_write_text, args=(tmp_path / "y.json", "{}")
        )
        worker.start()
        worker.join(5)
        assert not lockdep.lockdep_report()["blocking_on_loop"]
    finally:
        lockdep.reset_lockdep_for_test()


def test_the_loop_thread_after_its_loop_has_stopped_is_not_reported(tmp_path):
    """An atexit save runs on the thread the loop ran on, with no loop."""
    lockdep.reset_lockdep_for_test()
    try:
        lockdep.note_event_loop_thread()
        atomic_writer.atomic_write_text(tmp_path / "z.json", "{}")
        assert not lockdep.lockdep_report()["blocking_on_loop"]
    finally:
        lockdep.reset_lockdep_for_test()


def test_a_save_the_lane_refuses_still_leaves_the_loop(tmp_path, monkeypatch):
    """LIVE 2026-09-21: two seconds before the shutdown record, resource
    stakes saved behind the loop, the lane refused the drain because the
    fence was set, and the fallback paid inline — an fsync on the loop."""
    from core.runtime import executors

    def refuse(*_args, **_kwargs):
        raise RuntimeError("blocking lane closed for shutdown")

    monkeypatch.setattr(executors, "submit_blocking_io", refuse)
    target = tmp_path / "state.json"
    wrote_on: list[str] = []
    real = atomic_writer.atomic_write_bytes

    def counting(path, payload, **kwargs):
        wrote_on.append(threading.current_thread().name)
        return real(path, payload, **kwargs)

    monkeypatch.setattr(atomic_writer, "atomic_write_bytes", counting)

    async def save():
        assert atomic_writer.atomic_write_text_behind(target, "held") is False
        return threading.current_thread().name

    loop_thread = asyncio.run(save())
    for thread in threading.enumerate():
        if thread.name.startswith("write_behind:"):
            thread.join(2.0)
    assert target.read_text() == "held"
    assert wrote_on and wrote_on[0] != loop_thread, "the write ran on the loop thread"

"""The runtime-settings refresh lane survives any one failed read.

Every permission is read from the snapshot this lane keeps; a snapshot nobody refreshes goes stale in five seconds and
reads every permission as denied. On 3 November a test that made Path.stat raise AssertionError killed the lane, and
every later test in its process read screen and microphone as switched off.
"""
from __future__ import annotations

import threading

import pytest

from core.runtime import runtime_settings

pytestmark = pytest.mark.unit


def test_an_unexpected_error_in_one_read_does_not_end_the_lane(monkeypatch):
    calls: list[int] = []
    second = threading.Event()
    # Its own stop and wakeup, so the lane this process already runs is not touched.
    stop, wakeup = threading.Event(), threading.Event()
    monkeypatch.setattr(runtime_settings, "_refresh_stop", stop)
    monkeypatch.setattr(runtime_settings, "_refresh_wakeup", wakeup)
    monkeypatch.setattr(runtime_settings, "_WATCH_INTERVAL_SECONDS", 0.01)

    def refresh() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise AssertionError("one bad read")
        second.set()
        stop.set()

    monkeypatch.setattr(runtime_settings, "_refresh_settings_from_disk", refresh)
    worker = threading.Thread(target=runtime_settings._refresh_worker, daemon=True)
    worker.start()
    try:
        assert second.wait(5.0), "the lane died on the first error"
    finally:
        stop.set()
        worker.join(2.0)
    assert len(calls) >= 2

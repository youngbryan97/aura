"""An overlapping status call is not evidence that she is unhealthy (third day run, 6 November 2026).

The recursion guard on the orchestrator's status was a flag on the instance. Her Skynet probe calls it from a worker
thread, so a probe that overlapped another status call was told {"healthy": False} with no reason, and logged "health
probe failed (1/2): health check reported unhealthy" six times; each next probe read her healthy.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from core.orchestrator.handlers.status_manager import StatusManagerMixin


class _Status:
    initialized = True
    running = True
    cycle_count = 3
    is_processing = False
    mode = "neutral"
    skills_loaded = 1

    def model_dump(self):
        return {"initialized": True, "running": True, "cycle_count": 3}


class _Orchestrator(StatusManagerMixin):
    def __init__(self, during_check=None):
        self.start_time = time.time()
        self.stats = {}
        self.status = _Status()
        self._during_check = during_check
        self._last_health_check_result = True
        self._last_health_reason = ""

    def health_check(self):
        if self._during_check:
            self._during_check(self)
        return True


def test_a_call_from_another_thread_during_a_status_call_reads_the_last_verdict():
    seen = {}

    def overlap(orch):
        if "report" in seen:
            return
        worker = threading.Thread(target=lambda: seen.setdefault("report", orch.get_status()))
        worker.start()
        worker.join(5)

    orch = _Orchestrator(during_check=overlap)
    report = orch.get_status()
    assert report["healthy"] is True
    assert seen["report"]["healthy"] is True
    assert seen["report"]["status_call"] == "overlapping_call"


def test_a_call_from_inside_a_status_call_on_the_same_thread_is_still_stopped():
    seen = {}

    def recurse(orch):
        if "report" not in seen:
            seen["report"] = orch.get_status()

    orch = _Orchestrator(during_check=recurse)
    orch._last_health_check_result = False
    orch._last_health_reason = "failed checks: error_rate"
    orch.get_status()
    assert seen["report"]["status_call"] == "recursive_depth_guard"
    assert seen["report"]["healthy"] is False
    assert seen["report"]["health_reason"] == "failed checks: error_rate"

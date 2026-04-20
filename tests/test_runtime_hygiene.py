import asyncio
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from core.resilience.stability_guardian import StabilityGuardian
from core.runtime import runtime_hygiene as runtime_hygiene_module
from core.runtime.runtime_hygiene import MemorySample, RuntimeHygieneManager
from core.utils.task_tracker import TaskTracker


@pytest.mark.asyncio
async def test_task_tracker_loop_hygiene_observes_raw_asyncio_tasks():
    tracker = TaskTracker(name="RuntimeHygieneTest")
    tracker.install_loop_hygiene(asyncio.get_running_loop())
    release = asyncio.Event()

    async def _hold():
        await release.wait()

    try:
        task = asyncio.create_task(_hold(), name="runtime_hygiene.implicit")
        await asyncio.sleep(0)

        stats = tracker.get_stats()
        assert stats["implicit_active"] >= 1
        assert getattr(task, "_aura_task_supervision", "") == "implicit"
        assert getattr(task, "_aura_task_tracker", "") == "RuntimeHygieneTest"
    finally:
        release.set()
        await asyncio.sleep(0)
        tracker.restore_loop_hygiene()


@pytest.mark.asyncio
async def test_runtime_hygiene_tracks_non_daemon_threads():
    hygiene = RuntimeHygieneManager()
    hygiene.stale_thread_age_s = 0.0
    release = threading.Event()

    def _worker():
        release.wait(0.5)

    await hygiene.start(asyncio.get_running_loop())
    try:
        thread = threading.Thread(target=_worker, name="runtime-hygiene-thread", daemon=False)
        thread.start()
        await asyncio.sleep(0.05)

        report = hygiene.audit()

        assert report["threads"]["active_non_daemon"] >= 1
        assert report["healthy"]
        assert report["threads"]["stale_non_daemon"] >= 1
    finally:
        release.set()
        thread.join(timeout=1.0)
        await hygiene.stop()
        hygiene.reset_state()


@pytest.mark.asyncio
async def test_runtime_hygiene_tracks_subprocesses():
    hygiene = RuntimeHygieneManager()
    await hygiene.start(asyncio.get_running_loop())
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.25)"])

    try:
        await asyncio.sleep(0.05)
        report = hygiene.audit()
        assert report["processes"]["active_registered"] >= 1
        assert report["processes"]["active_subprocesses"] >= 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        await hygiene.stop()
        hygiene.reset_state()


@pytest.mark.asyncio
async def test_runtime_hygiene_adopts_existing_subprocesses_started_before_hygiene():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.25)"])
    hygiene = RuntimeHygieneManager()
    await hygiene.start(asyncio.get_running_loop())

    try:
        await asyncio.sleep(0.05)
        report = hygiene.audit()
        assert report["processes"]["active_registered"] >= 1
        assert report["processes"]["rogue_child_processes"] == 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        await hygiene.stop()
        hygiene.reset_state()


def test_runtime_hygiene_skips_tracemalloc_by_default(monkeypatch):
    calls = []

    monkeypatch.delenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC", raising=False)
    monkeypatch.setattr(runtime_hygiene_module.tracemalloc, "is_tracing", lambda: False)
    monkeypatch.setattr(runtime_hygiene_module.tracemalloc, "start", lambda frames=1: calls.append(frames))

    hygiene = RuntimeHygieneManager()
    hygiene._start_tracemalloc()

    assert calls == []


def test_runtime_hygiene_can_opt_in_tracemalloc(monkeypatch):
    calls = []

    monkeypatch.setenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC", "1")
    monkeypatch.setenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC_FRAMES", "3")
    monkeypatch.setattr(runtime_hygiene_module.tracemalloc, "is_tracing", lambda: False)
    monkeypatch.setattr(runtime_hygiene_module.tracemalloc, "start", lambda frames=1: calls.append(frames))

    hygiene = RuntimeHygieneManager()
    hygiene._start_tracemalloc()

    assert calls == [3]


def test_runtime_hygiene_treats_active_model_growth_as_transient(monkeypatch):
    hygiene = RuntimeHygieneManager()
    now = time.monotonic()
    hygiene._samples.clear()

    for idx in range(hygiene.memory_growth_window):
        hygiene._samples.append(
            MemorySample(
                timestamp=now + idx,
                rss_bytes=int((100 + (idx * 35)) * 1024 * 1024),
                traced_bytes=0,
                task_count=0,
                thread_count=1,
                child_process_count=1,
            )
        )

    monkeypatch.setattr(
        hygiene,
        "_active_local_model_activity",
        lambda: ["Qwen2.5-32B-Instruct-8bit:warming"],
    )

    summary = hygiene._memory_summary()

    assert summary["sustained_growth"] is False
    assert summary["transient_growth"] is True
    assert "local model activity" in summary["message"].lower()


@pytest.mark.asyncio
async def test_stability_guardian_surfaces_runtime_hygiene_findings(service_container):
    service_container.register_instance(
        "runtime_hygiene",
        SimpleNamespace(
            audit=lambda: {
                "healthy": False,
                "critical": False,
                "issues": ["1 long-lived implicit task(s) still running"],
                "repair_actions": ["gc.collect()"],
            }
        ),
        required=False,
    )
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))

    result = await guardian._check_runtime_hygiene()

    assert result.healthy is False
    assert result.severity == "warning"
    assert "long-lived implicit task" in result.message
    assert result.action_taken == "gc.collect()"

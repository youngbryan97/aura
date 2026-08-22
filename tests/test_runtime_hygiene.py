import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.resilience.stability_guardian import HealthCheckResult, StabilityGuardian
from core.runtime import runtime_hygiene as runtime_hygiene_module
from core.runtime.runtime_hygiene import MemorySample, RuntimeHygieneManager
from core.utils.task_tracker import TaskTracker

TMP_ROOT = Path(tempfile.gettempdir())


@pytest.mark.asyncio
async def test_task_tracker_loop_hygiene_observes_raw_asyncio_tasks():
    tracker = TaskTracker(name="RuntimeHygieneTest")
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(None)
    tracker.install_loop_hygiene(loop)
    release = asyncio.Event()

    async def _hold():
        await release.wait()

    try:
        task = loop.create_task(_hold(), name="runtime_hygiene.implicit")
        await asyncio.sleep(0)

        stats = tracker.get_stats()
        assert stats["implicit_active"] >= 1
        assert getattr(task, "_aura_task_supervision", "") == "implicit"
        assert getattr(task, "_aura_task_tracker", "") == "RuntimeHygieneTest"
    finally:
        release.set()
        await asyncio.sleep(0)
        tracker.restore_loop_hygiene(loop)
        loop.set_task_factory(previous_factory)


@pytest.mark.asyncio
async def test_task_tracker_shutdown_cancels_protected_tasks():
    tracker = TaskTracker(name="ProtectedShutdownTest")
    cancelled: list[str] = []

    async def _hold(label: str):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(label)

    ordinary = tracker.create_task(_hold("ordinary"), name="ordinary")
    protected = tracker.create_task(_hold("protected"), name="protected")
    protected._aura_protected = True
    await asyncio.sleep(0)

    await tracker.shutdown(timeout=0.2)
    await asyncio.sleep(0)

    assert ordinary.cancelled()
    assert protected.cancelled()
    assert set(cancelled) == {"ordinary", "protected"}
    assert tracker.active_count == 0


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
    proc = await asyncio.to_thread(
        subprocess.Popen,
        [sys.executable, "-c", "import time; time.sleep(0.25)"],
    )

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
@pytest.mark.host_observation
async def test_runtime_hygiene_adopts_existing_subprocesses_started_before_hygiene():
    assert runtime_hygiene_module._HAS_PSUTIL, "psutil unavailable in this environment"
    try:
        runtime_hygiene_module.psutil.Process().children(recursive=True)
    except PermissionError as exc:
        pytest.fail(f"psutil child-process inspection is blocked: {exc}")

    proc = await asyncio.to_thread(
        subprocess.Popen,
        [sys.executable, "-c", "import time; time.sleep(1.0)"],
    )
    hygiene = RuntimeHygieneManager()
    await hygiene.start(asyncio.get_running_loop())

    try:
        report = {}
        for _ in range(20):
            await asyncio.sleep(0.05)
            report = hygiene.audit()
            if report["processes"]["active_registered"] >= 1:
                break
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



def _observe_children(resource_observer, specs):
    """Feed the process-wide simulated observer (autouse fixture) the child
    census. Since 71b5598f the hygiene census reads the canonical
    resource-observation seam; legacy ``._proc`` doubles are never consulted."""
    from core.runtime.resource_observation import ProcessObservation

    parent_pid = os.getpid()
    observations = []
    for spec in specs:
        ppid = spec.get("ppid", parent_pid)
        ancestors = spec.get("ancestor_pids")
        if ancestors is None:
            ancestors = (parent_pid,) if ppid != parent_pid else ()
        observations.append(
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=spec["pid"],
                ppid=ppid,
                create_time=1_700_000_000.0,
                status=spec.get("status", "sleeping"),
                name=spec.get("name", "Python"),
                cmdline=tuple(str(part) for part in spec.get("cmdline", ())),
                rss_bytes=1024,
                ancestor_pids=tuple(ancestors),
            )
        )
    resource_observer.configure_processes(observations)


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


def test_runtime_hygiene_treats_recent_model_warmup_as_transient(monkeypatch):
    hygiene = RuntimeHygieneManager()
    hygiene.model_activity_grace_s = 120.0
    now = time.time()

    fake_client = SimpleNamespace(
        get_lane_status=lambda: {
            "state": "ready",
            "warmup_in_flight": False,
            "current_request_started_at": 0.0,
            "last_ready_at": now - 10.0,
            "last_progress_at": now - 12.0,
            "last_transition_at": now - 15.0,
        }
    )
    fake_mlx_module = SimpleNamespace(_CLIENTS={str(TMP_ROOT / "cortex"): fake_client})
    monkeypatch.setitem(sys.modules, "core.brain.llm.mlx_client", fake_mlx_module)

    assert hygiene._active_local_model_activity() == ["cortex:recent"]


def test_runtime_hygiene_tolerates_model_registry_churn(monkeypatch):
    hygiene = RuntimeHygieneManager()
    now = time.time()

    fake_client = SimpleNamespace(
        get_lane_status=lambda: {
            "state": "ready",
            "warmup_in_flight": False,
            "current_request_started_at": 0.0,
            "last_ready_at": now - 10.0,
            "last_progress_at": now - 12.0,
            "last_transition_at": now - 15.0,
        }
    )

    class FlakyRegistry(dict):
        def __init__(self):
            super().__init__({str(TMP_ROOT / "cortex"): fake_client})
            self.calls = 0

        def items(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("dictionary changed size during iteration")
            return super().items()

    fake_mlx_module = SimpleNamespace(_CLIENTS=FlakyRegistry())
    monkeypatch.setitem(sys.modules, "core.brain.llm.mlx_client", fake_mlx_module)

    assert hygiene._active_local_model_activity() == ["cortex:recent"]


def test_runtime_hygiene_adopts_late_active_children_before_flagging_rogue_processes(resource_observer):
    _observe_children(resource_observer, [{
        "pid": 43210,
        "cmdline": [sys.executable, "-m", "multiprocessing.spawn"],
        "name": "spawned-child",
    }])
    hygiene = RuntimeHygieneManager()

    hygiene._adopt_active_child_processes()
    summary = hygiene._process_summary()

    assert summary["active_registered"] == 1
    assert summary["active_subprocesses"] == 1
    assert summary["rogue_child_processes"] == 0


def test_runtime_hygiene_explicit_process_owner_registration_deduplicates_by_pid():
    class _OwnedProc:
        pid = 43212
        name = "MLXWorker-test"

        def is_alive(self):
            return True

    hygiene = RuntimeHygieneManager()
    proc = _OwnedProc()

    hygiene.register_process_handle(
        proc,
        kind="multiprocessing",
        name="MLXWorker-test",
        source="test.worker_owner",
        command="MLX worker for test",
    )
    hygiene.register_process_handle(
        proc,
        kind="multiprocessing",
        name="MLXWorker-test",
        source="test.worker_owner",
        command="MLX worker for test",
    )

    summary = hygiene._process_summary()

    assert summary["active_registered"] == 1
    assert summary["active_multiprocessing"] == 1


def test_runtime_hygiene_marks_asyncio_subprocess_finished_from_returncode():
    proc = SimpleNamespace(pid=43213, returncode=None)
    hygiene = RuntimeHygieneManager()
    hygiene.register_process_handle(
        proc,
        kind="subprocess",
        name="asyncio-subprocess-test",
        source="test.asyncio_subprocess",
        command="python -c pass",
    )

    summary = hygiene._process_summary()
    assert summary["active_registered"] == 1
    assert summary["active_subprocesses"] == 1

    proc.returncode = 0
    hygiene._refresh_process_records()
    summary = hygiene._process_summary()

    assert summary["active_registered"] == 0
    assert summary["active_subprocesses"] == 0


def test_runtime_hygiene_retires_an_owner_proven_dead_process_before_handle_close():
    class _OwnedProc:
        pid = 43214
        name = "MLXWorker-retired"
        exitcode = -9

        def is_alive(self):
            return False

    hygiene = RuntimeHygieneManager()
    proc = _OwnedProc()
    hygiene.register_process_handle(
        proc,
        kind="multiprocessing",
        name=proc.name,
        source="test.worker_owner",
    )

    assert hygiene.retire_process_handle(proc) is True
    assert hygiene.process_handle_is_registered(proc) is False
    assert id(proc) not in hygiene._process_refs
    assert hygiene._process_records[id(proc)].finished_at is not None
    assert hygiene._process_records[id(proc)].exit_code == -9


def test_runtime_hygiene_process_iter_system_error_is_nonfatal(monkeypatch):
    hygiene = RuntimeHygieneManager()
    hygiene._proc = SimpleNamespace(children=lambda recursive=True: [])
    if not runtime_hygiene_module._HAS_PSUTIL:
        hygiene._adopt_active_child_processes()
        assert hygiene._process_records == {}
        return

    monkeypatch.setattr(
        runtime_hygiene_module.psutil,
        "process_iter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SystemError("proc_cmdline permission wrapper failed")
        ),
    )

    hygiene._adopt_active_child_processes()

    assert hygiene._process_records == {}


def test_runtime_hygiene_classifies_registered_worker_descendants_as_owned(resource_observer):
    _observe_children(resource_observer, [
        {
            "pid": 61001,
            "ppid": 999,
            "name": "mlx-worker",
            "cmdline": [sys.executable, "-m", "mlx-worker"],
        },
        {
            "pid": 61002,
            "ppid": 61001,
            "name": "mlx-helper",
            "cmdline": [sys.executable, "-m", "mlx-helper"],
            "ancestor_pids": (os.getpid(), 61001),
        },
    ])
    hygiene = RuntimeHygieneManager()
    hygiene.register_process_handle(
        SimpleNamespace(pid=61001),
        kind="multiprocessing",
        name="mlx-worker",
        source="test.worker_owner",
        command="MLX worker",
    )

    summary = hygiene._process_summary()

    assert summary["active_registered"] == 1
    assert summary["owned_descendant_processes"] == 1
    assert summary["rogue_child_processes"] == 0
    assert summary["rogue_samples"] == []


def test_runtime_hygiene_adopts_direct_multiprocessing_spawn_during_summary(resource_observer):
    _observe_children(resource_observer, [{
        "pid": 62001,
        "cmdline": [
            sys.executable,
            "-c",
            "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=8, pipe_handle=20)",
            "--multiprocessing-fork",
        ],
    }])
    hygiene = RuntimeHygieneManager()

    summary = hygiene._process_summary()

    assert summary["active_registered"] == 1
    assert summary["active_multiprocessing"] == 1
    assert summary["rogue_child_processes"] == 0


def test_runtime_hygiene_adopts_python312_spawn_main_without_fork_flag(resource_observer):
    _observe_children(resource_observer, [{
        "pid": 62002,
        "cmdline": [
            sys.executable,
            "-c",
            "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=8, pipe_handle=20)",
        ],
    }])
    hygiene = RuntimeHygieneManager()

    summary = hygiene._process_summary()

    assert summary["active_registered"] == 1
    assert summary["active_multiprocessing"] == 1
    assert summary["rogue_child_processes"] == 0
    assert summary["rogue_samples"] == []


def test_runtime_hygiene_keeps_unowned_child_process_fail_closed(resource_observer):
    _observe_children(resource_observer, [{
        "pid": 62002,
        "ppid": 999,
        "cmdline": [sys.executable, "-m", "unexpected_worker"],
        "name": "unexpected-worker",
    }])
    hygiene = RuntimeHygieneManager()

    summary = hygiene._process_summary()

    assert summary["owned_descendant_processes"] == 0
    assert summary["rogue_child_processes"] == 1
    assert summary["rogue_samples"][0]["pid"] == 62002
    assert "unexpected-worker" in summary["rogue_samples"][0]["name"]


def test_runtime_hygiene_drops_completed_children_from_cached_process_table(
    resource_observer,
    monkeypatch,
):
    from core.runtime.resource_observation import ProcessIdsObservation

    _observe_children(resource_observer, [{
        "pid": 62004,
        "cmdline": ["git", "--no-optional-locks", "status", "--porcelain=v1"],
        "name": "git",
    }])
    monkeypatch.setattr(
        resource_observer,
        "process_ids",
        lambda: ProcessIdsObservation(
            provenance=resource_observer.provenance,
            pids=(),
        ),
    )
    hygiene = RuntimeHygieneManager()

    summary = hygiene._process_summary()

    assert summary["rogue_child_processes"] == 0
    assert summary["rogue_samples"] == []


def test_runtime_hygiene_audit_does_not_auto_adopt_unknown_late_child(
    resource_observer,
):
    _observe_children(resource_observer, [{
        "pid": 62003,
        "ppid": os.getpid(),
        "cmdline": [sys.executable, "-m", "unexpected_late_worker"],
        "name": "unexpected-late-worker",
    }])
    hygiene = RuntimeHygieneManager()

    report = hygiene.audit()

    assert report["healthy"] is False
    assert report["critical"] is True
    assert report["processes"]["active_registered"] == 0
    assert report["processes"]["rogue_child_processes"] == 1
    assert report["processes"]["rogue_samples"][0]["pid"] == 62003


@pytest.mark.asyncio
async def test_runtime_hygiene_ignores_python_resource_tracker_children():
    class _ResourceTrackerProc:
        pid = 43211

        def __init__(self):
            self.terminated = False
            self.killed = False
            self.waited = False

        def cmdline(self):
            return [
                sys.executable,
                "-c",
                "from multiprocessing.resource_tracker import main;main(11)",
            ]

        def name(self):
            return "Python"

        def is_running(self):
            return True

        def status(self):
            return "sleeping"

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True

        def kill(self):
            self.killed = True

    child = _ResourceTrackerProc()
    hygiene = RuntimeHygieneManager()
    hygiene._proc = SimpleNamespace(children=lambda recursive=True: [child])

    hygiene._adopt_active_child_processes()
    summary = hygiene._process_summary()
    await hygiene._cleanup_child_processes()

    assert summary["active_registered"] == 0
    assert summary["rogue_child_processes"] == 0
    assert child.terminated is False
    assert child.waited is False
    assert child.killed is False


def test_runtime_hygiene_thread_join_helper_skips_current_thread():
    current = threading.current_thread()

    RuntimeHygieneManager._join_thread_if_not_current(current, 0.01)


def test_runtime_hygiene_shutdown_thread_join_env_is_bounded(monkeypatch):
    monkeypatch.setenv("AURA_RUNTIME_HYGIENE_MAX_SHUTDOWN_THREAD_JOINS", "not-an-int")

    hygiene = RuntimeHygieneManager()

    assert hygiene.max_thread_joins_per_shutdown == 16


@pytest.mark.asyncio
async def test_runtime_hygiene_shutdown_thread_join_is_bounded(monkeypatch):
    recorded = []

    def fake_record_degradation(subsystem, error, **kwargs):
        recorded.append((subsystem, error, kwargs))

    monkeypatch.setattr(runtime_hygiene_module, "record_degradation", fake_record_degradation)

    class FakeThread:
        daemon = False

        def __init__(self, idx: int):
            self.ident = 10_000 + idx
            self.name = f"fake-thread-{idx}"
            self.joined = False

        def is_alive(self):
            return True

        def join(self, timeout=None):
            del timeout
            self.joined = True

    hygiene = RuntimeHygieneManager()
    hygiene.max_thread_joins_per_shutdown = 2
    threads = [FakeThread(idx) for idx in range(5)]
    hygiene._thread_refs = {idx: thread for idx, thread in enumerate(threads)}

    await hygiene._join_non_daemon_threads()

    assert [thread.joined for thread in threads] == [True, True, False, False, False]
    assert recorded
    subsystem, error, kwargs = recorded[0]
    assert subsystem == "runtime_hygiene_shutdown"
    assert "left for owner shutdown" in str(error)
    assert kwargs["severity"] == "warning"
    assert kwargs["enforce_failure_policy"] is False
    assert kwargs["extra"]["skipped_count"] == 3


@pytest.mark.asyncio
async def test_runtime_hygiene_shutdown_thread_join_errors_do_not_fail_closed(monkeypatch):
    recorded = []

    def fake_record_degradation(subsystem, error, **kwargs):
        recorded.append((subsystem, type(error).__name__, kwargs))

    monkeypatch.setattr(runtime_hygiene_module, "record_degradation", fake_record_degradation)

    class FailingThread:
        daemon = False
        ident = 20_000
        name = "failing-thread"

        def is_alive(self):
            return True

        def join(self, timeout=None):
            del timeout
            raise RuntimeError("join failed")

    hygiene = RuntimeHygieneManager()
    hygiene.max_thread_joins_per_shutdown = 1
    hygiene._thread_refs = {1: FailingThread()}

    await hygiene._join_non_daemon_threads()

    assert recorded == [
        (
            "runtime_hygiene_shutdown",
            "RuntimeError",
            {
                "severity": "warning",
                "action": "continued shutdown after a bounded thread join failed",
                "enforce_failure_policy": False,
            },
        )
    ]


@pytest.mark.asyncio
async def test_runtime_hygiene_child_cleanup_is_concurrent():
    class SlowTerminatingProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            time.sleep(0.15)
            raise subprocess.TimeoutExpired(cmd="slow", timeout=timeout or 0.0)

        def kill(self):
            self.killed = True

    hygiene = RuntimeHygieneManager()
    hygiene.process_shutdown_timeout_s = 0.2
    processes = [SlowTerminatingProcess(), SlowTerminatingProcess(), SlowTerminatingProcess()]
    hygiene._process_refs = {idx: proc for idx, proc in enumerate(processes)}

    started = time.monotonic()
    await hygiene._cleanup_child_processes()
    elapsed = time.monotonic() - started

    assert all(proc.terminated for proc in processes)
    assert all(proc.killed for proc in processes)
    assert elapsed < 0.6


@pytest.mark.asyncio
async def test_runtime_hygiene_cleans_adopted_psutil_children(monkeypatch, resource_observer):
    class PsutilChild:
        pid = 54321

        def __init__(self):
            self.terminated = False
            self.killed = False
            self.running = True

        def cmdline(self):
            return [sys.executable, "-m", "multiprocessing.spawn"]

        def name(self):
            return "spawned-child"

        def is_running(self):
            return self.running

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.running = False

        def kill(self):
            self.killed = True
            self.running = False

    child = PsutilChild()
    monkeypatch.setattr(runtime_hygiene_module, "_HAS_PSUTIL", True)

    _observe_children(resource_observer, [{
        "pid": 54321,
        "cmdline": [sys.executable, "-m", "multiprocessing.spawn"],
        "name": "spawned-child",
    }])
    hygiene = RuntimeHygieneManager()
    hygiene.process_shutdown_timeout_s = 0.2

    hygiene._adopt_active_child_processes()
    # Under a simulated observer adoption records the census without live
    # psutil handles; seed the ref exactly as host-mode adoption would so
    # the CLEANUP semantics under test stay honest.
    for key, record in hygiene._process_records.items():
        if record.pid == 54321:
            hygiene._process_refs[key] = child
    await hygiene._cleanup_child_processes()

    assert child.terminated
    assert child.running is False


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


@pytest.mark.asyncio
async def test_stability_guardian_rejects_runtime_hygiene_report_without_health_evidence(service_container):
    service_container.register_instance(
        "runtime_hygiene",
        SimpleNamespace(
            audit=lambda: {
                "critical": False,
                "issues": ["runtime hygiene did not emit healthy"],
                "repair_actions": [],
            }
        ),
        required=False,
    )
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))

    result = await guardian._check_runtime_hygiene()

    assert result.healthy is False
    assert result.severity == "warning"
    assert "runtime hygiene did not emit healthy" in result.message


def test_stability_guardian_treats_slow_user_facing_ticks_as_unhealthy():
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))
    now = time.time()

    for _ in range(5):
        guardian.record_tick_health(
            SimpleNamespace(
                tick_duration_ms=22000.0,
                origin="user",
                priority=True,
                is_user_facing=True,
            )
        )
    guardian._loop_lag_samples.append((now, 40.0))

    result = guardian._check_tick_rate()

    assert result.healthy is False
    assert result.severity == "warning"
    assert "Foreground turns are slow" in result.message
    assert "withhold healthy status" in (result.action_taken or "")


def test_stability_guardian_attributes_sustained_background_tick_latency():
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))

    for _ in range(3):
        guardian.record_tick_health(
            SimpleNamespace(
                tick_duration_ms=8200.0,
                origin="system",
                priority=False,
                is_user_facing=False,
                phase_durations_ms={"PhiConsciousnessPhase": 6100.0, "LearningPhase": 90.0},
            )
        )

    result = guardian._check_tick_rate()

    assert result.healthy is False
    assert "PhiConsciousnessPhase" in result.message
    assert "6100ms" in result.message


@pytest.mark.asyncio
async def test_stability_guardian_overall_health_false_for_slow_foreground(monkeypatch):
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))

    for _ in range(4):
        guardian.record_tick_health(
            SimpleNamespace(
                tick_duration_ms=24000.0,
                origin="user",
                priority=True,
                is_user_facing=True,
            )
        )

    def _healthy(name):
        return HealthCheckResult(name, True, "ok")

    for attr in (
        "_check_memory",
        "_check_asyncio_tasks",
        "_check_lock_watchdog",
        "_check_state_integrity",
        "_check_state_repository_pressure",
        "_check_llm_circuit",
        "_check_db_connections",
        "_check_backup_maintenance",
        "_check_runtime_hygiene",
        "_check_background_tasks",
    ):
        monkeypatch.setattr(guardian, attr, lambda attr=attr: _healthy(attr))

    report = await guardian.run_checks()

    assert report.overall_healthy is False
    assert any(check.name == "tick_rate" and check.healthy is False for check in report.checks)


def test_stability_guardian_flags_actual_event_loop_lag():
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time() - 1000.0))
    now = time.time()
    guardian.record_tick_health(
        SimpleNamespace(
            tick_duration_ms=450.0,
            origin="system",
            priority=False,
            is_user_facing=False,
        )
    )
    guardian._loop_lag_samples.append((now, guardian.MAX_EVENT_LOOP_LAG_MS + 250.0))

    result = guardian._check_tick_rate()

    assert result.healthy is False
    assert result.severity == "warning"
    assert "Event loop lag is elevated" in result.message


def test_stability_guardian_treats_stale_event_loop_lag_as_info():
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))
    guardian.record_tick_health(
        SimpleNamespace(
            tick_duration_ms=450.0,
            origin="system",
            priority=False,
            is_user_facing=False,
        )
    )
    guardian._loop_lag_samples.append(
        (
            time.time() - (guardian.EVENT_LOOP_LAG_WINDOW_S + 5.0),
            guardian.MAX_EVENT_LOOP_LAG_MS + 300.0,
        )
    )

    result = guardian._check_tick_rate()

    assert result.healthy is True
    assert result.severity in {"info", "warning"}
    assert "tick health ok" in result.message.lower()


@pytest.mark.asyncio
async def test_stability_guardian_restarts_missing_research_cycle_after_boot_grace(service_container):
    restarted = asyncio.Event()

    class ResearchCycleProbe:
        async def restart_async(self):
            restarted.set()

    service_container.register_instance(
        "research_cycle",
        ResearchCycleProbe(),
        required=True,
        failure_policy="degrade_with_receipt",
    )
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time() - 500.0))

    result = await guardian._check_background_tasks()
    await asyncio.sleep(0)

    assert result.healthy is False
    assert "aura.research_cycle" in result.message
    assert result.action_taken == "research_cycle.restart_async() triggered"
    assert restarted.is_set() is True


@pytest.mark.asyncio
async def test_stability_guardian_allows_research_cycle_boot_grace():
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time() - 20.0))

    result = await guardian._check_background_tasks()

    assert result.healthy is True


@pytest.mark.asyncio
async def test_finished_subprocess_refs_are_released_not_retained_for_life():
    """The registry must not BE the leak it exists to find.

    _process_refs used to hold a strong reference to every Popen ever
    created until shutdown, pinning each one's stdout/stderr wrappers and
    pipe buffers for process lifetime — the dominant cluster in the Jul 7
    soak's tracemalloc top-growth (longevity_leakrepro: 16k live io.open +
    21k TextIOWrapper objects). Finished procs must be released while
    their small records stay for reporting.
    """
    hygiene = RuntimeHygieneManager()
    await hygiene.start(asyncio.get_running_loop())
    proc = await asyncio.to_thread(
        subprocess.Popen, [sys.executable, "-c", "pass"],
    )
    try:
        await asyncio.to_thread(proc.wait, 5.0)
        hygiene._refresh_process_records()
        assert id(proc) not in hygiene._process_refs, (
            "a finished subprocess must not stay strongly referenced"
        )
        record = hygiene._process_records.get(id(proc))
        assert record is not None and record.finished_at is not None, (
            "the bounded post-mortem record should remain"
        )
    finally:
        await hygiene.stop()
        hygiene.reset_state()


@pytest.mark.asyncio
async def test_finished_records_are_bounded_not_unbounded():
    hygiene = RuntimeHygieneManager()
    await hygiene.start(asyncio.get_running_loop())
    try:
        retention = hygiene._FINISHED_RECORD_RETENTION
        for index in range(retention + 40):
            key = 10_000_000 + index
            hygiene._process_records[key] = runtime_hygiene_module.ProcessRecord(
                key=key, kind="subprocess", name=f"fake-{index}",
                source="test", command="true", pid=None,
            )
            hygiene._process_records[key].finished_at = float(index)
            hygiene._process_refs[key] = SimpleNamespace(poll=lambda: 0)
        hygiene._refresh_process_records()
        finished = [
            record for record in hygiene._process_records.values()
            if record.finished_at is not None
        ]
        assert len(finished) <= retention
        # Oldest finished entries are the ones dropped.
        assert all(
            record.finished_at >= 40.0
            for record in finished
            if record.name.startswith("fake-")
        )
    finally:
        await hygiene.stop()
        hygiene.reset_state()


@pytest.mark.asyncio
async def test_finished_thread_refs_are_released():
    hygiene = RuntimeHygieneManager()
    await hygiene.start(asyncio.get_running_loop())
    worker = threading.Thread(target=lambda: None, name="hygiene-evict-probe")
    worker.start()
    worker.join(timeout=5.0)
    try:
        key = id(worker)
        if key in hygiene._thread_refs:
            hygiene._refresh_thread_records()
            assert key not in hygiene._thread_refs, (
                "a finished thread must not stay strongly referenced"
            )
    finally:
        await hygiene.stop()
        hygiene.reset_state()


@pytest.mark.asyncio
async def test_finetune_pipe_flush_writes_inside_a_governed_scope(tmp_path, monkeypatch):
    """LRP-008: the live runtime refused every finetune-pipe flush as a
    governance violation ("append_text:adaptation.finetune_pipe.dataset
    called outside governed context", x12 Jul 18) and silently dropped
    the learning traces. The write must carry an active governance token
    in the thread that performs it."""
    from core.adaptation import finetune_pipe as pipe_module
    from core.governance_context import get_active_governance

    observed: dict = {}

    class _GatewayStub:
        def append_text(self, path, payload, source=""):
            observed["governed"] = get_active_governance() is not None
            observed["source"] = source
            Path(str(path)).write_text(payload)

        def write_text(self, path, payload, source=""):
            observed["rotate_governed"] = get_active_governance() is not None
            Path(str(path)).write_text(payload)

    monkeypatch.setattr(pipe_module, "get_file_write_gateway", lambda: _GatewayStub())
    pipe = pipe_module.FinetunePipe.__new__(pipe_module.FinetunePipe)
    pipe._batch = [{"text": "sample"}]
    pipe._flush_lock = asyncio.Lock()
    pipe.dataset_path = tmp_path / "dataset.jsonl"

    await pipe.flush()

    assert observed.get("source") == "adaptation.finetune_pipe.dataset"
    assert observed.get("governed") is True, (
        "the dataset append must run inside local_internal_governed_scope"
    )

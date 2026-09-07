from __future__ import annotations

import asyncio
import gc
import inspect
import logging
import multiprocessing as mp
import os
import stat
import subprocess
import sys
import threading
import time
import traceback
import tracemalloc
import warnings
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.resource_observation import (
    ResourceObserver,
    get_resource_observer,
)
from core.runtime.shutdown_execution import run_sync_shutdown_callable
from core.utils.task_tracker import (
    begin_shutdown_task_creation_scope,
    end_shutdown_task_creation_scope,
    get_task_tracker,
    shutdown_resource_creation_allowed,
)

try:
    import psutil

    _HAS_PSUTIL = True
    _PSUTIL_PROCESS_ERRORS = (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        psutil.ZombieProcess,
        psutil.Error,
    )
except ImportError:
    _HAS_PSUTIL = False
    _PSUTIL_PROCESS_ERRORS = ()

logger = logging.getLogger("Aura.RuntimeHygiene")
_PROCESS_INTROSPECTION_ERRORS = (
    RuntimeError,
    SystemError,
    AttributeError,
    TypeError,
    ValueError,
    OSError,
) + _PSUTIL_PROCESS_ERRORS
_THREAD_RUN_FAILURES = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    SystemError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _snapshot_mapping_items(mapping: Any) -> list[tuple[Any, Any]]:
    """Return a bounded snapshot of a live mapping without crashing on churn."""
    if not mapping:
        return []
    last_error: RuntimeError | None = None
    for _attempt in range(3):
        try:
            return list(mapping.items())
        except RuntimeError as exc:
            if "changed size" not in str(exc):
                raise
            last_error = exc
            time.sleep(0)
    logger.debug("RuntimeHygiene: skipped mutating registry snapshot: %s", last_error)
    return []


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _declared_float(name: str, default: float, description: str) -> float:
    """Read a float knob through the typed flag layer (C1 discipline)."""
    try:
        return float(
            declare(
                name,
                kind=FlagKind.FLOAT,
                default=default,
                description=description,
                owner="core.runtime.runtime_hygiene",
            ).value()
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return default


def _process_cmdline(proc: Any) -> list[str]:
    try:
        value = getattr(proc, "cmdline", ())
        raw = value() if callable(value) else value
        return [str(part) for part in (raw or [])]
    except _PROCESS_INTROSPECTION_ERRORS:
        return []


def _process_name(proc: Any) -> str:
    try:
        value = getattr(proc, "name", "")
        raw = value() if callable(value) else value
        return str(raw or "")
    except _PROCESS_INTROSPECTION_ERRORS:
        return ""


def _is_python_resource_tracker_process(proc: Any) -> bool:
    """Return true for Python's internal multiprocessing tracker processes.

    The resource tracker owns semaphore/shared-memory bookkeeping for the
    current interpreter. Terminating it during runtime cleanup causes noisy
    relaunches and can corrupt its unregister cache, so it is observed but not
    adopted, flagged as rogue, or force-reaped by Aura cleanup.
    """

    name = _process_name(proc).lower()
    cmdline = " ".join(_process_cmdline(proc)).lower()
    return (
        name in {"resource_tracker", "semaphore_tracker"}
        or "multiprocessing.resource_tracker" in cmdline
        or "multiprocessing.semaphore_tracker" in cmdline
    )


def _is_keep_awake_assertion_process(proc: Any) -> bool:
    """Return true for the macOS sleep assertion Aura spawns for itself.

    ``core.runtime.keep_awake`` starts ``caffeinate`` through the subprocess
    gateway and owns its lifecycle, so it is never a rogue child. Matching the
    binary AND its assertion flags keeps this from adopting an unrelated
    ``caffeinate`` a user happened to start.
    """

    name = _process_name(proc).lower()
    cmdline = " ".join(_process_cmdline(proc)).lower()
    if "caffeinate" not in name and "caffeinate" not in cmdline:
        return False
    return "-i" in _process_cmdline(proc) or "-m" in _process_cmdline(proc)


def _is_governed_applescript_process(proc: Any) -> bool:
    """Return true for the AppleScript helper Aura runs through its gateway.

    LIVE DEFECT, 2026-08-03 19:43. StabilityGuardian reported DEGRADED with
    "1 unregistered child process(es) detected; pid=30897 name=osascript".
    That osascript is Aura's own: every AppleScript in the desktop and
    web-interlocutor paths goes through
    ``core.runtime.desktop_action_gateway`` as a short-lived direct child. It
    is named for the macOS binary, matched none of the Aura worker tags, and
    so was reported as rogue — the same shape as the keep-awake assertion
    above, which produced a permanent DEGRADED card for a process the runtime
    deliberately started.

    Matched only as a DIRECT child, which the caller has already established,
    so an osascript the user started elsewhere is never adopted.
    """

    name = _process_name(proc).lower()
    cmdline = " ".join(_process_cmdline(proc)).lower()
    return name == "osascript" or cmdline.startswith("osascript")


def _is_python_multiprocessing_spawn_process(proc: Any) -> bool:
    """Return true for Python multiprocessing worker children owned by this runtime.

    Workers can appear between the adoption pass and the child-process summary
    scan. They are still Aura-owned if they are direct Python multiprocessing
    spawn children; adopt them instead of reporting a transient rogue child.
    """

    cmdline = " ".join(_process_cmdline(proc)).lower()
    has_spawn_module = "multiprocessing.spawn" in cmdline
    has_spawn_main = "spawn_main" in cmdline
    return bool(has_spawn_module and (has_spawn_main or "--multiprocessing-fork" in cmdline))


def _process_pid(proc: Any) -> int:
    try:
        return int(getattr(proc, "pid", 0) or 0)
    except _PROCESS_INTROSPECTION_ERRORS:
        return 0


def _process_ppid(proc: Any) -> int:
    try:
        value = getattr(proc, "ppid", 0)
        if value:
            raw = value() if callable(value) else value
            return int(raw or 0)
    except _PROCESS_INTROSPECTION_ERRORS:
        return 0
    info = getattr(proc, "info", None) or {}
    try:
        return int(info.get("ppid") or 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class MemorySample:
    timestamp: float
    rss_bytes: int
    traced_bytes: int
    task_count: int
    thread_count: int
    child_process_count: int
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""


@dataclass
class ThreadRecord:
    key: int
    name: str
    daemon: bool
    source: str
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    ident: int | None = None
    exception: str | None = None

    def age_s(self, now: float | None = None) -> float:
        current_time = now if now is not None else time.monotonic()
        origin = self.started_at or self.created_at
        return max(0.0, current_time - origin)


@dataclass
class ProcessRecord:
    key: int
    kind: str
    name: str
    source: str
    command: str
    created_at: float = field(default_factory=time.monotonic)
    pid: int | None = None
    exit_code: int | None = None
    finished_at: float | None = None
    successor_of_pid: int | None = None

    def age_s(self, now: float | None = None) -> float:
        current_time = now if now is not None else time.monotonic()
        return max(0.0, current_time - self.created_at)


@dataclass
class ShutdownResourceRecord:
    key: int
    kind: str
    name: str
    source: str
    resource: Any
    closer: Callable[[], Any] | None
    timeout_s: float
    required: bool
    blocking: bool
    sequence: int
    registered_at: float = field(default_factory=time.monotonic)
    status: str = "registered"
    duration_seconds: float | None = None
    error: str | None = None
    crossed_shutdown: bool = False


class RuntimeHygieneManager:
    """Tracks tasks, threads, child processes, and memory growth across the runtime."""

    def __init__(self, *, observer: ResourceObserver | None = None):
        self._observer = observer
        self._running = False
        self._thread_records: dict[int, ThreadRecord] = {}
        self._thread_refs: dict[int, threading.Thread] = {}
        self._process_records: dict[int, ProcessRecord] = {}
        self._process_refs: dict[int, Any] = {}
        self._resource_records: dict[int, ShutdownResourceRecord] = {}
        self._resource_sequence = 0
        self._resource_lock = threading.RLock()
        self._resource_admission_closed = False
        self._samples: deque[MemorySample] = deque(maxlen=36)
        self._task_tracker = get_task_tracker()
        self._last_gc_at = 0.0

        self.memory_growth_window = 6
        self.memory_growth_min_delta_mb = 128.0
        self.memory_growth_ratio = 0.12
        self.model_activity_grace_s = max(
            0.0,
            float(os.getenv("AURA_RUNTIME_HYGIENE_MODEL_GRACE_S", "120") or 120.0),
        )
        self.stale_thread_age_s = 900.0
        self.stale_task_age_s = 900.0
        self.process_shutdown_timeout_s = 1.0
        self.thread_join_timeout_s = 0.2
        self.max_thread_joins_per_shutdown = _env_int(
            "AURA_RUNTIME_HYGIENE_MAX_SHUTDOWN_THREAD_JOINS",
            16,
            low=1,
            high=256,
        )
        self.shutdown_timeout_s = max(
            1.5,
            float(os.getenv("AURA_RUNTIME_HYGIENE_SHUTDOWN_TIMEOUT_S", "4.0") or 4.0),
        )
        self.resource_shutdown_timeout_s = max(
            0.5, _declared_float("AURA_RUNTIME_RESOURCE_SHUTDOWN_TIMEOUT_S", 3.0,
                                 "Per-resource close budget during shutdown hygiene")
        )
        self.task_shutdown_timeout_s = max(
            0.5, _declared_float("AURA_RUNTIME_TASK_SHUTDOWN_TIMEOUT_S", 3.0,
                                 "Per-task cancel budget during shutdown hygiene")
        )
        self.executor_shutdown_timeout_s = max(
            0.25,
            _declared_float(
                "AURA_RUNTIME_EXECUTOR_SHUTDOWN_TIMEOUT_S",
                2.0,
                "Default-executor join budget during final shutdown hygiene",
            ),
        )
        self.tracemalloc_enabled = str(
            os.getenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.tracemalloc_frames = max(
            1,
            int(os.getenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC_FRAMES", "1") or 1),
        )
        self._tracemalloc_started_by_hygiene = False
        self._tracemalloc_baseline: Any = None
        self._tracemalloc_baseline_at: float = 0.0
        self._stop_lock = asyncio.Lock()
        self._shutdown_started = False
        self._shutdown_complete = False
        self._last_shutdown_report: dict[str, Any] | None = None

        self._original_thread_start = None
        self._original_popen_init = None
        self._original_mp_start = None
        self._original_new_event_loop = None

        self._proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None

    @property
    def resource_observer(self) -> ResourceObserver:
        return self._observer or get_resource_observer()

    async def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        try:
            from core.runtime.shutdown_coordinator import is_shutdown_requested

            if is_shutdown_requested():
                raise RuntimeError("runtime_shutdown")
        except ImportError:
            pass
        if self._running:
            target_loop = loop
            if target_loop is not None:
                self._task_tracker.install_loop_hygiene(target_loop)
            return

        self._running = True
        target_loop = loop or asyncio.get_running_loop()
        self._task_tracker.install_loop_hygiene(target_loop)
        self._patch_asyncio_new_event_loop()
        self._patch_threading()
        self._patch_subprocess()
        self._patch_multiprocessing()
        self._start_tracemalloc()
        self._adopt_active_child_processes()
        self.capture_sample()

    async def stop(self) -> dict[str, Any]:
        """Run one bounded final sweep and replay its evidence to later callers."""

        async with self._stop_lock:
            if self._shutdown_complete and self._last_shutdown_report is not None:
                return dict(self._last_shutdown_report)
            scope_token = begin_shutdown_task_creation_scope()
            try:
                return await self._execute_stop_sweep()
            finally:
                end_shutdown_task_creation_scope(scope_token)

    async def _execute_stop_sweep(self) -> dict[str, Any]:
        started = time.monotonic()
        self._shutdown_started = True
        self._running = False
        self._adopt_active_child_processes()
        self._refresh_thread_records()
        self._refresh_process_records()
        before = {
            "tasks": self._task_tracker.get_stats(),
            "threads": self._thread_summary(include_stacks=True),
            "processes": self._process_summary(),
            "resources": self._resource_summary(),
            "native_resources": self._native_resource_summary(),
        }

        resource_report = await self._cleanup_shutdown_resources()
        await self._cleanup_child_processes()
        await self._join_non_daemon_threads()
        try:
            from core.runtime.shutdown_coordinator import is_shutdown_requested

            shutdown_latched = is_shutdown_requested()
        except (ImportError, RuntimeError, AttributeError):
            shutdown_latched = False
        if shutdown_latched:
            task_report = await self._task_tracker.shutdown(
                timeout=self.task_shutdown_timeout_s
            )
        else:
            task_report = {
                "clean": True,
                "skipped": True,
                "reason": "runtime_shutdown_not_requested",
                "remaining": 0,
            }

        self._task_tracker.restore_loop_hygiene()
        self._restore_patches()
        executor_report = await self._shutdown_default_executor(
            asyncio.get_running_loop(),
            enabled=shutdown_latched,
        )
        if self._tracemalloc_started_by_hygiene and tracemalloc.is_tracing():
            try:
                tracemalloc.stop()
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('runtime_hygiene', exc)
                logger.debug("RuntimeHygiene: tracemalloc stop failed: %s", exc)
            finally:
                self._tracemalloc_started_by_hygiene = False

        self._adopt_active_child_processes()
        self._refresh_thread_records()
        self._refresh_process_records()
        after = {
            "tasks": self._task_tracker.get_stats(),
            "threads": self._thread_summary(include_stacks=True),
            "processes": self._process_summary(),
            "resources": self._resource_summary(),
            "native_resources": self._native_resource_summary(),
        }
        blockers: list[str] = []
        if task_report.get("clean") is not True:
            blockers.append("tasks_remaining")
        if executor_report.get("clean") is not True:
            blockers.append("default_executor_remaining")
        process_after = after["processes"]
        if (
            int(process_after.get("active_registered", 0) or 0) > 0
            or int(process_after.get("rogue_child_processes", 0) or 0) > 0
        ):
            blockers.append("child_processes_remaining")
        thread_after = after["threads"]
        if int(thread_after.get("active", 0) or 0) > 0:
            blockers.append("threads_remaining")
        if resource_report.get("clean") is not True:
            blockers.append("owned_resources_remaining")
        native_after = after["native_resources"]
        # The running asyncio loop necessarily owns an anonymous Unix
        # socketpair for cross-thread wakeups. It cannot disappear until after
        # asyncio.run() closes the loop, so only listeners are actionable at
        # this async boundary. The synchronous root-exit receipt verifies that
        # every socket and persistent-state handle is gone after loop close.
        if int(native_after.get("listening_socket_count", 0) or 0) > 0:
            blockers.append("listening_sockets_remaining")
        native_after["connections_deferred_to_root_exit"] = int(
            native_after.get("connection_count", 0) or 0
        )

        self._last_shutdown_report = {
            "clean": not blockers,
            "blockers": blockers,
            "started_at_unix": time.time() - (time.monotonic() - started),
            "completed_at_unix": time.time(),
            "duration_seconds": round(time.monotonic() - started, 6),
            "before": before,
            "after": after,
            "task_sweep": task_report,
            "resource_sweep": resource_report,
            "default_executor_sweep": executor_report,
        }
        self._shutdown_complete = True
        return dict(self._last_shutdown_report)

    async def _shutdown_default_executor(
        self,
        loop: Any,
        *,
        enabled: bool = True,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "attempted": False,
            "clean": True,
            "timeout_seconds": self.executor_shutdown_timeout_s,
            "workers_before": [],
            "workers_after": [],
            "warnings": [],
        }
        if not enabled:
            return report

        started = time.monotonic()
        report["attempted"] = True
        executor = getattr(loop, "_default_executor", None)
        report["workers_before"] = self._executor_worker_summary(executor)
        failure: BaseException | None = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            try:
                await loop.shutdown_default_executor(
                    timeout=self.executor_shutdown_timeout_s,
                )
            except (RuntimeError, TimeoutError, AttributeError) as exc:
                failure = exc

        warning_messages = [
            str(item.message)[:500]
            for item in caught
            if issubclass(item.category, RuntimeWarning)
        ]
        workers_after = self._executor_worker_summary(executor)
        report["warnings"] = warning_messages
        report["workers_after"] = workers_after
        report["duration_seconds"] = round(time.monotonic() - started, 6)
        if failure is not None:
            report.update(clean=False, error=repr(failure))
        elif warning_messages:
            report.update(
                clean=False,
                error="; ".join(warning_messages)[:500],
            )
        elif workers_after:
            report.update(
                clean=False,
                error=f"{len(workers_after)} default executor worker(s) remain alive",
            )

        if report["clean"] is not True:
            error = failure or RuntimeError(str(report.get("error") or "executor shutdown incomplete"))
            record_degradation(
                "runtime_hygiene_shutdown",
                error,
                severity="degraded",
                action="recorded incomplete default executor shutdown with worker evidence",
                extra={
                    "workers_after": workers_after,
                    "warnings": warning_messages,
                    "timeout_seconds": self.executor_shutdown_timeout_s,
                },
                enforce_failure_policy=False,
            )
        return report

    @staticmethod
    def _executor_worker_summary(executor: Any) -> list[dict[str, Any]]:
        workers = list(getattr(executor, "_threads", ()) or ())
        summary: list[dict[str, Any]] = []
        for worker in workers:
            try:
                alive = bool(worker.is_alive())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                alive = False
            if not alive:
                continue
            summary.append(
                {
                    "name": str(getattr(worker, "name", "executor-worker"))[:120],
                    "ident": getattr(worker, "ident", None),
                    "daemon": bool(getattr(worker, "daemon", False)),
                }
            )
        return summary[:64]

    async def on_stop_async(self) -> dict[str, Any]:
        return await self.stop()

    def cleanup(self) -> None:
        self._restore_patches()

    def reset_state(self) -> None:
        self._thread_records.clear()
        self._thread_refs.clear()
        self._process_records.clear()
        self._process_refs.clear()
        with self._resource_lock:
            self._resource_records.clear()
            self._resource_sequence = 0
            self._resource_admission_closed = False
        self._samples.clear()
        self._last_gc_at = 0.0
        self._shutdown_started = False
        self._shutdown_complete = False
        self._last_shutdown_report = None
        self._stop_lock = asyncio.Lock()

    def allocation_growth(self, top_n: int = 25) -> dict[str, Any]:
        """Attribute allocation GROWTH by call site since the baseline.

        First call under tracing arms the baseline snapshot; later calls
        return the top-N tracebacks by size growth. This is the surface the
        275MB/h idle-leak investigation needed: totals say THAT memory
        grows, only site diffs say WHERE (launch with
        AURA_RUNTIME_HYGIENE_TRACEMALLOC=1 and FRAMES>=10 for usable
        tracebacks). CPU-heavy on large heaps — callers run it off-loop.
        """
        if not tracemalloc.is_tracing():
            return {
                "available": False,
                "reason": "tracemalloc_off",
                "hint": "launch with AURA_RUNTIME_HYGIENE_TRACEMALLOC=1 "
                        "AURA_RUNTIME_HYGIENE_TRACEMALLOC_FRAMES=10",
            }
        try:
            snapshot = tracemalloc.take_snapshot().filter_traces((
                tracemalloc.Filter(False, tracemalloc.__file__),
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
            ))
        except (RuntimeError, MemoryError, ValueError) as exc:
            record_degradation('runtime_hygiene', exc)
            return {"available": False, "reason": f"snapshot_failed:{type(exc).__name__}"}

        traced_mb = tracemalloc.get_traced_memory()[0] / 1e6
        if self._tracemalloc_baseline is None:
            self._tracemalloc_baseline = snapshot
            self._tracemalloc_baseline_at = time.time()
            return {
                "available": True,
                "baseline_set": True,
                "baseline_at": self._tracemalloc_baseline_at,
                "traced_mb": round(traced_mb, 1),
                "frames": self.tracemalloc_frames,
            }

        top_n = max(1, min(int(top_n), 100))
        stats = snapshot.compare_to(self._tracemalloc_baseline, "traceback")
        sites = [
            {
                "size_diff_kb": round(stat.size_diff / 1024.0, 1),
                "count_diff": stat.count_diff,
                "size_kb": round(stat.size / 1024.0, 1),
                # format() emits two lines per frame (File + source); depth
                # follows tracemalloc.start(frames), so cap rather than slice
                # by our own frames setting or the File lines vanish.
                "traceback": stat.traceback.format()[-24:],
            }
            for stat in stats[:top_n]
        ]
        return {
            "available": True,
            "baseline_set": False,
            "baseline_at": self._tracemalloc_baseline_at,
            "window_s": round(time.time() - self._tracemalloc_baseline_at, 1),
            "traced_mb": round(traced_mb, 1),
            "growth_mb_total": round(sum(s.size_diff for s in stats) / 1e6, 1),
            "top_sites": sites,
        }

    def rearm_allocation_baseline(self) -> None:
        """Drop the baseline so the next allocation_growth() re-arms it."""
        self._tracemalloc_baseline = None
        self._tracemalloc_baseline_at = 0.0

    def capture_sample(self) -> MemorySample:
        memory = self.resource_observer.memory()
        rss_bytes = int(memory.process_rss_bytes)
        provenance = memory.provenance
        traced_bytes = 0
        try:
            if tracemalloc.is_tracing():
                traced_bytes, _peak = tracemalloc.get_traced_memory()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('runtime_hygiene', exc)
            logger.debug("RuntimeHygiene: tracemalloc snapshot failed: %s", exc)

        task_stats = self._task_tracker.get_stats()
        sample = MemorySample(
            timestamp=time.monotonic(),
            rss_bytes=rss_bytes,
            traced_bytes=traced_bytes,
            task_count=int(task_stats.get("active", 0)),
            thread_count=len(threading.enumerate()),
            child_process_count=self._count_child_processes(),
            observation_source=provenance.source.value,
            observation_scenario_id=provenance.scenario_id,
        )
        self._samples.append(sample)
        return sample

    def audit(self) -> dict[str, Any]:
        sample = self.capture_sample()
        self._refresh_thread_records()
        self._refresh_process_records()

        task_stats = self._task_tracker.get_stats()
        stale_tasks = self._task_tracker.get_stale_tasks(self.stale_task_age_s)
        thread_summary = self._thread_summary()
        process_summary = self._process_summary()
        resource_summary = self._resource_summary()
        memory_summary = self._memory_summary()

        repair_actions: list[str] = []
        issues: list[str] = []
        critical = False

        # Stale tasks and non-daemon threads are expected for long-lived components
        # (e.g. ThreadPoolExecutor, background event loops). We track them in the
        # telemetry payload but do not flag them as active issues to avoid noise.
        if process_summary["rogue_child_processes"]:
            issues.append(f"{process_summary['rogue_child_processes']} unregistered child process(es) detected")
            critical = True
        if memory_summary["sustained_growth"]:
            issues.append(memory_summary["message"])
            if time.monotonic() - self._last_gc_at > 60.0:
                gc.collect()
                self._last_gc_at = time.monotonic()
                repair_actions.append("gc.collect()")

        summary = {
            "healthy": not issues,
            "critical": critical,
            "issues": issues,
            "repair_actions": repair_actions,
            "tasks": {
                **task_stats,
                "stale_implicit_tasks": stale_tasks[:5],
            },
            "threads": thread_summary,
            "processes": process_summary,
            "resources": resource_summary,
            "memory": memory_summary,
            "latest_sample": {
                "rss_mb": round(sample.rss_bytes / (1024 * 1024), 1),
                "traced_mb": round(sample.traced_bytes / (1024 * 1024), 1),
                "task_count": sample.task_count,
                "thread_count": sample.thread_count,
                "child_process_count": sample.child_process_count,
            },
        }
        return summary

    def get_status(self) -> dict[str, Any]:
        report = self.audit()
        report["running"] = self._running
        report["shutdown_started"] = self._shutdown_started
        report["shutdown_complete"] = self._shutdown_complete
        report["shutdown_report"] = (
            dict(self._last_shutdown_report) if self._last_shutdown_report is not None else None
        )
        return report

    def get_shutdown_report(self) -> dict[str, Any] | None:
        return dict(self._last_shutdown_report) if self._last_shutdown_report is not None else None

    def get_root_exit_resource_report(self) -> dict[str, Any]:
        """Verify process resources after the event loop and logging close."""

        native = self._native_resource_summary()
        open_files = [str(path) for path in native.get("open_files", [])]
        open_file_set = set(open_files)
        persistent_state_files = [
            path
            for path in open_files
            if (
                path.lower().endswith(
                    (
                        ".db",
                        ".db-wal",
                        ".db-shm",
                        ".sqlite",
                        ".sqlite3",
                        ".sqlite-wal",
                        ".sqlite-shm",
                        "-wal",
                        "-shm",
                    )
                )
                or f"{path}-wal" in open_file_set
                or f"{path}-shm" in open_file_set
            )
        ]
        blockers: list[str] = []
        if int(native.get("connection_count", 0) or 0) > 0:
            blockers.append("native_sockets_remaining_after_loop_close")
        if persistent_state_files:
            blockers.append("persistent_state_files_open_after_shutdown")
        return {
            "clean": not blockers,
            "blockers": blockers,
            "native_resources": native,
            "persistent_state_files": persistent_state_files,
        }

    def close_root_exit_sockets(self) -> dict[str, Any]:
        """Close residual socket descriptors after loops and services are gone."""

        before = self._native_resource_summary()
        attempted: list[int] = []
        closed: list[int] = []
        failures: list[str] = []
        for connection in list(before.get("connections", [])):
            if not isinstance(connection, dict):
                continue
            try:
                fd = int(connection.get("fd", -1))
            except (TypeError, ValueError):
                continue
            if fd < 3 or fd in attempted:
                continue
            attempted.append(fd)
            try:
                if not stat.S_ISSOCK(os.fstat(fd).st_mode):
                    failures.append(f"fd:{fd}:not_socket")
                    continue
                os.close(fd)
                closed.append(fd)
            except OSError as exc:
                failures.append(f"fd:{fd}:{type(exc).__name__}: {exc}")
        after = self._native_resource_summary()
        residual = int(after.get("connection_count", 0) or 0)
        return {
            "clean": not failures and residual == 0,
            "attempted_fds": attempted,
            "closed_fds": closed,
            "failures": failures,
            "remaining_connections": residual,
        }

    def _patch_asyncio_new_event_loop(self) -> None:
        if self._original_new_event_loop is not None:
            return

        self._original_new_event_loop = asyncio.new_event_loop
        tracker = self._task_tracker

        def _patched_new_event_loop():
            loop = self._original_new_event_loop()
            try:
                tracker.install_loop_hygiene(loop)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('runtime_hygiene', exc)
                logger.debug("RuntimeHygiene: failed to install task factory on new loop: %s", exc)
            return loop

        asyncio.new_event_loop = _patched_new_event_loop

    def _patch_threading(self) -> None:
        if self._original_thread_start is not None:
            return

        self._original_thread_start = threading.Thread.start
        manager = self

        def _patched_start(thread: threading.Thread, *args, **kwargs):
            executor_teardown = manager._is_executor_shutdown_thread(thread)
            cleanup_critical = (
                shutdown_resource_creation_allowed() or executor_teardown
            )
            if not cleanup_critical and manager._shutdown_blocks_resource_start(
                operation=f"thread.start:{thread.name}", resource_kind="thread"
            ):
                manager._run_thread_suppression_cleanup(thread)
                raise RuntimeError("runtime_shutdown")
            if executor_teardown and manager._runtime_shutdown_latched():
                manager._record_creation_boundary(
                    operation=f"thread.start:{thread.name}",
                    resource_kind="thread",
                    outcome="allowed_teardown",
                    detail="asyncio_default_executor_shutdown",
                )
            thread._aura_shutdown_critical = cleanup_critical
            manager._register_thread(thread, source="thread.start")
            result = manager._original_thread_start(thread, *args, **kwargs)
            if manager._runtime_shutdown_latched() and not cleanup_critical:
                manager._record_creation_boundary(
                    operation=f"thread.start:{thread.name}",
                    resource_kind="thread",
                    outcome="crossed",
                    detail=f"ident={thread.ident}",
                )
                if not thread.is_alive():
                    manager._record_creation_boundary(
                        operation=f"thread.start:{thread.name}",
                        resource_kind="thread",
                        outcome="reaped",
                        detail="target_exited_at_shutdown_boundary",
                    )
            return result

        threading.Thread.start = _patched_start

    def _patch_subprocess(self) -> None:
        if self._original_popen_init is not None:
            return

        self._original_popen_init = subprocess.Popen.__init__
        manager = self

        def _patched_init(proc_self, *args, **kwargs):
            cleanup_critical = shutdown_resource_creation_allowed()
            command = kwargs.get("args") or (args[0] if args else "unknown")
            if manager._shutdown_blocks_resource_start(
                operation=f"subprocess.Popen:{str(command)[:160]}",
                resource_kind="subprocess",
            ):
                proc_self._child_created = False
                raise RuntimeError("runtime_shutdown")
            manager._original_popen_init(proc_self, *args, **kwargs)
            manager._register_subprocess(proc_self, args=args, kwargs=kwargs)
            if manager._runtime_shutdown_latched() and not cleanup_critical:
                operation = f"subprocess.Popen:{str(command)[:160]}"
                manager._record_creation_boundary(
                    operation=operation,
                    resource_kind="subprocess",
                    outcome="crossed",
                    detail=f"pid={getattr(proc_self, 'pid', None)}",
                )
                reaped = manager._reap_crossed_subprocess(proc_self)
                manager._record_creation_boundary(
                    operation=operation,
                    resource_kind="subprocess",
                    outcome="reaped" if reaped else "survived",
                    detail=f"pid={getattr(proc_self, 'pid', None)}",
                )
                raise RuntimeError("runtime_shutdown_after_subprocess_start")

        subprocess.Popen.__init__ = _patched_init

    def _patch_multiprocessing(self) -> None:
        if self._original_mp_start is not None:
            return

        self._original_mp_start = mp.process.BaseProcess.start
        manager = self

        def _patched_start(proc_self, *args, **kwargs):
            cleanup_critical = shutdown_resource_creation_allowed()
            if manager._shutdown_blocks_resource_start(
                operation=f"multiprocessing.start:{getattr(proc_self, 'name', 'unknown')}",
                resource_kind="multiprocessing",
            ):
                raise RuntimeError("runtime_shutdown")
            result = manager._original_mp_start(proc_self, *args, **kwargs)
            manager._register_multiprocessing_process(proc_self)
            if manager._runtime_shutdown_latched() and not cleanup_critical:
                operation = (
                    f"multiprocessing.start:{getattr(proc_self, 'name', 'unknown')}"
                )
                manager._record_creation_boundary(
                    operation=operation,
                    resource_kind="multiprocessing",
                    outcome="crossed",
                    detail=f"pid={getattr(proc_self, 'pid', None)}",
                )
                reaped = manager._reap_crossed_multiprocessing(proc_self)
                manager._record_creation_boundary(
                    operation=operation,
                    resource_kind="multiprocessing",
                    outcome="reaped" if reaped else "survived",
                    detail=f"pid={getattr(proc_self, 'pid', None)}",
                )
                raise RuntimeError("runtime_shutdown_after_multiprocessing_start")
            return result

        mp.process.BaseProcess.start = _patched_start

    @staticmethod
    def _is_executor_shutdown_thread(thread: threading.Thread) -> bool:
        target = getattr(thread, "_target", None)
        module = str(getattr(target, "__module__", "") or "")
        qualname = str(getattr(target, "__qualname__", "") or "")
        return module == "asyncio.base_events" and qualname.endswith(
            "BaseEventLoop._do_shutdown"
        )

    @staticmethod
    def _run_thread_suppression_cleanup(thread: threading.Thread) -> None:
        cleanup = getattr(thread, "_aura_shutdown_suppressed_cleanup", None)
        if not callable(cleanup):
            return
        try:
            cleanup()
        except Exception as exc:  # noqa: BLE001 - late-work ownership boundary
            record_degradation(
                "runtime_hygiene_shutdown",
                exc,
                severity="warning",
                action=f"recorded failed suppression cleanup for thread {thread.name}",
                enforce_failure_policy=False,
            )

    @staticmethod
    def _shutdown_blocks_resource_start(
        *,
        operation: str,
        resource_kind: str,
    ) -> bool:
        if shutdown_resource_creation_allowed():
            return False
        try:
            from core.runtime.shutdown_coordinator import (
                is_shutdown_requested,
                record_shutdown_admission_event,
            )

            if not is_shutdown_requested():
                return False
            record_shutdown_admission_event(
                operation,
                resource_kind=resource_kind,
                outcome="suppressed",
                detail="runtime_hygiene_creation_patch",
            )
            return True
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _runtime_shutdown_latched() -> bool:
        try:
            from core.runtime.shutdown_coordinator import is_shutdown_requested

            return bool(is_shutdown_requested())
        except (ImportError, RuntimeError, AttributeError):
            return False

    @staticmethod
    def _record_creation_boundary(
        *,
        operation: str,
        resource_kind: str,
        outcome: str,
        detail: str,
    ) -> None:
        try:
            from core.runtime.shutdown_coordinator import record_shutdown_admission_event

            record_shutdown_admission_event(
                operation,
                resource_kind=resource_kind,
                outcome=outcome,
                detail=detail,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            return

    @staticmethod
    def _reap_crossed_subprocess(proc: Any) -> bool:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=0.75)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=0.75)
            return proc.poll() is not None
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _reap_crossed_multiprocessing(proc: Any) -> bool:
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=0.75)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.75)
            return not proc.is_alive()
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
            return False

    def _restore_patches(self) -> None:
        if self._original_thread_start is not None:
            threading.Thread.start = self._original_thread_start
            self._original_thread_start = None
        if self._original_popen_init is not None:
            subprocess.Popen.__init__ = self._original_popen_init
            self._original_popen_init = None
        if self._original_mp_start is not None:
            mp.process.BaseProcess.start = self._original_mp_start
            self._original_mp_start = None
        if self._original_new_event_loop is not None:
            asyncio.new_event_loop = self._original_new_event_loop
            self._original_new_event_loop = None

    def _start_tracemalloc(self) -> None:
        if not self.tracemalloc_enabled:
            return
        if tracemalloc.is_tracing():
            return
        try:
            tracemalloc.start(self.tracemalloc_frames)
            self._tracemalloc_started_by_hygiene = True
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "runtime_hygiene",
                exc,
                severity="warning",
                action="continued runtime hygiene with tracemalloc disabled",
                extra={"tracemalloc_frames": self.tracemalloc_frames},
            )
            logger.debug("RuntimeHygiene: tracemalloc start failed: %s", exc)

    def _adopt_active_child_processes(self) -> None:
        try:
            parent_pid = int(os.getpid())
            children = [
                process
                for process in self.resource_observer.processes()
                if process.ppid == parent_pid or parent_pid in process.ancestor_pids
            ]
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation('runtime_hygiene', exc)
            logger.debug("RuntimeHygiene: existing child adoption skipped: %s", exc)
            return

        tracked_pids = {
            int(record.pid)
            for record in list(self._process_records.values())
            if record.finished_at is None and getattr(record, "pid", None)
        }
        for child in children:
            if _is_python_resource_tracker_process(child):
                continue
            try:
                pid = int(getattr(child, "pid", 0) or 0)
            except _PROCESS_INTROSPECTION_ERRORS:
                pid = 0
            if pid and pid in tracked_pids:
                continue
            command_parts = _process_cmdline(child)
            name = _process_name(child) or (f"pid:{pid}" if pid else "unknown_child")
            key = -(pid or len(self._process_records) + 1)
            self._process_records[key] = ProcessRecord(
                key=key,
                kind="subprocess",
                name=name,
                source="psutil.adopt_existing_child",
                command=" ".join(str(part) for part in command_parts)[:240] or name,
                pid=pid or None,
            )
            if self.resource_observer.provenance.host_observed and _HAS_PSUTIL:
                try:
                    self._process_refs[key] = psutil.Process(pid)
                except _PROCESS_INTROSPECTION_ERRORS:
                    pass
            if pid:
                tracked_pids.add(pid)

    def _register_thread(self, thread: threading.Thread, source: str) -> None:
        key = id(thread)
        record = self._thread_records.get(key)
        if record is None:
            record = ThreadRecord(
                key=key,
                name=thread.name,
                daemon=bool(thread.daemon),
                source=source,
            )
            self._thread_records[key] = record
            self._thread_refs[key] = thread
        else:
            record.name = thread.name
            record.daemon = bool(thread.daemon)

        if getattr(thread, "_aura_runtime_hygiene_wrapped", False):
            return

        original_run = thread.run

        def _wrapped_run(*args, **kwargs):
            record.started_at = time.monotonic()
            record.ident = threading.get_ident()
            try:
                if (
                    self._runtime_shutdown_latched()
                    and not getattr(thread, "_aura_shutdown_critical", False)
                ):
                    self._record_creation_boundary(
                        operation=f"thread.run:{thread.name}",
                        resource_kind="thread",
                        outcome="suppressed",
                        detail="shutdown_latched_before_target_entry",
                    )
                    self._run_thread_suppression_cleanup(thread)
                    return None
                return original_run(*args, **kwargs)
            except _THREAD_RUN_FAILURES as exc:
                record.exception = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record.finished_at = time.monotonic()

        thread.run = _wrapped_run
        thread._aura_runtime_hygiene_wrapped = True

    def _register_subprocess(self, proc: subprocess.Popen, *, args: tuple, kwargs: dict) -> None:
        command = kwargs.get("args")
        if command is None and args:
            command = args[0]
        if isinstance(command, (list, tuple)):
            command_text = " ".join(str(part) for part in command)
        else:
            command_text = str(command)

        key = id(proc)
        self._process_records[key] = ProcessRecord(
            key=key,
            kind="subprocess",
            name=getattr(proc, "args", command_text) if getattr(proc, "args", None) else command_text,
            source="subprocess.Popen",
            command=command_text[:240],
            pid=getattr(proc, "pid", None),
        )
        self._process_refs[key] = proc

    def register_shutdown_resource(
        self,
        resource: Any,
        *,
        kind: str,
        name: str,
        source: str,
        closer: Callable[[], Any] | None = None,
        timeout_s: float = 0.75,
        required: bool = True,
        blocking: bool = False,
    ) -> None:
        """Register a non-process resource for reverse-order final cleanup."""

        key = id(resource)
        try:
            from core.runtime.shutdown_coordinator import is_shutdown_requested

            crossed_shutdown = is_shutdown_requested()
        except (ImportError, RuntimeError, AttributeError):
            crossed_shutdown = False
        crossed_shutdown = self._shutdown_started or crossed_shutdown
        with self._resource_lock:
            if self._shutdown_complete or self._resource_admission_closed:
                self._record_resource_boundary(
                    name,
                    kind=kind,
                    outcome="survived",
                    detail="registered_after_final_sweep",
                )
                raise RuntimeError(f"resource registered after runtime final sweep: {name}")

            existing = self._resource_records.get(key)
            should_record_crossed = bool(
                crossed_shutdown
                and (existing is None or not existing.crossed_shutdown)
            )
            if existing is not None:
                existing.kind = str(kind or existing.kind)
                existing.name = str(name or existing.name)
                existing.source = str(source or existing.source)
                existing.closer = closer or existing.closer
                existing.timeout_s = max(0.05, float(timeout_s))
                existing.required = bool(required)
                existing.blocking = bool(blocking)
                existing.crossed_shutdown = (
                    existing.crossed_shutdown or crossed_shutdown
                )
            else:
                self._resource_sequence += 1
                self._resource_records[key] = ShutdownResourceRecord(
                    key=key,
                    kind=str(kind or "resource"),
                    name=str(name or type(resource).__name__),
                    source=str(source or "unknown"),
                    resource=resource,
                    closer=closer,
                    timeout_s=max(0.05, float(timeout_s)),
                    required=bool(required),
                    blocking=bool(blocking),
                    sequence=self._resource_sequence,
                    crossed_shutdown=crossed_shutdown,
                )
        if should_record_crossed:
            self._record_resource_boundary(
                name,
                kind=kind,
                outcome="crossed",
                detail="registered_during_shutdown",
            )

    def unregister_shutdown_resource(self, resource: Any) -> None:
        with self._resource_lock:
            record = self._resource_records.pop(id(resource), None)
        if record is not None and record.crossed_shutdown:
            self._record_resource_boundary(
                record.name,
                kind=record.kind,
                outcome="reaped",
                detail="owner_completed_cleanup",
            )

    @staticmethod
    def _record_resource_boundary(
        name: str,
        *,
        kind: str,
        outcome: str,
        detail: str,
    ) -> None:
        try:
            from core.runtime.shutdown_coordinator import record_shutdown_admission_event

            record_shutdown_admission_event(
                name,
                resource_kind=kind,
                outcome=outcome,
                detail=detail,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            return

    def _resource_summary(self) -> dict[str, Any]:
        with self._resource_lock:
            records = list(self._resource_records.values())
        active = [record for record in records if record.status not in {"completed", "released"}]
        return {
            "registered": len(records),
            "active": len(active),
            "required_active": sum(1 for record in active if record.required),
            "by_kind": {
                kind: sum(1 for record in active if record.kind == kind)
                for kind in sorted({record.kind for record in active})
            },
            "sample": [
                {
                    "kind": record.kind,
                    "name": record.name,
                    "source": record.source,
                    "status": record.status,
                    "required": record.required,
                    "error": record.error,
                }
                for record in active[:10]
            ],
        }

    def _native_resource_summary(self) -> dict[str, Any]:
        observer = self.resource_observer
        open_files = list(observer.open_files(pid=os.getpid()))
        connections = [
            connection
            for connection in observer.connections(kind="all")
            if connection.pid == os.getpid()
        ]
        owned_connections = [
            connection
            for connection in connections
            if int(connection.fd) >= 0
        ]

        connection_samples = [
            {
                "fd": connection.fd,
                "family": connection.family,
                "type": connection.socket_type,
                "status": connection.status,
                "local": (
                    {"host": connection.local_host, "port": connection.local_port}
                    if connection.local_port
                    else None
                ),
                "remote": (
                    {"host": connection.remote_host, "port": connection.remote_port}
                    if connection.remote_port
                    else None
                ),
            }
            for connection in owned_connections[:20]
        ]
        return {
            "available": True,
            "open_file_count": len(open_files),
            "connection_count": len(owned_connections),
            "listening_socket_count": sum(
                1
                for connection in owned_connections
                if connection.status.upper() == "LISTEN"
            ),
            "open_files": open_files[:20],
            "connections": connection_samples,
            "observation_source": observer.provenance.source.value,
            "observation_scenario_id": observer.provenance.scenario_id,
        }

    def register_process_handle(
        self,
        proc: Any,
        *,
        kind: str = "multiprocessing",
        name: str | None = None,
        source: str = "explicit_process_owner",
        command: str | None = None,
    ) -> None:
        """Register a child process from the subsystem that owns its lifecycle.

        Runtime hygiene patches process creation, but production model workers
        can be spawned from alternate multiprocessing contexts or after patches
        are temporarily restored during shutdown/restart edges. The owner still
        has the strongest provenance, so explicit registration is the canonical
        path for long-lived worker children.
        """

        pid = getattr(proc, "pid", None)
        for record in list(self._process_records.values()):
            if pid is not None and record.finished_at is None and record.pid == pid:
                record.kind = kind or record.kind
                record.name = str(name or record.name or getattr(proc, "name", kind))
                record.source = str(source or record.source)
                record.command = str(command or record.command or record.name)[:240]
                return
        key = id(proc)
        self._process_records[key] = ProcessRecord(
            key=key,
            kind=str(kind or "multiprocessing"),
            name=str(name or getattr(proc, "name", kind) or kind),
            source=str(source or "explicit_process_owner"),
            command=str(command or name or getattr(proc, "name", kind) or kind)[:240],
            pid=pid,
        )
        self._process_refs[key] = proc

    def process_handle_is_registered(self, proc: Any) -> bool:
        """Whether this exact live process is tracked for shutdown accounting.

        Registration that raised is visible to its caller; registration that
        silently did not take is not. A spawn path that owns a multi-gigabyte
        child needs to be able to CHECK rather than assume, because the cost
        of an untracked one is an orphan holding the model after shutdown.
        """
        pid = getattr(proc, "pid", None)
        key = id(proc)
        record = self._process_records.get(key)
        if record is not None and record.finished_at is None:
            return True
        if pid is None:
            return False
        return any(
            other.pid == pid and other.finished_at is None
            for other in list(self._process_records.values())
        )

    def handoff_successor(self, proc: Any, *, predecessor_pid: int) -> None:
        """Keep an exact registered successor alive through its parent's teardown."""
        key = id(proc)
        record = self._process_records.get(key)
        if predecessor_pid != os.getpid():
            raise ValueError("successor_predecessor_must_be_current_process")
        if record is None or self._process_refs.get(key) is not proc:
            raise ValueError("successor_handle_not_registered")
        if record.finished_at is not None or proc.poll() is not None:
            raise ValueError("successor_already_exited")
        record.successor_of_pid = predecessor_pid

    def retire_process_handle(self, proc: Any, *, exit_code: int | None = None) -> bool:
        """Retire one exact process handle after its owner proved termination.

        ``multiprocessing.Process.close()`` invalidates ``is_alive`` and
        ``exitcode``. Runtime hygiene used to retain that closed handle and
        probe it during every later audit, turning correct owner cleanup into
        repeated liveness faults. The lifecycle owner now marks the exact
        object finished before closing it; PID-only matches are deliberately
        ignored because a reused PID may already belong to a replacement.
        """

        key = id(proc)
        record = self._process_records.get(key)
        if record is None or self._process_refs.get(key) is not proc:
            return False
        if exit_code is None:
            try:
                observed = getattr(proc, "exitcode", None)
                exit_code = int(observed) if observed is not None else None
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
                exit_code = None
        record.exit_code = exit_code
        record.finished_at = record.finished_at or time.monotonic()
        self._process_refs.pop(key, None)
        self._evict_finished(self._process_records, self._process_refs)
        return True

    def _register_multiprocessing_process(self, proc: mp.Process) -> None:
        self.register_process_handle(
            proc,
            kind="multiprocessing",
            name=getattr(proc, "name", "multiprocessing"),
            source="multiprocessing.Process.start",
            command=getattr(proc, "name", "multiprocessing"),
        )

    # How many FINISHED records each registry retains for post-mortem
    # reporting. The registries used to retain every record — and every
    # strong ref — until shutdown: each Popen's stdout/stderr wrappers and
    # pipe buffers stayed reachable for process lifetime, which is the
    # dominant cluster in the Jul 7 soak's tracemalloc top-growth (16k
    # io.open + 21k TextIOWrapper live objects, longevity_leakrepro).
    _FINISHED_RECORD_RETENTION = 512

    def _evict_finished(self, records: dict, refs: dict) -> None:
        """Drop strong refs to finished resources; keep a bounded history.

        The ref is what pins pipes, buffers, and thread objects in memory.
        The record is a small dataclass kept for shutdown/health reporting,
        bounded to the most recent finished entries so a long-lived runtime
        cannot accumulate one record per subprocess it ever ran.
        """
        finished = [
            key for key, record in records.items()
            if record.finished_at is not None
        ]
        for key in finished:
            refs.pop(key, None)
        overflow = len(finished) - self._FINISHED_RECORD_RETENTION
        if overflow > 0:
            finished.sort(key=lambda key: records[key].finished_at)
            for key in finished[:overflow]:
                records.pop(key, None)

    def _refresh_thread_records(self) -> None:
        now = time.monotonic()
        live_idents = {thread.ident for thread in threading.enumerate()}
        for key, thread in list(self._thread_refs.items()):
            record = self._thread_records.get(key)
            if record is None:
                continue
            record.name = thread.name
            if thread.ident is not None:
                record.ident = thread.ident
            if thread.ident is not None and record.started_at is None:
                record.started_at = now
            if record.ident is not None and record.ident not in live_idents and record.finished_at is None:
                record.finished_at = now
        self._evict_finished(self._thread_records, self._thread_refs)

    def _refresh_process_records(self) -> None:
        now = time.monotonic()
        for key, proc in list(self._process_refs.items()):
            record = self._process_records.get(key)
            if record is None:
                continue
            if hasattr(proc, "poll"):
                try:
                    return_code = proc.poll()
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: subprocess poll failed: %s", exc)
                    return_code = None
                if return_code is not None:
                    record.exit_code = int(return_code)
                    record.finished_at = record.finished_at or now
            elif hasattr(proc, "is_alive"):
                try:
                    alive = proc.is_alive()
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: multiprocessing liveness failed: %s", exc)
                    alive = False
                if not alive:
                    record.exit_code = getattr(proc, "exitcode", None)
                    record.finished_at = record.finished_at or now
                else:
                    record.pid = getattr(proc, "pid", record.pid)
            elif hasattr(proc, "is_running"):
                try:
                    alive = bool(proc.is_running())
                    status = proc.status() if alive else "stopped"
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: adopted child liveness failed: %s", exc)
                    alive = False
                    status = "error"
                if not alive or status == "zombie":
                    record.finished_at = record.finished_at or now
            elif hasattr(proc, "returncode"):
                try:
                    return_code = getattr(proc, "returncode", None)
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: asyncio subprocess returncode read failed: %s", exc)
                    return_code = None
                if return_code is not None:
                    try:
                        record.exit_code = int(return_code)
                    except (RuntimeError, TypeError, ValueError):
                        record.exit_code = None
                    record.finished_at = record.finished_at or now
        # Release finished procs: OUR ref must not keep their pipes alive.
        # Callers that still hold a finished Popen keep it valid; when they
        # drop it, everything frees. Pipes are never closed from here — a
        # caller may legitimately read buffered output after exit.
        self._evict_finished(self._process_records, self._process_refs)

    def _thread_summary(self, *, include_stacks: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        frames = sys._current_frames() if include_stacks else {}
        active = 0
        active_non_daemon = 0
        active_daemon = 0
        stale_non_daemon = 0
        sample: list[dict[str, Any]] = []
        active_sample: list[dict[str, Any]] = []
        for record in list(self._thread_records.values()):
            if record.finished_at is not None:
                continue
            active += 1
            if len(active_sample) < 20:
                thread = self._thread_refs.get(record.key)
                item: dict[str, Any] = {
                    "name": record.name,
                    "ident": record.ident,
                    "daemon": record.daemon,
                    "shutdown_critical": bool(
                        getattr(thread, "_aura_shutdown_critical", False)
                    ),
                    "age_s": round(record.age_s(now), 1),
                    "source": record.source,
                    "exception": record.exception,
                }
                if include_stacks and record.ident is not None:
                    frame = frames.get(record.ident)
                    if frame is not None:
                        item["stack"] = [
                            {
                                "file": entry.filename[-240:],
                                "line": entry.lineno,
                                "function": entry.name,
                                "code": (entry.line or "")[:240],
                            }
                            for entry in traceback.extract_stack(frame, limit=12)
                        ]
                active_sample.append(item)
            if not record.daemon:
                active_non_daemon += 1
                if record.age_s(now) >= self.stale_thread_age_s:
                    stale_non_daemon += 1
                    sample.append(
                        {
                            "name": record.name,
                            "age_s": round(record.age_s(now), 1),
                            "source": record.source,
                        }
                    )
            else:
                active_daemon += 1
        return {
            "active": active,
            "active_non_daemon": active_non_daemon,
            "active_daemon": active_daemon,
            "stale_non_daemon": stale_non_daemon,
            "sample": sample[:5],
            "active_sample": active_sample,
        }

    def _process_summary(self) -> dict[str, Any]:
        active_registered = 0
        active_subprocesses = 0
        active_multiprocessing = 0
        active_registered_pids = set()
        active_samples: list[dict[str, Any]] = []
        for record in list(self._process_records.values()):
            if record.finished_at is not None:
                continue
            active_registered += 1
            if len(active_samples) < 20:
                active_samples.append(
                    {
                        "pid": record.pid,
                        "kind": record.kind,
                        "name": record.name,
                        "source": record.source,
                        "command": record.command,
                        "age_s": round(record.age_s(), 1),
                    }
                )
            if getattr(record, "pid", None):
                try:
                    active_registered_pids.add(int(record.pid))
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation(
                        "runtime_hygiene",
                        exc,
                        severity="warning",
                        action="ignored malformed registered process pid during hygiene summary",
                    )
                    logger.debug("RuntimeHygiene: malformed registered pid %r: %s", record.pid, exc)
            if record.kind == "subprocess":
                active_subprocesses += 1
            elif record.kind == "multiprocessing":
                active_multiprocessing += 1
        rogue_children = 0
        owned_descendants = 0
        rogue_samples: list[dict[str, Any]] = []
        live_process_ids: set[int] | None = None
        try:
            pid_observation = self.resource_observer.process_ids()
            if bool(getattr(pid_observation, "available", False)):
                live_process_ids = {
                    int(pid)
                    for pid in tuple(getattr(pid_observation, "pids", ()) or ())
                    if int(pid) > 0
                }
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation("runtime_hygiene", exc)
            logger.debug("RuntimeHygiene: live PID reconciliation skipped: %s", exc)
        try:
            parent_pid = int(os.getpid())
            active_children = [
                process
                for process in self.resource_observer.processes()
                if process.ppid == parent_pid or parent_pid in process.ancestor_pids
            ]
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation('runtime_hygiene', exc)
            logger.debug("RuntimeHygiene: child process scan failed: %s", exc)
            active_children = []
        if active_children:
            child_by_pid = {
                pid: child
                for child in active_children
                if (pid := _process_pid(child)) > 0
            }
            for child in active_children:
                child_pid = _process_pid(child)
                if child_pid in active_registered_pids or _is_python_resource_tracker_process(child):
                    continue
                # The enriched host process table is deliberately cached for
                # up to two seconds. A short-lived governed command can finish
                # and leave the process ledger before that cached row expires,
                # which used to turn successful git/osascript probes into
                # false rogue-child alerts. Reconcile only unregistered rows
                # against the cheap, uncached PID census. If the census itself
                # is unavailable, retain the fail-closed cached observation.
                if (
                    child_pid > 0
                    and live_process_ids is not None
                    and child_pid not in live_process_ids
                ):
                    continue
                if (
                    _process_ppid(child) == int(os.getpid())
                    and _is_python_multiprocessing_spawn_process(child)
                ):
                    self.register_process_handle(
                        child,
                        kind="multiprocessing",
                        name=_process_name(child) or "multiprocessing.spawn",
                        source="psutil.adopt_during_summary",
                        command=" ".join(_process_cmdline(child))[:240],
                    )
                    active_registered += 1
                    active_multiprocessing += 1
                    if child_pid > 0:
                        active_registered_pids.add(child_pid)
                    continue
                # [FIX] Auto-adopt direct children that match known Aura worker
                # name patterns (e.g. MLXWorker-*).  These are spawned via
                # multiprocessing but their cmdline may not match the generic
                # spawn signature.  A timing race between _spawn_worker_blocking
                # and register_process_handle can leave the worker unregistered
                # for one hygiene cycle, producing a transient "1 unregistered
                # child process(es)" StabilityGuardian alert.
                if _process_ppid(child) == int(os.getpid()):
                    child_name = _process_name(child)
                    # The keep-awake assertion is Aura's own: core.runtime.
                    # keep_awake spawns it through the subprocess gateway as a
                    # direct child and owns its lifecycle. It is named for the
                    # macOS binary rather than for Aura, so it matched none of
                    # the worker tags below and was reported as an unregistered
                    # child on EVERY boot — a permanent DEGRADED card in the
                    # user's neural feed for a process the runtime deliberately
                    # started.
                    if child_name and _is_governed_applescript_process(child):
                        self.register_process_handle(
                            child,
                            kind="subprocess",
                            name="desktop_action_gateway.osascript",
                            source="psutil.adopt_governed_applescript_during_summary",
                            command=" ".join(_process_cmdline(child))[:240],
                        )
                        active_registered += 1
                        active_subprocesses += 1
                        if child_pid > 0:
                            active_registered_pids.add(child_pid)
                        continue
                    if child_name and _is_keep_awake_assertion_process(child):
                        self.register_process_handle(
                            child,
                            kind="subprocess",
                            name="keep_awake.caffeinate",
                            source="psutil.adopt_keep_awake_during_summary",
                            command=" ".join(_process_cmdline(child))[:240],
                        )
                        active_registered += 1
                        active_subprocesses += 1
                        if child_pid > 0:
                            active_registered_pids.add(child_pid)
                        continue
                    if child_name and any(
                        tag in child_name for tag in ("MLXWorker", "AuraWorker", "Aura")
                    ):
                        self.register_process_handle(
                            child,
                            kind="multiprocessing",
                            name=child_name,
                            source="psutil.adopt_named_worker_during_summary",
                            command=" ".join(_process_cmdline(child))[:240],
                        )
                        active_registered += 1
                        active_multiprocessing += 1
                        if child_pid > 0:
                            active_registered_pids.add(child_pid)
                        continue
                if self._is_owned_descendant_process(
                    child,
                    active_registered_pids=active_registered_pids,
                    child_by_pid=child_by_pid,
                ):
                    owned_descendants += 1
                    continue
                rogue_children += 1
                if len(rogue_samples) < 5:
                    rogue_samples.append(
                        {
                            "pid": child_pid or None,
                            "ppid": _process_ppid(child) or None,
                            "name": _process_name(child)[:80],
                            "command": " ".join(_process_cmdline(child))[:160],
                        }
                    )
        return {
            "active_registered": max(0, active_registered),
            "active_subprocesses": max(0, active_subprocesses),
            "active_multiprocessing": max(0, active_multiprocessing),
            "owned_descendant_processes": max(0, owned_descendants),
            "rogue_child_processes": max(0, rogue_children),
            "active_samples": active_samples,
            "rogue_samples": rogue_samples,
        }

    def _is_owned_descendant_process(
        self,
        proc: Any,
        *,
        active_registered_pids: set[int],
        child_by_pid: dict[int, Any],
    ) -> bool:
        """Return true when a recursive child belongs to a registered owner.

        ``psutil.children(recursive=True)`` returns grandchildren as well as
        direct children. A registered model worker can legitimately spawn a
        short-lived helper below it; that helper should be visible in telemetry
        without being misclassified as an unregistered root process. The walk is
        bounded and stops at Aura's current process so an unrelated child still
        fails the hygiene check.
        """

        current_pid = int(os.getpid())
        seen: set[int] = set()
        parent_pid = _process_ppid(proc)
        for _ in range(16):
            if parent_pid <= 0 or parent_pid == current_pid or parent_pid in seen:
                return False
            if parent_pid in active_registered_pids:
                return True
            seen.add(parent_pid)
            parent = child_by_pid.get(parent_pid)
            if parent is None:
                return False
            parent_pid = _process_ppid(parent)
        return False

    def _memory_summary(self) -> dict[str, Any]:
        if len(self._samples) < self.memory_growth_window:
            latest = self._samples[-1] if self._samples else None
            return {
                "sustained_growth": False,
                "transient_growth": False,
                "message": "warming_up",
                "rss_mb": round((latest.rss_bytes if latest else 0) / (1024 * 1024), 1),
                "delta_mb": 0.0,
            }

        window = list(self._samples)[-self.memory_growth_window:]
        first = window[0]
        last = window[-1]
        delta_bytes = last.rss_bytes - first.rss_bytes
        delta_mb = delta_bytes / (1024 * 1024)
        baseline = max(float(first.rss_bytes), 1.0)
        positive_steps = sum(1 for idx in range(1, len(window)) if window[idx].rss_bytes >= window[idx - 1].rss_bytes)
        growth_ratio = delta_bytes / baseline
        sustained_growth = (
            delta_mb >= self.memory_growth_min_delta_mb
            or (growth_ratio >= self.memory_growth_ratio and positive_steps >= len(window) - 1)
        )
        transient_model_growth = []
        if sustained_growth:
            transient_model_growth = self._active_local_model_activity()
        message = "memory_growth_stable"
        if sustained_growth and transient_model_growth:
            message = "Transient RSS growth during local model activity: " + ", ".join(transient_model_growth[:3])
            sustained_growth = False
        elif sustained_growth:
            message = f"Sustained RSS growth detected (+{delta_mb:.1f}MB over {len(window)} samples)"
        return {
            "sustained_growth": sustained_growth,
            "transient_growth": bool(transient_model_growth),
            "message": message,
            "rss_mb": round(last.rss_bytes / (1024 * 1024), 1),
            "delta_mb": round(delta_mb, 1),
        }

    def _active_local_model_activity(self) -> list[str]:
        active: list[str] = []
        now = time.time()
        registries = (
            ("core.brain.llm.mlx_client", "_CLIENTS"),
        )
        for module_name, registry_attr in registries:
            try:
                module = __import__(module_name, fromlist=[registry_attr])
                registry_items = _snapshot_mapping_items(getattr(module, registry_attr, {}) or {})
            except (RuntimeError, AttributeError, TypeError):
                continue

            for client_path, client in registry_items:
                if client is None or not hasattr(client, "get_lane_status"):
                    continue
                try:
                    lane = client.get_lane_status()
                except (OSError, ConnectionError, TimeoutError):
                    continue
                state = str(lane.get("state", "") or "").strip().lower()
                current_request = float(lane.get("current_request_started_at", 0.0) or 0.0)
                if bool(lane.get("warmup_in_flight")) or current_request > 0.0 or state in {
                    "spawning",
                    "handshaking",
                    "warming",
                    "recovering",
                }:
                    active.append(f"{os.path.basename(str(client_path))}:{state or 'active'}")
                    continue

                recent_activity_at = max(
                    float(lane.get("last_ready_at", 0.0) or 0.0),
                    float(lane.get("last_progress_at", 0.0) or 0.0),
                    float(lane.get("last_transition_at", 0.0) or 0.0),
                )
                if (
                    self.model_activity_grace_s > 0.0
                    and recent_activity_at > 0.0
                    and (now - recent_activity_at) <= self.model_activity_grace_s
                ):
                    active.append(f"{os.path.basename(str(client_path))}:recent")
        return active

    def _count_child_processes(self) -> int:
        try:
            parent_pid = int(os.getpid())
            return sum(
                1
                for process in self.resource_observer.processes()
                if process.ppid == parent_pid or parent_pid in process.ancestor_pids
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation('runtime_hygiene', exc)
            logger.debug("RuntimeHygiene: child process scan failed: %s", exc)
            return 0

    async def _invoke_resource_callable(
        self,
        callback: Callable[[], Any],
        *,
        timeout_s: float,
    ) -> None:
        started = time.monotonic()
        if inspect.iscoroutinefunction(callback):
            result = callback()
        else:
            result = await run_sync_shutdown_callable(
                callback,
                timeout_s=max(0.05, timeout_s),
                name="runtime-resource-cleanup",
            )
        if inspect.isawaitable(result):
            remaining = max(0.05, timeout_s - (time.monotonic() - started))
            await asyncio.wait_for(result, timeout=remaining)

    async def _close_shutdown_resource(
        self,
        record: ShutdownResourceRecord,
        *,
        timeout_s: float,
    ) -> None:
        started = time.monotonic()
        record.status = "closing"
        closer = record.closer
        if closer is None:
            for method_name in ("stop", "close", "cancel"):
                candidate = getattr(record.resource, method_name, None)
                if callable(candidate):
                    closer = candidate
                    break
        try:
            if closer is None:
                raise RuntimeError(
                    f"no zero-argument closer for {record.kind}:{record.name}"
                )
            await self._invoke_resource_callable(
                closer,
                timeout_s=timeout_s,
            )
            for followup_name in ("wait_closed", "join_thread"):
                followup = getattr(record.resource, followup_name, None)
                if not callable(followup):
                    continue
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(
                        f"resource cleanup budget exhausted before {followup_name}"
                    )
                await self._invoke_resource_callable(
                    followup,
                    timeout_s=remaining,
                )
        except asyncio.CancelledError:
            record.status = "cancelled"
            record.duration_seconds = round(time.monotonic() - started, 6)
            raise
        except Exception as exc:  # noqa: BLE001 - final resource teardown boundary
            record.status = "failed"
            record.error = repr(exc)
            record.duration_seconds = round(time.monotonic() - started, 6)
            record_degradation(
                "runtime_hygiene_shutdown",
                exc,
                severity="degraded" if record.required else "warning",
                action=f"recorded failed final cleanup for {record.kind}:{record.name}",
                enforce_failure_policy=False,
                extra={"source": record.source, "required": record.required},
            )
            return

        record.status = "completed"
        record.error = None
        record.duration_seconds = round(time.monotonic() - started, 6)
        if record.crossed_shutdown:
            self._record_resource_boundary(
                record.name,
                kind=record.kind,
                outcome="reaped",
                detail="runtime_hygiene_final_sweep",
            )

    async def _cleanup_shutdown_resources(self) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + self.resource_shutdown_timeout_s
        attempted: set[int] = set()

        # Two bounded snapshots catch resources that crossed the shutdown latch
        # while the first reverse-order pass was already running.
        for _pass in range(2):
            with self._resource_lock:
                pending = sorted(
                    (
                        record
                        for record in self._resource_records.values()
                        if record.key not in attempted
                        and record.status not in {"completed", "released"}
                    ),
                    key=lambda record: record.sequence,
                    reverse=True,
                )
            if not pending:
                break
            for record in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                attempted.add(record.key)
                await self._close_shutdown_resource(
                    record,
                    timeout_s=min(record.timeout_s, remaining),
                )

        with self._resource_lock:
            self._resource_admission_closed = True
            all_records = list(self._resource_records.values())
        residual = [
            record
            for record in all_records
            if record.status not in {"completed", "released"}
        ]
        for record in residual:
            if record.status == "registered":
                record.status = "skipped"
                record.error = "resource cleanup budget exhausted"
            if record.crossed_shutdown:
                self._record_resource_boundary(
                    record.name,
                    kind=record.kind,
                    outcome="survived",
                    detail=record.error or record.status,
                )
        required_residual = [record for record in residual if record.required]
        return {
            "clean": not required_residual,
            "attempted": len(attempted),
            "completed": sum(
                1 for record in all_records if record.status == "completed"
            ),
            "residual": len(residual),
            "required_residual": len(required_residual),
            "duration_seconds": round(time.monotonic() - started, 6),
            "resources": [
                {
                    "kind": record.kind,
                    "name": record.name,
                    "source": record.source,
                    "required": record.required,
                    "status": record.status,
                    "duration_seconds": record.duration_seconds,
                    "error": record.error,
                }
                for record in sorted(
                    all_records,
                    key=lambda item: item.sequence,
                    reverse=True,
                )
            ],
        }

    async def _cleanup_child_processes(self) -> None:
        async def _cleanup_one(proc: Any) -> None:
            record = self._process_records.get(id(proc))
            if (
                record is not None
                and self._process_refs.get(id(proc)) is proc
                and record.successor_of_pid == os.getpid()
            ):
                return
            if _is_python_resource_tracker_process(proc):
                return
            if hasattr(proc, "poll"):
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            await run_sync_shutdown_callable(
                                lambda: proc.wait(self.process_shutdown_timeout_s),
                                timeout_s=self.process_shutdown_timeout_s + 0.25,
                                name="subprocess-wait",
                            )
                        except (RuntimeError, TimeoutError, AttributeError, subprocess.TimeoutExpired):
                            proc.kill()
                            try:
                                await run_sync_shutdown_callable(
                                    lambda: proc.wait(0.2),
                                    timeout_s=0.3,
                                    name="subprocess-kill-wait",
                                )
                            except (RuntimeError, TimeoutError, AttributeError, subprocess.TimeoutExpired) as exc:
                                record_degradation(
                                    "runtime_hygiene",
                                    exc,
                                    severity="warning",
                                    action="subprocess did not confirm exit after kill",
                                )
                                logger.debug("RuntimeHygiene: subprocess kill wait failed: %s", exc)
                except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: subprocess cleanup failed: %s", exc)
            elif hasattr(proc, "is_alive"):
                try:
                    if proc.is_alive():
                        proc.terminate()
                        await run_sync_shutdown_callable(
                            lambda: proc.join(self.process_shutdown_timeout_s),
                            timeout_s=self.process_shutdown_timeout_s + 0.25,
                            name="multiprocessing-join",
                        )
                        if proc.is_alive():
                            proc.kill()
                            try:
                                await run_sync_shutdown_callable(
                                    lambda: proc.join(0.2),
                                    timeout_s=0.3,
                                    name="multiprocessing-kill-join",
                                )
                            except (RuntimeError, TimeoutError, AttributeError, TypeError, ValueError) as exc:
                                record_degradation(
                                    "runtime_hygiene",
                                    exc,
                                    severity="warning",
                                    action="multiprocessing child did not confirm exit after kill",
                                )
                                logger.debug("RuntimeHygiene: multiprocessing kill join failed: %s", exc)
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: multiprocessing cleanup failed: %s", exc)
            elif _HAS_PSUTIL and hasattr(proc, "is_running"):
                try:
                    if proc.is_running():
                        proc.terminate()
                        try:
                            await run_sync_shutdown_callable(
                                lambda: proc.wait(self.process_shutdown_timeout_s),
                                timeout_s=self.process_shutdown_timeout_s + 0.25,
                                name="psutil-process-wait",
                            )
                        except (RuntimeError, TimeoutError, AttributeError, TypeError, ValueError):
                            if proc.is_running():
                                proc.kill()
                                try:
                                    await run_sync_shutdown_callable(
                                        lambda: proc.wait(0.2),
                                        timeout_s=0.3,
                                        name="psutil-process-kill-wait",
                                    )
                                except (RuntimeError, TimeoutError, AttributeError, TypeError, ValueError) as exc:
                                    record_degradation(
                                        "runtime_hygiene",
                                        exc,
                                        severity="warning",
                                        action="psutil child did not confirm exit after kill",
                                    )
                                    logger.debug("RuntimeHygiene: psutil kill wait failed: %s", exc)
                except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: psutil child cleanup failed: %s", exc)

        cleanup_coros = [_cleanup_one(proc) for proc in list(self._process_refs.values())]
        if not cleanup_coros:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*cleanup_coros, return_exceptions=True),
                timeout=max(1.0, self.process_shutdown_timeout_s + 0.75),
            )
        except TimeoutError as exc:
            record_degradation(
                "runtime_hygiene_shutdown",
                exc,
                severity="warning",
                action="continued shutdown after bounded concurrent child-process cleanup timed out",
                enforce_failure_policy=False,
            )

    async def _join_non_daemon_threads(self) -> None:
        join_candidates: list[threading.Thread] = []
        for thread in list(self._thread_refs.values()):
            if thread.daemon:
                continue
            if not thread.is_alive():
                continue
            if thread.ident == threading.get_ident():
                continue
            join_candidates.append(thread)
        if not join_candidates:
            return

        selected = join_candidates[: self.max_thread_joins_per_shutdown]
        skipped = join_candidates[self.max_thread_joins_per_shutdown :]
        if skipped:
            record_degradation(
                "runtime_hygiene_shutdown",
                RuntimeError(f"{len(skipped)} non-daemon thread(s) left for owner shutdown"),
                severity="warning",
                action=(
                    "bounded runtime hygiene shutdown thread joins; remaining live threads "
                    "are left to their owning services"
                ),
                extra={
                    "skipped_threads": [getattr(thread, "name", "unknown") for thread in skipped[:10]],
                    "selected_count": len(selected),
                    "skipped_count": len(skipped),
                },
                enforce_failure_policy=False,
            )

        join_coros = [
            run_sync_shutdown_callable(
                lambda thread=thread: self._join_thread_if_not_current(
                    thread,
                    self.thread_join_timeout_s,
                ),
                timeout_s=self.thread_join_timeout_s + 0.2,
                name=f"thread-join:{thread.name}",
            )
            for thread in selected
        ]
        if not join_coros:
            return
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*join_coros, return_exceptions=True),
                timeout=max(0.5, self.thread_join_timeout_s + 0.3),
            )
        except TimeoutError as exc:
            record_degradation(
                "runtime_hygiene_shutdown",
                exc,
                severity="warning",
                action="continued shutdown after bounded concurrent thread join timed out",
                extra={
                    "selected_count": len(selected),
                    "skipped_count": len(skipped),
                    "selected_threads": [getattr(thread, "name", "unknown") for thread in selected[:10]],
                },
                enforce_failure_policy=False,
            )
            return
        for result in results:
            if isinstance(result, (RuntimeError, AttributeError, TypeError, ValueError)):
                record_degradation(
                    "runtime_hygiene_shutdown",
                    result,
                    severity="warning",
                    action="continued shutdown after a bounded thread join failed",
                    enforce_failure_policy=False,
                )
                logger.debug("RuntimeHygiene: thread join failed: %s", result)

    @staticmethod
    def _join_thread_if_not_current(thread: threading.Thread, timeout_s: float) -> None:
        if thread.ident == threading.get_ident():
            return
        thread.join(timeout_s)


_runtime_hygiene: RuntimeHygieneManager | None = None


def get_runtime_hygiene() -> RuntimeHygieneManager:
    global _runtime_hygiene
    if _runtime_hygiene is None:
        _runtime_hygiene = RuntimeHygieneManager()
    return _runtime_hygiene

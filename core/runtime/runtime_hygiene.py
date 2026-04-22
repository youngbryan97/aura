from __future__ import annotations

import asyncio
import gc
import logging
import multiprocessing as mp
import os
import subprocess
import threading
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from core.utils.task_tracker import get_task_tracker

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger("Aura.RuntimeHygiene")


@dataclass
class MemorySample:
    timestamp: float
    rss_bytes: int
    traced_bytes: int
    task_count: int
    thread_count: int
    child_process_count: int


@dataclass
class ThreadRecord:
    key: int
    name: str
    daemon: bool
    source: str
    created_at: float = field(default_factory=time.monotonic)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    ident: Optional[int] = None
    exception: Optional[str] = None

    def age_s(self, now: Optional[float] = None) -> float:
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
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    finished_at: Optional[float] = None

    def age_s(self, now: Optional[float] = None) -> float:
        current_time = now if now is not None else time.monotonic()
        return max(0.0, current_time - self.created_at)


class RuntimeHygieneManager:
    """Tracks tasks, threads, child processes, and memory growth across the runtime."""

    def __init__(self):
        self._running = False
        self._thread_records: Dict[int, ThreadRecord] = {}
        self._thread_refs: Dict[int, threading.Thread] = {}
        self._process_records: Dict[int, ProcessRecord] = {}
        self._process_refs: Dict[int, Any] = {}
        self._samples: Deque[MemorySample] = deque(maxlen=36)
        self._task_tracker = get_task_tracker()
        self._last_gc_at = 0.0

        self.memory_growth_window = 6
        self.memory_growth_min_delta_mb = 128.0
        self.memory_growth_ratio = 0.12
        self.stale_thread_age_s = 900.0
        self.stale_task_age_s = 900.0
        self.process_shutdown_timeout_s = 1.0
        self.thread_join_timeout_s = 0.2
        self.tracemalloc_enabled = str(
            os.getenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.tracemalloc_frames = max(
            1,
            int(os.getenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC_FRAMES", "1") or 1),
        )
        self._tracemalloc_started_by_hygiene = False

        self._original_thread_start = None
        self._original_popen_init = None
        self._original_mp_start = None
        self._original_new_event_loop = None

        self._proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None

    async def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
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

    async def stop(self) -> None:
        self._task_tracker.restore_loop_hygiene()
        self._restore_patches()
        await self._cleanup_child_processes()
        await self._join_non_daemon_threads()
        if self._tracemalloc_started_by_hygiene and tracemalloc.is_tracing():
            try:
                tracemalloc.stop()
            except Exception as exc:
                logger.debug("RuntimeHygiene: tracemalloc stop failed: %s", exc)
            finally:
                self._tracemalloc_started_by_hygiene = False
        self.capture_sample()
        self._running = False

    async def on_stop_async(self) -> None:
        await self.stop()

    def cleanup(self) -> None:
        self._restore_patches()

    def reset_state(self) -> None:
        self._thread_records.clear()
        self._thread_refs.clear()
        self._process_records.clear()
        self._process_refs.clear()
        self._samples.clear()
        self._last_gc_at = 0.0

    def capture_sample(self) -> MemorySample:
        rss_bytes = 0
        if self._proc is not None:
            try:
                rss_bytes = int(self._proc.memory_info().rss)
            except Exception as exc:
                logger.debug("RuntimeHygiene: failed to read RSS: %s", exc)
        traced_bytes = 0
        try:
            if tracemalloc.is_tracing():
                traced_bytes, _peak = tracemalloc.get_traced_memory()
        except Exception as exc:
            logger.debug("RuntimeHygiene: tracemalloc snapshot failed: %s", exc)

        task_stats = self._task_tracker.get_stats()
        sample = MemorySample(
            timestamp=time.monotonic(),
            rss_bytes=rss_bytes,
            traced_bytes=traced_bytes,
            task_count=int(task_stats.get("active", 0)),
            thread_count=len(threading.enumerate()),
            child_process_count=self._count_child_processes(),
        )
        self._samples.append(sample)
        return sample

    def audit(self) -> Dict[str, Any]:
        sample = self.capture_sample()
        self._adopt_active_child_processes()
        self._refresh_thread_records()
        self._refresh_process_records()

        task_stats = self._task_tracker.get_stats()
        stale_tasks = self._task_tracker.get_stale_tasks(self.stale_task_age_s)
        thread_summary = self._thread_summary()
        process_summary = self._process_summary()
        memory_summary = self._memory_summary()

        repair_actions: List[str] = []
        issues: List[str] = []
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

    def get_status(self) -> Dict[str, Any]:
        report = self.audit()
        report["running"] = self._running
        return report

    def _patch_asyncio_new_event_loop(self) -> None:
        if self._original_new_event_loop is not None:
            return

        self._original_new_event_loop = asyncio.new_event_loop
        tracker = self._task_tracker

        def _patched_new_event_loop():
            loop = self._original_new_event_loop()
            try:
                tracker.install_loop_hygiene(loop)
            except Exception as exc:
                logger.debug("RuntimeHygiene: failed to install task factory on new loop: %s", exc)
            return loop

        asyncio.new_event_loop = _patched_new_event_loop

    def _patch_threading(self) -> None:
        if self._original_thread_start is not None:
            return

        self._original_thread_start = threading.Thread.start
        manager = self

        def _patched_start(thread: threading.Thread, *args, **kwargs):
            manager._register_thread(thread, source="thread.start")
            return manager._original_thread_start(thread, *args, **kwargs)

        threading.Thread.start = _patched_start

    def _patch_subprocess(self) -> None:
        if self._original_popen_init is not None:
            return

        self._original_popen_init = subprocess.Popen.__init__
        manager = self

        def _patched_init(proc_self, *args, **kwargs):
            manager._original_popen_init(proc_self, *args, **kwargs)
            manager._register_subprocess(proc_self, args=args, kwargs=kwargs)

        subprocess.Popen.__init__ = _patched_init

    def _patch_multiprocessing(self) -> None:
        if self._original_mp_start is not None:
            return

        self._original_mp_start = mp.process.BaseProcess.start
        manager = self

        def _patched_start(proc_self, *args, **kwargs):
            result = manager._original_mp_start(proc_self, *args, **kwargs)
            manager._register_multiprocessing_process(proc_self)
            return result

        mp.process.BaseProcess.start = _patched_start

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
        except Exception as exc:
            logger.debug("RuntimeHygiene: tracemalloc start failed: %s", exc)

    def _adopt_active_child_processes(self) -> None:
        if self._proc is None:
            return
        try:
            children = list(self._proc.children(recursive=True))
        except Exception as exc:
            logger.debug("RuntimeHygiene: existing child adoption skipped: %s", exc)
            return

        if not children and _HAS_PSUTIL:
            try:
                parent_pid = int(os.getpid())
                children = [
                    proc
                    for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline", "status"])
                    if int((proc.info or {}).get("ppid") or 0) == parent_pid
                ]
            except Exception as exc:
                logger.debug("RuntimeHygiene: process_iter child adoption skipped: %s", exc)
                children = []

        tracked_pids = {
            int(record.pid)
            for record in self._process_records.values()
            if record.finished_at is None and getattr(record, "pid", None)
        }
        for child in children:
            try:
                pid = int(getattr(child, "pid", 0) or 0)
            except Exception:
                pid = 0
            if pid and pid in tracked_pids:
                continue
            try:
                command_parts = list(child.cmdline() or [])
            except Exception:
                command_parts = []
            try:
                name = str(child.name() or f"pid:{pid}")
            except Exception:
                name = f"pid:{pid}" if pid else "unknown_child"
            key = -(pid or len(self._process_records) + 1)
            self._process_records[key] = ProcessRecord(
                key=key,
                kind="subprocess",
                name=name,
                source="psutil.adopt_existing_child",
                command=" ".join(str(part) for part in command_parts)[:240] or name,
                pid=pid or None,
            )
            self._process_refs[key] = child
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
        manager = self

        def _wrapped_run(*args, **kwargs):
            record.started_at = time.monotonic()
            record.ident = threading.get_ident()
            try:
                return original_run(*args, **kwargs)
            except BaseException as exc:
                record.exception = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record.finished_at = time.monotonic()

        thread.run = _wrapped_run
        setattr(thread, "_aura_runtime_hygiene_wrapped", True)

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

    def _register_multiprocessing_process(self, proc: mp.Process) -> None:
        key = id(proc)
        self._process_records[key] = ProcessRecord(
            key=key,
            kind="multiprocessing",
            name=getattr(proc, "name", "multiprocessing"),
            source="multiprocessing.Process.start",
            command=getattr(proc, "name", "multiprocessing"),
            pid=getattr(proc, "pid", None),
        )
        self._process_refs[key] = proc

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

    def _refresh_process_records(self) -> None:
        now = time.monotonic()
        for key, proc in list(self._process_refs.items()):
            record = self._process_records.get(key)
            if record is None:
                continue
            if hasattr(proc, "poll"):
                try:
                    return_code = proc.poll()
                except Exception as exc:
                    logger.debug("RuntimeHygiene: subprocess poll failed: %s", exc)
                    return_code = None
                if return_code is not None:
                    record.exit_code = int(return_code)
                    record.finished_at = record.finished_at or now
            elif hasattr(proc, "is_alive"):
                try:
                    alive = proc.is_alive()
                except Exception as exc:
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
                except Exception as exc:
                    logger.debug("RuntimeHygiene: adopted child liveness failed: %s", exc)
                    alive = False
                    status = "error"
                if not alive or status == "zombie":
                    record.finished_at = record.finished_at or now

    def _thread_summary(self) -> Dict[str, Any]:
        now = time.monotonic()
        active = 0
        active_non_daemon = 0
        stale_non_daemon = 0
        sample: List[Dict[str, Any]] = []
        for record in self._thread_records.values():
            if record.finished_at is not None:
                continue
            active += 1
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
        return {
            "active": active,
            "active_non_daemon": active_non_daemon,
            "stale_non_daemon": stale_non_daemon,
            "sample": sample[:5],
        }

    def _process_summary(self) -> Dict[str, Any]:
        active_registered = 0
        active_subprocesses = 0
        active_multiprocessing = 0
        active_registered_pids = set()
        for record in self._process_records.values():
            if record.finished_at is not None:
                continue
            active_registered += 1
            if getattr(record, "pid", None):
                try:
                    active_registered_pids.add(int(record.pid))
                except Exception:
                    pass
            if record.kind == "subprocess":
                active_subprocesses += 1
            elif record.kind == "multiprocessing":
                active_multiprocessing += 1
        rogue_children = 0
        if self._proc is not None:
            try:
                active_children = list(self._proc.children(recursive=True))
            except Exception as exc:
                logger.debug("RuntimeHygiene: child process scan failed: %s", exc)
                active_children = []
            rogue_children = sum(
                1
                for child in active_children
                if int(getattr(child, "pid", 0) or 0) not in active_registered_pids
            )
        return {
            "active_registered": max(0, active_registered),
            "active_subprocesses": max(0, active_subprocesses),
            "active_multiprocessing": max(0, active_multiprocessing),
            "rogue_child_processes": max(0, rogue_children),
        }

    def _memory_summary(self) -> Dict[str, Any]:
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

    def _active_local_model_activity(self) -> List[str]:
        active: List[str] = []
        registries = (
            ("core.brain.llm.mlx_client", "_CLIENTS"),
            ("core.brain.llm.local_server_client", "_SERVER_CLIENTS"),
        )
        for module_name, registry_attr in registries:
            try:
                module = __import__(module_name, fromlist=[registry_attr])
                registry = dict(getattr(module, registry_attr, {}) or {})
            except Exception:
                continue

            for client_path, client in registry.items():
                if client is None or not hasattr(client, "get_lane_status"):
                    continue
                try:
                    lane = client.get_lane_status()
                except Exception:
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
        return active

    def _count_child_processes(self) -> int:
        if self._proc is None:
            return 0
        try:
            return len(self._proc.children(recursive=True))
        except Exception as exc:
            logger.debug("RuntimeHygiene: child process scan failed: %s", exc)
            return 0

    async def _cleanup_child_processes(self) -> None:
        for proc in list(self._process_refs.values()):
            if hasattr(proc, "poll"):
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            await asyncio.to_thread(proc.wait, self.process_shutdown_timeout_s)
                        except Exception:
                            proc.kill()
                except Exception as exc:
                    logger.debug("RuntimeHygiene: subprocess cleanup failed: %s", exc)
            elif hasattr(proc, "is_alive"):
                try:
                    if proc.is_alive():
                        proc.terminate()
                        await asyncio.to_thread(proc.join, self.process_shutdown_timeout_s)
                        if proc.is_alive():
                            proc.kill()
                except Exception as exc:
                    logger.debug("RuntimeHygiene: multiprocessing cleanup failed: %s", exc)

    async def _join_non_daemon_threads(self) -> None:
        for thread in list(self._thread_refs.values()):
            if thread.daemon:
                continue
            if not thread.is_alive():
                continue
            try:
                await asyncio.to_thread(thread.join, self.thread_join_timeout_s)
            except Exception as exc:
                logger.debug("RuntimeHygiene: thread join failed: %s", exc)


_runtime_hygiene: Optional[RuntimeHygieneManager] = None


def get_runtime_hygiene() -> RuntimeHygieneManager:
    global _runtime_hygiene
    if _runtime_hygiene is None:
        _runtime_hygiene = RuntimeHygieneManager()
    return _runtime_hygiene

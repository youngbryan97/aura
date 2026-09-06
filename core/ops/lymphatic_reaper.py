"""
core/ops/lymphatic_reaper.py

Enterprise maintenance for stale temporary files, zombie children, and runtime
cache pressure. The reaper is intentionally conservative: it does not terminate
long-lived child processes unless explicitly enabled.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

from core.observability.metrics import get_metrics
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.resource_observation import get_resource_observer
from core.runtime.task_ownership import create_tracked_task
from core.utils.task_tracker import mark_task_protected

logger = logging.getLogger("Aura.Reaper")
metrics = get_metrics()

STALE_TMP_AGE_S = 86_400
LONG_CHILD_AGE_S = 3_600

_REAPER_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    psutil.Error,
)


def _record_reaper_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "lymphatic_reaper",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation("lymphatic_reaper", error, severity=severity, action=action)
        except TypeError:
            logger.debug("LymphaticReaper degradation could not be recorded: %s", signature_exc)


def _default_data_dir() -> Path:
    """The data directory under whichever state root this process has."""
    try:
        from core.runtime.state_ownership import state_root

        return Path(state_root()) / "data"
    except (ImportError, AttributeError, OSError, RuntimeError):
        return Path("~/.aura/data")


class LymphaticReaper:
    def __init__(self, interval_s: float = 300.0, *, data_dir: Path | None = None):
        self._interval = max(1.0, float(interval_s))
        self._running = False
        self._task: asyncio.Task | None = None
        # Under the state root, not hardcoded to the live one. This fell back
        # to ~/.aura/data, so a process with its own AURA_STATE_ROOT — every
        # test, every probe — had a reaper sweeping the live instance's data.
        self._data_dir = Path(
            data_dir or os.environ.get("AURA_DATA_DIR") or _default_data_dir()
        ).expanduser()
        self._terminate_long_children = os.getenv("AURA_REAPER_TERMINATE_LONG_CHILDREN", "0") == "1"
        self._last_sweep_at = 0.0
        self._last_error = ""
        self._last_error_at = 0.0
        self._consecutive_failures = 0
        self._last_step_errors: dict[str, str] = {}
        self._last_sweep_status: dict[str, Any] = {}

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = create_tracked_task(
            self._run_loop(),
            name="lymphatic_reaper.loop",
        )
        mark_task_protected(self._task, owner="lymphatic_reaper")
        logger.info("Lymphatic Reaper active (interval %.1fs)", self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed %s in core.ops.lymphatic_reaper: %s", type(_exc).__name__, _exc)
            except _REAPER_ERRORS as exc:
                self._remember_error(exc)
                _record_reaper_degradation(
                    exc,
                    action="completed lymphatic reaper shutdown after loop task ended with a known failure",
                    severity="warning",
                )
            self._task = None
        logger.info("Lymphatic Reaper shutdown.")

    def is_alive(self) -> bool:
        """Return True only when the reaper loop is actively supervised."""
        return bool(self._running and self._task is not None and not self._task.done())

    async def _run_loop(self) -> None:
        while self._running:
            sleep_s = self._interval
            try:
                await self.sweep()
                self._consecutive_failures = 0
            except _REAPER_ERRORS as exc:
                self._remember_error(exc)
                sleep_s = min(self._interval * (1 + self._consecutive_failures), self._interval * 6)
                _record_reaper_degradation(
                    exc,
                    action="kept reaper loop alive and backed off after sweep failure",
                    severity="degraded",
                    extra={"consecutive_failures": self._consecutive_failures},
                )
                logger.error("Reaper sweep failed: %s", exc)
            await asyncio.sleep(sleep_s)

    async def sweep(self) -> dict[str, Any]:
        """Execute all maintenance tasks independently.

        Every step is blocking work (process-table walks, directory scans,
        file deletion) and runs on a worker thread — a sweep over a large
        artifact backlog must never stall the event loop it exists to keep
        healthy.
        """
        start_time = time.time()
        logger.debug("Starting lymphatic sweep")

        proc_cleaned = await self._run_step("hunt_orphans", self._hunt_orphans, default=0)
        fs_cleaned = await self._run_step("filesystem_sweep", self._filesystem_sweep, default=0)
        crash_cleaned = await self._run_step(
            "crash_artifact_sweep", self._crash_artifact_sweep, default=0
        )
        memory_defragmented = await self._run_step(
            "defragment_memory", self._defragment_memory, default=False
        )

        duration = time.time() - start_time
        self._last_sweep_at = time.time()
        logger.info(
            "Sweep complete: %d procs reaped, %.1fMB storage reclaimed. (Duration: %.2fs)",
            proc_cleaned,
            (fs_cleaned + crash_cleaned) / (1024 * 1024),
            duration,
        )

        self._emit_metrics(duration, memory_defragmented)
        self._last_sweep_status = {
            "processes_reaped": proc_cleaned,
            "storage_reclaimed_bytes": fs_cleaned + crash_cleaned,
            "crash_artifact_bytes": crash_cleaned,
            "memory_defragmented": memory_defragmented,
            "duration_s": duration,
            "step_errors": dict(self._last_step_errors),
        }
        return dict(self._last_sweep_status)

    async def _run_step(self, name: str, fn: Callable[[], Any], *, default: Any) -> Any:
        try:
            result = await asyncio.to_thread(fn)
            self._last_step_errors.pop(name, None)
            return result
        except _REAPER_ERRORS as exc:
            self._remember_error(exc)
            self._last_step_errors[name] = f"{type(exc).__name__}: {exc}"
            _record_reaper_degradation(
                exc,
                action="skipped one lymphatic maintenance step and continued remaining sweep",
                severity="warning",
                extra={"step": name},
            )
            logger.debug("Reaper step %s failed: %s", name, exc)
            return default

    def _hunt_orphans(self) -> int:
        """Reap zombies; optionally terminate long-lived children when explicitly enabled."""
        count = 0
        table = get_resource_observer().process_table()
        if not table.available:
            raise RuntimeError(f"process_table_unavailable:{table.error}")
        children = [
            process
            for process in table.processes
            if os.getpid() in process.ancestor_pids
        ]
        for child in children:
            try:
                handle = psutil.Process(child.pid)
                if child.status.lower() == "zombie":
                    handle.wait(timeout=0)
                    count += 1
                    continue

                age_s = time.time() - child.create_time
                if age_s > LONG_CHILD_AGE_S and self._terminate_long_children:
                    handle.terminate()
                    count += 1
                elif age_s > LONG_CHILD_AGE_S:
                    logger.debug("Long-lived child retained by policy: pid=%s age_s=%.1f", child.pid, age_s)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as exc:
                _record_reaper_degradation(
                    exc,
                    action="skipped inaccessible child process during orphan sweep",
                    severity="debug",
                    extra={"pid": getattr(child, "pid", None)},
                )
        return count

    def _filesystem_sweep(self) -> int:
        """Clean temporary files, stale locks, and logs below Aura's data tmp directory."""
        reclaimed = 0
        tmp_dir = self._data_dir / "tmp"
        if not tmp_dir.exists():
            return 0

        tmp_root = tmp_dir.resolve()
        for path in tmp_dir.iterdir():
            try:
                if not self._is_safe_tmp_child(path, tmp_root):
                    continue
                if time.time() - self._path_mtime(path) <= STALE_TMP_AGE_S:
                    continue
                if path.is_symlink() or path.is_file():
                    reclaimed += path.lstat().st_size
                    path.unlink()
                elif path.is_dir():
                    reclaimed += self._directory_size(path)
                    shutil.rmtree(path)
            except OSError as exc:
                _record_reaper_degradation(
                    exc,
                    action="left stale tmp path in place after cleanup failed",
                    severity="warning",
                    extra={"path": str(path)[:240]},
                )
                logger.debug("Reaper failed to clean %s: %s", path, exc)
        return reclaimed

    # Crash-artifact retention: newest N files kept per family. One live
    # instance accumulated 18,226 stall dumps (558MB) because pruning only
    # ran when NEW stalls happened — a healthy runtime never drained the
    # backlog. Deletions are batch-bounded so one sweep stays cheap.
    CRASH_ARTIFACT_POLICIES: tuple[tuple[str, str, int], ...] = (
        ("error_logs/stalls", "stall_*.txt", 500),
        ("error_logs/memory", "death_syslog_*.log", 20),
        ("error_logs/memory", "oom_tombstone_*.json", 50),
        ("error_logs/memory", "sentinel_tombstone_*.json", 50),
    )
    CRASH_SWEEP_DELETE_BATCH = 500

    def _crash_artifact_root(self) -> Path:
        try:
            from core.config import config

            return config.paths.project_root / "data"
        except _REAPER_ERRORS:
            return Path(__file__).resolve().parents[2] / "data"

    def _crash_artifact_sweep(self) -> int:
        """Drain crash-artifact backlogs beyond retention. Returns bytes freed."""
        import fnmatch

        reclaimed = 0
        deletions_left = self.CRASH_SWEEP_DELETE_BATCH
        root = self._crash_artifact_root()
        for subdir, pattern, keep in self.CRASH_ARTIFACT_POLICIES:
            if deletions_left <= 0:
                break
            target_dir = root / subdir
            if not target_dir.is_dir():
                continue
            try:
                # Names embed epoch timestamps, so name order is time order.
                names = sorted(
                    entry.name
                    for entry in os.scandir(target_dir)
                    if fnmatch.fnmatch(entry.name, pattern)
                )
            except OSError:
                continue
            excess = len(names) - keep
            if excess <= 0:
                continue
            for name in names[: min(excess, deletions_left)]:
                victim = target_dir / name
                try:
                    reclaimed += victim.stat().st_size
                    victim.unlink()
                    deletions_left -= 1
                except OSError:
                    continue
        return reclaimed

    @staticmethod
    def _is_safe_tmp_child(path: Path, tmp_root: Path) -> bool:
        try:
            if path.is_symlink():
                return path.parent.resolve() == tmp_root
            return path.resolve().parent == tmp_root
        except OSError:
            return False

    @staticmethod
    def _directory_size(path: Path) -> int:
        total = 0
        for child in path.rglob("*"):
            try:
                if child.is_file() and not child.is_symlink():
                    total += child.stat().st_size
            except OSError:
                continue
        return total

    @staticmethod
    def _path_mtime(path: Path) -> float:
        if path.is_symlink():
            return path.lstat().st_mtime
        return path.stat().st_mtime

    def _defragment_memory(self) -> bool:
        """Clear internal Python caches and trigger GC."""
        import gc

        gc.collect()
        return True

    def _emit_metrics(self, duration: float, memory_defragmented: bool) -> None:
        try:
            metrics.gauge("reaper.sweep_duration_s", duration)
            metrics.gauge("reaper.memory_defragmented", 1.0 if memory_defragmented else 0.0)
            metrics.increment("reaper.sweeps_total")
        except _REAPER_ERRORS as exc:
            _record_reaper_degradation(
                exc,
                action="completed sweep while metrics emission failed",
                severity="warning",
            )

    def _remember_error(self, exc: BaseException) -> None:
        self._consecutive_failures += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._last_error_at = time.time()

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "alive": self.is_alive(),
            "interval_s": self._interval,
            "data_dir": str(self._data_dir),
            "terminate_long_children": self._terminate_long_children,
            "last_sweep_at": self._last_sweep_at,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at,
            "consecutive_failures": self._consecutive_failures,
            "last_step_errors": dict(self._last_step_errors),
            "last_sweep_status": dict(self._last_sweep_status),
        }


_reaper: LymphaticReaper | None = None


def get_reaper() -> LymphaticReaper:
    global _reaper
    if _reaper is None:
        _reaper = LymphaticReaper()
    return _reaper

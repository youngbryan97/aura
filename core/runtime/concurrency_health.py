"""Unified concurrency health sampling.

Aura already has a task tracker, lock watchdog, stall watchdog, degradation
tracker, and dead-letter queue.  This module does not duplicate those organs;
it composes their evidence into one fail-visible report so activation audits
can prove the async substrate is alive and not just defensively instrumented.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.errors import record_degradation


@dataclass(frozen=True)
class ConcurrencyHealthReport:
    sampled_at: float
    active_tasks: int
    stale_tasks: int
    failed_tasks_total: int
    recently_failed_tasks: int
    stalled_locks: int
    dlq_total: int
    dlq_recent: int
    degradation_total: int
    pressure: float
    passed: bool
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConcurrencyHealthMonitor:
    """Sample live async/concurrency evidence without owning the subsystems."""

    def __init__(self, *, stale_task_age_s: float = 300.0, recent_window_s: float = 900.0) -> None:
        self.stale_task_age_s = stale_task_age_s
        self.recent_window_s = recent_window_s
        self.last_report: ConcurrencyHealthReport | None = None

    def sample(self) -> ConcurrencyHealthReport:
        now = time.time()
        evidence: dict[str, Any] = {}
        task_stats = self._task_stats()
        lock_snapshot = self._lock_snapshot()
        dlq_stats = self._dlq_stats(now)
        degradation_stats = self._degradation_stats()

        stale_tasks = len(task_stats.get("stale_tasks", []) or [])
        failed_total = int(task_stats.get("failed_total", 0) or 0)
        recent_failures = sum(
            1
            for item in task_stats.get("recently_completed", []) or []
            if item.get("failed") is True
        )
        stalled_locks = sum(
            1
            for lock in lock_snapshot.get("locks", []) or []
            if float(lock.get("held_duration_s", 0.0) or 0.0) > float(lock_snapshot.get("threshold_s", 0.0) or 0.0)
        )
        dlq_recent = int(dlq_stats.get("recent_count", 0) or 0)
        degradation_total = int(degradation_stats.get("total_degradations", 0) or 0)
        pressure = min(
            1.0,
            0.18 * stale_tasks
            + 0.12 * recent_failures
            + 0.20 * stalled_locks
            + 0.04 * min(10, dlq_recent)
            + 0.01 * min(20, degradation_total),
        )
        passed = stale_tasks == 0 and stalled_locks == 0 and recent_failures == 0 and pressure < 0.85

        evidence["task_tracker"] = task_stats
        evidence["lock_watchdog"] = lock_snapshot
        evidence["dead_letter_queue"] = dlq_stats
        evidence["degradations"] = degradation_stats
        report = ConcurrencyHealthReport(
            sampled_at=now,
            active_tasks=int(task_stats.get("active", 0) or 0),
            stale_tasks=stale_tasks,
            failed_tasks_total=failed_total,
            recently_failed_tasks=recent_failures,
            stalled_locks=stalled_locks,
            dlq_total=int(dlq_stats.get("total", 0) or 0),
            dlq_recent=dlq_recent,
            degradation_total=degradation_total,
            pressure=round(pressure, 4),
            passed=passed,
            evidence=evidence,
        )
        self.last_report = report
        return report

    def status(self) -> dict[str, Any]:
        if self.last_report is None:
            return self.sample().to_dict()
        return self.last_report.to_dict()

    def _task_stats(self) -> dict[str, Any]:
        try:
            from core.utils.task_tracker import get_task_tracker

            tracker = get_task_tracker()
            stats = tracker.get_stats()
            stats["stale_tasks"] = tracker.get_stale_tasks(min_age_s=self.stale_task_age_s, include_supervised=True)[:10]
            return stats
        except Exception as exc:
            record_degradation("concurrency_health", exc, severity="warning", action="task tracker unavailable in health sample")
            return {"active": 0, "failed_total": 0, "recently_completed": [], "stale_tasks": [], "error": repr(exc)}

    @staticmethod
    def _lock_snapshot() -> dict[str, Any]:
        try:
            from core.resilience.lock_watchdog import get_lock_watchdog

            return get_lock_watchdog().get_snapshot()
        except Exception as exc:
            record_degradation("concurrency_health", exc, severity="warning", action="lock watchdog unavailable in health sample")
            return {"active_count": 0, "locks": [], "threshold_s": 0.0, "error": repr(exc)}

    def _dlq_stats(self, now: float) -> dict[str, Any]:
        try:
            from core.dead_letter_queue import get_dlq

            queue = get_dlq()
            stats = queue.stats(recent_window_s=self.recent_window_s)
            return stats
        except Exception as exc:
            record_degradation("concurrency_health", exc, severity="warning", action="dead letter queue unavailable in health sample")
            return {"total": 0, "recent_count": 0, "recent": [], "error": repr(exc)}

    @staticmethod
    def _degradation_stats() -> dict[str, Any]:
        try:
            from core.runtime.errors import get_degradation_tracker

            return get_degradation_tracker().status()
        except Exception as exc:
            record_degradation("concurrency_health", exc, severity="warning", action="degradation tracker unavailable in health sample")
            return {"total_degradations": 0, "error": repr(exc)}


async def start_concurrency_health_monitor() -> ConcurrencyHealthMonitor:
    from core.container import ServiceContainer

    monitor = ServiceContainer.get("concurrency_health", default=None)
    if monitor is None:
        monitor = ConcurrencyHealthMonitor()
        ServiceContainer.register_instance("concurrency_health", monitor, required=False)
    monitor.sample()
    return monitor


__all__ = ["ConcurrencyHealthMonitor", "ConcurrencyHealthReport", "start_concurrency_health_monitor"]

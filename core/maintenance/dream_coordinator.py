"""core/maintenance/dream_coordinator.py
Single exclusive gate for all dream/sleep/consolidation subsystems.

Prevents concurrent memory writes from DreamProcessor, DreamerV2,
maintenance dream_cycle, and resilience DreamCycle — which previously
had no coordination and could write to the same episodic store simultaneously.

Priority order (highest → lowest):
  1. resilience/DLQ re-ingestion          (every 5 min if DLQ non-empty)
  2. maintenance/WAL checkpoint + pruning  (every hour)
  3. DreamerV2 full biological sleep cycle (when idle > 10 min, every 2h)
  4. DreamProcessor                        (DEPRECATED — do not re-enable)
"""
import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, Optional

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.DreamCoordinator")

_coordinator: Optional["DreamCoordinator"] = None


def get_dream_coordinator() -> "DreamCoordinator":
    global _coordinator
    if _coordinator is None:
        _coordinator = DreamCoordinator()
    return _coordinator


class DreamCoordinator:
    """Single exclusive async lock for all memory consolidation subsystems."""

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._last_run: dict[str, float] = {}
        self._running: dict[str, bool] = {}
        self._run_count: dict[str, int] = {}
        self._pending: dict[str, dict[str, Any]] = {}

    async def run_if_due(
        self,
        name: str,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        interval_s: float,
        *,
        priority: int = 0,
    ) -> bool:
        """Run the named dream subsystem if its interval has elapsed.

        Returns True if the coroutine ran, False if skipped.
        Thread-safe: the internal asyncio.Lock prevents concurrent runs.
        """
        now = time.monotonic()
        last = self._last_run.get(name, 0.0)
        if now - last < interval_s:
            return False

        try:
            from core.runtime.background_policy import (
                MAINTENANCE_BACKGROUND_POLICY,
                background_activity_reason,
            )

            orchestrator = get_runtime_service("orchestrator", default=None)
            allow_no_user_anchor = name in {"dlq_recovery"}

            reason = background_activity_reason(
                orchestrator,
                profile=MAINTENANCE_BACKGROUND_POLICY,
                allow_no_user_anchor=allow_no_user_anchor,
            )
            if reason:
                self._mark_deferred(name, reason)
                return False
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "dream_coordinator",
                exc,
                action=(
                    f"Skipped dream subsystem '{name}' because maintenance "
                    "admission policy was unavailable"
                ),
            )
            logger.warning(
                "DreamCoordinator: '%s' skipped; maintenance policy unavailable: %s",
                name,
                exc,
            )
            return False

        if self._lock.locked():
            logger.debug(
                "DreamCoordinator: '%s' skipped — another subsystem is running.", name
            )
            return False

        async with self._lock:
            # Double-check after acquiring lock
            now2 = time.monotonic()
            if now2 - self._last_run.get(name, 0.0) < interval_s:
                return False

            self._running[name] = True
            self._run_count[name] = self._run_count.get(name, 0) + 1
            self._pending.pop(name, None)
            logger.info(
                "🌙 DreamCoordinator: running '%s' (run #%d, priority=%d)",
                name, self._run_count[name], priority,
            )
            started = time.monotonic()
            try:
                await coro_factory()
                elapsed = time.monotonic() - started
                self._last_run[name] = time.monotonic()
                logger.info("✅ DreamCoordinator: '%s' complete in %.1fs", name, elapsed)
                return True
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                elapsed = time.monotonic() - started
                record_degradation(
                    "dream_coordinator",
                    exc,
                    action=f"Subsystem '{name}' failed after {elapsed:.1f}s; retrying next interval",
                )
                logger.error("DreamCoordinator: '%s' failed: %s", name, exc)
                return False
            finally:
                self._running[name] = False

    def status(self) -> dict[str, Any]:
        return {
            "last_run_monotonic": {k: round(v, 1) for k, v in self._last_run.items()},
            "currently_running": {k: v for k, v in self._running.items() if v},
            "run_counts": dict(self._run_count),
            "pending": {k: dict(v) for k, v in self._pending.items()},
        }

    def _mark_deferred(self, name: str, reason: str) -> None:
        """Record deferred work without turning normal boot gating into log noise."""

        now = time.monotonic()
        previous = self._pending.get(name)
        count = int((previous or {}).get("count", 0)) + 1
        first_seen = float((previous or {}).get("first_seen_monotonic", now))
        previous_reason = str((previous or {}).get("reason", "") or "")
        self._pending[name] = {
            "reason": reason,
            "count": count,
            "first_seen_monotonic": round(first_seen, 1),
            "last_seen_monotonic": round(now, 1),
        }
        if previous_reason == reason:
            logger.debug(
                "DreamCoordinator: '%s' still queued until admission clears (%s, count=%d).",
                name,
                reason,
                count,
            )
            return
        logger.info(
            "DreamCoordinator: '%s' queued until admission clears (%s).",
            name,
            reason,
        )

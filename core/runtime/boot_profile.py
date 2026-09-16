"""Boot-phase wall-clock profiler.

A live desktop boot has taken 13 minutes with nothing in the logs naming the
slow phase. This module gives boot a flight recorder: each phase's duration
is recorded as it completes, slow phases are called out in real time, and a
summary line plus a JSON artifact land when the runtime reaches ready.

Two APIs:
- ``mark(name)`` — attribute everything since the previous mark to ``name``.
  One-line insertions between existing boot steps; no re-indentation.
- ``phase(name)`` — context manager for isolated timed blocks.

Thread-safe; import-light; never raises into the boot path.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.BootProfile")

SLOW_PHASE_WARN_S = 2.0
PHASE_WARN_S = {
    # This phase starts the full orchestrator and resident 27B worker. A
    # healthy measured desktop boot is about 9-13 seconds here; retain a real
    # alert at 15 seconds rather than labelling every normal launch slow.
    "orchestrator_runtime_start": 15.0,
    # This phase deliberately waits up to 12 seconds for asynchronous model
    # warmup before sealing the signed runtime manifest. Warn only when it
    # exceeds that contract instead of labelling normal readiness convergence
    # as a slow boot.
    "readiness_health_snapshot": 13.0,
}


#: How long the desktop shell waits for the runtime to finish booting before
#: it reloads the window (interface/gui_actor.py reads this). Everything that
#: is part of boot, including the deferred initialisers, has to land inside
#: it — and registration is locked only once they have, or at this line.
DESKTOP_BOOT_WINDOW_S = 90.0


class BootProfiler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phases: list[dict[str, Any]] = []
        self._started_monotonic = time.perf_counter()
        self._started_at = time.time()
        self._last_mark_monotonic = self._started_monotonic

    def _record(self, name: str, duration_s: float, offset_s: float) -> None:
        with self._lock:
            self._phases.append(
                {
                    "name": str(name),
                    "duration_s": round(max(0.0, duration_s), 3),
                    "offset_s": round(max(0.0, offset_s), 3),
                }
            )
        warn_s = PHASE_WARN_S.get(str(name), SLOW_PHASE_WARN_S)
        if duration_s >= warn_s:
            logger.warning(
                "🐢 [BOOT] phase '%s' took %.1fs (offset +%.1fs)",
                name,
                duration_s,
                offset_s,
            )

    def elapsed_s(self) -> float:
        """Seconds since the profiler was created, which is process boot."""
        return max(0.0, time.perf_counter() - self._started_monotonic)

    def mark(self, name: str) -> float:
        """Attribute the time since the previous mark to ``name``."""
        now = time.perf_counter()
        with self._lock:
            since = self._last_mark_monotonic
            self._last_mark_monotonic = now
        duration = now - since
        self._record(name, duration, since - self._started_monotonic)
        return duration

    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._record(name, time.perf_counter() - t0, t0 - self._started_monotonic)

    def total_s(self) -> float:
        return time.perf_counter() - self._started_monotonic

    def phases(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(p) for p in self._phases]

    def summary(self, top: int = 6) -> str:
        phases = self.phases()
        if not phases:
            return f"boot {self.total_s():.1f}s (no phases recorded)"
        slowest = sorted(phases, key=lambda p: p["duration_s"], reverse=True)[:top]
        parts = ", ".join(f"{p['name']}={p['duration_s']:.1f}s" for p in slowest)
        return (
            f"boot {self.total_s():.1f}s across {len(phases)} phases; "
            f"slowest: {parts}"
        )

    def to_report(self) -> dict[str, Any]:
        return {
            "schema": "aura.boot_profile.v1",
            "started_at_unix": self._started_at,
            "total_s": round(self.total_s(), 3),
            "slow_phase_warn_s": SLOW_PHASE_WARN_S,
            "phase_warn_s": dict(PHASE_WARN_S),
            "phases": self.phases(),
        }

    def write_artifact(self, path: Path | None = None) -> Path | None:
        """Persist the profile for post-mortems. Never raises."""
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            target = Path(path) if path is not None else (
                Path("artifacts") / "current" / "boot_profile.json"
            )
            with local_internal_governed_scope(
                "boot_profile.write_artifact",
                receipt_prefix="boot-profile-artifact",
            ):
                get_file_write_gateway().write_text(
                    target,
                    json.dumps(self.to_report(), indent=2),
                    source="boot_profile.write_artifact",
                    durable=False,
                )
            return target
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("boot profile artifact write skipped: %s", exc)
            return None


_profiler: BootProfiler | None = None
_profiler_lock = threading.Lock()


def get_boot_profiler() -> BootProfiler:
    global _profiler
    with _profiler_lock:
        if _profiler is None:
            _profiler = BootProfiler()
        return _profiler


def reset_boot_profiler() -> BootProfiler:
    """Fresh profiler (tests and in-process reboots)."""
    global _profiler
    with _profiler_lock:
        _profiler = BootProfiler()
        return _profiler


__all__ = [
    "BootProfiler",
    "PHASE_WARN_S",
    "SLOW_PHASE_WARN_S",
    "get_boot_profiler",
    "reset_boot_profiler",
]

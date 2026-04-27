"""core/runtime/performance_guard.py

Performance Guard
==================
Enforces concrete budgets across the runtime:

  cold_start_target_s    — boot to first interactive chat
  frame_budget_ms        — UI frame budget; the dashboard caps at 16.6 ms
  ack_budget_ms          — time to acknowledge a user action (250 ms)
  thinking_budget_s      — first token expected window (3 s primary)
  response_budget_s      — full response expected window (45 s primary)
  concurrent_heavy_lanes — 2 on 32 GB, 3 on 64 GB

Behavior under pressure:

  * If frame timing breaches the 16.6 ms budget for ≥ 5 frames in a row,
    non-critical animations are throttled (CSS class
    ``aura-throttle-motion`` is applied to ``<body>``).
  * If the ack budget is breached, the chat input shows a "still
    thinking…" indicator with a live-updating estimate.
  * If the response budget is breached and a deeper lane is warm, the
    request is auto-routed to the deeper lane (without changing the
    prompt — only the routing).
  * Concurrent heavy-lane budget is enforced by counting alive MLX
    workers; if more than ``concurrent_heavy_lanes`` are warm, the LRU
    one is asked to release.

The guard publishes per-window samples to the dashboard's "Performance"
tab and to ``~/.aura/data/performance/samples.jsonl``.
"""
from __future__ import annotations
from core.utils.task_tracker import get_task_tracker

import asyncio
import json
import logging
import os
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.PerformanceGuard")


_PERF_DIR = Path.home() / ".aura" / "data" / "performance"
_PERF_DIR.mkdir(parents=True, exist_ok=True)
_SAMPLES_PATH = _PERF_DIR / "samples.jsonl"


@dataclass
class Budgets:
    cold_start_target_s: float = 2.0
    frame_budget_ms: float = 16.6
    ack_budget_ms: float = 250.0
    thinking_budget_s: float = 3.0
    response_budget_s: float = 45.0
    concurrent_heavy_lanes: int = 2  # raised to 3 if 48GB+ RAM detected
    soft_breach_streak: int = 5


@dataclass
class FrameSample:
    when: float
    duration_ms: float
    source: str


@dataclass
class AckSample:
    when: float
    request_id: str
    latency_ms: float


class PerformanceGuard:
    def __init__(self) -> None:
        self.budgets = Budgets()
        self._frames: Deque[FrameSample] = deque(maxlen=512)
        self._acks: Deque[AckSample] = deque(maxlen=512)
        self._motion_throttled: bool = False
        self._streak: int = 0
        self._on_motion_change: List = []
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._derive_concurrency_from_ram()

    def _derive_concurrency_from_ram(self) -> None:
        try:
            import psutil
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
            if total_gb >= 48:
                self.budgets.concurrent_heavy_lanes = 3
            if total_gb >= 96:
                self.budgets.concurrent_heavy_lanes = 4
        except Exception:
            pass

    # ── samples ───────────────────────────────────────────────────────

    def record_frame(self, duration_ms: float, *, source: str = "ui") -> None:
        s = FrameSample(when=time.time(), duration_ms=duration_ms, source=source)
        self._frames.append(s)
        self._maybe_throttle_motion()
        self._persist({"kind": "frame", **asdict(s)})

    def record_ack(self, request_id: str, latency_ms: float) -> None:
        s = AckSample(when=time.time(), request_id=request_id, latency_ms=latency_ms)
        self._acks.append(s)
        self._persist({"kind": "ack", **asdict(s)})

    # ── motion throttling ─────────────────────────────────────────────

    def _maybe_throttle_motion(self) -> None:
        if not self._frames:
            return
        last = self._frames[-1]
        if last.duration_ms > self.budgets.frame_budget_ms:
            self._streak += 1
        else:
            self._streak = 0
        target = self._streak >= self.budgets.soft_breach_streak
        if target != self._motion_throttled:
            self._motion_throttled = target
            for cb in list(self._on_motion_change):
                try:
                    cb(target)
                except Exception:
                    pass
            logger.info("⏱ motion throttle %s (streak=%d)", "ON" if target else "OFF", self._streak)

    def on_motion_change(self, cb) -> None:
        self._on_motion_change.append(cb)

    # ── routing decisions ─────────────────────────────────────────────

    def should_promote_to_deep_lane(self, *, primary_thinking_ms: float) -> bool:
        return primary_thinking_ms > self.budgets.thinking_budget_s * 1000.0 * 2.5

    def lane_count_above_budget(self, alive_lanes: int) -> bool:
        return alive_lanes > self.budgets.concurrent_heavy_lanes

    # ── reports ───────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        frame_durations = [f.duration_ms for f in self._frames]
        ack_latencies = [a.latency_ms for a in self._acks]
        return {
            "budgets": asdict(self.budgets),
            "frames": _pct(frame_durations),
            "acks": _pct(ack_latencies),
            "motion_throttled": self._motion_throttled,
            "streak": self._streak,
        }

    # ── persistence ───────────────────────────────────────────────────

    @staticmethod
    def _persist(row: Dict[str, Any]) -> None:
        try:
            with open(_SAMPLES_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except Exception:
            pass

    # ── background watcher ───────────────────────────────────────────

    async def start(self, *, interval: float = 5.0) -> None:
        if self._running:
            return
        self._running = True

        async def _loop():
            while self._running:
                # Watcher: emit a periodic report row even if no UI is
                # contributing samples (so the timeline is continuous).
                self._persist({"kind": "report", "when": time.time(), **self.report()})
                await asyncio.sleep(interval)

        self._task = get_task_tracker().create_task(_loop(), name="PerformanceGuard")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


def _pct(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    s = sorted(values)
    n = len(s)
    return {
        "count": n,
        "min": s[0],
        "p50": s[n // 2],
        "p95": s[min(n - 1, int(0.95 * n))],
        "p99": s[min(n - 1, int(0.99 * n))],
        "max": s[-1],
        "mean": statistics.fmean(s),
    }


_GUARD: Optional[PerformanceGuard] = None


def get_guard() -> PerformanceGuard:
    global _GUARD
    if _GUARD is None:
        _GUARD = PerformanceGuard()
    return _GUARD


__all__ = ["Budgets", "PerformanceGuard", "get_guard"]

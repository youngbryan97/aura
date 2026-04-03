from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.FlowControl")


@dataclass
class FlowSnapshot:
    queue_depth: int
    queue_capacity: int
    reply_depth: int
    reply_capacity: int
    busy: bool
    inference_busy: bool
    lag_seconds: float
    governor_mode: str
    load: float
    overloaded: bool


@dataclass
class FlowDecision:
    allow: bool
    priority: int
    defer_seconds: float = 0.0
    reason: str = "allow"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class CognitiveFlowController:
    """Adaptive backpressure and admission control for Aura cognition."""

    def _capacity_of(self, queue_obj: Any, default: int) -> int:
        for attr in ("maxsize", "_maxsize"):
            value = getattr(queue_obj, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return default

    def snapshot(self, orch: Any) -> FlowSnapshot:
        message_queue = getattr(orch, "message_queue", None)
        reply_queue = getattr(orch, "reply_queue", None)
        governor = getattr(orch, "system_governor", None)
        event_loop_monitor = getattr(orch, "_event_loop_monitor", None)
        inference_gate = getattr(orch, "_inference_gate", None)

        queue_depth = int(message_queue.qsize()) if message_queue and hasattr(message_queue, "qsize") else 0
        queue_capacity = self._capacity_of(message_queue, 100)
        reply_depth = int(reply_queue.qsize()) if reply_queue and hasattr(reply_queue, "qsize") else 0
        reply_capacity = self._capacity_of(reply_queue, 50)
        busy = bool(getattr(orch, "is_busy", False))
        inference_busy = bool(inference_gate and getattr(inference_gate, "is_alive", lambda: False)())
        lag_seconds = float(getattr(event_loop_monitor, "_last_lag", 0.0) or 0.0)
        governor_mode = str(getattr(getattr(governor, "current_mode", None), "value", "FULL"))

        queue_ratio = queue_depth / max(1, queue_capacity)
        reply_ratio = reply_depth / max(1, reply_capacity)
        lag_ratio = _clamp01(lag_seconds / 0.25)
        mode_penalty = {
            "FULL": 0.0,
            "DEGRADED_NO_PROACTIVE": 0.2,
            "DEGRADED_CORE_ONLY": 0.4,
        }.get(governor_mode, 0.0)

        load = _clamp01(
            (queue_ratio * 0.40)
            + (reply_ratio * 0.15)
            + (0.20 if busy else 0.0)
            + (0.15 if inference_busy else 0.0)
            + (lag_ratio * 0.10)
            + mode_penalty
        )

        return FlowSnapshot(
            queue_depth=queue_depth,
            queue_capacity=queue_capacity,
            reply_depth=reply_depth,
            reply_capacity=reply_capacity,
            busy=busy,
            inference_busy=inference_busy,
            lag_seconds=lag_seconds,
            governor_mode=governor_mode,
            load=load,
            overloaded=load >= 0.75,
        )

    def admit(self, orch: Any, origin: str, priority: int) -> FlowDecision:
        normalized_origin = str(origin or "").strip().lower()
        is_user_facing = normalized_origin in {
            "user", "voice", "admin", "api", "gui", "websocket", "direct", "external",
        }

        if is_user_facing:
            return FlowDecision(True, priority, 0.0, "user_facing")

        snap = self.snapshot(orch)
        if snap.governor_mode == "DEGRADED_CORE_ONLY" and priority >= 15:
            return FlowDecision(False, priority, 0.0, "degraded_core_only")

        if snap.load >= 0.90 and priority >= 15:
            return FlowDecision(False, priority, 0.0, "hard_backpressure")

        if snap.load >= 0.75 and priority >= 15:
            defer = min(2.0, 0.25 + (snap.load - 0.75) * 5.0)
            return FlowDecision(True, priority + 10, defer, "defer_under_load")

        if snap.load >= 0.60 and priority >= 15:
            return FlowDecision(True, priority + 5, 0.0, "priority_downgrade")

        return FlowDecision(True, priority, 0.0, "allow")

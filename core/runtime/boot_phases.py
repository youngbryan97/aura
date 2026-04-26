"""core/runtime/boot_phases.py

Boot UX state machine
=========================
The user-facing layer that turns the existing boot status (
``core/health/boot_status.py``) into a clean staged screen with the six
named states the polished UX expects:

  starting        — process is up but no organs initialized
  core_ready      — core orchestrator + memory facade healthy
  memory_ready    — memory subsystems hot
  cortex_warming  — local cortex (32B) loading
  cortex_ready    — local cortex hot
  voice_waiting   — voice IO awaiting permission
  autonomy_paused — autonomy is gated until viability is HEALTHY
  ready           — everything required for normal interaction is up
  degraded        — running but some non-critical organ is down
  offline         — not reachable
  recovering      — currently performing self-repair

The dashboard's boot panel reads ``snapshot()`` and renders the linear
checklist; the staged-update layer reads ``ready()`` to gate the chat
input.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.BootPhases")


class BootState(str, Enum):
    STARTING = "starting"
    CORE_READY = "core_ready"
    MEMORY_READY = "memory_ready"
    CORTEX_WARMING = "cortex_warming"
    CORTEX_READY = "cortex_ready"
    VOICE_WAITING = "voice_waiting"
    AUTONOMY_PAUSED = "autonomy_paused"
    READY = "ready"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    RECOVERING = "recovering"


@dataclass
class BootSnapshot:
    state: str
    progress: float  # 0..1
    blocked_on: List[str] = field(default_factory=list)
    organs: Dict[str, str] = field(default_factory=dict)
    started_at: float = 0.0
    last_transition_at: float = field(default_factory=time.time)
    last_change: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BootPhases:
    def __init__(self) -> None:
        self.state: BootState = BootState.STARTING
        self.started_at = time.time()
        self.last_transition_at = self.started_at
        self.last_change: Optional[str] = None
        self.organs: Dict[str, str] = {
            "core": "starting",
            "memory": "starting",
            "cortex": "starting",
            "voice": "waiting",
            "autonomy": "paused",
        }

    def update_organ(self, organ: str, state: str) -> None:
        prev = self.organs.get(organ)
        self.organs[organ] = state
        if prev != state:
            self.last_change = f"{organ}: {prev or '?'} -> {state}"
            self._recompute()

    def _recompute(self) -> None:
        # Simple precedence: any organ in "failed" → DEGRADED; "warming" or
        # "starting" before all-ready → state matches first non-ready organ.
        now = time.time()

        if any(v == "failed" for v in self.organs.values()):
            self._transition(BootState.DEGRADED)
            return
        if self.organs.get("core") in ("starting",):
            self._transition(BootState.STARTING)
            return
        if self.organs.get("core") == "ready" and self.organs.get("memory") in ("starting", "warming"):
            self._transition(BootState.CORE_READY)
            return
        if self.organs.get("memory") == "ready" and self.organs.get("cortex") in ("starting",):
            self._transition(BootState.MEMORY_READY)
            return
        if self.organs.get("cortex") == "warming":
            self._transition(BootState.CORTEX_WARMING)
            return
        if self.organs.get("cortex") == "ready" and self.organs.get("voice") == "waiting":
            self._transition(BootState.VOICE_WAITING)
            return
        if self.organs.get("autonomy") == "paused":
            self._transition(BootState.AUTONOMY_PAUSED)
            return
        if all(v == "ready" for k, v in self.organs.items() if k != "voice"):
            self._transition(BootState.READY)
            return

    def _transition(self, new: BootState) -> None:
        if new == self.state:
            return
        self.state = new
        self.last_transition_at = time.time()
        logger.info("🟢 boot %s", new.value)

    def snapshot(self) -> BootSnapshot:
        # Progress: monotonic across the linear states
        order = [
            BootState.STARTING, BootState.CORE_READY, BootState.MEMORY_READY,
            BootState.CORTEX_WARMING, BootState.CORTEX_READY,
            BootState.VOICE_WAITING, BootState.AUTONOMY_PAUSED, BootState.READY,
        ]
        try:
            idx = order.index(self.state)
            progress = (idx + 1) / len(order)
        except ValueError:
            progress = 0.0
        blocked_on = [k for k, v in self.organs.items() if v not in ("ready", "waiting", "paused")]
        return BootSnapshot(
            state=self.state.value,
            progress=progress,
            blocked_on=blocked_on,
            organs=dict(self.organs),
            started_at=self.started_at,
            last_transition_at=self.last_transition_at,
            last_change=self.last_change,
        )

    def ready(self) -> bool:
        return self.state == BootState.READY


_PHASES: Optional[BootPhases] = None


def get_boot_phases() -> BootPhases:
    global _PHASES
    if _PHASES is None:
        _PHASES = BootPhases()
    return _PHASES


__all__ = ["BootState", "BootSnapshot", "BootPhases", "get_boot_phases"]

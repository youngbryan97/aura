"""core/runtime/conditions.py — typed component conditions (roadmap K6).

Kubernetes objects expose their state as typed conditions — `Ready`,
`Progressing`, `Degraded` — each with a machine-readable reason, a human
message, and the time of the last *transition* (not the last update: a
condition that has been False for six hours says so, which is exactly what
an operator and the incident narrator need to distinguish "just broke"
from "broken all along").

Aura's managed components adopt the same contract. A component owns a
``ComponentConditions`` set; observers read the process-wide registry:

    conditions = get_component_conditions("cortex_lane")
    conditions.set(ConditionType.READY, False,
                   reason="CrashLoopBackOff",
                   message="respawn refused: trip=2, retry in 58s")

Transition timestamps move only when the boolean status flips, so flapping
is visible as recent transitions while steady state shows its true age.
Consumers: the lane reconciler (first adopter), the health surface, and
the incident narrator.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ConditionType(StrEnum):
    READY = "Ready"
    PROGRESSING = "Progressing"
    DEGRADED = "Degraded"


@dataclass(frozen=True)
class Condition:
    type: ConditionType
    status: bool
    reason: str
    message: str
    last_transition_at: float
    last_update_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": str(self.type),
            "status": self.status,
            "reason": self.reason,
            "message": self.message,
            "last_transition_at": self.last_transition_at,
            "last_update_at": self.last_update_at,
        }


class ComponentConditions:
    """Thread-safe condition set for one managed component."""

    def __init__(self, component: str) -> None:
        self.component = component
        self._conditions: dict[ConditionType, Condition] = {}
        self._lock = threading.Lock()

    def set(
        self,
        kind: ConditionType,
        status: bool,
        *,
        reason: str,
        message: str = "",
    ) -> Condition:
        now = time.time()
        flipped = False
        with self._lock:
            existing = self._conditions.get(kind)
            if existing is None:
                condition = Condition(
                    type=kind,
                    status=status,
                    reason=reason,
                    message=message,
                    last_transition_at=now,
                    last_update_at=now,
                )
            elif existing.status != status:
                condition = Condition(
                    type=kind,
                    status=status,
                    reason=reason,
                    message=message,
                    last_transition_at=now,
                    last_update_at=now,
                )
                flipped = True
            else:
                # Same status: refresh reason/message/update time, KEEP the
                # transition time — steady state must show its true age.
                condition = replace(
                    existing, reason=reason, message=message, last_update_at=now
                )
            self._conditions[kind] = condition
        if flipped:
            # A status FLIP is a mind-moment: feed the A5 black box so a later
            # hard fault shows the condition history that led in. OUTSIDE the
            # lock — the recorder must never be able to re-enter this state.
            try:
                from core.runtime.flight_recorder import record_event

                record_event(
                    kind="condition_transition",
                    source=self.component,
                    summary=f"{kind}={status} ({reason}) {message}".strip(),
                )
            except (ImportError, AttributeError, RuntimeError) as exc:
                # Best-effort by design, and the black box is read after a
                # hard fault — a gap in it should be visible in the log.
                logger.debug(
                    "condition transition %s=%s not recorded to the black box (%s: %s)",
                    kind,
                    status,
                    type(exc).__name__,
                    exc,
                )
        return condition

    def get(self, kind: ConditionType) -> Condition | None:
        with self._lock:
            return self._conditions.get(kind)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                str(kind): condition.to_dict()
                for kind, condition in self._conditions.items()
            }


_REGISTRY: dict[str, ComponentConditions] = {}
_REGISTRY_LOCK = threading.Lock()


def get_component_conditions(component: str) -> ComponentConditions:
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(component)
        if existing is None:
            existing = ComponentConditions(component)
            _REGISTRY[component] = existing
        return existing


def all_conditions_report() -> dict[str, dict[str, Any]]:
    """Every managed component's conditions — for /health and the narrator."""
    with _REGISTRY_LOCK:
        components = dict(_REGISTRY)
    return {name: conditions.snapshot() for name, conditions in components.items()}


def reset_conditions_for_test() -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.clear()

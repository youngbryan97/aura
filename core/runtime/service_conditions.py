"""core/runtime/service_conditions.py — "alive" and "ready" are different claims.

Clean-room adoption of the Kubernetes condition convention and finalizer
guarantee. No upstream code is used; the vocabulary and semantics below are
Aura's own, derived from the published API conventions.

Aura's control plane already reconciles desired against observed state and
carries a generation counter — that half of the Kubernetes pattern was adopted
long ago. What it did not have is the part that makes a status *readable*: a
single collapsed `observed_state` cannot express that a model is loaded but not
accepting foreground work, or that a service is running but its dependency is
missing, or that a subsystem is degraded while recovering normally. Those are
independent facts about one object, and squashing them into one enum forces
every consumer to re-derive the distinctions from prose reasons.

Two things are added here.

**Conditions.** A list of independent, named claims, each with its own status,
reason and transition time, plus the object generation the claim was made
about. That last field is what stops a stale status from being read as current:
a condition whose ``observed_generation`` is behind the object's generation
describes a configuration that no longer exists, and callers can see that
rather than being misled by it.

**Finalizers.** A subsystem is not "stopped" because something asked it to
stop. It is stopped when its cleanup has actually completed: transactions
closed, leases released, child processes reaped, temporary credentials
destroyed. A finalizer holds an object in TERMINATING until its named cleanup
runs, so "shut down" stops being a hope and becomes a receipt. Aura has
learned this the hard way — a lane declared cold while its worker still held
20GB is exactly the class of bug this prevents.

Pure policy over data. No I/O, nothing imported above the runtime foundation,
so conditions can be evaluated on boot and shutdown paths alike.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "ConditionStatus",
    "ConditionType",
    "FinalizerOutcome",
    "FinalizerSet",
    "ServiceCondition",
    "ConditionSet",
]


class ConditionStatus(StrEnum):
    """Tri-state on purpose.

    UNKNOWN is not a failure and must not be read as one — it means the probe
    could not run. Collapsing it into FALSE is how "we could not check" becomes
    "it is broken", which then triggers recovery for a healthy service.
    """

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class ConditionType(StrEnum):
    """The independent claims worth making about a runtime service.

    Deliberately small. Each one exists because Aura has a consumer that needs
    exactly this distinction and currently re-derives it from a reason string.
    """

    #: The process/object exists and its own loop is running.
    ALIVE = "alive"
    #: Initialisation finished; internal state is usable.
    READY = "ready"
    #: Will take new foreground work right now. READY without this is the
    #: "loaded but saturated" state that a single enum cannot express.
    ACCEPTING_WORK = "accepting_work"
    #: Everything it needs is present.
    DEPENDENCIES_SATISFIED = "dependencies_satisfied"
    #: Running, but not at full capability. Degraded is not the same as broken.
    DEGRADED = "degraded"
    #: Actively recovering. Distinguishes "failing" from "failing and handling it".
    RECOVERING = "recovering"
    #: Cleanup is in progress and not yet complete.
    TERMINATING = "terminating"


@dataclass
class ServiceCondition:
    """One named claim about a service, and when it last changed."""

    type: ConditionType
    status: ConditionStatus
    reason: str = ""
    message: str = ""
    #: The object generation this claim was made about. A condition behind the
    #: current generation describes configuration that no longer exists.
    observed_generation: int = 0
    last_transition_at: float = field(default_factory=lambda: time.time())

    def is_stale(self, current_generation: int) -> bool:
        return self.observed_generation < current_generation

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "status": self.status.value,
            "reason": self.reason,
            "message": self.message,
            "observed_generation": self.observed_generation,
            "last_transition_at": round(self.last_transition_at, 3),
        }


class ConditionSet:
    """The conditions currently asserted about one service.

    Transition times only move when the STATUS changes. A condition re-asserted
    with the same status keeps its original timestamp, so "how long has this
    been true?" stays answerable — which is the question that actually matters
    when deciding whether something is stuck.
    """

    def __init__(self, generation: int = 0) -> None:
        self._conditions: dict[ConditionType, ServiceCondition] = {}
        self._generation = int(generation)

    @property
    def generation(self) -> int:
        return self._generation

    def bump_generation(self) -> int:
        """Record that the desired configuration changed.

        Every existing condition is now potentially stale, and says so, until a
        controller re-observes at the new generation.
        """
        self._generation += 1
        return self._generation

    def set(
        self,
        condition_type: ConditionType,
        status: ConditionStatus,
        *,
        reason: str = "",
        message: str = "",
        now: float | None = None,
    ) -> ServiceCondition:
        now = time.time() if now is None else now
        existing = self._conditions.get(condition_type)
        changed = existing is None or existing.status is not status
        condition = ServiceCondition(
            type=condition_type,
            status=status,
            reason=reason,
            message=message,
            observed_generation=self._generation,
            # Preserve the original transition when the status is unchanged.
            last_transition_at=now if changed else existing.last_transition_at,
        )
        self._conditions[condition_type] = condition
        return condition

    def get(self, condition_type: ConditionType) -> ServiceCondition | None:
        return self._conditions.get(condition_type)

    def is_true(self, condition_type: ConditionType) -> bool:
        """True only on an explicit TRUE. UNKNOWN is not a yes."""
        condition = self._conditions.get(condition_type)
        return condition is not None and condition.status is ConditionStatus.TRUE

    def stale(self) -> list[ServiceCondition]:
        return [c for c in self._conditions.values() if c.is_stale(self._generation)]

    def duration(self, condition_type: ConditionType, now: float | None = None) -> float | None:
        """How long this condition has held its current status."""
        condition = self._conditions.get(condition_type)
        if condition is None:
            return None
        return max(0.0, (time.time() if now is None else now) - condition.last_transition_at)

    def summary(self) -> str:
        """A short human-readable state, derived rather than asserted."""
        if self.is_true(ConditionType.TERMINATING):
            return "terminating"
        if not self.is_true(ConditionType.ALIVE):
            return "down"
        if not self.is_true(ConditionType.READY):
            return "starting"
        if self.is_true(ConditionType.RECOVERING):
            return "recovering"
        if self.is_true(ConditionType.DEGRADED):
            return "degraded"
        if not self.is_true(ConditionType.ACCEPTING_WORK):
            return "ready_but_saturated"
        return "healthy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self._generation,
            "summary": self.summary(),
            "conditions": [c.to_dict() for c in self._conditions.values()],
            "stale_conditions": [c.type.value for c in self.stale()],
        }


# ── finalizers: stopped means cleaned up ───────────────────────────────────


@dataclass
class FinalizerOutcome:
    """What happened when cleanup ran."""

    name: str
    completed: bool
    error: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "completed": self.completed,
            "error": self.error,
            "duration_s": round(self.duration_s, 4),
        }


class FinalizerSet:
    """Named cleanups that must complete before a service counts as stopped.

    A service is not stopped because something asked it to stop. It is stopped
    when its transactions are closed, its leases released, its children reaped.
    Holding the object in TERMINATING until those finish turns "shut down" from
    a hope into a receipt — and makes an incomplete shutdown *visible* rather
    than silently leaving a 20GB worker alive behind a lane marked cold.
    """

    def __init__(self) -> None:
        self._finalizers: dict[str, Callable[[], Any]] = {}
        self._outcomes: list[FinalizerOutcome] = []

    def add(self, name: str, cleanup: Callable[[], Any]) -> None:
        self._finalizers[str(name)] = cleanup

    def remove(self, name: str) -> bool:
        return self._finalizers.pop(str(name), None) is not None

    @property
    def pending(self) -> tuple[str, ...]:
        return tuple(self._finalizers)

    @property
    def is_clear(self) -> bool:
        return not self._finalizers

    def run_all(self) -> list[FinalizerOutcome]:
        """Run every finalizer, keeping going after a failure.

        One cleanup failing must not strand the others — a released lease is
        worth having even if a temp file could not be removed. A finalizer that
        fails is RETAINED so the object stays terminating and the failure stays
        visible, rather than being dropped to make the shutdown look clean.
        """
        outcomes: list[FinalizerOutcome] = []
        for name in list(self._finalizers):
            cleanup = self._finalizers.get(name)
            if cleanup is None:
                continue
            started = time.monotonic()
            try:
                cleanup()
            except Exception as exc:  # noqa: BLE001 — cleanup must be total
                outcomes.append(
                    FinalizerOutcome(
                        name=name,
                        completed=False,
                        error=f"{type(exc).__name__}: {exc}",
                        duration_s=time.monotonic() - started,
                    )
                )
                continue
            self._finalizers.pop(name, None)
            outcomes.append(
                FinalizerOutcome(
                    name=name,
                    completed=True,
                    duration_s=time.monotonic() - started,
                )
            )
        self._outcomes.extend(outcomes)
        return outcomes

    def to_dict(self) -> dict[str, Any]:
        return {
            "pending": list(self.pending),
            "clear": self.is_clear,
            "outcomes": [o.to_dict() for o in self._outcomes[-16:]],
        }

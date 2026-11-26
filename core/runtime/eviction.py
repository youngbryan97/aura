"""core/runtime/eviction.py — graded eviction under pressure.

Clean-room adoption of the kubelet's eviction manager and PodDisruptionBudget.

The OOM policy in :mod:`core.runtime.oom_policy` answers "who dies when we
are already out of memory". That is the last resort and it is deliberately
crude. The kubelet's real contribution is everything *before* that point:

* **Soft and hard thresholds.** A soft threshold must hold for a grace
  period before it acts, so a transient spike does not shed anything. A
  hard threshold acts immediately, because by then waiting costs more than
  acting. One threshold value cannot serve both purposes, and a system
  with only one either thrashes or reacts too late — Aura has done both.
* **Eviction order derived from declared QoS, not from size.** BestEffort
  organs go first, then Burstable ones ranked by how far they have
  exceeded their own request, and Guaranteed organs are not evicted for
  using what they asked for. This is why declaring requests is worth
  doing: it buys protection.
* **Disruption budgets.** A group may declare a minimum number of members
  that must stay up. Eviction that would breach it is refused, and the
  refusal is recorded rather than silently ignored — otherwise a pressure
  event takes the last replica of something the system needs.
* **Reclaim before eviction.** Ask organs to drop caches first; evict only
  if that did not clear the threshold. Killing something that would have
  released 2GB on request is pure loss.

The kubelet also maps QoS class onto `oom_score_adj` so the kernel's
last-resort choice agrees with the scheduler's policy. This module does the
same to :mod:`core.runtime.oom_policy`, so both layers shed in the same
order rather than fighting.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.runtime.quota import QosClass, get_quota_registry

logger = logging.getLogger("Aura.Eviction")

#: QoS → oom_score_adj, so the OOM killer's last-resort choice agrees with
#: the eviction manager's policy instead of contradicting it.
QOS_OOM_SCORE_ADJ: dict[QosClass, int] = {
    QosClass.GUARANTEED: -998,
    QosClass.BURSTABLE: 100,
    QosClass.BEST_EFFORT: 900,
}


class Signal(StrEnum):
    """Pressure signals thresholds are written against."""

    MEMORY_AVAILABLE_FRACTION = "memory.available_fraction"
    MEMORY_PRESSURE_FULL = "memory.psi_full"
    INFERENCE_PRESSURE_FULL = "inference.psi_full"
    IO_PRESSURE_FULL = "io.psi_full"
    DISK_AVAILABLE_FRACTION = "disk.available_fraction"
    CPU_PRESSURE_FULL = "cpu.psi_full"


class Comparison(StrEnum):
    BELOW = "below"
    ABOVE = "above"


@dataclass(frozen=True)
class Threshold:
    signal: Signal
    comparison: Comparison
    value: float
    #: Soft thresholds must hold this long before acting. Zero makes it hard.
    grace_period_s: float = 0.0
    reason: str = ""

    @property
    def hard(self) -> bool:
        return self.grace_period_s <= 0.0

    def breached(self, observed: float) -> bool:
        if self.comparison is Comparison.BELOW:
            return observed < self.value
        return observed > self.value

    def describe(self, observed: float) -> str:
        return (
            f"{self.signal} is {observed:.4g}, "
            f"{'below' if self.comparison is Comparison.BELOW else 'above'} "
            f"{self.value:.4g}"
            + ("" if self.hard else f" for over {self.grace_period_s:.0f}s")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": str(self.signal),
            "comparison": str(self.comparison),
            "value": self.value,
            "grace_period_s": self.grace_period_s,
            "hard": self.hard,
            "reason": self.reason,
        }


#: Defaults chosen against this host's real shape: 64GB with ~20GB wired
#: by the resident model, so "12% available" is roughly 7.5GB — enough
#: headroom to shed deliberately, not enough to ignore.
DEFAULT_THRESHOLDS: tuple[Threshold, ...] = (
    Threshold(
        signal=Signal.MEMORY_AVAILABLE_FRACTION,
        comparison=Comparison.BELOW,
        value=0.15,
        grace_period_s=60.0,
        reason="sustained memory shortage",
    ),
    Threshold(
        signal=Signal.MEMORY_AVAILABLE_FRACTION,
        comparison=Comparison.BELOW,
        value=0.08,
        reason="acute memory shortage",
    ),
    Threshold(
        signal=Signal.MEMORY_PRESSURE_FULL,
        comparison=Comparison.ABOVE,
        value=0.25,
        grace_period_s=30.0,
        reason="a quarter of wall time lost to memory reclaim",
    ),
    Threshold(
        signal=Signal.DISK_AVAILABLE_FRACTION,
        comparison=Comparison.BELOW,
        value=0.05,
        reason="durable writes are about to start failing",
    ),
)


@dataclass
class Evictable:
    """A candidate the eviction manager may act on."""

    name: str
    #: Free caches without going away. Returns bytes released.
    reclaim: Callable[[], int] | None = None
    #: Stop entirely. Returns bytes released.
    evict: Callable[[], int] | None = None
    #: Current usage per resource kind, for ranking against requests.
    usage: Callable[[], dict[str, float]] | None = None
    #: Group membership for disruption budgets.
    group: str = ""
    #: Higher survives longer within the same QoS class.
    priority: int = 0
    alive: bool = True

    def qos_class(self) -> QosClass:
        return get_quota_registry().qos_class(self.name)


@dataclass(frozen=True)
class DisruptionBudget:
    """The floor a group must keep. Eviction below it is refused."""

    group: str
    min_available: int
    reason: str = ""


@dataclass
class EvictionEvent:
    at: float
    victim: str
    action: str
    qos_class: str
    freed_bytes: int
    threshold: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "victim": self.victim,
            "action": self.action,
            "qos_class": self.qos_class,
            "freed_bytes": self.freed_bytes,
            "threshold": self.threshold,
            "detail": self.detail,
        }


class EvictionManager:
    def __init__(self, thresholds: tuple[Threshold, ...] = DEFAULT_THRESHOLDS) -> None:
        self._lock = threading.RLock()
        self._thresholds = list(thresholds)
        self._candidates: dict[str, Evictable] = {}
        self._budgets: dict[str, DisruptionBudget] = {}
        self._breach_started: dict[str, float] = {}
        self._events: list[EvictionEvent] = []
        self._refusals: list[dict[str, Any]] = []
        self.reclaims = 0
        self.evictions = 0

    # ── registration ──────────────────────────────────────────────────
    def register(self, candidate: Evictable) -> Evictable:
        with self._lock:
            self._candidates[candidate.name] = candidate
        self.sync_oom_scores()
        return candidate

    def unregister(self, name: str) -> None:
        with self._lock:
            self._candidates.pop(name, None)

    def set_budget(self, budget: DisruptionBudget) -> None:
        with self._lock:
            self._budgets[budget.group] = budget

    def add_threshold(self, threshold: Threshold) -> None:
        with self._lock:
            self._thresholds.append(threshold)

    # ── QoS → OOM agreement ───────────────────────────────────────────
    def sync_oom_scores(self) -> dict[str, int]:
        """Push QoS-derived oom_score_adj into the OOM policy.

        Without this the two layers can disagree: eviction protects a
        Guaranteed organ while the OOM killer picks it because it is
        large. The kubelet solves it by making one derive from the other.
        """
        from core.runtime.oom_policy import register_organ

        applied: dict[str, int] = {}
        with self._lock:
            candidates = list(self._candidates.values())
        for candidate in candidates:
            qos = candidate.qos_class()
            adj = QOS_OOM_SCORE_ADJ[qos] - candidate.priority
            register_organ(
                candidate.name,
                oom_score_adj=adj,
                footprint=_bytes_probe(candidate),
                shed=candidate.reclaim or candidate.evict,
                rationale=f"QoS {qos} (priority {candidate.priority})",
                recoverable=candidate.reclaim is not None,
            )
            applied[candidate.name] = adj
        return applied

    # ── signals ───────────────────────────────────────────────────────
    def observe(self) -> dict[str, float]:
        """Read every signal thresholds are written against."""
        observed: dict[str, float] = {}
        try:
            from core.runtime.resource_observation import get_resource_observer

            observer = get_resource_observer()
            memory = observer.memory()
            if memory.available and memory.total_bytes > 0:
                observed[str(Signal.MEMORY_AVAILABLE_FRACTION)] = (
                    memory.available_bytes / memory.total_bytes
                )
            disk = observer.disk("/")
            total = float(getattr(disk, "total_bytes", 0) or 0)
            free = float(getattr(disk, "free_bytes", 0) or 0)
            if total > 0:
                observed[str(Signal.DISK_AVAILABLE_FRACTION)] = free / total
        except Exception:  # noqa: BLE001 — a missing signal must not stop the others
            logger.debug("resource observation for eviction failed", exc_info=True)
        try:
            from core.runtime.pressure_stall import Resource, pressure

            observed[str(Signal.MEMORY_PRESSURE_FULL)] = pressure(Resource.MEMORY)
            observed[str(Signal.INFERENCE_PRESSURE_FULL)] = pressure(Resource.INFERENCE)
            observed[str(Signal.IO_PRESSURE_FULL)] = pressure(Resource.IO)
            observed[str(Signal.CPU_PRESSURE_FULL)] = pressure(Resource.CPU)
        except Exception:  # noqa: BLE001
            logger.debug("PSI read for eviction failed", exc_info=True)
        return observed

    def breached(self, observed: dict[str, float] | None = None) -> list[Threshold]:
        """Thresholds currently breached, honouring soft grace periods."""
        signals = observed if observed is not None else self.observe()
        now = time.monotonic()
        active: list[Threshold] = []
        with self._lock:
            thresholds = list(self._thresholds)
        for threshold in thresholds:
            key = f"{threshold.signal}:{threshold.value}:{threshold.grace_period_s}"
            value = signals.get(str(threshold.signal))
            if value is None or not threshold.breached(value):
                self._breach_started.pop(key, None)
                continue
            if threshold.hard:
                active.append(threshold)
                continue
            started = self._breach_started.setdefault(key, now)
            if now - started >= threshold.grace_period_s:
                active.append(threshold)
        return active

    # ── ranking ───────────────────────────────────────────────────────
    def _overage(self, candidate: Evictable) -> float:
        """How far past its own request this candidate is, as a fraction."""
        spec = get_quota_registry().spec(candidate.name)
        if spec is None or candidate.usage is None:
            return 0.0
        try:
            usage = candidate.usage() or {}
        except Exception:  # noqa: BLE001
            logger.debug("usage probe failed for eviction candidate %s", candidate.name, exc_info=True)
            return 0.0
        worst = 0.0
        for kind, used in usage.items():
            requested = spec.requests.get(str(kind))
            if not requested:
                continue
            worst = max(worst, (float(used) - requested) / requested)
        return worst

    def eviction_order(self) -> list[Evictable]:
        """BestEffort first, then Burstable by overage, Guaranteed never.

        Within a class, lower priority goes first; ties break by name so
        the order is deterministic and therefore explainable.
        """
        with self._lock:
            candidates = [c for c in self._candidates.values() if c.alive]
        rank = {
            QosClass.BEST_EFFORT: 0,
            QosClass.BURSTABLE: 1,
            QosClass.GUARANTEED: 2,
        }
        eligible = [c for c in candidates if c.qos_class() is not QosClass.GUARANTEED]
        return sorted(
            eligible,
            key=lambda c: (rank[c.qos_class()], c.priority, -self._overage(c), c.name),
        )

    def _budget_blocks(self, candidate: Evictable) -> str:
        if not candidate.group:
            return ""
        with self._lock:
            budget = self._budgets.get(candidate.group)
            if budget is None:
                return ""
            alive = sum(
                1
                for c in self._candidates.values()
                if c.group == candidate.group and c.alive
            )
        if alive - 1 < budget.min_available:
            return (
                f"disruption budget for group {candidate.group!r} requires "
                f"{budget.min_available} available; evicting would leave {alive - 1}"
            )
        return ""

    # ── acting ────────────────────────────────────────────────────────
    def enforce(self, *, dry_run: bool = False) -> dict[str, Any]:
        """One evaluation pass: reclaim first, evict only if still breached."""
        observed = self.observe()
        active = self.breached(observed)
        if not active:
            return {"breached": [], "actions": [], "observed": observed}

        worst = max(active, key=lambda t: (t.hard, t.grace_period_s == 0))
        detail = worst.describe(observed.get(str(worst.signal), float("nan")))
        actions: list[dict[str, Any]] = []

        # Phase 1 — reclaim. Killing something that would have released
        # memory on request is pure loss.
        for candidate in self.eviction_order():
            if candidate.reclaim is None:
                continue
            if dry_run:
                actions.append({"victim": candidate.name, "action": "reclaim", "dry_run": True})
                continue
            freed = _safe_call(candidate.reclaim, candidate.name, "reclaim")
            if freed <= 0:
                continue
            self.reclaims += 1
            event = EvictionEvent(
                at=time.time(),
                victim=candidate.name,
                action="reclaim",
                qos_class=str(candidate.qos_class()),
                freed_bytes=freed,
                threshold=str(worst.signal),
                detail=detail,
            )
            self._record(event)
            actions.append(event.to_dict())
            if not self.breached():
                return {
                    "breached": [t.to_dict() for t in active],
                    "actions": actions,
                    "resolved_by": "reclaim",
                    "observed": observed,
                }

        # Phase 2 — eviction, hard thresholds only. A soft threshold that
        # reclaim could not clear becomes hard on its own if it persists.
        if not any(t.hard for t in active):
            return {
                "breached": [t.to_dict() for t in active],
                "actions": actions,
                "resolved_by": "soft threshold: reclaim only",
                "observed": observed,
            }

        for candidate in self.eviction_order():
            if candidate.evict is None:
                continue
            blocked = self._budget_blocks(candidate)
            if blocked:
                self._refusals.append(
                    {"at": time.time(), "victim": candidate.name, "reason": blocked}
                )
                logger.warning("eviction of %s refused: %s", candidate.name, blocked)
                actions.append({"victim": candidate.name, "action": "refused", "reason": blocked})
                continue
            if dry_run:
                actions.append({"victim": candidate.name, "action": "evict", "dry_run": True})
                continue
            freed = _safe_call(candidate.evict, candidate.name, "evict")
            candidate.alive = False
            self.evictions += 1
            event = EvictionEvent(
                at=time.time(),
                victim=candidate.name,
                action="evict",
                qos_class=str(candidate.qos_class()),
                freed_bytes=freed,
                threshold=str(worst.signal),
                detail=detail,
            )
            self._record(event)
            actions.append(event.to_dict())
            if not self.breached():
                break

        return {
            "breached": [t.to_dict() for t in active],
            "actions": actions,
            "observed": observed,
        }

    def _record(self, event: EvictionEvent) -> None:
        with self._lock:
            self._events.append(event)
            if len(self._events) > 128:
                del self._events[:-128]
        logger.warning(
            "eviction %s: %s (%s) freed %.1fMB — %s",
            event.action,
            event.victim,
            event.qos_class,
            event.freed_bytes / 1e6,
            event.detail,
        )
        if event.action == "evict":
            from core.runtime.taint import TaintFlag, taint

            taint(
                TaintFlag.OOM_SHED,
                f"evicted {event.victim} ({event.qos_class}) — {event.detail}",
                subsystem="eviction",
            )

    def report(self) -> dict[str, Any]:
        with self._lock:
            candidates = list(self._candidates.values())
            budgets = [
                {"group": b.group, "min_available": b.min_available, "reason": b.reason}
                for b in self._budgets.values()
            ]
            events = [e.to_dict() for e in self._events[-8:]]
            refusals = list(self._refusals[-8:])
            thresholds = [t.to_dict() for t in self._thresholds]
        return {
            "thresholds": thresholds,
            "candidates": [
                {
                    "name": c.name,
                    "qos_class": str(c.qos_class()),
                    "group": c.group,
                    "priority": c.priority,
                    "alive": c.alive,
                    "can_reclaim": c.reclaim is not None,
                    "can_evict": c.evict is not None,
                }
                for c in candidates
            ],
            "eviction_order": [c.name for c in self.eviction_order()],
            "disruption_budgets": budgets,
            "reclaims": self.reclaims,
            "evictions": self.evictions,
            "recent_events": events,
            "recent_refusals": refusals,
            "currently_breached": [t.to_dict() for t in self.breached()],
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._candidates.clear()
            self._budgets.clear()
            self._breach_started.clear()
            self._events.clear()
            self._refusals.clear()
            self._thresholds = list(DEFAULT_THRESHOLDS)
            self.reclaims = 0
            self.evictions = 0


def _safe_call(fn: Callable[[], int], name: str, action: str) -> int:
    try:
        return max(0, int(fn() or 0))
    except Exception as exc:  # noqa: BLE001
        logger.error("%s of %s failed: %s", action, name, exc)
        return 0


def _bytes_probe(candidate: Evictable) -> Callable[[], int] | None:
    if candidate.usage is None:
        return None

    def probe() -> int:
        try:
            usage = candidate.usage() or {}
        except Exception:  # noqa: BLE001
            logger.debug(
                "memory usage probe failed for eviction candidate %s",
                candidate.name,
                exc_info=True,
            )
            return 0
        return int(usage.get("memory_bytes", 0) or 0)

    return probe


_MANAGER = EvictionManager()


def get_eviction_manager() -> EvictionManager:
    return _MANAGER


def register_evictable(
    name: str,
    *,
    reclaim: Callable[[], int] | None = None,
    evict: Callable[[], int] | None = None,
    usage: Callable[[], dict[str, float]] | None = None,
    group: str = "",
    priority: int = 0,
) -> Evictable:
    return _MANAGER.register(
        Evictable(
            name=name,
            reclaim=reclaim,
            evict=evict,
            usage=usage,
            group=group,
            priority=priority,
        )
    )


def eviction_report() -> dict[str, Any]:
    return _MANAGER.report()


def reset_eviction_for_test() -> None:
    _MANAGER.reset_for_test()


__all__ = [
    "DEFAULT_THRESHOLDS",
    "QOS_OOM_SCORE_ADJ",
    "Comparison",
    "DisruptionBudget",
    "EvictionEvent",
    "EvictionManager",
    "Evictable",
    "Signal",
    "Threshold",
    "eviction_report",
    "get_eviction_manager",
    "register_evictable",
    "reset_eviction_for_test",
]

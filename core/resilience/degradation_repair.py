"""Route degradation records into resilience pressure and repair dispatch.

``record_degradation`` is the visibility boundary. This module makes it causal:
degraded records now affect the body/resilience model, and repeated or critical
incidents can enter the existing self-modification error pipeline with cooldowns.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.task_ownership import (
    close_awaitable,
    create_tracked_task,
    runtime_shutdown_blocks_new_work,
)

logger = logging.getLogger("Aura.Resilience.DegradationRepair")


ServiceGetter = Callable[[str], Any | None]


@dataclass
class DegradationRepairAction:
    subsystem: str
    severity: str
    incident_id: str = ""
    routed_at: float = field(default_factory=time.time)
    resilience_state: str = "unavailable"
    self_modification_status: str = "not_requested"
    self_modification_dispatched: bool = False
    autonomous_repair_status: str = "not_requested"
    immune_status: str = "not_requested"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _service_get(name: str) -> Any | None:
    try:
        from core.runtime.service_registry import get_runtime_service

        return get_runtime_service(name, default=None)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


class DegradationRepairRouter:
    """Connect degradation records to resilience and repair systems."""

    SELF_MODIFICATION_COOLDOWN_S = 300.0
    INCIDENT_REPEAT_THRESHOLD = 5

    def __init__(
        self,
        *,
        service_getter: ServiceGetter | None = None,
        cooldown_seconds: float | None = None,
    ) -> None:
        self._service_getter = service_getter or _service_get
        self._cooldown_seconds = (
            float(cooldown_seconds)
            if cooldown_seconds is not None
            else self.SELF_MODIFICATION_COOLDOWN_S
        )
        self._last_self_modification_dispatch: dict[str, float] = {}
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._owner_loop_lock = threading.RLock()

    def bind_owner_loop(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        *,
        replace: bool = False,
    ) -> bool:
        """Bind cross-thread repair dispatch to the canonical runtime loop."""

        try:
            target = loop or asyncio.get_running_loop()
        except RuntimeError:
            return False
        if target.is_closed() or not target.is_running():
            return False
        with self._owner_loop_lock:
            existing = self._owner_loop
            if (
                not replace
                and existing is not None
                and existing is not target
                and not existing.is_closed()
                and existing.is_running()
            ):
                return False
            self._owner_loop = target
        return True

    def route(
        self,
        *,
        record: Any,
        error: BaseException,
        incident: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> DegradationRepairAction:
        extra = dict(extra or {})
        action = DegradationRepairAction(
            subsystem=str(getattr(record, "subsystem", "unknown")),
            severity=str(getattr(record, "severity", "degraded")),
            incident_id=str(getattr(incident, "incident_id", "") or ""),
        )
        self._route_to_resilience(record, error, action)
        self._route_to_adaptive_immunity(record, error, incident, extra, action)
        self._route_to_self_modification(record, error, incident, extra, action)
        return action

    def _get_service(self, name: str) -> Any | None:
        try:
            return self._service_getter(name)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Service lookup failed for %s: %s", name, exc)
            return None

    def _route_to_resilience(
        self, record: Any, error: BaseException, action: DegradationRepairAction
    ) -> None:
        resilience = self._get_service("resilience_engine") or self._get_service("soma")
        if resilience is None or not hasattr(resilience, "record_failure"):
            action.notes.append("resilience_engine_unavailable")
            return

        severity = str(getattr(record, "severity", "degraded"))
        signal = 0.95 if severity == "critical" else 0.55
        stakes = 0.9 if severity == "critical" else 0.6
        try:
            # Name WHICH degradation this is, so the same one recurring is
            # one standing fact rather than a world getting steadily worse.
            subsystem = getattr(record, "subsystem", "unknown")
            reason = str(
                getattr(record, "reason", None)
                or getattr(record, "detail", None)
                or type(error).__name__
            )[:120]
            try:
                state = resilience.record_failure(
                    domain=f"degradation:{subsystem}",
                    severity=signal,
                    stakes=stakes,
                    signature=f"degradation:{subsystem}:{reason}",
                )
            except TypeError:
                # An engine that predates the signature must still receive the
                # failure. Losing the record entirely because it could not be
                # named is worse than recording it unnamed.
                action.notes.append("resilience_signature_unsupported")
                state = resilience.record_failure(
                    domain=f"degradation:{subsystem}",
                    severity=signal,
                    stakes=stakes,
                )
            action.resilience_state = str(getattr(state, "value", state))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            action.notes.append(f"resilience_route_failed:{type(exc).__name__}")
            logger.debug("Resilience routing failed: %s", exc)

    def _route_to_self_modification(
        self,
        record: Any,
        error: BaseException,
        incident: Any | None,
        extra: dict[str, Any],
        action: DegradationRepairAction,
    ) -> None:
        if not self._should_dispatch_self_modification(record, incident, extra):
            action.self_modification_status = "not_eligible"
            return

        key = f"{getattr(record, 'subsystem', 'unknown')}:{getattr(record, 'error_type', '')}"
        now = time.time()
        last = self._last_self_modification_dispatch.get(key, 0.0)
        if now - last < self._cooldown_seconds:
            action.self_modification_status = "cooldown"
            action.notes.append(f"cooldown_remaining_s:{round(self._cooldown_seconds - (now - last), 1)}")
            return

        engine = self._get_service("self_modification_engine")
        if engine is None or not hasattr(engine, "on_error"):
            action.self_modification_status = "engine_unavailable"
            return

        context = {
            "subsystem": getattr(record, "subsystem", "unknown"),
            "severity": getattr(record, "severity", "degraded"),
            "error_type": getattr(record, "error_type", type(error).__qualname__),
            "error_message": getattr(record, "error_message", str(error)),
            "incident_id": getattr(incident, "incident_id", ""),
            "incident_occurrence_count": int(getattr(incident, "occurrence_count", 0) or 0),
            "degradation_action": getattr(record, "action", ""),
            "error_already_logged": True,
            "extra": extra,
        }
        repair_error = error if isinstance(error, Exception) else RuntimeError(context["error_message"])
        try:
            engine.on_error(
                repair_error,
                context,
                skill_name=str(getattr(record, "subsystem", "degradation")),
                goal=f"Repair degradation in {getattr(record, 'subsystem', 'unknown')}",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            action.self_modification_status = f"dispatch_failed:{type(exc).__name__}"
            logger.debug("Self-modification routing failed: %s", exc)
            return

        self._last_self_modification_dispatch[key] = now
        action.self_modification_dispatched = True
        action.self_modification_status = "dispatched"
        self._route_to_autonomous_repair(record, error, incident, extra, context, action)

    def _route_to_adaptive_immunity(
        self,
        record: Any,
        error: BaseException,
        incident: Any | None,
        extra: dict[str, Any],
        action: DegradationRepairAction,
    ) -> None:
        try:
            immune = self._get_service("adaptive_immune_system")
            if immune is None and (
                bool(extra.get("repair_requested"))
                or str(getattr(record, "severity", "")) == "critical"
            ):
                from core.adaptation.adaptive_immunity import get_adaptive_immune_system

                immune = get_adaptive_immune_system()
            if immune is None or not hasattr(immune, "observe_event"):
                action.immune_status = "service_unavailable"
                return
            event = {
                "source": "runtime_degradation",
                # Runtime degradations originate inside Aura's own substrate.
                # "runtime" was never a valid antigen domain and poisoned
                # persisted replay with an ambiguous origin on every boot.
                "source_domain": "substrate",
                "subsystem": getattr(record, "subsystem", "unknown"),
                "component": getattr(record, "subsystem", "unknown"),
                "severity": getattr(record, "severity", "degraded"),
                "error_type": getattr(record, "error_type", type(error).__qualname__),
                "error_message": getattr(record, "error_message", str(error)),
                "incident_id": getattr(incident, "incident_id", ""),
                "occurrence_count": int(getattr(incident, "occurrence_count", 0) or 0),
                "repair_requested": bool(extra.get("repair_requested")),
            }
            scheduled, reason = self._schedule_coro(
                immune.observe_event(
                    event,
                    anomaly_score=0.9 if event["severity"] == "critical" else 0.65,
                    state_snapshot={"extra": extra},
                ),
                label="adaptive_immunity.observe_degradation",
            )
            action.immune_status = "scheduled" if scheduled else f"deferred:{reason}"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            action.immune_status = f"unavailable:{type(exc).__name__}"
            logger.debug("Adaptive immunity routing failed: %s", exc)

    def _route_to_autonomous_repair(
        self,
        record: Any,
        error: BaseException,
        incident: Any | None,
        extra: dict[str, Any],
        context: dict[str, Any],
        action: DegradationRepairAction,
    ) -> None:
        try:
            from core.resilience.autonomous_repair_executor import (
                AutonomousRepairRequest,
                get_autonomous_repair_executor,
            )

            request = AutonomousRepairRequest(
                subsystem=str(getattr(record, "subsystem", "unknown")),
                severity=str(getattr(record, "severity", "degraded")),
                error_type=str(getattr(record, "error_type", type(error).__qualname__)),
                error_message=str(getattr(record, "error_message", str(error))),
                incident_id=str(getattr(incident, "incident_id", "") or ""),
                occurrence_count=int(getattr(incident, "occurrence_count", 0) or 0),
                goal=f"Repair degradation in {getattr(record, 'subsystem', 'unknown')}",
                context={**context, "extra": extra},
            )
            decision = get_autonomous_repair_executor().enqueue_background(request)
            action.autonomous_repair_status = str(decision.get("status", "unknown"))
            if decision.get("reason"):
                action.notes.append(str(decision["reason"]))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            action.autonomous_repair_status = f"unavailable:{type(exc).__name__}"
            logger.debug("Autonomous repair routing failed: %s", exc)

    def _should_dispatch_self_modification(
        self,
        record: Any,
        incident: Any | None,
        extra: dict[str, Any],
    ) -> bool:
        if bool(extra.get("repair_requested")):
            return True
        if str(getattr(record, "severity", "")) == "critical":
            return True
        error_type = str(getattr(record, "error_type", "") or "").lower()
        subsystem = str(getattr(record, "subsystem", "") or "").lower()
        action = str(getattr(record, "action", "") or "").lower()
        if any(token in error_type for token in ("timeout", "runtimeerror", "attributeerror")):
            return True
        if any(token in subsystem for token in ("cognitive", "mind_tick", "inference", "desktop", "tool")):
            return True
        if "repair" in action or "failed closed" in action:
            return True
        occurrence_count = int(getattr(incident, "occurrence_count", 0) or 0)
        return occurrence_count >= min(2, self.INCIDENT_REPEAT_THRESHOLD)

    def _schedule_on_owner_loop(self, coro: Any, label: str) -> bool:
        if runtime_shutdown_blocks_new_work(
            label,
            resource_kind="adaptive_immunity_task",
        ):
            close_awaitable(coro)
            return False
        try:
            create_tracked_task(
                coro,
                name=label,
                owner="degradation_repair_router",
                bounded=True,
            )
            return True
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            close_awaitable(coro)
            logger.warning("%s task scheduling failed: %s", label, exc)
            return False

    def _schedule_coro(self, coro: Any, *, label: str) -> tuple[bool, str]:
        if runtime_shutdown_blocks_new_work(
            label,
            resource_kind="adaptive_immunity_task",
        ):
            close_awaitable(coro)
            return False, "runtime_shutdown"
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        with self._owner_loop_lock:
            owner_loop = self._owner_loop
            if owner_loop is not None and (
                owner_loop.is_closed() or not owner_loop.is_running()
            ):
                self._owner_loop = None
                owner_loop = None

        if owner_loop is None and running_loop is not None:
            self.bind_owner_loop(running_loop)
            owner_loop = running_loop

        if running_loop is not None and running_loop is owner_loop:
            scheduled = self._schedule_on_owner_loop(coro, label)
            return (
                (True, "current_owner_loop")
                if scheduled
                else (False, "task_schedule_failed")
            )

        if (
            owner_loop is None
            or owner_loop.is_closed()
            or not owner_loop.is_running()
        ):
            close_awaitable(coro)
            return False, "no_owner_loop"
        try:
            owner_loop.call_soon_threadsafe(
                self._schedule_on_owner_loop,
                coro,
                label,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError):
            close_awaitable(coro)
            return False, "owner_loop_unavailable"
        return True, "bound_owner_loop"


_router: DegradationRepairRouter | None = None


def get_degradation_repair_router() -> DegradationRepairRouter:
    global _router
    if _router is None:
        _router = DegradationRepairRouter()
    return _router


def set_degradation_repair_router_for_tests(router: DegradationRepairRouter | None) -> None:
    global _router
    _router = router

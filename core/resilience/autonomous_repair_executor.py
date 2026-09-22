"""Nonblocking autonomous repair execution for runtime faults.

This module is the bridge between "Aura noticed a real defect" and "Aura
attempted a governed repair."  It keeps repair work out of foreground chat,
deduplicates repeated incidents, and delegates code mutation to the existing
self-modification stack instead of inventing a second patch pathway.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger("Aura.Resilience.AutonomousRepair")

ServiceGetter = Callable[[str], Any | None]


def _default_service_getter(name: str) -> Any | None:
    try:
        from core.runtime.service_registry import get_runtime_service

        return get_runtime_service(name, default=None)
    # not a failure: no service here, so the caller falls back to its own default.
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class AutonomousRepairRequest:
    subsystem: str
    error_type: str
    error_message: str
    severity: str = "degraded"
    goal: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    incident_id: str = ""
    occurrence_count: int = 0
    created_at: float = field(default_factory=lambda: time.time())

    @property
    def fingerprint(self) -> str:
        payload = f"{self.subsystem}|{self.error_type}|{self.error_message[:180]}"
        return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fingerprint"] = self.fingerprint
        return data


class AutonomousRepairExecutor:
    """Schedule and run safe autonomous repair cycles without blocking callers."""

    def __init__(
        self,
        *,
        service_getter: ServiceGetter | None = None,
        cooldown_seconds: float = 120.0,
        max_concurrent: int = 1,
        cycle_timeout_seconds: float = 90.0,
        enabled: bool = True,
    ) -> None:
        self._service_getter = service_getter or _default_service_getter
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.max_concurrent = max(1, int(max_concurrent))
        self.cycle_timeout_seconds = max(5.0, float(cycle_timeout_seconds))
        self.enabled = bool(enabled)
        self._last_attempt: dict[str, float] = {}
        self._active = 0
        self._lock = threading.Lock()
        self.stats: dict[str, int] = {
            "scheduled": 0,
            "started": 0,
            "completed": 0,
            "failed": 0,
            "cooldown": 0,
            "skipped": 0,
        }
        self.last_result: dict[str, Any] | None = None

    def enqueue_background(self, request: AutonomousRepairRequest) -> dict[str, Any]:
        """Start a repair request in the background when eligible."""

        decision = self._reserve(request)
        if decision["status"] != "scheduled":
            return decision

        self._schedule(self._run_request(request))
        return decision

    async def execute_now(self, request: AutonomousRepairRequest) -> dict[str, Any]:
        """Run a repair request synchronously for tests or explicit probes."""

        decision = self._reserve(request)
        if decision["status"] != "scheduled":
            return decision
        return await self._run_request(request)

    async def attempt_patch_for_antigen(self, artifact: Any, antigen: Any) -> dict[str, Any]:
        """Adaptive-immune patch proposal adapter."""

        source_domain = str(getattr(antigen, "source_domain", "substrate") or "substrate")
        source = str(getattr(antigen, "source", "") or "")
        error_signature = str(getattr(antigen, "error_signature", "") or "")
        if (
            source_domain == "environment"
            or source.endswith(":metabolism")
            or error_signature == "resource_pressure"
        ):
            self.stats["skipped"] += 1
            return {
                "attempted": False,
                "applied": False,
                "status": "environmental_observation",
                "notes": (
                    "environmental resource telemetry is handled by resource "
                    "policy, not source-code repair"
                ),
                "fingerprint": "",
            }

        request = AutonomousRepairRequest(
            subsystem=str(getattr(antigen, "subsystem", "") or getattr(artifact, "component", "immune")),
            error_type=str(getattr(antigen, "error_signature", "") or getattr(artifact, "kind", "immune_patch")),
            error_message=str(getattr(artifact, "notes", "") or getattr(antigen, "source", "immune patch proposal")),
            severity="degraded",
            goal=f"Repair immune antigen in {getattr(artifact, 'component', 'unknown component')}",
            context={
                "origin": "adaptive_immune_system",
                "artifact_id": getattr(artifact, "artifact_id", ""),
                "artifact_kind": getattr(getattr(artifact, "kind", ""), "value", str(getattr(artifact, "kind", ""))),
                "component": getattr(artifact, "component", ""),
                "antigen_id": getattr(antigen, "antigen_id", ""),
                "danger": getattr(antigen, "danger", None),
            },
        )
        decision = self.enqueue_background(request)
        return {
            "attempted": decision["status"] == "scheduled",
            "applied": False,
            "status": decision["status"],
            "notes": decision.get("reason", "immune patch scheduled for autonomous repair"),
            "fingerprint": request.fingerprint,
        }

    def _reserve(self, request: AutonomousRepairRequest) -> dict[str, Any]:
        if not self.enabled:
            self.stats["skipped"] += 1
            return {"status": "disabled", "fingerprint": request.fingerprint}

        now = time.time()
        with self._lock:
            last = self._last_attempt.get(request.fingerprint, 0.0)
            if now - last < self.cooldown_seconds:
                self.stats["cooldown"] += 1
                return {
                    "status": "cooldown",
                    "fingerprint": request.fingerprint,
                    "reason": f"cooldown_remaining_s:{round(self.cooldown_seconds - (now - last), 1)}",
                }
            if self._active >= self.max_concurrent:
                self.stats["skipped"] += 1
                return {
                    "status": "busy",
                    "fingerprint": request.fingerprint,
                    "reason": "repair executor already active",
                }
            self._active += 1
            self._last_attempt[request.fingerprint] = now
            self.stats["scheduled"] += 1

        return {
            "status": "scheduled",
            "fingerprint": request.fingerprint,
            "request": request.to_dict(),
        }

    async def _run_request(self, request: AutonomousRepairRequest) -> dict[str, Any]:
        self.stats["started"] += 1
        result: dict[str, Any] = {
            "status": "started",
            "fingerprint": request.fingerprint,
            "request": request.to_dict(),
        }
        try:
            engine = self._service_getter("self_modification_engine")
            if engine is None:
                result.update({"status": "engine_unavailable", "success": False})
                return result

            repair_error = RuntimeError(request.error_message)
            context = dict(request.context)
            context.update(
                {
                    "subsystem": request.subsystem,
                    "severity": request.severity,
                    "error_type": request.error_type,
                    "error_message": request.error_message,
                    "incident_id": request.incident_id,
                    "incident_occurrence_count": request.occurrence_count,
                    "autonomous_repair": True,
                }
            )
            on_error = getattr(engine, "on_error", None)
            if callable(on_error) and not bool(context.get("error_already_logged")):
                on_error(
                    repair_error,
                    context,
                    skill_name=request.subsystem,
                    goal=request.goal or f"Repair degradation in {request.subsystem}",
                )
                await asyncio.sleep(0)

            cycle = getattr(engine, "run_autonomous_cycle", None)
            if not callable(cycle):
                result.update({"status": "cycle_unavailable", "success": False})
                return result

            cycle_result = await asyncio.wait_for(
                cycle(),
                timeout=self.cycle_timeout_seconds,
            )
            result.update(
                {
                    "status": "completed",
                    "success": bool(cycle_result.get("success", False))
                    if isinstance(cycle_result, dict)
                    else False,
                    "cycle_result": cycle_result,
                }
            )
            return result
        except (TimeoutError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.warning("Autonomous repair failed for %s: %s", request.fingerprint, exc)
            result.update(
                {
                    "status": "failed",
                    "success": False,
                    "error_type": type(exc).__qualname__,
                    "error": str(exc)[:500],
                }
            )
            return result
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)
            if result.get("status") == "completed":
                self.stats["completed"] += 1
            elif result.get("status") in {"failed", "engine_unavailable", "cycle_unavailable"}:
                self.stats["failed"] += 1
            self.last_result = result

    @staticmethod
    def _schedule(coro: Any) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            def _run_in_thread() -> None:
                try:
                    asyncio.run(coro)
                except asyncio.CancelledError:
                    return
                except (RuntimeError, OSError, TypeError, ValueError) as exc:
                    logger.warning("Autonomous repair thread failed: %s", exc)

            thread = threading.Thread(
                target=_run_in_thread,
                name="aura-autonomous-repair",
                daemon=True,
            )
            thread.start()
            return

        try:
            create_tracked_task(
                coro,
                name="aura-autonomous-repair",
                owner="autonomous_repair_executor",
                bounded=True,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.warning("Autonomous repair task scheduling failed: %s", exc)


_EXECUTOR: AutonomousRepairExecutor | None = None


def get_autonomous_repair_executor() -> AutonomousRepairExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = AutonomousRepairExecutor()
    return _EXECUTOR


def set_autonomous_repair_executor_for_tests(executor: AutonomousRepairExecutor | None) -> None:
    global _EXECUTOR
    _EXECUTOR = executor

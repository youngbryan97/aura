"""Skynet — DistributedResilienceCore.

Health monitoring with a repair step: an unhealthy subsystem is asked to
recover itself, on a cooldown, through a method it publishes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from core.fictional.common import record_fictional_degradation

logger = logging.getLogger("Aura.FictionalSynthesis")


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 5: SKYNET — DistributedResilienceCore
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SubsystemStatus:
    name: str
    healthy: bool
    failure_count: int
    last_error: str = ""
    last_checked_at: float = field(default_factory=time.time)
    last_transition_at: float = field(default_factory=time.time)
    last_reported_at: float = 0.0


class DistributedResilienceCore:
    """
    Derived from: Skynet (Terminator)
    """

    def __init__(self):
        self._subsystems: dict[str, SubsystemStatus] = {}
        self._recovery_attempts: dict[str, float] = {}
        self._pending_repairs: dict[str, dict[str, Any]] = {}
        self._recovery_tasks: set[asyncio.Task] = set()
        self._running = False
        # Created in start_monitoring (needs a running loop); declared here so
        # stop() before start is a no-op rather than an AttributeError.
        self._stop_event: asyncio.Event | None = None
        try:
            self._failure_threshold = max(
                1,
                int(os.getenv("AURA_RESILIENCE_FAILURE_THRESHOLD", "2")),
            )
        except (TypeError, ValueError):
            self._failure_threshold = 2
        try:
            self._reminder_interval_s = max(
                60.0,
                float(os.getenv("AURA_RESILIENCE_REMINDER_INTERVAL_S", "600")),
            )
        except (TypeError, ValueError):
            self._reminder_interval_s = 600.0

    def register_subsystem(self, name: str):
        self._subsystems[name] = SubsystemStatus(name=name, healthy=True, failure_count=0)

    def record_failure(self, name: str, error: str = "") -> bool:
        if name in self._subsystems:
            status = self._subsystems[name]
            was_healthy = status.healthy
            status.failure_count += 1
            status.last_error = error
            status.last_checked_at = time.time()
            if status.failure_count >= self._failure_threshold:
                status.healthy = False
            if was_healthy and not status.healthy:
                status.last_transition_at = status.last_checked_at
                return True
        return False

    def record_success(self, name: str) -> bool:
        if name in self._subsystems:
            status = self._subsystems[name]
            recovered = not status.healthy
            status.failure_count = 0
            status.healthy = True
            status.last_error = ""
            status.last_checked_at = time.time()
            if recovered:
                status.last_transition_at = status.last_checked_at
            return recovered
        return False

    def _report_failure(self, name: str, error: str) -> None:
        became_unhealthy = self.record_failure(name, error)
        status = self._subsystems[name]
        now = time.time()
        if became_unhealthy:
            status.last_reported_at = now
            logger.error(
                "🛡️  Skynet: Subsystem '%s' became UNHEALTHY after %d consecutive probes: %s",
                name,
                status.failure_count,
                error,
            )
            self._request_recovery(name, error)
            return
        if status.healthy:
            if status.failure_count == 1:
                status.last_reported_at = now
                logger.warning(
                    "🛡️  Skynet: Subsystem '%s' health probe failed (1/%d): %s",
                    name,
                    self._failure_threshold,
                    error,
                )
            return
        if now - status.last_reported_at >= self._reminder_interval_s:
            status.last_reported_at = now
            logger.error(
                "🛡️  Skynet: Subsystem '%s' remains UNHEALTHY for %.0fs: %s",
                name,
                max(0.0, now - status.last_transition_at),
                error,
            )

    #: A subsystem may be asked to recover at most this often. Repair that
    #: retries without a floor is how a wedged service becomes a hot loop.
    RECOVERY_COOLDOWN_S = 600.0
    #: Methods a subsystem may expose to repair itself, most specific
    #: first. Anything not on this list is not called: "restart" on an
    #: unknown object is not a contract, it is a guess with side effects.
    RECOVERY_METHODS = ("recover", "reinitialize", "reconnect", "restart")

    def _request_recovery(self, name: str, error: str) -> bool:
        """Ask a failed subsystem to repair itself, once, on a cooldown.

        The monitor detected failures and logged them, and that was the
        whole of "resilience" — no restart, no quarantine, no repair, and
        no record that repair was never attempted (CP126 ``2b26af7d``).

        What it will NOT do is decide on its own to restart a process or
        tear down a service it did not create. It calls a repair method
        the subsystem itself publishes, and when there is none it records
        a repair request so the absence is visible instead of silent.
        """
        now = time.time()
        last = self._recovery_attempts.get(name, 0.0)
        if now - last < self.RECOVERY_COOLDOWN_S:
            return False
        self._recovery_attempts[name] = now

        from core.container import ServiceContainer

        service = ServiceContainer.get(name, default=None)
        method = None
        if service is not None:
            for candidate in self.RECOVERY_METHODS:
                attr = getattr(service, candidate, None)
                if callable(attr):
                    method = (candidate, attr)
                    break

        if method is None:
            record_fictional_degradation(
                RuntimeError(f"subsystem {name} is unhealthy and publishes no repair method"),
                severity="warning",
                action=(
                    "recorded a repair request; the resilience monitor will not "
                    "restart a service it did not create"
                ),
            )
            self._pending_repairs[name] = {
                "error": str(error)[:200],
                "requested_at": now,
                "reason": "no_repair_method_published",
            }
            return False

        label, callable_method = method
        self._pending_repairs[name] = {
            "error": str(error)[:200],
            "requested_at": now,
            "reason": f"calling {label}()",
        }
        logger.warning("🛡️  Skynet: asking '%s' to %s() after repeated failures.", name, label)
        try:
            result = callable_method()
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_fictional_degradation(
                exc,
                severity="warning",
                action=f"subsystem {name} raised while attempting {label}()",
            )
            return False
        if asyncio.iscoroutine(result):
            # Repair on the monitor's own loop, bounded, so a hung repair
            # cannot stall health reporting for every other subsystem.
            from core.utils.task_tracker import get_task_tracker

            task = get_task_tracker().track(
                self._await_recovery(name, label, result),
                name=f"skynet.recovery.{name}",
            )
            self._recovery_tasks.add(task)
            task.add_done_callback(self._recovery_tasks.discard)
        return True

    async def _await_recovery(self, name: str, label: str, awaitable: Any) -> None:
        try:
            await asyncio.wait_for(awaitable, timeout=self.HEALTH_PROBE_TIMEOUT_S * 4)
        except (TimeoutError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_fictional_degradation(
                exc,
                severity="warning",
                action=f"subsystem {name} did not complete {label}() within the repair window",
            )

    def pending_repairs(self) -> dict[str, dict[str, Any]]:
        """Repairs requested and not yet observed to have worked."""
        return dict(self._pending_repairs)

    def _report_success(self, name: str) -> None:
        status = self._subsystems[name]
        unhealthy_since = status.last_transition_at
        if self.record_success(name):
            recovered_at = time.time()
            status.last_reported_at = recovered_at
            self._pending_repairs.pop(name, None)
            logger.info(
                "🛡️  Skynet: Subsystem '%s' RECOVERED after %.0fs.",
                name,
                max(0.0, recovered_at - unhealthy_since),
            )

    @staticmethod
    def _monitor_targets(service_container: Any) -> list[str]:
        required_targets = ["orchestrator", "capability_engine", "memory_facade"]
        optional_targets = ["server", "voice_engine", "live_learner"]
        targets = list(required_targets)
        for target in optional_targets:
            try:
                if service_container.has(target):
                    targets.append(target)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
        return targets

    async def start_monitoring(self):
        if self._running:
            return
        self._running = True
        from core.container import ServiceContainer
        
        # Required organs are always monitored. Optional surfaces are monitored
        # only when their boot profile registered them.
        core_targets = self._monitor_targets(ServiceContainer)
        for target in core_targets:
            self.register_subsystem(target)

        logger.info("🛡️  Skynet ResilienceCore monitoring %d subsystems.", len(core_targets))
        
        # Event-driven interval so stop() is honored immediately rather than
        # after a full monitor period.
        self._stop_event = asyncio.Event()
        try:
            while self._running:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=60)
                    break  # stop() fired
                except TimeoutError:
                    pass  # normal interval elapsed
                if not self._running:
                    break
                for name, _status in list(self._subsystems.items()):
                    try:
                        service = ServiceContainer.get(name, default=None)
                    except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                        self._report_failure(name, f"service lookup failed: {e}")
                        record_fictional_degradation(
                            e,
                            action=f"marked resilience target {name} degraded after service lookup failed",
                        )
                        logger.debug("Skynet service lookup error for %s: %s", name, e)
                        continue
                    if service is None:
                        try:
                            from core.runtime.ablation_policy import service_intentionally_lesioned

                            if service_intentionally_lesioned(name):
                                logger.info(
                                    "Resilience monitor observed intentional ablation for subsystem '%s'; missing-service repair suppressed.",
                                    name,
                                )
                                continue
                        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                            record_fictional_degradation(
                                e,
                                action=f"continued resilience monitoring without ablation marker for {name}",
                            )
                        self._report_failure(name, "service missing from container")
                        continue

                    verdict, failure_detail = await self._probe_service_health(name, service)

                    if verdict is True:
                        self._report_success(name)
                    elif verdict is False:
                        self._report_failure(name, failure_detail)
                    else:
                        # UNVERIFIED, not healthy. Reporting success here
                        # manufactured positive health evidence for a service
                        # that never answered a health question — and a
                        # recovery counter advanced on the strength of it.
                        logger.debug(
                            "Skynet: subsystem '%s' exposes no recognized health "
                            "convention; leaving state unchanged (%s).",
                            name,
                            failure_detail,
                        )
        finally:
            self._running = False

    # A subsystem's health probe is arbitrary third-party code called from the
    # monitor loop; without a deadline one slow probe stalls health reporting
    # for EVERY other subsystem.
    HEALTH_PROBE_TIMEOUT_S = 5.0

    async def _probe_service_health(
        self, name: str, service: Any
    ) -> tuple[bool | None, str]:
        """Return (True healthy, False unhealthy, None unverified) plus detail.

        Recognizes the health conventions actually used across this codebase
        instead of only an exact ``healthy is False`` field — a service
        reporting ``is_alive: False``, ``ok: False``, or ``status: "failed"``
        was previously read as healthy.
        """
        probe = None
        for attr in ("get_status", "health", "get_health"):
            candidate = getattr(service, attr, None)
            if callable(candidate):
                probe = candidate
                break
        if probe is None:
            for attr in ("is_alive", "is_ready", "is_healthy"):
                candidate = getattr(service, attr, None)
                if callable(candidate):
                    probe = candidate
                    break
        if probe is None:
            return None, "no recognized health probe"

        try:
            if asyncio.iscoroutinefunction(probe):
                stats = await asyncio.wait_for(probe(), timeout=self.HEALTH_PROBE_TIMEOUT_S)
            else:
                # Off-loop: a synchronous probe may block on IO or a lock.
                stats = await asyncio.wait_for(
                    asyncio.to_thread(probe), timeout=self.HEALTH_PROBE_TIMEOUT_S
                )
        except TimeoutError:
            return False, f"health probe exceeded {self.HEALTH_PROBE_TIMEOUT_S:.0f}s"
        except (OSError, ConnectionError, RuntimeError, TypeError, ValueError, AttributeError) as e:
            record_fictional_degradation(
                e,
                action=f"marked resilience target {name} degraded after health probe failed",
            )
            logger.debug("Skynet health check error for %s: %s", name, e)
            return False, f"health probe raised {type(e).__name__}: {e}"

        if isinstance(stats, bool):
            return stats, "" if stats else "health probe returned False"
        if not isinstance(stats, dict):
            return None, f"unrecognized health payload {type(stats).__name__}"

        detail = str(
            stats.get("health_reason")
            or stats.get("reason")
            or stats.get("last_error")
            or "health check reported unhealthy"
        )[:240]

        for key in ("healthy", "ok", "is_alive", "is_ready", "alive", "ready"):
            if key in stats:
                if stats[key] is False:
                    return False, detail
                if stats[key] is True:
                    return True, ""
        status_text = str(stats.get("status") or stats.get("state") or "").strip().lower()
        if status_text:
            if status_text in {"healthy", "ok", "ready", "running", "alive", "active", "up"}:
                return True, ""
            if status_text in {"failed", "error", "dead", "down", "unhealthy", "crashed", "stopped"}:
                return False, detail
        if stats.get("degraded") is True:
            return False, detail
        return None, "health payload carried no recognized verdict"

    def stop(self):
        self._running = False
        # Wake the monitor immediately instead of letting it finish a full
        # sleep interval — shutdown should be deterministic, not up to 60s.
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            try:
                stop_event.set()
            except RuntimeError:
                logger.debug("Skynet stop event could not be set; loop will exit on next tick.")


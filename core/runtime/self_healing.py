"""core/runtime/self_healing.py

Self-Healing Loop
==================
A small, hot-path loop that watches for cognitive-module hangs and
restarts the offending module without losing receipts. It is
complementary to StabilityGuardian (broad health checks) and the
OrganSupervisor (subprocess restart): self_healing operates *inside*
the same process, on async tasks that have stopped progressing.

Detection signals:

  * a registered "heartbeat" callable hasn't been called in N seconds
  * an asyncio.Task referenced in the registry is in a "hanging" state
    (still pending after a grace window beyond its declared budget)

Repair actions:

  * cancel the hanging task
  * call the module's ``restart_async()`` if available; otherwise
    re-instantiate via ServiceContainer
  * record an action receipt + a phenomenal envelope (severity = 0.5)
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.service_registry import get_runtime_service
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.state_ownership import state_root
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger("Aura.SelfHealing")

_SELF_HEALING_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

_DIR = state_root() / "data" / "self_healing"
_DIR.mkdir(parents=True, exist_ok=True)
_LEDGER = _DIR / "events.jsonl"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _bounded_env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        configured = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        configured = default
    return min(maximum, max(minimum, configured))


def _which_invariants_stopped_holding(subsystem: str) -> tuple[str, ...]:
    """The invariants that are actually failing, rather than a guess at one.

    A repair verified against an invariant nobody checked is verified against
    nothing. The declared invariants are already scoped, so this asks the
    scope named after the failing subsystem and falls back to every scope
    when there is no such scope — a wider check, never a silent empty one.
    """

    try:
        from core.verify.invariants import get_registry, verify
    except ImportError:
        return ()
    try:
        scopes = set(get_registry().scopes())
        wanted = [one for one in (subsystem, subsystem.split("_")[0]) if one in scopes]
        report = verify(*wanted) if wanted else verify()
        return tuple(sorted({str(one.invariant) for one in report.violations}))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Could not read which invariants stopped holding: %s", exc)
        return ()


def _diagnose_this_failure(subsystem: str, reason: str) -> Any:
    """Recognise a failure before escalating, or None when that is impossible.

    Runs on a worker thread. Everything it touches is either bounded or
    optional, and a diagnosis that cannot be made must never stop a repair
    that can.
    """

    try:
        from core.resilience.unknown_failure import look_at_this_failure
    except ImportError:
        return None
    try:
        return look_at_this_failure(
            subsystem,
            reason,
            broken_invariants=_which_invariants_stopped_holding(subsystem),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        logger.debug("Could not diagnose %s: %s", subsystem, exc)
        return None


def _learn_this_failure_class(diagnosis: Any, module_path: str) -> str:
    """Name a novel failure whose repair held, so next time it is known."""

    try:
        from core.resilience.unknown_failure import a_repair_that_held
    except ImportError:
        return ""
    try:
        called = str(module_path).replace("/", ".").removesuffix(".py")
        return a_repair_that_held(diagnosis, called=called)
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        logger.debug("Could not learn the failure class for %s: %s", module_path, exc)
        return ""


def _deep_repair_block_reason(origin: str = "self_healing_deep_repair") -> str:
    """Return a reason deep repair must not run in this runtime mode."""

    if not _env_flag("AURA_ENABLE_DEEP_REPAIR", True):
        return "deep_repair_disabled"
    try:
        from core.runtime.background_policy import background_loop_start_reason

        return background_loop_start_reason(
            origin,
        )
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation(
            "self_healing",
            exc,
            action="blocked deep repair because background policy was unavailable",
            receipt_required=True,
        )
        return "background_policy_unavailable"


@dataclass
class WatchEntry:
    name: str
    last_heartbeat_at: float = field(default_factory=time.time)
    expected_interval_s: float = 30.0
    restart_async: Callable[[], Awaitable[None]] | None = None
    container_key: str | None = None
    restarts: int = 0
    restart_failures: int = 0
    last_restart_at: float = 0.0
    last_restart_error: str = ""


class SelfHealing:
    shutdown_timeout_s = 8.0

    def __init__(self) -> None:
        self._watches: dict[str, WatchEntry] = {}
        self._deep_repairs: dict[str, asyncio.Task] = {}
        self._module_path_cache: dict[str, str | None] = {}
        self._task: asyncio.Task | None = None
        self._running = False
        self._stop_lock = asyncio.Lock()
        self._ledger_write_timeout_s = _bounded_env_float(
            "AURA_SELF_HEALING_LEDGER_TIMEOUT_S",
            5.0,
            minimum=1.0,
            maximum=30.0,
        )
        self._restart_timeout_s = _bounded_env_float(
            "AURA_SELF_HEALING_RESTART_TIMEOUT_S",
            15.0,
            minimum=1.0,
            maximum=120.0,
        )

    def watch(
        self,
        name: str,
        *,
        expected_interval_s: float = 30.0,
        restart_async: Callable[[], Awaitable[None]] | None = None,
        container_key: str | None = None,
    ) -> None:
        self._watches[name] = WatchEntry(
            name=name,
            expected_interval_s=expected_interval_s,
            restart_async=restart_async,
            container_key=container_key,
        )

    def heartbeat(self, name: str) -> None:
        w = self._watches.get(name)
        if w is None:
            return
        w.last_heartbeat_at = time.time()
        if w.restart_failures:
            w.restart_failures = 0
            w.last_restart_error = ""

    async def start(self, *, interval: float = 5.0) -> None:
        if self._running and self._task and not self._task.done():
            return
        self._consume_loop_failure()
        self._running = True

        async def _loop():
            while self._running:
                try:
                    await self._tick()
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    if not self._running or is_shutdown_requested():
                        break
                    logger.warning("SelfHealing loop spuriously cancelled. Ignoring.")
                    continue
                except Exception as e:  # noqa: BLE001 - watchdog loop boundary
                    record_degradation('self_healing', e)
                    logger.error("SelfHealing loop error: %s", e)
                    await asyncio.sleep(1.0)

        self._task = create_tracked_task(_loop(), name="SelfHealing")

    async def stop(self) -> None:
        async with self._stop_lock:
            self._running = False
            owned_tasks = [
                task
                for task in (self._task, *self._deep_repairs.values())
                if task is not None
                and task is not asyncio.current_task()
                and not task.done()
            ]
            self._task = None
            for task in owned_tasks:
                task.cancel()
            if owned_tasks:
                await asyncio.gather(*owned_tasks, return_exceptions=True)
            self._deep_repairs.clear()

    def get_status(self) -> dict[str, Any]:
        now = time.time()
        return {
            "running": bool(self._running and self._task and not self._task.done()),
            "watch_count": len(self._watches),
            "restart_timeout_s": self._restart_timeout_s,
            "deep_repairs_active": sum(
                1 for task in self._deep_repairs.values() if not task.done()
            ),
            "watches": {
                name: {
                    "heartbeat_age_s": round(max(0.0, now - watch.last_heartbeat_at), 2),
                    "expected_interval_s": watch.expected_interval_s,
                    "restart_count": watch.restarts,
                    "restart_failure_count": watch.restart_failures,
                    "last_restart_at": watch.last_restart_at,
                    "last_restart_error": watch.last_restart_error,
                    "container_key": watch.container_key,
                }
                for name, watch in sorted(self._watches.items())
            },
        }

    def _consume_loop_failure(self) -> None:
        task = self._task
        if task is None or not task.done():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        except _SELF_HEALING_RECOVERABLE_ERRORS as exc:
            error = exc
        if error is not None:
            record_degradation(
                "self_healing",
                error,
                action="recovered the self-healing loop after its prior owned task failed",
                receipt_required=True,
            )

    async def _tick(self) -> None:
        now = time.time()
        for w in list(self._watches.values()):
            age = now - w.last_heartbeat_at
            if age <= w.expected_interval_s * 2.5:
                continue
            defer_reason = self._healing_defer_reason()
            if defer_reason:
                w.last_heartbeat_at = now
                await self._append_record_async(
                    {
                        "when": now,
                        "name": w.name,
                        "stale_for_s": age,
                        "result": self._deferred_result(defer_reason),
                    }
                )
                continue
            await self._heal(w, age)

    @staticmethod
    def _deferred_result(reason: str) -> str:
        if reason == "foreground_busy":
            return "deferred_foreground_busy"
        return f"deferred_{reason}"

    def _healing_defer_reason(self) -> str:
        if is_shutdown_requested():
            return "shutdown_requested"

        try:
            from core.runtime.proof_policy import proof_headless_run

            if proof_headless_run():
                return "proof_run_active"
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation(
                "self_healing",
                exc,
                action="deferred healing because proof policy was unavailable",
                receipt_required=True,
            )
            return "proof_policy_unavailable"

        if self._foreground_runtime_busy():
            return "foreground_busy"
        return ""

    def _foreground_runtime_busy(self) -> bool:
        try:
            from core.runtime.foreground_guard import foreground_activity_reason

            if foreground_activity_reason():
                return True
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("self_healing", exc)
            return True

        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            status_getter = getattr(gate, "get_conversation_status", None)
            if callable(status_getter):
                status = status_getter() or {}
                if isinstance(status, dict):
                    return bool(
                        status.get("active")
                        or status.get("foreground_owned")
                        or int(status.get("active_generations", 0) or 0) > 0
                        or status.get("kernel_lock_held")
                        or str(status.get("state", "")).lower()
                        in {"spawning", "handshaking", "warming", "recovering"}
                    )
                if bool(getattr(status, "active", False)):
                    return True
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("self_healing", exc)
            return True

        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None) or ServiceContainer.get("aura_runtime", default=None)
            if orch is None:
                return False
            status = getattr(orch, "status", None)
            if not bool(getattr(status, "is_processing", False)):
                return False
            return not bool(getattr(orch, "_current_task_is_autonomous", False))
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("self_healing", exc)
            return False

    async def _heal(self, w: WatchEntry, age: float) -> None:
        record = {
            "when": time.time(),
            "name": w.name,
            "stale_for_s": age,
            "restart_count": w.restarts,
            "restart_failure_count": w.restart_failures,
        }
        try:
            defer_reason = self._healing_defer_reason()
            if defer_reason:
                record["result"] = self._deferred_result(defer_reason)
                w.last_heartbeat_at = time.time()
                await self._append_record_async(record)
                return

            if w.restarts >= 3 or w.restart_failures >= 3:
                module_path = await asyncio.to_thread(self._module_path_for_watch, w)
                block_reason = _deep_repair_block_reason("self_healing_watchdog_deep_repair")
                if module_path and not block_reason:
                    logger.warning("Deep repair triggered for %s (%s)", w.name, module_path)
                    scheduled = self.schedule_deep_repair(
                        module_path,
                        reason="watchdog_restart_exhausted",
                        watch_name=w.name,
                        metadata={
                            "stale_for_s": age,
                            "restart_count": w.restarts,
                            "restart_failure_count": w.restart_failures,
                            "last_restart_error": w.last_restart_error,
                        },
                    )
                    record.update(scheduled)
                    w.restarts = 0
                    w.restart_failures = 0
                    w.last_heartbeat_at = time.time()
                else:
                    record["result"] = (
                        block_reason
                        if module_path
                        else "deep_repair_failed_no_module_path"
                    )
                    w.restarts = 0
                    w.restart_failures = 0
                    w.last_heartbeat_at = time.time()
                    await self._append_record_async(record)
                    return

            if record.get("result") not in (
                "deep_repair_scheduled",
                "deep_repair_already_running",
                "deep_repair_failed_no_module_path",
                "deep_repair_failed_no_lab",
            ):
                try:
                    restarted = await asyncio.wait_for(
                        self._restart_watch(w),
                        timeout=self._restart_timeout_s,
                    )
                except asyncio.CancelledError:
                    raise
                except TimeoutError as exc:
                    self._record_restart_failure(w, exc)
                    record["result"] = (
                        f"restart_timeout_after_{self._restart_timeout_s:g}s"
                    )
                    record["restart_failure_count"] = w.restart_failures
                    record_degradation(
                        "self_healing",
                        exc,
                        action=(
                            f"contained timed-out restart for {w.name} and kept the "
                            "self-healing loop alive"
                        ),
                        receipt_required=True,
                    )
                    restarted = False
                except Exception as exc:  # noqa: BLE001 - watched-service boundary
                    self._record_restart_failure(w, exc)
                    record["result"] = f"restart_failed:{type(exc).__name__}:{exc}"
                    record["restart_failure_count"] = w.restart_failures
                    record_degradation(
                        "self_healing",
                        exc,
                        action=(
                            f"contained failed restart for {w.name} and kept the "
                            "self-healing loop alive"
                        ),
                        receipt_required=True,
                    )
                    restarted = False

                if restarted:
                    w.restarts += 1
                    w.restart_failures = 0
                    w.last_restart_error = ""
                    w.last_restart_at = time.time()
                    record["result"] = "restarted"
                elif "result" not in record:
                    module_path = await asyncio.to_thread(self._module_path_for_watch, w)
                    block_reason = _deep_repair_block_reason(
                        "self_healing_restart_unavailable"
                    )
                    if module_path and not block_reason:
                        record.update(
                            self.schedule_deep_repair(
                                module_path,
                                reason="restart_interface_unavailable",
                                watch_name=w.name,
                                metadata={"stale_for_s": age},
                            )
                        )
                    else:
                        record["result"] = block_reason or "restart_interface_unavailable"
                        w.restart_failures += 1
                        w.last_restart_error = str(record["result"])
                        record["restart_failure_count"] = w.restart_failures
                w.last_heartbeat_at = time.time()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - watchdog repair boundary
            record_degradation('self_healing', exc)
            self._record_restart_failure(w, exc)
            record["result"] = f"restart_failed:{type(exc).__name__}:{exc}"
            record["restart_failure_count"] = w.restart_failures
            w.last_heartbeat_at = time.time()
        await self._append_record_async(record)

    async def _restart_watch(self, w: WatchEntry) -> bool:
        if w.restart_async is not None:
            restarted = w.restart_async()
            if inspect.isawaitable(restarted):
                await restarted
            return True
        if not w.container_key:
            return False

        from core.container import ServiceContainer

        instance = ServiceContainer.get(w.container_key, default=None)
        if instance is None:
            return False
        restart = getattr(instance, "restart_async", None)
        if callable(restart):
            restarted = restart()
            if inspect.isawaitable(restarted):
                await restarted
            return True

        stop = getattr(instance, "stop", None)
        start = getattr(instance, "start", None)
        if not callable(stop) or not callable(start):
            return False
        stopped = stop()
        if inspect.isawaitable(stopped):
            await stopped
        started = start()
        if inspect.isawaitable(started):
            await started
        return True

    @staticmethod
    def _record_restart_failure(w: WatchEntry, error: BaseException) -> None:
        w.restart_failures += 1
        w.last_restart_at = time.time()
        detail = str(error).strip() or "no error message"
        w.last_restart_error = f"{type(error).__name__}: {detail}"

    def _module_path_for_watch(self, w: WatchEntry) -> str | None:
        if not w.container_key:
            return None
        cached = self._module_path_cache.get(w.container_key)
        if w.container_key in self._module_path_cache:
            return cached
            
        fallbacks = {
            "orchestrator": "core/orchestrator/main.py",
            "mind_tick": "core/mind_tick.py",
            "scheduler": "core/scheduler.py",
            "morphogenetic_runtime": "core/morphogenesis/runtime.py",
            "motor_cortex": "core/somatic/motor_cortex.py"
        }
        fallback = fallbacks.get(w.container_key)
        
        try:
            from core.config import config
            from core.container import ServiceContainer

            instance = ServiceContainer.get(w.container_key, default=None)
            if instance is None:
                self._module_path_cache[w.container_key] = fallback
                return fallback
                
            # Unpack proxies if present
            if hasattr(instance, "__wrapped__"):
                instance = instance.__wrapped__
            elif hasattr(instance, "_instance"):
                instance = instance._instance or instance

            try:
                source_file = inspect.getsourcefile(type(instance)) or inspect.getfile(type(instance))
            except (TypeError, OSError, AttributeError, RuntimeError, ValueError) as _exc:
                logger.debug(
                    "Watched service %s has no resolvable source file: %s",
                    w.name,
                    _exc,
                )
                source_file = None
            if source_file:
                source_path = Path(source_file)
                try:
                    base_dir = Path(config.paths.base_dir)
                    resolved = str(source_path.relative_to(base_dir))
                    self._module_path_cache[w.container_key] = resolved
                    return resolved
                except ValueError as _exc:
                    logger.debug("Suppressed %s in core.runtime.self_healing: %s", type(_exc).__name__, _exc)

            module_name = getattr(type(instance), "__module__", "")
            if module_name and module_name != "builtins":
                candidate = module_name.replace(".", "/") + ".py"
                if (config.paths.base_dir / candidate).exists():
                    self._module_path_cache[w.container_key] = candidate
                    return candidate
                
            self._module_path_cache[w.container_key] = fallback
            return fallback
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('self_healing', exc)
            logger.debug("Could not resolve watched module path for %s: %s", w.name, exc)
            self._module_path_cache[w.container_key] = fallback
            return fallback

    def schedule_deep_repair(
        self,
        module_path: str,
        *,
        reason: str,
        watch_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Schedule a ReimplementationLab repair without blocking the watchdog."""

        key = str(module_path)
        block_reason = _deep_repair_block_reason("self_healing_schedule_deep_repair")
        if block_reason:
            return {
                "result": block_reason,
                "module_path": key,
                "reason": reason,
            }
        existing = self._deep_repairs.get(key)
        if existing is not None and not existing.done():
            return {
                "result": "deep_repair_already_running",
                "module_path": key,
                "reason": reason,
            }

        async def _runner() -> None:
            await self.request_deep_repair(
                key,
                reason=reason,
                watch_name=watch_name,
                metadata=metadata,
                max_attempts=max_attempts,
            )

        try:
            task = create_tracked_task(_runner(), name=f"SelfHealing.deep_repair.{key}")
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "self_healing",
                exc,
                action="failed closed when deep repair task could not be scheduled through task ownership",
                receipt_required=True,
            )
            return {
                "result": "deep_repair_schedule_failed",
                "module_path": key,
                "reason": reason,
            }
        self._deep_repairs[key] = task
        task.add_done_callback(lambda _task: self._deep_repairs.pop(key, None))
        return {
            "result": "deep_repair_scheduled",
            "module_path": key,
            "reason": reason,
        }

    async def request_deep_repair(
        self,
        module_path: str,
        *,
        reason: str,
        watch_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Run ReimplementationLab as a deep repair strategy.

        This is intentionally separate from restart healing so patch-based
        repair systems can call the same hardened path when local diffs fail.
        """

        record: dict[str, Any] = {
            "when": time.time(),
            "name": watch_name or "deep_repair",
            "module_path": str(module_path),
            "reason": reason,
            "metadata": metadata or {},
        }
        block_reason = _deep_repair_block_reason("self_healing_request_deep_repair")
        if block_reason:
            record["result"] = block_reason
            await self._append_record_async(record)
            return record
        # Is this anything we know?
        #
        # The ladder reached governed reconstruction without ever asking. A
        # failure the system has a concept for should be repaired by the
        # concept's runbook; one it does not is the hard case, and the hard
        # case is worth naming so the next occurrence is cheap. Off the loop,
        # because inferring the broken invariant runs the verifier.
        diagnosis = await asyncio.to_thread(
            _diagnose_this_failure, watch_name or str(module_path), reason
        )
        if diagnosis is not None:
            record["diagnosis"] = diagnosis.to_dict()
        try:
            from core.service_names import ServiceNames

            lab = (
                get_runtime_service(ServiceNames.REIMPLEMENTATION_LAB, default=None)
                or get_runtime_service(ServiceNames.PROGRAM_DNA_RECONSTRUCTION, default=None)
            )
            if lab is None:
                from core.self_improvement.reimplementation_lab import (
                    register_reimplementation_lab,
                )

                lab = register_reimplementation_lab()
            if lab is None:
                record["result"] = "deep_repair_failed_no_program_dna_lab"
                return record

            lab_metadata = {
                "trigger": "self_healing",
                "reason": reason,
                "watch_name": watch_name,
                **(metadata or {}),
            }
            result = await lab.run_reconstruction(
                str(module_path),
                max_attempts=max_attempts,
                metadata=lab_metadata,
            )
            result_dict = result.to_dict() if hasattr(result, "to_dict") else {"success": False}
            record["result"] = "deep_repair_succeeded" if result_dict.get("success") else "deep_repair_rejected"
            record["lab_result"] = result_dict
            # Step six, and only on a repair that held. A concept minted for
            # a failure that is still happening teaches the recogniser to
            # expect the broken state, and every later occurrence comes back
            # KNOWN with nothing known about it.
            if diagnosis is not None and result_dict.get("success"):
                learned = _learn_this_failure_class(diagnosis, module_path)
                if learned:
                    record["failure_concept"] = learned
            return record
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('self_healing', exc)
            record["result"] = f"deep_repair_failed:{exc}"
            return record
        finally:
            await self._append_record_async(record)

    async def _append_record_async(self, record: dict[str, Any]) -> None:
        """Persist a healing receipt without blocking the main asyncio loop."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._append_record, record),
                timeout=self._ledger_write_timeout_s,
            )
        except TimeoutError:
            logger.warning("SelfHealing ledger write timed out; preserving live loop responsiveness.")
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("SelfHealing ledger write failed asynchronously: %s", exc)

    def _append_record(self, record: dict[str, Any]) -> None:
        try:
            with local_internal_governed_scope(
                "runtime.self_healing.ledger",
                domain="file_write",
            ):
                get_file_write_gateway().append_text(
                    _LEDGER,
                    json.dumps(record, default=str) + "\n",
                    source="runtime.self_healing.ledger",
                )
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("SelfHealing ledger append failed: %s", exc)


_HEALER: SelfHealing | None = None


def get_healer() -> SelfHealing:
    global _HEALER
    if _HEALER is None:
        _HEALER = SelfHealing()
    return _HEALER


__all__ = ["SelfHealing", "WatchEntry", "get_healer"]

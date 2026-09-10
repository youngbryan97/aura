import asyncio
import logging
from functools import partial
from typing import Any

from core.config import Environment, config
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.service_registry import register_runtime_service

logger = logging.getLogger(__name__)

HARDENING_BOOT_TIMEOUT_SECONDS = 10.0


def _noop_service_callback() -> None:
    return None


def _record_hardening_degradation(
    error: BaseException,
    *,
    component: str,
    action: str,
    severity: Severity = "degraded",
) -> None:
    record_degradation(
        "hardening",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SILENT_LOSS_OF_CAPABILITY,
        extra={
            "component": component,
            "repair_requested": True,
        },
    )


def _component_is_alive(component: Any) -> bool:
    liveness = getattr(component, "is_alive", None)
    if callable(liveness):
        return bool(liveness())
    task = getattr(component, "_task", None)
    running = bool(getattr(component, "_running", True))
    if task is not None:
        return running and not task.done()
    return running


def _component_is_running(component: Any) -> bool:
    # A monitor's health can be false precisely because its sampling task is
    # working. Restart only the task, not the evidence it is collecting.
    running = getattr(component, "is_running", None)
    if callable(running):
        return bool(running())
    return _component_is_alive(component)


async def _stop_failed_component(component: Any) -> None:
    stop = getattr(component, "stop", None)
    if not callable(stop):
        return
    result = stop()
    if asyncio.iscoroutine(result):
        await asyncio.wait_for(result, timeout=2.0)


def _fail_required_component(name: str, error: BaseException) -> None:
    if config.env == Environment.PROD:
        raise RuntimeError(f"Production hardening component failed: {name}") from error


async def _start_supervisor(
    *,
    name: str,
    container_key: str,
    component: Any,
    boot_timeout_s: float,
    status: dict[str, dict[str, Any]],
) -> bool:
    try:
        await asyncio.wait_for(component.start(), timeout=boot_timeout_s)
    except asyncio.CancelledError:
        raise
    except TimeoutError as exc:
        try:
            await _stop_failed_component(component)
        except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as stop_exc:
            _record_hardening_degradation(
                stop_exc,
                component=name,
                action=f"{name} stop after boot timeout failed; component left unregistered",
                severity="warning",
            )
        status[name] = {
            "state": "failed",
            "registered": False,
            "error": f"startup timed out after {boot_timeout_s:.1f}s",
        }
        _record_hardening_degradation(
            exc,
            component=name,
            action=f"{name} boot timed out; component stopped if possible and left unregistered",
            severity="critical" if config.env == Environment.PROD else "degraded",
        )
        logger.error("%s boot timed out after %.1fs.", name, boot_timeout_s)
        _fail_required_component(name, exc)
        return False
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        status[name] = {
            "state": "failed",
            "registered": False,
            "error": str(exc),
        }
        _record_hardening_degradation(
            exc,
            component=name,
            action=f"{name} boot failed; component left unregistered for retry by runtime recovery",
            severity="critical" if config.env == Environment.PROD else "degraded",
        )
        logger.error("%s boot failed: %s", name, exc)
        _fail_required_component(name, exc)
        return False

    if not _component_is_alive(component):
        exc = RuntimeError(f"{name} start returned without a live background task")
        status[name] = {
            "state": "failed",
            "registered": False,
            "error": str(exc),
        }
        _record_hardening_degradation(
            exc,
            component=name,
            action=f"{name} failed liveness verification and was not registered",
            severity="critical" if config.env == Environment.PROD else "degraded",
        )
        logger.error("%s failed liveness verification.", name)
        _fail_required_component(name, exc)
        return False

    register_runtime_service(container_key, component, failure_policy="degrade_with_receipt", owner="core/orchestrator/initializers/hardening.py", registered_by="_register_component")
    status[name] = {
        "state": "online",
        "registered": True,
        "container_key": container_key,
    }
    return True


async def init_hardening_layer(orchestrator: Any):
    """Initialize growth scanning, reaper, and hypervisor layer."""

    hardening_status: dict[str, dict[str, Any]] = {}
    managed_components: dict[str, Any] = {}
    orchestrator.hardening_status = hardening_status

    # 7.5 Platform Root (Hardware Binding)
    # Do NOT start PlatformRoot before multiprocessing spawn, otherwise Metal GPU
    # bindings can corrupt child processes. This is an intentional deferral, not
    # a degraded boot path.
    hardening_status["platform_root"] = {
        "state": "deferred",
        "registered": False,
        "reason": "deferred until after multiprocessing spawn to protect Metal bindings",
    }
    logger.info("Platform Root deferred until post-spawn runtime.")

    # 8. Startup Validation (Pre-flight)
    from core.startup.validator import get_validator

    validator = get_validator()
    v_passed = await validator.run_all()
    if not v_passed:
        logger.critical("Startup Validation Failed")
        hardening_status["startup_validator"] = {
            "state": "failed",
            "registered": False,
            "error": "startup validator returned False",
        }
        if config.env == Environment.PROD:
            raise RuntimeError("Production pre-flight failure: Critical startup checks failed.")
    else:
        hardening_status["startup_validator"] = {
            "state": "online",
            "registered": False,
        }

    # 9. Enterprise Hardening: Reaper & Hypervisor
    from core.ops.hypervisor import get_hypervisor
    from core.ops.lymphatic_reaper import get_reaper

    orchestrator.reaper = get_reaper()
    orchestrator.hypervisor = get_hypervisor()

    if await _start_supervisor(
        name="reaper",
        container_key="reaper",
        component=orchestrator.reaper,
        boot_timeout_s=HARDENING_BOOT_TIMEOUT_SECONDS,
        status=hardening_status,
    ):
        managed_components["reaper"] = orchestrator.reaper
    if await _start_supervisor(
        name="hypervisor",
        container_key="hypervisor",
        component=orchestrator.hypervisor,
        boot_timeout_s=HARDENING_BOOT_TIMEOUT_SECONDS,
        status=hardening_status,
    ):
        managed_components["hypervisor"] = orchestrator.hypervisor

    # EventLoopMonitor: Watchdog for stall detection (>0.1s)
    try:
        from core.utils.concurrency import EventLoopMonitor

        monitor = EventLoopMonitor()
        monitor.start()
        if not _component_is_alive(monitor):
            raise RuntimeError("EventLoopMonitor start returned without a live task")
        register_runtime_service("event_loop_monitor", monitor, failure_policy="degrade_with_receipt", owner="core/orchestrator/initializers/hardening.py", registered_by="initialize_event_loop_monitor")
        hardening_status["event_loop_monitor"] = {
            "state": "online",
            "registered": True,
            "container_key": "event_loop_monitor",
            "threshold_s": monitor.threshold,
            "active_threshold_s": monitor.active_threshold,
        }
        managed_components["event_loop_monitor"] = monitor
        logger.info(
            "EventLoopMonitor active (threshold=%.2fs active_threshold=%.2fs)",
            monitor.threshold,
            monitor.active_threshold,
        )
    except asyncio.CancelledError:
        raise
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        hardening_status["event_loop_monitor"] = {
            "state": "failed",
            "registered": False,
            "error": str(exc),
        }
        _record_hardening_degradation(
            exc,
            component="event_loop_monitor",
            action="event-loop monitor boot failed; left unregistered so runtime can retry",
            severity="critical" if config.env == Environment.PROD else "degraded",
        )
        logger.error("EventLoopMonitor failed to start: %s", exc)
        _fail_required_component("event_loop_monitor", exc)

    # Unified runtime pressure: pull-based provider for the health-contract
    # entry. Before this registration existed the contract required a service
    # nobody provided, so the runtime could pin DEGRADED against a phantom
    # (observed live 2026-07-05). Pull-based: no loop, nothing to die.
    try:
        from core.runtime.runtime_pressure import get_unified_runtime_pressure

        pressure = get_unified_runtime_pressure()
        pressure.runtime_pressure_snapshot()  # prove it can sample at boot
        register_runtime_service(
            "unified_runtime_pressure",
            pressure,
            failure_policy="degrade_with_receipt",
            owner="core/orchestrator/initializers/hardening.py",
            registered_by="initialize_unified_runtime_pressure",
        )
        hardening_status["unified_runtime_pressure"] = {
            "state": "online",
            "registered": True,
            "container_key": "unified_runtime_pressure",
        }
        managed_components["unified_runtime_pressure"] = pressure
        logger.info("UnifiedRuntimePressure active (pull-based, no loop).")
    except asyncio.CancelledError:
        raise
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        hardening_status["unified_runtime_pressure"] = {
            "state": "failed",
            "registered": False,
            "error": str(exc),
        }
        _record_hardening_degradation(
            exc,
            component="unified_runtime_pressure",
            action="runtime pressure provider boot failed; contract entry stays unhealthy",
            severity="degraded",
        )
        logger.error("UnifiedRuntimePressure failed to start: %s", exc)

    # Allostasis engine: predictive interoception over the pressure snapshots
    # the provider above produces. Loop-free by design — the metabolic
    # coordinator supplies the 60 s pulse, so there is no task here to die.
    # Registration is what makes the anticipatory signal reachable from the
    # body-state hot path (container lookup only, never construction there).
    try:
        from core.autonomic.allostasis import get_allostasis_engine

        allostasis = get_allostasis_engine()
        register_runtime_service(
            "allostasis_engine",
            allostasis,
            failure_policy="degrade_with_receipt",
            owner="core/autonomic/allostasis.py",
            registered_by="initialize_allostasis_engine",
        )
        hardening_status["allostasis_engine"] = {
            "state": "online" if allostasis.enabled else "disabled",
            "registered": True,
            "container_key": "allostasis_engine",
        }
        managed_components["allostasis_engine"] = allostasis
        logger.info(
            "AllostasisEngine active (predictive interoception, pulse via metabolic coordinator)."
        )
    except asyncio.CancelledError:
        raise
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        hardening_status["allostasis_engine"] = {
            "state": "failed",
            "registered": False,
            "error": str(exc),
        }
        _record_hardening_degradation(
            exc,
            component="allostasis_engine",
            action="allostasis boot failed; body runs reactive-only (no anticipation)",
            severity="degraded",
        )
        logger.error("AllostasisEngine failed to start: %s", exc)

    # Adopt hardening components into the canonical desired-state reconciler.
    # They are already running at this point; future probes/restarts flow
    # through one owner instead of independent ad hoc watchdog policy.
    try:
        from core.runtime.control_plane import (
            DesiredServiceSpec,
            WorkClass,
            get_runtime_control_plane,
        )

        control_plane = get_runtime_control_plane()
        for component_name, component in managed_components.items():
            if control_plane.has_service(component_name):
                continue
            if component_name == "unified_runtime_pressure":
                start = _noop_service_callback
                stop = _noop_service_callback
                probe = component.runtime_pressure_snapshot
                critical = True
                restart_on_unhealthy = False
            elif component_name == "allostasis_engine":
                # Pull-based like unified_runtime_pressure: no loop to
                # restart; the probe is readiness, not task liveness.
                start = _noop_service_callback
                stop = _noop_service_callback
                probe = component.is_ready
                critical = False
                restart_on_unhealthy = False
            else:
                start = getattr(component, "start", _noop_service_callback)
                stop = getattr(component, "stop", _noop_service_callback)
                probe = partial(_component_is_running, component)
                critical = component_name == "event_loop_monitor"
                restart_on_unhealthy = True
            control_plane.register_service(
                DesiredServiceSpec(
                    name=component_name,
                    critical=critical,
                    restart_on_unhealthy=restart_on_unhealthy,
                    admission_class=WorkClass.SERVICE_START,
                    metadata={"boot_layer": "hardening"},
                ),
                start=start,
                stop=stop,
                probe=probe,
                adopt_running=True,
            )

        control_report = await control_plane.reconcile_once()
        register_runtime_service(
            "runtime_control_plane",
            control_plane,
            failure_policy="fail-closed",
            owner="core/runtime/control_plane.py",
            registered_by="init_hardening_layer",
            required_for="desired-state reconciliation and resource admission",
        )
        register_runtime_service(
            "resource_admission",
            control_plane.admission,
            failure_policy="fail-closed",
            owner="core/runtime/control_plane.py",
            registered_by="init_hardening_layer",
            required_for="pressure-aware constrained work leases",
        )
        hardening_status["runtime_control_plane"] = {
            "state": "online" if control_report.get("critical_ready") else "degraded",
            "registered": True,
            "container_key": "runtime_control_plane",
            "converged": bool(control_report.get("converged")),
            "critical_ready": bool(control_report.get("critical_ready")),
            "managed_services": sorted(managed_components),
        }
    except asyncio.CancelledError:
        raise
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        hardening_status["runtime_control_plane"] = {
            "state": "failed",
            "registered": False,
            "error": str(exc),
        }
        _record_hardening_degradation(
            exc,
            component="runtime_control_plane",
            action="desired-state control plane boot failed; resource admission remains unhealthy",
            severity="critical" if config.env == Environment.PROD else "degraded",
        )
        _fail_required_component("runtime_control_plane", exc)

    logger.info("Hardening Layer status: %s", hardening_status)

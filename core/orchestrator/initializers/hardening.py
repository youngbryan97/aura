import asyncio
import logging
from typing import Any

from core.config import Environment, config
from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, Severity, record_degradation

logger = logging.getLogger(__name__)

HARDENING_BOOT_TIMEOUT_SECONDS = 10.0


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

    ServiceContainer.register_instance(container_key, component)
    status[name] = {
        "state": "online",
        "registered": True,
        "container_key": container_key,
    }
    return True


async def init_hardening_layer(orchestrator: Any):
    """Initialize growth scanning, reaper, and hypervisor layer."""

    hardening_status: dict[str, dict[str, Any]] = {}
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

    await _start_supervisor(
        name="reaper",
        container_key="reaper",
        component=orchestrator.reaper,
        boot_timeout_s=HARDENING_BOOT_TIMEOUT_SECONDS,
        status=hardening_status,
    )
    await _start_supervisor(
        name="hypervisor",
        container_key="hypervisor",
        component=orchestrator.hypervisor,
        boot_timeout_s=HARDENING_BOOT_TIMEOUT_SECONDS,
        status=hardening_status,
    )

    # EventLoopMonitor: Watchdog for stall detection (>0.1s)
    try:
        from core.utils.concurrency import EventLoopMonitor

        monitor = EventLoopMonitor()
        monitor.start()
        if not _component_is_alive(monitor):
            raise RuntimeError("EventLoopMonitor start returned without a live task")
        ServiceContainer.register_instance("event_loop_monitor", monitor)
        hardening_status["event_loop_monitor"] = {
            "state": "online",
            "registered": True,
            "container_key": "event_loop_monitor",
            "threshold_s": monitor.threshold,
        }
        logger.info("EventLoopMonitor active (threshold=%.2fs)", monitor.threshold)
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

    logger.info("Hardening Layer status: %s", hardening_status)

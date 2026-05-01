from core.runtime.errors import record_degradation
import asyncio
import logging
from typing import Any
from core.container import ServiceContainer
from core.config import config, Environment

logger = logging.getLogger(__name__)

async def init_hardening_layer(orchestrator: Any):
    """Initialize growth scanning, reaper, and hypervisor layer."""
    
    # 7.5 Platform Root (Hardware Binding)
    try:
        # Do NOT start PlatformRoot before multiprocessing spawn,
        # otherwise Metal GPU bindings will corrupt the child process!
        # platform_root = ServiceContainer.get("platform_root", default=None)
        # if platform_root:
        #     get_task_tracker().create_task(platform_root.start_monitor())
        #     logger.info("🌿 [BOOT] Platform Root persistence monitor started.")
        logger.info("🌿 [BOOT] Platform Root DEFERRED to prevent spawn corruption.")
    except Exception as e:
        record_degradation('hardening', e)
        logger.warning("🌿 [BOOT] Platform Root monitor failed to start: %s", e)
    
    # 8. Startup Validation (Pre-flight)
    from core.startup.validator import get_validator
    validator = get_validator()
    v_passed = await validator.run_all()
    if not v_passed:
        logger.critical("🚨 Startup Validation Failed")
        if config.env == Environment.PROD:
            raise RuntimeError("Production pre-flight failure: Critical startup checks failed.")
    
    # 9. Enterprise Hardening: Reaper & Hypervisor
    from core.ops.lymphatic_reaper import get_reaper
    from core.ops.hypervisor import get_hypervisor
    orchestrator.reaper = get_reaper()
    orchestrator.hypervisor = get_hypervisor()
    
    # [WATCHDOG] 10s Boot Timers
    try:
        await asyncio.wait_for(orchestrator.reaper.start(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.error("🛑 Reaper boot timed out. Proceeding in degraded state.")
    except Exception as e:
        record_degradation('hardening', e)
        logger.error("❌ Reaper boot failed: %s", e)

    try:
        await asyncio.wait_for(orchestrator.hypervisor.start(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.error("🛑 Hypervisor boot timed out. Proceeding in degraded state.")
    except Exception as e:
        record_degradation('hardening', e)
        logger.error("❌ Hypervisor boot failed: %s", e)
        
    ServiceContainer.register_instance("reaper", orchestrator.reaper)
    ServiceContainer.register_instance("hypervisor", orchestrator.hypervisor)
    
    # EventLoopMonitor: Watchdog for stall detection (>0.1s)
    try:
        from core.utils.concurrency import EventLoopMonitor
        monitor = EventLoopMonitor(threshold=0.25)
        from core.utils.task_tracker import get_task_tracker
        get_task_tracker().track(monitor.start(), name="event_loop_monitor")
        ServiceContainer.register_instance("event_loop_monitor", monitor)
        logger.info("🛡️ [BOOT] EventLoopMonitor active (Threshold: 0.1s)")
    except Exception as e:
        record_degradation('hardening', e)
        logger.error("❌ [BOOT] EventLoopMonitor failed to start: %s", e)

    logger.info("🛡️ [BOOT] Hardening Layer (Reaper/Hypervisor/Watchdog) online.")

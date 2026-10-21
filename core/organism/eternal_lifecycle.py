import asyncio
import logging
import time

import mlx.core as mx

from core.container import ServiceContainer
from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Eternal")

async def eternal_lifecycle():
    """
    Indefinite health monitoring loop (1Hz).
    Manages thermal emergencies, memory pressure, and scheduled maintenance.
    """
    from core.runtime import CoreRuntime
    
    logger.info("♾️ Eternal Lifecycle engaged.")
    
    from core.runtime.shutdown_coordinator import is_shutdown_requested

    last_emergency = 0.0
    while not is_shutdown_requested():
        try:
            CoreRuntime.get_sync()
            mem = psutil.virtual_memory().percent

            # Thermal/Memory Emergency, at most once a minute: this loop turns every second, and each turn above 88%
            # tried to save a snapshot. It called StateManager.snapshot(), which does not exist (LIVE 21 October 2026:
            # an AttributeError a second, escalated to an emergency incident); the orchestrator's own save is used.
            if mem > 88 and time.monotonic() - last_emergency >= 60.0:
                last_emergency = time.monotonic()
                logger.warning(
                    "⚠️ High memory pressure (%.1f%%). Triggering emergency eviction.",
                    mem,
                )
                save = getattr(ServiceContainer.get("orchestrator", default=None), "_save_state_async", None)
                if callable(save):
                    await save("thermal_emergency")
                try:
                    from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel
                    sentinel = get_gpu_sentinel()
                    if sentinel.acquire(priority=GPUPriority.REFLEX, timeout=5.0):
                        try:
                            if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                                mx.metal.clear_cache()
                            else:
                                mx.clear_cache()
                        finally:
                            sentinel.release()
                except (ImportError, AttributeError, RuntimeError) as e:
                    record_degradation('eternal_lifecycle', e)
                    logger.debug("[MLX] Cache clear skipped in eternal: %s", e)
                
                # Evict episodic memory if pressure persists
                if mem > 92:
                    mem_system = ServiceContainer.get("dual_memory", default=None)
                    if mem_system:
                        await mem_system.episodic.evict_oldest(0.3)
            
            # Scheduled Maintenance (3 AM)
            lt = time.localtime()
            if lt.tm_hour == 3 and lt.tm_min == 0 and lt.tm_sec == 0:
                logger.info("🌙 Starting nightly maintenance sequence...")
                try:
                    from core.adaptation.nightly_lora import nightly_lora_finetune
                    await nightly_lora_finetune()
                except ImportError:
                    logger.debug("Nightly LoRA module not found, skipping.")
                # ImportError above shadowed this clause's copy of it, so a
                # nightly-maintenance failure that raised one was silently
                # logged as "module not found" and never recorded.
                except (AttributeError, RuntimeError) as e:
                    record_degradation('eternal_lifecycle', e)
                    logger.error("Nightly maintenance failed: %s", e)

        except RuntimeError as _e:
            # Runtime not yet initialized
            logger.debug('Ignored RuntimeError in eternal_lifecycle.py: %s', _e)
        # RuntimeError is caught above and never reaches here, so every real
        # eternal-loop RuntimeError was debug-logged as "not yet initialized"
        # and never recorded as a degradation.
        except (ImportError, AttributeError) as e:
            record_degradation('eternal_lifecycle', e)
            logger.error("Error in eternal_loop: %s", e)
            
        await asyncio.sleep(1)

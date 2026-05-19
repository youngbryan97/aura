from core.runtime.errors import record_degradation
import asyncio
import time
import psutil
import mlx.core as mx
import logging
from core.container import ServiceContainer

logger = logging.getLogger("Aura.Eternal")

async def eternal_lifecycle():
    """
    Indefinite health monitoring loop (1Hz).
    Manages thermal emergencies, memory pressure, and scheduled maintenance.
    """
    from core.runtime import CoreRuntime
    
    logger.info("♾️ Eternal Lifecycle engaged.")
    
    from core.runtime.shutdown_coordinator import is_shutdown_requested

    while not is_shutdown_requested():
        try:
            rt = CoreRuntime.get_sync()
            mem = psutil.virtual_memory().percent
            
            # Thermal/Memory Emergency
            if mem > 88:
                logger.warning("⚠️ High memory pressure (%s%). Triggering emergency eviction.", mem)
                await rt.state_manager.snapshot("thermal_emergency")
                try:
                    from core.utils.gpu_sentinel import get_gpu_sentinel, GPUPriority
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
                    from core.dual_memory import DualMemorySystem
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
                except (ImportError, AttributeError, RuntimeError) as e:
                    record_degradation('eternal_lifecycle', e)
                    logger.error("Nightly maintenance failed: %s", e)

        except RuntimeError as _e:
            # Runtime not yet initialized
            logger.debug('Ignored RuntimeError in eternal_lifecycle.py: %s', _e)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('eternal_lifecycle', e)
            logger.error("Error in eternal_loop: %s", e)
            
        await asyncio.sleep(1)

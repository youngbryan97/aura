from core.runtime.errors import record_degradation
import mlx.core as mx
from dataclasses import dataclass, field
import psutil
import logging
import asyncio

logger = logging.getLogger("Aura.Memory")

@dataclass
class MemoryBudget:
    max_kv_cache_mb: int = 16384          # KV-cache cap for the 32B cortex lane
    max_vector_memory_mb: int = 4096
    max_sqlite_mb: int = 2048
    soft_limit_mb: int = 56000           # 56 GB soft limit for 64GB M5 Pro

    def enforce(self):
        """
        Enforces the memory budget by clearing caches and evicting volatile memory.
        """
        used_mb = psutil.virtual_memory().used / (1024**2)
        
        if used_mb > self.soft_limit_mb:
            logger.warning("🚨 Memory pressure high: %.1fMB used. Enforcing budget.", used_mb)
            
            # 1. Clear MLX Cache safely
            try:
                from core.utils.gpu_sentinel import get_gpu_sentinel, GPUPriority
                sentinel = get_gpu_sentinel()
                if sentinel.acquire(priority=GPUPriority.REFLEX, timeout=5.0):
                    try:
                        import mlx.core as mx
                        if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                            mx.metal.clear_cache()
                        else:
                            mx.clear_cache()
                    finally:
                        sentinel.release()
            except Exception as e:
                record_degradation('governor', e)
                logger.debug(f"[MLX] Cache clear skipped in governor: {e}")
            
            # 2. Evict Episodic Memory
            try:
                from core.runtime import CoreRuntime
                rt = CoreRuntime.get_sync()
                dual_memory = rt.container.get("dual_memory")
                if dual_memory:
                    logger.info("Evicting oldest 30% of episodic memories.")
                    # Assuming episodic has an async evict_oldest or we run it in loop
                    # For synchronous enforcement, we might need a sync wrapper or just trigger it
                    # ISSUE 33 fix: Safe task creation from sync context
                    try:
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(dual_memory.episodic.evict_oldest(0.3))
                        except RuntimeError:
                            asyncio.run(dual_memory.episodic.evict_oldest(0.3))
                    except RuntimeError:
                        # No loop running, just run it
                        asyncio.run(dual_memory.episodic.evict_oldest(0.3))
            except Exception as e:
                record_degradation('governor', e)
                logger.error("Failed to evict episodic memory: %s", e)
            
            # 3. GC
            import gc
            gc.collect()

class MemoryGovernor:
    def __init__(self, budget: MemoryBudget = None):
        self.budget = budget or MemoryBudget()
        
    async def start(self):
        logger.info("🧠 MemoryGovernor active.")
        
    def check(self):
        self.budget.enforce()

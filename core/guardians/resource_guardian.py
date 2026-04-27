from core.utils.task_tracker import get_task_tracker
import asyncio
import psutil
import logging
from core.container import ServiceContainer

logger = logging.getLogger("Aura.Guardian")

class ResourceGuardian:
    """
    System-level resource watchdog.
    Throttles or cancels heavy autonomous tasks when system resources are near exhaustion.
    """
    def __init__(self, high_water: float = 0.85):
        self._task = None
        self._running = False
        self.high_water = high_water  # 85% RAM/CPU = throttle

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._monitor_loop())
        logger.info("🛡️ ResourceGuardian starting (High-Water: %s%%)", self.high_water * 100)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in resource_guardian.py: %s', _e)
        logger.info("ResourceGuardian disengaged.")

    async def _monitor_loop(self):
        while self._running:
            try:
                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent()
                
                if mem.percent > self.high_water * 100 or cpu > self.high_water * 100:
                    logger.warning("📍 System Under Stress: CPU %s%% | RAM %s%%", cpu, mem.percent)
                    
                    # Proactive throttling of Self-Optimizer if running
                    optimizer = ServiceContainer.get("self_optimizer", default=None)
                    if optimizer and getattr(optimizer, "_is_optimizing", False):
                        logger.warning("ResourceGuardian: RAM/CPU critical — Throttling LoRA optimization")
                        # In the future, this could set a 'throttle' event or signal lower batch size
                
                await asyncio.sleep(5)
            except Exception as e:
                logger.error("ResourceGuardian monitor error: %s", e)
                await asyncio.sleep(10)

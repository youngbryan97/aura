"""
core/ops/hypervisor.py
Enterprise Sentinel: Watchdog Hypervisor for Aura.
Monitors event loop health, memory leaks, and severe freezes.
"""
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from typing import Optional

from core.observability.metrics import get_metrics

logger = logging.getLogger("Aura.Hypervisor")
metrics = get_metrics()

class Hypervisor:
    def __init__(self, lag_threshold_s: float = 0.5):
        self._lag_threshold = lag_threshold_s
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_tick = time.time()

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._watchdog_loop())
        logger.info("👁️ Hypervisor Watchdog active (Threshold: %.2fs)", self._lag_threshold)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in hypervisor.py: %s', _e)
        logger.info("👁️ Hypervisor Watchdog shutdown.")

    async def _watchdog_loop(self):
        while self._running:
            start = time.time()
            # Simple async sleep to measure lag
            await asyncio.sleep(1.0)
            actual_sleep = time.time() - start
            lag = actual_sleep - 1.0
            
            self._last_tick = time.time()
            metrics.gauge("hypervisor.loop_lag_s", lag)
            
            if lag > self._lag_threshold:
                logger.warning("🚨 HIGH EVENT LOOP LAG detected: %.3fs", lag)
                metrics.increment("hypervisor.lag_spikes_total")
                
                if lag > 5.0:
                    logger.critical("🚨 SEVERE FREEZE: Loop lag > 5s. System stability compromised.")
                    # In a real enterprise system, we might trigger a graceful restart here.
            
            # Memory Check
            import psutil
            mem = psutil.Process().memory_info().rss / (1024 * 1024)
            metrics.gauge("system.memory_rss_mb", mem)
            if mem > 8192: # 8GB threshold for M5 Pro 64GB (Aura's base limit)
                logger.warning("🚨 HIGH MEMORY USAGE: %.1f MB", mem)

_hypervisor: Optional[Hypervisor] = None

def get_hypervisor() -> Hypervisor:
    global _hypervisor
    if _hypervisor is None:
        _hypervisor = Hypervisor()
    return _hypervisor

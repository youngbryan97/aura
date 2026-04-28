from core.runtime.errors import record_degradation
import asyncio
import logging
import psutil

logger = logging.getLogger("Aura.MemoryMonitor")

class AppleSiliconMemoryMonitor:
    """Monitors Unified Memory pressure on Apple Silicon (M1/M2/M3/M4/M5).
    
    Aura uses this to throttle background reasoning (ReasoningQueue)
    when memory pressure is high to avoid system swap lag.
    """
    def __init__(self, interval: float = 2.0, threshold: int = 85):
        self.interval = interval
        self.threshold = threshold
        self.is_running = False
        self._pressure = 0
        self._loop_task = None

    async def start(self):
        self.is_running = True
        # Use our new task tracker helper (hoisted from Part 5)
        from .task_tracker import fire_and_track
        self._loop_task = fire_and_track(self._monitor_loop(), name="MemoryMonitor")
        logger.info("Apple Silicon Memory Monitor active.")

    async def stop(self):
        self.is_running = False
        if self._loop_task:
            self._loop_task.cancel()

    @property
    def pressure(self) -> int:
        """Returns 0-100 indicating memory pressure."""
        return self._pressure

    async def _monitor_loop(self):
        import gc as _gc
        last_gc_at = 0.0
        last_purge_at = 0.0
        while self.is_running:
            try:
                # Sample memory pressure off the event loop so watchdogs never
                # see a shell command or psutil hiccup as a global stall.
                self._pressure = await asyncio.to_thread(self._get_pressure_sysctl)
                if self._pressure >= self.threshold:
                    logger.warning(f"⚠️ HIGH MEMORY PRESSURE: {self._pressure}% (Threshold: {self.threshold}%)")
                    import time as _time
                    now = _time.monotonic()
                    # Run a generational gc once per minute when pressure is up.
                    # Sustained-growth recovery had no eviction step between the
                    # 85% warning and the 90% VRAM purge, so RAM kept climbing
                    # through the gap.
                    if now - last_gc_at > 60.0:
                        await asyncio.to_thread(_gc.collect)
                        last_gc_at = now
                    # Trigger VRAM purge if critical (kept on its own cooldown
                    # so we never spin-purge the GPU heap).
                    if self._pressure > 90 and now - last_purge_at > 30.0:
                        from core.managers.vram_manager import get_vram_manager
                        await asyncio.to_thread(get_vram_manager().purge)
                        last_purge_at = now

                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('memory_monitor', e)
                logger.error(f"Memory monitor error: {e}")
                await asyncio.sleep(5)

    def _get_pressure_sysctl(self) -> int:
        """Legacy name: return a safe process-wide pressure sample."""
        try:
            mem = psutil.virtual_memory()
            return int(mem.percent)
        except Exception:
            return 0

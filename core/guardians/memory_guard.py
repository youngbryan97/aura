from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import psutil
import logging
from typing import Optional
from core.container import ServiceContainer

logger = logging.getLogger("Aura.MemoryGuard")

class MemoryGuard:
    """
    Zenith Resource Guardian: Monitors system RAM usage and prevents OOM.
    Specifically tuned for 64GB Apple M5 Pro.
    """
    def __init__(self, threshold_percent: float = 82.0):
        self.threshold_percent = threshold_percent
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.consecutive_strikes = 0

    async def start(self):
        if self._running or self._task:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._watch_loop())
        logger.info("🛡️ MemoryGuard active (Threshold: %s%%)", self.threshold_percent)

    async def stop(self):
        self._running = False
        task = self._task
        if task and not task.done():
            self._task = None
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.debug('Ignored Exception in memory_guard.py: %s', "unknown_error")
        logger.info("MemoryGuard disengaged.")

    async def _watch_loop(self):
        import gc
        import time
        from core.container import ServiceContainer
        
        base_threshold = self.threshold_percent
        adaptive_threshold = base_threshold
        self.consecutive_strikes = 0

        while self._running:
            try:
                mem = psutil.virtual_memory()
                
                # Check if we are above the ADAPTIVE threshold
                if mem.percent > adaptive_threshold:
                    self.consecutive_strikes = self.consecutive_strikes + 1
                    logger.warning("🔴 RAM CRITICAL (%s%%) - Strike %s. Defensive Mode Active.", mem.percent, self.consecutive_strikes)
                    
                    # 1. Emergency Garbage Collection
                    # GUARD: Focus Area 3 - Building the reference graph for gen-2 GC can cause MemoryError
                    # If we are above 96%, building the graph might be the final blow. Skip deep GC.
                    # v2026 FIX: Reduced to generation=1 to prevent loop stalls during boot
                    if mem.percent < 94:
                        gc.collect(generation=1) 
                    else:
                        logger.warning("⚠️ RAM > 94%: Using minimal GC to avoid stall.")
                        gc.collect(generation=0)
                    
                    # 2. Adaptive Throttling: Slightly lower threshold, but don't create a death spiral
                    # v2026 FIX: Increased min adaptive threshold to 85.0 (was 80.0)
                    adaptive_threshold = max(85.0, adaptive_threshold - 0.2)
                    
                    # 3. Component Cache Purging
                    if mem.percent > 95:
                        try:
                            router = ServiceContainer.get("llm_router", default=None)
                            gate = ServiceContainer.get("inference_gate", default=None)
                            foreground_busy = False
                            if gate:
                                try:
                                    foreground_busy = bool(
                                        (hasattr(gate, "_foreground_user_turn_active") and gate._foreground_user_turn_active())
                                        or (hasattr(gate, "_foreground_owner_active") and gate._foreground_owner_active())
                                    )
                                except Exception:
                                    foreground_busy = False
                            if foreground_busy:
                                logger.warning(
                                    "MemoryGuard: Skipping router cache purge during active foreground inference (%s%%)",
                                    mem.percent,
                                )
                            elif router and hasattr(router, "clear_cache"):
                                logger.info("MemoryGuard: Clearing LLM Router cache")
                                router.clear_cache()
                        except Exception as e:
                            record_degradation('memory_guard', e)
                            logger.error("MemoryGuard: Cache purge failed: %s", e)
                    
                    # 4. LoRA Abort
                    optimizer = ServiceContainer.get("self_optimizer", default=None)
                    if optimizer and getattr(optimizer, "_is_optimizing", False):
                        logger.critical("MemoryGuard: Aborting LoRA training to prevent system OOM")
                        if hasattr(optimizer, "abort"):
                            optimizer.abort()
                    
                    # 5. Task Throttling (Phase 31: Advanced Throttling)
                    try:
                        router = ServiceContainer.get("llm_router", default=None)
                        if router and not getattr(router, "high_pressure_mode", False):
                            logger.warning("MemoryGuard: Triggering HIGH PRESSURE mode in LLM Router (%s%%)", mem.percent)
                            router.high_pressure_mode = True
                    except Exception as e:
                        record_degradation('memory_guard', e)
                        logger.error("MemoryGuard: Setting high pressure mode failed: %s", e)

                    if mem.percent > max(adaptive_threshold + 1.5, 84.0):
                        try:
                            gate = ServiceContainer.get("inference_gate", default=None)
                            if gate and hasattr(gate, "_shed_background_workers_for_memory_pressure"):
                                logger.warning("MemoryGuard: Shedding background local-runtime workers to protect Cortex (%s%%)", mem.percent)
                                await gate._shed_background_workers_for_memory_pressure()
                        except Exception as e:
                            record_degradation('memory_guard', e)
                            logger.error("MemoryGuard: Background MLX shed failed: %s", e)

                else:
                    # Cooling down
                    if self.consecutive_strikes > 0:
                        try:
                            router = ServiceContainer.get("llm_router", default=None)
                            if router and getattr(router, "high_pressure_mode", False):
                                logger.info("MemoryGuard: System stabilized. Disabling HIGH PRESSURE mode.")
                                router.high_pressure_mode = False
                        except Exception:
                            logger.debug("MemoryGuard: Manual gc.collect() failed.")
                        
                        self.consecutive_strikes = 0
                        logger.info("🟢 RAM Stabilized. Cooling down adaptive threshold.")
                    
                    # Slowly restore base threshold
                    if adaptive_threshold < base_threshold:
                        adaptive_threshold = min(base_threshold, adaptive_threshold + 0.1)
                
                # Dynamic sleep: Check more frequently when under pressure
                # v2026 FIX: Increased check interval to reduce overhead
                sleep_time = 5 if self.consecutive_strikes > 0 else 15
                await asyncio.sleep(sleep_time)

            except Exception as e:
                record_degradation('memory_guard', e)
                # Focus Area 3 - Catch Exception (not BaseException) to allow CancelledError
                # to propagate and shut down the task cleanly.
                if not self._running:
                    break
                logger.critical("🛑 MemoryGuard SEIZURE: %s. Attempting auto-restart in 10s...", type(e).__name__)
                await asyncio.sleep(10)
                # Loop continues

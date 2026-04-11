import asyncio
import logging
import psutil
import time
import os
import gc
from typing import Dict, Any
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.AutonomicCore")

class AutonomicCore:
    """Unified Autonomic Nervous System.
    Replaces Governor, ImmuneSystem, ExistentialAwareness, and OptimizationEngine.
    Provides a single, deterministic heartbeat for system survival.
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        
        # Unified Thresholds — raised for M5 64GB hardware where the 32B model
        # alone consumes ~31% of unified memory.
        self.defrag_ram_percent = 85.0     # Substrate defrag (SnapKV + episodic eviction)
        self.throttle_ram_percent = 94.0   # Skip deep thoughts / background tasks
        self.cleanup_ram_percent = 96.0    # Aggressive GC
        self.critical_ram_percent = 98.0   # Emergency purge / model unload

        self.running = False
        self._task = None
        self.uptime_start = time.time()
        self._last_defrag_time = 0.0  # Cooldown to avoid defrag spam
        
        # Restoration Phase Integration
        from .survival_driver import SurvivalDriver
        self.survival_driver = SurvivalDriver(self.orchestrator)
        self.survival_status = {}
        
    async def start(self):
        """Boot the unified autonomic heartbeat."""
        self.running = True
        self._task = get_task_tracker().create_task(
            self._heartbeat_loop(),
            name="autonomic_core.heartbeat",
        )
        logger.info("🛡️ Autonomic Core online. Unified survival heartbeat started.")
        
    async def stop(self):
        """Shutdown the autonomic heartbeat."""
        self.running = False
        if self._task:
            self._task.cancel()
            
    async def _heartbeat_loop(self):
        """Single loop for all background survival checks."""
        while self.running:
            try:
                await self._manage_metabolism()
                await self._enforce_governance()
                await self._check_survival()
                await self._check_idle_model_swap()
                
                # Deterministic 10-second heartbeat to minimize CPU overhead
                await asyncio.sleep(10.0) 
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Autonomic Core heartbeat error: %s", e)
                await asyncio.sleep(10.0)
                
    async def _manage_metabolism(self):
        """Consolidated memory and existential checks.

        Tier cascade (lowest → highest severity):
          85% → Substrate Defrag (SnapKV eviction, episodic pruning, MLX cache clear)
          94% → Throttle (skip deep thoughts / background tasks)
          96% → Aggressive GC
          98% → Emergency purge + auto cognitive recovery
        """
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            now = time.time()

            # 1. Critical Existential Threat — auto-recovery (Zero-Touch)
            if mem.percent >= self.critical_ram_percent or disk.percent > 98.0:
                logger.critical("Critical resource pressure (RAM: %s%%, Disk: %s%%). Auto-recovery.", mem.percent, disk.percent)
                gc.collect()
                await self._substrate_defrag()
                await self._auto_cognitive_recovery()
                await self._emit_status("CRITICAL: Auto-recovery triggered at %.0f%% RAM" % mem.percent)

            # 2. Hard Cleanup Needed
            elif mem.percent >= self.cleanup_ram_percent:
                logger.warning("High RAM (%s%%). Running aggressive garbage collection.", mem.percent)
                gc.collect()
                await self._substrate_defrag()
                await self._emit_status("Memory load high. Optimizing...")

            # 3. Throttling
            elif mem.percent >= self.throttle_ram_percent:
                if self.orchestrator:
                    self.orchestrator.status.memory_pressure = True

            # 4. Substrate Defrag — consolidate BEFORE hitting the throttle wall
            elif mem.percent >= self.defrag_ram_percent:
                if self.orchestrator:
                    self.orchestrator.status.memory_pressure = False
                # Defrag at most once every 5 minutes to avoid churn
                if (now - self._last_defrag_time) > 300.0:
                    logger.info("Substrate Defrag: RAM at %.1f%%. Consolidating caches.", mem.percent)
                    await self._substrate_defrag()
                    self._last_defrag_time = now

            else:
                if self.orchestrator:
                    self.orchestrator.status.memory_pressure = False

        except Exception as e:
            logger.debug("Vitals check failed: %s", e)

    async def _substrate_defrag(self):
        """Consolidate SnapKV cache, evict stale episodic memories, and clear MLX metal cache.

        Called at 85% RAM to prevent hitting the throttle/critical wall.
        This is the automated equivalent of the manual BRAIN button.
        """
        try:
            # 1. Clear MLX metal cache (free GPU-side allocations)
            try:
                import mlx.core as mx
                if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                    mx.metal.clear_cache()
                    logger.info("Substrate Defrag: MLX metal cache cleared.")
            except Exception as e:
                logger.debug("Substrate Defrag: MLX cache clear skipped: %s", e)

            # 2. SnapKV eviction — compress the KV cache
            try:
                from core.container import ServiceContainer
                evictor = ServiceContainer.get("snap_kv_evictor", default=None)
                if evictor and hasattr(evictor, 'get_compressed_context'):
                    current_gb = psutil.virtual_memory().used / (1024 ** 3)
                    if evictor.check_memory_pressure(current_gb):
                        logger.info("Substrate Defrag: SnapKV eviction triggered at %.1fGB.", current_gb)
            except Exception as e:
                logger.debug("Substrate Defrag: SnapKV eviction skipped: %s", e)

            # 3. Episodic memory compaction: compress weak episodes into semantic
            # summaries instead of just deleting them. Preserves knowledge while
            # freeing the SQLite store.
            try:
                from core.container import ServiceContainer
                dual_memory = ServiceContainer.get("dual_memory", default=None)
                if dual_memory and hasattr(dual_memory, 'episodic'):
                    if hasattr(dual_memory.episodic, 'compact_to_semantic'):
                        result = await dual_memory.episodic.compact_to_semantic(batch_size=30)
                        if result.get("compacted", 0) > 0:
                            logger.info("Substrate Defrag: Compacted %d episodes to semantic.", result["compacted"])
                    elif hasattr(dual_memory.episodic, 'evict_oldest'):
                        await dual_memory.episodic.evict_oldest(0.2)
                        logger.info("Substrate Defrag: Evicted oldest 20%% of episodic memories.")
            except Exception as e:
                logger.debug("Substrate Defrag: Episodic compaction skipped: %s", e)

            # 4. Force garbage collection
            gc.collect()
            logger.info("Substrate Defrag: GC complete. RAM now at %.1f%%.", psutil.virtual_memory().percent)

        except Exception as e:
            logger.error("Substrate Defrag failed: %s", e)

    async def _auto_cognitive_recovery(self):
        """Zero-Touch: Automatically perform what the BRAIN/REGEN buttons do.

        Resets circuit breakers, re-initializes cognitive engine, and clears
        rate limits — no manual intervention required.
        """
        try:
            if not self.orchestrator:
                return

            from core.orchestrator.handlers.recovery import retry_cognitive_connection
            success = await retry_cognitive_connection(self.orchestrator)
            if success:
                logger.info("Zero-Touch: Cognitive auto-recovery SUCCEEDED.")
                await self._emit_status("Cognitive lane auto-recovered.")
            else:
                logger.warning("Zero-Touch: Cognitive auto-recovery failed. System degraded.")
                await self._emit_status("Auto-recovery attempted but cortex remains offline.")

        except Exception as e:
            logger.error("Zero-Touch auto-recovery error: %s", e)

    async def _check_idle_model_swap(self):
        """Model hot-swap budget: unload 32B cortex when idle to reclaim ~15GB.

        After 5 minutes with no user interaction, automatically swap the 32B
        model for the 7B brainstem. The inference gate will lazy-reload the 32B
        when the next user message arrives.

        This is the single biggest RAM reclamation available — the 32B model
        alone consumes ~20GB vs ~5GB for the 7B.
        """
        if not self.orchestrator:
            return

        try:
            last_user = getattr(self.orchestrator, '_last_user_interaction_time', 0)
            if last_user == 0:
                return

            idle_seconds = time.time() - last_user
            IDLE_THRESHOLD = 300.0  # 5 minutes

            if idle_seconds < IDLE_THRESHOLD:
                return

            # Only swap if the 32B is actually loaded
            from core.container import ServiceContainer
            mlx_client = ServiceContainer.get("mlx_client", default=None)
            if not mlx_client or not hasattr(mlx_client, 'is_alive') or not mlx_client.is_alive():
                return

            # Check if we already swapped (avoid re-triggering)
            if getattr(self, '_idle_swap_done', False):
                return

            import psutil
            ram_pct = psutil.virtual_memory().percent
            # Only swap if RAM is above 70% — if plenty of room, let the model stay warm
            if ram_pct < 70.0:
                return

            logger.info(
                "Idle model swap: No user interaction for %.0fs, RAM at %.1f%%. "
                "Unloading 32B cortex to reclaim memory.",
                idle_seconds, ram_pct,
            )

            await mlx_client.reboot_worker(reason="idle_budget_swap")
            import gc
            gc.collect()

            # Warm up brainstem (7B) so it's ready for the next request
            try:
                brainstem = ServiceContainer.get("brainstem_client", default=None)
                if brainstem and hasattr(brainstem, 'warmup'):
                    await brainstem.warmup()
                    logger.info("Idle model swap: 7B brainstem warmed up.")
            except Exception as bs_err:
                logger.debug("Brainstem warmup after idle swap skipped: %s", bs_err)

            self._idle_swap_done = True
            await self._emit_status("Cortex hibernated (idle). Brainstem active.")

        except Exception as e:
            logger.debug("Idle model swap check failed: %s", e)

    def _reset_idle_swap(self):
        """Called when a user message arrives to clear the idle swap flag."""
        self._idle_swap_done = False

    async def _enforce_governance(self):
        """Immune system watchdog: detect runaway threads and hung processes."""
        try:
            import threading
            active = threading.active_count()
            if active > 200:
                logger.warning("Thread count high (%d) — possible leak.", active)
                if self.orchestrator:
                    self.orchestrator.status.memory_pressure = True
        except Exception as e:
            logger.debug("Governance check failed: %s", e)
        
    async def _unload_llm(self):
        """Request local model unload to free VRAM/RAM under memory pressure."""
        logger.info("Requesting local model memory optimization...")
        try:
            from core.container import ServiceContainer
            mlx_client = ServiceContainer.get("mlx_client", default=None)
            if mlx_client and hasattr(mlx_client, "unload"):
                await mlx_client.unload()
                logger.info("MLX model unloaded to reclaim memory.")
                return
            # Fallback: force garbage collection of model tensors
            import gc
            gc.collect()
        except Exception as e:
            logger.debug("Model unload failed: %s", e)

    async def _check_survival(self):
        """Phase 8: Check for existential threats via SurvivalDriver."""
        try:
            self.survival_status = self.survival_driver.check_vitals()
            imperative = self.survival_driver.get_imperatives(self.survival_status)
            if imperative:
                self.survival_driver.publish_threat(imperative)
        except Exception as e:
            logger.debug("Survival check error: %s", e)

    def get_survival_report(self) -> Dict[str, Any]:
        """Provides the latest survival metrics."""
        return self.survival_status

    async def _emit_status(self, message: str) -> None:
        """Publish a status message to the event bus."""
        from core.event_bus import get_event_bus
        await get_event_bus().publish("autonomic/status", {"message": message})

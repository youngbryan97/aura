import asyncio
import gc
import inspect
import logging
import time
from typing import Any

from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation
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
        # Monotonic: wall-clock cooldowns let a clock rollback block defrag
        # indefinitely and a forward jump fire it immediately.
        self._last_defrag_monotonic = 0.0
        self._snapkv_gap_reported = False
        
        # Restoration Phase Integration
        from .survival_driver import SurvivalDriver
        self.survival_driver = SurvivalDriver(self.orchestrator)
        self.survival_status = {}
        
    async def start(self):
        """Boot the unified autonomic heartbeat (idempotent)."""
        # Single-owner guard. start() previously set running=True and created
        # a task unconditionally, so a repeated or concurrent start produced
        # DUPLICATE survival loops — two heartbeats independently running GC,
        # defrag, recovery and model swaps against the same orchestrator. A
        # failure inside create_task also left running=True with no loop,
        # advertising a heartbeat that did not exist.
        if self.running and self._task is not None and not self._task.done():
            logger.debug("Autonomic Core already running; ignoring duplicate start().")
            return
        try:
            self._task = get_task_tracker().create_task(
                self._heartbeat_loop(),
                name="autonomic_core.heartbeat",
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            self.running = False
            self._task = None
            record_degradation('core_monitor', exc)
            logger.error("Autonomic Core heartbeat failed to start: %s", exc)
            raise
        self.running = True
        logger.info("🛡️ Autonomic Core online. Unified survival heartbeat started.")

    async def stop(self):
        """Shutdown the autonomic heartbeat and prove termination."""
        self.running = False
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        # Join it: stop() used to return while the loop could still be inside
        # GC, defrag, or a model swap, so shutdown raced live resource
        # mutation and liveness could never prove the heartbeat had ended.
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=15.0)
        except (asyncio.CancelledError, TimeoutError):
            pass
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('core_monitor', exc)
            logger.debug("Autonomic heartbeat join reported: %s", exc)
            
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
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('core_monitor', e)
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
            from core.runtime.disk_budget import (
                DISK_RED_PERCENT,
                state_volume_percent,
            )

            disk_percent = state_volume_percent()
            # 1. Critical Existential Threat — auto-recovery (Zero-Touch)
            if mem.percent >= self.critical_ram_percent or disk_percent > DISK_RED_PERCENT:
                logger.critical("Critical resource pressure (RAM: %s%%, Disk: %s%%). Auto-recovery.", mem.percent, disk_percent)
                # The MOST severe tier must not leave memory_pressure reading
                # False from an earlier healthy sample: only the 94-96%
                # throttle branch used to set it, so at 96-100% every consumer
                # of orchestrator.status.memory_pressure saw "no pressure"
                # precisely when pressure was worst.
                if self.orchestrator:
                    self.orchestrator.status.memory_pressure = True
                gc.collect()
                await self._substrate_defrag()
                # The documented 98% tier is "emergency purge / model unload".
                # _unload_llm existed but was never called from here, so the
                # single largest reclaimable allocation (the resident model)
                # was never released at the tier that promised it.
                await self._unload_llm()
                await self._auto_cognitive_recovery()
                await self._emit_status(
                    f"CRITICAL: Auto-recovery triggered at {mem.percent:.0f}% RAM"
                )

            # 2. Hard Cleanup Needed
            elif mem.percent >= self.cleanup_ram_percent:
                logger.warning("High RAM (%s%%). Running aggressive garbage collection.", mem.percent)
                if self.orchestrator:
                    self.orchestrator.status.memory_pressure = True
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
                if (time.monotonic() - self._last_defrag_monotonic) > 300.0:
                    logger.info("Substrate Defrag: RAM at %.1f%%. Consolidating caches.", mem.percent)
                    await self._substrate_defrag()
                    self._last_defrag_monotonic = time.monotonic()

            else:
                if self.orchestrator:
                    self.orchestrator.status.memory_pressure = False

        except (ImportError, OSError, AttributeError) as e:
            record_degradation('core_monitor', e)
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

                from core.runtime.desktop_boot_safety import mlx_process_uses_metal

                if (
                    mlx_process_uses_metal()
                    and hasattr(mx, "metal")
                    and hasattr(mx.metal, "clear_cache")
                ):
                    mx.metal.clear_cache()
                    logger.info("Substrate Defrag: MLX metal cache cleared.")
                else:
                    logger.debug(
                        "Substrate Defrag: this process does not own Metal; "
                        "worker-side cache reclamation remains authoritative."
                    )
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('core_monitor', e)
                logger.debug("Substrate Defrag: MLX cache clear skipped: %s", e)

            # 2. SnapKV pressure probe.
            #
            # HONEST ACTION ACCOUNTING: this branch used to log "SnapKV
            # eviction triggered" while calling nothing that frees memory —
            # the advertised defrag stage was a pure no-op, so a defrag under
            # pressure reported work it had not done. The evictor exposes
            # calculate_eviction_targets() (a computation) and
            # get_compressed_context(context, ...) (compresses a string the
            # CALLER supplies); neither releases memory when invoked here
            # without a live KV context. The pressure signal is still useful,
            # so it is recorded as an unmet capability rather than claimed as
            # a completed eviction.
            try:
                from core.container import ServiceContainer
                evictor = ServiceContainer.get("snap_kv_evictor", default=None)
                if evictor and hasattr(evictor, "check_memory_pressure"):
                    current_gb = psutil.virtual_memory().used / (1024 ** 3)
                    if evictor.check_memory_pressure(current_gb):
                        if not self._snapkv_gap_reported:
                            self._snapkv_gap_reported = True
                            record_degradation(
                                'core_monitor',
                                RuntimeError(
                                    "snapkv_evictor_exposes_no_standalone_eviction"
                                ),
                                severity="warning",
                                action=(
                                    "reported SnapKV pressure without claiming an "
                                    "eviction; defrag relies on MLX cache clear, "
                                    "episodic compaction and GC"
                                ),
                            )
                        logger.info(
                            "Substrate Defrag: SnapKV reports pressure at %.1fGB "
                            "(no standalone eviction available; other stages proceed).",
                            current_gb,
                        )
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('core_monitor', e)
                logger.debug("Substrate Defrag: SnapKV pressure probe skipped: %s", e)

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
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('core_monitor', e)
                logger.debug("Substrate Defrag: Episodic compaction skipped: %s", e)

            # 4. Force garbage collection
            gc.collect()
            logger.info("Substrate Defrag: GC complete. RAM now at %.1f%%.", psutil.virtual_memory().percent)

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('core_monitor', e)
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

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('core_monitor', e)
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
            idle_threshold = 300.0  # 5 minutes

            if idle_seconds < idle_threshold:
                return

            # Only swap if the 32B is actually loaded.
            from core.container import ServiceContainer
            mlx_client = ServiceContainer.get("mlx_client", default=None)
            if not mlx_client or not hasattr(mlx_client, 'is_alive'):
                return
            # is_alive may be async. Calling it bare returned a COROUTINE,
            # which is always truthy — so the "model is loaded" check passed
            # unconditionally (and leaked an un-awaited coroutine each pass).
            alive = mlx_client.is_alive()
            if inspect.isawaitable(alive):
                alive = await alive
            if not alive:
                return

            # Check if we already swapped (avoid re-triggering)
            if getattr(self, '_idle_swap_done', False):
                return

            from core.runtime import resource_psutil as psutil
            ram_pct = psutil.virtual_memory().percent
            # Only swap if RAM is above 70% — if plenty of room, let the model stay warm
            if ram_pct < 70.0:
                return

            logger.info(
                "Idle model swap: No user interaction for %.0fs, RAM at %.1f%%. "
                "Unloading 32B cortex to reclaim memory.",
                idle_seconds, ram_pct,
            )

            # UNLOAD, not reboot. The docstring promises the 32B is unloaded
            # to reclaim ~15GB, but reboot_worker RESTARTS the same worker on
            # the same model — which reloads those weights rather than
            # releasing them, so the advertised reclamation never happened.
            # Prefer a real unload and fall back to reboot only if this client
            # exposes no unload path.
            if hasattr(mlx_client, "unload"):
                await mlx_client.unload()
            else:
                logger.warning(
                    "Idle model swap: mlx_client exposes no unload(); falling back to "
                    "reboot_worker, which does NOT reclaim the model's memory."
                )
                await mlx_client.reboot_worker(reason="idle_budget_swap")
            import gc
            gc.collect()

            # Warm up brainstem (7B) so it's ready for the next request
            brainstem_ready = False
            try:
                brainstem = ServiceContainer.get("brainstem_client", default=None)
                if brainstem and hasattr(brainstem, 'warmup'):
                    warmup_result = await brainstem.warmup()
                    lane = (
                        brainstem.get_lane_status()
                        if hasattr(brainstem, "get_lane_status")
                        else {}
                    )
                    if warmup_result is not False and lane.get("conversation_ready", False):
                        brainstem_ready = True
                        logger.info("Idle model swap: 7B brainstem warmed up.")
                    else:
                        logger.warning(
                            "Idle model swap: 7B brainstem warmup did not establish readiness "
                            "(state=%s, reason=%s).",
                            lane.get("state", "unknown"),
                            lane.get("last_error", "warmup_not_ready"),
                        )
            except (ImportError, AttributeError, RuntimeError) as bs_err:
                record_degradation('core_monitor', bs_err)
                logger.debug("Brainstem warmup after idle swap skipped: %s", bs_err)

            # The swap itself completed — the cortex was unloaded and its
            # memory reclaimed — so the flag is set to stop re-triggering.
            # Brainstem warmup is a separate best-effort step, and the status
            # emitted below states plainly when it did not establish
            # readiness, so "hibernated" is never an unqualified claim of a
            # ready replacement lane. A deferred warmup is still recorded so
            # the incomplete swap is visible in health rather than only in a
            # log line.
            self._idle_swap_done = True
            if not brainstem_ready:
                record_degradation(
                    'core_monitor',
                    RuntimeError("idle_swap_left_no_warm_conversation_lane"),
                    severity="warning",
                    action=(
                        "unloaded the cortex for the idle budget but no brainstem "
                        "lane warmed; the next foreground turn pays a cold start"
                    ),
                )
            if brainstem_ready:
                await self._emit_status("Cortex hibernated (idle). Brainstem active.")
            else:
                await self._emit_status(
                    "Cortex hibernated (idle). Brainstem warmup incomplete; "
                    "foreground demand will restore the Cortex."
                )

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('core_monitor', e)
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
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('core_monitor', e)
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
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('core_monitor', e)
            logger.debug("Model unload failed: %s", e)

    async def _check_survival(self):
        """Phase 8: Check for existential threats via SurvivalDriver."""
        try:
            self.survival_status = self.survival_driver.check_vitals()
            imperative = self.survival_driver.get_imperatives(self.survival_status)
            if imperative:
                self.survival_driver.publish_threat(imperative)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('core_monitor', e)
            logger.debug("Survival check error: %s", e)

    def get_survival_report(self) -> dict[str, Any]:
        """Provides the latest survival metrics (a copy).

        The live dictionary was returned by reference, so any consumer
        could rewrite the monitor's own survival evidence — and every
        later reader, including the display, would see the altered
        values as though they had been measured.
        """
        return dict(self.survival_status or {})

    async def _emit_status(self, message: str) -> None:
        """Publish a status message to the event bus."""
        from core.event_bus import get_event_bus
        await get_event_bus().publish("autonomic/status", {"message": message})

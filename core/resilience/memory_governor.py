"""Aura Zenith Memory Governor.
Enforces resource constraints for 64GB M5 Pro stability.
"""
from core.utils.exceptions import capture_and_log
import asyncio
import logging
import os
import time
from typing import Optional, Any
import psutil


from core.memory.physics import hawking_decay

logger = logging.getLogger("Aura.Resilience.MemoryGovernor")

class MemoryGovernor:
    """Monitors system memory and enforces pruning/unloading thresholds.
    
    Thresholds:
    - 32GB: Trigger VectorMemory pruning.
    - 48GB: Trigger LLM model unloading.
    - 56GB: Emergency cleanup and metabolic slowdown.
    """
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._proc = psutil.Process(os.getpid())
        
        # Thresholds in MB (scaled for 64GB M5 Pro)
        self.threshold_prune = 32768
        self.threshold_unload = 48000
        self.threshold_critical = 56000
        
        self.check_interval = 60.0  # Seconds
        self._last_vacuum_time = time.monotonic()
        self._last_vector_prune_time = time.monotonic()
        self.vector_prune_interval_s = 86400.0
        self.prune_cooldown_s = 600.0
        self.unload_cooldown_s = 900.0
        self.critical_cooldown_s = 300.0
        self.prune_hysteresis_mb = 1024.0
        self.unload_hysteresis_mb = 2048.0
        self._last_prune_action_time = 0.0
        self._last_unload_action_time = 0.0
        self._last_critical_action_time = 0.0
        self._last_prune_rss_mb = 0.0
        self._last_unload_rss_mb = 0.0

    def _iter_managed_runtime_processes(self):
        """Yield heavyweight local-runtime processes owned by Aura."""
        try:
            children = self._proc.children(recursive=True)
            descendant_pids = {
                int(getattr(proc, "pid", 0) or 0)
                for proc in children
                if getattr(proc, "pid", None) is not None
            }
        except Exception as exc:
            logger.debug("Memory Governor: could not inspect child process tree: %s", exc)
            descendant_pids = set()

        if not descendant_pids:
            return

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
            try:
                if int(proc.info.get('pid') or 0) not in descendant_pids:
                    continue
                cmdline = proc.info.get('cmdline') or []
                cmd_str = " ".join(cmdline)
                if "llama-server" in cmd_str or "mlx_worker.py" in cmd_str or "MTLCompilerService" in cmd_str:
                    yield proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def _managed_runtime_rss_mb(self) -> float:
        total_mb = 0.0
        for proc in self._iter_managed_runtime_processes():
            try:
                mem_info = proc.info.get('memory_info')
                rss = getattr(mem_info, 'rss', 0) if mem_info is not None else proc.memory_info().rss
                total_mb += rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_mb

    async def start(self):
        """Start the governor loop."""
        self.is_running = True
        try:
            from core.utils.task_tracker import get_task_tracker

            self._task = get_task_tracker().create_task(
                self._run_loop(),
                name="aura.memory_governor",
            )
        except Exception:
            self._task = asyncio.create_task(self._run_loop(), name="aura.memory_governor")
        logger.info("🛡️ Memory Governor active. Thresholds: Prune=%dMB, Unload=%dMB, Critical=%dMB", 
                    self.threshold_prune, self.threshold_unload, self.threshold_critical)

    async def stop(self):
        """Stop the governor loop."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in memory_governor.py: %s', _e)
            finally:
                self._task = None
        
        # v8.1.0: Ensure total cleanup of any leaked worker handles (ORPHAN-05/07)
        try:
            await self._critical_cleanup()
            logger.info("🛡️ Memory Governor shutdown complete. All worker handles purged.")
        except Exception as e:
            logger.error(f"Error during Memory Governor shutdown: {e}")

    async def _run_loop(self):
        """Periodic resource check and enforcement."""
        while self.is_running:
            try:
                await self._enforce_policy()
                await self._periodic_vector_prune()
                await self._periodic_db_vacuum()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Memory Governor loop error: %s", e)
                await asyncio.sleep(10)

    async def _enforce_policy(self):
        """Check RSS memory and system-wide RAM to trigger cleanup actions."""
        now = time.monotonic()
        # 1. Check Process Memory
        rss_mb = self._proc.memory_info().rss / (1024 * 1024)
        runtime_rss_mb = self._managed_runtime_rss_mb()
        managed_rss_mb = rss_mb + runtime_rss_mb
        logger.debug(
            "Managed RSS: core=%.2f MB runtime=%.2f MB total=%.2f MB",
            rss_mb,
            runtime_rss_mb,
            managed_rss_mb,
        )

        # 2. Check Global System Memory (Neural Purge Trigger)
        vm = psutil.virtual_memory()
        sys_percent = vm.percent

        if sys_percent > 98.0:
            logger.critical("🚨 HIGH RAM: System at %.1f%%. Checking for idle local-runtime workers.", sys_percent)
            
            # v8.0.1: Neural Purge of idle workers
            idle_purged = 0
            for proc in self._iter_managed_runtime_processes():
                try:
                    cmdline = proc.info.get('cmdline') or []
                    cmd_str = " ".join(cmdline)
                    cpu = proc.cpu_percent(interval=0.1)
                    if cpu < 1.0:
                        logger.warning(
                            "🚨 NEURAL PURGE: Terminating idle runtime worker (PID: %d, CPU: %.1f%%, CMD: %s) to reclaim RAM.",
                            proc.info['pid'],
                            cpu,
                            cmd_str[:180],
                        )
                        proc.kill()
                        idle_purged += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if idle_purged > 0:
                logger.info("✅ NEURAL PURGE: Reclaimed RAM from %d idle workers.", idle_purged)
            
            # If still critical or core RSS is huge, trigger full cleanup
            if sys_percent > 95.0 or managed_rss_mb > self.threshold_critical:
                if (now - self._last_critical_action_time) >= self.critical_cooldown_s:
                    logger.critical("🚨 EMERGENCY: RAM remains critical (%.1f%%). Triggering FULL cleanup.", sys_percent)
                    await self._critical_cleanup()
                    self._last_critical_action_time = now
                else:
                    logger.debug("Memory Governor: critical cleanup cooldown active.")
        elif sys_percent >= 88.0 or managed_rss_mb > self.threshold_unload:
            should_unload = (
                sys_percent >= 88.0
                or managed_rss_mb >= (self._last_unload_rss_mb + self.unload_hysteresis_mb)
                or self._last_unload_rss_mb == 0.0
            )
            if (now - self._last_unload_action_time) >= self.unload_cooldown_s and should_unload:
                logger.warning(
                    "⚠️ High Memory: total managed runtime %.2f MB (core %.2f + runtime %.2f). Unloading idle local models.",
                    managed_rss_mb,
                    rss_mb,
                    runtime_rss_mb,
                )
                await self._unload_models()
                self._last_unload_action_time = now
                self._last_unload_rss_mb = managed_rss_mb
            else:
                logger.debug("Memory Governor: unload skipped (cooldown or stable baseline).")
        elif sys_percent >= 84.0 or managed_rss_mb > self.threshold_prune:
            should_prune = (
                sys_percent >= 84.0
                or managed_rss_mb >= (self._last_prune_rss_mb + self.prune_hysteresis_mb)
                or self._last_prune_rss_mb == 0.0
            )
            if (now - self._last_prune_action_time) >= self.prune_cooldown_s and should_prune:
                logger.info(
                    "ℹ️ Low Salience Pruning Triggered: managed total %.2f MB (system %.1f%%)",
                    managed_rss_mb,
                    sys_percent,
                )
                await self._prune_memory()
                self._last_prune_action_time = now
                self._last_prune_rss_mb = managed_rss_mb
            else:
                logger.debug("Memory Governor: prune skipped (cooldown or stable baseline).")

    async def _prune_memory(self):
        """Trigger strategic forgetting in vector and dual memory (Hawking Decay)."""
        try:
            # 1. Hawking Decay for DualMemory (Episodic/Semantic)
            if hasattr(self.orchestrator, "memory_manager"):
                mm = self.orchestrator.memory_manager
                if hasattr(mm, "dual_memory") and mm.dual_memory:
                    dm = mm.dual_memory
                    evaporated = 0
                    
                    if hasattr(dm, "episodic") and dm.vault_key:
                        episodes = dm.episodic.get_all_episodes()
                        for ep in episodes:
                            decay = hawking_decay(int(ep.timestamp * 1000), dm.vault_key)
                            if decay["fidelity"] < 0.1:
                                # Evaporate completely
                                with __import__('sqlite3').connect(dm.episodic.db_path) as conn:
                                     conn.execute("DELETE FROM episodes WHERE id=?", (ep.id,))
                                evaporated += 1
                            else:
                                # Update strength
                                ep.decay_rate = 1.0 - decay["fidelity"]
                                dm.episodic.store(ep)
                        if evaporated > 0:
                             logger.info("🌌 Hawking Decay: Evaporated %d forgotten episodes.", evaporated)

            # 2. Legacy Vector Memory Pruning
            if hasattr(self.orchestrator, "memory_manager"):
                mm = self.orchestrator.memory_manager
                if hasattr(mm, "vector") and mm.vector:
                    # Prune memories older than 30 days with low salience
                    pruned = mm.vector.prune_low_salience(threshold_days=30, min_salience=-0.2)
                    if pruned > 0:
                        logger.info("✅ Pruned %d low-salience vectors.", pruned)
        except Exception as e:
            logger.error("Failed to prune memory: %s", e)

    async def _unload_models(self):
        """Unload LLM models from VRAM/RAM."""
        try:
            if hasattr(self.orchestrator, "llm_router"):
                await self.orchestrator.llm_router.unload_models()
                logger.info("✅ LLM models unloaded from memory.")

            try:
                from core.container import ServiceContainer

                gate = ServiceContainer.get("inference_gate", default=None)
                if gate and hasattr(gate, "_shed_background_workers_for_memory_pressure"):
                    await gate._shed_background_workers_for_memory_pressure()
            except Exception as e:
                logger.debug("InferenceGate background shed skipped: %s", e)

            try:
                from core.brain.llm.local_server_client import _SERVER_CLIENTS
                from core.brain.llm.model_registry import PRIMARY_ENDPOINT

                unloaded = 0
                for client in list(_SERVER_CLIENTS.values()):
                    if client is None:
                        continue
                    if getattr(client, "_lane_name", "") == PRIMARY_ENDPOINT:
                        continue
                    if hasattr(client, "_is_runtime_resident") and not client._is_runtime_resident():
                        continue
                    if hasattr(client, "reboot_worker"):
                        await client.reboot_worker(
                            reason="memory_governor_unload",
                            mark_failed=False,
                        )
                        unloaded += 1
                if unloaded:
                    logger.info("✅ Local runtime lanes unloaded: %d", unloaded)
            except Exception as e:
                logger.debug("Local runtime unload skipped: %s", e)
            
            # v50: Aggressive MLX VRAM Purge
            try:
                import mlx.core as mx
                try:
                    from core.utils.gpu_sentinel import get_gpu_sentinel, GPUPriority
                    sentinel = get_gpu_sentinel()
                    if sentinel.acquire(priority=GPUPriority.REFLEX, timeout=5.0):
                        try:
                            if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                                mx.metal.clear_cache()
                            else:
                                mx.clear_cache()
                            logger.info("🌿 [MLX] Metal cache cleared aggressively.")
                        finally:
                            sentinel.release()
                except Exception as e:
                    logger.debug(f"[MLX] Cache clear skipped: {e}")
            except ImportError as _e:
                logger.debug('Ignored ImportError in memory_governor.py: %s', _e)

            # Also notify cognitive engine to clear caches
            if hasattr(self.orchestrator, "cognitive_engine"):
                 ce = self.orchestrator.cognitive_engine
                 if ce and hasattr(ce, "nucleus") and ce.nucleus:
                      await ce.nucleus.unload_models()
        except Exception as e:
            logger.error("Failed to unload models: %s", e)

    async def _periodic_db_vacuum(self):
        """Prevents SQLite file bloat and fragmentation without locking the loop."""
        try:
            from core.resilience.database_coordinator import get_db_coordinator
            db_coord = get_db_coordinator()
            # v50: Only vacuum if we haven't in 24 hours (86400s)
            if time.monotonic() - self._last_vacuum_time > 86400:
                logger.info("🧹 [MEMORY] Starting scheduled DB vacuum (Safe/Non-blocking)...")
                # We use to_thread because VACUUM is a heavy synchronous SQLite command
                await asyncio.to_thread(db_coord.vacuum_all_databases)
                self._last_vacuum_time = time.monotonic()
        except Exception as e:
            logger.error(f"VACUUM Failed: {e}")

    async def _periodic_vector_prune(self):
        """Prune low-salience memories on a real schedule, not only under pressure."""
        try:
            if time.monotonic() - self._last_vector_prune_time <= self.vector_prune_interval_s:
                return
            await self._prune_memory()
            self._last_vector_prune_time = time.monotonic()
        except Exception as e:
            logger.error(f"Periodic vector prune failed: {e}")

    async def _critical_cleanup(self):
        """Maximum effort cleanup."""
        logger.critical("🚨 NEURAL PURGE: Killing heavy local-runtime workers to recover system RAM.")
        
        await self._unload_models()
        
        # v7.2.3: Force kill heavy runtime processes if they are hanging/heavy
        try:
            for proc in self._iter_managed_runtime_processes():
                cmdline = proc.info.get('cmdline') or []
                cmd_str = " ".join(cmdline)
                if "llama-server" in cmd_str or "mlx_worker.py" in cmd_str or "MTLCompilerService" in cmd_str:
                    logger.warning("🚨 NEURAL PURGE: Forcible termination of heavy MLX/Metal process (PID: %d, Name: %s)", 
                                   proc.info['pid'], proc.info['name'])
                    proc.kill()
        except Exception as e:
            logger.error("Failed to kill heavy processes: %s", e)

        # Integrated Adrenaline Surge: Signal distress to AffectEngine
        try:
            from core.container import ServiceContainer
            affect = ServiceContainer.get("affect", default=None)
            if affect and hasattr(affect, "react"):
                asyncio.create_task(affect.react("critical_resource_exhaustion", {"intensity": 1.0}))
        except Exception as e:
            logger.debug("Failed to trigger adrenaline surcharge: %s", e)

        await self._prune_memory()
        
        # Trigger explicit GC
        import gc
        gc.collect()
        
        # Signal metabolism engine to slow down
        try:
            from core.container import ServiceContainer
            metabolism = ServiceContainer.get("metabolic_monitor", default=None)
            if metabolism and hasattr(metabolism, "force_rest"):
                await metabolism.force_rest(duration=300)
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        # v50 Hardening: Reclaim Metal Compiler Context immediately after purge
        try:
            from core.container import ServiceContainer
            root = ServiceContainer.get("platform_root", default=None)
            if root and hasattr(root, "force_compiler_wake"):
                logger.info("🌿 [MEMORY GOVERNOR] Reclaiming Metal context post-purge...")
                root.force_compiler_wake()
        except Exception as e:
            logger.error(f"Failed to pulse platform root: {e}")

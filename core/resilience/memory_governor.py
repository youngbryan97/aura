import asyncio
import gc
import logging
import os
import sqlite3
import time
from typing import Any

import psutil

from core.memory.physics import hawking_decay
from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Resilience.MemoryGovernor")


_PROCESS_INSPECTION_ERRORS = (
    psutil.NoSuchProcess,
    psutil.AccessDenied,
    psutil.ZombieProcess,
)
_MODEL_UNLOAD_ERRORS = (
    AttributeError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_RSS_SAMPLE_ERRORS = (
    AttributeError,
    RuntimeError,
    TypeError,
) + _PROCESS_INSPECTION_ERRORS
_PROCESS_TERMINATION_ERRORS = _PROCESS_INSPECTION_ERRORS + (OSError,)


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
        self._task: asyncio.Task | None = None
        self._proc = psutil.Process(os.getpid())

        # Thresholds in MB (scaled for 64GB M5 Pro)
        self.threshold_prune = 32768
        self.threshold_unload = 48000
        self.threshold_critical = 56000

        self.check_interval = 60.0  # Seconds
        self._last_vacuum_time = time.monotonic()
        self._last_vector_prune_time = time.monotonic()
        self.vector_prune_interval_s = 86400.0
        self.prune_cooldown_s = 120.0
        self.unload_cooldown_s = 180.0
        self.critical_cooldown_s = 60.0
        self.prune_hysteresis_mb = 1024.0
        self.unload_hysteresis_mb = 2048.0
        self._last_prune_action_time = 0.0
        self._last_unload_action_time = 0.0
        self._last_critical_action_time = 0.0
        self._last_prune_rss_mb = 0.0
        self._last_unload_rss_mb = 0.0
        self._loop_failure_count = 0
        self._last_policy_sample: dict[str, float] = {}
        self._cleanup_events: list[dict[str, Any]] = []

    def health_snapshot(self) -> dict[str, Any]:
        """Return a compact operational view for health probes and dashboards."""
        return {
            "running": self.is_running,
            "task_alive": self._task is not None and not self._task.done(),
            "loop_failure_count": self._loop_failure_count,
            "thresholds_mb": {
                "prune": self.threshold_prune,
                "unload": self.threshold_unload,
                "critical": self.threshold_critical,
            },
            "last_policy_sample": dict(self._last_policy_sample),
            "recent_cleanup_events": list(self._cleanup_events[-10:]),
        }

    def _remember_cleanup_event(self, action: str, status: str, detail: str = "") -> None:
        self._cleanup_events.append(
            {
                "at": time.time(),
                "action": action,
                "status": status,
                "detail": detail[:240],
            }
        )
        if len(self._cleanup_events) > 40:
            self._cleanup_events = self._cleanup_events[-40:]

    def _record_degradation(
        self,
        exc: BaseException,
        *,
        action: str,
        severity: str = "degraded",
    ):
        self._remember_cleanup_event(action, "degraded", f"{type(exc).__name__}: {exc}")
        return record_degradation(
            "memory_governor",
            exc,
            severity=severity,
            action=action,
        )

    def _iter_managed_runtime_processes(self):
        """Yield heavyweight local-runtime processes owned by Aura."""
        try:
            children = self._proc.children(recursive=True)
            descendant_pids = {
                int(getattr(proc, "pid", 0) or 0)
                for proc in children
                if getattr(proc, "pid", None) is not None
            }
        except (RuntimeError, AttributeError, TypeError) as exc:
            self._record_degradation(
                exc,
                severity="warning",
                action="skipped managed-runtime RSS scan after child process inspection failed",
            )
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
            except _PROCESS_INSPECTION_ERRORS:
                continue

    def _managed_runtime_rss_mb(self) -> float:
        total_mb = 0.0
        for proc in self._iter_managed_runtime_processes():
            try:
                mem_info = proc.info.get('memory_info')
                rss = getattr(mem_info, 'rss', 0) if mem_info is not None else proc.memory_info().rss
                total_mb += rss / (1024 * 1024)
            except _PROCESS_INSPECTION_ERRORS:
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
        except (ImportError, AttributeError, RuntimeError):
            self._task = get_task_tracker().create_task(self._run_loop(), name="aura.memory_governor")
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            self._record_degradation(
                e,
                action="completed memory-governor shutdown after critical cleanup failed",
            )
            logger.error("Error during Memory Governor shutdown: %s", e)

    async def _run_loop(self):
        """Periodic resource check and enforcement."""
        while self.is_running:
            try:
                await self._enforce_policy()
                await self._periodic_vector_prune()
                await self._periodic_db_vacuum()
                self._loop_failure_count = 0
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self._loop_failure_count += 1
                self._record_degradation(
                    e,
                    action=(
                        "backed off memory-governor loop after policy failure "
                        f"#{self._loop_failure_count}"
                    ),
                )
                logger.error("Memory Governor loop error: %s", e)
                if self._loop_failure_count >= 3:
                    reclaimed = gc.collect()
                    self._remember_cleanup_event(
                        "loop_failure_gc",
                        "ok",
                        f"collected={reclaimed}",
                    )
                await asyncio.sleep(10)

    async def _enforce_policy(self):
        """Check RSS memory and system-wide RAM to trigger cleanup actions."""
        now = time.monotonic()
        # 1. Check Process Memory
        try:
            rss_mb = self._proc.memory_info().rss / (1024 * 1024)
        except _RSS_SAMPLE_ERRORS as exc:
            self._record_degradation(
                exc,
                severity="warning",
                action="continued memory policy with zero core RSS after process sample failed",
            )
            rss_mb = 0.0
        runtime_rss_mb = self._managed_runtime_rss_mb()
        managed_rss_mb = rss_mb + runtime_rss_mb
        logger.debug(
            "Managed RSS: core=%.2f MB runtime=%.2f MB total=%.2f MB",
            rss_mb,
            runtime_rss_mb,
            managed_rss_mb,
        )

        # 2. Check Global System Memory (Neural Purge Trigger)
        try:
            from core.utils.memory_monitor import AppleSiliconMemoryMonitor
            sys_percent = AppleSiliconMemoryMonitor()._get_pressure_sysctl()
        except (ImportError, AttributeError, RuntimeError):
            vm = psutil.virtual_memory()
            sys_percent = vm.percent
        self._last_policy_sample = {
            "core_rss_mb": rss_mb,
            "runtime_rss_mb": runtime_rss_mb,
            "managed_rss_mb": managed_rss_mb,
            "system_percent": float(sys_percent),
            "sampled_at": time.time(),
        }

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
                except _PROCESS_INSPECTION_ERRORS as exc:
                    self._record_degradation(
                        exc,
                        severity="warning",
                        action="continued neural purge after idle runtime worker disappeared or was denied",
                    )
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

    async def _prune_memory(self) -> dict[str, int]:
        """Trigger strategic forgetting in vector and dual memory without cross-lane blockage."""
        result = {"episodes_evaporated": 0, "vectors_pruned": 0}
        mm = getattr(self.orchestrator, "memory_manager", None)
        if mm is None:
            self._remember_cleanup_event("prune_memory", "skipped", "memory_manager unavailable")
            return result

        dm = getattr(mm, "dual_memory", None)
        if dm:
            try:
                episodic = getattr(dm, "episodic", None)
                vault_key = getattr(dm, "vault_key", None)
                if episodic and vault_key:
                    episodes = episodic.get_all_episodes()
                    for ep in episodes:
                        decay = hawking_decay(int(ep.timestamp * 1000), vault_key)
                        if decay["fidelity"] < 0.1:
                            with sqlite3.connect(episodic.db_path) as conn:
                                conn.execute("DELETE FROM episodes WHERE id=?", (ep.id,))
                            result["episodes_evaporated"] += 1
                        else:
                            ep.decay_rate = 1.0 - decay["fidelity"]
                            episodic.store(ep)
                    if result["episodes_evaporated"] > 0:
                        logger.info(
                            "🌌 Hawking Decay: Evaporated %d forgotten episodes.",
                            result["episodes_evaporated"],
                        )
            except (sqlite3.Error, OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
                self._record_degradation(
                    e,
                    action="continued vector pruning after dual-memory Hawking decay failed",
                )
                logger.error("Dual-memory pruning failed: %s", e)

        vector = getattr(mm, "vector", None)
        if vector:
            try:
                pruned = vector.prune_low_salience(threshold_days=30, min_salience=-0.2) or 0
                result["vectors_pruned"] = int(pruned)
                if result["vectors_pruned"] > 0:
                    logger.info("✅ Pruned %d low-salience vectors.", result["vectors_pruned"])
            except (RuntimeError, AttributeError, TypeError, ValueError, sqlite3.Error, OSError) as e:
                self._record_degradation(
                    e,
                    action="completed memory prune pass after vector pruning failed",
                )
                logger.error("Vector memory pruning failed: %s", e)

        self._remember_cleanup_event(
            "prune_memory",
            "ok",
            f"episodes={result['episodes_evaporated']} vectors={result['vectors_pruned']}",
        )
        return result

    def _clear_mlx_cache(self, mx: Any) -> None:
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        else:
            mx.clear_cache()

    async def _unload_models(self) -> dict[str, int]:
        """Unload LLM models from VRAM/RAM while keeping each recovery lane independent."""
        result = {
            "router_unloaded": 0,
            "background_workers_shed": 0,
            "local_runtime_lanes_rebooted": 0,
            "mlx_cache_cleared": 0,
            "cognitive_nucleus_unloaded": 0,
        }

        llm_router = getattr(self.orchestrator, "llm_router", None)
        if llm_router and hasattr(llm_router, "unload_models"):
            try:
                await llm_router.unload_models()
                result["router_unloaded"] = 1
                logger.info("✅ LLM models unloaded from memory.")
            except _MODEL_UNLOAD_ERRORS as e:
                self._record_degradation(
                    e,
                    action="continued model unload sweep after llm_router unload failed",
                )
                logger.error("LLM router unload failed: %s", e)

        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_shed_background_workers_for_memory_pressure"):
                await gate._shed_background_workers_for_memory_pressure()
                result["background_workers_shed"] = 1
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            self._record_degradation(
                e,
                severity="warning",
                action="continued model unload sweep without inference-gate background shedding",
            )
            logger.debug("InferenceGate background shed skipped: %s", e)

        try:
            from core.brain.llm.local_server_client import _SERVER_CLIENTS
            from core.brain.llm.model_registry import PRIMARY_ENDPOINT

            for client in list(_SERVER_CLIENTS.values()):
                if client is None:
                    continue
                try:
                    if getattr(client, "_lane_name", "") == PRIMARY_ENDPOINT:
                        continue
                    if hasattr(client, "_is_runtime_resident") and not client._is_runtime_resident():
                        continue
                    if hasattr(client, "reboot_worker"):
                        await client.reboot_worker(
                            reason="memory_governor_unload",
                            mark_failed=False,
                        )
                        result["local_runtime_lanes_rebooted"] += 1
                except _MODEL_UNLOAD_ERRORS as e:
                    self._record_degradation(
                        e,
                        action="continued unloading remaining local runtime lanes after client reboot failed",
                    )
                    logger.debug("Local runtime client unload failed: %s", e)
            if result["local_runtime_lanes_rebooted"]:
                logger.info("✅ Local runtime lanes unloaded: %d", result["local_runtime_lanes_rebooted"])
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            self._record_degradation(
                e,
                severity="warning",
                action="continued model unload sweep without local runtime registry access",
            )
            logger.debug("Local runtime unload skipped: %s", e)

        try:
            import mlx.core as mx
        except ImportError as e:
            logger.debug("MLX cache clear skipped because mlx is unavailable: %s", e)
        else:
            sentinel = None
            acquired = False
            try:
                from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel

                sentinel = get_gpu_sentinel()
                acquired = sentinel.acquire(priority=GPUPriority.REFLEX, timeout=5.0)
                if not acquired:
                    raise TimeoutError("GPU sentinel was busy during memory pressure cleanup")
                self._clear_mlx_cache(mx)
                result["mlx_cache_cleared"] = 1
                logger.info("🌿 [MLX] Metal cache cleared aggressively.")
            except (ImportError, AttributeError) as e:
                self._record_degradation(
                    e,
                    severity="warning",
                    action="cleared MLX cache without GPU sentinel because sentinel was unavailable",
                )
                try:
                    self._clear_mlx_cache(mx)
                    result["mlx_cache_cleared"] = 1
                except _MODEL_UNLOAD_ERRORS as cache_exc:
                    self._record_degradation(
                        cache_exc,
                        action="continued model unload sweep after direct MLX cache clear failed",
                    )
                    logger.debug("[MLX] Direct cache clear failed: %s", cache_exc)
            except _MODEL_UNLOAD_ERRORS as e:
                self._record_degradation(
                    e,
                    action="continued model unload sweep after MLX cache clear failed",
                )
                logger.debug("[MLX] Cache clear skipped: %s", e)
            finally:
                if acquired and sentinel is not None:
                    try:
                        sentinel.release()
                    except RuntimeError as e:
                        self._record_degradation(
                            e,
                            severity="warning",
                            action="completed MLX cache cleanup after GPU sentinel release failed",
                        )

        ce = getattr(self.orchestrator, "cognitive_engine", None)
        nucleus = getattr(ce, "nucleus", None) if ce else None
        if nucleus and hasattr(nucleus, "unload_models"):
            try:
                await nucleus.unload_models()
                result["cognitive_nucleus_unloaded"] = 1
            except _MODEL_UNLOAD_ERRORS as e:
                self._record_degradation(
                    e,
                    action="completed model unload sweep after cognitive nucleus unload failed",
                )
                logger.error("Cognitive nucleus unload failed: %s", e)

        self._remember_cleanup_event("unload_models", "ok", str(result))
        return result

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
        except (ImportError, AttributeError, RuntimeError) as e:
            self._record_degradation(
                e,
                severity="warning",
                action="scheduled database vacuum skipped after coordinator failure",
            )
            logger.error("VACUUM Failed: %s", e)

    async def _periodic_vector_prune(self):
        """Prune low-salience memories on a real schedule, not only under pressure."""
        try:
            if time.monotonic() - self._last_vector_prune_time <= self.vector_prune_interval_s:
                return
            await self._prune_memory()
            self._last_vector_prune_time = time.monotonic()
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            self._record_degradation(
                e,
                action="left vector-prune schedule unchanged after periodic prune failure",
            )
            logger.error("Periodic vector prune failed: %s", e)

    async def _critical_cleanup(self):
        """Maximum effort cleanup."""
        logger.critical("🚨 NEURAL PURGE: Killing heavy local-runtime workers to recover system RAM.")
        
        unload_result = await self._unload_models()

        # v7.2.3: Force kill heavy runtime processes if they are hanging/heavy
        killed = 0
        try:
            for proc in self._iter_managed_runtime_processes():
                try:
                    cmdline = proc.info.get('cmdline') or []
                    cmd_str = " ".join(cmdline)
                    if "llama-server" in cmd_str or "mlx_worker.py" in cmd_str or "MTLCompilerService" in cmd_str:
                        logger.warning(
                            "🚨 NEURAL PURGE: Forcible termination of heavy MLX/Metal process "
                            "(PID: %d, Name: %s)",
                            proc.info['pid'],
                            proc.info['name'],
                        )
                        proc.kill()
                        killed += 1
                except _PROCESS_TERMINATION_ERRORS as e:
                    self._record_degradation(
                        e,
                        severity="warning",
                        action="continued critical cleanup after runtime process termination failed",
                    )
                    logger.debug("Failed to kill managed runtime process: %s", e)
        except (OSError, ConnectionError, TimeoutError, RuntimeError, AttributeError, TypeError) as e:
            self._record_degradation(
                e,
                action="continued critical cleanup after runtime process sweep failed",
            )
            logger.error("Failed to kill heavy processes: %s", e)

        # Integrated Adrenaline Surge: Signal distress to AffectEngine
        try:
            from core.container import ServiceContainer
            affect = ServiceContainer.get("affect", default=None)
            if affect and hasattr(affect, "react"):
                get_task_tracker().create_task(affect.react("critical_resource_exhaustion", {"intensity": 1.0}))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            self._record_degradation(
                e,
                severity="warning",
                action="continued critical cleanup without affect distress signal",
            )
            logger.debug("Failed to trigger adrenaline surcharge: %s", e)

        prune_result = await self._prune_memory()

        # Trigger explicit GC
        collected = gc.collect()

        # Signal metabolism engine to slow down
        try:
            from core.container import ServiceContainer
            metabolism = ServiceContainer.get("metabolic_monitor", default=None)
            if metabolism and hasattr(metabolism, "force_rest"):
                await metabolism.force_rest(duration=300)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            self._record_degradation(
                e,
                severity="warning",
                action="continued critical cleanup without metabolic rest signal",
            )
            capture_and_log(e, {'module': __name__})

        # v50 Hardening: Reclaim Metal Compiler Context immediately after purge
        try:
            from core.container import ServiceContainer
            root = ServiceContainer.get("platform_root", default=None)
            if root and hasattr(root, "force_compiler_wake"):
                logger.info("🌿 [MEMORY GOVERNOR] Reclaiming Metal context post-purge...")
                root.force_compiler_wake()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            self._record_degradation(
                e,
                severity="warning",
                action="completed critical cleanup without platform-root compiler pulse",
            )
            logger.error("Failed to pulse platform root: %s", e)
        self._remember_cleanup_event(
            "critical_cleanup",
            "ok",
            f"killed={killed} gc={collected} unload={unload_result} prune={prune_result}",
        )

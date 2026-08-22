import asyncio
import gc
import logging
import os
import sqlite3
import sys
import time
from typing import Any

from core.memory.physics import hawking_decay
from core.resilience.runaway_budget import RunawayPolicy, get_runaway_budget
from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation
from core.runtime.sqlite_support import connecting
from core.utils.exceptions import capture_and_log
from core.utils.memory_monitor import get_memory_pressure_snapshot, process_memory_bytes
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Resilience.MemoryGovernor")


def _mlx_registry_snapshot(module) -> dict:
    """The MLX client registry as an atomic dict, or {} when unavailable."""
    if module is None:
        return {}
    snapshot = getattr(module, "clients_snapshot", None)
    if callable(snapshot):
        try:
            return dict(snapshot())
        except (RuntimeError, TypeError, ValueError):
            return {}
    return {}


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
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
) + _PROCESS_INSPECTION_ERRORS
_PROCESS_TERMINATION_ERRORS = _PROCESS_INSPECTION_ERRORS + (OSError,)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


class MemoryGovernor:
    """Monitors system memory and enforces pruning/unloading thresholds.
    
    Daily-use thresholds are ordered rungs of the same host-derived envelope
    used by worker admission and the external watchdog.
    """
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.is_running = False
        self._task: asyncio.Task | None = None
        self._proc = psutil.Process(os.getpid())

        # Thresholds in MB. These intentionally sit below the out-of-band
        # MemoryWatchdog hard/lethal ladder so graceful cleanup runs first.
        try:
            total_mb = psutil.virtual_memory().total / (1024 * 1024)
        except _RSS_SAMPLE_ERRORS:
            total_mb = 65536.0
        try:
            from core.runtime.desktop_boot_safety import compute_desktop_memory_envelope

            envelope = compute_desktop_memory_envelope(int(total_mb * 1024 * 1024))
            prune_default = envelope.governor_prune_mb
            unload_default = envelope.governor_unload_mb
            critical_default = envelope.governor_critical_mb
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            prune_default = min(28672.0, total_mb * 0.44)
            unload_default = min(34816.0, total_mb * 0.53)
            critical_default = min(40960.0, total_mb * 0.62)
        self.threshold_prune = int(_env_float("AURA_GOVERNOR_PRUNE_MB", prune_default))
        self.threshold_unload = int(_env_float("AURA_GOVERNOR_UNLOAD_MB", unload_default))
        self.threshold_critical = int(_env_float("AURA_GOVERNOR_CRITICAL_MB", critical_default))

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
        self._last_policy_sample: dict[str, Any] = {}
        self._cleanup_events: list[dict[str, Any]] = []
        self._model_resource_state = "boot"
        self._model_allocation_identity: tuple[tuple[str, int, str, bool], ...] = ()
        self._model_settle_until = 0.0
        self._model_settling_s = max(
            0.0, _env_float("AURA_RUNAWAY_MODEL_SETTLING_S", 180.0)
        )
        self._trend_provisional = True

        # Watches the RSS *trend* and whether our own cleanup is achieving
        # anything. The thresholds above answer "how bad is it now"; this
        # answers "is it getting worse and is anything I do about it helping" —
        # the judgement a receipt-only system structurally cannot make.
        self._runaway = get_runaway_budget().detector(
            "managed_rss_mb", RunawayPolicy.for_memory_mb()
        )

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
            "resource_lifecycle": {
                "state": self._model_resource_state,
                "trend_provisional": self._trend_provisional,
                "settle_remaining_s": round(
                    max(0.0, self._model_settle_until - time.monotonic()), 3
                ),
                "allocation_identity": [list(item) for item in self._model_allocation_identity],
            },
            "recent_cleanup_events": list(self._cleanup_events[-10:]),
        }

    async def check(self) -> dict[str, Any]:
        """Run one bounded memory-policy check and return the current health snapshot."""
        await self._enforce_policy()
        return self.health_snapshot()

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

        # Every mitigation this governor performs passes through here, which
        # makes it the honest place to tell the runaway detector "I tried
        # something". Repeated attempts with RSS still climbing is precisely the
        # evidence that our cleanup does not address this particular growth —
        # and that is a hard failure, not another receipt.
        self._runaway.record_mitigation()

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

    @staticmethod
    def _registered_mlx_worker_pids() -> set[int]:
        """Return worker PIDs from the authoritative in-process client registry."""
        module = sys.modules.get("core.brain.llm.mlx_client")
        # Through the registry's own lock — copying it directly iterates a
        # dict another thread registers into, which raises mid-copy.
        clients = _mlx_registry_snapshot(module)
        if not isinstance(clients, dict):
            return set()
        pids: set[int] = set()
        for client in list(clients.values()):
            process = getattr(client, "_process", None)
            try:
                pid = int(getattr(process, "pid", 0) or 0)
                alive = bool(process is not None and process.is_alive())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                continue
            if pid > 0 and alive:
                pids.add(pid)
        return pids

    def _iter_managed_runtime_processes(self):
        """Yield heavyweight MLX runtime processes owned by Aura.

        Iterates the child tree directly instead of scanning the global
        process table — the full-table scan ran on the event loop and was
        a measurable lag source under memory pressure.
        """
        try:
            children = self._proc.children(recursive=True)
        except _RSS_SAMPLE_ERRORS as exc:
            self._record_degradation(
                exc,
                severity="warning",
                action="skipped managed-runtime RSS scan after child process inspection failed",
            )
            logger.debug("Memory Governor: could not inspect child process tree: %s", exc)
            return

        registered_pids = self._registered_mlx_worker_pids()
        seen: set[int] = set()
        for proc in children:
            try:
                info = proc.as_dict(attrs=['pid', 'name', 'cmdline', 'memory_info'])
                pid = int(info.get("pid", getattr(proc, "pid", 0)) or 0)
                if pid <= 0 or pid in seen:
                    continue
                cmd_str = " ".join(info.get('cmdline') or [])
                if pid in registered_pids or "mlx_worker.py" in cmd_str:
                    seen.add(pid)
                    proc.info = info  # match the process_iter contract callers rely on
                    yield proc
            except _RSS_SAMPLE_ERRORS:
                continue

    def _managed_runtime_rss_mb(self) -> float:
        total_mb = 0.0
        for proc in self._iter_managed_runtime_processes():
            try:
                mem_info = proc.info.get('memory_info')
                pid = int(proc.info.get("pid", getattr(proc, "pid", 0)) or 0)
                if pid > 0:
                    total_mb += process_memory_bytes(pid) / (1024 * 1024)
                else:
                    rss = (
                        getattr(mem_info, 'rss', 0)
                        if mem_info is not None
                        else proc.memory_info().rss
                    )
                    total_mb += rss / (1024 * 1024)
            except _PROCESS_INSPECTION_ERRORS:
                continue
        return total_mb

    @staticmethod
    def _model_lifecycle_snapshot() -> dict[str, Any]:
        """Read passive model lifecycle plus a stable resident-allocation identity."""
        try:
            from core.runtime.runtime_pressure import _model_resource_lifecycle_snapshot

            snapshot = dict(_model_resource_lifecycle_snapshot())
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            snapshot = {"state": "cold", "load_active": False}

        module = sys.modules.get("core.brain.llm.mlx_client")
        # Through the registry's own lock — copying it directly iterates a
        # dict another thread registers into, which raises mid-copy.
        clients = _mlx_registry_snapshot(module)
        identity: list[tuple[str, int, str, bool]] = []
        if isinstance(clients, dict):
            for path, client in list(clients.items()):
                process = getattr(client, "_process", None)
                try:
                    pid = int(getattr(process, "pid", 0) or 0)
                    alive = bool(process is not None and process.is_alive())
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pid = 0
                    alive = False
                if not alive:
                    continue
                identity.append(
                    (
                        str(path),
                        pid,
                        str(getattr(client, "_lane_state", "cold") or "cold").lower(),
                        bool(getattr(client, "_init_done", False)),
                    )
                )
        snapshot["allocation_identity"] = tuple(sorted(identity))
        return snapshot

    def _update_runaway_lifecycle(self, now: float) -> tuple[str, bool]:
        snapshot = self._model_lifecycle_snapshot()
        reported = str(snapshot.get("state") or "cold").lower()
        identity = tuple(snapshot.get("allocation_identity") or ())
        previous_state = self._model_resource_state
        boundary = reported != previous_state or identity != self._model_allocation_identity

        if boundary:
            self._runaway.reset()
            self._model_resource_state = reported
            self._model_allocation_identity = identity
            if reported == "model_loading":
                self._model_settle_until = 0.0
            elif previous_state == "model_loading" or identity:
                self._model_settle_until = now + self._model_settling_s
            else:
                self._model_settle_until = 0.0
            logger.info(
                "Memory Governor resource lifecycle %s → %s; trend baseline reset.",
                previous_state,
                reported,
            )

        if reported == "model_loading":
            state = "model_loading"
            provisional = True
        elif now < self._model_settle_until:
            state = "settling"
            provisional = True
        else:
            state = reported
            provisional = False

        if provisional:
            # Keep allocation ramps out of the next steady-state epoch. Absolute
            # critical cleanup below remains active on the current measurement.
            self._runaway.reset()
        self._trend_provisional = provisional
        return state, provisional

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
            await self._critical_cleanup(reason="shutdown")
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

    def _sample_rss_sync(self) -> tuple[float, float]:
        """Sample the canonical process tree exactly once, off the event loop."""
        try:
            rss_mb = process_memory_bytes(self._proc.pid) / (1024 * 1024)
        except _RSS_SAMPLE_ERRORS as exc:
            self._record_degradation(
                exc,
                severity="warning",
                action="continued memory policy with zero core RSS after process sample failed",
            )
            rss_mb = 0.0
        try:
            snapshot = get_memory_pressure_snapshot(force_refresh=True)
            process_tree_mb = max(
                rss_mb,
                float(snapshot.process_rss_gb) * 1024.0,
            )
        except _RSS_SAMPLE_ERRORS as exc:
            self._record_degradation(
                exc,
                severity="warning",
                action="continued memory policy with core-only RSS after canonical tree sample failed",
            )
            process_tree_mb = rss_mb
        # The canonical tree already includes the root. Runtime is the residual,
        # not another tree total to add on top of it.
        return rss_mb, max(0.0, process_tree_mb - rss_mb)

    async def _enforce_policy(self):
        """Check RSS memory and system-wide RAM to trigger cleanup actions."""
        now = time.monotonic()
        # 1. Check Process Memory (sampled off-loop: child-tree RSS walks
        # hit the disk/kernel hard during swap pressure and must never
        # block the event loop).
        rss_mb, runtime_rss_mb = await asyncio.to_thread(self._sample_rss_sync)
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

        lifecycle_state, trend_provisional = self._update_runaway_lifecycle(now)
        self._last_policy_sample["resource_lifecycle"] = lifecycle_state
        self._last_policy_sample["trend_provisional"] = trend_provisional

        # Trend, not just level. The thresholds below are level-triggered: at
        # the ~242MB/h growth the 4h soak measured, the 28GB prune trigger is
        # DAYS away — the trend is unmistakable the whole time and nothing looks
        # at it. Worse, when pruning fires and RSS keeps climbing, this loop
        # just prunes again forever, emitting a receipt each time saying it
        # handled things. This is what turns "the mitigation is not working"
        # into an actual hard failure instead of a nicer log line.
        if trend_provisional:
            verdict = self._runaway.assess()
            runaway_status = verdict.to_dict()
            runaway_status.update(
                {
                    "state": "provisional",
                    "reason": f"resource_lifecycle:{lifecycle_state}",
                }
            )
        else:
            self._runaway.observe(managed_rss_mb)
            verdict = self._runaway.assess()
            runaway_status = verdict.to_dict()
        self._last_policy_sample["runaway"] = runaway_status

        if sys_percent > 98.0 or managed_rss_mb > self.threshold_critical:
            # The managed-RSS condition must live at this level: the old
            # structure only reached the critical cleanup when system-wide
            # RAM crossed 98%, which macOS memory compression can prevent
            # even while this process tree balloons toward host freeze.
            logger.critical(
                "🚨 CRITICAL MEMORY: system %.1f%%, managed RSS %.0fMB "
                "(threshold %.0fMB). Checking for idle MLX runtime workers.",
                sys_percent,
                managed_rss_mb,
                float(self.threshold_critical),
            )

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

            if (now - self._last_critical_action_time) >= self.critical_cooldown_s:
                logger.critical(
                    "🚨 EMERGENCY: memory remains critical (system %.1f%%, managed %.0fMB). Triggering FULL cleanup.",
                    sys_percent,
                    managed_rss_mb,
                )
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
                            with connecting(sqlite3.connect(episodic.db_path)) as conn:
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
            "mlx_runtime_lanes_rebooted": 0,
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

    async def _critical_cleanup(self, *, reason: str = "pressure"):
        """Maximum effort cleanup."""
        if reason == "shutdown":
            logger.info("Releasing heavy MLX runtime workers during orderly shutdown.")
        else:
            logger.critical("🚨 NEURAL PURGE: Killing heavy MLX runtime workers to recover system RAM.")
        
        unload_result = await self._unload_models()

        # v7.2.3: Force kill heavy runtime processes if they are hanging/heavy
        killed = 0
        try:
            for proc in self._iter_managed_runtime_processes():
                try:
                    cmdline = proc.info.get('cmdline') or []
                    cmd_str = " ".join(cmdline)
                    if "mlx_worker.py" in cmd_str or "MTLCompilerService" in cmd_str:
                        log = logger.info if reason == "shutdown" else logger.warning
                        log(
                            "%s heavy MLX/Metal process (PID: %d, Name: %s)",
                            "Stopping" if reason == "shutdown" else "🚨 NEURAL PURGE: Forcibly terminating",
                            proc.info["pid"],
                            proc.info["name"],
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
        if reason != "shutdown":
            try:
                from core.container import ServiceContainer
                affect = ServiceContainer.get("affect", default=None)
                if affect and hasattr(affect, "react"):
                    get_task_tracker().create_task(
                        affect.react(
                            "critical_resource_exhaustion",
                            {
                                "source": "memory_governor",
                                "intensity": 1.0,
                                "evidence": {"kind": "resource_governor", "reason": reason},
                            },
                        ),
                        name="memory_governor.affect_resource_exhaustion",
                    )
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

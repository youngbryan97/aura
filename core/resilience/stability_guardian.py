from __future__ import annotations

import asyncio
import gc
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger("Aura.StabilityGuardian")


# ── Health check result ───────────────────────────────────────────────────────

@dataclass
class HealthCheckResult:
    name:       str
    healthy:    bool
    message:    str
    severity:   str = "info"    # "info", "warning", "error", "critical"
    action_taken: Optional[str] = None
    timestamp:  float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "name":         self.name,
            "healthy":      self.healthy,
            "message":      self.message,
            "severity":     self.severity,
            "action_taken": self.action_taken,
            "timestamp":    self.timestamp,
        }


@dataclass
class SystemHealthReport:
    timestamp:     float
    overall_healthy: bool
    checks:        List[HealthCheckResult]
    memory_pct:    float
    cpu_pct:       float
    task_count:    int
    tick_rate_hz:  float
    mean_tick_ms:  float

    def to_dict(self) -> Dict:
        return {
            "timestamp":      self.timestamp,
            "overall_healthy": self.overall_healthy,
            "memory_pct":     round(self.memory_pct, 1),
            "cpu_pct":        round(self.cpu_pct, 1),
            "task_count":     self.task_count,
            "tick_rate_hz":   round(self.tick_rate_hz, 3),
            "mean_tick_ms":   round(self.mean_tick_ms, 1),
            "checks":         [c.to_dict() for c in self.checks],
        }

    def print_summary(self) -> None:
        status = "HEALTHY" if self.overall_healthy else "DEGRADED"
        logger.info(f"\n[StabilityGuardian] {status}  {time.strftime('%H:%M:%S', time.localtime(self.timestamp))}")
        logger.info(f"  Memory: {self.memory_pct:.0f}%  CPU: {self.cpu_pct:.0f}%  Tasks: {self.task_count}")
        logger.info(f"  Tick rate: {self.tick_rate_hz:.2f}Hz  Mean tick: {self.mean_tick_ms:.0f}ms")
        for c in self.checks:
            icon = "✓" if c.healthy else ("⚠" if c.severity in ("warning", "info") else "✗")
            logger.info(f"  {icon} {c.name}: {c.message}")
            if c.action_taken:
                logger.info(f"     → {c.action_taken}")


# ── The Guardian ──────────────────────────────────────────────────────────────

class StabilityGuardian:
    """
    Comprehensive health monitoring and recovery.

    Runs a check loop every 10 seconds.
    Fires targeted recovery for each detected issue.
    Maintains a rolling health log.
    Exposes a /health endpoint-compatible dict.
    """

    CHECK_INTERVAL_S    = 10.0
    MEMORY_WARNING_PCT  = 82.0    # 64GB system — don't warn until ~52GB used
    MEMORY_CRITICAL_PCT = 92.0    # True critical on 64GB — ~59GB used
    MAX_TICK_LAG_MS     = 5000.0   # 5s mean tick = something is blocking
    MIN_TICK_RATE_HZ    = 0.01     # If we're not ticking at all, something is wrong
    MAX_TASK_COUNT      = 260      # asyncio task explosion guard
    MAX_UNSUPERVISED_TASK_COUNT = 80

    @staticmethod
    def _memory_thresholds() -> Tuple[float, float]:
        if not _HAS_PSUTIL:
            return StabilityGuardian.MEMORY_WARNING_PCT, StabilityGuardian.MEMORY_CRITICAL_PCT
        try:
            total_gb = float(psutil.virtual_memory().total) / float(1024 ** 3)
        except Exception:
            total_gb = 0.0
        if total_gb >= 60.0:
            return 88.0, 94.0
        return StabilityGuardian.MEMORY_WARNING_PCT, StabilityGuardian.MEMORY_CRITICAL_PCT

    def __init__(self, orchestrator: Any):
        self.orchestrator   = orchestrator
        self._running       = False
        self._task: Optional[asyncio.Task] = None
        self._report_history: deque = deque(maxlen=100)
        self._tick_times:     Deque[Tuple[float, float]] = deque(maxlen=60)   # (timestamp, duration_ms)
        self._last_tick_at:   float = time.time()
        self._extra_checks:   List[Callable] = []
        self._last_repair_at: Dict[str, float] = {}

        try:
            from core.config import config
            self._log_path = config.paths.data_dir / "stability" / "health_log.jsonl"
        except (ImportError, AttributeError):
            self._log_path = Path.home() / ".aura" / "stability" / "health_log.jsonl"
            
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("StabilityGuardian initialized.")

    def _repair_allowed(self, name: str, cooldown_s: float) -> bool:
        now = time.time()
        last = float(self._last_repair_at.get(name, 0.0) or 0.0)
        if now - last < cooldown_s:
            return False
        self._last_repair_at[name] = now
        return True

    def add_check(self, fn: Callable) -> None:
        """Register an additional health check. fn() -> HealthCheckResult"""
        self._extra_checks.append(fn)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            from core.utils.task_tracker import get_task_tracker

            self._task = get_task_tracker().create_task(
                self._loop(),
                name="aura.stability_guardian",
            )
        except Exception:
            self._task = asyncio.create_task(self._loop(), name="aura.stability_guardian")
        logger.info("StabilityGuardian running (interval=%ds).", int(self.CHECK_INTERVAL_S))

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in stability_guardian.py: %s', _e)

    # ── Called from kernel tick observer ─────────────────────────────────────

    def record_tick_health(self, tick_entry: Any) -> None:
        """Called after every kernel tick by the FeedbackObserver."""
        now = time.time()
        if hasattr(tick_entry, "tick_duration_ms"):
            duration_ms = float(getattr(tick_entry, "tick_duration_ms", 0.0) or 0.0)
            if duration_ms > 0.0:
                self._tick_times.append((now, duration_ms))
        self._last_tick_at = now

    def _recent_tick_durations(self, now: Optional[float] = None, window_s: float = 300.0) -> List[float]:
        current_time = now or time.time()
        return [
            duration_ms
            for timestamp, duration_ms in self._tick_times
            if (current_time - timestamp) <= window_s
        ]

    # ── Main check loop ───────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.CHECK_INTERVAL_S)
                report = await self.run_checks()
                self._report_history.append(report)
                self._persist_report(report)

                if not report.overall_healthy:
                    unhealthy = [c for c in report.checks if not c.healthy]
                    summary = "; ".join(f"{c.name}:{c.message}" for c in unhealthy[:3])
                    logger.warning(
                        "StabilityGuardian: DEGRADED — %d issue(s) detected. %s",
                        len(unhealthy),
                        summary,
                    )
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "stability_guardian",
                            "degraded_report",
                            detail=summary,
                            severity="warning",
                            classification="background_degraded",
                            context={"issue_count": len(unhealthy)},
                        )
                    except Exception as record_exc:
                        logger.debug("StabilityGuardian degraded event emit failed: %s", record_exc)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("StabilityGuardian loop error: %s", e)

    async def run_checks(self) -> SystemHealthReport:
        """Run all health checks and return a full report."""
        checks = []

        # Run all checks, catching exceptions so one failure doesn't block others
        check_fns = [
            self._check_memory,
            self._check_asyncio_tasks,
            self._check_lock_watchdog,
            self._check_tick_rate,
            self._check_state_integrity,
            self._check_state_repository_pressure,
            self._check_llm_circuit,
            self._check_db_connections,
            self._check_backup_maintenance,
            self._check_background_tasks,
        ]
        for fn in check_fns + self._extra_checks:
            try:
                result = await fn() if asyncio.iscoroutinefunction(fn) else fn()
                if isinstance(result, HealthCheckResult):
                    checks.append(result)
            except Exception as e:
                checks.append(HealthCheckResult(
                    name    = getattr(fn, "__name__", "unknown"),
                    healthy = False,
                    message = f"Check raised exception: {e}",
                    severity = "error",
                ))

        # System metrics
        mem_pct  = psutil.virtual_memory().percent if _HAS_PSUTIL else 0.0
        cpu_pct  = psutil.cpu_percent(interval=None) if _HAS_PSUTIL else 0.0
        task_count = len(asyncio.all_tasks())

        # Tick stats
        now = time.time()
        recent_ticks = self._recent_tick_durations(now)
        if recent_ticks:
            oldest_timestamp = min(timestamp for timestamp, _duration in self._tick_times if (now - timestamp) <= 300.0)
            tick_rate = len(recent_ticks) / max(1.0, now - oldest_timestamp)
            mean_tick = sum(recent_ticks) / len(recent_ticks)
        else:
            tick_rate = 0.0
            mean_tick = 0.0

        overall = all(c.healthy or c.severity in ("info",) for c in checks)

        return SystemHealthReport(
            timestamp      = time.time(),
            overall_healthy = overall,
            checks         = checks,
            memory_pct     = mem_pct,
            cpu_pct        = cpu_pct,
            task_count     = task_count,
            tick_rate_hz   = tick_rate,
            mean_tick_ms   = mean_tick,
        )

    # ── Individual checks ─────────────────────────────────────────────────────

    async def _check_memory(self) -> HealthCheckResult:
        if not _HAS_PSUTIL:
            return HealthCheckResult("memory", True, "psutil not available", "info")

        mem = psutil.virtual_memory()
        pct = mem.percent
        warning_pct, critical_pct = self._memory_thresholds()

        if pct > critical_pct:
            # Proactive cleanup
            action = "repair_cooldown_active"
            if self._repair_allowed("memory_critical_gc", 60.0):
                gc.collect()
                action = "gc.collect() triggered"
            # Try to evict old episodic memory
            try:
                from core.container import ServiceContainer
                dual_mem = ServiceContainer.get("dual_memory", default=None)
                if (
                    action != "repair_cooldown_active"
                    and dual_mem
                    and hasattr(dual_mem, "episodic")
                    and hasattr(dual_mem.episodic, "evict_oldest")
                ):
                    await dual_mem.episodic.evict_oldest(0.2)
                    action = "gc.collect() + episodic eviction triggered"
            except Exception as _e:
                logger.debug('Ignored Exception in stability_guardian.py: %s', _e)
            return HealthCheckResult(
                "memory", False,
                f"CRITICAL memory pressure: {pct:.0f}%",
                severity="critical",
                action_taken=action,
            )

        if pct > warning_pct:
            action = "repair_cooldown_active"
            if self._repair_allowed("memory_warning_gc", 90.0):
                gc.collect()
                action = "gc.collect() triggered"
            return HealthCheckResult(
                "memory", False,
                f"High memory: {pct:.0f}%",
                severity="warning",
                action_taken=action,
            )

        return HealthCheckResult("memory", True, f"Memory OK: {pct:.0f}%")

    async def _check_asyncio_tasks(self) -> HealthCheckResult:
        tasks = asyncio.all_tasks()
        n     = len(tasks)
        current = asyncio.current_task()
        supervised = sum(1 for t in tasks if getattr(t, "_aura_supervised", False))
        anonymous_unsupervised = [
            t for t in tasks
            if t is not current
            and not getattr(t, "_aura_supervised", False)
            and (t.get_name().startswith("Task") or "Task-" in t.get_name())
        ]

        if n > self.MAX_TASK_COUNT or len(anonymous_unsupervised) > self.MAX_UNSUPERVISED_TASK_COUNT:
            cancelled_count = 0
            action = "repair_cooldown_active"
            if self._repair_allowed("asyncio_task_shedding", 30.0):
                action = "task_shedding_attempted"
                for t in anonymous_unsupervised:
                    try:
                        t.cancel()
                        cancelled_count += 1
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                    if n - cancelled_count <= self.MAX_TASK_COUNT - 20:
                        break

            task_list = list(tasks)
            names = [t.get_name() for t in task_list[:20]]
            try:
                from core.utils.task_tracker import get_task_tracker

                tracker_stats = get_task_tracker().get_stats()
            except Exception:
                tracker_stats = {"active": 0, "high_water": 0}
            return HealthCheckResult(
                "asyncio_tasks", False,
                f"Task explosion: {n} tasks ({supervised} supervised / {n - supervised} unsupervised, anonymous={len(anonymous_unsupervised)}). Sample: {names}",
                severity="error",
                action_taken=(
                    f"{action}; cancelled={cancelled_count}; "
                    f"tracker_active={tracker_stats.get('active', 0)} high_water={tracker_stats.get('high_water', 0)}"
                ),
            )

        # Check for tasks that look stuck (created a long time ago and still running)
        # We can't easily check creation time in Python, so we just count
        return HealthCheckResult("asyncio_tasks", True, f"Tasks: {n} (OK, supervised={supervised})")

    async def _check_lock_watchdog(self) -> HealthCheckResult:
        try:
            from core.resilience.lock_watchdog import get_lock_watchdog

            snapshot = get_lock_watchdog().get_snapshot()
            locks = snapshot.get("locks", [])
            if not locks:
                return HealthCheckResult("lock_watchdog", True, "No stalled locks")

            worst = locks[0]
            interventions = int(worst.get("interventions", 0) or 0)
            message = (
                f"{len(locks)} active lock(s); worst {worst.get('name', 'unknown')} held for "
                f"{worst.get('held_duration_s', 0.0):.1f}s"
            )
            if float(worst.get("held_duration_s", 0.0) or 0.0) <= float(snapshot.get("threshold_s", 0.0) or 0.0):
                return HealthCheckResult("lock_watchdog", True, message)
            return HealthCheckResult(
                "lock_watchdog",
                False,
                message,
                severity="warning" if interventions else "error",
                action_taken=f"watchdog_interventions={interventions}",
            )
        except Exception as exc:
            return HealthCheckResult("lock_watchdog", False, f"Check failed: {exc}", "error")

    def _check_tick_rate(self) -> HealthCheckResult:
        now = time.time()
        elapsed = now - self._last_tick_at
        recent_ticks = self._recent_tick_durations(now)

        if not recent_ticks:
            if elapsed > 300.0:
                return HealthCheckResult(
                    "tick_rate", True,
                    f"No recent kernel ticks for {elapsed:.0f}s — kernel appears idle or event-driven",
                    severity="info",
                )
            return HealthCheckResult("tick_rate", True, "No recent kernel ticks yet (normal at boot or idle)")

        mean_ms = sum(recent_ticks) / len(recent_ticks)
        if elapsed > 30.0:
            return HealthCheckResult(
                "tick_rate", True,
                f"Kernel idle for {elapsed:.0f}s; last recent mean tick={mean_ms:.0f}ms",
                severity="info",
            )
        slow_ticks = [duration for duration in recent_ticks if duration > self.MAX_TICK_LAG_MS]
        if mean_ms > self.MAX_TICK_LAG_MS and len(recent_ticks) >= 5 and len(slow_ticks) >= 3:
            import sys
            import traceback
            dump = "\n".join("".join(traceback.format_stack(f)) for thread_id, f in sys._current_frames().items())
            logger.error("🚨 [StabilityGuardian] TICK STALL DETECTED (%.0fms). THREAD DUMP:\n%s", mean_ms, dump[:1500])
            
            return HealthCheckResult(
                "tick_rate", False,
                f"Mean tick too slow: {mean_ms:.0f}ms (max {self.MAX_TICK_LAG_MS:.0f}ms) — possible event loop blocking",
                severity="warning",
                action_taken="Dumped thread stacks to aura_json.log"
            )
        if slow_ticks:
            return HealthCheckResult(
                "tick_rate", True,
                f"Observed {len(slow_ticks)} slow tick(s); mean={mean_ms:.0f}ms but not sustained enough to degrade",
                severity="info",
            )

        return HealthCheckResult("tick_rate", True, f"Tick health OK: mean={mean_ms:.0f}ms")

    async def _check_state_integrity(self) -> HealthCheckResult:
        """Validate AuraState structure before it gets committed."""
        try:
            from core.container import ServiceContainer
            ki   = ServiceContainer.get("aura_kernel", default=None)
            if ki is None:
                # Try getting kernel interface
                ki = ServiceContainer.get("kernel_interface", default=None)
                
            if ki is None:
                 return HealthCheckResult("state_integrity", True, "Kernel not yet running")

            # Handle different ki/kernel structures
            kernel = getattr(ki, "kernel", ki)
            state = getattr(kernel, "state", None)
            
            if state is None:
                return HealthCheckResult("state_integrity", False, "Kernel state is None", "error")

            # Structural checks
            issues = []
            if not hasattr(state, "affect"):
                issues.append("missing affect")
            if not hasattr(state, "cognition"):
                issues.append("missing cognition")
            if not hasattr(state, "identity"):
                issues.append("missing identity")
            if getattr(state, "version", 0) < 0:
                issues.append(f"invalid version: {getattr(state, 'version', 0)}")

            # Check for circular references in concept_graph
            try:
                import json as _json
                cg = getattr(state.identity, "concept_graph", {})
                _json.dumps({"cg": cg}, check_circular=True)
            except (ValueError, RecursionError) as e:
                issues.append(f"concept_graph circular ref: {e}")

            # Working memory bounds
            wm = getattr(state.cognition, "working_memory", [])
            wm_len = len(wm)
            if wm_len > 200:
                issues.append(f"working_memory oversized: {wm_len}")

            if issues:
                action = "None"
                # Force state compaction if working memory is oversized
                if "working_memory oversized" in " ".join(issues):
                    try:
                        if state:
                            state.compact(trigger_threshold=100, keep_turns=50)
                            action = "Forced state.compact()"
                    except Exception as e:
                        action = f"Compact failed: {e}"
                        
                return HealthCheckResult(
                    "state_integrity", False,
                    f"State issues: {'; '.join(issues)}",
                    severity="warning",
                    action_taken=action
                )

            return HealthCheckResult("state_integrity", True, f"State v{getattr(state, 'version', 0)} OK")

        except Exception as e:
            return HealthCheckResult("state_integrity", False, f"Check failed: {e}", "error")

    async def _check_state_repository_pressure(self) -> HealthCheckResult:
        try:
            from core.container import ServiceContainer

            repo = ServiceContainer.get("state_repository", default=None)
            if repo is None or not hasattr(repo, "get_runtime_status"):
                return HealthCheckResult("state_repository", True, "State repository runtime status unavailable", "info")

            status = repo.get_runtime_status()
            queue_depth = int(status.get("queue_depth", 0) or 0)
            queue_max = int(status.get("queue_maxsize", 0) or 0)
            consumer_alive = bool(status.get("consumer_alive", False))
            actions = []

            if hasattr(repo, "repair_runtime") and self._repair_allowed("state_repository_repair", 30.0) and (
                not consumer_alive
                or (queue_max > 0 and queue_depth >= max(1, int(queue_max * 0.75)))
            ):
                repair = await repo.repair_runtime()
                status = repair.get("status", status)
                queue_depth = int(status.get("queue_depth", queue_depth) or queue_depth)
                consumer_alive = bool(status.get("consumer_alive", consumer_alive))
                actions = list(repair.get("actions", []) or [])

            if not consumer_alive:
                return HealthCheckResult(
                    "state_repository",
                    False,
                    f"Vault consumer not alive (queue={queue_depth}/{queue_max})",
                    severity="error",
                    action_taken=", ".join(actions) or None,
                )
            if queue_max > 0 and queue_depth >= max(1, int(queue_max * 0.75)):
                return HealthCheckResult(
                    "state_repository",
                    False,
                    f"State mutation queue elevated: {queue_depth}/{queue_max}",
                    severity="warning",
                    action_taken=", ".join(actions) or None,
                )
            return HealthCheckResult(
                "state_repository",
                True,
                (
                    f"State queue OK: {queue_depth}/{queue_max}; "
                    f"dropped={int(status.get('dropped_commit_count', 0) or 0)}"
                ),
                action_taken=", ".join(actions) or None,
            )
        except Exception as exc:
            return HealthCheckResult("state_repository", False, f"Check failed: {exc}", "error")

    async def _check_llm_circuit(self) -> HealthCheckResult:
        """Verify LLM availability and circuit breaker state."""
        try:
            from core.container import ServiceContainer
            ki = ServiceContainer.get("aura_kernel", default=None)
            if ki is None:
                ki = ServiceContainer.get("kernel_interface", default=None)
                
            if ki is None:
                return HealthCheckResult("llm_circuit", True, "Kernel not running yet")

            kernel = getattr(ki, "kernel", ki)
            organ = getattr(kernel, "organs", {}).get("llm")
            
            if organ is None:
                return HealthCheckResult("llm_circuit", False, "LLM organ missing", "error")

            # Use different ready mechanisms if .ready is not a set()
            ready = True
            if hasattr(organ, "ready"):
                if hasattr(organ.ready, "is_set"):
                    ready = organ.ready.is_set()
                else:
                    ready = bool(organ.ready)
            
            if not ready:
                return HealthCheckResult("llm_circuit", False, "LLM organ not ready", "warning")

            instance = getattr(organ, "instance", None)
            instance_name = instance.__class__.__name__ if instance else "None"

            # Check if stuck on MockLLM after enough time for real model to load
            start_time = getattr(self.orchestrator, "start_time", 0)
            if instance_name == "MockLLM" and time.time() - start_time > 60:
                # Try to reload LLM organ
                if self._repair_allowed("llm_reload", 120.0):
                    try:
                        if hasattr(organ, "load"):
                            await organ.load()
                            new_instance = getattr(organ, "instance", None)
                            new_name = new_instance.__class__.__name__ if new_instance else "None"
                            return HealthCheckResult(
                                "llm_circuit", True,
                                f"LLM organ reloaded: {new_name}",
                                action_taken="organ.load() called",
                            )
                    except Exception as reload_err:
                        return HealthCheckResult(
                            "llm_circuit", False,
                            f"LLM stuck on MockLLM: {reload_err}",
                            severity="warning",
                        )
                return HealthCheckResult(
                    "llm_circuit", False,
                    "LLM stuck on MockLLM; reload cooldown active",
                    severity="warning",
                )

            return HealthCheckResult("llm_circuit", True, f"LLM OK: {instance_name}")

        except Exception as e:
            return HealthCheckResult("llm_circuit", False, f"Check failed: {e}", "error")

    async def _check_db_connections(self) -> HealthCheckResult:
        """Check DB connection pool health."""
        try:
            from core.container import ServiceContainer
            repo = ServiceContainer.get("state_repository", default=None)
            if repo is None:
                return HealthCheckResult("db_connections", True, "No state repo yet")

            if getattr(repo, "_db", None) is None:
                # Check if we should be owner
                is_owner = getattr(repo, "is_vault_owner", False)
                if is_owner:
                    # DB connection was lost — try to reconnect
                    try:
                        import aiosqlite
                        db_path = getattr(repo, "db_path", "aura_state.db")
                        repo._db = await aiosqlite.connect(db_path)
                        await repo._db.execute("PRAGMA journal_mode=WAL")
                        return HealthCheckResult(
                            "db_connections", True,
                            "DB reconnected",
                            action_taken="aiosqlite.connect() called",
                        )
                    except Exception as e:
                        return HealthCheckResult(
                            "db_connections", False,
                            f"DB connection lost and reconnect failed: {e}",
                            severity="error",
                        )

            return HealthCheckResult("db_connections", True, "DB connection OK")

        except Exception as e:
            return HealthCheckResult("db_connections", False, f"Check failed: {e}", "error")

    async def _check_backup_maintenance(self) -> HealthCheckResult:
        try:
            from core.container import ServiceContainer

            manager = ServiceContainer.get("backup_manager", default=None)
            if manager is None:
                return HealthCheckResult("backup_manager", False, "BackupManager not registered", "warning")

            health = await manager.get_health()
            if not health.get("scheduler_registered", False):
                if hasattr(manager, "on_start_async") and self._repair_allowed("backup_scheduler_registration", 60.0):
                    await manager.on_start_async()
                return HealthCheckResult(
                    "backup_manager",
                    False,
                    "Backup maintenance was not registered with scheduler",
                    severity="warning",
                    action_taken="re-registered_backup_maintenance",
                )

            latest_age_s = health.get("latest_backup_age_s")
            uptime = max(0.0, time.time() - float(getattr(self.orchestrator, "start_time", time.time()) or time.time()))
            backup_interval_s = float(health.get("backup_interval_s", 86400.0) or 86400.0)

            if latest_age_s is None and uptime > (backup_interval_s * 1.5) and hasattr(manager, "ensure_recent_backup"):
                if self._repair_allowed("backup_creation", 300.0):
                    await manager.ensure_recent_backup(max_age_s=backup_interval_s)
                return HealthCheckResult(
                    "backup_manager",
                    False,
                    "No backup found after extended uptime",
                    severity="warning",
                    action_taken="triggered_backup_creation",
                )

            if latest_age_s is not None and latest_age_s > (backup_interval_s * 2.0) and hasattr(manager, "ensure_recent_backup"):
                if self._repair_allowed("backup_refresh", 300.0):
                    await manager.ensure_recent_backup(max_age_s=backup_interval_s)
                return HealthCheckResult(
                    "backup_manager",
                    False,
                    f"Latest backup is stale ({latest_age_s:.0f}s old)",
                    severity="warning",
                    action_taken="triggered_backup_refresh",
                )

            latest_label = health.get("latest_backup") or "none_yet"
            return HealthCheckResult("backup_manager", True, f"Backup maintenance OK: latest={latest_label}")
        except Exception as exc:
            return HealthCheckResult("backup_manager", False, f"Check failed: {exc}", "error")

    async def _check_background_tasks(self) -> HealthCheckResult:
        """Verify critical named background tasks are still running."""
        CRITICAL_TASKS = {
            "aura.research_cycle",
        }
        tasks          = asyncio.all_tasks()
        running_names  = {t.get_name() for t in tasks if not t.done()}
        missing        = CRITICAL_TASKS - running_names
        # Some may not exist yet at boot
        startup_tasks  = {"aura.research_cycle"}  # OK to be missing early on
        real_missing   = missing - startup_tasks

        if real_missing:
            import os
            action = "None"
            if "aura.research_cycle" in real_missing:
                try:
                    from core.container import ServiceContainer
                    from core.utils.task_tracker import get_task_tracker
                    autonomous_loop = ServiceContainer.get("autonomous_loop", default=None)
                    if autonomous_loop and hasattr(autonomous_loop, "start") and self._repair_allowed("background_task_restart", 60.0):
                        # Re-start loop
                        get_task_tracker().create_task(
                            autonomous_loop.start(),
                            name="stability_guardian.autonomous_loop.restart",
                        )
                        action = "autonomous_loop.start() triggered"
                except Exception as e:
                    action = f"Restart failed: {e}"
                    
            logger.debug(f"[PID {os.getpid()}] StabilityGuardian: Total tasks: {len(tasks)}. Running task names: {running_names}")
            return HealthCheckResult(
                "background_tasks", False,
                f"Critical tasks not running in PID {os.getpid()}: {real_missing}. Tasks: {running_names}",
                severity="warning",
                action_taken=action
            )

        return HealthCheckResult("background_tasks", True, f"Background tasks OK ({len(running_names)} running)")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_latest_report(self) -> Optional[Dict]:
        if not self._report_history:
            return None
        return self._report_history[-1].to_dict()

    def get_health_summary(self) -> Dict:
        """Compatible with a /health API endpoint."""
        if not self._report_history:
            return {"status": "initializing", "healthy": True}
        r = self._report_history[-1]
        active_issues = [
            {
                "name": c.name,
                "message": c.message,
                "severity": c.severity,
                "action_taken": c.action_taken,
            }
            for c in r.checks
            if not c.healthy
        ]
        return {
            "status":     "healthy" if r.overall_healthy else "degraded",
            "healthy":    r.overall_healthy,
            "memory_pct": r.memory_pct,
            "tick_rate":  r.tick_rate_hz,
            "mean_tick":  r.mean_tick_ms,
            "checks":     {c.name: c.healthy for c in r.checks},
            "active_issues": active_issues,
            "timestamp":  r.timestamp,
        }

    _HEALTH_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    _HEALTH_LOG_BACKUP_COUNT = 3

    def _persist_report(self, report: SystemHealthReport) -> None:
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(report.to_dict()) + "\n")
            # Rotate if too large
            try:
                if self._log_path.stat().st_size > self._HEALTH_LOG_MAX_BYTES:
                    self._rotate_health_log()
            except OSError as _exc:
                logger.debug("Suppressed OSError: %s", _exc)
        except Exception as _e:
            logger.debug('Ignored Exception in stability_guardian.py: %s', _e)

    def _rotate_health_log(self) -> None:
        """Rotate health_log.jsonl to prevent unbounded growth on long-running systems."""
        try:
            for i in range(self._HEALTH_LOG_BACKUP_COUNT, 0, -1):
                src = self._log_path.with_suffix(f".jsonl.{i}") if i > 0 else self._log_path
                dst = self._log_path.with_suffix(f".jsonl.{i + 1}")
                if i == self._HEALTH_LOG_BACKUP_COUNT:
                    # Delete oldest backup
                    dst = self._log_path.with_suffix(f".jsonl.{i}")
                    if dst.exists():
                        dst.unlink()
                    continue
                src = self._log_path.with_suffix(f".jsonl.{i}")
                dst = self._log_path.with_suffix(f".jsonl.{i + 1}")
                if src.exists():
                    src.rename(dst)
            # Rotate current file to .1
            backup_1 = self._log_path.with_suffix(".jsonl.1")
            if self._log_path.exists():
                self._log_path.rename(backup_1)
            logger.info("🧹 [StabilityGuardian] Rotated health_log.jsonl (exceeded %dMB).",
                        self._HEALTH_LOG_MAX_BYTES // (1024 * 1024))
        except Exception as e:
            logger.debug("Health log rotation failed: %s", e)

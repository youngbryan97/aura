"""Metabolic Coordinator — background tasks, pacing, memory hygiene, world decay,
autonomous thought triggers, RL training, and self-update.

Extracted from orchestrator.py as part of the orchestrator ownership split.
"""
import asyncio
import gc
import json
import logging
import math
import os
import time
from collections import deque

from core.config import config
from core.container import ServiceContainer
from core.executive.standing_authority import (
    BACKGROUND_REFLECTION_MAX_ACTIONS,
    BACKGROUND_REFLECTION_WINDOW_SECONDS,
)
from core.memory.retention_policy import working_history_retention_policy
from core.runtime.background_policy import (
    IDLE_COGNITION_BACKGROUND_POLICY,
    background_activity_reason,
)
from core.runtime.errors import record_degradation
from core.runtime.impulse_governance import run_governed_impulse
from core.runtime.safe_mode import runtime_feature_enabled, runtime_mode_value
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger(__name__)

_METABOLIC_SUBSYSTEM = "metabolic_coordinator"
_METABOLIC_BOUNDARY_ERRORS = (
    AttributeError, ImportError, LookupError, OSError,
    RuntimeError, TimeoutError, TypeError, ValueError,
    asyncio.InvalidStateError,
)
_BOOT_WARMUP_CYCLES = 5
_BCI_EVENT_POLL_SECONDS = 1.0
_AUTONOMOUS_REFLECTION_TIMEOUT_SECONDS = 120.0
_AUTONOMOUS_REFLECTION_INTERVAL_SECONDS = 1800.0
_AUTONOMOUS_REFLECTION_FAILURE_BACKOFF_SECONDS = 300.0
_AUTONOMOUS_REFLECTION_MIN_INTERVAL_SECONDS = (
    BACKGROUND_REFLECTION_WINDOW_SECONDS
    / max(1, BACKGROUND_REFLECTION_MAX_ACTIONS - 1)
)
_RECOVERY_RESTART_PAUSE_SECONDS = 2.0
_MAX_RECOVERY_DROPPED_MESSAGES = working_history_retention_policy(
    "AURA_METABOLIC_RECOVERY_DROPPED_MAX"
).max_items


def _record_metabolic_degradation(
    error: BaseException,
    *,
    action: str = "metabolic operation degraded and isolated",
    severity: str = "degraded",
) -> None:
    record_degradation(_METABOLIC_SUBSYSTEM, error, severity=severity, action=action)


def _coerce_float(value, default: float, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if minimum is not None:
        return max(minimum, number)
    return number


class MetabolicCoordinator:
    """Handles all background / metabolic operations for the orchestrator."""

    def __init__(self, orch=None, container=None):
        self._orch = orch
        self._container = container
        # [UNITY] Metabolic Token Bucket
        self._metabolic_energy: float = 1.0  # 0.0 - 1.0
        self._last_energy_refill = time.time()
        self._energy_refill_rate = 0.05  # 5% per second

        # Neural Event Buffer — bounded to prevent accumulation under stalled drain
        self._neural_events: deque = deque(maxlen=100)
        self._event_bus = None
        self._bci_subscription_task = None

        # Background Resource Guard
        self._bg_llm_semaphore = asyncio.Semaphore(1) # Guard background LLM slots
        self._last_gc_time = 0
        self._is_processing = False  # Re-entry Guard
        from core.runtime.flags import FlagKind, declare

        self._autonomous_reflection_interval_s = max(
            _AUTONOMOUS_REFLECTION_MIN_INTERVAL_SECONDS,
            float(
                declare(
                    "AURA_AUTONOMOUS_REFLECTION_INTERVAL_S",
                    kind=FlagKind.FLOAT,
                    default=_AUTONOMOUS_REFLECTION_INTERVAL_SECONDS,
                    description=(
                        "Seconds between autonomous reflection passes; clamped to the "
                        "signed standing-authority budget"
                    ),
                    owner="core.coordinators.metabolic_coordinator",
                ).value()
            ),
        )
        self._autonomous_reflection_failure_backoff_s = max(
            30.0,
            float(
                declare(
                    "AURA_AUTONOMOUS_REFLECTION_FAILURE_BACKOFF_S",
                    kind=FlagKind.FLOAT,
                    default=_AUTONOMOUS_REFLECTION_FAILURE_BACKOFF_SECONDS,
                    description="Backoff after a failed reflection pass (floor 30)",
                    owner="core.coordinators.metabolic_coordinator",
                ).value()
            ),
        )
        self._autonomous_reflection_last_attempt_at = 0.0
        self._autonomous_reflection_last_completed_at = 0.0
        self._autonomous_reflection_next_eligible_at = 0.0
        self._autonomous_reflection_failure_count = 0
        self._autonomous_reflection_last_outcome = "never_run"

        # Proactive Cleanup
        self._cleanup_stale_locks()

    def _cleanup_stale_locks(self):
        """Remove only stale PID locks without destroying active singleton guards."""
        try:
            lock_dir = config.paths.home_dir / "locks"
            if lock_dir.exists():
                logger.info("🧹 Inspecting PID locks in %s", lock_dir)
                for lock_file in lock_dir.glob("*.lock"):
                    if not self._lock_file_is_stale(lock_file):
                        continue
                    try:
                        lock_file.unlink()
                        logger.info("🧹 Removed stale lock: %s", lock_file.name)
                    except _METABOLIC_BOUNDARY_ERRORS as _exc:
                        _record_metabolic_degradation(
                            _exc,
                            action=f"left stale lock in place: {lock_file.name}",
                            severity="warning",
                        )
                        logger.debug("Unable to remove stale lock %s: %s", lock_file, _exc)
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="stale PID lock cleanup skipped")
            logger.debug("Stale lock cleanup failed: %s", e)

    @staticmethod
    def _lock_file_is_stale(lock_file) -> bool:
        try:
            raw = lock_file.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            logger.debug("Unable to read PID lock %s: %s", lock_file, exc)
            return False

        if not raw:
            return False

        pid_text = raw.splitlines()[0].strip()
        if not pid_text.isdigit():
            return False

        pid = int(pid_text)
        if pid <= 0:
            return False

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except OSError as exc:
            logger.debug("Unable to inspect PID %s from %s: %s", pid, lock_file, exc)
            return False
        return False

    @staticmethod
    def _extract_bci_event_data(raw_event):
        if isinstance(raw_event, tuple):
            if len(raw_event) < 3:
                return None
            payload = raw_event[2]
        else:
            payload = raw_event

        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        return data if data is not None else payload

    @staticmethod
    def _reflection_topic_for_status(status) -> str:
        """Build a bounded reflection prompt from the canonical runtime status."""
        if status is None:
            return "Self-reflection on current runtime status: unavailable"

        legacy_state = getattr(status, "state", None)
        if legacy_state:
            return f"Self-reflection on current state: {str(legacy_state)[:160]}"

        message = getattr(status, "message", None)
        fields = []
        for name in (
            "running",
            "healthy",
            "is_processing",
            "is_throttled",
            "cycle_count",
            "uptime",
        ):
            if hasattr(status, name):
                fields.append(f"{name}={getattr(status, name)}")

        details = ", ".join(fields) if fields else "no structured status fields"
        if message:
            return f"Self-reflection on current runtime status: {str(message)[:120]} ({details})"
        return f"Self-reflection on current runtime status: {details[:200]}"

    @staticmethod
    def _autonomous_reflection_result_ok(result) -> bool:
        """Require a successful tool envelope with a non-empty swarm result."""
        if not isinstance(result, dict) or result.get("ok") is not True:
            return False
        output = result.get("output")
        if output is None:
            return False
        if isinstance(output, (dict, list, tuple, set)):
            return bool(output)
        text = str(output).strip()
        if not text:
            return False
        lowered = text.casefold()
        return not any(
            marker in lowered
            for marker in (
                "swarm capacity reached",
                "swarm failed to produce",
                "debate cancelled",
                "execution failure",
            )
        )

    def autonomous_reflection_status(self) -> dict[str, object]:
        now = time.time()
        return {
            "last_attempt_at": self._autonomous_reflection_last_attempt_at or None,
            "last_completed_at": self._autonomous_reflection_last_completed_at or None,
            "next_eligible_at": self._autonomous_reflection_next_eligible_at or None,
            "next_eligible_in_s": round(
                max(0.0, self._autonomous_reflection_next_eligible_at - now),
                3,
            ),
            "interval_s": self._autonomous_reflection_interval_s,
            "failure_count": self._autonomous_reflection_failure_count,
            "last_outcome": self._autonomous_reflection_last_outcome,
        }

    async def _run_autonomous_reflection(
        self,
        *,
        topic: str | None = None,
        roles: list[str] | None = None,
        reason: str = "metabolic_idle_reflection",
        objective: str = "bounded autonomous runtime reflection",
    ) -> None:
        orch = self.orch
        if orch is None:
            self._record_autonomous_reflection_failure("orchestrator_unavailable")
            return
        try:
            payload: dict[str, object] = {
                "topic": topic or self._reflection_topic_for_status(
                    getattr(orch, "status", None)
                )
            }
            if roles:
                payload["roles"] = list(roles)
            result = await asyncio.wait_for(
                orch.execute_tool(
                    "swarm_debate",
                    payload,
                    is_background=True,
                    origin="background_reflection",
                    payload_context={
                        "objective": objective,
                        "reason": reason,
                    },
                ),
                timeout=_AUTONOMOUS_REFLECTION_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._record_autonomous_reflection_failure("timeout")
            return
        except _METABOLIC_BOUNDARY_ERRORS as exc:
            _record_metabolic_degradation(exc, action="autonomous reflection failed")
            self._record_autonomous_reflection_failure(type(exc).__name__)
            return

        completed_at = time.time()
        self._autonomous_reflection_last_completed_at = completed_at
        if not self._autonomous_reflection_result_ok(result):
            reason = "non_meaningful_result"
            if isinstance(result, dict):
                reason = str(result.get("error") or result.get("output") or reason)[:160]
            self._record_autonomous_reflection_failure(reason, now=completed_at)
            return
        self._autonomous_reflection_failure_count = 0
        self._autonomous_reflection_last_outcome = "completed"
        self._autonomous_reflection_next_eligible_at = (
            completed_at + self._autonomous_reflection_interval_s
        )

    def _record_autonomous_reflection_failure(
        self,
        reason: str,
        *,
        now: float | None = None,
    ) -> None:
        completed_at = float(now if now is not None else time.time())
        self._autonomous_reflection_last_completed_at = completed_at
        self._autonomous_reflection_failure_count += 1
        backoff = max(
            _AUTONOMOUS_REFLECTION_MIN_INTERVAL_SECONDS,
            min(
                self._autonomous_reflection_interval_s,
                self._autonomous_reflection_failure_backoff_s
                * (2 ** min(8, max(0, self._autonomous_reflection_failure_count - 1))),
            ),
        )
        self._autonomous_reflection_next_eligible_at = completed_at + backoff
        self._autonomous_reflection_last_outcome = f"failed:{str(reason)[:160]}"
        logger.warning(
            "Autonomous reflection failed (%s); retry eligible in %.0fs.",
            reason,
            backoff,
        )

    def _maybe_schedule_autonomous_reflection(
        self,
        *,
        idle_time: float,
        now: float,
        topic: str | None = None,
        roles: list[str] | None = None,
        reason: str = "metabolic_idle_reflection",
        objective: str = "bounded autonomous runtime reflection",
        task_name: str = "metabolic.autonomous_reflection",
    ) -> bool:
        orch = self.orch
        if (
            orch is None
            or idle_time <= 300.0
            or bool(getattr(orch, "is_busy", False))
            or now < self._autonomous_reflection_next_eligible_at
        ):
            return False
        policy_reason = background_activity_reason(
            orch,
            profile=IDLE_COGNITION_BACKGROUND_POLICY,
            max_failure_pressure=0.20,
        )
        if policy_reason:
            self._autonomous_reflection_last_outcome = f"deferred:{policy_reason}"
            return False
        if self._allostasis_defers():
            # A reflection debate is heavy LLM work and deferrable by
            # definition; the body has asked for headroom it can see it
            # will need. Expected backpressure — outcome note, not degradation.
            self._autonomous_reflection_last_outcome = "deferred:allostasis"
            return False

        self._autonomous_reflection_last_attempt_at = now
        self._autonomous_reflection_last_outcome = "running"
        # Reserve the cadence slot before task creation so a subsequent
        # metabolic pulse cannot race a second debate into the same lane.
        self._autonomous_reflection_next_eligible_at = (
            now + self._autonomous_reflection_interval_s
        )
        task = self.track_metabolic_task(
            task_name,
            self._run_autonomous_reflection(
                topic=topic,
                roles=roles,
                reason=reason,
                objective=objective,
            ),
        )
        if task is None:
            self._autonomous_reflection_last_outcome = "deferred:scheduler_unavailable"
            self._autonomous_reflection_next_eligible_at = (
                now + self._autonomous_reflection_failure_backoff_s
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Main Tick
    # ------------------------------------------------------------------

    def _is_resource_constrained(self) -> bool:
        """Return True only for resource pressure that background throttling can relieve.

        CPU bursts during local model generation are expected foreground work.
        Treating a single CPU sample as lockdown pressure creates false sleeps
        in the live/proof path without fixing the underlying load.

        Anticipatory constraint: the allostasis engine can flag pressure that
        is not here yet but credibly will be (a forecast red-line crossing) —
        deferring background load is exactly the relief it is asking for.
        """
        try:
            from core.container import ServiceContainer
            allostasis = ServiceContainer.get("allostasis_engine", default=None)
            if allostasis is not None:
                defer, reason = allostasis.should_defer_heavy_work()
                if defer:
                    logger.info("Metabolism: deferring on allostasis signal — %s", reason)
                    return True
        except _METABOLIC_BOUNDARY_ERRORS as exc:
            logger.debug("Allostasis constraint check unavailable: %s", exc)
        try:
            from core.runtime import resource_psutil as psutil
            mem = psutil.virtual_memory().percent
            from core.runtime.disk_budget import state_volume_percent

            disk = state_volume_percent()
            return mem > 90 or disk > 95
        except (ImportError, OSError, AttributeError, RuntimeError, TypeError, ValueError):
            return False

    def _allostasis_defers(self) -> bool:
        """True when the allostasis engine asks for deferrable load to wait.

        Gates only work that is deferrable by definition (RL training,
        self-update, autonomous reflection) — never the relief work (GC,
        model scavenge, memory hygiene), which must keep running under
        pressure because it is what frees the resources being fought over.
        """
        try:
            from core.container import ServiceContainer
            allostasis = ServiceContainer.get("allostasis_engine", default=None)
            if allostasis is None:
                return False
            defer, _reason = allostasis.should_defer_heavy_work()
            return bool(defer)
        except _METABOLIC_BOUNDARY_ERRORS:
            return False

    async def _allostasis_pulse(self) -> None:
        """One allostatic sample per metabolic cycle (the 60 s pulse).

        Runs before the throttle early-returns so the vitals keep being
        sampled precisely when the system is under pressure — the moment
        trend forecasting matters most. Fail-soft: a broken pulse degrades
        prediction, never the cycle.
        """
        try:
            from core.autonomic.allostasis import get_allostasis_engine
            await get_allostasis_engine().sample_and_regulate()
        except _METABOLIC_BOUNDARY_ERRORS as exc:
            _record_metabolic_degradation(exc, action="allostasis pulse skipped this cycle")
            logger.debug("Allostasis pulse unavailable: %s", exc)

    @property
    def orch(self):
        """Authoritative lazy resolution of orchestrator from container."""
        if getattr(self, "_orch", None) is not None:
            return self._orch

        # Strict avoidance of resolution recursion
        from core.container import ServiceContainer
        obj = ServiceContainer.get("orchestrator", default=None)
        if obj:
            self._orch = obj
            return obj
        return None

    async def process_cycle(self):
        """v31.4 Enterprise Hardening: Semaphore-guarded cycle."""
        if not hasattr(self, "_cycle_semaphore"):
            self._cycle_semaphore = asyncio.Semaphore(1)

        if self._cycle_semaphore.locked():
            logger.debug("Metabolism: Cycle already in progress. Skipping overlap.")
            return False

        async with self._cycle_semaphore:
            return await self._process_cycle_inner()

    async def _process_cycle_inner(self):
        if self._is_processing:
            logger.debug("Metabolism: Cycle already in progress. Skipping overlap.")
            return False

        self._is_processing = True
        _tick_start = time.monotonic()
        # Reclaim idle model memory BEFORE any throttle/early-return below, so it
        # still fires under memory pressure — the exact condition where unloading
        # an idle model frees the most-needed RAM. Cheap when lanes are busy.
        await self._scavenge_idle_model_vram()
        # Allostatic pulse likewise runs before every early-return: vitals must
        # keep being sampled while the system is under the pressure being
        # forecast, or the forecasts go stale exactly when they matter.
        await self._allostasis_pulse()
        try:
            orch = self.orch
            if not orch:
                return False
            kernel = getattr(orch, 'kernel', None)
            volition = getattr(kernel, 'volition_level', 0) if kernel else 0

            # Level 0 (Lockdown) is extremely conservative
            if volition == 0 and self._is_resource_constrained():
                logger.warning("Metabolism: Throttling due to resource pressure (Lockdown active).")
                await asyncio.sleep(10)
                return False

            # Levels 1-3 are progressively more willing to spend resources
            if volition > 0:
                # Level 1-3 allows higher thresholds (95% mem instead of 90%)
                try:
                    from core.runtime import resource_psutil as psutil
                    mem = psutil.virtual_memory().percent
                    if mem > (90 + volition): # Level 3 allows up to 93%
                         logger.warning("Metabolism: Throttling. System saturated.")
                         return False
                except ImportError as _exc:
                    logger.debug("Suppressed ImportError: %s", _exc)

            result = await self._process_metabolic_tasks(volition)

            # ── Metrics: record tick duration ─────────────────────────
            try:
                from core.observability.metrics import get_metrics
                tick_ms = (time.monotonic() - _tick_start) * 1000.0
                get_metrics().record_tick(tick_ms)
            except _METABOLIC_BOUNDARY_ERRORS as exc:
                _record_metabolic_degradation(exc, action="metabolic tick metrics skipped")
                logger.debug("Metrics tick recording unavailable: %s", exc)

            # ── Boring Mode: periodic check ───────────────────────────
            # Every ~100 cycles, check if critical incidents warrant
            # entering boring mode for self-preservation.
            cycle_count = getattr(getattr(orch, 'status', None), 'cycle_count', 0)
            if cycle_count > 0 and cycle_count % 100 == 0:
                try:
                    from core.resilience.boring_mode import get_boring_mode
                    from core.resilience.incident_manager import get_incident_manager
                    bm = get_boring_mode()
                    if not bm.is_active:
                        mgr = get_incident_manager()
                        summary = mgr.get_summary()
                        critical_count = summary.get("by_severity", {}).get("critical", 0)
                        if critical_count >= 3:
                            bm.enter(
                                f"auto_trigger: {critical_count} critical incidents active"
                            )
                            logger.warning(
                                "🧊 Boring Mode ENTERED: %d critical incidents detected.",
                                critical_count,
                            )
                except _METABOLIC_BOUNDARY_ERRORS as exc:
                    _record_metabolic_degradation(exc, action="boring-mode incident probe skipped")
                    logger.debug("Boring mode incident probe unavailable: %s", exc)

            # ── DB Maintenance: periodic pass ─────────────────────────
            # Run every ~50 cycles (checkpoint, retention, vacuum).
            if cycle_count > 0 and cycle_count % 50 == 0:
                try:
                    from core.persistence.db_maintenance import get_db_maintenance
                    maint = get_db_maintenance()
                    maint_result = await maint.run_maintenance_async()
                    if maint_result.total_rows_deleted > 0:
                        logger.info(
                            "🗄️ DB Maintenance: deleted %d expired rows.",
                            maint_result.total_rows_deleted,
                        )
                except _METABOLIC_BOUNDARY_ERRORS as exc:
                    _record_metabolic_degradation(exc, action="database maintenance pass skipped")
                    logger.debug("DB maintenance pass unavailable: %s", exc)

            # ── Resource Governor: periodic sample ────────────────────
            if cycle_count > 0 and cycle_count % 10 == 0:
                try:
                    from core.resource.resource_governor import get_resource_governor
                    gov = get_resource_governor()
                    snap = gov.sample()
                    if snap.eviction_tier.value != "none":
                        gov.execute_eviction(snap.eviction_tier)
                except _METABOLIC_BOUNDARY_ERRORS as exc:
                    _record_metabolic_degradation(exc, action="resource-governor sample skipped")
                    logger.debug("Resource governor sample unavailable: %s", exc)

            # ── Initiative Overflow: cap adjustment ───────────────────
            if cycle_count > 0 and cycle_count % 60 == 0:
                try:
                    from core.autonomy.initiative_overflow import get_initiative_overflow
                    get_initiative_overflow().adjust_cap()
                except _METABOLIC_BOUNDARY_ERRORS as exc:
                    _record_metabolic_degradation(exc, action="initiative overflow cap adjustment skipped")
                    logger.debug("Initiative overflow cap adjustment unavailable: %s", exc)

            return result
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="metabolic cycle returned failure")
            logger.error("Metabolic cycle failed: %s", e)
            return False
        finally:
            self._is_processing = False

    async def _process_metabolic_tasks(self, volition: int = 0):
        """Internal metabolic processing (formerly process_cycle)."""
        # [UNITY] Dynamic Refill (Phase 23.5)
        now = time.time()
        delta = now - self._last_energy_refill

        # Recovery slows down when system Integrity is low
        refill_rate = 0.01 if self._metabolic_energy < 0.2 else 0.05

        self._metabolic_energy = min(1.0, self._metabolic_energy + (delta * refill_rate))
        self._last_energy_refill = now

        # [UNITY] Calculate idle time for autonomous triggers
        orch = self.orch
        last_user_interaction = (
            _coerce_float(
                getattr(orch.status, "last_user_interaction_time", 0.0)
                or getattr(orch, "_last_user_interaction_time", 0.0)
                or 0.0,
                0.0,
                minimum=0.0,
            )
            if orch
            else 0.0
        )
        idle_time = (now - last_user_interaction) if last_user_interaction > 0.0 else 0.0

        # Boot Warmup Grace Period
        # Prevent heavy MLX/GPU tasks from starving the system during initial boot.
        cycle_count = getattr(orch.status, "cycle_count", 0) if orch else 0
        if cycle_count < _BOOT_WARMUP_CYCLES:
            if cycle_count > 0:
                logger.debug(
                    "🍼 Metabolic: Grace period active (Cycle %s/%s). Skipping background tasks.",
                    cycle_count,
                    _BOOT_WARMUP_CYCLES,
                )
            return

        # Lazy Event Bus Registration
        if self._event_bus is None or (
            self._bci_subscription_task is not None and self._bci_subscription_task.done()
        ):
            try:
                from core.event_bus import get_event_bus
                self._event_bus = get_event_bus()
                # Subscription task for background thread safety
                async def _sub():
                    q = await self._event_bus.subscribe("core/senses/bci_event")
                    while not is_shutdown_requested():
                        try:
                            raw_event = await asyncio.wait_for(
                                q.get(),
                                timeout=_BCI_EVENT_POLL_SECONDS,
                            )
                        except TimeoutError:
                            continue
                        data = self._extract_bci_event_data(raw_event)
                        if data is not None:
                            self._neural_events.append(data)
                self._bci_subscription_task = self.track_metabolic_task(
                    "metabolic.bci_event_subscription",
                    _sub(),
                )
            except _METABOLIC_BOUNDARY_ERRORS as e:
                _record_metabolic_degradation(e, action="BCI event subscription not started")
                logger.debug("Failed to subscribe to BCI events: %s", e)

        orch = self.orch
        if not orch:
            return
        try:
            # Cycle count increment moved to MindTick (authority)
            # to prevent conflicting updates and "stuck" status reporting.

            # Trigger metabolic hooks (Non-blocking)
            self.track_metabolic_task(
                "metabolic.on_cycle_hook",
                orch.hooks.trigger("on_cycle", {"cycle": orch.status.cycle_count}),
            )
            if orch.status.cycle_count % 500 == 0:
                self.track_metabolic_task(
                    "metabolic.periodic_state_save",
                    orch._save_state_async("periodic"),
                )
            if orch.status.cycle_count % 1000 == 0:
                logger.info("Alive: Cycle %s", orch.status.cycle_count)
                try:
                    from core.runtime import resource_psutil as psutil
                    mem_percent = psutil.virtual_memory().percent
                    # Use status.volition_level (likely intended) instead of undefined variable
                    volition = getattr(orch.status, 'volition_level', 0)
                    # Level 2+ required for background RL; allostasis can defer it.
                    if (volition >= 2 and not orch.status.is_processing
                            and mem_percent < 80 and not self._allostasis_defers()):
                        self.track_metabolic_task("rl_training", self.run_rl_training())
                    else:
                        logger.info(
                            "Skipping RL training: Volition low (%d), system busy, or allostasis deferral.",
                            volition,
                        )
                except _METABOLIC_BOUNDARY_ERRORS as e:
                    _record_metabolic_degradation(e, action="RL training resource check skipped")
                    logger.debug("Dependency missing for memory check, skipping RL training: %s", e)
            if orch.status.cycle_count % 5000 == 0:
                try:
                    from core.runtime import resource_psutil as psutil
                    mem_percent = psutil.virtual_memory().percent
                    volition = getattr(orch.status, 'volition_level', 0)
                    # Level 3 required for background Self-Update; allostasis can defer it.
                    if (volition >= 3 and not orch.status.is_processing
                            and mem_percent < 80 and not self._allostasis_defers()):
                        self.track_metabolic_task("self_update", self.run_self_update())
                    else:
                        logger.info(
                            "Skipping Evo update: Volition low (%d), system busy, or allostasis deferral.",
                            volition,
                        )
                except _METABOLIC_BOUNDARY_ERRORS as e:
                    _record_metabolic_degradation(e, action="self-update resource check skipped")
                    logger.debug("Dependency missing for memory check, skipping Evo update: %s", e)
            # 1. Internal Pacing & Mood updates
            if orch.drive_controller:
                # Avoid calling MotivationEngine.update() blindly as it expects args and is async
                if getattr(orch.drive_controller, "name", "") != "motivation_engine":
                    try:
                        if hasattr(orch.drive_controller, 'update'):
                            res = orch.drive_controller.update()
                            if asyncio.iscoroutine(res):
                                self.track_metabolic_task(
                                    "metabolic.drive_controller_update",
                                    res,
                                )
                    except TypeError as _e:
                        logger.debug('Ignored TypeError in metabolic_coordinator.py: %s', _e)

            if hasattr(orch, 'drives') and orch.drives:
                try:
                    res = orch.drives.update()
                    if asyncio.iscoroutine(res):
                        self.track_metabolic_task(
                            "metabolic.drives_update",
                            res,
                        )
                except TypeError as _e:
                    logger.debug('Ignored TypeError in metabolic_coordinator.py: %s', _e)

            # 4. Trigger Autonomous Reflection if idle. This is a tracked job,
            # not inline heartbeat work, and has explicit cadence/backoff.
            cycle_count = int(getattr(orch.status, "cycle_count", 0) or 0)
            narrative_due = bool(
                cycle_count > 0
                and cycle_count % 25000 == 0
                and getattr(orch, "swarm", None)
            )
            if narrative_due:
                scheduled = self._maybe_schedule_autonomous_reflection(
                    idle_time=idle_time,
                    now=now,
                    topic=(
                        "Aura's current persona stability and transcendental evolution "
                        "path. Assessment of current objective: "
                        f"{str(getattr(orch, '_current_objective', '') or '')[:1000]}"
                    ),
                    roles=["philosopher", "critic", "architect"],
                    reason="metabolic_narrative_reflection",
                    objective="bounded recursive narrative reflection",
                    task_name="metabolic.narrative_reflection",
                )
                if scheduled:
                    orch._emit_thought_stream(
                        "🌀 Initiating Recursive Narrative Reflection..."
                    )
            else:
                self._maybe_schedule_autonomous_reflection(
                    idle_time=idle_time,
                    now=now,
                )
            # Grounded Introspection — Latent Core Heartbeat
            if hasattr(orch, 'latent_core') and orch.latent_core:
                try:
                    latent_summary = orch.latent_core.get_summary()
                    if hasattr(orch, 'predictive_model') and orch.predictive_model:
                        # Heavy predictive math runs on the thread-pool executor so
                        # it never blocks the event loop. This is the design — there
                        # is no separate broker task for it — so we always run it
                        # in-process rather than attempting a (nonexistent) offload.
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None,
                            orch.predictive_model.observe_and_update,
                            latent_summary,
                        )
                except _METABOLIC_BOUNDARY_ERRORS as lc_err:
                    _record_metabolic_degradation(lc_err, action="latent core heartbeat skipped")
                    logger.debug("Latent core heartbeat skipped: %s", lc_err)

            orch = self.orch
            kernel = getattr(orch, 'kernel', None) or getattr(orch, 'kernel_interface', None)

            # [COOKIE] Accelerated Thought Reflection
            cookie = kernel.organs.get("cookie") if kernel and hasattr(kernel, 'organs') else None
            state = getattr(orch, "state", None)
            if cookie and cookie.instance and state and hasattr(state.cognition, 'active_goals') and state.cognition.active_goals:
                top_goal = state.cognition.active_goals[0].get("description", "System Integrity")
                if state.affect.focus > 0.7:  # Only dilate when highly focused
                    self.track_metabolic_task(
                        "metabolic.cookie_reflection",
                        cookie.instance.reflect(state, f"Optimizing for: {top_goal}", cycles=7),
                    )
                    # We don't await here to keep the metabolic cycle moving
                    # but the result will be logged by the cookie.

            # [TRICORDER] Multi-modal Diagnostic Scan
            tricorder = kernel.organs.get("tricorder") if kernel and hasattr(kernel, 'organs') else None
            if tricorder and tricorder.instance and state:
                self.track_metabolic_task(
                    "metabolic.tricorder_scan",
                    tricorder.instance.scan(state),
                )

            # [CONTINUITY] Knowledge Distillation (Persistence)
            # Only distill during 'cool' periods to save energy
            continuity = kernel.organs.get("continuity") if kernel and hasattr(kernel, 'organs') else None
            if continuity and continuity.instance and state:
                if state.cognition.current_mode in ("dormant", "dreaming") or self._metabolic_energy < 0.1:
                    self.track_metabolic_task(
                        "metabolic.continuity_distill",
                        continuity.instance.distill(state),
                    )

            # 2. Acquire Work (Queue or Volition)
            # [COGNITIVE COOLING] Decay acceleration over time (Claude Prompt 1)
            orch.status.acceleration_factor = max(1.0, orch.status.acceleration_factor * 0.999)

            # [PRIORITY INFERENCE] Check for user-lane thoughts (Claude Prompt 1)
            # Use access to the queue to see if there's high priority work
            if hasattr(orch.message_queue, '_q'):
                high_priority = any(getattr(m, 'priority', 0) >= 50 for m in list(orch.message_queue._q._queue))
            else:
                high_priority = any(getattr(m, 'priority', 0) >= 50 for m in list(getattr(orch.message_queue, '_queue', [])))

            if high_priority and orch.status.is_processing:
                logger.debug("⚠️ [HARDENING] High-priority user thought detected. Yielding...")
                await asyncio.sleep(0.05) # Subtle yield

            # Drain Neural Events into Percepts
            while self._neural_events:
                ne = self._neural_events.popleft()
                if not isinstance(ne, dict):
                    logger.debug("Dropping malformed neural event: %r", ne)
                    continue
                cmd = ne.get("command")
                conf = ne.get("confidence", 0.0)
                # Inject as a high-intensity percept if confidence is high
                if hasattr(orch, 'world') and hasattr(orch.world, 'recent_percepts'):
                    orch.world.recent_percepts.append({
                        "type": "neural_decode",
                        "command": cmd,
                        "intensity": conf,
                        "timestamp": now
                    })
                    logger.debug("🧠 [METABOLIC] Injected neural percept: %s", cmd)

            message = await orch._acquire_next_message()
            # 3. Dispatch Work
            if message:
                orch._dispatch_message(message)
            # 4. Background Cognition & Maintenance
            if self._consume_energy(0.05):
                self.manage_memory_hygiene()

            if self._consume_energy(0.02):
                await self.process_world_decay()
            # Ensure liquid state & heartbeat are updated every cycle
            self.update_liquid_pacing()
            # 5. Autonomous Agency Triggers
            #    Morphogenesis can suppress autonomous initiative when field
            #    danger/resource_pressure/inhibition is elevated, preventing
            #    expensive background tasks from competing during crises.
            _morph_suppress = False
            try:
                from core.morphogenesis.hooks import should_suppress_autonomous_initiative
                _morph_suppress = should_suppress_autonomous_initiative()
            except _METABOLIC_BOUNDARY_ERRORS as exc:
                _record_metabolic_degradation(exc, action="morphogenesis suppression probe skipped")
                logger.debug("Morphogenesis initiative suppression probe failed: %s", exc)
            if self._consume_energy(0.1) and not _morph_suppress:
                await self.trigger_autonomous_thought(bool(message))
                await orch._pulse_agency_core()

            if self._consume_energy(0.01):
                await self.run_terminal_self_heal()
            # 6. Persona Evolution (Phase 12)
            if runtime_feature_enabled(orch, "persona_evolution", default=True) and orch.status.cycle_count % 10000 == 0:
                if hasattr(orch, 'persona_evolver') and orch.persona_evolver:
                    self.track_metabolic_task("persona_evolution", orch.persona_evolver.run_evolution_cycle())
            # 8. Eternal Record (Phase 21 Singularity)
            if orch.status.singularity_threshold and orch.status.cycle_count % 1000 == 0:
                self.emit_eternal_record()
            return bool(message)
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="metabolic task cycle returned failure")
            logging.getLogger("Aura.Critical").error("Error in process cycle: %s", e)
            # Feed the exception into the morphogenetic field so the cell ecology
            # can react (emit repair signals, trigger immunity bridge, modulate
            # resource allocation).
            try:
                from core.morphogenesis.hooks import observe_orchestrator_exception
                observe_orchestrator_exception(subsystem="metabolic_coordinator", exc=e)
            except _METABOLIC_BOUNDARY_ERRORS as exc:
                _record_metabolic_degradation(exc, action="morphogenesis exception observer skipped")
                logger.debug("Morphogenesis exception observer failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Liquid Pacing & Telemetry
    # ------------------------------------------------------------------

    def update_liquid_pacing(self):
        """Update emotional state and heartbeat sync."""
        orch = self.orch
        if not orch:
            return
        if getattr(orch, 'liquid_state', None) is None:
            return
        # [TELEMETRY SYNC] Pull VAD from affect engine (authoritative) and push to substrate
        vad_kwargs = {}
        ae = ServiceContainer.get("affect_engine", default=None)
        if ae:
            try:
                affect_status = ae.get_status()
                vad_kwargs = {
                    "valence": affect_status.get("valence"),
                    "arousal": affect_status.get("arousal")
                }
            except _METABOLIC_BOUNDARY_ERRORS as e:
                _record_metabolic_degradation(e, action="authoritative affect status unavailable")
                logger.debug("Failed to pull authoritative affect status: %s", e)
        elif hasattr(orch, 'affect_engine') and orch.affect_engine:
            try:
                affect_status = orch.affect_engine.get_status()
                vad_kwargs = {
                    "valence": affect_status.get("valence"),
                    "arousal": affect_status.get("arousal")
                }
            except _METABOLIC_BOUNDARY_ERRORS as e:
                _record_metabolic_degradation(e, action="background affect status unavailable")
                logger.debug("Failed to pull background affect status: %s", e)

        # liquid_state.update() is async — schedule it properly
        try:
            asyncio.get_running_loop()
            self.track_metabolic_task(
                "metabolic.liquid_state_update",
                orch.liquid_state.update(**vad_kwargs),
            )
        except RuntimeError as _e:
            logger.debug("Liquid-state update deferred outside an event loop: %s", _e)
        if hasattr(orch, '_watchdog') and orch._watchdog:
            orch._watchdog.heartbeat("orchestrator")
        if orch.lnn:
            stimuli = {
                "curiosity": orch.liquid_state.current.curiosity,
                "frustration": orch.liquid_state.current.frustration,
                "energy": orch.liquid_state.current.energy
            }
            self.track_metabolic_task("lnn_pulse", orch.lnn.pulse(stimuli))
        if hasattr(orch, 'mortality') and orch.mortality:
            self.track_metabolic_task("mortality_pulse", orch.mortality.heartbeat())
            if orch.status.cycle_count % 100 == 0:
                self.track_metabolic_task("threat_assessment", orch.mortality.assess_threats())
        sm = getattr(orch, 'singularity_monitor', None)
        if sm:
            sm.pulse()
        if hasattr(orch, 'affect_engine') and orch.affect_engine:
            if "affect_decay" not in self._active_task_names(orch):
                self.track_metabolic_task("affect_decay", orch.affect_engine.decay_tick())
        idle_time = time.time() - orch._last_thought_time
        curiosity = orch.liquid_state.current.curiosity
        if orch.homeostasis:
            curiosity = orch.homeostasis.curiosity
        if curiosity < 0.2 and idle_time > 60:
            if time.time() - orch._last_boredom_impulse > 300:
                self.trigger_boredom_impulse()
        frustration = orch.liquid_state.current.frustration
        if frustration > 0.6:
            if time.time() - orch._last_reflection_impulse > 300:
                self.trigger_reflection_impulse()
        if time.time() - orch._last_pulse > 5:
            self.emit_neural_pulse()
            self.emit_telemetry_pulse()
        if hasattr(orch, 'liquid_state') and orch.liquid_state:
            orch.status.agency = orch.liquid_state.current.energy
            orch.status.curiosity = orch.liquid_state.current.curiosity

    def emit_telemetry_pulse(self):
        """Emit real-time liquid state telemetry."""
        orch = self.orch
        if not orch:
            return
        try:
            ls = orch.liquid_state
            if ls:
                ls_status = ls.get_status()
                orch._publish_telemetry({
                    "energy": ls_status.get("energy", 80),
                    "curiosity": ls_status.get("curiosity", 50),
                    "frustration": ls_status.get("frustration", 0),
                    "confidence": ls_status.get("focus", 50),
                    "mood": ls_status.get("mood", "NEUTRAL"),
                    "acceleration_factor": orch.status.acceleration_factor,
                    "singularity_active": orch.status.singularity_threshold
                })
        except _METABOLIC_BOUNDARY_ERRORS as exc:
            _record_metabolic_degradation(exc, action="telemetry pulse failed; recovery scheduled")
            logger.error("Telemetry pulse failure: %s", exc)
            if hasattr(orch, "_recover_from_stall"):
                self.track_metabolic_task("metabolic.recover_from_stall", self.recover_from_stall())

    def emit_eternal_record(self):
        """Archives a snapshot of the system's current state into the Eternal Record."""
        try:
            from core.config import config
            from core.resilience.eternal_record import EternalRecord
            record_store = config.paths.home_dir / "eternal_archive"
            archivist = EternalRecord(record_store)
            kg_path = config.paths.data_dir / "knowledge.db"
            snapshot_dir = archivist.create_snapshot(kg_path)
            if snapshot_dir:
                self.orch._emit_thought_stream(f"🏺 Eternal Record Snapshot secured: {snapshot_dir.name}")
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="eternal record snapshot skipped")
            logger.debug("Eternal record snapshot failed: %s", e)

    # ------------------------------------------------------------------
    # Impulses
    # ------------------------------------------------------------------

    def trigger_boredom_impulse(self):
        """Inject a curiosity-driven autonomous goal."""
        orch = self.orch
        if not orch:
            return
        reason = background_activity_reason(orch, min_idle_seconds=300.0, max_memory_percent=78.0)
        if reason:
            logger.debug("Skipping boredom impulse: %s", reason)
            return
        logger.info("🥱 BOREDOM TRIGGERED: Generating curiosity impulse.")
        orch._last_boredom_impulse = time.time()
        try:
            from core.autonomy.topic_selection import select_autonomous_topic

            state = getattr(getattr(orch, "kernel", None), "state", None)
            candidate = select_autonomous_topic(orch, state)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_metabolic_degradation(
                exc,
                action="left boredom impulse idle because no grounded topic could be selected",
            )
            candidate = None
        if candidate is None:
            logger.debug("Boredom impulse found no grounded unresolved topic or interest.")
            return
        topic = candidate.text
        try:
            asyncio.get_running_loop()
            self.track_metabolic_task(
                "metabolic.boredom_impulse",
                run_governed_impulse(
                    orch,
                    source="metabolic_coordinator",
                    summary=f"metabolic_boredom_impulse:{candidate.source}:{topic}",
                    message=f"Impulse: I am bored. I want to research {topic}.",
                    urgency=0.3,
                    state_cause="metabolic_boredom_shift",
                    state_update={"delta_curiosity": 0.5},
                    enqueue_priority=25,
                ),
            )
        except RuntimeError as _e:
            logger.debug("Boredom impulse deferred outside an event loop: %s", _e)

    def trigger_reflection_impulse(self):
        """Inject a self-reflection goal due to frustration."""
        orch = self.orch
        if not orch:
            return
        reason = background_activity_reason(orch, profile=IDLE_COGNITION_BACKGROUND_POLICY)
        if reason:
            logger.debug("Skipping reflection impulse: %s", reason)
            return
        logger.info("😤 FRUSTRATION TRIGGERED: Generating reflection impulse.")
        orch._last_reflection_impulse = time.time()
        try:
            asyncio.get_running_loop()
            self.track_metabolic_task(
                "metabolic.reflection_impulse",
                run_governed_impulse(
                    orch,
                    source="metabolic_coordinator",
                    summary="metabolic_reflection_impulse",
                    message="Impulse: I feel frustrated. I need to reflect on my recent interactions.",
                    urgency=0.3,
                    state_cause="metabolic_reflection_shift",
                    state_update={"delta_frustration": -0.3},
                    enqueue_priority=15,
                ),
            )
        except RuntimeError as _e:
            logger.debug("Reflection impulse deferred outside an event loop: %s", _e)

    def emit_neural_pulse(self):
        """Emit system health to thought stream."""
        orch = self.orch
        if not orch:
            return
        try:
            from core.thought_stream import get_emitter
            mood = orch.liquid_state.get_mood() if hasattr(orch, 'liquid_state') else "Stable"
            get_emitter().emit("Neural Pulse", f"System Active (Mood: {mood})", level="info", cycle=orch.status.cycle_count)
            orch._last_pulse = time.time()
        except _METABOLIC_BOUNDARY_ERRORS as _e:
            _record_metabolic_degradation(_e, action="neural pulse skipped")
            logger.debug("Neural pulse emit failed: %s", _e)

    # ------------------------------------------------------------------
    # Task Tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _active_task_names(orch):
        active = getattr(orch, "_active_metabolic_tasks", None)
        if active is None:
            active = set()
            try:
                orch._active_metabolic_tasks = active
            except _METABOLIC_BOUNDARY_ERRORS as exc:
                _record_metabolic_degradation(
                    exc,
                    action="using local active-task set for metabolic dedupe",
                    severity="warning",
                )
        return active

    def track_metabolic_task(self, name: str, coro):
        """Ensures metabolic tasks don't pile up and exhaust resources."""
        import inspect

        if not coro or not inspect.isawaitable(coro):
            # If it's already done (sync) or None, don't track it
            return

        orch = self.orch
        if not orch:
            if hasattr(coro, "close"):
                coro.close()
            return
        active_tasks = self._active_task_names(orch)
        if name in active_tasks:
            # v31.1 FIX: Explicitly close the coroutine if we skip tracking
            # to prevent 'coroutine was never awaited' RuntimeWarning.
            if hasattr(coro, "close"):
                coro.close()
            return
        active_tasks.add(name)
        tracker = get_task_tracker()
        try:
            schedule = tracker.track
        except AttributeError:
            schedule = tracker.create_task
        try:
            task = schedule(coro, name=name)
        except RuntimeError as exc:
            active_tasks.discard(name)
            if hasattr(coro, "close"):
                coro.close()
            logger.debug("Metabolic task %s deferred outside an event loop: %s", name, exc)
            return None
        except _METABOLIC_BOUNDARY_ERRORS as exc:
            active_tasks.discard(name)
            if hasattr(coro, "close"):
                coro.close()
            _record_metabolic_degradation(exc, action=f"metabolic task {name} was not scheduled")
            logger.debug("Metabolic task %s scheduling failed: %s", name, exc)
            return None

        def _cleanup(t):
            active_tasks.discard(name)
            if t.cancelled():
                return
            try:
                task_error = t.exception()
            except asyncio.CancelledError:
                return
            except _METABOLIC_BOUNDARY_ERRORS as exc:
                _record_metabolic_degradation(exc, action=f"metabolic task {name} cleanup degraded")
                logger.debug("Metabolic task %s cleanup failed: %s", name, exc)
                return
            if task_error:
                _record_metabolic_degradation(task_error, action=f"metabolic task {name} failed")
                logger.error("Metabolic task %s failed: %s", name, task_error)

        try:
            task.add_done_callback(_cleanup)
        except _METABOLIC_BOUNDARY_ERRORS as exc:
            active_tasks.discard(name)
            _record_metabolic_degradation(exc, action=f"metabolic task {name} cleanup callback not attached")
            logger.debug("Metabolic task %s cleanup callback registration failed: %s", name, exc)

        return task

    def _consume_energy(self, amount: float) -> bool:
        """Consume metabolic energy. Returns False if insufficient energy."""
        if self._metabolic_energy >= amount:
            self._metabolic_energy -= amount
            return True
        return False

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    async def recover_from_stall(self):
        """Attempts to recover from a cognitive loop stall."""
        from core.config import config
        from core.container import ServiceContainer
        orch = self.orch
        if not orch:
            return
        orch._recovery_attempts += 1
        logger.warning("🚑 RECOVERY ATTEMPT #%s initiated...", orch._recovery_attempts)
        try:
            dlq = ServiceContainer.get("dead_letter_queue", default=None)
            if dlq:
                dlq.capture_failure(
                    message=getattr(orch, "_current_objective", "None"),
                    context={"recovery_attempt": orch._recovery_attempts},
                    error=RuntimeError("Cognitive Stall Detected"),
                    source="orchestrator_stall"
                )
        except _METABOLIC_BOUNDARY_ERRORS as dlq_e:
            _record_metabolic_degradation(dlq_e, action="stall recovery DLQ receipt unavailable")
            logger.error("CRITICAL: Failed to log to DLQ during stall: %s", dlq_e)
        try:
            if orch._current_thought_task and not orch._current_thought_task.done():
                logger.info("Cancelling hanging thought task...")
                orch._current_thought_task.cancel()
            if orch.message_queue.qsize() > 50:
                logger.warning("Message queue overflow detected. Clearing and moving to DLQ...")
                dropped = []
                while not orch.message_queue.empty() and len(dropped) < _MAX_RECOVERY_DROPPED_MESSAGES:
                    raw = orch.message_queue.get_nowait()
                    # Handle both 3-tuple and 4-tuple formats for safety during cleanup
                    if isinstance(raw, tuple):
                        msg = raw[-1]
                    else:
                        msg = raw
                    dropped.append(msg)
                if not orch.message_queue.empty():
                    logger.warning(
                        "Message queue still contains items after bounded recovery drain (%s).",
                        _MAX_RECOVERY_DROPPED_MESSAGES,
                    )
                if dropped:
                    try:
                        dlq_path = config.paths.data_dir / "dlq.jsonl"
                        payload = [
                            json.dumps({"timestamp": time.time(), "message": msg}, default=str) + "\n"
                            for msg in dropped
                        ]

                        def _append_lines() -> None:
                            from core.runtime.file_write_gateway import get_file_write_gateway

                            get_file_write_gateway().append_text(
                                dlq_path,
                                "".join(payload),
                                encoding="utf-8",
                                source="metabolic_coordinator.recovery_dlq",
                            )

                        await asyncio.to_thread(_append_lines)
                    except _METABOLIC_BOUNDARY_ERRORS as e:
                        _record_metabolic_degradation(e, action="dropped messages were not persisted to DLQ file")
                        logger.error("Failed to dump dropped messages to DLQ file: %s", e)
            await orch.retry_cognitive_connection()
            if orch._recovery_attempts >= 2 and hasattr(orch, 'lazarus') and orch.lazarus:
                logger.warning("🚨 [RECOVERY] Escalating to Lazarus Brainstem...")
                await orch.lazarus.attempt_recovery()
            if orch._recovery_attempts >= 3:
                logger.critical("🚨 STALL PERSISTS: Escalating to full orchestrator restart.")
                orch.status.running = False
                await asyncio.sleep(_RECOVERY_RESTART_PAUSE_SECONDS)
                await orch.start()
                orch._recovery_attempts = 0
            logger.info("✅ Recovery logic applied.")
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="stall recovery sequence failed")
            logger.error("Recovery sequence failed: %s", e)

    # ------------------------------------------------------------------
    # Memory Hygiene
    # ------------------------------------------------------------------

    def manage_memory_hygiene(self):
        from core.config import config
        from core.container import ServiceContainer
        orch = self.orch
        if not orch:
            return
        if isinstance(orch.conversation_history, list):
            if len(orch.conversation_history) > 150:
                orch.conversation_history = orch.conversation_history[-150:]
        if len(orch.conversation_history) > 2:
            self.deduplicate_history()
        if len(orch.conversation_history) > 100:
            self.track_metabolic_task(
                "metabolic.prune_history",
                self.prune_history_async(),
            )
        if orch.status.cycle_count % 1000 == 0:
            # Phase XIV: Reduced VACUUM frequency to prevent SQLite locks
            async def _optimize_dbs():
                audit = ServiceContainer.get("subsystem_audit", default=None)
                try:
                    from core.resilience.database_coordinator import get_db_coordinator
                    db_coord = get_db_coordinator()
                    logger.info("🧹 Enqueueing deep database hygiene (VACUUM)...")
                    # ZENITH: Wrap glob in thread
                    db_files = await asyncio.to_thread(lambda: list(config.paths.data_dir.glob("*.db")))
                    for db_file in db_files:
                        await db_coord.execute_write(str(db_file), "VACUUM")
                except _METABOLIC_BOUNDARY_ERRORS as e:
                    _record_metabolic_degradation(e, action="database hygiene pass failed")
                    logger.error("Database hygiene failed: %s", e)
                    if audit:
                        audit.report_failure("database_hygiene", str(e))
                finally:
                    # Always emit heartbeat — proves the hygiene task ran
                    if audit:
                        audit.heartbeat("database_hygiene")
            self.track_metabolic_task(
                "metabolic.optimize_databases",
                _optimize_dbs(),
            )

        if len(orch.conversation_history) > 10 and orch.memory_manager:
            # Circuit Breaker: Only consolidate if memory subsystem is healthy
            audit = ServiceContainer.get("subsystem_audit", default=None)
            if audit and audit.get_status("memory").get("degraded", False):
                logger.warning("Memory consolidated SKIPPED: Subsystem is DEGRADED.")
            else:
                self.track_metabolic_task(
                    "metabolic.consolidate_long_term_memory",
                    self.consolidate_long_term_memory(),
                )

        if orch.status.cycle_count % 1000 == 0:
            if hasattr(orch, 'memory') and orch.memory:
                try:
                    prune_result = orch.memory.prune_low_salience(threshold_days=14)
                    if asyncio.iscoroutine(prune_result):
                        self.track_metabolic_task(
                            "metabolic.prune_low_salience",
                            prune_result,
                        )
                except _METABOLIC_BOUNDARY_ERRORS as e:
                    _record_metabolic_degradation(e, action="vector salience pruning skipped")
                    logger.debug("Vector pruning skipped: %s", e)

        # ZENITH LOCKDOWN: Periodic Garbage Collection
        if orch.status.cycle_count % 500 == 0:
            try:
                from core.runtime import resource_psutil as psutil
                mem_percent = psutil.virtual_memory().percent
                # Proactive GC if RAM > 85% or every 30s-ish
                if mem_percent > 85 or (time.time() - self._last_gc_time > 30):
                    logger.debug("♻️ Metabolic RAM-aware GC Triggered (RAM: %s%%).", mem_percent)
                    gc.collect()
                    self._last_gc_time = time.time()
            except ImportError:
                gc.collect()

    def deduplicate_history(self):
        """Remove consecutive identical messages."""
        orch = self.orch
        if not orch:
            return
        if not orch.conversation_history:
            return
        first_msg = orch.conversation_history[0] if orch.conversation_history else None
        if not first_msg:
            return
        deduped = [first_msg]
        for msg in orch.conversation_history[1:]:
            if msg.get("content") != deduped[-1].get("content"):
                deduped.append(msg)
        orch.conversation_history = deduped

    async def prune_history_async(self):
        """Asynchronously prune history via context pruner."""
        orch = self.orch
        if not orch:
            return
        try:
            from core.memory.context_pruner import context_pruner
            orch.conversation_history = await context_pruner.prune_history(
                orch.conversation_history, orch.cognitive_engine
            )
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="history pruner failed; applying local trim")
            logger.debug("History pruning failed: %s", e)
            if isinstance(orch.conversation_history, list) and len(orch.conversation_history) > 50:
                orch.conversation_history = orch.conversation_history[-50:]

    async def consolidate_long_term_memory(self):
        """Summarize and move important session highlights to long-term vector memory."""
        from core.container import ServiceContainer
        orch = self.orch
        if not orch:
            return
        try:
            if len(orch.conversation_history) % 15 != 0:
                return
            logger.info("🧠 Consolidating session highlights to long-term memory...")
            recent = orch.conversation_history[-20:] if isinstance(orch.conversation_history, list) else []
            if not recent:
                return
            chat_text = "\n".join([f"{m['role']}: {m.get('content', '')}" for m in recent])
            from core.brain.cognitive_engine import ThinkingMode
            summary_prompt = (
                "Review this recent conversation fragment and extract 3-5 key 'long-term' facts "
                "or user preferences learned. Format as single-sentence declarations. "
                "Focus on what's important for future context, ignoring fluff.\n\n"
                f"Conversation:\n{chat_text}"
            )
            summary_thought = await orch.cognitive_engine.think(
                objective=summary_prompt,
                context={"history": []},
                mode=ThinkingMode.FAST,
                is_background=True
            )
            if summary_thought and summary_thought.content:
                highlights = summary_thought.content
                logger.info("✨ Key Highlights Extracted: %s", (highlights or "")[:100])
                if orch.memory_manager:
                    await orch.memory_manager.log_event(
                        "session_consolidation",
                        highlights,
                        metadata={"type": "summary", "session_start": orch.start_time}
                    )
                    orch._emit_telemetry("Memory", "Session highlights consolidated to long-term storage.")
                archive_eng = ServiceContainer.get("archive_engine", default=None)
                if archive_eng and hasattr(archive_eng, 'archive_vital_logs'):
                    logger.info("📦 Deep Sleep Cycle: Triggering Metabolic Archival Compression...")
                    await archive_eng.archive_vital_logs()
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="long-term memory consolidation failed")
            logger.error("Memory consolidation failed: %s", e)
            # Circuit Breaker: Report degradation
            audit = ServiceContainer.get("subsystem_audit", default=None)
            if audit:
                audit.report_failure("memory", str(e))

    # ------------------------------------------------------------------
    # World Decay
    # ------------------------------------------------------------------

    async def process_world_decay(self):
        """Apply entropy to internal belief systems."""
        from core.container import ServiceContainer
        orch = self.orch
        if not orch:
            return
        if orch.status.cycle_count % 60 == 0:
            try:
                from core.world_model.belief_graph import belief_graph
                # ZENITH: Wrap sync decay in executor to prevent loop blocking
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, belief_graph.decay, 0.001)
            except _METABOLIC_BOUNDARY_ERRORS as e:
                _record_metabolic_degradation(e, action="belief graph decay skipped")
                logger.error("World decay error: %s", e)
        if orch.status.cycle_count % 600 == 0:
            try:
                if orch.metabolic_monitor:
                    health_snapshot = orch.metabolic_monitor.get_current_metabolism()
                    health = health_snapshot.health_score
                    if health < 0.2:
                        archive_eng = ServiceContainer.get("archive_engine", default=None)
                        if archive_eng:
                            logger.info("📦 Metabolic Pressure Detected (Health: %.2f). Triggering Emergency Archival.", health)
                            self.track_metabolic_task(
                                "metabolic.emergency_archive",
                                archive_eng.archive_vital_logs(),
                            )
            except _METABOLIC_BOUNDARY_ERRORS as e:
                _record_metabolic_degradation(e, action="emergency archival trigger skipped")
                logger.debug("Metabolic Archival trigger failed: %s", e)
        if runtime_feature_enabled(orch, "persona_evolution", default=True) and orch.status.cycle_count % 3600 == 0:
            try:
                from core.evolution.persona_evolver import PersonaEvolver
                evolver = PersonaEvolver(orch)
                self.track_metabolic_task(
                    "metabolic.persona_evolution_cycle",
                    evolver.run_evolution_cycle(),
                )
            except _METABOLIC_BOUNDARY_ERRORS as e:
                _record_metabolic_degradation(e, action="persona evolution trigger skipped")
                logger.debug("Persona Evolution trigger failed: %s", e)

    # ------------------------------------------------------------------
    # Autonomous Thought
    # ------------------------------------------------------------------

    async def trigger_autonomous_thought(self, has_message: bool):
        """Trigger idle-time search for autonomous goals."""
        orch = self.orch
        if not orch:
            return
        if not orch.cognitive_engine or has_message:
            return
        is_thinking = orch._current_thought_task is not None and not orch._current_thought_task.done()
        if not is_thinking:
            idle = time.time() - orch._last_thought_time
            sm = getattr(orch, 'singularity_monitor', None)

            # [VOLITION] Accelerated Thought Factor
            factor = getattr(sm, 'acceleration_factor', 1.0) if sm else 1.0
            if hasattr(orch.cognitive_engine, 'singularity_factor'):
                factor = orch.cognitive_engine.singularity_factor

            factor = _coerce_float(factor, 1.0, minimum=1.0)
            configured_min_interval = _coerce_float(
                runtime_mode_value(orch, "autonomous_thought_interval_s", 45.0),
                45.0,
                minimum=1.0,
            )
            threshold = 45.0 / factor

            kernel = getattr(self.orch, 'kernel', None)
            volition = getattr(kernel, 'volition_level', 0) if kernel else 0

            # Level 1 (Reflective): Only triggers internal reflection
            # Level 2 (Perceptive): Normal threshold
            # Level 3 (Agentic): Aggressive (Threshold / 2)
            if volition == 0:
                return # No autonomous thought in Lockdown
            elif volition == 3:
                threshold /= 2.0

            threshold = max(configured_min_interval, threshold)

            if idle >= threshold:
                orch.boredom = int(idle)
                logger.info("🧠 Accelerated Thought (Volition: L%d, Factor: %.1fx, Threshold: %.1fs)", volition, factor, threshold)
                orch._current_thought_task = self.track_metabolic_task(
                    "metabolic.autonomous_thought",
                    orch._perform_autonomous_thought(),
                )

    # ------------------------------------------------------------------
    # Terminal Self-Heal
    # ------------------------------------------------------------------

    async def run_terminal_self_heal(self):
        """Check terminal monitor for errors to fix."""
        orch = self.orch
        if not orch:
            return
        try:
            try:
                policy_reason = background_activity_reason(
                    orch,
                    profile=IDLE_COGNITION_BACKGROUND_POLICY,
                    max_failure_pressure=0.25,
                    allow_no_user_anchor=True,
                )
                if policy_reason:
                    logger.debug(
                        "Terminal Monitor: metabolic auto-fix deferred by background policy: %s",
                        policy_reason,
                    )
                    return
            except _METABOLIC_BOUNDARY_ERRORS as policy_exc:
                _record_metabolic_degradation(policy_exc, action="terminal self-heal policy probe skipped")
                logger.debug("Metabolic terminal self-heal policy probe failed: %s", policy_exc)

            from core.terminal_monitor import get_terminal_monitor
            monitor = get_terminal_monitor()
            if monitor:
                error_goal = await monitor.check_for_errors()
                if error_goal and not (orch._current_thought_task is not None and not orch._current_thought_task.done()):
                    logger.info("🔧 Terminal Monitor: Auto-fix triggered")
                    if orch.self_modifier:
                        error_text = error_goal.get("error", "Unknown")
                        orch.self_modifier.on_error(
                            Exception(f"Terminal Command Failure: {error_text}")
                            if isinstance(error_text, str)
                            else Exception("Terminal Command Failure"),
                            {"command": error_goal.get("command"), "output": error_goal.get("output")},
                            skill_name="TerminalMonitor"
                        )
                    runner = getattr(orch, "_run_cognitive_loop", None) or getattr(orch, "_handle_incoming_message", None)
                    objective = error_goal.get("objective")
                    if runner is not None and objective:
                        orch._current_thought_task = self.track_metabolic_task(
                            "metabolic.terminal_self_heal",
                            runner(objective, origin="terminal_monitor"),
                        )
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="terminal self-heal check failed")
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "terminal_monitor",
                    "self_heal_check_failed",
                    detail=f"{type(e).__name__}: {e}",
                    severity="warning",
                    classification="background_degraded",
                    exc=e,
                )
            except _METABOLIC_BOUNDARY_ERRORS as _exc:
                _record_metabolic_degradation(_exc, action="terminal degraded-event receipt unavailable")
                logger.debug("Terminal degraded-event receipt failed: %s", _exc)
            logger.debug("Terminal monitor check failed: %s", e)

    # ------------------------------------------------------------------
    # Background Reflection & Learning
    # ------------------------------------------------------------------

    def trigger_background_reflection(self, response: str):
        from core.orchestrator.types import _bg_task_exception_handler
        orch = self.orch
        if not orch:
            return
        reflect_coro = None
        reflect_task = None
        try:
            from core.conversation_reflection import get_reflector
            reflect_coro = get_reflector().maybe_reflect(
                orch.conversation_history,
                orch.cognitive_engine,
                mood=orch._get_current_mood(),
                time_str=orch._get_current_time_str(),
            )
            reflect_task = self.track_metabolic_task(
                "metabolic.background_reflection",
                reflect_coro,
            )
            if reflect_task is not None:
                try:
                    reflect_task.add_done_callback(_bg_task_exception_handler)
                except _METABOLIC_BOUNDARY_ERRORS as exc:
                    _record_metabolic_degradation(exc, action="background reflection callback not attached")
                    logger.debug("Background reflection callback registration failed: %s", exc)
                    reflect_task.cancel()
                    raise
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="background reflection setup failed")
            if reflect_coro is not None and reflect_task is None:
                reflect_coro.close()
            logger.debug("Background reflection setup failed: %s", e)

    def trigger_background_learning(self, message: str, response: str):
        from core.orchestrator.types import _bg_task_exception_handler
        orch = self.orch
        if not orch:
            return
        learn_coro = None
        learn_task = None
        try:
            original_msg = message.replace("Impulse: ", "").replace("Thought: ", "")
            learn_coro = orch._learn_from_exchange(original_msg, response)
            learn_task = self.track_metabolic_task(
                "metabolic.background_learning",
                learn_coro,
            )
            if learn_task is not None:
                try:
                    learn_task.add_done_callback(_bg_task_exception_handler)
                except _METABOLIC_BOUNDARY_ERRORS as exc:
                    _record_metabolic_degradation(exc, action="background learning callback not attached")
                    logger.debug("Background learning callback registration failed: %s", exc)
                    learn_task.cancel()
                    raise
            if orch.curiosity and hasattr(orch.curiosity, 'extract_curiosity_from_conversation'):
                orch.curiosity.extract_curiosity_from_conversation(original_msg)
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="background learning setup failed")
            if learn_coro is not None and learn_task is None:
                learn_coro.close()
            logger.debug("Background learning setup failed: %s", e)

    # ------------------------------------------------------------------
    # RL & Self-Update
    # ------------------------------------------------------------------

    async def _scavenge_idle_model_vram(self) -> None:
        """Unload idle local model lanes to reclaim unified memory.

        Delegates to the lane-safe scavenger (it refuses any busy/foreground
        lane and respawns transparently on the next request). Best-effort: never
        let a maintenance reclaim disturb the metabolic cycle.
        """
        try:
            from core.brain.llm.mlx_client import scavenge_idle_model_vram

            outcome = await scavenge_idle_model_vram()
            if outcome.get("unloaded"):
                logger.info(
                    "🧹 Metabolism: idle VRAM scavenge unloaded %d model lane(s).",
                    outcome["unloaded"],
                )
        except _METABOLIC_BOUNDARY_ERRORS as exc:
            _record_metabolic_degradation(exc, action="idle VRAM scavenge skipped")
            logger.debug("Idle VRAM scavenge skipped: %s", exc)

    async def run_rl_training(self):
        """Trigger autonomous RL training."""
        logger.info("🧠 RL: Triggering policy optimization...")
        try:
            from core.tasks import dispatch_background
            dispatch_background("core.tasks.run_rl_training")
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="RL training task dispatch failed")
            logger.error("RL training trigger failed: %s", e)

    async def run_self_update(self):
        """Trigger autonomous self-update (Fine-tuning)."""
        logger.info("🧬 EVO: Triggering self-update (GPU low-load window)...")
        try:
            from core.tasks import dispatch_background
            dispatch_background("core.tasks.run_self_update")
        except _METABOLIC_BOUNDARY_ERRORS as e:
            _record_metabolic_degradation(e, action="self-update task dispatch failed")
            logger.error("Self-update trigger failed: %s", e)

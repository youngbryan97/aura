"""core/autonomy/sleep_trigger.py

Autonomous Sleep Trigger — Homeostatic Idle Consolidation Drive.

Watches `_last_user_interaction_time` on the orchestrator. When the system
has been idle for the threshold period (default 30 min) and the time since
the last sleep cycle is sufficient (default 2 hr), triggers DreamerV2's full
biological maintenance pipeline.

Guards:
  - Active conversation → never interrupt
  - Already sleeping → skip
  - Recent sleep within cooldown window → skip
  - High CPU load → defer (don't consolidate while actively thinking)

This is what makes idle time productive instead of dead time.
"""
import asyncio
import logging
import time

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.SleepTrigger")

_IDLE_THRESHOLD_SEC  = 30 * 60    # 30 minutes idle before first trigger
_SLEEP_COOLDOWN_SEC  = 2 * 60 * 60  # 2 hours between sleep cycles
_CPU_DEFER_THRESHOLD = 60.0         # % CPU — defer if system is under load
_CHECK_INTERVAL_SEC  = 60           # Poll every 60 seconds


class AutonomousSleepTrigger:
    """
    Monitors idle state and autonomously initiates memory consolidation
    (DreamerV2.engage_sleep_cycle()) when conditions are right.
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._boot_idle_baseline: float = float(
            getattr(orchestrator, "start_time", 0.0) or time.time()
        )
        self._last_sleep_time: float = 0.0
        self._sleeping: bool = False
        self._task: asyncio.Task | None = None
        self.running = False

    async def start(self):
        self.running = True
        self._task = get_task_tracker().create_task(self._watch_loop(), name="SleepTrigger")
        logger.info("😴 SleepTrigger active (idle=%.0fm, cooldown=%.0fh)",
                    _IDLE_THRESHOLD_SEC / 60, _SLEEP_COOLDOWN_SEC / 3600)

    async def stop(self):
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()

    # ─── Core loop ────────────────────────────────────────────────────────────

    async def _watch_loop(self):
        while self.running:
            await asyncio.sleep(_CHECK_INTERVAL_SEC)
            try:
                await self._evaluate()
            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('sleep_trigger', e)
                logger.debug("SleepTrigger eval error (non-fatal): %s", e)

    async def _evaluate(self):
        orch = self._get_orchestrator()
        if not orch:
            return

        # Guard: active conversation
        if getattr(orch.status, "is_processing", False):
            return

        # Guard: already in a sleep cycle
        if self._sleeping:
            return

        # Guard: sleep cooldown
        if (time.time() - self._last_sleep_time) < _SLEEP_COOLDOWN_SEC:
            return

        # Guard: idle threshold
        last_user = float(getattr(orch, "_last_user_interaction_time", 0.0) or 0.0)
        if last_user <= 0.0:
            logger.debug(
                "SleepTrigger: waiting for a real user anchor (boot idle %.0fs).",
                max(0.0, time.time() - self._boot_idle_baseline),
            )
            return
        idle_sec = time.time() - last_user
        if idle_sec < _IDLE_THRESHOLD_SEC:
            return

        # Guard: CPU load — defer if system is busy with something else
        try:
            from core.runtime import resource_psutil as psutil
            cpu = psutil.cpu_percent(interval=None)
            if cpu > _CPU_DEFER_THRESHOLD:
                logger.debug("SleepTrigger: CPU %.0f%% > threshold, deferring.", cpu)
                return
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('sleep_trigger', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        # All guards passed — initiate sleep
        await self._initiate_sleep(orch, idle_sec)

    async def _initiate_sleep(self, orch, idle_sec: float):
        self._sleeping = True
        idle_min = idle_sec / 60

        logger.info("🌙 SleepTrigger: %.0f min idle — initiating consolidation.", idle_min)

        try:
            from core.thought_stream import get_emitter
            get_emitter().emit(
                "Sleep",
                f"Idle for {idle_min:.0f} minutes. Beginning memory consolidation.",
                level="info",
                category="SleepCycle",
            )
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('sleep_trigger', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            dreamer = self._find_dreamer(orch)
            if dreamer:
                results = await dreamer.engage_sleep_cycle()
                dreamed = (results.get("dream") or {}).get("dreamed", False)
                # consolidation is a ConsolidationReport (success, field
                # `duplicates_merged`) or a {"error": ...} dict (failure path),
                # never a {"merged": ...} dict — read both shapes defensively.
                consolidation = results.get("consolidation")
                if isinstance(consolidation, dict):
                    consolidated = consolidation.get(
                        "duplicates_merged", consolidation.get("merged", 0)
                    )
                else:
                    consolidated = getattr(consolidation, "duplicates_merged", 0)
                logger.info("🌅 Sleep cycle complete: dreamed=%s consolidated=%s",
                            dreamed, consolidated)
                try:
                    from core.thought_stream import get_emitter
                    get_emitter().emit(
                        "Waking",
                        f"Consolidation complete. Dreamed: {'yes' if dreamed else 'no'}.",
                        level="info",
                        category="SleepCycle",
                    )
                except (ImportError, AttributeError, RuntimeError) as _exc:
                    record_degradation('sleep_trigger', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
            else:
                logger.warning("SleepTrigger: DreamerV2 not found — skipping cycle.")
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('sleep_trigger', e)
            logger.error("SleepTrigger: Sleep cycle failed: %s", e)
        finally:
            self._last_sleep_time = time.time()
            self._sleeping = False

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_orchestrator(self):
        if self.orchestrator:
            return self.orchestrator
        try:
            return get_runtime_service("orchestrator", default=None)
        except (ImportError, AttributeError, RuntimeError):
            # not a failure: no orchestrator registered, which the default
            # on the lookup above already returns None for.
            return None

    @staticmethod
    def _find_dreamer(orch):
        """Locate DreamerV2 via runtime registry or orchestrator attributes."""
        try:
            dreamer = get_runtime_service("dreamer_v2", default=None)
            if dreamer:
                return dreamer
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('sleep_trigger', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        # Fallback: build one from orchestrator's sub-systems
        for attr in ("dreamer_v2", "dreamer", "dream_engine"):
            d = getattr(orch, attr, None)
            if d and hasattr(d, "engage_sleep_cycle"):
                return d

        # Last resort: instantiate with available services
        try:
            from core.sleep.dreamer_v2 import DreamerV2
            brain = get_runtime_service("brain", default=None) or getattr(orch, "brain", None)
            kg    = get_runtime_service("knowledge_graph", default=None) or getattr(orch, "knowledge_graph", None)
            vm    = get_runtime_service("vector_memory", default=None) or getattr(orch, "vector_memory", None)
            bg    = get_runtime_service("belief_graph", default=None)
            if brain and kg:
                return DreamerV2(brain=brain, knowledge_graph=kg, vector_memory=vm, belief_graph=bg)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('sleep_trigger', e)
            logger.debug("SleepTrigger: could not instantiate DreamerV2: %s", e)

        return None


# ── Singleton ──────────────────────────────────────────────────────────────────
_trigger: AutonomousSleepTrigger | None = None


def get_sleep_trigger(orchestrator=None) -> AutonomousSleepTrigger:
    global _trigger
    if _trigger is None:
        _trigger = AutonomousSleepTrigger(orchestrator)
    elif orchestrator and not _trigger.orchestrator:
        _trigger.orchestrator = orchestrator
    return _trigger

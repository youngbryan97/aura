"""Aura Zenith Cognitive Loop Service.
Decouples the cognitive cycle from the Orchestrator.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.service_access import (
    optional_service,
    resolve_affect_engine,
    resolve_belief_graph,
    resolve_curiosity_engine,
    resolve_identity_model,
    resolve_motivation_engine,
    resolve_self_prediction,
)
from core.utils.queues import unpack_priority_message
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.CognitiveLoop")

_COGNITIVE_LOOP_RECOVERABLE_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_cognitive_loop_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "cognitive_loop",
        exc,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


class CognitiveLoop:
    """Service responsible for the cognitive thinking cycle."""

    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.is_running = False
        self._task: asyncio.Task | None = None
        self.cycle_count = 0
        self.last_cycle_time = time.monotonic()
        self.stall_threshold = 30.0  # Seconds
        # Two worker PROCESSES used to be forked here, in __init__, whether or
        # not this loop ever deliberated — and the only thing that reaped them
        # was __del__ calling shutdown(wait=False), which depends on GC timing
        # and does not wait for the children to actually go. Constructing a
        # CognitiveLoop is cheap and common; forking is neither.
        self._deliberation_pool: Any | None = None
        self._active_deliberation_task: asyncio.Task | None = None

    def _get_deliberation_pool(self):
        """The process pool, created the first time deliberation needs one."""

        if self._deliberation_pool is None:
            from concurrent.futures import ProcessPoolExecutor

            self._deliberation_pool = ProcessPoolExecutor(max_workers=2)
        return self._deliberation_pool

    def shutdown_deliberation_pool(self, *, wait: bool = False) -> None:
        """Reap the deliberation workers, if any were ever started."""

        pool, self._deliberation_pool = self._deliberation_pool, None
        if pool is None:
            return
        try:
            pool.shutdown(wait=wait)
        except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as e:
            logger.debug("Failed to shutdown deliberation pool: %s", e)

    def __del__(self):
        # Retained as a backstop, but no longer the only path: stop() and
        # shutdown_deliberation_pool() reap deterministically.
        try:
            self.shutdown_deliberation_pool()
        except Exception as exc:  # noqa: BLE001 — a finalizer must not raise
            logger.debug("Cognitive loop finalizer could not close its pool: %s", exc)

    async def start(self):
        """Start the cognitive cycle."""
        if self.is_running:
            return
        self.is_running = True
        self._task = get_task_tracker().create_task(self.run(), name="cognitive-loop")
        logger.info("🧠 Cognitive Loop service started.")

    async def stop(self):
        """Stop the cognitive cycle."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.CancelledError as _e:
                logger.debug("Cognitive loop task acknowledged cancellation: %s", _e)
            except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as exc:
                _record_cognitive_loop_degradation(
                    exc,
                    action="continued shutdown after cognitive loop task did not stop cleanly",
                    severity="warning",
                )
            self._task = None
        if self._active_deliberation_task:
            self._active_deliberation_task.cancel()
            try:
                await asyncio.wait_for(self._active_deliberation_task, timeout=2.0)
            except asyncio.CancelledError:
                logger.debug("Active deliberation task acknowledged cancellation.")
            except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as exc:
                _record_cognitive_loop_degradation(
                    exc,
                    action="continued shutdown after active deliberation task did not stop cleanly",
                    severity="warning",
                )
            self._active_deliberation_task = None
        self.shutdown_deliberation_pool()
        logger.info("🧠 Cognitive Loop service stopped.")

    async def run(self):
        """Main cognitive cycle."""
        while self.is_running:
            try:
                async with asyncio.timeout(self.stall_threshold):
                    await self._process_cycle()
                self.last_cycle_time = time.monotonic()

                # Dynamic pacing based on load
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                break
            except TimeoutError as e:
                _record_cognitive_loop_degradation(
                    e,
                    action="recovered from stalled cognitive cycle after watchdog timeout",
                    severity="degraded",
                    extra={
                        "cycle_count": self.cycle_count,
                        "stall_threshold_seconds": self.stall_threshold,
                    },
                )
                logger.critical(
                    "Cognitive cycle exceeded %ss; running stall recovery.", self.stall_threshold
                )
                await self._recover_from_stall()
                if self.is_running:
                    await asyncio.sleep(1.0)
            except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as e:
                _record_cognitive_loop_degradation(
                    e,
                    action="kept cognitive loop alive after failed cycle and applied backoff",
                    severity="degraded",
                    extra={"cycle_count": self.cycle_count},
                )
                logger.error("Error in Cognitive Loop cycle: %s", e, exc_info=True)
                await asyncio.sleep(1.0)  # Error backoff

    async def _process_cycle(self):
        """Single cognitive cycle iteration."""
        self.cycle_count += 1
        if hasattr(self.orchestrator, "status"):
            self.orchestrator.status.cycle_count = self.cycle_count

        # 1. Free Energy Heartbeat (Metabolic Metric)
        await self._tick()

        # Unified Cognitive Pipeline: Health Check every 20 cycles
        if self.cycle_count % 20 == 0:
            await self._check_coordinators_health()

        # 2. Message Acquisition (Priority Weighted)
        message = await self._acquire_next_message()
        if message:
            await self._dispatch_message(message)

        # 3. Autonomous Thinking (Legacy fallthrough)
        await self._trigger_autonomous_thought(bool(message))

        # 4. Health & Heartbeat via Orchestrator
        if hasattr(self.orchestrator, "_update_heartbeat"):
            self.orchestrator._update_heartbeat()

    async def _tick(self):
        """
        The real cognitive heartbeat. Every cycle, Aura:
        1. Computes her free energy (replaces phi)
        2. Derives her emotional state from it
        3. Decides what autonomous action (if any) to take
        4. Updates her self-model
        """
        from core.consciousness.free_energy import get_free_energy_engine

        fe_engine = get_free_energy_engine()
        spl = resolve_self_prediction(default=None)
        belief_system = resolve_belief_graph(default=None)
        affect = resolve_affect_engine(default=None)

        # 1. Get prediction error from SelfPredictionLoop
        prediction_error = (
            spl.get_surprise_signal() if spl and hasattr(spl, "get_surprise_signal") else 0.2
        )

        # 2. Compute free energy
        recent_actions = getattr(self.orchestrator, "_recent_action_count", 0)
        user_interaction_gap = time.time() - getattr(
            self.orchestrator, "_last_user_interaction_time", 0
        )
        user_present = user_interaction_gap < 30

        fe_state = fe_engine.compute(
            prediction_error=prediction_error,
            belief_system=belief_system,
            recent_action_count=recent_actions,
            user_present=user_present,
        )

        # 3. Push emotional state to affect engine (grounded in actual internal state)
        if affect and hasattr(affect, "update_from_free_energy"):
            affect.update_from_free_energy(fe_state.valence, fe_state.arousal)
        elif affect and hasattr(affect, "update_emotion"):
            # Fallback for AffectEngineV2
            affect.update_emotion("free_energy", fe_state.valence, fe_state.arousal)

        # 4. Autonomous action — driven by FE (Active Inference), not a timer
        # Decouple heavy deliberation asynchronously to prevent 1Hz heartbeat tick drift
        if not user_present and fe_state.dominant_action in (
            "act_on_world",
            "explore",
            "update_beliefs",
        ):
            if self._active_deliberation_task is None or self._active_deliberation_task.done():
                if self._active_deliberation_task and self._active_deliberation_task.done():
                    try:
                        self._active_deliberation_task.result()
                    except asyncio.CancelledError:
                        logger.debug("Async deep deliberation was cancelled before completion.")
                    except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as e:
                        _record_cognitive_loop_degradation(
                            e,
                            action="observed failed deep deliberation task and allowed next cycle to schedule a new one",
                            severity="warning",
                            extra={"dominant_action": fe_state.dominant_action},
                        )
                        logger.error("Async deep deliberation failed: %s", e)

                self._active_deliberation_task = get_task_tracker().create_task(
                    self._autonomous_action_async(fe_state),
                    name="cognitive-loop-deliberation",
                )

        # 5. Update self-model with real internal telemetry
        self._update_self_telemetry(fe_state, prediction_error)

    async def _autonomous_action_async(self, fe_state):
        """Asynchronous wrapper for autonomous actions to prevent main cognitive heartbeat drift."""
        try:
            await self._autonomous_action(fe_state)
        except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as e:
            _record_cognitive_loop_degradation(
                e,
                action="contained autonomous action failure inside deliberation task",
                severity="warning",
                extra={"dominant_action": getattr(fe_state, "dominant_action", "unknown")},
            )
            logger.error("Error in async autonomous action: %s", e)

    async def _autonomous_action(self, fe_state):
        """
        Aura acts because her free energy demands it, not because a timer fired.
        """
        spl = resolve_self_prediction(default=None)
        motivation = resolve_motivation_engine(default=None)

        # Appraise the moment through the one control ladder before the specific drives fire.
        # This is what makes the hierarchical agency causal rather than an island: the live
        # free-energy + nociception signals become a Situation, and the tier that wins does
        # real work (reflex/scientific-hypothesis/self-improvement-signal/governance) and opens
        # an outcome-ledger receipt — binding the action engines into the live heartbeat.
        await self._appraise_through_agency(fe_state, spl)

        # Emotional regulation gates impulse: if arousal is high but deliberation is thin and
        # no real damage backs it, hold this cycle instead of acting on the spike. This is the
        # maturity layer made causal — it decides whether to act, not just how to feel.
        if self._regulation_says_hold(fe_state):
            logger.debug("🧘 [Regulation] holding autonomous action this cycle (impulse regulated).")
            return

        if fe_state.dominant_action == "explore":
            # Find the most unpredictable dimension and explore it
            if spl and hasattr(spl, "get_most_unpredictable_dimension"):
                unpredictable_dim = spl.get_most_unpredictable_dimension()
                curiosity = resolve_curiosity_engine(default=None)
                if curiosity and hasattr(curiosity, "explore_dimension"):
                    await curiosity.explore_dimension(unpredictable_dim)

        elif fe_state.dominant_action == "act_on_world":
            # High FE + inaction = generate an intention from the belief system
            if motivation and hasattr(motivation, "pulse"):
                intention = await motivation.pulse()
                if intention and hasattr(motivation, "_dispatch_intention"):
                    await motivation._dispatch_intention(intention)

    def _regulation_says_hold(self, fe_state) -> bool:
        """Ask the emotional regulator whether to hold the impulse this cycle (fail-open)."""
        try:
            from core.affect.emotional_regulation import get_emotional_regulator
            reg = get_emotional_regulator().regulate(
                arousal=float(getattr(fe_state, "arousal", 0.0) or 0.0),
                valence=float(getattr(fe_state, "valence", 0.0) or 0.0),
                deliberation=0.3,  # the autonomous lane is fast/reactive, not deeply deliberated
            )
            return bool(reg.hold)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # False lets the autonomous lane act. Not being able to ask
            # whether she should hold must not look like being told no.
            logger.debug(
                "emotional regulation could not be consulted (%s: %s); not holding",
                type(exc).__name__,
                exc,
            )
            return False

    async def _appraise_through_agency(self, fe_state, spl):
        """Run the live moment through the hierarchical control ladder (fail-open).

        Builds a Situation from the real internal signals — felt damage (nociception),
        prediction error / surprise (uncertainty), and free-energy arousal (urgency) — and
        dispatches it. The winning tier does causal work and records an outcome receipt, so
        the agency ladder, scientific engine, RSI signal path, nociception, and outcome ledger
        are all exercised by the heartbeat instead of sitting idle.
        """
        try:
            from core.agency.hierarchical_agency import Situation, get_hierarchical_agency

            threat = 0.0
            try:
                from core.affect.nociception import get_nociception_engine
                threat = float(get_nociception_engine().nociceptive_pressure())
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug(
                    "nociceptive pressure unreadable (%s: %s); the situation "
                    "is judged with no threat",
                    type(exc).__name__,
                    exc,
                )

            uncertainty = 0.2
            if spl is not None and hasattr(spl, "get_surprise_signal"):
                try:
                    uncertainty = float(spl.get_surprise_signal())
                except (RuntimeError, TypeError, ValueError) as exc:
                    logger.debug(
                        "surprise signal unreadable (%s: %s); the situation "
                        "keeps its default uncertainty",
                        type(exc).__name__,
                        exc,
                    )

            arousal = float(getattr(fe_state, "arousal", 0.0) or 0.0)
            dominant = str(getattr(fe_state, "dominant_action", "") or "")

            situation = Situation(
                description=f"heartbeat:{dominant or 'idle'}",
                threat=threat,
                uncertainty=uncertainty,
                novelty=arousal if dominant == "explore" else 0.0,
                goal_horizon=0.5 if dominant == "act_on_world" else 0.0,
                context={"dominant_action": dominant, "arousal": arousal},
            )
            # Dispatch off the event loop — the tier handlers may touch SQLite / Will.
            result = await asyncio.to_thread(get_hierarchical_agency().dispatch, situation)
            logger.debug(
                "🪜 [Agency] heartbeat appraisal: %s → %s (handled=%s)",
                result.starting_tier.name, result.final_tier.name, result.handled,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_cognitive_loop_degradation(
                exc, action="skipped hierarchical-agency appraisal this cycle", severity="debug",
            )

    def _update_self_telemetry(self, fe_state, prediction_error: float):
        """Update Aura's self-model with ACTUAL internal readings."""
        self_model = resolve_identity_model(default=None)
        if not self_model:
            return

        telemetry = {
            "free_energy": fe_state.free_energy,
            "prediction_error": prediction_error,
            "emotional_valence": fe_state.valence,
            "emotional_arousal": fe_state.arousal,
            "processing_intent": fe_state.dominant_action,
            "is_distressed": fe_state.free_energy > 0.7,
            "timestamp": time.time(),
        }

        if hasattr(self_model, "update_telemetry"):
            self_model.update_telemetry(telemetry)

    async def _acquire_next_message(self) -> dict | None:
        """Get next message from orchestrator queue (PriorityQueue support)."""
        try:
            q = self.orchestrator.message_queue
            if q.empty():
                return None

            item = await q.get()
            payload, origin = unpack_priority_message(item)

            # Unpack the wrapped dict (content, origin)
            if isinstance(payload, dict) and "content" in payload:
                return payload
            elif isinstance(payload, str):
                return {"content": payload, "origin": origin or "unknown"}

            return {"content": str(payload), "origin": origin or "raw"}
        except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as e:
            _record_cognitive_loop_degradation(
                e,
                action="continued cognitive cycle without a queued message",
                severity="warning",
            )
            logger.error("Failed to acquire message from queue: %s", e)
            return None

    async def _dispatch_message(self, message: dict):
        """Process incoming message using immediate priority bypass.
        Bypassing process_user_input prevents infinite recursion.
        """
        content = message.get("content", "")
        origin = message.get("origin", "user")

        try:
            if hasattr(self.orchestrator, "process_user_input_priority"):
                await self.orchestrator.process_user_input_priority(content, origin=origin)
            elif hasattr(self.orchestrator, "process_user_input"):
                # Legacy fallback (may still recurse if orchestrator doesn't have priority method)
                await self.orchestrator.process_user_input(content)
            else:
                raise RuntimeError("orchestrator has no supported input dispatch method")
        except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as exc:
            _record_cognitive_loop_degradation(
                exc,
                action="dropped one queued message after orchestrator dispatch failed",
                severity="degraded",
                extra={"origin": origin, "content_preview": str(content)[:160]},
            )
            logger.error("CognitiveLoop dispatch failed: %s", exc, exc_info=True)

    async def _check_coordinators_health(self):
        """Audit critical coordinators and trigger restarts if needed."""
        # Coordinator names in container
        critical_components = {
            "agency_core": "Agency Core",
            "memory_coordinator": "Memory Coordinator",
            "affect_engine": "Affect Engine",
            "personality_engine": "Personality Engine",
        }

        for key, name in critical_components.items():
            try:
                comp = optional_service(key, default=None)
                if comp is None:
                    try:
                        from core.runtime.ablation_policy import service_intentionally_lesioned

                        if service_intentionally_lesioned(key):
                            logger.info(
                                "Ablation active for critical component '%s' (%s); repair suppressed for lesion validity.",
                                name,
                                key,
                            )
                            continue
                    except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as exc:
                        _record_cognitive_loop_degradation(
                            exc,
                            action="continued coordinator health audit without ablation marker",
                            severity="warning",
                            extra={"component": key},
                        )
                    logger.warning(
                        "🚨 [RECOVERY] Critical component '%s' (%s) missing from container. Triggering Orchestrator repair...",
                        name,
                        key,
                    )
                    if hasattr(self.orchestrator, "retry_cognitive_connection"):
                        # This often fixes the container state
                        await self.orchestrator.retry_cognitive_connection()
            except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as exc:
                _record_cognitive_loop_degradation(
                    exc,
                    action="continued health audit after coordinator recovery check failed",
                    severity="warning",
                    extra={"component": key},
                )

    async def _trigger_autonomous_thought(self, was_responding: bool):
        """
        [DEPRECATED] v15: Autonomous thoughts are now driven by _tick's FE logic.
        Legacy boredom timers are disabled to prevent periodic 'cron' behavior.
        """
        logger.debug("🧠 CognitiveLoop: Legacy autonomous thought trigger ignored (deprecated).")

    async def _recover_from_stall(self):
        """Emergency recovery logic for cognitive stalls."""
        if hasattr(self.orchestrator, "_recover_from_stall"):
            try:
                await self.orchestrator._recover_from_stall()
            except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as exc:
                _record_cognitive_loop_degradation(
                    exc,
                    action="fell back to local cognitive stall recovery after orchestrator recovery failed",
                    severity="degraded",
                )
                self.cycle_count = 0
                self.last_cycle_time = time.monotonic()
        else:
            # Basic internal recovery
            logger.warning("Falling back to basic cognitive recovery.")
            self.cycle_count = 0
            self.last_cycle_time = time.monotonic()
            # Maybe reset LLM circuit breakers
            if hasattr(self.orchestrator, "cognitive_engine"):
                ce = self.orchestrator.cognitive_engine
                if ce and hasattr(ce, "unload_models"):
                    try:
                        await ce.unload_models()
                    except _COGNITIVE_LOOP_RECOVERABLE_ERRORS as e:
                        _record_cognitive_loop_degradation(
                            e,
                            action="completed basic stall recovery without unloading cognitive models",
                            severity="warning",
                        )
                        logger.error("Failed to unload models during recovery: %s", e)

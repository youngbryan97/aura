"""The impulses metabolism can raise.

Lifted whole out of `metabolic_coordinator`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import time


class _TriggersTheImpulses:
    """Lifted whole out of MetabolicCoordinator; see metabolic_coordinator.py."""

    def trigger_boredom_impulse(self):
        """Inject a curiosity-driven autonomous goal."""
        from .metabolic_coordinator import (
            _record_metabolic_degradation,
            background_activity_reason,
            logger,
            run_governed_impulse,
        )

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
        from .metabolic_coordinator import (
            IDLE_COGNITION_BACKGROUND_POLICY,
            background_activity_reason,
            logger,
            run_governed_impulse,
        )

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

    async def trigger_autonomous_thought(self, has_message: bool):
        """Trigger idle-time search for autonomous goals."""
        from .metabolic_coordinator import (
            _coerce_float,
            logger,
            runtime_mode_value,
        )

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

    def trigger_background_reflection(self, response: str):
        from core.orchestrator.types import _bg_task_exception_handler

        from .metabolic_coordinator import (
            _METABOLIC_BOUNDARY_ERRORS,
            _record_metabolic_degradation,
            logger,
        )
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

        from .metabolic_coordinator import (
            _METABOLIC_BOUNDARY_ERRORS,
            _record_metabolic_degradation,
            logger,
        )
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


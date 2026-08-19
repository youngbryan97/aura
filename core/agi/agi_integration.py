"""core/agi/agi_integration.py
===========================

Operationally: this starts, stops and health-checks four subsystems —
dimensional expansion, actuator synthesis, the unified inference engine and
the consciousness loop — and records a degradation when one of them fails.
"AGI" names the group of modules it coordinates; nothing here measures
generality, and no test in the tree treats the name as a claim.
AGI Integration Layer coordinates the dimensional expansion, actuator synthesis,
unified inference engine, and consciousness loop modules of Aura.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from typing import Any

import numpy as np

from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Aura.AGI.Integration")

#: Failures that must leave start() rolled back rather than half-running.
_AGI_START_ERRORS = (
    ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError,
)


def _task_done(task: Any) -> bool:
    """Best-effort completion check for any task-like handle."""
    done = getattr(task, "done", None)
    if callable(done):
        try:
            return bool(done())
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return False
    return False


async def _await_quietly(task: asyncio.Task) -> None:
    """Await a cancelled task without re-raising its CancelledError."""
    try:
        await task
    except asyncio.CancelledError:
        return

_AGI_RUNTIME_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
)


def _record_agi_degradation(
    subsystem: str,
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        subsystem,
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


# Thread safety lock for the singleton instance
_SINGLETON_LOCK = threading.Lock()
_agi_integration_instance: AGIIntegrationLayer | None = None


class AGIIntegrationLayer:
    """Singleton coordinator that ties AGI subsystems together."""

    def __init__(self) -> None:
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._lock = threading.Lock()

        # Metrics & Telemetry
        self.tick_count = 0
        self.last_tick_time = 0.0
        self.last_save_time = time.time()
        self.start_time = time.time()

        # Prefer the CANONICAL container-owned organs. Constructing private
        # instances here forked the system: inference feedback and runtime
        # telemetry accumulated in two places, so whichever one a consumer
        # happened to hold saw a different picture of the same body. A private
        # instance is now only a last resort, and it is recorded when it
        # happens rather than silently federating.
        from core.brain.homeostatic_modulator import HomeostaticModulator
        from core.brain.inference_feedback import InferenceFeedbackLoop

        self.modulator = ServiceContainer.get("homeostatic_modulator", default=None)
        self.feedback_loop = ServiceContainer.get("inference_feedback_loop", default=None)
        forked = []
        if self.modulator is None:
            self.modulator = HomeostaticModulator()
            forked.append("homeostatic_modulator")
            ServiceContainer.register_instance(
                "homeostatic_modulator", self.modulator, required=False
            )
        if self.feedback_loop is None:
            self.feedback_loop = InferenceFeedbackLoop()
            forked.append("inference_feedback_loop")
            ServiceContainer.register_instance(
                "inference_feedback_loop", self.feedback_loop, required=False
            )
        if forked:
            # Registering what we built keeps the NEXT consumer on the same
            # instance, so at most one fork can ever exist.
            logger.info(
                "AGIIntegrationLayer created and published canonical %s.",
                ", ".join(forked),
            )

        # Phase 2 Proprioception, Grounding, and World Modeling
        from core.embodiment.digital_body import get_digital_body
        from core.grounding.affordance_model import get_affordance_model
        from core.world_model.transition_model import get_transition_model

        self.digital_body = get_digital_body()
        self.affordance_model = get_affordance_model()
        self.transition_model = get_transition_model()

        logger.info(
            "AGIIntegrationLayer initialized with proprioception, grounding, and transition modeling."
        )

    async def start(self) -> None:
        """Starts the integration layer background tasks."""
        with self._lock:
            if self._running:
                logger.warning("AGIIntegrationLayer is already running.")
                return
            # The running flag and the container registration used to be set
            # BEFORE the task existed, so any failure below left the instance
            # advertised as running with no loop behind it — permanently, since
            # a later start() would return early on that same flag. Everything
            # is now rolled back on failure and the flag is only set once a
            # live task exists.
            try:
                ServiceContainer.register("agi_integration", self)

                from core.utils.task_tracker import get_task_tracker

                tracker = get_task_tracker()
                self._loop_task = tracker.create_task(
                    self._run_loop(), name="agi.integration_loop"
                )
            except _AGI_START_ERRORS as exc:
                self._loop_task = None
                self._running = False
                record_degradation(
                    "agi_integration", exc, severity="warning",
                    action="AGI integration not started; left stopped rather than falsely running",
                )
                logger.error("AGIIntegrationLayer failed to start: %s", exc)
                raise
            self._running = True
            logger.info("AGIIntegrationLayer started background tick task.")

    async def stop(self) -> None:
        """Stops the integration layer and performs final state saves.

        The task is AWAITED before the final save. Previously stop() cancelled,
        dropped the reference, and saved immediately — so an in-flight tick
        could still be mutating the very state being written, and the save
        raced the loop it was supposed to be finalising.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            task, self._loop_task = self._loop_task, None

        # The tracker may hand back something that is not a real asyncio.Task
        # (a stub, a wrapper, a test double). Shutdown must not depend on the
        # exact shape of that handle — cancel what can be cancelled, join what
        # can be joined, and never crash the stop path over a missing method.
        if task is not None and not _task_done(task):
            cancel = getattr(task, "cancel", None)
            if callable(cancel):
                cancel()
            try:
                # Bounded: a wedged tick must not hold shutdown open forever,
                # but the common case joins immediately.
                if inspect.isawaitable(task) or isinstance(task, asyncio.Task):
                    await asyncio.wait_for(
                        asyncio.shield(_await_quietly(task)), timeout=5.0
                    )
            except (TimeoutError, asyncio.CancelledError):
                logger.warning(
                    "AGIIntegrationLayer tick did not stop within 5s; saving anyway."
                )
            except _AGI_START_ERRORS as exc:
                record_degradation(
                    "agi_integration", exc, severity="warning",
                    action="continued shutdown after the tick loop errored on cancel",
                )

        # Final save of logit projection weights, now that nothing is ticking.
        self._save_projection_weights()
        logger.info("AGIIntegrationLayer stopped.")

    async def _run_loop(self) -> None:
        """Background loop executing homeostatic ticks every 1 second."""
        while self._running:
            try:
                start_time = time.time()
                await self._run_tick()
                elapsed = time.time() - start_time
                # Sleep remaining time of the 1-second interval
                await asyncio.sleep(max(0.1, 1.0 - elapsed))
            except asyncio.CancelledError:
                break
            except _AGI_RUNTIME_ERRORS as exc:
                _record_agi_degradation(
                    "agi_integration_loop",
                    exc,
                    action="kept AGI integration loop alive after recoverable tick failure",
                    severity="degraded",
                )
                logger.error("Error in AGI integration tick loop: %s", exc)
                await asyncio.sleep(1.0)

    async def _run_tick(self) -> None:
        """Executes a single step across all AGI subsystems."""
        self.tick_count += 1
        self.last_tick_time = time.time()

        # 0. Update proprioceptive telemetry and linear-causal transition modeling
        try:
            self.digital_body.update_telemetry()

            # Retrieve last executed action from commitments
            last_action = "reflect"
            if self.digital_body.current_commitments:
                active = [
                    c for c in self.digital_body.current_commitments if c.get("status") == "active"
                ]
                if active:
                    last_action = active[-1].get("action", "reflect")

            err = self.transition_model.process_step(last_action)

            # Inject transition prediction error surprise directly into FreeEnergyEngine
            if err > 0.5:
                free_energy = ServiceContainer.get("free_energy_engine", default=None)
                if free_energy and hasattr(free_energy, "accept_surprise_signal"):
                    free_energy.accept_surprise_signal(err)
        except _AGI_RUNTIME_ERRORS as exc:
            _record_agi_degradation(
                "agi_grounding_tick",
                exc,
                action="skipped proprioceptive transition tick and preserved loop cadence",
            )
            logger.debug("Failed to step proprioceptive/transition systems: %s", exc)

        # 1. Step the PrecisionEngine (FitzHugh-Nagumo oscillator)
        precision = ServiceContainer.get("precision_engine", default=None)
        if precision:
            try:
                # Advancing FHN oscillator
                precision.step()
            except _AGI_RUNTIME_ERRORS as exc:
                _record_agi_degradation(
                    "agi_precision_step",
                    exc,
                    action="skipped precision oscillator step and continued AGI integration tick",
                )
                logger.debug("Failed to step PrecisionEngine: %s", exc)

        # 2. Periodically trigger dimensional expansion contractions
        expansion = ServiceContainer.get("dimensional_expansion", default=None)
        if expansion and self.tick_count % 30 == 0:  # Every 30 ticks (30s)
            try:
                # Clean up under-used dimensions
                retired_axes = expansion.evaluate_contraction()
                if retired_axes:
                    logger.info("Dimensional expansion retired axes: %s", retired_axes)
            except _AGI_RUNTIME_ERRORS as exc:
                _record_agi_degradation(
                    "agi_expansion_contraction",
                    exc,
                    action="skipped dimensional contraction pass and preserved active dimensions",
                )
                logger.debug("Failed contraction evaluation: %s", exc)

        # 3. Periodically persist SubstrateLogitProjection weights
        if time.time() - self.last_save_time >= 300.0:  # Every 5 minutes
            self._save_projection_weights()
            self.last_save_time = time.time()

    def _save_projection_weights(self) -> None:
        """Saves weights from SubstrateLogitProjection to persistence path."""
        try:
            if hasattr(self.modulator, "projection") and self.modulator.projection:
                self.modulator.projection.save()
                logger.info("Persisted SubstrateLogitProjection weights successfully.")
        except _AGI_RUNTIME_ERRORS as exc:
            _record_agi_degradation(
                "agi_projection_save",
                exc,
                action="kept in-memory projection active after persistence failure",
                severity="degraded",
            )
            logger.error("Failed to save logit projection weights: %s", exc)

    def on_inference_complete(
        self, output_text: str, token_ids: list[int], logprobs: list[float] | None, modulation: Any
    ) -> dict[str, float]:
        """Inference callback to compute and propagate feedback metrics."""
        try:
            return self.feedback_loop.process_output(
                output_text=output_text,
                token_ids=token_ids,
                logprobs=logprobs,
                modulation=modulation,
                modulator_projection=self.modulator.projection,
            )
        except _AGI_RUNTIME_ERRORS as exc:
            _record_agi_degradation(
                "agi_inference_complete_feedback",
                exc,
                action="returned conservative feedback metrics after inference feedback failure",
                severity="degraded",
            )
            logger.error("Failed executing on_inference_complete callback: %s", exc)
            return {"surprise": 0.5, "coherence": 0.0}

    def get_modulation(self) -> Any:
        """Fetches the active homeostatic inference modulation."""
        try:
            return self.modulator.compute_modulation()
        except _AGI_RUNTIME_ERRORS as exc:
            _record_agi_degradation(
                "agi_get_modulation",
                exc,
                action="returned conservative homeostatic modulation after modulator failure",
                severity="degraded",
            )
            # Safe default fallback modulation
            from core.brain.homeostatic_modulator import InferenceModulation

            # head_weights used to be a hardcoded 32-element vector, invented
            # here without consulting the resident checkpoint. If the active
            # model does not have 32 attention heads that array is
            # shape-incompatible — and it LOOKS usable, so it reaches the
            # steering path and either errors deep inside or silently
            # mis-weights heads. A fallback must not assert an architecture it
            # never read.
            #
            # None means "no head weighting", which every consumer must already
            # handle (the modulator legitimately returns unweighted modulation).
            # If the true head count is discoverable we use it; otherwise we
            # decline to guess.
            return InferenceModulation(
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                logit_bias={},
                head_weights=self._neutral_head_weights(),
                urgency=0.5,
            )

    def _neutral_head_weights(self) -> np.ndarray | None:
        """Unweighted head vector sized to the ACTUAL model, or None.

        Guessing a head count produces an array that is confidently the wrong
        shape. Reading it costs one attribute lookup; not knowing it is a
        legitimate answer.
        """
        heads = None
        try:
            registry = ServiceContainer.get("model_registry", default=None)
            config = getattr(registry, "active_config", None) or {}
            if isinstance(config, dict):
                for key in ("num_attention_heads", "n_heads", "num_heads"):
                    value = config.get(key)
                    if isinstance(value, int) and 0 < value <= 512:
                        heads = value
                        break
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Head count not discoverable for fallback modulation: %s", exc)
        if heads is None:
            return None
        return np.ones(heads, dtype=np.float32)

    def get_unified_telemetry(self) -> dict[str, Any]:
        """Aggregates and returns state telemetry from all subsystems."""
        telemetry: dict[str, Any] = {
            "integration": {
                "ticks": self.tick_count,
                "uptime_seconds": round(time.time() - self.start_time, 2),
                "last_tick": self.last_tick_time,
            }
        }

        # 1. Precision Engine (FitzHugh-Nagumo)
        precision = ServiceContainer.get("precision_engine", default=None)
        if precision:
            try:
                telemetry["precision"] = precision.get_state_dict()
            except _AGI_RUNTIME_ERRORS as exc:
                _record_agi_degradation(
                    "agi_telemetry_precision",
                    exc,
                    action="omitted precision telemetry section from unified AGI telemetry",
                    severity="debug",
                )

        # 2. Liquid Substrate
        substrate = ServiceContainer.get("liquid_substrate", default=None)
        if substrate:
            try:
                with substrate.sync_lock:
                    telemetry["substrate"] = {
                        "valence": round(float(substrate.x[substrate.idx_valence]), 4),
                        "arousal": round(float(substrate.x[substrate.idx_arousal]), 4),
                        "frustration": round(float(substrate.x[substrate.idx_frustration]), 4),
                        "curiosity": round(float(substrate.x[substrate.idx_curiosity]), 4),
                        "focus": round(float(substrate.x[substrate.idx_focus]), 4),
                    }
            except _AGI_RUNTIME_ERRORS as exc:
                _record_agi_degradation(
                    "agi_telemetry_substrate",
                    exc,
                    action="omitted substrate telemetry section from unified AGI telemetry",
                    severity="debug",
                )

        # 3. Free Energy Engine
        free_energy = ServiceContainer.get("free_energy_engine", default=None)
        if free_energy:
            try:
                telemetry["free_energy"] = {
                    "smoothed_free_energy": round(float(free_energy.smoothed_fe), 4),
                    "current_action": getattr(free_energy, "current_action", None),
                }
            except _AGI_RUNTIME_ERRORS as exc:
                _record_agi_degradation(
                    "agi_telemetry_free_energy",
                    exc,
                    action="omitted free-energy telemetry section from unified AGI telemetry",
                    severity="debug",
                )

        # 4. Dimensional Expansion
        expansion = ServiceContainer.get("dimensional_expansion", default=None)
        if expansion:
            try:
                status = expansion.get_status()
                telemetry["dimensional_expansion"] = {
                    "current_dim": status.get("current_dim"),
                    "expanded_count": status.get("expanded_count"),
                }
            except _AGI_RUNTIME_ERRORS as exc:
                _record_agi_degradation(
                    "agi_telemetry_dimensional_expansion",
                    exc,
                    action="omitted dimensional expansion telemetry section from unified AGI telemetry",
                    severity="debug",
                )

        # 5. Actuator Registry
        registry = ServiceContainer.get("actuator_registry", default=None)
        if registry:
            try:
                telemetry["actuators"] = {
                    "synthesized_count": len(getattr(registry, "synthesized_actuators", {})),
                    "total_count": len(getattr(registry, "actuators", {})),
                }
            except _AGI_RUNTIME_ERRORS as exc:
                _record_agi_degradation(
                    "agi_telemetry_actuators",
                    exc,
                    action="omitted actuator telemetry section from unified AGI telemetry",
                    severity="debug",
                )

        # 6. Digital Body Schema (Proprioception)
        if hasattr(self, "digital_body"):
            try:
                telemetry["digital_body"] = self.digital_body.get_state_dict()
            except _AGI_RUNTIME_ERRORS as exc:
                _record_agi_degradation(
                    "agi_telemetry_digital_body",
                    exc,
                    action="omitted digital body telemetry section from unified AGI telemetry",
                    severity="debug",
                )

        # 7. Transition Model (Causal Predictive World Model)
        if hasattr(self, "transition_model"):
            try:
                telemetry["transition_model"] = self.transition_model.get_state_dict()
            except _AGI_RUNTIME_ERRORS as exc:
                _record_agi_degradation(
                    "agi_telemetry_transition_model",
                    exc,
                    action="omitted transition model telemetry section from unified AGI telemetry",
                    severity="debug",
                )

        return telemetry


def get_agi_integration() -> AGIIntegrationLayer:
    """Thread-safe accessor for the AGIIntegrationLayer singleton."""
    global _agi_integration_instance
    if _agi_integration_instance is None:
        with _SINGLETON_LOCK:
            if _agi_integration_instance is None:
                _agi_integration_instance = AGIIntegrationLayer()
    return _agi_integration_instance

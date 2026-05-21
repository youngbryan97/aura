"""core/agi/agi_integration.py
===========================
AGI Integration Layer coordinates the dimensional expansion, actuator synthesis,
unified inference engine, and consciousness loop modules of Aura.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import numpy as np

from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Aura.AGI.Integration")

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

        # Local feedback & modulator instances
        from core.brain.homeostatic_modulator import HomeostaticModulator
        from core.brain.inference_feedback import InferenceFeedbackLoop

        self.modulator = HomeostaticModulator()
        self.feedback_loop = InferenceFeedbackLoop()

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
            self._running = True

            # Register in ServiceContainer
            ServiceContainer.register("agi_integration", self)

            # Spawn background tick loop using task tracker
            from core.utils.task_tracker import get_task_tracker

            tracker = get_task_tracker()
            self._loop_task = tracker.create_task(self._run_loop(), name="agi.integration_loop")
            logger.info("AGIIntegrationLayer started background tick task.")

    async def stop(self) -> None:
        """Stops the integration layer and performs final state saves."""
        with self._lock:
            if not self._running:
                return
            self._running = False

            if self._loop_task:
                self._loop_task.cancel()
                self._loop_task = None

            # Final save of logit projection weights
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
                # Clean up under-utilized dimensions
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

            return InferenceModulation(
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                logit_bias={},
                head_weights=np.ones(32, dtype=np.float32),
                urgency=0.5,
            )

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

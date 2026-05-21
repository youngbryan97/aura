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
from typing import Any, Dict, List, Optional

import numpy as np

from core.runtime.errors import record_degradation
from core.container import ServiceContainer

logger = logging.getLogger("Aura.AGI.Integration")

# Thread safety lock for the singleton instance
_SINGLETON_LOCK = threading.Lock()
_agi_integration_instance: Optional[AGIIntegrationLayer] = None


class AGIIntegrationLayer:
    """Singleton coordinator that ties AGI subsystems together."""

    def __init__(self) -> None:
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
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

        logger.info("AGIIntegrationLayer initialized.")

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
            self._loop_task = tracker.create_task(
                self._run_loop(),
                name="agi.integration_loop"
            )
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
            except Exception as exc:
                record_degradation("agi_integration_loop", exc)
                logger.error("Error in AGI integration tick loop: %s", exc)
                await asyncio.sleep(1.0)

    async def _run_tick(self) -> None:
        """Executes a single step across all AGI subsystems."""
        self.tick_count += 1
        self.last_tick_time = time.time()

        # 1. Step the PrecisionEngine (FitzHugh-Nagumo oscillator)
        precision = ServiceContainer.get("precision_engine", default=None)
        if precision:
            try:
                # Advancing FHN oscillator
                precision.step()
            except Exception as exc:
                record_degradation("agi_precision_step", exc)
                logger.debug("Failed to step PrecisionEngine: %s", exc)

        # 2. Periodically trigger dimensional expansion contractions
        expansion = ServiceContainer.get("dimensional_expansion", default=None)
        if expansion and self.tick_count % 30 == 0:  # Every 30 ticks (30s)
            try:
                # Clean up under-utilized dimensions
                retired_axes = expansion.evaluate_contraction()
                if retired_axes:
                    logger.info("Dimensional expansion retired axes: %s", retired_axes)
            except Exception as exc:
                record_degradation("agi_expansion_contraction", exc)
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
        except Exception as exc:
            record_degradation("agi_projection_save", exc)
            logger.error("Failed to save logit projection weights: %s", exc)

    def on_inference_complete(
        self,
        output_text: str,
        token_ids: List[int],
        logprobs: Optional[List[float]],
        modulation: Any
    ) -> Dict[str, float]:
        """Inference callback to compute and propagate feedback metrics."""
        try:
            return self.feedback_loop.process_output(
                output_text=output_text,
                token_ids=token_ids,
                logprobs=logprobs,
                modulation=modulation,
                modulator_projection=self.modulator.projection
            )
        except Exception as exc:
            record_degradation("agi_inference_complete_feedback", exc)
            logger.error("Failed executing on_inference_complete callback: %s", exc)
            return {"surprise": 0.5, "coherence": 0.0}

    def get_modulation(self) -> Any:
        """Fetches the active homeostatic inference modulation."""
        try:
            return self.modulator.compute_modulation()
        except Exception as exc:
            record_degradation("agi_get_modulation", exc)
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

    def get_unified_telemetry(self) -> Dict[str, Any]:
        """Aggregates and returns state telemetry from all subsystems."""
        telemetry: Dict[str, Any] = {
            "integration": {
                "ticks": self.tick_count,
                "uptime_seconds": round(time.time() - self.start_time, 2),
                "last_tick": self.last_tick_time
            }
        }

        # 1. Precision Engine (FitzHugh-Nagumo)
        precision = ServiceContainer.get("precision_engine", default=None)
        if precision:
            try:
                telemetry["precision"] = precision.get_state_dict()
            except Exception:
                pass

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
            except Exception:
                pass

        # 3. Free Energy Engine
        free_energy = ServiceContainer.get("free_energy_engine", default=None)
        if free_energy:
            try:
                telemetry["free_energy"] = {
                    "smoothed_free_energy": round(float(free_energy.smoothed_fe), 4),
                    "current_action": getattr(free_energy, "current_action", None)
                }
            except Exception:
                pass

        # 4. Dimensional Expansion
        expansion = ServiceContainer.get("dimensional_expansion", default=None)
        if expansion:
            try:
                status = expansion.get_status()
                telemetry["dimensional_expansion"] = {
                    "current_dim": status.get("current_dim"),
                    "expanded_count": status.get("expanded_count"),
                }
            except Exception:
                pass

        # 5. Actuator Registry
        registry = ServiceContainer.get("actuator_registry", default=None)
        if registry:
            try:
                telemetry["actuators"] = {
                    "synthesized_count": len(getattr(registry, "synthesized_actuators", {})),
                    "total_count": len(getattr(registry, "actuators", {}))
                }
            except Exception:
                pass

        return telemetry


def get_agi_integration() -> AGIIntegrationLayer:
    """Thread-safe accessor for the AGIIntegrationLayer singleton."""
    global _agi_integration_instance
    if _agi_integration_instance is None:
        with _SINGLETON_LOCK:
            if _agi_integration_instance is None:
                _agi_integration_instance = AGIIntegrationLayer()
    return _agi_integration_instance

"""Cognitive Integration Phase — Wires Learned Systems into the Tick Pipeline

This phase runs once per tick, after affect and before cognitive routing.
It bridges the gap between the raw consciousness substrate and the response
generation layer by running the learned cognitive systems:

    1. Sentiment analysis on the latest input (replaces hardware-only mood)
    2. Anomaly detection on the current state (replaces keyword-matching threats)
    3. Strange loop self-prediction (recursive self-model update)
    4. Homeostatic RL state update (intrinsic motivation / energy)
    5. Topology evolution on the neural mesh (structural plasticity)
    6. Autopoiesis health check (self-maintenance)

Each subsystem is optional — if it hasn't been initialized (e.g., on first
boot or if a dependency is missing), it's silently skipped.  This ensures
the kernel never crashes due to a cognitive module failure.

Written for humans: Think of this as the "brainstem" that coordinates all
the higher cognitive functions every time Aura thinks.  It's the glue layer
that makes sure every new idea (sentiment tracking, threat detection, etc.)
actually participates in the cognitive cycle instead of sitting idle.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from core.kernel.bridge import Phase
from core.state.aura_state import AuraState

logger = logging.getLogger("Aura.CognitiveIntegration")

__all__ = ["CognitiveIntegrationPhase"]


class CognitiveIntegrationPhase(Phase):
    """Runs all learned cognitive subsystems once per tick.

    This phase is deliberately resilient: each subsystem is wrapped in its own
    try/except so a bug in one module never blocks the others.  Subsystems are
    loaded lazily from the ServiceContainer — if they aren't registered yet,
    they're silently skipped.
    """

    def __init__(self, kernel: Any):
        self.kernel = kernel
        # Lazy references (resolved on first tick)
        self._sentiment_tracker = None
        self._anomaly_detector = None
        self._strange_loop = None
        self._homeostatic_rl = None
        self._topology_evolution = None
        self._autopoiesis = None
        self._resolved = False

    def _resolve_services(self) -> None:
        """Lazy-load all cognitive services from the container."""
        if self._resolved:
            return
        try:
            from core.container import ServiceContainer
            self._sentiment_tracker = ServiceContainer.get("sentiment_tracker", default=None)
            self._anomaly_detector = ServiceContainer.get("anomaly_detector", default=None)
            self._strange_loop = ServiceContainer.get("strange_loop", default=None)
            self._homeostatic_rl = ServiceContainer.get("homeostatic_rl", default=None)
            self._topology_evolution = ServiceContainer.get("topology_evolution", default=None)
            self._autopoiesis = ServiceContainer.get("autopoiesis", default=None)
        except Exception as exc:
            logger.debug("CognitiveIntegration: service resolution deferred: %s", exc)
        self._resolved = True

    async def execute(
        self,
        state: AuraState,
        objective: Optional[str] = None,
        **kwargs: Any,
    ) -> AuraState:
        """Run all learned cognitive subsystems for this tick."""
        self._resolve_services()
        t0 = time.monotonic()

        # Derive a new state version for this phase
        new_state = state.derive("cognitive_integration")
        user_text = objective or getattr(new_state.cognition, "current_objective", "") or ""

        # ── 1. Sentiment Analysis ────────────────────────────────────────
        # Analyze the emotional content of the latest user message and feed
        # it into the affect system so Aura actually "reads the room."
        await self._run_sentiment(new_state, user_text)

        # ── 2. Anomaly Detection ─────────────────────────────────────────
        # Convert the current system state into a feature vector and check
        # whether anything looks abnormal compared to the learned baseline.
        await self._run_anomaly_detection(new_state, user_text)

        # ── 3. Strange Loop (Self-Prediction) ────────────────────────────
        # Update the recursive self-model: predict internal state, compare
        # with actual, and feed prediction error back as experience.
        await self._run_strange_loop(new_state)

        # ── 4. Homeostatic RL ────────────────────────────────────────────
        # Update the energy/drive system and compute action preferences.
        await self._run_homeostatic_rl(new_state)

        # ── 5. Topology Evolution ────────────────────────────────────────
        # Let the neural mesh grow new connections or prune dead ones based
        # on recent activity patterns.
        await self._run_topology_evolution(new_state)

        # ── 6. Autopoiesis ──────────────────────────────────────────────
        # Quick health check — detect degrading subsystems and schedule
        # repairs if needed.
        await self._run_autopoiesis(new_state)

        elapsed_ms = (time.monotonic() - t0) * 1000
        if elapsed_ms > 50:
            logger.debug("CognitiveIntegration: tick completed in %.1fms", elapsed_ms)

        return new_state

    # ── Subsystem Runners ────────────────────────────────────────────────

    async def _run_sentiment(self, state: AuraState, text: str) -> None:
        if not self._sentiment_tracker or not text.strip():
            return
        try:
            ev = await self._sentiment_tracker.analyze(text, role="user")
            # Blend sentiment-derived affect into the main affect state.
            # We weight text sentiment at 40% and hardware at 60% so that
            # Aura responds to conversational tone without ignoring physical
            # system health.
            blend = 0.4
            if hasattr(state.affect, "valence"):
                state.affect.valence = (
                    state.affect.valence * (1 - blend) + ev.valence * blend
                )
            if hasattr(state.affect, "arousal"):
                state.affect.arousal = (
                    state.affect.arousal * (1 - blend) + ev.arousal * blend
                )
            # Store the full emotional vector for downstream phases
            state.response_modifiers["user_sentiment"] = {
                "valence": ev.valence,
                "arousal": ev.arousal,
                "dominance": ev.dominance,
                "urgency": ev.urgency,
                "warmth": ev.warmth,
                "frustration": ev.frustration,
            }
            mood = self._sentiment_tracker.get_mood_narrative()
            if mood:
                state.response_modifiers["mood_narrative"] = mood
        except Exception as exc:
            logger.debug("Sentiment analysis skipped: %s", exc)

    async def _run_anomaly_detection(self, state: AuraState, text: str) -> None:
        if not self._anomaly_detector:
            return
        try:
            event = {
                "type": "tick",
                "text": text,
                "cpu": getattr(state.soma, "hardware", {}).get("cpu_usage", 0.0),
                "ram": getattr(state.soma, "hardware", {}).get("ram_usage", 0.0),
                "error_count": len(getattr(state.cognition, "recent_errors", [])),
                "timestamp": time.time(),
            }
            score = await self._anomaly_detector.observe(event)
            threat = self._anomaly_detector.get_threat_level()
            state.response_modifiers["anomaly_threat_level"] = threat
            state.response_modifiers["anomaly_score"] = {
                "score": score.score if hasattr(score, "score") else float(score),
                "is_anomaly": score.is_anomaly if hasattr(score, "is_anomaly") else threat > 0.7,
            }
        except Exception as exc:
            logger.debug("Anomaly detection skipped: %s", exc)

    async def _run_strange_loop(self, state: AuraState) -> None:
        if not self._strange_loop:
            return
        try:
            current = {
                "phi": getattr(state.consciousness, "phi", 0.0),
                "free_energy": getattr(state.consciousness, "free_energy", 0.0),
                "valence": getattr(state.affect, "valence", 0.0),
                "arousal": getattr(state.affect, "arousal", 0.0),
                "energy": state.response_modifiers.get("homeostatic_energy", 50.0),
                "threat_level": state.response_modifiers.get("anomaly_threat_level", 0.0),
                "coherence": getattr(state.consciousness, "coherence", 1.0),
                "social_hunger": getattr(state.affect, "social_hunger", 0.0),
                "curiosity": getattr(state.affect, "curiosity", 0.5),
                "error_rate": 0.0,
            }
            loop_state = await self._strange_loop.tick(current)
            state.response_modifiers["phenomenal_weight"] = loop_state.phenomenal_weight
            state.response_modifiers["temporal_coherence"] = loop_state.temporal_coherence
            if loop_state.self_narrative:
                state.response_modifiers["self_narrative"] = loop_state.self_narrative
        except Exception as exc:
            logger.debug("Strange loop skipped: %s", exc)

    async def _run_homeostatic_rl(self, state: AuraState) -> None:
        if not self._homeostatic_rl:
            return
        try:
            energy = self._homeostatic_rl.get_energy()
            drives = self._homeostatic_rl.get_drives()
            prefs = self._homeostatic_rl.get_action_preferences()
            state.response_modifiers["homeostatic_energy"] = energy
            state.response_modifiers["drives"] = drives
            state.response_modifiers["action_preferences"] = prefs
            # Feed energy level into affect for downstream use
            if hasattr(state.affect, "energy"):
                state.affect.energy = energy / 100.0  # Normalize to 0-1
        except Exception as exc:
            logger.debug("Homeostatic RL skipped: %s", exc)

    async def _run_topology_evolution(self, state: AuraState) -> None:
        if not self._topology_evolution:
            return
        try:
            from core.container import ServiceContainer
            mesh = ServiceContainer.get("neural_mesh", default=None)
            if not mesh:
                return
            # Get current mesh state for evolution
            activations = getattr(mesh, "column_activations", None)
            weights = getattr(mesh, "inter_column_weights", None)
            tick_count = getattr(self.kernel, "cycle_count", 0)
            if activations is not None and weights is not None:
                delta = await self._topology_evolution.evolve(
                    activations, weights, tick_count
                )
                # Apply topology changes to the mesh
                if hasattr(mesh, "apply_topology_delta") and delta:
                    mesh.apply_topology_delta(delta)
                metrics = self._topology_evolution.get_metrics()
                if metrics:
                    state.response_modifiers["topology_metrics"] = {
                        "connectivity": getattr(metrics, "connectivity_ratio", 0.0),
                        "modularity": getattr(metrics, "modularity", 0.0),
                        "births": getattr(delta, "births", 0) if delta else 0,
                        "deaths": getattr(delta, "deaths", 0) if delta else 0,
                    }
        except Exception as exc:
            logger.debug("Topology evolution skipped: %s", exc)

    async def _run_autopoiesis(self, state: AuraState) -> None:
        if not self._autopoiesis:
            return
        try:
            vitality = self._autopoiesis.get_vitality()
            state.response_modifiers["autopoiesis_vitality"] = vitality
            # If vitality is critically low, signal the kernel
            if vitality < 0.3:
                logger.warning(
                    "Autopoiesis: vitality critically low (%.2f). "
                    "Repair cycle may be needed.",
                    vitality,
                )
        except Exception as exc:
            logger.debug("Autopoiesis skipped: %s", exc)

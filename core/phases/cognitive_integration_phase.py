"""Cognitive Integration Phase — Wires ALL Learned Systems into the Tick Pipeline

This phase runs once per tick, after affect and before cognitive routing.
It bridges the gap between the raw consciousness substrate and the response
generation layer by running every learned cognitive and artificial life system:

    LEARNED COGNITIVE SYSTEMS:
    1. Sentiment analysis on the latest input (replaces hardware-only mood)
    2. Anomaly detection on the current state (replaces keyword-matching threats)
    3. Adaptive immunity on live antigens (clonal repair ecology)
    4. Strange loop self-prediction (recursive self-model update)
    5. Homeostatic RL state update (intrinsic motivation / energy)
    6. Topology evolution on the neural mesh (structural plasticity)
    7. Autopoiesis health check (self-maintenance)

    ARTIFICIAL LIFE SYSTEMS (from Avida, Tierra, Lenia, EcoSim, Evochora):
    8. Criticality regulation — tune neural dynamics toward edge-of-chaos
    9. ALife dynamics — Lenia kernels, entropy tracking, differential CPU allocation
   10. ALife extensions — pattern replication, speciation, toroidal topology,
       thermodynamic costs, ownership-based access costs
   11. Endogenous fitness — survival-based evolution (Tierra) + behavioral rules (EcoSim)

Each subsystem is optional — if it hasn't been initialized (e.g., on first
boot or if a dependency is missing), it's silently skipped.

Written for humans: Think of this as the "brainstem" that coordinates all
cognitive functions every time Aura thinks. The ALife systems give Aura the
properties of a living organism: she self-organizes toward criticality,
evolves her own neural wiring, maintains dual thermodynamic constraints
(energy AND entropy), replicates successful patterns, and forms functional
species within her cortical columns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

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
        self._autonomous_resilience_mesh = None
        self._adaptive_immune_system = None
        self._strange_loop = None
        self._homeostatic_rl = None
        self._topology_evolution = None
        self._autopoiesis = None
        # ALife systems
        self._criticality_regulator = None
        self._alife_dynamics = None
        self._alife_extensions = None
        self._endogenous_fitness = None
        self._resolved = False

    def _resolve_services(self) -> None:
        """Lazy-load all cognitive services from the container."""
        if self._resolved:
            return
        try:
            from core.container import ServiceContainer
            self._sentiment_tracker = ServiceContainer.get("sentiment_tracker", default=None)
            self._anomaly_detector = ServiceContainer.get("anomaly_detector", default=None)
            self._autonomous_resilience_mesh = ServiceContainer.get("autonomous_resilience_mesh", default=None)
            self._adaptive_immune_system = ServiceContainer.get("adaptive_immune_system", default=None)
            self._strange_loop = ServiceContainer.get("strange_loop", default=None)
            self._homeostatic_rl = ServiceContainer.get("homeostatic_rl", default=None)
            self._topology_evolution = ServiceContainer.get("topology_evolution", default=None)
            self._autopoiesis = ServiceContainer.get("autopoiesis", default=None)
            # ALife systems
            self._criticality_regulator = ServiceContainer.get("criticality_regulator", default=None)
            self._alife_dynamics = ServiceContainer.get("alife_dynamics", default=None)
            self._alife_extensions = ServiceContainer.get("alife_extensions", default=None)
            self._endogenous_fitness = ServiceContainer.get("endogenous_fitness", default=None)
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
        anomaly_score = await self._run_anomaly_detection(new_state, user_text)

        # ── 3. Autonomous Resilience Mesh ────────────────────────────────
        # Audit runtime, service wiring, security signals, and static code
        # risks so the immune system has more truthful threat context.
        resilience_report = await self._run_autonomous_resilience(new_state, user_text)

        # ── 4. Adaptive Immunity ─────────────────────────────────────────
        # Convert the current state into a richer antigen that the adaptive
        # immune population can learn against and emit bounded effectors for.
        await self._run_adaptive_immunity(new_state, user_text, anomaly_score, resilience_report)

        # ── 5. Strange Loop (Self-Prediction) ────────────────────────────
        # Update the recursive self-model: predict internal state, compare
        # with actual, and feed prediction error back as experience.
        await self._run_strange_loop(new_state)

        # ── 6. Homeostatic RL ────────────────────────────────────────────
        # Update the energy/drive system and compute action preferences.
        await self._run_homeostatic_rl(new_state)

        # ── 7. Topology Evolution ────────────────────────────────────────
        # Let the neural mesh grow new connections or prune dead ones based
        # on recent activity patterns.
        await self._run_topology_evolution(new_state)

        # ── 8. Autopoiesis ──────────────────────────────────────────────
        # Quick health check — detect degrading subsystems and schedule
        # repairs if needed.
        await self._run_autopoiesis(new_state)

        # ── 9. Criticality Regulation (Wolfram/CA research) ─────────────
        # Tune the neural mesh toward the edge of chaos — the critical
        # point where computation is richest. Adjusts gain, noise, and
        # excitation/inhibition balance via a PID controller.
        await self._run_criticality(new_state)

        # ── 10. ALife Dynamics (Lenia + Evochora + Avida) ───────────────
        # Lenia continuous convolution kernels for inter-column coupling,
        # entropy tracking (Evochora's dual thermodynamic constraint),
        # and differential CPU allocation (Avida's compute-as-reward).
        await self._run_alife_dynamics(new_state)

        # ── 11. ALife Extensions (pattern replication, speciation, etc.) ─
        # Autopoietic pattern replication (Avida), speciation-driven
        # column specialization (EcoSim), toroidal wrapping, thermodynamic
        # operation costs, and ownership-based access costs (Evochora).
        await self._run_alife_extensions(new_state)

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

    async def _run_anomaly_detection(self, state: AuraState, text: str):
        if not self._anomaly_detector:
            return None
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
            return score
        except Exception as exc:
            logger.debug("Anomaly detection skipped: %s", exc)
            return None

    async def _run_autonomous_resilience(self, state: AuraState, text: str) -> Dict[str, Any] | None:
        if not self._autonomous_resilience_mesh:
            return None
        try:
            report = await self._autonomous_resilience_mesh.tick(
                user_text=text,
                state_snapshot={
                    "error_count": len(getattr(state.cognition, "recent_errors", [])),
                    "anomaly_threat_level": state.response_modifiers.get("anomaly_threat_level", 0.0),
                },
            )
            state.response_modifiers["autonomous_resilience"] = report
            return report
        except Exception as exc:
            logger.debug("Autonomous resilience skipped: %s", exc)
            return None

    async def _run_adaptive_immunity(
        self,
        state: AuraState,
        text: str,
        anomaly_score: Any | None,
        resilience_report: Dict[str, Any] | None,
    ) -> None:
        if not self._adaptive_immune_system:
            return
        try:
            resilience_threat = float((resilience_report or {}).get("threat_score", 0.0) or 0.0)
            event = {
                "type": "tick",
                "text": text,
                "source": "cognitive_integration",
                "subsystem": "cognitive_integration",
                "cpu": getattr(state.soma, "hardware", {}).get("cpu_usage", 0.0),
                "ram": getattr(state.soma, "hardware", {}).get("ram_usage", 0.0),
                "resource_pressure": max(
                    0.0,
                    min(
                        1.0,
                        max(
                            float(getattr(state.soma, "hardware", {}).get("cpu_usage", 0.0)) / 100.0,
                            float(getattr(state.soma, "hardware", {}).get("ram_usage", 0.0)) / 100.0,
                        ),
                    ),
                ),
                "error_count": len(getattr(state.cognition, "recent_errors", [])),
                "threat_probability": resilience_threat,
                "timestamp": time.time(),
            }
            immune_response = await self._adaptive_immune_system.observe_event(
                event,
                anomaly_score=anomaly_score,
                state_snapshot={
                    "health_pressure": state.response_modifiers.get("anomaly_threat_level", 0.0),
                    "resilience": resilience_report or {},
                },
            )
            state.response_modifiers["adaptive_immunity"] = immune_response.to_dict()
            selected = immune_response.selected_artifact
            if selected is not None:
                state.response_modifiers["adaptive_effector"] = {
                    "kind": selected.kind.value,
                    "component": selected.component,
                    "confidence": selected.confidence,
                    "executed": selected.executed,
                    "success": selected.success,
                }
            if immune_response.diagnostic_verdict:
                state.response_modifiers["adaptive_immune_verdict"] = immune_response.diagnostic_verdict
            if immune_response.coverage_report:
                state.response_modifiers["adaptive_immune_coverage"] = immune_response.coverage_report
            if immune_response.verification_report:
                state.response_modifiers["adaptive_immune_verification"] = immune_response.verification_report
            if resilience_report and resilience_report.get("immune_events"):
                secondary_responses = []
                for issue_event in list(resilience_report.get("immune_events", []))[:2]:
                    extra_response = await self._adaptive_immune_system.observe_event(
                        issue_event,
                        anomaly_score=anomaly_score,
                        state_snapshot={
                            "health_pressure": state.response_modifiers.get("anomaly_threat_level", 0.0),
                            "resilience": resilience_report,
                        },
                    )
                    secondary_responses.append(
                        {
                            "antigen": extra_response.antigen.to_dict(),
                            "verdict": extra_response.diagnostic_verdict,
                            "selected_artifact": (
                                extra_response.selected_artifact.to_dict()
                                if extra_response.selected_artifact
                                else None
                            ),
                        }
                    )
                if secondary_responses:
                    state.response_modifiers["autonomous_resilience_immune"] = secondary_responses
        except Exception as exc:
            logger.debug("Adaptive immunity skipped: %s", exc)

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
            if vitality < 0.3:
                logger.warning(
                    "Autopoiesis: vitality critically low (%.2f). "
                    "Repair cycle may be needed.",
                    vitality,
                )
        except Exception as exc:
            logger.debug("Autopoiesis skipped: %s", exc)

    # ── ALife System Runners ─────────────────────────────────────────────

    async def _run_criticality(self, state: AuraState) -> None:
        """Criticality Regulator — tunes neural mesh toward edge-of-chaos.

        Biological brains self-organize toward the critical point between
        order and chaos, where computation is richest. This regulator
        measures the branching ratio and avalanche statistics of the neural
        mesh and adjusts gain, noise, and excitation/inhibition balance
        to drive the system toward criticality.
        """
        if not self._criticality_regulator:
            return
        try:
            from core.container import ServiceContainer
            mesh = ServiceContainer.get("neural_mesh", default=None)
            if not mesh:
                return
            activations = getattr(mesh, "column_activations", None)
            weights = getattr(mesh, "inter_column_weights", None)
            if activations is None or weights is None:
                return

            crit_state = await self._criticality_regulator.tick(activations, weights)
            score = self._criticality_regulator.get_criticality_score()
            adjustments = self._criticality_regulator.get_adjustments()

            state.response_modifiers["criticality_score"] = score
            state.response_modifiers["branching_ratio"] = crit_state.branching_ratio
            state.response_modifiers["avalanche_exponent"] = crit_state.avalanche_exponent

            # Apply adjustments to the mesh
            if hasattr(mesh, "set_modulatory_state"):
                mesh.set_modulatory_state(
                    modulatory_gain=adjustments.get("gain", 1.0),
                    modulatory_noise=adjustments.get("noise", 1.0),
                )

            # Feed E/I ratio to neurochemical system
            ei_ratio = adjustments.get("ei_ratio", 1.0)
            try:
                neurochems = ServiceContainer.get("neurochemical_system", default=None)
                if neurochems and hasattr(neurochems, "set_ei_target"):
                    neurochems.set_ei_target(ei_ratio)
            except Exception:
                pass
        except Exception as exc:
            logger.debug("Criticality regulation skipped: %s", exc)

    async def _run_alife_dynamics(self, state: AuraState) -> None:
        """ALife Dynamics — Lenia kernels, entropy tracking, CPU allocation.

        Lenia: Replaces fixed inter-column connectivity with continuous
        convolution kernels that create richer emergent dynamics.
        Entropy: Adds Evochora's dual thermodynamic constraint (energy + entropy).
        CPU allocation: Avida's innovation — columns earn compute time through
        useful contribution to the global workspace.
        """
        if not self._alife_dynamics:
            return
        try:
            from core.container import ServiceContainer
            mesh = ServiceContainer.get("neural_mesh", default=None)
            if not mesh:
                return
            activations = getattr(mesh, "column_activations", None)
            weights = getattr(mesh, "inter_column_weights", None)
            projection = getattr(mesh, "projection_weights", None)
            if activations is None or weights is None:
                return

            alife_state = await self._alife_dynamics.tick(
                activations, weights, projection
            )

            # Store entropy and credit info in state
            if hasattr(alife_state, "entropy"):
                state.response_modifiers["entropy"] = alife_state.entropy
                state.response_modifiers["entropy_pressure"] = alife_state.entropy_pressure
            if hasattr(alife_state, "compute_credits"):
                state.response_modifiers["compute_credits_gini"] = (
                    self._alife_dynamics.get_status().get("gini_coefficient", 0.0)
                    if hasattr(self._alife_dynamics, "get_status") else 0.0
                )
        except Exception as exc:
            logger.debug("ALife dynamics skipped: %s", exc)

    async def _run_alife_extensions(self, state: AuraState) -> None:
        """ALife Extensions — replication, speciation, toroidal topology, costs.

        Pattern replication (Avida): Successful column configs replicate to
        struggling neighbors. Speciation (EcoSim): Columns form functional
        species. Toroidal wrapping: Mesh boundaries connect. Thermodynamic
        costs (Evochora): Per-operation energy/entropy charges. Ownership
        costs: Cross-subsystem access is more expensive.
        """
        if not self._alife_extensions:
            return
        try:
            from core.container import ServiceContainer
            mesh = ServiceContainer.get("neural_mesh", default=None)
            mesh_state = {}
            if mesh:
                mesh_state = {
                    "column_activations": getattr(mesh, "column_activations", None),
                    "inter_column_weights": getattr(mesh, "inter_column_weights", None),
                }
            tick_count = getattr(self.kernel, "cycle_count", 0)
            ext_state = await self._alife_extensions.tick(
                mesh_state=mesh_state,
                evolution_state={},
                tick_count=tick_count,
            )
            if hasattr(ext_state, "species_info") and ext_state.species_info:
                state.response_modifiers["species_count"] = getattr(
                    ext_state.species_info, "species_count", 0
                )
        except Exception as exc:
            logger.debug("ALife extensions skipped: %s", exc)

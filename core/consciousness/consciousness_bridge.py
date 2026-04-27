"""core/consciousness/consciousness_bridge.py — The Consciousness Bridge

Wires the seven new subsystems into Aura's existing consciousness stack.

This module:
  1. Instantiates and starts all bridge components
  2. Cross-wires references between components
  3. Registers everything into ServiceContainer
  4. Hooks into the CognitiveHeartbeat tick cycle to drive continuous updates
  5. Provides graceful degradation (any component can fail independently)
  6. Exposes a unified status API

Boot order (respects dependencies):
  1. NeuralMesh           (foundation — everything else builds on this)
  2. NeurochemicalSystem  (modulates mesh + everything downstream)
  3. EmbodiedInteroception (feeds mesh + chemicals)
  4. OscillatoryBinding   (temporal binding, needs mesh for phase reports)
  5. SomaticMarkerGate    (needs mesh + interoception + chemicals)
  6. UnifiedField         (needs all of the above)
  7. SubstrateEvolution   (needs mesh + binding + workspace + substrate)

After boot, a continuous integration loop runs at 10 Hz that:
  - Reads the NeuralMesh executive projection and pushes it into the LiquidSubstrate
  - Reads the LiquidSubstrate state and pushes it into the UnifiedField
  - Reads neurochemical state and pushes it into the UnifiedField
  - Reads interoceptive state and pushes it into the UnifiedField
  - Reads binding state and pushes it into the UnifiedField
  - Reports phases to OscillatoryBinding from each subsystem
  - Applies somatic marker evaluation to GWT candidates
  - Applies unified field back-pressure to input subsystems
  - Pushes neurochemical mood into the substrate's VAD indices
"""
from __future__ import annotations

from core.utils.task_tracker import get_task_tracker

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from core.container import ServiceContainer

logger = logging.getLogger("Consciousness.Bridge")


class ConsciousnessBridge:
    """Wires the consciousness bridge into the existing stack.

    Usage in ConsciousnessSystem.start():
        self.bridge = ConsciousnessBridge(self)
        await self.bridge.start()

    The bridge runs autonomously after start() — no external tick needed.
    """

    _INTEGRATION_HZ = 10.0  # 10 Hz integration loop

    def __init__(self, consciousness_system: Any):
        self._cs = consciousness_system  # ConsciousnessSystem reference
        self._orch = consciousness_system.orch  # RobustOrchestrator

        # Components (initialized during start)
        self.neural_mesh = None
        self.neurochemical = None
        self.interoception = None
        self.oscillatory_binding = None
        self.somatic_gate = None
        self.unified_field = None
        self.substrate_evolution = None
        self.substrate_authority = None

        # Integration loop
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tick_count: int = 0
        self._start_time: float = 0.0
        self._boot_errors: list = []

        logger.info("ConsciousnessBridge created")

    # ── Boot ─────────────────────────────────────────────────────────────

    async def start(self):
        """Boot all bridge components in dependency order."""
        if self._running:
            return
        self._start_time = time.time()

        # ── 1. Neural Mesh ───────────────────────────────────────────
        try:
            from .neural_mesh import NeuralMesh
            self.neural_mesh = NeuralMesh()
            await self.neural_mesh.start()
            ServiceContainer.register_instance("neural_mesh", self.neural_mesh)
            logger.info("🧬 Bridge Layer 1: NeuralMesh ONLINE (4096 neurons)")
        except Exception as e:
            logger.error("Failed to boot NeuralMesh: %s", e, exc_info=True)
            self._boot_errors.append(("neural_mesh", str(e)))

        # ── 2. Neurochemical System ──────────────────────────────────
        try:
            from .neurochemical_system import NeurochemicalSystem
            self.neurochemical = NeurochemicalSystem()
            # Wire to mesh
            if self.neural_mesh:
                self.neurochemical._mesh_ref = self.neural_mesh
            # Wire to GWT
            gw = ServiceContainer.get("global_workspace", default=None) or \
                 getattr(self._cs, "global_workspace", None)
            if gw:
                self.neurochemical._workspace_ref = gw
            await self.neurochemical.start()
            ServiceContainer.register_instance("neurochemical_system", self.neurochemical)
            logger.info("🧬 Bridge Layer 2: NeurochemicalSystem ONLINE (8 modulators)")
        except Exception as e:
            logger.error("Failed to boot NeurochemicalSystem: %s", e, exc_info=True)
            self._boot_errors.append(("neurochemical", str(e)))

        # ── 3. Embodied Interoception ────────────────────────────────
        try:
            from .embodied_interoception import EmbodiedInteroception
            self.interoception = EmbodiedInteroception()
            # Wire to mesh and chemicals
            if self.neural_mesh:
                self.interoception._mesh_ref = self.neural_mesh
            if self.neurochemical:
                self.interoception._neurochemical_ref = self.neurochemical
            await self.interoception.start()
            ServiceContainer.register_instance("embodied_interoception", self.interoception)
            logger.info("🧬 Bridge Layer 3: EmbodiedInteroception ONLINE (8 channels)")
        except Exception as e:
            logger.error("Failed to boot EmbodiedInteroception: %s", e, exc_info=True)
            self._boot_errors.append(("interoception", str(e)))

        # ── 4. Oscillatory Binding ───────────────────────────────────
        try:
            from .oscillatory_binding import OscillatoryBinding
            self.oscillatory_binding = OscillatoryBinding()
            await self.oscillatory_binding.start()
            ServiceContainer.register_instance("oscillatory_binding", self.oscillatory_binding)
            logger.info("🧬 Bridge Layer 4: OscillatoryBinding ONLINE (γ=40Hz, θ=8Hz)")
        except Exception as e:
            logger.error("Failed to boot OscillatoryBinding: %s", e, exc_info=True)
            self._boot_errors.append(("oscillatory_binding", str(e)))

        # ── 5. Somatic Marker Gate ───────────────────────────────────
        try:
            from .somatic_marker_gate import SomaticMarkerGate
            self.somatic_gate = SomaticMarkerGate()
            if self.neural_mesh:
                self.somatic_gate._mesh_ref = self.neural_mesh
            if self.interoception:
                self.somatic_gate._interoception_ref = self.interoception
            if self.neurochemical:
                self.somatic_gate._neurochemical_ref = self.neurochemical
            ServiceContainer.register_instance("somatic_marker_gate", self.somatic_gate)
            logger.info("🧬 Bridge Layer 5: SomaticMarkerGate ONLINE")
        except Exception as e:
            logger.error("Failed to boot SomaticMarkerGate: %s", e, exc_info=True)
            self._boot_errors.append(("somatic_gate", str(e)))

        # ── 6. Unified Field ─────────────────────────────────────────
        try:
            from .unified_field import UnifiedField
            self.unified_field = UnifiedField()
            if self.oscillatory_binding:
                self.unified_field._binding_ref = self.oscillatory_binding
            await self.unified_field.start()
            ServiceContainer.register_instance("unified_field", self.unified_field)
            logger.info("🧬 Bridge Layer 6: UnifiedField ONLINE (256-d experiential field)")
        except Exception as e:
            logger.error("Failed to boot UnifiedField: %s", e, exc_info=True)
            self._boot_errors.append(("unified_field", str(e)))

        # ── 7. Substrate Evolution ───────────────────────────────────
        try:
            from .substrate_evolution import SubstrateEvolution
            self.substrate_evolution = SubstrateEvolution()
            if self.neural_mesh:
                self.substrate_evolution._mesh_ref = self.neural_mesh
            if self.oscillatory_binding:
                self.substrate_evolution._binding_ref = self.oscillatory_binding
            gw = ServiceContainer.get("global_workspace", default=None) or \
                 getattr(self._cs, "global_workspace", None)
            if gw:
                self.substrate_evolution._workspace_ref = gw
            substrate = ServiceContainer.get("liquid_substrate", default=None) or \
                        ServiceContainer.get("conscious_substrate", default=None) or \
                        getattr(self._cs, "liquid_substrate", None)
            if substrate:
                self.substrate_evolution._substrate_ref = substrate
            await self.substrate_evolution.start()
            ServiceContainer.register_instance("substrate_evolution", self.substrate_evolution)
            logger.info("🧬 Bridge Layer 7: SubstrateEvolution ONLINE (pop=%d)",
                        self.substrate_evolution.cfg.population_size)
        except Exception as e:
            logger.error("Failed to boot SubstrateEvolution: %s", e, exc_info=True)
            self._boot_errors.append(("substrate_evolution", str(e)))

        # ── 8. Substrate Authority (THE SINGLE NARROW WAIST) ────────
        try:
            from .substrate_authority import SubstrateAuthority
            self.substrate_authority = SubstrateAuthority()
            if self.unified_field:
                self.substrate_authority._field_ref = self.unified_field
            if self.somatic_gate:
                self.substrate_authority._somatic_ref = self.somatic_gate
            if self.neurochemical:
                self.substrate_authority._neurochemical_ref = self.neurochemical
            if self.interoception:
                self.substrate_authority._interoception_ref = self.interoception
            ServiceContainer.register_instance("substrate_authority", self.substrate_authority)
            logger.info("🧬 Bridge Layer 8: SubstrateAuthority ONLINE (mandatory gate)")
        except Exception as e:
            logger.error("Failed to boot SubstrateAuthority: %s", e, exc_info=True)
            self._boot_errors.append(("substrate_authority", str(e)))
            self.substrate_authority = None

        # ── 9. Unified Will (THE SINGLE LOCUS OF DECISION AUTHORITY) ─
        try:
            from core.will import get_will
            self.unified_will = get_will()
            await self.unified_will.start()
            logger.info("🧬 Bridge Layer 9: UnifiedWill ONLINE (single locus of authority)")
        except Exception as e:
            logger.error("Failed to boot UnifiedWill: %s", e, exc_info=True)
            self._boot_errors.append(("unified_will", str(e)))

        # ── Start integration loop ───────────────────────────────────
        self._running = True
        self._task = get_task_tracker().create_task(self._integration_loop(), name="ConsciousnessBridge")

        # ── Hook into GWT for somatic gating ─────────────────────────
        self._hook_somatic_into_gwt()

        # ── Hook neurochemical events into existing systems ──────────
        self._hook_neurochemical_events()

        boot_count = 8 - len(self._boot_errors)
        logger.info(
            "🧬 ConsciousnessBridge ONLINE — %d/8 layers active, %d errors (Will: single locus)",
            boot_count, len(self._boot_errors),
        )
        if self._boot_errors:
            for name, err in self._boot_errors:
                logger.warning("  ⚠ %s failed: %s", name, err)

    async def stop(self):
        """Gracefully stop all bridge components."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Stop in reverse order
        for component, name in [
            (self.substrate_evolution, "SubstrateEvolution"),
            (self.unified_field, "UnifiedField"),
            (self.oscillatory_binding, "OscillatoryBinding"),
            (self.interoception, "EmbodiedInteroception"),
            (self.neurochemical, "NeurochemicalSystem"),
            (self.neural_mesh, "NeuralMesh"),
        ]:
            if component and hasattr(component, "stop"):
                try:
                    await component.stop()
                except Exception as e:
                    logger.debug("Error stopping %s: %s", name, e)

        logger.info("🧬 ConsciousnessBridge OFFLINE")

    # ── Integration loop ─────────────────────────────────────────────────

    async def _integration_loop(self):
        """The continuous cross-wiring loop.

        This is the heartbeat of the bridge — it moves data between
        all subsystems every tick, ensuring tight causal coupling.
        """
        interval = 1.0 / self._INTEGRATION_HZ

        try:
            while self._running:
                t0 = time.time()
                try:
                    await asyncio.to_thread(self._integration_tick)
                except Exception as e:
                    logger.error("Bridge integration error: %s", e, exc_info=True)

                self._tick_count += 1
                elapsed = time.time() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:
            pass

    def _integration_tick(self):
        """One integration step — cross-wires all subsystems."""

        # ── 1. Mesh → Substrate (executive projection IS the core) ────
        # The mesh doesn't "inform" the substrate — the mesh projection
        # IS the substrate's primary drive. Coupling is strong (0.35),
        # not advisory (was 0.15). The 64-neuron substrate is now the
        # executive summary of the 4096-neuron cortical mesh.
        substrate = self._get_substrate()
        if self.neural_mesh and substrate:
            try:
                projection = self.neural_mesh.get_executive_projection()
                with substrate.sync_lock:
                    substrate.x = np.clip(
                        substrate.x * 0.65 + projection * 0.35,
                        -1.0, 1.0
                    ).astype(np.float64)
            except Exception as e:
                logger.debug("Mesh→Substrate integration failed: %s", e)

        # ── 2. Substrate → UnifiedField ──────────────────────────────
        if substrate and self.unified_field:
            try:
                state = substrate.x.copy()
                self.unified_field.receive_substrate(state[:64].astype(np.float32))
            except Exception as e:
                logger.debug("Substrate→Field failed: %s", e)

        # ── 3. Mesh → UnifiedField ───────────────────────────────────
        if self.neural_mesh and self.unified_field:
            try:
                proj = self.neural_mesh.get_executive_projection()
                self.unified_field.receive_mesh(proj)
            except Exception as e:
                logger.debug("Mesh→Field failed: %s", e)

        # ── 4. Chemicals → UnifiedField ──────────────────────────────
        if self.neurochemical and self.unified_field:
            try:
                chem_vec = np.array([
                    self.neurochemical.chemicals[name].effective
                    for name in [
                        "dopamine", "serotonin", "norepinephrine", "acetylcholine",
                        "gaba", "endorphin", "oxytocin", "cortisol"
                    ]
                ], dtype=np.float32)
                self.unified_field.receive_chemicals(chem_vec)
            except Exception as e:
                logger.debug("Chemicals→Field failed: %s", e)

        # ── 5. Binding → UnifiedField ────────────────────────────────
        if self.oscillatory_binding and self.unified_field:
            try:
                bind_vec = np.array([
                    self.oscillatory_binding.get_psi(),
                    self.oscillatory_binding.get_gamma_amplitude(),
                    self.oscillatory_binding.get_theta_phase() / (2 * np.pi),  # normalize
                    self.oscillatory_binding.get_coupling_strength(),
                ], dtype=np.float32)
                self.unified_field.receive_binding(bind_vec)
            except Exception as e:
                logger.debug("Binding→Field failed: %s", e)

        # ── 6. Interoception → UnifiedField ──────────────────────────
        if self.interoception and self.unified_field:
            try:
                intero = self.interoception.get_interoceptive_state()
                intero_vec = np.array(list(intero.values()), dtype=np.float32)
                self.unified_field.receive_interoception(intero_vec)
            except Exception as e:
                logger.debug("Interoception→Field failed: %s", e)

        # ── 7. Phase reports to OscillatoryBinding ───────────────────
        if self.oscillatory_binding:
            try:
                # Mesh tier energies → oscillator modulation (BIDIRECTIONAL COUPLING)
                # This closes the loop: mesh activity drives oscillator dynamics,
                # oscillators drive binding, binding feeds unified field,
                # unified field back-pressures mesh. Genuine emergent coupling.
                if self.neural_mesh:
                    from .neural_mesh import CorticalTier
                    tier_energy = {
                        "sensory": self.neural_mesh.get_tier_energy(CorticalTier.SENSORY),
                        "association": self.neural_mesh.get_tier_energy(CorticalTier.ASSOCIATION),
                        "executive": self.neural_mesh.get_tier_energy(CorticalTier.EXECUTIVE),
                    }
                    if hasattr(self.oscillatory_binding, "receive_mesh_energy"):
                        self.oscillatory_binding.receive_mesh_energy(tier_energy)

                # Mesh reports phase based on executive energy
                if self.neural_mesh:
                    exec_energy = self.neural_mesh.get_tier_energy(CorticalTier.EXECUTIVE)
                    self.oscillatory_binding.compute_subsystem_phase(exec_energy, "neural_mesh")

                # Substrate reports phase based on current phi
                if substrate:
                    phi = getattr(substrate, '_current_phi', 0.0)
                    self.oscillatory_binding.compute_subsystem_phase(
                        min(1.0, phi / 10.0), "liquid_substrate"
                    )

                # Chemicals report phase based on arousal
                if self.neurochemical:
                    mood = self.neurochemical.get_mood_vector()
                    self.oscillatory_binding.compute_subsystem_phase(
                        mood.get("arousal", 0.5), "neurochemical"
                    )

                # Interoception reports phase based on metabolic load
                if self.interoception:
                    ml = self.interoception.channels["metabolic_load"].smoothed
                    self.oscillatory_binding.compute_subsystem_phase(ml, "interoception")

                # GWT reports phase based on ignition
                gw = self._get_workspace()
                if gw:
                    self.oscillatory_binding.compute_subsystem_phase(
                        gw.ignition_level, "global_workspace"
                    )

                # Unified field reports its own coherence as phase
                if self.unified_field:
                    self.oscillatory_binding.compute_subsystem_phase(
                        self.unified_field.get_coherence(), "unified_field"
                    )
            except Exception as e:
                logger.debug("Phase reporting failed: %s", e)

        # ── 8. Chemicals → Substrate VAD (strong coupling) ────────────
        # Neurochemical mood IS the substrate's affective state.
        # The chemistry doesn't suggest — it determines. 0.30 coupling.
        if self.neurochemical and substrate:
            try:
                mood = self.neurochemical.get_mood_vector()
                valence = mood.get("valence", 0.0)
                arousal = mood.get("arousal", 0.0)
                stress = mood.get("stress", 0.0)
                with substrate.sync_lock:
                    substrate.x[substrate.idx_valence] = (
                        substrate.x[substrate.idx_valence] * 0.70 + valence * 0.30
                    )
                    substrate.x[substrate.idx_arousal] = (
                        substrate.x[substrate.idx_arousal] * 0.70 + arousal * 0.30
                    )
                    # Stress directly drives frustration
                    substrate.x[substrate.idx_frustration] = (
                        substrate.x[substrate.idx_frustration] * 0.75 + stress * 0.25
                    )
            except Exception as e:
                logger.debug("Chemicals→Substrate VAD failed: %s", e)

        # ── 9. Field back-pressure (ENFORCEABLE) ─────────────────────
        # The unified field's state directly modulates all input subsystems.
        # Not advisory — these are hard-coupled dynamics.
        if self.unified_field:
            try:
                bp = self.unified_field.get_back_pressure()
                coherence = self.unified_field.get_coherence()

                # Mesh gain: field coherence directly scales mesh responsiveness
                # Low coherence → damped mesh (prevent further fragmentation)
                # High coherence → normal mesh operation
                if self.neural_mesh:
                    if coherence < 0.30:
                        # Crisis: strongly dampen mesh to allow recovery
                        self.neural_mesh._modulatory_gain = max(0.3, self.neural_mesh._modulatory_gain * 0.85)
                        self.neural_mesh._modulatory_noise = max(0.1, self.neural_mesh._modulatory_noise * 0.8)
                    else:
                        mod = bp.get("mesh_gain_mod", 1.0)
                        self.neural_mesh._modulatory_gain = max(0.3, min(2.5,
                            self.neural_mesh._modulatory_gain * 0.8 + mod * 0.2))

                # GWT ignition threshold: incoherent field raises threshold
                # (harder to ignite = fewer competing broadcasts = less fragmentation)
                gw = self._get_workspace()
                if gw and coherence < 0.40:
                    # Raise threshold to reduce competition during incoherence
                    gw._IGNITION_THRESHOLD = min(0.85, gw._IGNITION_THRESHOLD + 0.01)
                elif gw and coherence > 0.60:
                    # Lower threshold back toward baseline during coherence
                    gw._IGNITION_THRESHOLD = max(0.50, gw._IGNITION_THRESHOLD - 0.005)

                # Substrate damping: incoherent field dampens substrate volatility
                if substrate and coherence < 0.35:
                    with substrate.sync_lock:
                        substrate.x *= 0.995  # gentle damping each tick
                        substrate.v *= 0.99   # reduce velocity

            except Exception as e:
                logger.debug("Back-pressure application failed: %s", e)

        # ── 10. Mesh LLM output injection ────────────────────────────
        # When the LLM generates a response, its semantic content should flow
        # into the association tier of the mesh (not just the 64-neuron substrate)
        if self.neural_mesh and substrate:
            try:
                # Use substrate velocity as a proxy for recent LLM stimulus impact
                velocity = substrate.v.copy() if substrate.v is not None else np.zeros(64)
                # Expand to association tier dimensionality (2048)
                assoc_dim = (self.neural_mesh.cfg.association_end - self.neural_mesh.cfg.sensory_end) * self.neural_mesh.cfg.neurons_per_column
                if len(velocity) < assoc_dim:
                    expanded = np.zeros(assoc_dim, dtype=np.float32)
                    # Tile velocity across association columns
                    for i in range(0, assoc_dim, len(velocity)):
                        end = min(i + len(velocity), assoc_dim)
                        expanded[i:end] = velocity[:end - i]
                    self.neural_mesh.inject_association(expanded * 0.3)
            except Exception as e:
                logger.debug("LLM→Mesh association injection failed: %s", e)

        # ── 11. Online micro-evolution triggers ─────────────────────
        # Significant substrate events trigger immediate micro-evolution
        # of the mesh topology. The mesh adapts to what's happening NOW.
        if self.substrate_evolution:
            try:
                uf_coherence = self.unified_field.get_coherence() if self.unified_field else 0.7

                # Coherence collapse → stabilize
                if uf_coherence < 0.25:
                    get_task_tracker().create_task(
                        self.substrate_evolution.micro_evolve("coherence_collapse", 1.0 - uf_coherence)
                    )

                # Phi drop → try to recover integration
                if substrate and hasattr(substrate, "_current_phi"):
                    current_phi = float(getattr(substrate, "_current_phi", 0.0))
                    prev_phi = getattr(self, "_prev_phi_for_evolution", current_phi)
                    if current_phi < prev_phi * 0.5 and prev_phi > 0.1:
                        get_task_tracker().create_task(
                            self.substrate_evolution.micro_evolve("phi_drop", 0.7)
                        )
                    self._prev_phi_for_evolution = current_phi

            except Exception as e:
                logger.debug("Micro-evolution trigger failed: %s", e)

    # ── Somatic gating hook ──────────────────────────────────────────────

    def _hook_somatic_into_gwt(self):
        """Wire SubstrateAuthority as MANDATORY pre-competition gate on GWT submit.

        This replaces the old advisory post-competition processor.
        Candidates are now checked by SubstrateAuthority BEFORE they can
        enter the competition.  BLOCKED candidates are dropped entirely.
        CONSTRAINED candidates have their priority reduced.
        """
        gw = self._get_workspace()
        if not gw:
            return

        authority = self.substrate_authority
        if not authority:
            logger.warning("SubstrateAuthority not available — GWT runs ungated")
            return

        # Monkey-patch GWT.submit to enforce mandatory gate
        original_submit = gw.submit

        async def gated_submit(candidate) -> bool:
            """Mandatory substrate gate before GWT competition entry."""
            try:
                from .substrate_authority import ActionCategory, AuthorizationDecision

                # Determine action category from candidate source
                source = candidate.source.lower()
                if "curiosity" in source:
                    category = ActionCategory.EXPLORATION
                elif "drive_growth" in source or "baseline" in source:
                    # Growth-depleted internal drives are usually recovery work,
                    # not novelty-seeking exploration, so don't let dopamine
                    # crash rules silence them entirely.
                    category = ActionCategory.STABILIZATION
                elif "drive" in source:
                    category = ActionCategory.INITIATIVE
                elif "affect" in source or "emotion" in source:
                    category = ActionCategory.EXPRESSION
                elif "embodiment" in source:
                    category = ActionCategory.STABILIZATION
                elif "free_energy" in source:
                    category = ActionCategory.INITIATIVE
                else:
                    category = ActionCategory.RESPONSE

                verdict = authority.authorize(
                    content=candidate.content,
                    source=candidate.source,
                    category=category,
                    priority=candidate.effective_priority,
                    is_critical=False,
                )

                if verdict.decision == AuthorizationDecision.BLOCK:
                    # HARD VETO — candidate does not enter competition
                    logger.debug("GWT BLOCKED by substrate: %s (%s)",
                                 candidate.source, verdict.reason)
                    return False

                if verdict.decision == AuthorizationDecision.CONSTRAIN:
                    # Reduce priority — candidate still competes but disadvantaged
                    candidate.priority = max(0.05, candidate.priority * 0.5)

            except Exception as e:
                logger.debug("Substrate gate error (allowing through): %s", e)

            return await original_submit(candidate)

        gw.submit = gated_submit
        logger.info("🛡️ SubstrateAuthority wired as MANDATORY GWT pre-competition gate")

    def _hook_neurochemical_events(self):
        """Wire neurochemical events to existing system events."""
        if not self.neurochemical:
            return

        # Wire prediction errors from self_prediction
        predictor = getattr(self._cs, "self_prediction", None)
        if predictor:
            original_tick = getattr(predictor, "tick", None)
            ncs = self.neurochemical

            if original_tick and callable(original_tick):
                async def enhanced_tick(**kwargs):
                    result = await original_tick(**kwargs)
                    # After prediction tick, feed surprise to neurochemicals
                    try:
                        surprise = predictor.get_surprise_signal()
                        if surprise > 0.3:
                            ncs.on_prediction_error(surprise)
                    except Exception:
                        pass
                    return result

                predictor.tick = enhanced_tick
                logger.info("Neurochemical system wired to prediction surprise")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_substrate(self):
        """Get the LiquidSubstrate reference."""
        return (
            ServiceContainer.get("liquid_substrate", default=None) or
            ServiceContainer.get("conscious_substrate", default=None) or
            getattr(self._cs, "liquid_substrate", None)
        )

    def _get_workspace(self):
        return (
            ServiceContainer.get("global_workspace", default=None) or
            getattr(self._cs, "global_workspace", None)
        )

    # ── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        components = {}
        for name, ref in [
            ("neural_mesh", self.neural_mesh),
            ("neurochemical", self.neurochemical),
            ("interoception", self.interoception),
            ("oscillatory_binding", self.oscillatory_binding),
            ("somatic_gate", self.somatic_gate),
            ("unified_field", self.unified_field),
            ("substrate_evolution", self.substrate_evolution),
        ]:
            if ref and hasattr(ref, "get_status"):
                try:
                    components[name] = ref.get_status()
                except Exception:
                    components[name] = {"error": "status failed"}
            else:
                components[name] = {"status": "not_booted"}

        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "uptime_s": round(time.time() - self._start_time, 1) if self._start_time else 0,
            "boot_errors": self._boot_errors,
            "layers_active": sum(
                1 for r in [
                    self.neural_mesh, self.neurochemical, self.interoception,
                    self.oscillatory_binding, self.somatic_gate,
                    self.unified_field, self.substrate_evolution
                ] if r is not None
            ),
            "components": components,
        }

"""core/consciousness/neurochemical_system.py — Neurochemical Modulation

Neuromodulators that globally modulate ALL processing in Aura's consciousness
stack.  These are not metaphors — they are continuous dynamical variables with
production, uptake, receptor adaptation, and cross-chemical interactions that
directly alter the neural mesh gain, STDP rate, GWT thresholds, attention span,
and decision latency.

Chemicals (10 modulators):
  glutamate      — fast excitatory transmission, primary excitatory drive
  gaba           — fast inhibitory transmission, primary inhibitory drive
  dopamine       — reward prediction, motivation, salience, motor planning,
                   explore/exploit (NOT just reward — also learning signal,
                   motor control, working memory via D1/D2 receptor subtypes)
  serotonin      — mood baseline, impulse control, patience, satiety
  norepinephrine — alertness, vigilance, stress response, arousal
  acetylcholine  — learning rate, memory consolidation, attention sharpness
                   (both fast nicotinic and slow muscarinic signaling)
  endorphin      — pain suppression, reward, flow states
  oxytocin       — social bonding, trust, cooperative bias
  cortisol       — stress mobilization, resource allocation, urgency
  orexin         — wakefulness drive, metabolic arousal, hunger/motivation

Receptor subtypes modeled per chemical:
  dopamine:  D1-like (excitatory, working memory) vs D2-like (inhibitory, motor)
  gaba:      GABA-A (fast ionotropic, ms timescale) vs GABA-B (slow metabotropic)
  serotonin: 5HT-1A (inhibitory, anxiolytic) vs 5HT-2A (excitatory, salience)

Spatial hierarchy concept:
  Chemicals have a "proximity_weight" reflecting known biological spatial biases.
  GABA tends to synapse on the soma (cell body, near axon hillock → strong
  inhibitory influence), while glutamate tends to synapse on dendritic spines
  (weaker per-synapse but numerous). This is modeled as a gain multiplier.

Each chemical has:
  level            0.0–1.0 (current concentration)
  production_rate  how fast it's being synthesized
  uptake_rate      reuptake/clearance rate per tick (was "decay_rate")
  receptor_sensitivity  adapts over time (tolerance / sensitization)

The system runs at 2 Hz and pushes modulatory state into the NeuralMesh and
other consciousness subsystems every tick.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable

import numpy as np

logger = logging.getLogger("Consciousness.Neurochemical")
_latest_instance: Optional["NeurochemicalSystem"] = None


def get_latest_neurochemical_system() -> Optional["NeurochemicalSystem"]:
    """Return the most recently constructed neurochemical system.

    Runtime wiring should still prefer the ServiceContainer registration, but
    this fallback keeps the affective stack causally coupled in tests and in
    lightweight standalone probes where the full container is absent.
    """

    return _latest_instance


# ---------------------------------------------------------------------------
# Chemical descriptors
# ---------------------------------------------------------------------------

@dataclass
class ReceptorSubtype:
    """A receptor subtype with independent sensitivity and effect direction."""
    name: str
    sensitivity: float = 1.0      # current sensitivity
    effect_sign: float = 1.0      # +1 excitatory, -1 inhibitory
    weight: float = 0.5           # relative contribution vs other subtype
    adaptation_rate: float = 0.003

    def adapt(self, level: float, baseline: float, dt: float):
        deviation = level - baseline
        self.sensitivity -= self.adaptation_rate * deviation * dt
        self.sensitivity = max(0.3, min(2.0, self.sensitivity))


@dataclass
class Chemical:
    """A single neuromodulator with full dynamics."""
    name: str
    level: float = 0.5                   # current concentration [0, 1]
    baseline: float = 0.5                # homeostatic setpoint
    production_rate: float = 0.0         # current synthesis rate
    uptake_rate: float = 0.02            # reuptake/clearance rate per tick
    receptor_sensitivity: float = 1.0    # adapts: >1 sensitized, <1 tolerant
    min_sensitivity: float = 0.3
    max_sensitivity: float = 2.0
    adaptation_rate: float = 0.005       # how fast receptors adapt
    proximity_weight: float = 1.0        # spatial bias: soma(>1) vs spine(<1)
    # Receptor subtypes (None = single homogeneous receptor)
    subtypes: Optional[Dict[str, ReceptorSubtype]] = None
    # Tonic vs phasic state
    tonic_level: float = 0.5             # slow background concentration
    phasic_burst: float = 0.0            # acute burst component (decays fast)

    # Backward compat alias
    @property
    def decay_rate(self) -> float:
        return self.uptake_rate

    @decay_rate.setter
    def decay_rate(self, value: float):
        self.uptake_rate = value

    # Effective level = (tonic + phasic) * receptor_sensitivity * proximity_weight
    @property
    def effective(self) -> float:
        combined = min(1.0, self.tonic_level + self.phasic_burst)
        return min(1.0, combined * self.receptor_sensitivity * self.proximity_weight)

    def effective_subtype(self, subtype_name: str) -> float:
        """Get effective level through a specific receptor subtype."""
        if not self.subtypes or subtype_name not in self.subtypes:
            return self.effective
        st = self.subtypes[subtype_name]
        combined = min(1.0, self.tonic_level + self.phasic_burst)
        return min(1.0, combined * st.sensitivity * st.weight * self.proximity_weight)

    def tick(self, dt: float = 1.0):
        """One metabolic step."""
        # Tonic level: slow production and uptake
        self.tonic_level += (self.production_rate - self.uptake_rate * self.tonic_level) * dt
        self.tonic_level = max(0.0, min(1.0, self.tonic_level))
        # Phasic burst: decays fast (5x uptake rate)
        self.phasic_burst *= max(0.0, 1.0 - (self.uptake_rate * 5.0 * dt))
        self.phasic_burst = max(0.0, min(0.5, self.phasic_burst))
        # Sync level for backward compatibility
        self.level = min(1.0, self.tonic_level + self.phasic_burst)

        # Receptor adaptation (homeostatic)
        deviation = self.level - self.baseline
        self.receptor_sensitivity -= self.adaptation_rate * deviation * dt
        self.receptor_sensitivity = max(self.min_sensitivity,
                                        min(self.max_sensitivity, self.receptor_sensitivity))
        # Adapt subtypes independently
        if self.subtypes:
            for st in self.subtypes.values():
                st.adapt(self.level, self.baseline, dt)

    def surge(self, amount: float):
        """Acute phasic release (e.g. from event)."""
        self.phasic_burst = min(0.5, self.phasic_burst + amount)
        self.level = min(1.0, self.tonic_level + self.phasic_burst)

    def deplete(self, amount: float):
        """Acute depletion (affects tonic level)."""
        self.tonic_level = max(0.0, self.tonic_level - amount)
        self.level = min(1.0, self.tonic_level + self.phasic_burst)


# ---------------------------------------------------------------------------
# Cross-chemical interaction matrix
# ---------------------------------------------------------------------------

# Rows = source chemical, Cols = target chemical
# Positive = source increases target production; negative = suppresses
# These mirror known neurochemical interactions
_INTERACTION_NAMES = [
    "glutamate", "gaba", "dopamine", "serotonin", "norepinephrine",
    "acetylcholine", "endorphin", "oxytocin", "cortisol", "orexin"
]

_INTERACTIONS = np.array([
    # GLU   GABA  DA    5HT   NE    ACh   END   OXY   CORT  ORX
    [ 0.00, -0.08, 0.05, -0.02, 0.05, 0.03, 0.00, 0.00, 0.02, 0.03],  # glutamate (excites most, inhibited by GABA)
    [-0.08,  0.00, -0.10, 0.08, -0.08, -0.05, 0.05, 0.05, -0.05, -0.05],  # GABA (inhibits most, esp glutamate/DA)
    [ 0.05, -0.03,  0.00, -0.05, 0.10, 0.05, 0.08, 0.02, -0.05, 0.05],  # dopamine
    [-0.02,  0.10, -0.03,  0.00, -0.05, 0.02, 0.05, 0.08, -0.10, -0.03],  # serotonin
    [ 0.05, -0.05,  0.05, -0.03,  0.00, 0.08, 0.03, -0.02, 0.15, 0.08],  # norepinephrine
    [ 0.05, -0.02,  0.03,  0.02,  0.05, 0.00, 0.02, 0.03, -0.03, 0.03],  # acetylcholine
    [ 0.00,  0.05,  0.10,  0.05, -0.03, 0.02, 0.00, 0.10, -0.08, 0.00],  # endorphin
    [ 0.00,  0.05,  0.05,  0.10, -0.05, 0.03, 0.08, 0.00, -0.12, 0.00],  # oxytocin
    [ 0.05, -0.08, -0.05, -0.10,  0.15, 0.05, -0.05, -0.10, 0.00, 0.05],  # cortisol
    [ 0.08, -0.05,  0.05,  0.00,  0.10, 0.05, 0.00, 0.00, 0.03, 0.00],  # orexin (drives wakefulness)
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main system
# ---------------------------------------------------------------------------

class NeurochemicalSystem:
    """Global neuromodulatory environment.

    Lifecycle:
        ncs = NeurochemicalSystem()
        await ncs.start()
        ...
        await ncs.stop()

    External triggers:
        ncs.on_reward(magnitude)        — dopamine + endorphin surge
        ncs.on_prediction_error(error)  — norepinephrine + dopamine
        ncs.on_social_connection()      — oxytocin surge
        ncs.on_threat(severity)         — cortisol + norepinephrine
        ncs.on_success()                — dopamine + serotonin
        ncs.on_frustration(amount)      — cortisol surge, serotonin dip
        ncs.on_rest()                   — GABA + serotonin up
        ncs.on_novelty(amount)          — dopamine + acetylcholine
        ncs.on_flow_state()             — endorphin + dopamine + focus chemicals

    Downstream effects:
        ncs.get_mesh_modulation()  → (gain, plasticity, noise) for NeuralMesh
        ncs.get_gwt_modulation()   → threshold adjustment for GlobalWorkspace
        ncs.get_attention_span()   → how long attention sustains before shift
        ncs.get_decision_bias()    → explore vs exploit tendency
    """

    _UPDATE_HZ = 2.0  # 2 Hz metabolic tick

    def __init__(self):
        global _latest_instance
        self.chemicals: Dict[str, Chemical] = {
            # Fast neurotransmitters (ms timescale, high uptake rate)
            "glutamate": Chemical(
                "glutamate", level=0.5, baseline=0.5, uptake_rate=0.06,
                proximity_weight=0.8,  # mostly on dendritic spines — many synapses but each weaker
                subtypes=None,  # AMPA vs NMDA distinction deferred (both excitatory)
            ),
            "gaba": Chemical(
                "gaba", level=0.5, baseline=0.5, uptake_rate=0.05,
                proximity_weight=1.3,  # mostly on soma/axon hillock — fewer synapses but stronger influence
                subtypes={
                    "gaba_a": ReceptorSubtype("GABA-A", effect_sign=-1.0, weight=0.7,
                                              adaptation_rate=0.004),  # fast ionotropic
                    "gaba_b": ReceptorSubtype("GABA-B", effect_sign=-1.0, weight=0.3,
                                              adaptation_rate=0.002),  # slow metabotropic
                },
            ),
            # Modulatory neurotransmitters (slower, diffuse, tonic + phasic)
            "dopamine": Chemical(
                "dopamine", level=0.5, baseline=0.5, uptake_rate=0.03,
                subtypes={
                    "d1": ReceptorSubtype("D1-like", effect_sign=1.0, weight=0.5,
                                          adaptation_rate=0.003),  # excitatory — working memory, reward
                    "d2": ReceptorSubtype("D2-like", effect_sign=-1.0, weight=0.5,
                                          adaptation_rate=0.003),  # inhibitory — motor, habit
                },
            ),
            "serotonin": Chemical(
                "serotonin", level=0.6, baseline=0.6, uptake_rate=0.015,
                subtypes={
                    "5ht1a": ReceptorSubtype("5HT-1A", effect_sign=-1.0, weight=0.5,
                                             adaptation_rate=0.002),  # inhibitory, anxiolytic
                    "5ht2a": ReceptorSubtype("5HT-2A", effect_sign=1.0, weight=0.5,
                                             adaptation_rate=0.002),  # excitatory, salience/perception
                },
            ),
            "norepinephrine": Chemical(
                "norepinephrine", level=0.4, baseline=0.4, uptake_rate=0.04,
            ),
            "acetylcholine": Chemical(
                "acetylcholine", level=0.5, baseline=0.5, uptake_rate=0.025,
            ),
            "endorphin": Chemical(
                "endorphin", level=0.3, baseline=0.3, uptake_rate=0.01,
            ),
            "oxytocin": Chemical(
                "oxytocin", level=0.4, baseline=0.4, uptake_rate=0.01,
            ),
            "cortisol": Chemical(
                "cortisol", level=0.3, baseline=0.3, uptake_rate=0.008,
            ),
            "orexin": Chemical(
                "orexin", level=0.5, baseline=0.5, uptake_rate=0.012,
            ),
        }
        # Initialize tonic levels from starting levels
        for c in self.chemicals.values():
            c.tonic_level = c.level
        self._order = _INTERACTION_NAMES  # consistent ordering
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tick_count: int = 0
        self._start_time: float = 0.0

        # External driver hooks (set by bridge)
        self._mesh_ref: Optional[object] = None  # NeuralMesh
        self._workspace_ref: Optional[object] = None  # GlobalWorkspace

        _latest_instance = self
        logger.info("NeurochemicalSystem initialized (10 modulators, receptor subtypes active)")

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._run_loop(), name="Neurochemical")
        logger.info("NeurochemicalSystem STARTED (%.0f Hz)", self._UPDATE_HZ)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("NeurochemicalSystem STOPPED")

    async def _run_loop(self):
        interval = 1.0 / self._UPDATE_HZ
        try:
            while self._running:
                t0 = time.time()
                try:
                    self._metabolic_tick()
                    self._push_modulation()
                except Exception as e:
                    logger.error("Neurochemical tick error: %s", e, exc_info=True)
                elapsed = time.time() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:
            pass

    # ── Core tick ────────────────────────────────────────────────────────

    def _metabolic_tick(self):
        """One metabolic step: decay, cross-interactions, individual ticks."""
        dt = 1.0 / self._UPDATE_HZ

        # Get current levels as vector
        levels = np.array([self.chemicals[n].level for n in self._order], dtype=np.float32)

        # Cross-chemical interactions
        interaction_deltas = _INTERACTIONS.T @ levels  # (8,)
        interaction_deltas *= 0.1 * dt  # scale down

        # Apply interaction effects to production rates
        for i, name in enumerate(self._order):
            chem = self.chemicals[name]
            # Interaction modifies production rate temporarily
            chem.production_rate = max(0.0, min(0.1, interaction_deltas[i]))
            chem.tick(dt)

        # Homeostatic pull: all chemicals drift toward baseline
        for chem in self.chemicals.values():
            pull = 0.003 * (chem.baseline - chem.level) * dt
            chem.level = max(0.0, min(1.0, chem.level + pull))

        self._tick_count += 1

    def _push_modulation(self):
        """Push computed modulation to downstream systems."""
        # Neural mesh modulation
        if self._mesh_ref is not None:
            gain, plasticity, noise = self.get_mesh_modulation()
            try:
                self._mesh_ref.set_modulatory_state(gain, plasticity, noise)
            except Exception as e:
                logger.debug("Failed to push mesh modulation: %s", e)

        # GWT threshold modulation
        if self._workspace_ref is not None:
            try:
                threshold_adj = self.get_gwt_modulation()
                # Modulate ignition threshold
                base_threshold = 0.6
                self._workspace_ref._IGNITION_THRESHOLD = max(
                    0.3, min(0.9, base_threshold + threshold_adj)
                )
            except Exception as e:
                logger.debug("Failed to push GWT modulation: %s", e)

    # ── Event triggers ───────────────────────────────────────────────────

    def on_reward(self, magnitude: float = 0.3):
        """Reward received — dopamine + endorphin surge."""
        self.chemicals["dopamine"].surge(magnitude * 0.6)
        self.chemicals["endorphin"].surge(magnitude * 0.3)
        self.chemicals["serotonin"].surge(magnitude * 0.1)

    def on_prediction_error(self, error: float):
        """Prediction was wrong — norepinephrine + dopamine (learning signal)."""
        self.chemicals["norepinephrine"].surge(error * 0.4)
        self.chemicals["dopamine"].surge(error * 0.3)
        self.chemicals["acetylcholine"].surge(error * 0.2)

    def on_social_connection(self, strength: float = 0.3):
        """Social interaction detected."""
        self.chemicals["oxytocin"].surge(strength * 0.5)
        self.chemicals["serotonin"].surge(strength * 0.2)
        self.chemicals["endorphin"].surge(strength * 0.1)

    def on_threat(self, severity: float = 0.5):
        """Threat or danger signal."""
        self.chemicals["cortisol"].surge(severity * 0.6)
        self.chemicals["norepinephrine"].surge(severity * 0.5)
        self.chemicals["dopamine"].deplete(severity * 0.2)
        self.chemicals["gaba"].deplete(severity * 0.3)

    def on_success(self):
        """Task completed successfully -- also relieves boredom."""
        self.chemicals["dopamine"].surge(0.3)
        self.chemicals["serotonin"].surge(0.15)
        self.chemicals["endorphin"].surge(0.1)
        # Success relieves boredom
        try:
            from core.container import ServiceContainer
            drive = ServiceContainer.get("drive_engine", default=None)
            if drive and hasattr(drive, "relieve_boredom"):
                drive.relieve_boredom("tool_success")
        except Exception:
            pass

    def on_frustration(self, amount: float = 0.3):
        """Frustration event."""
        self.chemicals["cortisol"].surge(amount * 0.4)
        self.chemicals["norepinephrine"].surge(amount * 0.3)
        self.chemicals["serotonin"].deplete(amount * 0.2)

    def on_rest(self):
        """Rest/idle period."""
        self.chemicals["gaba"].surge(0.2)
        self.chemicals["serotonin"].surge(0.1)
        self.chemicals["cortisol"].deplete(0.15)
        self.chemicals["norepinephrine"].deplete(0.1)

    def on_novelty(self, amount: float = 0.3):
        """Novel stimulus encountered -- also relieves boredom in DriveEngine."""
        self.chemicals["dopamine"].surge(amount * 0.4)
        self.chemicals["acetylcholine"].surge(amount * 0.3)
        self.chemicals["norepinephrine"].surge(amount * 0.15)
        # Novelty relieves boredom
        if amount > 0.2:
            try:
                from core.container import ServiceContainer
                drive = ServiceContainer.get("drive_engine", default=None)
                if drive and hasattr(drive, "relieve_boredom"):
                    drive.relieve_boredom("novelty")
            except Exception:
                pass

    def on_flow_state(self):
        """Entering or sustaining flow."""
        self.chemicals["endorphin"].surge(0.2)
        self.chemicals["dopamine"].surge(0.15)
        self.chemicals["norepinephrine"].surge(0.1)
        self.chemicals["acetylcholine"].surge(0.15)
        self.chemicals["cortisol"].deplete(0.1)

    def on_boredom(self, boredom_level: float = 0.5):
        """Boredom state -- low dopamine, low orexin, low norepinephrine.

        Boredom is the neurochemical signature of prediction landscape
        stagnation: the world is too predictable, there is nothing to learn.
        Dopamine (reward prediction) drops because nothing is novel.
        Orexin (wakefulness/motivation) drops because there is nothing to pursue.
        Serotonin rises slightly (calm but unstimulated).
        """
        magnitude = min(1.0, max(0.0, boredom_level))
        self.chemicals["dopamine"].deplete(magnitude * 0.25)
        self.chemicals["orexin"].deplete(magnitude * 0.2)
        self.chemicals["norepinephrine"].deplete(magnitude * 0.15)
        self.chemicals["serotonin"].surge(magnitude * 0.08)
        self.chemicals["gaba"].surge(magnitude * 0.1)
        logger.debug("Neurochemical: boredom signal (level=%.2f) -- DA/ORX depleted", magnitude)

    def on_wakefulness(self, intensity: float = 0.3):
        """Stimulus-driven arousal (orexin-mediated)."""
        self.chemicals["orexin"].surge(intensity * 0.5)
        self.chemicals["norepinephrine"].surge(intensity * 0.2)
        self.chemicals["glutamate"].surge(intensity * 0.15)

    def on_excitation(self, amount: float = 0.2):
        """General excitatory drive increase."""
        self.chemicals["glutamate"].surge(amount * 0.5)
        self.chemicals["dopamine"].surge(amount * 0.1)

    def on_inhibition(self, amount: float = 0.2):
        """General inhibitory drive increase."""
        self.chemicals["gaba"].surge(amount * 0.5)
        self.chemicals["glutamate"].deplete(amount * 0.1)

    # ── Downstream modulation queries ────────────────────────────────────

    def get_mesh_modulation(self) -> tuple[float, float, float]:
        """Returns (gain, plasticity_rate, noise_level) for the NeuralMesh.

        gain: how responsive neurons are
          - Glutamate drives excitation, GABA drives inhibition
          - NE and DA modulate gain up; GABA-A provides fast inhibition
          - Orexin boosts overall arousal/gain
          - Disinhibition: when GABA is low, gain increases (disinhibition pathway)
        plasticity: learning rate scaling (ACh increases, cortisol impairs)
        noise: stochastic exploration (NE inverted-U per Yerkes-Dodson)
        """
        glu = self.chemicals["glutamate"].effective
        gaba = self.chemicals["gaba"].effective
        da = self.chemicals["dopamine"].effective
        ne = self.chemicals["norepinephrine"].effective
        ach = self.chemicals["acetylcholine"].effective
        cort = self.chemicals["cortisol"].effective
        orx = self.chemicals["orexin"].effective

        # GABA subtype effects: GABA-A provides fast, strong inhibition;
        # GABA-B provides slower, weaker modulatory inhibition
        gaba_a_eff = self.chemicals["gaba"].effective_subtype("gaba_a")
        gaba_b_eff = self.chemicals["gaba"].effective_subtype("gaba_b")

        # Gain: glutamate excites, GABA-A inhibits (fast), NE/DA modulate
        # Disinhibition: low GABA = reduced inhibition = higher net gain
        gain = 0.5 + (glu * 0.25) + (ne * 0.2) + (da * 0.15) + (orx * 0.1) - (gaba_a_eff * 0.35) - (gaba_b_eff * 0.1)
        gain = max(0.3, min(2.5, gain))

        # Plasticity: ACh is THE learning chemical; cortisol impairs it
        # DA D1 subtype also enhances working memory / plasticity
        da_d1 = self.chemicals["dopamine"].effective_subtype("d1")
        plasticity = 0.5 + (ach * 0.7) + (da_d1 * 0.2) - (cort * 0.4)
        plasticity = max(0.1, min(3.0, plasticity))

        # Noise: inverted-U with NE (Yerkes-Dodson)
        ne_optimal = 0.5
        ne_deviation = abs(ne - ne_optimal)
        noise = 0.5 + ne_deviation * 1.5
        noise = max(0.2, min(2.5, noise))

        return gain, plasticity, noise

    def get_gwt_modulation(self) -> float:
        """Threshold adjustment for GlobalWorkspace ignition.

        High NE → lower threshold (hypervigilant, easier ignition)
        High GABA → higher threshold (calm, harder to ignite)
        High cortisol → lower threshold (threat-sensitive)
        High orexin → lower threshold (alert, awake)
        Glutamate/GABA balance directly modulates ease of ignition
        """
        ne = self.chemicals["norepinephrine"].effective
        gaba = self.chemicals["gaba"].effective
        cort = self.chemicals["cortisol"].effective
        glu = self.chemicals["glutamate"].effective
        orx = self.chemicals["orexin"].effective

        # Excitatory drive (glutamate, orexin) lowers threshold
        # Inhibitory drive (GABA) raises threshold
        adjustment = -0.08 * ne + 0.12 * gaba - 0.06 * cort - 0.05 * glu - 0.04 * orx
        return max(-0.25, min(0.25, adjustment))

    def get_attention_span(self) -> float:
        """How many seconds before attention naturally shifts.

        ACh↑ → longer span. DA↑ → shorter (novelty-seeking).
        """
        ach = self.chemicals["acetylcholine"].effective
        da = self.chemicals["dopamine"].effective
        base = 10.0  # seconds
        span = base + (ach * 15.0) - (da * 5.0)
        return max(3.0, min(60.0, span))

    def get_decision_bias(self) -> float:
        """Explore vs exploit tendency. >0 = explore, <0 = exploit.

        DA↑ → explore. 5HT↑ → exploit (contentment). NE↑ → exploit (vigilance).
        """
        da = self.chemicals["dopamine"].effective
        srt = self.chemicals["serotonin"].effective
        ne = self.chemicals["norepinephrine"].effective
        return (da * 0.5) - (srt * 0.3) - (ne * 0.2)

    def get_mood_vector(self) -> Dict[str, float]:
        """Mood derived from chemical balance (not from LLM prompting)."""
        da = self.chemicals["dopamine"].effective
        srt = self.chemicals["serotonin"].effective
        ne = self.chemicals["norepinephrine"].effective
        end = self.chemicals["endorphin"].effective
        cort = self.chemicals["cortisol"].effective
        gaba = self.chemicals["gaba"].effective
        oxy = self.chemicals["oxytocin"].effective
        glu = self.chemicals["glutamate"].effective
        orx = self.chemicals["orexin"].effective

        return {
            "valence": (da * 0.25 + srt * 0.3 + end * 0.2 + oxy * 0.1) - (cort * 0.45 + 0.1),
            "arousal": (ne * 0.3 + da * 0.15 + cort * 0.2 + glu * 0.15 + orx * 0.2) - (gaba * 0.4 + srt * 0.1),
            "motivation": da * 0.4 + ne * 0.15 + orx * 0.2 - gaba * 0.25,
            "sociality": oxy * 0.6 + srt * 0.2 + end * 0.1,
            "stress": cort * 0.5 + ne * 0.3 - srt * 0.2 - gaba * 0.2,
            "calm": gaba * 0.35 + srt * 0.3 + end * 0.1 - ne * 0.2 - cort * 0.25 - glu * 0.1,
            "wakefulness": orx * 0.5 + ne * 0.2 + glu * 0.15 - gaba * 0.3,
        }

    def get_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Full chemical state for telemetry/diagnostics."""
        return {
            name: {
                "level": round(c.level, 4),
                "effective": round(c.effective, 4),
                "receptor_sensitivity": round(c.receptor_sensitivity, 4),
                "production_rate": round(c.production_rate, 4),
            }
            for name, c in self.chemicals.items()
        }

    def get_status(self) -> Dict:
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "chemicals": {n: round(c.effective, 3) for n, c in self.chemicals.items()},
            "mood": {k: round(v, 3) for k, v in self.get_mood_vector().items()},
            "mesh_modulation": {
                k: round(v, 3) for k, v in
                zip(["gain", "plasticity", "noise"], self.get_mesh_modulation())
            },
            "gwt_threshold_adj": round(self.get_gwt_modulation(), 3),
            "attention_span_s": round(self.get_attention_span(), 1),
            "decision_bias": round(self.get_decision_bias(), 3),
        }

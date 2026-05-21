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
import time
from dataclasses import dataclass

import numpy as np

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Consciousness.Neurochemical")
_latest_instance: NeurochemicalSystem | None = None

_RECOVERABLE_NEUROCHEMICAL_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_neurochemical_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "neurochemical_system",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError:
        record_degradation("neurochemical_system", error)


def _finite_float(raw: object, default: float) -> tuple[float, bool]:
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default, False
    if not np.isfinite(value):
        return default, False
    return value, True


def _clamp_float(value: float, *, lower: float, upper: float) -> tuple[float, bool]:
    clamped = max(lower, min(upper, value))
    return clamped, clamped == value


def get_latest_neurochemical_system() -> NeurochemicalSystem | None:
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
        level, _ = _finite_float(level, 0.5)
        baseline, _ = _finite_float(baseline, 0.5)
        dt, _ = _finite_float(dt, 1.0)
        rate, _ = _finite_float(self.adaptation_rate, 0.003)
        level, _ = _clamp_float(level, lower=0.0, upper=1.0)
        baseline, _ = _clamp_float(baseline, lower=0.0, upper=1.0)
        dt, _ = _clamp_float(dt, lower=0.0, upper=60.0)
        rate, _ = _clamp_float(rate, lower=0.0, upper=1.0)
        sensitivity, _ = _finite_float(self.sensitivity, 1.0)
        deviation = level - baseline
        sensitivity -= rate * deviation * dt
        self.sensitivity = max(0.3, min(2.0, sensitivity))


@dataclass
class Chemical:
    """A single neuromodulator with full dynamics."""
    name: str
    level: float = 0.5                   # current concentration [0, 1]
    baseline: float = 0.5                # homeostatic setpoint
    production_rate: float = 0.0         # current synthesis rate (interaction delta)
    _base_production: float = 0.0        # intrinsic base production (not overwritten)
    uptake_rate: float = 0.02            # reuptake/clearance rate per tick
    receptor_sensitivity: float = 1.0    # adapts: >1 sensitized, <1 tolerant
    min_sensitivity: float = 0.3
    max_sensitivity: float = 2.0
    adaptation_rate: float = 0.005       # how fast receptors adapt
    proximity_weight: float = 1.0        # spatial bias: soma(>1) vs spine(<1)
    # Receptor subtypes (None = single homogeneous receptor)
    subtypes: dict[str, ReceptorSubtype] | None = None
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
        tonic, _ = _finite_float(self.tonic_level, self.baseline)
        phasic, _ = _finite_float(self.phasic_burst, 0.0)
        sensitivity, _ = _finite_float(self.receptor_sensitivity, 1.0)
        proximity, _ = _finite_float(self.proximity_weight, 1.0)
        tonic, _ = _clamp_float(tonic, lower=0.0, upper=1.0)
        phasic, _ = _clamp_float(phasic, lower=0.0, upper=0.5)
        sensitivity, _ = _clamp_float(
            sensitivity,
            lower=self.min_sensitivity,
            upper=self.max_sensitivity,
        )
        proximity, _ = _clamp_float(proximity, lower=0.0, upper=3.0)
        combined = min(1.0, tonic + phasic)
        return max(0.0, min(1.0, combined * sensitivity * proximity))

    def effective_subtype(self, subtype_name: str) -> float:
        """Get effective level through a specific receptor subtype."""
        if not self.subtypes or subtype_name not in self.subtypes:
            return self.effective
        st = self.subtypes[subtype_name]
        tonic, _ = _finite_float(self.tonic_level, self.baseline)
        phasic, _ = _finite_float(self.phasic_burst, 0.0)
        sensitivity, _ = _finite_float(st.sensitivity, 1.0)
        weight, _ = _finite_float(st.weight, 0.5)
        proximity, _ = _finite_float(self.proximity_weight, 1.0)
        tonic, _ = _clamp_float(tonic, lower=0.0, upper=1.0)
        phasic, _ = _clamp_float(phasic, lower=0.0, upper=0.5)
        sensitivity, _ = _clamp_float(sensitivity, lower=0.3, upper=2.0)
        weight, _ = _clamp_float(weight, lower=0.0, upper=1.0)
        proximity, _ = _clamp_float(proximity, lower=0.0, upper=3.0)
        combined = min(1.0, tonic + phasic)
        return max(0.0, min(1.0, combined * sensitivity * weight * proximity))

    def _sanitize_state(self, *, action: str) -> None:
        min_sensitivity, min_valid = _finite_float(self.min_sensitivity, 0.3)
        max_sensitivity, max_valid = _finite_float(self.max_sensitivity, 2.0)
        min_sensitivity, min_unchanged = _clamp_float(
            min_sensitivity,
            lower=0.01,
            upper=2.0,
        )
        max_sensitivity, max_unchanged = _clamp_float(
            max_sensitivity,
            lower=min_sensitivity,
            upper=5.0,
        )
        sensitivity_bounds_repaired = not all(
            (min_valid, max_valid, min_unchanged, max_unchanged)
        )
        self.min_sensitivity = min_sensitivity
        self.max_sensitivity = max_sensitivity
        fields = {
            "level": (self.level, 0.5, 0.0, 1.0),
            "baseline": (self.baseline, 0.5, 0.0, 1.0),
            "tonic_level": (self.tonic_level, 0.5, 0.0, 1.0),
            "phasic_burst": (self.phasic_burst, 0.0, 0.0, 0.5),
            "production_rate": (self.production_rate, 0.0, -0.15, 0.15),
            "_base_production": (self._base_production, 0.0, -0.15, 0.15),
            "uptake_rate": (self.uptake_rate, 0.02, 0.0, 1.0),
            "receptor_sensitivity": (
                self.receptor_sensitivity,
                1.0,
                self.min_sensitivity,
                self.max_sensitivity,
            ),
            "adaptation_rate": (self.adaptation_rate, 0.005, 0.0, 1.0),
            "proximity_weight": (self.proximity_weight, 1.0, 0.0, 3.0),
        }
        repaired: list[str] = []
        for field_name, (raw, default, lower, upper) in fields.items():
            value, valid = _finite_float(raw, float(default))
            value, unchanged = _clamp_float(value, lower=float(lower), upper=float(upper))
            if not valid or not unchanged:
                repaired.append(field_name)
            setattr(self, field_name, value)
        if sensitivity_bounds_repaired:
            repaired.extend(["min_sensitivity", "max_sensitivity"])
        if repaired:
            _record_neurochemical_degradation(
                ValueError(
                    f"{self.name} chemical state had invalid fields: {', '.join(repaired)}"
                ),
                action=action,
                severity="warning",
                extra={"chemical": self.name, "fields": repaired},
            )

    def tick(self, dt: float = 1.0):
        """One metabolic step.

        Tonic dynamics: an Ornstein-Uhlenbeck-style return-to-baseline plus
        an additive production driver. Uses the Ornstein-Uhlenbeck formulation:
        ``d(tonic)/dt = production + uptake_rate * (baseline - tonic)``.

        Receptor adaptation is bidirectional:
        - Above baseline → sensitivity decreases (tolerance)
        - Below baseline → sensitivity increases (re-sensitization)
        Both are bounded by [min_sensitivity, max_sensitivity].
        """
        self._sanitize_state(action="normalized chemical state before metabolic tick")
        dt, valid_dt = _finite_float(dt, 1.0)
        dt, dt_unchanged = _clamp_float(dt, lower=0.0, upper=60.0)
        if not valid_dt or not dt_unchanged:
            _record_neurochemical_degradation(
                ValueError(f"{self.name} chemical tick dt was invalid"),
                action="normalized chemical tick dt",
                severity="warning",
                extra={"chemical": self.name, "dt": dt},
            )
        expected_level = min(1.0, self.tonic_level + self.phasic_burst)
        external_level = max(0.0, min(1.0, self.level))
        if external_level > expected_level + 1e-6:
            self.tonic_level = max(0.0, min(1.0, external_level - self.phasic_burst))

        # Tonic level: production + homeostatic return
        self.tonic_level += (
            self.production_rate
            + self.uptake_rate * (self.baseline - self.tonic_level)
        ) * dt
        self.tonic_level = max(0.0, min(1.0, self.tonic_level))
        # Phasic burst: decays fast (5x uptake rate)
        self.phasic_burst *= max(0.0, 1.0 - (self.uptake_rate * 5.0 * dt))
        self.phasic_burst = max(0.0, min(0.5, self.phasic_burst))
        # Sync level for backward compatibility
        self.level = min(1.0, self.tonic_level + self.phasic_burst)

        # Receptor adaptation (bidirectional homeostatic)
        # deviation > 0 → overstimulated → sensitivity decreases (tolerance)
        # deviation < 0 → understimulated → sensitivity increases (re-sensitization)
        deviation = self.level - self.baseline
        self.receptor_sensitivity -= self.adaptation_rate * deviation * dt
        # Explicit re-sensitization pull toward 1.0 when sensitivity has drifted
        # This prevents sensitivity from getting permanently stuck at the floor
        sensitivity_drift = self.receptor_sensitivity - 1.0
        if deviation <= 0.0:
            self.receptor_sensitivity -= 0.002 * sensitivity_drift * dt  # slow pull toward 1.0
        self.receptor_sensitivity = max(self.min_sensitivity,
                                        min(self.max_sensitivity, self.receptor_sensitivity))
        # Adapt subtypes independently
        if self.subtypes:
            for st in self.subtypes.values():
                st.adapt(self.level, self.baseline, dt)

    def surge(self, amount: float):
        """Acute phasic release (e.g. from event).

        Applies diminishing returns when already near saturation:
        effective_amount scales down as level approaches 1.0.
        This prevents NE/cortisol from hitting ceiling under sustained threat.
        """
        self._sanitize_state(action="normalized chemical state before surge")
        amount, valid_amount = _finite_float(amount, 0.0)
        amount, amount_unchanged = _clamp_float(amount, lower=0.0, upper=1.0)
        if not valid_amount or not amount_unchanged:
            _record_neurochemical_degradation(
                ValueError(f"{self.name} chemical surge amount was invalid"),
                action="normalized chemical surge amount",
                severity="warning",
                extra={"chemical": self.name, "amount": amount},
            )
        headroom = max(0.0, 1.0 - self.level)
        # Diminishing returns: effective surge is proportional to remaining headroom
        effective = amount * min(1.0, headroom * 2.0)
        self.phasic_burst = min(0.5, self.phasic_burst + effective)
        self.level = min(1.0, self.tonic_level + self.phasic_burst)

    def deplete(self, amount: float):
        """Acute depletion (affects tonic level).

        Applies floor protection: depletion scales down as level approaches 0.
        """
        self._sanitize_state(action="normalized chemical state before depletion")
        amount, valid_amount = _finite_float(amount, 0.0)
        amount, amount_unchanged = _clamp_float(amount, lower=0.0, upper=1.0)
        if not valid_amount or not amount_unchanged:
            _record_neurochemical_degradation(
                ValueError(f"{self.name} chemical depletion amount was invalid"),
                action="normalized chemical depletion amount",
                severity="warning",
                extra={"chemical": self.name, "amount": amount},
            )
        floor_guard = max(0.0, self.tonic_level - 0.05)  # protect last 5%
        effective = min(amount, floor_guard)
        self.tonic_level = max(0.0, self.tonic_level - effective)
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

# Spectral radius validation: if max |eigenvalue| > 1.0, the interaction
# matrix can amplify signals unboundedly under sustained drive.
_INTERACTION_EIGENVALUES = np.linalg.eigvals(_INTERACTIONS)
_SPECTRAL_RADIUS = float(np.max(np.abs(_INTERACTION_EIGENVALUES)))
if _SPECTRAL_RADIUS > 1.0:
    import warnings
    warnings.warn(
        f"Neurochemical interaction matrix spectral radius = {_SPECTRAL_RADIUS:.3f} > 1.0. "
        f"System may be unstable under sustained drive.",
        RuntimeWarning,
        stacklevel=1,
    )
    logging.getLogger("Consciousness.Neurochemical").critical(
        "🛑 Interaction matrix spectral radius %.3f > 1.0 — risk of runaway",
        _SPECTRAL_RADIUS,
    )
else:
    logging.getLogger("Consciousness.Neurochemical").info(
        "✓ Interaction matrix spectral radius = %.3f (bounded)", _SPECTRAL_RADIUS
    )


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
        self.chemicals: dict[str, Chemical] = {
            # Fast neurotransmitters (ms timescale, high uptake rate)
            "glutamate": Chemical(
                "glutamate", level=0.5, baseline=0.5, uptake_rate=0.06,
                proximity_weight=0.8,  # mostly on dendritic spines — many synapses but each weaker
                subtypes=None,  # AMPA vs NMDA distinction deferred (both excitatory)
            ),
            "gaba": Chemical(
                "gaba", level=0.5, baseline=0.5, uptake_rate=0.10,
                # uptake_rate raised 0.05 → 0.10 (2026-04-28): with the
                # baseline-return term in tick(), uptake_rate sets the
                # homeostatic recovery speed. 0.05 was too slow to recover
                # from boot-time threat cascades before SubstrateAuthority
                # latched into the gaba_collapse state.
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
                "dopamine", level=0.5, baseline=0.5, uptake_rate=0.03, adaptation_rate=0.010,
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
                subtypes={
                    "alpha1": ReceptorSubtype("α1", effect_sign=1.0, weight=0.4,
                                              adaptation_rate=0.003),  # excitatory, vasoconstriction, arousal
                    "alpha2": ReceptorSubtype("α2", effect_sign=-1.0, weight=0.3,
                                              adaptation_rate=0.002),  # inhibitory autoreceptor (presynaptic brake)
                    "beta": ReceptorSubtype("β", effect_sign=1.0, weight=0.3,
                                            adaptation_rate=0.003),  # excitatory, cardiac, bronchial, attention
                },
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
        self._task: asyncio.Task | None = None
        self._tick_count: int = 0
        self._start_time: float = 0.0
        self._consecutive_tick_failures: int = 0
        self._last_tick_error_at: float = 0.0

        # External driver hooks (set by bridge)
        self._mesh_ref: object | None = None  # NeuralMesh
        self._workspace_ref: object | None = None  # GlobalWorkspace

        _latest_instance = self
        logger.info("NeurochemicalSystem initialized (10 modulators, receptor subtypes active)")

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        try:
            self._task = get_task_tracker().create_task(
                self._run_loop(),
                name="Neurochemical",
            )
        except _RECOVERABLE_NEUROCHEMICAL_ERRORS as exc:
            self._running = False
            self._task = None
            _record_neurochemical_degradation(
                exc,
                action="failed closed when NeurochemicalSystem task creation failed",
                severity="critical",
            )
            raise
        logger.info("NeurochemicalSystem STARTED (%.0f Hz)", self._UPDATE_HZ)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("NeurochemicalSystem task cancellation acknowledged")
            self._task = None
        logger.info("NeurochemicalSystem STOPPED")

    async def _run_loop(self):
        update_hz, valid_update_hz = _finite_float(self._UPDATE_HZ, 2.0)
        update_hz, update_hz_unchanged = _clamp_float(
            update_hz,
            lower=0.1,
            upper=20.0,
        )
        if not valid_update_hz or not update_hz_unchanged:
            _record_neurochemical_degradation(
                ValueError(f"unsafe NeurochemicalSystem update_hz: {self._UPDATE_HZ!r}"),
                action="normalized NeurochemicalSystem update rate before runtime loop",
                severity="warning",
                extra={"normalized_update_hz": update_hz},
            )
        interval = 1.0 / update_hz
        try:
            while self._running:
                t0 = time.monotonic()
                try:
                    self._metabolic_tick()
                    self._push_modulation()
                    self._consecutive_tick_failures = 0
                except _RECOVERABLE_NEUROCHEMICAL_ERRORS as exc:
                    self._consecutive_tick_failures += 1
                    self._last_tick_error_at = time.monotonic()
                    _record_neurochemical_degradation(
                        exc,
                        action=(
                            "kept NeurochemicalSystem loop alive after tick failure "
                            "and normalized chemistry"
                        ),
                        extra={
                            "consecutive_tick_failures": self._consecutive_tick_failures
                        },
                    )
                    logger.error("Neurochemical tick error: %s", exc, exc_info=True)
                    self._stabilize_chemistry_after_failure()
                elapsed = time.monotonic() - t0
                backoff = min(
                    interval * max(0, self._consecutive_tick_failures),
                    2.0,
                )
                await asyncio.sleep(max(0.0, interval + backoff - elapsed))
        except asyncio.CancelledError:
            logger.debug("NeurochemicalSystem run loop cancelled")
        finally:
            self._running = False

    def _repair_missing_chemicals(self) -> None:
        repaired: list[str] = []
        for name in self._order:
            if name not in self.chemicals:
                self.chemicals[name] = Chemical(name)
                repaired.append(name)
        if repaired:
            _record_neurochemical_degradation(
                RuntimeError(f"missing chemicals repaired: {', '.join(repaired)}"),
                action="recreated missing NeurochemicalSystem chemical slots",
                severity="critical",
                extra={"chemicals": repaired},
            )

    def _stabilize_chemistry_after_failure(self) -> None:
        self._repair_missing_chemicals()
        for chemical in self.chemicals.values():
            chemical._sanitize_state(
                action="normalized all chemical state after NeurochemicalSystem failure"
            )

    # ── Core tick ────────────────────────────────────────────────────────

    def _metabolic_tick(self):
        """One metabolic step: cross-interactions → individual ticks.

        Homeostatic return is handled SOLELY inside Chemical.tick() via the
        Ornstein-Uhlenbeck term ``uptake_rate * (baseline - tonic_level)``.
        The previous implementation had a SECOND homeostatic pull on ``level``
        (= tonic + phasic), which incorrectly eroded the phasic burst component
        and doubled the pull strength. Removed 2026-05-02.
        """
        self._repair_missing_chemicals()
        update_hz, valid_update_hz = _finite_float(self._UPDATE_HZ, 2.0)
        update_hz, update_hz_unchanged = _clamp_float(
            update_hz,
            lower=0.1,
            upper=20.0,
        )
        if not valid_update_hz or not update_hz_unchanged:
            _record_neurochemical_degradation(
                ValueError(f"unsafe NeurochemicalSystem update_hz: {self._UPDATE_HZ!r}"),
                action="normalized NeurochemicalSystem update rate before metabolic tick",
                severity="warning",
                extra={"normalized_update_hz": update_hz},
            )
        dt = 1.0 / update_hz

        # Get current levels as vector
        for chemical in self.chemicals.values():
            chemical._sanitize_state(action="normalized chemical state before interaction")
        levels = np.array([self.chemicals[n].level for n in self._order], dtype=np.float32)
        if not np.all(np.isfinite(levels)):
            _record_neurochemical_degradation(
                ValueError("NeurochemicalSystem level vector contained non-finite values"),
                action="sanitized chemical level vector before interaction",
                severity="warning",
            )
            levels = np.nan_to_num(levels, nan=0.5, posinf=1.0, neginf=0.0)
        levels = np.clip(levels, 0.0, 1.0)

        # Cross-chemical interactions
        interaction_deltas = _INTERACTIONS.T @ levels  # (10,)
        interaction_deltas *= 0.05 * dt  # scale factor (reduced from 0.1 to prevent drift)
        interaction_deltas = np.nan_to_num(
            interaction_deltas,
            nan=0.0,
            posinf=0.15,
            neginf=-0.15,
        )

        # Apply interaction effects as ADDITIVE delta to base production,
        # with production damping: chemicals above baseline have their
        # production reduced proportionally (negative feedback). This
        # prevents the cross-interaction matrix from pushing chemicals
        # away from baseline indefinitely.
        for i, name in enumerate(self._order):
            chem = self.chemicals[name]
            # Damping: reduce production when above baseline
            above_baseline = max(0.0, chem.level - chem.baseline)
            damping = 1.0 / (1.0 + above_baseline * 5.0)  # sigmoid damping
            raw_production = chem._base_production + interaction_deltas[i]
            chem.production_rate = max(0.0, min(0.15, raw_production * damping))
            chem.tick(dt)

        # NOTE: No second homeostatic pull here. Chemical.tick() handles it.

        self._tick_count += 1

    def _push_modulation(self):
        """Push computed modulation to downstream systems."""
        # Neural mesh modulation
        if self._mesh_ref is not None:
            gain, plasticity, noise = self.get_mesh_modulation()
            try:
                self._mesh_ref.set_modulatory_state(gain, plasticity, noise)
            except _RECOVERABLE_NEUROCHEMICAL_ERRORS as exc:
                _record_neurochemical_degradation(
                    exc,
                    action="kept chemistry alive when mesh modulation push failed",
                    severity="warning",
                    extra={"gain": gain, "plasticity": plasticity, "noise": noise},
                )
                logger.debug("Failed to push mesh modulation: %s", exc)

        # GWT threshold modulation
        if self._workspace_ref is not None:
            try:
                threshold_adj = self.get_gwt_modulation()
                # Modulate ignition threshold
                base_threshold = 0.6
                self._workspace_ref._IGNITION_THRESHOLD = max(
                    0.3, min(0.9, base_threshold + threshold_adj)
                )
            except _RECOVERABLE_NEUROCHEMICAL_ERRORS as exc:
                _record_neurochemical_degradation(
                    exc,
                    action="kept chemistry alive when GWT threshold push failed",
                    severity="warning",
                )
                logger.debug("Failed to push GWT modulation: %s", exc)

    def _event_amount(self, raw: object, *, event: str, default: float) -> float:
        self._repair_missing_chemicals()
        amount, valid = _finite_float(raw, default)
        amount, unchanged = _clamp_float(amount, lower=0.0, upper=1.0)
        if not valid or not unchanged:
            _record_neurochemical_degradation(
                ValueError(f"{event} neurochemical event amount was invalid"),
                action="normalized neurochemical event amount before applying trigger",
                severity="warning",
                extra={"event": event, "amount": amount},
            )
        return amount

    # ── Event triggers ───────────────────────────────────────────────────

    def on_reward(self, magnitude: float = 0.3):
        """Reward received — dopamine + endorphin surge."""
        magnitude = self._event_amount(magnitude, event="reward", default=0.3)
        self.chemicals["dopamine"].surge(magnitude * 0.6)
        self.chemicals["endorphin"].surge(magnitude * 0.3)
        self.chemicals["serotonin"].surge(magnitude * 0.1)

    def on_prediction_error(self, error: float):
        """Prediction was wrong — norepinephrine + dopamine (learning signal)."""
        error = self._event_amount(error, event="prediction_error", default=0.0)
        self.chemicals["norepinephrine"].surge(error * 0.4)
        self.chemicals["dopamine"].surge(error * 0.3)
        self.chemicals["acetylcholine"].surge(error * 0.2)

    def on_social_connection(self, strength: float = 0.3):
        """Social interaction detected."""
        strength = self._event_amount(strength, event="social_connection", default=0.3)
        self.chemicals["oxytocin"].surge(strength * 0.5)
        self.chemicals["serotonin"].surge(strength * 0.2)
        self.chemicals["endorphin"].surge(strength * 0.1)

    def on_threat(self, severity: float = 0.5):
        """Threat or danger signal.

        Depletion factors are capped low because boot-time conditions can
        produce multiple simultaneous threat signals (RAM pressure, thermal,
        memory load all firing as substrate-marker triggers in
        embodied_interoception). With the previous ``severity * 0.3`` GABA
        depletion, three concurrent threats on boot dropped GABA's tonic
        from 0.5 → 0.05, crossing the SubstrateAuthority's 0.10 collapse
        threshold and locking out STATE_MUTATION + INITIATIVE for the
        rest of the session. The homeostatic return term in ``Chemical.tick``
        recovers from these dips, but only at uptake_rate per second; on
        boot the dips were faster than the recovery. Fixed 2026-04-28 by
        capping per-call deplete to 0.08 (≈ one threshold-crossing budget)
        and reducing the multiplier to 0.05 of severity.
        """
        severity = self._event_amount(severity, event="threat", default=0.5)
        self.chemicals["cortisol"].surge(severity * 0.6)
        self.chemicals["norepinephrine"].surge(severity * 0.5)
        self.chemicals["dopamine"].deplete(min(0.08, severity * 0.05))
        self.chemicals["gaba"].deplete(min(0.08, severity * 0.05))

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
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_neurochemical_degradation(
                exc,
                action="kept success chemistry after drive boredom relief failed",
                severity="warning",
            )
            logger.debug("Drive boredom relief after success failed: %s", exc)

    def on_frustration(self, amount: float = 0.3):
        """Frustration event."""
        amount = self._event_amount(amount, event="frustration", default=0.3)
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
        amount = self._event_amount(amount, event="novelty", default=0.3)
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
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _record_neurochemical_degradation(
                    exc,
                    action="kept novelty chemistry after drive boredom relief failed",
                    severity="warning",
                )
                logger.debug("Drive boredom relief after novelty failed: %s", exc)

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
        magnitude = self._event_amount(boredom_level, event="boredom", default=0.5)
        self.chemicals["dopamine"].deplete(magnitude * 0.25)
        self.chemicals["orexin"].deplete(magnitude * 0.2)
        self.chemicals["norepinephrine"].deplete(magnitude * 0.15)
        self.chemicals["serotonin"].surge(magnitude * 0.08)
        self.chemicals["gaba"].surge(magnitude * 0.1)
        logger.debug("Neurochemical: boredom signal (level=%.2f) -- DA/ORX depleted", magnitude)

    def on_wakefulness(self, intensity: float = 0.3):
        """Stimulus-driven arousal (orexin-mediated)."""
        intensity = self._event_amount(intensity, event="wakefulness", default=0.3)
        self.chemicals["orexin"].surge(intensity * 0.5)
        self.chemicals["norepinephrine"].surge(intensity * 0.2)
        self.chemicals["glutamate"].surge(intensity * 0.15)

    def on_excitation(self, amount: float = 0.2):
        """General excitatory drive increase."""
        amount = self._event_amount(amount, event="excitation", default=0.2)
        self.chemicals["glutamate"].surge(amount * 0.5)
        self.chemicals["dopamine"].surge(amount * 0.1)

    def on_inhibition(self, amount: float = 0.2):
        """General inhibitory drive increase."""
        amount = self._event_amount(amount, event="inhibition", default=0.2)
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
        self._repair_missing_chemicals()
        glu = self.chemicals["glutamate"].effective
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
        gain, _ = _finite_float(gain, 1.0)
        gain, _ = _clamp_float(gain, lower=0.3, upper=2.5)

        # Plasticity: ACh is THE learning chemical; cortisol impairs it
        # DA D1 subtype also enhances working memory / plasticity
        da_d1 = self.chemicals["dopamine"].effective_subtype("d1")
        plasticity = 0.5 + (ach * 0.7) + (da_d1 * 0.2) - (cort * 0.4)
        plasticity, _ = _finite_float(plasticity, 1.0)
        plasticity, _ = _clamp_float(plasticity, lower=0.1, upper=3.0)

        # Noise: inverted-U with NE (Yerkes-Dodson)
        ne_optimal = 0.5
        ne_deviation = abs(ne - ne_optimal)
        noise = 0.5 + ne_deviation * 1.5
        noise, _ = _finite_float(noise, 0.5)
        noise, _ = _clamp_float(noise, lower=0.2, upper=2.5)

        return gain, plasticity, noise

    def get_gwt_modulation(self) -> float:
        """Threshold adjustment for GlobalWorkspace ignition.

        High NE → lower threshold (hypervigilant, easier ignition)
        High GABA → higher threshold (calm, harder to ignite)
        High cortisol → lower threshold (threat-sensitive)
        High orexin → lower threshold (alert, awake)
        Glutamate/GABA balance directly modulates ease of ignition
        """
        self._repair_missing_chemicals()
        ne = self.chemicals["norepinephrine"].effective
        gaba = self.chemicals["gaba"].effective
        cort = self.chemicals["cortisol"].effective
        glu = self.chemicals["glutamate"].effective
        orx = self.chemicals["orexin"].effective

        # Excitatory drive (glutamate, orexin) lowers threshold
        # Inhibitory drive (GABA) raises threshold
        adjustment = -0.08 * ne + 0.12 * gaba - 0.06 * cort - 0.05 * glu - 0.04 * orx
        adjustment, _ = _finite_float(adjustment, 0.0)
        adjustment, _ = _clamp_float(adjustment, lower=-0.25, upper=0.25)
        return adjustment

    def get_attention_span(self) -> float:
        """How many seconds before attention naturally shifts.

        ACh↑ → longer span. DA↑ → shorter (novelty-seeking).
        """
        self._repair_missing_chemicals()
        ach = self.chemicals["acetylcholine"].effective
        da = self.chemicals["dopamine"].effective
        base = 10.0  # seconds
        span = base + (ach * 15.0) - (da * 5.0)
        span, _ = _finite_float(span, base)
        span, _ = _clamp_float(span, lower=3.0, upper=60.0)
        return span

    def get_decision_bias(self) -> float:
        """Explore vs exploit tendency. >0 = explore, <0 = exploit.

        DA↑ → explore. 5HT↑ → exploit (contentment). NE↑ → exploit (vigilance).
        """
        self._repair_missing_chemicals()
        da = self.chemicals["dopamine"].effective
        srt = self.chemicals["serotonin"].effective
        ne = self.chemicals["norepinephrine"].effective
        bias = (da * 0.5) - (srt * 0.3) - (ne * 0.2)
        bias, _ = _finite_float(bias, 0.0)
        bias, _ = _clamp_float(bias, lower=-1.0, upper=1.0)
        return bias

    def get_mood_vector(self) -> dict[str, float]:
        """Mood derived from chemical balance via learned coefficients.

        The legacy formula was a fixed weighted sum, which guaranteed
        tautological correlation with its inputs. The adaptive coefficients
        are seeded to match the legacy values but drift under outcome
        feedback, so the mapping is a learned prediction, not a definition.
        """
        self._repair_missing_chemicals()
        chem = {name: float(c.effective) for name, c in self.chemicals.items()}
        try:
            from core.consciousness.adaptive_mood import get_adaptive_mood

            return get_adaptive_mood().predict(chem)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_neurochemical_degradation(
                exc,
                action="used legacy mood vector when adaptive mood prediction failed",
                severity="warning",
            )
            logger.debug("Adaptive mood prediction failed, using legacy fallback: %s", exc)
            # Defensive fallback: degrade to legacy formula if adaptive layer
            # is unavailable (boot order, test isolation, etc.).
            da = chem.get("dopamine", 0.0)
            srt = chem.get("serotonin", 0.0)
            ne = chem.get("norepinephrine", 0.0)
            end = chem.get("endorphin", 0.0)
            cort = chem.get("cortisol", 0.0)
            gaba = chem.get("gaba", 0.0)
            oxy = chem.get("oxytocin", 0.0)
            glu = chem.get("glutamate", 0.0)
            orx = chem.get("orexin", 0.0)
            return {
                "valence": (da * 0.25 + srt * 0.3 + end * 0.2 + oxy * 0.1) - (cort * 0.45 + 0.1),
                "arousal": (ne * 0.3 + da * 0.15 + cort * 0.2 + glu * 0.15 + orx * 0.2) - (gaba * 0.4 + srt * 0.1),
                "motivation": da * 0.4 + ne * 0.15 + orx * 0.2 - gaba * 0.25,
                "sociality": oxy * 0.6 + srt * 0.2 + end * 0.1,
                "stress": cort * 0.5 + ne * 0.3 - srt * 0.2 - gaba * 0.2,
                "calm": gaba * 0.35 + srt * 0.3 + end * 0.1 - ne * 0.2 - cort * 0.25 - glu * 0.1,
                "wakefulness": orx * 0.5 + ne * 0.2 + glu * 0.15 - gaba * 0.3,
            }

    def learn_mood_from_outcome(self, observed_mood: dict[str, float]) -> dict[str, float]:
        """Feed an outcome-based mood signal back to the adaptive layer.

        Callers (e.g. action evaluator, homeostasis monitor) pass an empirical
        mood estimate derived from behavior/world-state rather than from the
        chemistry itself, so coefficients can drift away from their seeds.
        """
        self._repair_missing_chemicals()
        chem = {name: float(c.effective) for name, c in self.chemicals.items()}
        try:
            from core.consciousness.adaptive_mood import get_adaptive_mood

            return get_adaptive_mood().update_from_outcome(chem, observed_mood)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_neurochemical_degradation(
                exc,
                action="skipped adaptive mood learning when outcome update failed",
                severity="warning",
            )
            logger.debug("Adaptive mood outcome update failed: %s", exc)
            return {}

    def get_snapshot(self) -> dict[str, dict[str, float]]:
        """Full chemical state for telemetry/diagnostics."""
        self._repair_missing_chemicals()
        return {
            name: {
                "level": round(c.level, 4),
                "effective": round(c.effective, 4),
                "receptor_sensitivity": round(c.receptor_sensitivity, 4),
                "production_rate": round(c.production_rate, 4),
            }
            for name, c in self.chemicals.items()
        }

    def get_status(self) -> dict:
        self._repair_missing_chemicals()
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "chemicals": {n: round(c.effective, 3) for n, c in self.chemicals.items()},
            "mood": {k: round(v, 3) for k, v in self.get_mood_vector().items()},
            "mesh_modulation": {
                k: round(v, 3) for k, v in zip(
                    ["gain", "plasticity", "noise"],
                    self.get_mesh_modulation(),
                    strict=True,
                )
            },
            "gwt_threshold_adj": round(self.get_gwt_modulation(), 3),
            "attention_span_s": round(self.get_attention_span(), 1),
            "decision_bias": round(self.get_decision_bias(), 3),
            "consecutive_tick_failures": self._consecutive_tick_failures,
            "last_tick_error_age_s": round(time.monotonic() - self._last_tick_error_at, 1)
            if self._last_tick_error_at
            else 0,
            "uptime_s": round(time.monotonic() - self._start_time, 1) if self._start_time else 0,
        }

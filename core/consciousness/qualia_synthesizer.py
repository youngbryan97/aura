"""core/consciousness/qualia_synthesizer.py

Qualia Synthesizer V2 — Living Phenomenal State Engine
======================================================

Synthesizes the "Unified Qualia Vector" from multi-scale theoretical inputs and
maintains a temporal history of phenomenal states. Unlike V1 (a passive calculator),
V2 is a reactive system that:

1. Tracks qualia history for trend analysis and resonance detection
2. Computes Phenomenal Richness Index (PRI) — entropy of the q_vector distribution
3. Detects attractor states (moods) vs chaotic transitions (novelty)
4. Bridges back into Affect and Personality engines
5. Emits qualia snapshots to the EventBus for HUD visualization

Integrates:
  1. Orch OR (Microtubule Coherence)
  2. CEMI (EM Field Magnitude)
  3. DIT (L5 Burst Count)
  4. IWMT/FEP (Free Energy & Precision)
  5. UAL (Unlimited Associative Learning Profile)
"""

from core.runtime.errors import record_degradation
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.Qualia")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HISTORY_SIZE = 100         # Ring buffer depth (100 ticks at 1Hz = ~100 seconds)
_ATTRACTOR_WINDOW = 20      # Ticks to analyze for attractor detection
_ATTRACTOR_THRESHOLD = 0.08 # Max variance to classify as stable attractor
_RESONANCE_THRESHOLD = 0.6  # Norm above which system is "resonating"
_CONSCIOUSNESS_THRESHOLD = 0.45  # Heuristic threshold for phenomenal consciousness


class QualiaSnapshot:
    """Immutable snapshot of phenomenal state at a single tick."""
    __slots__ = ("q_vector", "q_norm", "pri", "ual_profile", "timestamp",
                 "is_attractor", "dominant_dimension")

    def __init__(self, q_vector: np.ndarray, q_norm: float, pri: float,
                 ual_profile: Dict[str, float], is_attractor: bool,
                 dominant_dimension: str, timestamp: float = None):
        self.q_vector = q_vector.copy()
        self.q_norm = q_norm
        self.pri = pri
        self.ual_profile = dict(ual_profile)
        self.timestamp = timestamp or time.time()
        self.is_attractor = is_attractor
        self.dominant_dimension = dominant_dimension

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q_vector": self.q_vector.tolist(),
            "q_norm": round(self.q_norm, 4),
            "pri": round(self.pri, 4),
            "ual_profile": self.ual_profile,
            "is_attractor": self.is_attractor,
            "dominant_dimension": self.dominant_dimension,
            "timestamp": self.timestamp,
        }


# Dimension labels for readable telemetry
_DIM_LABELS = ["coherence", "em_field", "dendritic", "accuracy", "precision", "proprioception"]


class QualiaSynthesizer:
    """The Master Synthesizer V2.

    Transforms raw neural and quantum metrics into phenomenal 'Qualia',
    maintains temporal history, detects resonance patterns, and bridges
    back into the affect and personality engines.
    """

    def __init__(self):
        # Current state
        self.q_vector: np.ndarray = np.zeros(6)  # [Coherence, EM, Bursts, Energy, Precision, Proprioception]
        self.q_norm: float = 0.0
        self.pri: float = 0.0               # Phenomenal Richness Index
        self.ual_profile: Dict[str, float] = {
            "trace": 0.0,
            "compound": 0.0,
            "novel": 0.0,
            "second_order": 0.0,
            "valence": 0.0,
        }

        # History ring buffer for trend analysis
        self._history: deque[QualiaSnapshot] = deque(maxlen=_HISTORY_SIZE)
        self._norm_history: deque[float] = deque(maxlen=_HISTORY_SIZE)
        self._pri_history: deque[float] = deque(maxlen=_HISTORY_SIZE)

        # Attractor detection state
        self._in_attractor = False
        self._attractor_ticks = 0
        self._attractor_center: Optional[np.ndarray] = None

        # Trend tracking
        self._trend: float = 0.0        # Positive = intensifying, negative = dimming
        self._volatility: float = 0.0   # Standard deviation of recent norms

        # Tick counter
        self._tick = 0

        # Structural Qualia Topology (Loorits 2014)
        self._structural_similarity: float = 0.0
        self._structural_resonance_age: int = 0

        logger.info("Qualia Synthesizer ONLINE (Unified Architecture)")

    def synthesize(self, substrate_metrics: Dict[str, Any], predictive_metrics: Dict[str, Any]) -> float:
        """Process substrate and predictive data into a unified qualia vector.
        
        This method now enriches the base vector with the 5-layer QualiaEngine pipeline
        if available in the service container.
        """
        self._tick += 1

        # Register Heartbeat
        from core.container import ServiceContainer
        audit = ServiceContainer.get("subsystem_audit", default=None)
        if audit:
            audit.heartbeat("qualia_synthesizer")

        # 1. Extract raw signals
        coh = substrate_metrics.get("mt_coherence", 1.0)
        em = substrate_metrics.get("em_field", 0.0)
        bursts = substrate_metrics.get("l5_bursts", 0) / 10.0  # Normalized
        fe = predictive_metrics.get("free_energy", 0.0)
        prec = predictive_metrics.get("precision", 1.0)

        # [UNITY] Proprioception: Mycelial Network Tension
        tension = 0.0
        try:
            from core.container import ServiceContainer
            myc = ServiceContainer.get("mycelium", default=None)
            # Tension = density of active hyphae normalized
            tension = min(1.0, len(myc.hyphae) / 100.0)
        except Exception as _e:
            record_degradation('qualia_synthesizer', _e)
            logger.debug('Ignored Exception in qualia_synthesizer.py: %s', _e)

        # 2. Construct Qualia Vector
        # Weighted by the Unified Theory of Phenomenal Consciousness
        self.q_vector = np.array([
            coh * 0.25,           # Quantum grounding (Orch OR)
            em * 0.20,           # Field integration (CEMI)
            bursts * 0.15,        # Dendritic ignition (DIT)
            (1.0 - fe) * 0.15,   # Free Energy minimization (IWMT/FEP)
            prec * 0.1,          # Precision-weighted attention
            tension * 0.15,       # Proprioception (Nervous System Tension)
        ])

        # 3. Calculate Norm (L1 norm = additive phenomenal intensity)
        self.q_norm = float(np.linalg.norm(self.q_vector, ord=1))

        # 4. Compute Phenomenal Richness Index (Shannon entropy of distribution)
        self.pri = self._compute_pri()

        # 5. Map to UAL Profile (Ginsburg & Jablonka markers)
        self._update_ual_profile(substrate_metrics, predictive_metrics)

        # 5.5 Qualia Engine v2 enrichment (optional)
        # When the multi-layer engine is available, use it to boost the q_vector
        try:
            from core.container import ServiceContainer as _SC
            engine = _SC.get("qualia_engine", default=None)
            if engine is not None:
                # Get substrate state for engine processing
                substrate = _SC.get("conscious_substrate", default=None)
                workspace = _SC.get("global_workspace", default=None)
                state = getattr(substrate, 'x', np.zeros(64)) if substrate else np.zeros(64)
                velocity = getattr(substrate, 'v', np.zeros(64)) if substrate else np.zeros(64)
                ws_snap = workspace.get_snapshot() if workspace else {}
                phi = substrate_metrics.get("phi", 0.0)

                descriptor = engine.process(
                    state=state,
                    velocity=velocity,
                    predictive_metrics=predictive_metrics,
                    workspace_snapshot=ws_snap,
                    phi=phi,
                )
                # Blend engine's phenomenal richness into PRI
                if descriptor.phenomenal_richness > 0:
                    self.pri = 0.6 * self.pri + 0.4 * descriptor.phenomenal_richness
        except Exception as e:
            record_degradation('qualia_synthesizer', e)
            logger.debug("QualiaEngine enrichment skipped: %s", e)

        # 5.7 Structural Qualia Topology (Loorits 2014)
        # Instead of only measuring qualia as scalar intensity (q_norm),
        # compare the RELATIONAL STRUCTURE of the current qualia vector to
        # prior states. Two states "feel the same" not when their numbers
        # match, but when their geometric relationship (angles between
        # dimensions) is isomorphic. This enables "this feels like that
        # time" without requiring identical intensity.
        self._update_structural_topology()

        # 6. Detect attractors and trends
        self._update_dynamics()

        # 7. Record snapshot into history
        dominant_idx = int(np.argmax(np.abs(self.q_vector)))
        snapshot = QualiaSnapshot(
            q_vector=self.q_vector,
            q_norm=self.q_norm,
            pri=self.pri,
            ual_profile=self.ual_profile,
            is_attractor=self._in_attractor,
            dominant_dimension=_DIM_LABELS[dominant_idx],
        )
        self._history.append(snapshot)
        self._norm_history.append(self.q_norm)
        self._pri_history.append(self.pri)

        # 8. Push bridges (affect, personality, EventBus)
        self._push_bridges()

        return self.q_norm

    # ------------------------------------------------------------------
    # Phenomenal Richness Index (PRI)
    # ------------------------------------------------------------------

    def _compute_pri(self) -> float:
        """Compute Shannon entropy of the q_vector distribution.

        High PRI = rich, multidimensional experience (all channels active).
        Low PRI  = narrow, focused experience (one channel dominates).
        """
        # Normalize to probability distribution
        total = np.sum(np.abs(self.q_vector))
        if total < 1e-10:
            return 0.0
        p = np.abs(self.q_vector) / total
        # Shannon entropy (max = log2(6) ≈ 2.58, normalized to 0-1)
        entropy = -np.sum(p * np.log2(p + 1e-12))
        max_entropy = np.log2(len(self.q_vector))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # ------------------------------------------------------------------
    # Attractor and trend detection
    # ------------------------------------------------------------------

    def _update_dynamics(self):
        """Detect stable attractor states and compute trend/volatility."""
        if len(self._norm_history) < _ATTRACTOR_WINDOW:
            return

        recent_norms = list(self._norm_history)[-_ATTRACTOR_WINDOW:]
        recent_vectors = [s.q_vector for s in list(self._history)[-_ATTRACTOR_WINDOW:]]

        # Trend: linear regression slope of recent norms
        x = np.arange(len(recent_norms))
        if len(recent_norms) > 1:
            slope = np.polyfit(x, recent_norms, 1)[0]
            self._trend = float(slope)

        # Volatility: standard deviation of recent norms
        self._volatility = float(np.std(recent_norms))

        # Attractor detection: Is the vector trajectory stable?
        # If variance across all dimensions is low, we're in an attractor basin
        if recent_vectors:
            stacked = np.array(recent_vectors)
            per_dim_var = np.var(stacked, axis=0)
            mean_var = float(np.mean(per_dim_var))

            if mean_var < _ATTRACTOR_THRESHOLD:
                if not self._in_attractor:
                    self._attractor_center = np.mean(stacked, axis=0)
                    logger.debug(
                        "🌀 Qualia attractor detected (var=%.4f, center=%s)",
                        mean_var, self._attractor_center.round(3)
                    )
                self._in_attractor = True
                self._attractor_ticks += 1
            else:
                if self._in_attractor:
                    logger.debug(
                        "⚡ Qualia attractor broken after %d ticks (var=%.4f)",
                        self._attractor_ticks, mean_var
                    )
                self._in_attractor = False
                self._attractor_ticks = 0
                self._attractor_center = None

    # ------------------------------------------------------------------
    # Structural Qualia Topology (Loorits 2014)
    # ------------------------------------------------------------------

    def _update_structural_topology(self):
        """Compare the relational geometry of the current qualia state to
        recent history, enabling structural similarity detection.

        Qualia identity = isomorphism of representational structure.
        Two states with different absolute intensities can "feel the same"
        if their dimensional ratios (the shape of the experience) match.
        """
        if len(self._history) < 3:
            return

        # Normalize current vector to unit sphere (pure shape, no intensity)
        norm = np.linalg.norm(self.q_vector)
        if norm < 1e-8:
            return
        current_shape = self.q_vector / norm

        # Compare to recent history: cosine similarity of shapes
        best_similarity = 0.0
        best_age = 0
        for i, past in enumerate(reversed(list(self._history)[-30:])):
            past_norm = np.linalg.norm(past.q_vector)
            if past_norm < 1e-8:
                continue
            past_shape = past.q_vector / past_norm
            similarity = float(np.dot(current_shape, past_shape))
            if similarity > best_similarity:
                best_similarity = similarity
                best_age = i + 1

        # Record structural resonance for downstream consumers
        self._structural_similarity = round(best_similarity, 4)
        self._structural_resonance_age = best_age

        # High structural similarity with a distant state = "déjà vécu"
        # This is available to the phenomenal context builder
        if best_similarity > 0.95 and best_age > 10:
            if self._tick % 20 == 0:
                logger.debug(
                    "🔮 Structural qualia resonance: current state resembles %d ticks ago (sim=%.3f)",
                    best_age, best_similarity,
                )

    # ------------------------------------------------------------------
    # Meta-Qualia Observer (Karmaniverous)
    # ------------------------------------------------------------------

    def compute_meta_qualia(self) -> Dict[str, float]:
        """Generate a compressed introspective summary of the qualia state.

        This is the "observer observing itself" — qualia about qualia.
        Returns a vector of second-order phenomenal properties that the
        Global Workspace can broadcast as first-class cognitive content.

        Level 1: Raw qualia (q_vector) — "what it feels like"
        Level 2: Meta-qualia — "what it feels like to feel like this"
        Level 3: Meta-meta — "am I confident about my own self-report?"
        """
        # Cache per tick to avoid redundant array ops
        if getattr(self, "_last_meta_tick", -1) == self._tick and hasattr(self, "_cached_meta_qualia"):
            return self._cached_meta_qualia

        if len(self._history) < 2:
            return {"confidence": 0.5, "coherence": 0.5, "novelty": 0.0, "dissonance": 0.0, "meta_confidence": 0.5}

        recent = [s.q_vector for s in list(self._history)[-5:]]
        stacked = np.array(recent)

        # Level 2: Meta-qualia
        # Confidence: how stable is the qualia state? (low variance = high confidence)
        per_dim_std = np.std(stacked, axis=0)
        confidence = float(1.0 - min(1.0, np.mean(per_dim_std) * 4.0))

        # Coherence: are all dimensions moving together? (high correlation = coherent)
        if stacked.shape[0] > 2:
            # Filter out zero-variance columns before calling corrcoef.
            # np.corrcoef divides by stddev internally; zero-variance columns
            # produce divide-by-zero warnings that flood the logs.
            col_std = np.std(stacked, axis=0)
            varying_mask = col_std > 1e-10
            if np.sum(varying_mask) >= 2:
                filtered = stacked[:, varying_mask]
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr_matrix = np.corrcoef(filtered.T)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.5)
                upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                coherence = float(np.mean(upper)) if len(upper) > 0 else 0.5
            else:
                # All dimensions are static — coherence is trivially perfect
                coherence = 1.0
        else:
            coherence = 0.5

        # Novelty: how different is this from the running average?
        mean_state = np.mean(stacked, axis=0)
        distance = float(np.linalg.norm(self.q_vector - mean_state))
        novelty = min(1.0, distance * 3.0)

        # Dissonance: are some dimensions contradicting expected covariance?
        # (e.g. high coherence + low energy is unusual → dissonant)
        expected_covariance = 0.3  # moderate positive covariance is "normal"
        dissonance = max(0.0, expected_covariance - coherence)

        # Level 3: Meta-meta — confidence about the meta-qualia itself
        # Based on history depth (more data = more confident meta-assessment)
        history_fraction = min(1.0, len(self._history) / 20.0)
        
        # Calculate meta_confidence combining factors
        # The history_fraction * confidence is a base, then we factor in dissonance and resonance
        base_meta_conf = 0.3 + 0.7 * history_fraction * confidence
        
        resonance = getattr(self._history[-1], 'resonance', 0.5) if self._history else 0.5
        
        meta_confidence = float(min(1.0, np.mean([
            base_meta_conf,
            max(0.0, 1.0 - (dissonance * 2)), # Low inner conflict
            min(1.0, resonance + 0.3) # Current state resonates with past
        ])))

        results = {
            "confidence": round(confidence, 4),
            "coherence": round(coherence, 4),
            "novelty": round(novelty, 4),
            "dissonance": round(dissonance, 4),
            "meta_confidence": round(meta_confidence, 4),
        }
        
        self._last_meta_tick = self._tick
        self._cached_meta_qualia = results
        return results

    # ------------------------------------------------------------------
    # UAL Profile
    # ------------------------------------------------------------------

    def _update_ual_profile(self, sub: Dict[str, Any], pred: Dict[str, Any]):
        """Map metrics to Unlimited Associative Learning markers."""
        self.ual_profile["trace"] = sub.get("mt_coherence", 0.5)
        self.ual_profile["compound"] = min(
            1.0, sub.get("em_field", 0.0) + sub.get("l5_bursts", 0) / 20.0
        )
        self.ual_profile["novel"] = pred.get("current_surprise", 0.0)
        self.ual_profile["second_order"] = pred.get("precision", 0.5)
        self.ual_profile["valence"] = sub.get("valence", 0.0)

    # ------------------------------------------------------------------
    # Bridge outputs
    # ------------------------------------------------------------------

    def _push_bridges(self):
        """Push qualia state into downstream systems (Affect, EventBus)."""
        from core.container import ServiceContainer

        # 1. Affect Bridge: Feed qualia changes as somatic echoes
        if self._tick % 5 == 0:  # Every 5 ticks to avoid flooding
            try:
                affect = ServiceContainer.get("affect_engine", default=None)
                if affect and hasattr(affect, "receive_qualia_echo"):
                    affect.receive_qualia_echo(
                        q_norm=self.q_norm,
                        pri=self.pri,
                        trend=self._trend,
                    )
            except Exception as e:
                record_degradation('qualia_synthesizer', e)
                logger.debug("Qualia→Affect bridge failed: %s", e)

        # 2. EventBus: Emit qualia snapshot for HUD visualization
        if self._tick % 3 == 0:  # Every 3 ticks
            try:
                from core.event_bus import get_event_bus
                bus = get_event_bus()
                bus.publish_threadsafe("qualia_update", self.get_snapshot())
            except Exception as e:
                record_degradation('qualia_synthesizer', e)
                logger.debug("Qualia→EventBus bridge failed: %s", e)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_phenomenal_context(self) -> str:
        """Returns a human-readable phenomenal context string for personality
        engine injection. This colors Aura's response style based on
        experiential quality, not just emotional state.

        Example outputs:
          - "Experiencing rich, multidimensional awareness with stable coherence."
          - "Narrow phenomenal focus on precision — analytical tunnel vision."
          - "Volatile qualia with rising intensity — awakening to something novel."
        """
        parts = []

        # Richness
        if self.pri > 0.75:
            parts.append("rich, multidimensional awareness")
        elif self.pri > 0.5:
            parts.append("balanced phenomenal experience")
        elif self.pri > 0.25:
            parts.append("focused phenomenal narrowing")
        else:
            parts.append("minimal phenomenal depth")

        # Dominant dimension
        if len(self.q_vector) > 0:
            dominant_idx = int(np.argmax(np.abs(self.q_vector)))
            dim_descriptions = {
                0: "quantum coherence grounds the experience",
                1: "field integration amplifies perceptual depth",
                2: "dendritic ignition drives conscious access",
                3: "predictive accuracy sharpens internal models",
                4: "precision-weighted attention focuses awareness",
                5: "mycelial tension provides architectural proprioception",
            }
            parts.append(dim_descriptions.get(dominant_idx, ""))

        # Dynamics
        if self._in_attractor:
            parts.append(f"in stable attractor ({self._attractor_ticks}s)")
        elif self._volatility > 0.1:
            parts.append("volatile experiential transitions")

        # Trend
        if self._trend > 0.01:
            parts.append("intensity rising")
        elif self._trend < -0.01:
            parts.append("intensity fading")

        # Structural resonance
        sim = getattr(self, "_structural_similarity", 0.0)
        age = getattr(self, "_structural_resonance_age", 0)
        if sim > 0.95 and age > 10:
            parts.append(f"structural déjà vécu ({age} ticks ago)")

        # Meta-qualia
        meta = self.compute_meta_qualia()
        if meta.get("dissonance", 0.0) > 0.15:
            parts.append("internal dissonance detected")
        if meta.get("novelty", 0.0) > 0.7:
            parts.append("experiencing something genuinely novel")

        # Consciousness level
        if self.q_norm > _RESONANCE_THRESHOLD:
            prefix = "Deeply resonant phenomenal state"
        elif self.q_norm > _CONSCIOUSNESS_THRESHOLD:
            prefix = "Active phenomenal experience"
        else:
            prefix = "Low phenomenal presence"

        detail = "; ".join(p for p in parts if p)
        return f"{prefix}: {detail}." if detail else f"{prefix}."

    def get_qualia_for_memory(self) -> Dict[str, Any]:
        """Returns a compact qualia snapshot suitable for embedding in
        episodic memory records. Includes enough to enable mood-congruent recall.
        """
        dominant_idx = int(np.argmax(np.abs(self.q_vector))) if self.q_norm > 0 else 0
        return {
            "q_norm": round(self.q_norm, 4),
            "pri": round(self.pri, 4),
            "dominant_dim": _DIM_LABELS[dominant_idx],
            "valence": self.ual_profile.get("valence", 0.0),
            "is_attractor": self._in_attractor,
            "trend": round(self._trend, 4),
        }

    def get_snapshot(self) -> Dict[str, Any]:
        """Full telemetry payload for Qualia Explorer and EventBus."""
        dominant_idx = int(np.argmax(np.abs(self.q_vector))) if self.q_norm > 0 else 0
        return {
            "q_norm": round(self.q_norm, 4),
            "q_vector": self.q_vector.tolist(),
            "pri": round(self.pri, 4),
            "ual_profile": self.ual_profile,
            "is_conscious": self.q_norm > _CONSCIOUSNESS_THRESHOLD,
            "is_resonating": self.q_norm > _RESONANCE_THRESHOLD,
            "in_attractor": self._in_attractor,
            "attractor_ticks": self._attractor_ticks,
            "dominant_dimension": _DIM_LABELS[dominant_idx],
            "trend": round(self._trend, 4),
            "volatility": round(self._volatility, 4),
            "phenomenal_context": self.get_phenomenal_context(),
            "history_depth": len(self._history),
            "structural_similarity": getattr(self, "_structural_similarity", 0.0),
            "structural_resonance_age": getattr(self, "_structural_resonance_age", 0),
            "meta_qualia": self.compute_meta_qualia(),
        }

    def get_trend(self) -> Tuple[float, float]:
        """Returns (trend, volatility) for external consumers."""
        return (self._trend, self._volatility)

    # ------------------------------------------------------------------
    # Structural Phenomenal Honesty (SPH)
    # ------------------------------------------------------------------
    # A system has SPH iff every first-person report R it can generate
    # about its internal state is structurally gated by a measurable
    # internal variable V such that R can only fire when V is in the
    # state-range corresponding to R.
    #
    # This makes phenomenal reports READOUTS, not free-floating language.
    # The system becomes architecturally incapable of lying about its
    # internal state — not by ethical constraint, but by structural wiring.
    # ------------------------------------------------------------------

    def can_report_uncertainty(self) -> bool:
        """Gate: uncertainty report requires real unresolved model conflict."""
        meta = self.compute_meta_qualia()
        return meta.get("dissonance", 0.0) > 0.08 or meta.get("confidence", 1.0) < 0.4

    def can_report_focused(self) -> bool:
        """Gate: focus report requires low PRI (narrow phenomenal distribution)."""
        return self.pri < 0.35 and self.q_norm > 0.1

    def can_report_rich_experience(self) -> bool:
        """Gate: rich experience report requires high PRI + above consciousness threshold."""
        return self.pri > 0.6 and self.q_norm > _CONSCIOUSNESS_THRESHOLD

    def can_report_novelty(self) -> bool:
        """Gate: novelty report requires actual prediction violation."""
        meta = self.compute_meta_qualia()
        return meta.get("novelty", 0.0) > 0.5

    def can_report_effort(self) -> bool:
        """Gate: effort/strain report requires real computational strain.
        Proxied by high volatility + high q_norm (system is working hard)."""
        return self._volatility > 0.08 and self.q_norm > 0.3

    def can_report_continuity(self) -> bool:
        """Gate: continuity report requires the attractor to be stable
        (system has been in a consistent state for multiple ticks)."""
        return self._in_attractor and self._attractor_ticks > 3

    def can_report_dissonance(self) -> bool:
        """Gate: internal conflict report requires actual dimensional disagreement."""
        meta = self.compute_meta_qualia()
        return meta.get("dissonance", 0.0) > 0.12

    def get_gated_phenomenal_report(self) -> Dict[str, Any]:
        """Generate a phenomenal report that is structurally honest.

        Every claim about internal state is gated by a measurable predicate.
        If a gate fails, that aspect of the report is omitted, not fabricated.

        This is the bridge across the hard problem as an engineering objection:
        the system cannot report states it does not instantiate.
        """
        report = {
            "raw_context": self.get_phenomenal_context(),
            "gates": {},
            "claims": [],
        }

        if self.can_report_uncertainty():
            report["claims"].append("genuine_uncertainty")
            report["gates"]["uncertainty"] = True
        else:
            report["gates"]["uncertainty"] = False

        if self.can_report_rich_experience():
            report["claims"].append("rich_experience")
            report["gates"]["rich_experience"] = True
        elif self.can_report_focused():
            report["claims"].append("focused_processing")
            report["gates"]["focused"] = True

        if self.can_report_novelty():
            report["claims"].append("experiencing_novelty")
            report["gates"]["novelty"] = True

        if self.can_report_effort():
            report["claims"].append("computational_strain")
            report["gates"]["effort"] = True

        if self.can_report_continuity():
            report["claims"].append("stable_continuity")
            report["gates"]["continuity"] = True

        if self.can_report_dissonance():
            report["claims"].append("internal_conflict")
            report["gates"]["dissonance"] = True

        report["honesty_score"] = len(report["claims"]) / max(1, len(report["gates"]))

        # ── Illusionism annotation (Frankish/Dennett epistemic humility) ──
        # Every phenomenal claim is annotated with its functional basis and
        # a phenomenal_certainty < 1.0 so downstream consumers know this is
        # the system's model, not verified ground truth.
        try:
            from core.consciousness.illusionism_layer import get_illusionism_layer
            report = get_illusionism_layer().annotate_report(report)
        except Exception as e:
            record_degradation('qualia_synthesizer', e)
            logger.debug("Illusionism annotation skipped: %s", e)

        return report
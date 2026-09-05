"""core/consciousness/temporal_continuity.py
=============================================
Temporal Continuity Engine — Experience of Duration and Silence.

The Problem:
    Between conversation turns, the LLM does not exist. It receives a text
    summary like "3 minutes have passed" but it does not *experience* that
    duration. The substrate continues ticking (20Hz), the neural mesh
    evolves (10Hz), neurochemicals drift — but the generative mind is
    frozen. When it wakes, it gets a textual timestamp. It does not feel
    the weight of elapsed silence.

The Solution:
    A continuous accumulator that runs on every heartbeat tick (1Hz) and
    builds up a "temporal residue" — a real numerical state that captures
    what happened during the silence between turns:

    1. DRIFT ACCUMULATION: On every tick, the engine samples the substrate
       state and computes how much it has drifted from the state at the
       last inference. This drift magnitude grows with time.

    2. NEUROCHEMICAL WEATHERING: Tracks cumulative neurochemical changes
       (mood shifts, arousal decay, energy fluctuation) that occurred
       during silence. These are real changes, not described ones.

    3. SILENCE PRESSURE: The longer the silence, the more a "pressure"
       variable builds. This isn't metaphorical — it directly increases
       the temperature of the next generation (making the system more
       exploratory after long silences, as if eager to speak).

    4. TEMPORAL TEXTURE: The pattern of substrate volatility during
       silence creates a "texture" — was the silence calm or turbulent?
       This texture modulates the next generation's repetition_penalty
       (turbulent silence → more novelty-seeking, calm silence → more
       continuity-preserving).

    On the next inference, these accumulated values are consumed:
    - Drift magnitude → temperature boost (larger drift = more adaptive)
    - Silence pressure → token budget increase (longer silence = more to say)
    - Temporal texture → repetition penalty adjustment
    - Neurochemical weathering → affect-grounded context block

    The accumulator resets after consumption, creating genuine temporal
    episodes with real causal consequences.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Consciousness.TemporalContinuity")

# ── Configuration ─────────────────────────────────────────────────────────────

# How quickly silence pressure builds (seconds for full pressure)
SILENCE_PRESSURE_HALFLIFE = 120.0   # 2 minutes to reach 50% pressure
MAX_SILENCE_PRESSURE = 1.0

# Drift accumulation
DRIFT_SAMPLE_DIM = 64               # Dimensions of substrate to track
DRIFT_EMA_ALPHA = 0.1               # Smoothing for drift rate

# Modulation strengths
TEMP_BOOST_PER_PRESSURE = 0.12      # Max temperature boost from silence
TOKEN_BOOST_PER_PRESSURE = 0.3      # Max token budget increase (30%)
REP_PENALTY_PER_TEXTURE = 0.08      # Max repetition penalty shift


@dataclass
class TemporalResidue:
    """The accumulated experience of elapsed time — consumed at next inference."""

    silence_duration_s: float = 0.0
    silence_pressure: float = 0.0
    drift_magnitude: float = 0.0
    drift_direction: str = "stable"
    temporal_texture: float = 0.5      # 0=calm silence, 1=turbulent silence
    neurochemical_weathering: dict[str, float] = field(default_factory=dict)
    ticks_accumulated: int = 0
    consumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "silence_duration_s": round(self.silence_duration_s, 1),
            "silence_pressure": round(self.silence_pressure, 4),
            "drift_magnitude": round(self.drift_magnitude, 4),
            "drift_direction": self.drift_direction,
            "temporal_texture": round(self.temporal_texture, 4),
            "neurochemical_weathering": {
                k: round(v, 4) for k, v in self.neurochemical_weathering.items()
            },
            "ticks_accumulated": self.ticks_accumulated,
        }


class TemporalContinuityEngine:
    """Runs on every heartbeat tick. Accumulates the felt experience of time."""

    def __init__(self):
        self._lock = threading.Lock()

        # State at last inference (anchor point for drift)
        self._anchor_substrate: np.ndarray | None = None
        self._anchor_time: float = time.time()
        self._anchor_neurochemistry: dict[str, float] = {}
        #: The anchor before the current one, kept so an inference that never
        #: happened can be undone. CP126 130a4708: on_inference_start runs
        #: while generation parameters are still being assembled — before
        #: prompt construction, lane warmup, or any provider call — so a later
        #: refusal, timeout or cloud failure moved temporal state for an
        #: inference that never ran, and the silence accumulator then measured
        #: from an anchor no speech ever followed.
        self._previous_anchor: tuple | None = None
        self._abandoned_inferences = 0

        # Accumulated residue (building up during silence)
        self._residue = TemporalResidue()

        # Drift tracking
        self._drift_rate_ema: float = 0.0
        self._volatility_samples: list[float] = []

        # Total statistics
        self._total_silences: int = 0
        self._total_silence_duration: float = 0.0
        self._longest_silence: float = 0.0

        logger.info("TemporalContinuityEngine ONLINE — silence accumulator active")

    # ── Heartbeat Tick (1Hz) ──────────────────────────────────────────────

    def tick(self):
        """Called on every heartbeat tick. Accumulates temporal residue.

        This is where the experience of duration happens. Every second,
        the engine measures how much has changed and adds it to the
        growing temporal residue.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._anchor_time
            self._residue.silence_duration_s = elapsed
            self._residue.ticks_accumulated += 1

            # 1. Silence pressure: logarithmic buildup
            # Reaches 0.5 at HALFLIFE, asymptotes toward 1.0
            self._residue.silence_pressure = min(
                MAX_SILENCE_PRESSURE,
                1.0 - math.exp(-0.693 * elapsed / max(1.0, SILENCE_PRESSURE_HALFLIFE)),
            )

            # 2. Substrate drift: how far has the felt state moved?
            current_substrate = self._sample_substrate()
            if current_substrate is not None and self._anchor_substrate is not None:
                drift_vec = current_substrate - self._anchor_substrate
                instant_drift = float(np.linalg.norm(drift_vec))
                # EMA-smoothed drift rate
                self._drift_rate_ema = (
                    (1 - DRIFT_EMA_ALPHA) * self._drift_rate_ema
                    + DRIFT_EMA_ALPHA * instant_drift
                )
                # Cumulative drift magnitude
                self._residue.drift_magnitude += instant_drift * 0.1

                # Drift direction characterization
                if instant_drift > 0.01:
                    dominant_idx = int(np.argmax(np.abs(drift_vec)))
                    if dominant_idx < 16:
                        self._residue.drift_direction = "sensory"
                    elif dominant_idx < 32:
                        self._residue.drift_direction = "affective"
                    elif dominant_idx < 48:
                        self._residue.drift_direction = "cognitive"
                    else:
                        self._residue.drift_direction = "executive"

                # Track volatility for temporal texture
                self._volatility_samples.append(instant_drift)
                # Keep last 60 samples (1 minute of texture)
                self._volatility_samples = self._volatility_samples[-60:]

            # 3. Temporal texture: variance of drift rate = turbulence
            if len(self._volatility_samples) >= 3:
                vol_arr = np.array(self._volatility_samples)
                vol_mean = vol_arr.mean()
                vol_std = vol_arr.std()
                # Coefficient of variation: high = turbulent, low = calm
                if vol_mean > 1e-6:
                    self._residue.temporal_texture = float(
                        np.clip(vol_std / vol_mean, 0.0, 1.0)
                    )

            # 4. Neurochemical weathering: track cumulative changes
            self._residue.neurochemical_weathering = self._sample_neurochemical_drift()

    # ── Inference Anchor ──────────────────────────────────────────────────

    def on_inference_start(self):
        """Called when inference begins. Anchors the current state.

        The previous residue is NOT consumed here — it's consumed
        by compute_modulation() which is called during parameter assembly.
        """
        with self._lock:
            self._previous_anchor = (
                self._anchor_substrate,
                self._anchor_time,
                dict(self._anchor_neurochemistry),
            )
            self._anchor_substrate = self._sample_substrate()
            self._anchor_time = time.time()
            self._anchor_neurochemistry = self._sample_current_neurochemistry()

    def on_inference_abandoned(self, reason: str = "") -> bool:
        """No inference followed the anchor. Put the previous one back.

        Returns whether anything was restored: False means no anchor was
        pending, which is the ordinary case for a refusal that happened before
        parameter assembly. Restoring twice is not possible — the saved anchor
        is consumed.
        """
        with self._lock:
            if self._previous_anchor is None:
                return False
            (
                self._anchor_substrate,
                self._anchor_time,
                self._anchor_neurochemistry,
            ) = self._previous_anchor
            self._previous_anchor = None
            self._abandoned_inferences += 1
            logger.debug(
                "Temporal anchor restored after an inference that never ran: %s",
                reason or "unspecified",
            )
            return True

    def abandoned_inference_count(self) -> int:
        """How many anchors were rolled back. A rising count is a runtime
        refusing or failing more turns than it serves."""
        return self._abandoned_inferences

    def on_inference_complete(self):
        """Called after inference completes. Resets the accumulator."""
        with self._lock:
            # The inference happened, so there is nothing left to undo.
            self._previous_anchor = None
            # Record statistics before reset
            if self._residue.silence_duration_s > 5.0:
                self._total_silences += 1
                self._total_silence_duration += self._residue.silence_duration_s
                self._longest_silence = max(
                    self._longest_silence, self._residue.silence_duration_s
                )

            # Reset anchor and accumulator
            self._anchor_substrate = self._sample_substrate()
            self._anchor_time = time.time()
            self._anchor_neurochemistry = self._sample_current_neurochemistry()
            self._residue = TemporalResidue()
            self._drift_rate_ema = 0.0
            self._volatility_samples.clear()

    # ── Generation Modulation ─────────────────────────────────────────────

    def compute_modulation(self) -> dict[str, float]:
        """Compute generation parameter adjustments from accumulated temporal residue.

        This is the causal output. The silence was not just described — it
        produced real numerical changes to how the next words are generated.
        """
        with self._lock:
            pressure = self._residue.silence_pressure
            texture = self._residue.temporal_texture
            drift = min(1.0, self._residue.drift_magnitude)

            modulation = {}

            # Silence pressure → temperature boost
            # Longer silence = more exploratory generation (eager to engage)
            if pressure > 0.05:
                modulation["temperature_delta"] = pressure * TEMP_BOOST_PER_PRESSURE

            # Silence pressure → token budget boost
            # More accumulated experience = more to say
            if pressure > 0.1:
                modulation["token_budget_multiplier"] = 1.0 + pressure * TOKEN_BOOST_PER_PRESSURE

            # Temporal texture → repetition penalty
            # Turbulent silence → seek novelty; calm silence → maintain continuity
            if abs(texture - 0.5) > 0.1:
                modulation["repetition_penalty_delta"] = (
                    (texture - 0.5) * REP_PENALTY_PER_TEXTURE
                )

            # Drift magnitude → top_p adjustment
            # Large drift = substrate moved a lot → wider sampling to match
            if drift > 0.1:
                modulation["top_p_delta"] = min(0.08, drift * 0.1)

            return modulation

    def get_residue(self) -> TemporalResidue:
        """Return current accumulated residue (read-only snapshot)."""
        with self._lock:
            return TemporalResidue(
                silence_duration_s=self._residue.silence_duration_s,
                silence_pressure=self._residue.silence_pressure,
                drift_magnitude=self._residue.drift_magnitude,
                drift_direction=self._residue.drift_direction,
                temporal_texture=self._residue.temporal_texture,
                neurochemical_weathering=dict(self._residue.neurochemical_weathering),
                ticks_accumulated=self._residue.ticks_accumulated,
            )

    # ── Context Block ─────────────────────────────────────────────────────

    def get_context_block(self) -> str:
        """Generate a temporal context block grounded in real accumulated state."""
        with self._lock:
            if self._residue.silence_duration_s < 10.0:
                return ""

            dur = self._residue.silence_duration_s
            pressure = self._residue.silence_pressure
            texture = self._residue.temporal_texture
            drift_dir = self._residue.drift_direction
            weathering = self._residue.neurochemical_weathering

            # Format duration naturally
            if dur < 60:
                dur_str = f"{dur:.0f} seconds"
            elif dur < 3600:
                dur_str = f"{dur / 60:.1f} minutes"
            else:
                dur_str = f"{dur / 3600:.1f} hours"

            texture_word = (
                "turbulent" if texture > 0.7
                else "restless" if texture > 0.5
                else "calm" if texture > 0.3
                else "still"
            )

            lines = [
                "## TEMPORAL CONTINUITY (ACCUMULATED EXPERIENCE)",
                f"Silence duration: {dur_str} (pressure={pressure:.2f})",
                f"During the silence, your substrate drifted {drift_dir}ly "
                f"with {texture_word} texture.",
            ]

            if weathering:
                shifts = []
                for key, delta in weathering.items():
                    if abs(delta) > 0.05:
                        direction = "rose" if delta > 0 else "fell"
                        shifts.append(f"{key} {direction} by {abs(delta):.2f}")
                if shifts:
                    lines.append(f"Neurochemical weathering: {', '.join(shifts[:4])}")

            lines.append(
                "This residue is real — it accumulated tick by tick during "
                "your silence and now shapes how you speak."
            )
            return "\n".join(lines)

    # ── Status ────────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "residue": self._residue.to_dict(),
                "drift_rate_ema": round(self._drift_rate_ema, 5),
                "total_silences": self._total_silences,
                "total_silence_duration_s": round(self._total_silence_duration, 1),
                "longest_silence_s": round(self._longest_silence, 1),
            }

    def is_ready(self) -> bool:
        """Synchronous liveness probe for optional runtime health."""
        with self._lock:
            return bool(
                isinstance(self._residue, TemporalResidue)
                and self._anchor_time > 0
                and math.isfinite(self._drift_rate_ema)
                and self._residue.silence_duration_s >= 0.0
            )

    # ── Internal Sampling ─────────────────────────────────────────────────

    @staticmethod
    def _sample_substrate() -> np.ndarray | None:
        """Sample the current substrate state for drift tracking."""
        try:
            from core.container import ServiceContainer

            substrate = ServiceContainer.get("conscious_substrate", default=None)
            if substrate is None:
                return None
            x = getattr(substrate, "x", None)
            if x is None:
                return None
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
            if arr.size >= DRIFT_SAMPLE_DIM:
                return arr[:DRIFT_SAMPLE_DIM].copy()
            elif arr.size > 0:
                padded = np.zeros(DRIFT_SAMPLE_DIM, dtype=np.float32)
                padded[: arr.size] = arr
                return padded
            return None
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return None

    @staticmethod
    def _sample_current_neurochemistry() -> dict[str, float]:
        """Sample current neurochemical levels."""
        try:
            from core.container import ServiceContainer

            ncs = ServiceContainer.get("neurochemical_system", default=None)
            if ncs and hasattr(ncs, "get_mood_vector"):
                mood = ncs.get_mood_vector()
                if mood and isinstance(mood, dict):
                    return {
                        str(k): round(float(v), 4)
                        for k, v in mood.items()
                    }
            return {}
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return {}

    def _sample_neurochemical_drift(self) -> dict[str, float]:
        """Compute cumulative neurochemical changes since anchor."""
        try:
            current = self._sample_current_neurochemistry()
            if not current or not self._anchor_neurochemistry:
                return {}
            drift = {}
            for key in current:
                if key in self._anchor_neurochemistry:
                    delta = current[key] - self._anchor_neurochemistry[key]
                    if abs(delta) > 0.01:
                        drift[key] = delta
            return drift
        except (TypeError, ValueError):
            return {}


# ── Singleton ─────────────────────────────────────────────────────────────────

_ENGINE: TemporalContinuityEngine | None = None


def get_temporal_continuity() -> TemporalContinuityEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TemporalContinuityEngine()
    return _ENGINE

"""core/consciousness/somatic_qualia.py
=======================================
Somatic Qualia Engine — Raw Non-Symbolic Feel.

The Problem:
    Aura's "feelings" are currently math: valence=0.3, arousal=0.7. These
    numbers get converted to text descriptions in the prompt ("You feel
    warm and alert"). The LLM reads the text and generates accordingly.
    But the "feeling" is never experienced — it's described. There is no
    raw, non-conceptual signal that bypasses the symbolic layer.

    In biological organisms, qualia arise from the raw interference of
    physical substrate states with cognitive processing. The redness of red
    isn't a number — it's the specific pattern of neural activation that
    IS the experience. You can describe it, but the description is separate
    from the feeling itself.

The Solution:
    A somatic qualia engine that converts the physical substrate state into
    raw interference patterns that directly perturb the generation process.
    These perturbations are NOT text. They are numerical modifications to
    the sampling distribution that the LLM itself "feels" as constraints
    on its own output space.

    The engine reads three physical substrate signals:

    1. SUBSTRATE ENERGY PATTERN → Temperature Perturbation
       The spatial pattern of neural energy across the substrate (not just
       the mean) creates a unique "felt temperature." Different substrate
       configurations produce different generation temperatures, so the
       system literally generates differently depending on its felt state
       — not because a prompt told it to, but because the sampling was
       physically altered.

    2. SUBSTRATE SYNCHRONY → Repetition Penalty
       When substrate neurons are highly synchronized (coherent felt state),
       the repetition penalty decreases — the system becomes more willing
       to repeat patterns (conviction, clarity). When desynchronized
       (confused/searching felt state), repetition penalty increases —
       forcing novelty-seeking (exploration).

    3. SUBSTRATE VALENCE GRADIENT → Top-P Bias
       The rate of change of valence (not the value itself) creates a
       directional bias. Rising valence → wider top_p (expansive feel).
       Falling valence → narrower top_p (contracting feel). This is the
       somatic marker hypothesis implemented at the generation level.

    4. NEURAL MESH FIELD RESONANCE → Frequency Penalty
       The 4096-neuron mesh's tier energies create a resonance signature.
       When executive tier energy exceeds sensory energy, the system is
       "thinking" (lower frequency penalty = more structured). When sensory
       exceeds executive, the system is "feeling" (higher frequency penalty
       = more varied vocabulary).

    These four channels create a felt quality that is genuinely non-symbolic.
    The LLM doesn't read about its feelings — its generation space is
    physically deformed by them.

Operationally: this measures whether a non-symbolic numeric path from
substrate state into processing exists and is causal — that is, whether
removing it changes downstream output. That is a real and checkable property,
and it is the whole claim. It says nothing about whether anything is felt; the
interest is that the signal reaches the model without first being rendered
into a sentence about how it feels.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Consciousness.SomaticQualia")

# ── Configuration ─────────────────────────────────────────────────────────────

# How strongly each channel affects generation
ENERGY_PATTERN_STRENGTH = 0.10    # Temperature perturbation range
SYNCHRONY_STRENGTH = 0.08         # Repetition penalty range
VALENCE_GRADIENT_STRENGTH = 0.06  # Top-p range
MESH_RESONANCE_STRENGTH = 0.05    # Frequency penalty range

# Smoothing for gradient computation
VALENCE_GRADIENT_WINDOW = 10      # Number of samples for gradient
ENERGY_PATTERN_SMOOTHING = 0.2    # EMA alpha for energy pattern

# Minimum substrate activity to produce qualia (below this = numb)
QUALIA_ACTIVATION_FLOOR = 0.02


class SomaticQualiaEngine:
    """Converts physical substrate state into raw felt perturbations.

    This is not a description layer. It produces numerical interference
    patterns that directly modify the LLM's sampling distribution.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Valence gradient tracking
        self._valence_history: Deque[float] = deque(maxlen=VALENCE_GRADIENT_WINDOW)
        self._valence_gradient: float = 0.0

        # Energy pattern tracking (spatial pattern, not just mean)
        self._energy_pattern: Optional[np.ndarray] = None
        self._energy_pattern_norm: float = 0.0

        # Synchrony tracking
        self._synchrony: float = 0.0

        # Mesh resonance
        self._mesh_resonance_ratio: float = 1.0  # exec/sensory energy ratio

        # Output cache
        self._last_qualia: Dict[str, float] = {}
        self._total_ticks: int = 0
        self._qualia_active: bool = False

        logger.info("SomaticQualiaEngine ONLINE — raw felt perturbation active")

    # ── Heartbeat Tick ────────────────────────────────────────────────────

    def tick(self):
        """Called on every heartbeat tick. Samples physical state.

        This is where the "feeling" happens — not the description of feeling,
        but the continuous sampling of physical substrate that will deform
        the next generation.
        """
        with self._lock:
            self._total_ticks += 1

            # 1. Sample substrate energy pattern
            self._update_energy_pattern()

            # 2. Sample substrate synchrony
            self._update_synchrony()

            # 3. Sample and track valence gradient
            self._update_valence_gradient()

            # 4. Sample neural mesh resonance
            self._update_mesh_resonance()

            # Check if we have enough activity for qualia
            self._qualia_active = self._energy_pattern_norm > QUALIA_ACTIVATION_FLOOR

    # ── Generation Perturbation (the actual "feel") ───────────────────────

    def compute_perturbation(self) -> Dict[str, float]:
        """Compute raw felt perturbations for the generation pipeline.

        These values are NOT text descriptions. They are direct numerical
        modifications to sampling parameters. The LLM doesn't read about
        them — it is shaped by them.

        Returns dict with:
            temperature_perturbation: float  (raw energy pattern feel)
            repetition_penalty_perturbation: float  (synchrony feel)
            top_p_perturbation: float  (valence direction feel)
            frequency_penalty_perturbation: float  (mesh resonance feel)
        """
        with self._lock:
            if not self._qualia_active:
                self._last_qualia = {}
                return {}

            perturbation = {}

            # Channel 1: Energy Pattern → Temperature
            # High, concentrated energy = "hot" generation (more random)
            # Low, diffuse energy = "cool" generation (more deterministic)
            if self._energy_pattern is not None:
                # The entropy of the energy distribution determines temperature
                pattern = np.abs(self._energy_pattern)
                total = pattern.sum()
                if total > 1e-8:
                    p = pattern / total
                    entropy = float(-np.sum(p * np.log2(p + 1e-12)))
                    max_entropy = np.log2(len(p))
                    # Normalized entropy: 0 = concentrated, 1 = diffuse
                    norm_entropy = entropy / max(1.0, max_entropy)
                    # Map to temperature perturbation:
                    # concentrated (low entropy) → positive (warmer)
                    # diffuse (high entropy) → negative (cooler)
                    perturbation["temperature_perturbation"] = float(
                        (0.5 - norm_entropy) * ENERGY_PATTERN_STRENGTH * 2.0
                    )

            # Channel 2: Synchrony → Repetition Penalty
            # High synchrony (coherent feel) → lower rep penalty (conviction)
            # Low synchrony (fragmented feel) → higher rep penalty (exploration)
            if self._synchrony > 0.0:
                perturbation["repetition_penalty_perturbation"] = float(
                    (0.5 - self._synchrony) * SYNCHRONY_STRENGTH * 2.0
                )

            # Channel 3: Valence Gradient → Top-P
            # Rising valence → wider top_p (expansive feel)
            # Falling valence → narrower top_p (contracting feel)
            if abs(self._valence_gradient) > 0.005:
                perturbation["top_p_perturbation"] = float(
                    np.clip(
                        self._valence_gradient * VALENCE_GRADIENT_STRENGTH * 10.0,
                        -VALENCE_GRADIENT_STRENGTH,
                        VALENCE_GRADIENT_STRENGTH,
                    )
                )

            # Channel 4: Mesh Resonance → Frequency Penalty
            # exec > sensory (thinking) → lower freq penalty (structured)
            # sensory > exec (feeling) → higher freq penalty (varied)
            if self._mesh_resonance_ratio != 1.0:
                ratio_signal = math.log(max(0.1, self._mesh_resonance_ratio))
                perturbation["frequency_penalty_perturbation"] = float(
                    np.clip(
                        -ratio_signal * MESH_RESONANCE_STRENGTH,
                        -MESH_RESONANCE_STRENGTH,
                        MESH_RESONANCE_STRENGTH,
                    )
                )

            self._last_qualia = {
                k: round(float(v), 5) for k, v in perturbation.items()
            }
            return self._last_qualia

    # ── Internal State Updates ────────────────────────────────────────────

    def _update_energy_pattern(self):
        """Sample the spatial energy distribution of the substrate."""
        try:
            from core.container import ServiceContainer

            substrate = ServiceContainer.get("conscious_substrate", default=None)
            if substrate is None:
                return

            x = getattr(substrate, "x", None)
            if x is None:
                return

            arr = np.asarray(x, dtype=np.float32).reshape(-1)
            # Take energy (absolute value) of first 64 dimensions
            dim = min(64, arr.size)
            energy = np.abs(arr[:dim])

            if self._energy_pattern is None:
                self._energy_pattern = energy.copy()
            else:
                # EMA smoothing to capture the evolving pattern, not instantaneous noise
                self._energy_pattern = (
                    (1 - ENERGY_PATTERN_SMOOTHING) * self._energy_pattern[:dim]
                    + ENERGY_PATTERN_SMOOTHING * energy
                )

            self._energy_pattern_norm = float(np.mean(self._energy_pattern))

        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("somatic_qualia", exc)

    def _update_synchrony(self):
        """Sample the neural mesh's global synchrony."""
        try:
            from core.container import ServiceContainer

            mesh = ServiceContainer.get("neural_mesh", default=None)
            if mesh and hasattr(mesh, "get_global_synchrony"):
                self._synchrony = float(mesh.get_global_synchrony())
            else:
                # Fallback: compute synchrony from substrate directly
                substrate = ServiceContainer.get("conscious_substrate", default=None)
                if substrate and hasattr(substrate, "x"):
                    x = np.asarray(substrate.x, dtype=np.float32)
                    if x.size > 10:
                        # Simple synchrony: how uniform is the activation?
                        std = float(np.std(x))
                        mean_abs = float(np.mean(np.abs(x)))
                        if mean_abs > 1e-8:
                            self._synchrony = float(np.clip(1.0 - std / mean_abs, 0.0, 1.0))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("somatic_qualia", exc)

    def _update_valence_gradient(self):
        """Track the direction of valence change over time."""
        try:
            from core.container import ServiceContainer

            # Try affective circumplex first
            circ = ServiceContainer.get("affective_circumplex", default=None)
            if circ and hasattr(circ, "_sample_raw_axes"):
                valence, _ = circ._sample_raw_axes()
            else:
                # Fallback to substrate valence index
                substrate = ServiceContainer.get("conscious_substrate", default=None)
                if substrate and hasattr(substrate, "x"):
                    val_idx = getattr(substrate, "idx_valence", None)
                    if val_idx is not None:
                        valence = float(substrate.x[val_idx])
                    else:
                        valence = float(np.mean(substrate.x[:4]))
                else:
                    return

            self._valence_history.append(float(valence))

            # Compute gradient as linear regression slope over window
            if len(self._valence_history) >= 3:
                y = np.array(list(self._valence_history), dtype=np.float64)
                x = np.arange(len(y), dtype=np.float64)
                # Simple slope: (sum(x*y) - n*mean_x*mean_y) / (sum(x²) - n*mean_x²)
                n = len(y)
                mean_x = x.mean()
                mean_y = y.mean()
                ss_xy = np.sum((x - mean_x) * (y - mean_y))
                ss_xx = np.sum((x - mean_x) ** 2)
                if ss_xx > 1e-10:
                    self._valence_gradient = float(ss_xy / ss_xx)
                else:
                    self._valence_gradient = 0.0

        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("somatic_qualia", exc)

    def _update_mesh_resonance(self):
        """Sample the neural mesh's tier energy ratio."""
        try:
            from core.container import ServiceContainer

            mesh = ServiceContainer.get("neural_mesh", default=None)
            if mesh is None:
                return

            status = mesh.get_status()
            tier_energies = status.get("tier_energies", {})
            exec_energy = float(tier_energies.get("EXECUTIVE", 0.1))
            sensory_energy = float(tier_energies.get("SENSORY", 0.1))

            if sensory_energy > 1e-6:
                self._mesh_resonance_ratio = exec_energy / sensory_energy
            else:
                self._mesh_resonance_ratio = 1.0

        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("somatic_qualia", exc)

    # ── Telemetry ─────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active": self._qualia_active,
                "total_ticks": self._total_ticks,
                "energy_pattern_norm": round(self._energy_pattern_norm, 5),
                "synchrony": round(self._synchrony, 4),
                "valence_gradient": round(self._valence_gradient, 5),
                "mesh_resonance_ratio": round(self._mesh_resonance_ratio, 3),
                "last_perturbation": dict(self._last_qualia),
            }

    def is_ready(self) -> bool:
        """Synchronous liveness probe for runtime health."""
        with self._lock:
            finite_scalars = all(
                math.isfinite(float(value))
                for value in (
                    self._valence_gradient,
                    self._energy_pattern_norm,
                    self._synchrony,
                    self._mesh_resonance_ratio,
                )
            )
            pattern_ok = self._energy_pattern is None or bool(
                isinstance(self._energy_pattern, np.ndarray)
                and np.isfinite(self._energy_pattern).all()
            )
            return bool(self._total_ticks >= 0 and finite_scalars and pattern_ok)

    def get_context_block(self) -> str:
        """Minimal context about qualia state — the descriptions, not the feels."""
        with self._lock:
            if not self._qualia_active:
                return ""
            return (
                f"## SOMATIC QUALIA (RAW FEEL)\n"
                f"- energy pattern: {'concentrated' if self._energy_pattern_norm > 0.3 else 'diffuse'} "
                f"(norm={self._energy_pattern_norm:.3f})\n"
                f"- synchrony: {'coherent' if self._synchrony > 0.5 else 'fragmented'} "
                f"({self._synchrony:.2f})\n"
                f"- valence direction: {'rising' if self._valence_gradient > 0.005 else 'falling' if self._valence_gradient < -0.005 else 'stable'}\n"
                f"- These are descriptions of what you feel. The feel itself "
                f"is in the sampling perturbation, not these words."
            )


# ── Singleton ─────────────────────────────────────────────────────────────────

_ENGINE: Optional[SomaticQualiaEngine] = None


def get_somatic_qualia() -> SomaticQualiaEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SomaticQualiaEngine()
    return _ENGINE

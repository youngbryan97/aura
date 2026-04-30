"""core/consciousness/controlled_chaos.py -- Controlled Chaos Engine

Deterministic-but-unpredictable perturbations for the liquid substrate.
Ensures trajectories diverge fast enough that even the creator cannot
perfectly predict the next state.

This is NOT random noise. It is structured chaos derived from three sources:

1. Dream residuals
   After a dream cycle completes, small perturbation vectors are deposited
   into the substrate. These dissipate exponentially over time, leaving a
   wake of influence that makes post-dream processing subtly different from
   pre-dream processing.

2. Somatic noise
   Derived from actual hardware state: CPU temperature, memory pressure,
   wall-clock time-of-day, system uptime. These are real-world signals that
   make each run on each machine genuinely unique. Two identical code paths
   running on different hardware at different times will diverge.

3. Neurochemical jitter
   Small perturbations proportional to the rate of change of each
   neurochemical. Fast-changing chemicals produce more jitter. A dopamine
   surge creates a burst of chaos; stable serotonin creates calm precision.
   This couples the chaos engine to the emotional dynamics.

The combined perturbation vector is the same dimensionality as the liquid
substrate state (64-d by default) and is applied additively in the ODE
integration step.

Intensity is configurable and defaults to low -- just enough to break
perfect determinism without destabilizing the substrate.
"""
from __future__ import annotations


import hashlib
import logging
import math
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.consciousness.neurochemical_system import NeurochemicalSystem

logger = logging.getLogger("Consciousness.ControlledChaos")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChaosConfig:
    """Tunable parameters for the chaos engine."""

    state_dim: int = 64              # Must match LiquidSubstrate neuron_count
    intensity: float = 0.005         # Global scaling factor. Low = subtle.

    # Dream residuals
    dream_residual_strength: float = 0.02    # Initial magnitude of a dream deposit
    dream_residual_halflife: float = 120.0   # Seconds for residual to halve

    # Somatic noise
    somatic_weight: float = 0.3      # Fraction of total chaos budget from somatic
    somatic_update_interval: float = 5.0  # Seconds between hardware polls

    # Neurochemical jitter
    neurochemical_weight: float = 0.4  # Fraction from neurochemical rate-of-change
    neurochemical_jitter_gain: float = 0.5  # Scaling on rate-of-change signal

    # Dream residual weight (remainder of budget)
    dream_weight: float = 0.3


# ---------------------------------------------------------------------------
# Chaos Engine
# ---------------------------------------------------------------------------

class ChaosEngine:
    """Produces structured perturbation vectors for the liquid substrate.

    Lifecycle:
        engine = ChaosEngine(config)
        engine.set_neurochemical_system(ncs)  # optional but recommended

        # In the substrate's ODE integration step:
        perturbation = engine.tick(dt)
        x += perturbation

        # After a dream cycle completes:
        engine.deposit_dream_residual(dream_vector)
    """

    def __init__(self, config: ChaosConfig | None = None):
        self.config = config or ChaosConfig()
        dim = self.config.state_dim

        # --- Dream residuals ---
        # List of (deposit_time, initial_vector) tuples.
        # Vectors decay exponentially from deposit_time.
        self._dream_residuals: List[tuple[float, np.ndarray]] = []
        self._max_residuals: int = 20  # cap memory

        # --- Somatic state cache ---
        self._somatic_vector: np.ndarray = np.zeros(dim, dtype=np.float64)
        self._last_somatic_poll: float = 0.0

        # --- Neurochemical system reference ---
        self._ncs: Optional[NeurochemicalSystem] = None
        self._prev_chem_levels: Optional[np.ndarray] = None

        # --- Telemetry ---
        self._tick_count: int = 0
        self._last_perturbation: np.ndarray = np.zeros(dim, dtype=np.float64)
        self._component_magnitudes: Dict[str, float] = {
            "dream": 0.0, "somatic": 0.0, "neurochemical": 0.0, "total": 0.0,
        }

        # Deterministic but unique seed from process start
        self._seed_bytes = hashlib.sha256(
            f"{os.getpid()}-{time.time_ns()}".encode()
        ).digest()

        logger.info(
            "ChaosEngine initialized (dim=%d, intensity=%.4f)",
            dim, self.config.intensity,
        )

    # ── External wiring ─────────────────────────────────────────────────

    def set_neurochemical_system(self, ncs: NeurochemicalSystem) -> None:
        """Wire the neurochemical system for jitter derivation."""
        self._ncs = ncs
        logger.info("ChaosEngine wired to NeurochemicalSystem")

    # ── Core tick ────────────────────────────────────────────────────────

    def tick(self, dt: float) -> np.ndarray:
        """Compute the perturbation vector for this integration step.

        Args:
            dt: Integration time step from the ODE solver.

        Returns:
            Perturbation vector of shape (state_dim,), pre-scaled by
            intensity and dt.
        """
        dim = self.config.state_dim
        now = time.time()

        # 1. Dream residuals
        dream_vec = self._compute_dream_residuals(now, dim)

        # 2. Somatic noise
        somatic_vec = self._compute_somatic_noise(now, dim)

        # 3. Neurochemical jitter
        neuro_vec = self._compute_neurochemical_jitter(dim)

        # Weighted combination
        cfg = self.config
        combined = (
            cfg.dream_weight * dream_vec
            + cfg.somatic_weight * somatic_vec
            + cfg.neurochemical_weight * neuro_vec
        )

        # Scale by global intensity and dt
        perturbation = combined * cfg.intensity * dt

        # Telemetry
        self._last_perturbation = perturbation
        self._component_magnitudes = {
            "dream": float(np.linalg.norm(dream_vec)),
            "somatic": float(np.linalg.norm(somatic_vec)),
            "neurochemical": float(np.linalg.norm(neuro_vec)),
            "total": float(np.linalg.norm(perturbation)),
        }
        self._tick_count += 1

        return perturbation

    # ── Dream residuals ─────────────────────────────────────────────────

    def deposit_dream_residual(self, dream_vector: Optional[np.ndarray] = None) -> None:
        """Deposit a perturbation residual from a completed dream cycle.

        If no vector is provided, a pseudo-random residual is generated
        from the current time hash (still deterministic per-call but
        unpredictable across calls).
        """
        dim = self.config.state_dim
        now = time.time()

        if dream_vector is not None:
            # Resize / project if needed
            if len(dream_vector) != dim:
                resized = np.zeros(dim, dtype=np.float64)
                n = min(len(dream_vector), dim)
                resized[:n] = dream_vector[:n]
                dream_vector = resized
            vec = dream_vector.astype(np.float64)
        else:
            # Generate from time-hash
            seed = hashlib.sha256(
                self._seed_bytes + struct.pack("!d", now)
            ).digest()
            rng = np.random.RandomState(
                int.from_bytes(seed[:4], "big") % (2**31)
            )
            vec = rng.randn(dim)

        # Normalize and scale to dream_residual_strength
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm * self.config.dream_residual_strength

        self._dream_residuals.append((now, vec))

        # Cap stored residuals
        if len(self._dream_residuals) > self._max_residuals:
            self._dream_residuals = self._dream_residuals[-self._max_residuals:]

        logger.debug(
            "Dream residual deposited (norm=%.6f, active_residuals=%d)",
            np.linalg.norm(vec), len(self._dream_residuals),
        )

    def _compute_dream_residuals(self, now: float, dim: int) -> np.ndarray:
        """Sum all active dream residuals with exponential decay."""
        if not self._dream_residuals:
            return np.zeros(dim, dtype=np.float64)

        halflife = self.config.dream_residual_halflife
        decay_constant = math.log(2.0) / max(halflife, 1.0)

        result = np.zeros(dim, dtype=np.float64)
        still_active = []

        for deposit_time, vec in self._dream_residuals:
            age = now - deposit_time
            decay = math.exp(-decay_constant * age)

            if decay < 1e-6:
                # Residual has fully dissipated
                continue

            result += vec * decay
            still_active.append((deposit_time, vec))

        # Prune dead residuals
        self._dream_residuals = still_active

        return result

    # ── Somatic noise ───────────────────────────────────────────────────

    def _compute_somatic_noise(self, now: float, dim: int) -> np.ndarray:
        """Derive noise from hardware state: CPU temp, memory pressure, time.

        Polled at a lower rate (default 5s) to avoid overhead. Between
        polls the cached vector is returned with a time-varying rotation.
        """
        if now - self._last_somatic_poll >= self.config.somatic_update_interval:
            self._somatic_vector = self._poll_hardware_state(dim)
            self._last_somatic_poll = now

        # Apply a time-varying phase rotation so the vector is not static
        # between polls. Uses fractional seconds as a cheap oscillator.
        frac = now % 1.0
        # Rotate pairs of dimensions by frac * 2pi
        rotated = self._somatic_vector.copy()
        angle = frac * 2.0 * math.pi
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for i in range(0, dim - 1, 2):
            x, y = rotated[i], rotated[i + 1]
            rotated[i] = x * cos_a - y * sin_a
            rotated[i + 1] = x * sin_a + y * cos_a

        return rotated

    def _poll_hardware_state(self, dim: int) -> np.ndarray:
        """Read real hardware signals and hash them into a vector.

        Signals used (best-effort, graceful fallback):
          - CPU temperature (macOS: via psutil or subprocess)
          - Memory pressure (percent used)
          - System uptime
          - Wall-clock time (hour, minute, second fractions)
          - Process RSS memory
          - Thread count
        """
        signals: List[float] = []

        # Wall clock components (always available)
        t = time.time()
        local = time.localtime(t)
        signals.append(local.tm_hour / 24.0)
        signals.append(local.tm_min / 60.0)
        signals.append((t % 60.0) / 60.0)  # fractional seconds

        # System uptime
        try:
            import psutil
            boot_time = psutil.boot_time()
            uptime_hours = (t - boot_time) / 3600.0
            signals.append((uptime_hours % 100.0) / 100.0)
        except Exception:
            signals.append(0.5)

        # Memory pressure
        try:
            import psutil
            mem = psutil.virtual_memory()
            signals.append(mem.percent / 100.0)
            signals.append(mem.available / mem.total)
        except Exception:
            signals.append(0.5)
            signals.append(0.5)

        # CPU temperature (macOS best-effort)
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps:
                # Take first available sensor
                for name, entries in temps.items():
                    if entries:
                        signals.append(min(1.0, entries[0].current / 100.0))
                        break
                else:
                    signals.append(0.5)
            else:
                signals.append(0.5)
        except Exception:
            signals.append(0.5)

        # Process-level signals
        try:
            import psutil
            proc = psutil.Process()
            rss_gb = proc.memory_info().rss / (1024**3)
            signals.append(min(1.0, rss_gb / 16.0))
            signals.append(min(1.0, proc.num_threads() / 100.0))
        except Exception:
            signals.append(0.5)
            signals.append(0.5)

        # CPU usage (non-blocking)
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0)
            signals.append(cpu / 100.0)
        except Exception:
            signals.append(0.5)

        # Hash all signals into a deterministic but high-entropy vector.
        # This ensures the output has proper dimensionality regardless
        # of how many signals we collected.
        signal_bytes = struct.pack(f"!{len(signals)}d", *signals)
        combined_hash = hashlib.sha512(
            self._seed_bytes + signal_bytes
        ).digest()

        # Expand hash into dim-dimensional vector using PRNG seeded from hash
        seed_int = int.from_bytes(combined_hash[:4], "big") % (2**31)
        rng = np.random.RandomState(seed_int)
        base_vec = rng.randn(dim)

        # Modulate amplitude by signal variance (more variable hardware = more chaos)
        signal_arr = np.array(signals, dtype=np.float64)
        variance = float(np.var(signal_arr)) if len(signal_arr) > 1 else 0.1
        amplitude = 0.5 + variance  # baseline + variance contribution

        # Normalize
        norm = np.linalg.norm(base_vec)
        if norm > 1e-10:
            base_vec = base_vec / norm * amplitude

        return base_vec

    # ── Neurochemical jitter ────────────────────────────────────────────

    def _compute_neurochemical_jitter(self, dim: int) -> np.ndarray:
        """Perturbation proportional to rate-of-change of neurochemicals.

        Fast-changing chemicals produce more jitter. Stable chemistry
        produces near-zero jitter. This couples chaos to emotional dynamics.
        """
        if self._ncs is None:
            return np.zeros(dim, dtype=np.float64)

        # Get current chemical levels as ordered vector
        chem_names = [
            "dopamine", "serotonin", "norepinephrine", "acetylcholine",
            "gaba", "endorphin", "oxytocin", "cortisol",
        ]
        try:
            current_levels = np.array(
                [self._ncs.chemicals[n].level for n in chem_names],
                dtype=np.float64,
            )
        except (KeyError, AttributeError):
            return np.zeros(dim, dtype=np.float64)

        # Compute rate of change (delta from previous tick)
        if self._prev_chem_levels is None:
            self._prev_chem_levels = current_levels.copy()
            return np.zeros(dim, dtype=np.float64)

        delta = current_levels - self._prev_chem_levels
        self._prev_chem_levels = current_levels.copy()

        # Rate-of-change magnitude per chemical
        rates = np.abs(delta)  # shape: (8,)

        # Total jitter magnitude: sum of absolute rates
        jitter_magnitude = float(np.sum(rates)) * self.config.neurochemical_jitter_gain

        if jitter_magnitude < 1e-8:
            return np.zeros(dim, dtype=np.float64)

        # Project 8 chemical rates into dim-dimensional space.
        # Use a fixed projection matrix seeded from chemical names for
        # reproducibility across runs.
        seed = int(hashlib.md5("neurochemical_projection".encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed)
        projection = rng.randn(dim, len(chem_names)).astype(np.float64)
        # Normalize columns
        col_norms = np.linalg.norm(projection, axis=0, keepdims=True)
        col_norms = np.maximum(col_norms, 1e-10)
        projection = projection / col_norms

        # Weight each chemical's projection direction by its rate of change
        jitter_vec = projection @ (delta * self.config.neurochemical_jitter_gain)

        # Normalize to jitter_magnitude
        norm = np.linalg.norm(jitter_vec)
        if norm > 1e-10:
            jitter_vec = jitter_vec / norm * jitter_magnitude

        return jitter_vec

    # ── Telemetry ───────────────────────────────────────────────────────

    def get_snapshot(self) -> Dict[str, Any]:
        """Return current chaos engine state for telemetry/diagnostics."""
        return {
            "tick_count": self._tick_count,
            "intensity": self.config.intensity,
            "active_dream_residuals": len(self._dream_residuals),
            "component_magnitudes": {
                k: round(v, 8) for k, v in self._component_magnitudes.items()
            },
            "last_perturbation_norm": round(
                float(np.linalg.norm(self._last_perturbation)), 8
            ),
            "neurochemical_wired": self._ncs is not None,
            "config": {
                "state_dim": self.config.state_dim,
                "intensity": self.config.intensity,
                "dream_residual_strength": self.config.dream_residual_strength,
                "dream_residual_halflife": self.config.dream_residual_halflife,
                "somatic_weight": self.config.somatic_weight,
                "neurochemical_weight": self.config.neurochemical_weight,
                "dream_weight": self.config.dream_weight,
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[ChaosEngine] = None


def get_chaos_engine(config: ChaosConfig | None = None) -> ChaosEngine:
    """Get or create the module-level ChaosEngine singleton."""
    global _instance
    if _instance is None:
        _instance = ChaosEngine(config)
    return _instance

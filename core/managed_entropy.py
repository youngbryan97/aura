"""core/managed_entropy.py — Managed Entropy Injection

Prevents deterministic convergence in Aura's cognitive systems by injecting
controlled, hardware-sourced randomness at key decision points.

Injection points:
  1. Curiosity pressure (agency_core.py) — jitters the monotonic increment
  2. Goal generation (volition.py) — mutates topic selection
  3. Predictive self-model — perturbs observations to prevent weight collapse
  4. LLM temperature — already handled by PhysicalEntropyInjector (brain/entropy.py)

All entropy is budgeted per tick to prevent chaotic system state.
"""
import math
from core.utils.exceptions import capture_and_log
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("Aura.ManagedEntropy")


class ManagedEntropy:
    """Central entropy manager with per-tick budget and scoped injection."""

    def __init__(self, max_entropy_per_tick: float = 0.15):
        """
        Args:
            max_entropy_per_tick: Maximum total entropy injected per cognitive tick.
                Higher = more creative/chaotic. Lower = more stable/predictable.
                Recommended range: 0.05 (conservative) to 0.25 (exploratory).
        """
        self._max_per_tick = max_entropy_per_tick
        self._budget_remaining = max_entropy_per_tick
        self._last_reset = time.time()
        self._tick_duration = 5.0  # Budget resets every N seconds
        self._total_injected = 0.0
        self._injection_count = 0

        # Lazy import to avoid circular dependency at module load
        self._anchor = None

    def _get_anchor(self):
        """Lazy-load the hardware entropy anchor."""
        if self._anchor is None:
            try:
                from core.senses.entropy_anchor import entropy_anchor
                self._anchor = entropy_anchor
            except ImportError:
                logger.warning("PhysicalEntropyAnchor unavailable, using numpy fallback")
                self._anchor = None
        return self._anchor

    def _raw_float(self) -> float:
        """Get a raw entropy float [0.0, 1.0] from quantum, hardware, or fallback."""
        # Priority 1: Quantum entropy (ANU QRNG)
        try:
            from core.consciousness.quantum_entropy import get_quantum_entropy
            qe = get_quantum_entropy()
            return qe.get_quantum_float()
        except Exception as e:
            capture_and_log(e, {'module': __name__})
        
        # Priority 2: Hardware entropy anchor
        anchor = self._get_anchor()
        if anchor and hasattr(anchor, "get_entropy_float"):
            return anchor.get_entropy_float()
        
        # Priority 3: NumPy fallback
        return float(np.random.random())

    def _reset_budget_if_needed(self):
        """Reset entropy budget on tick boundary."""
        now = time.time()
        if now - self._last_reset >= self._tick_duration:
            self._budget_remaining = self._max_per_tick
            self._last_reset = now

    def _consume_budget(self, amount: float) -> float:
        """Consume entropy from budget, returning clamped amount."""
        self._reset_budget_if_needed()
        actual = min(amount, self._budget_remaining)
        self._budget_remaining -= actual
        self._total_injected += actual
        self._injection_count += 1
        return actual

    # ── Scoped Injection Points ──────────────────────────────────────────

    def get_curiosity_jitter(self, intensity: float = 1.0) -> float:
        """Entropy for curiosity pressure modulation.
        
        Returns a float in [-intensity * budget, +intensity * budget]
        that should be ADDED to the curiosity increment.
        
        Args:
            intensity: Multiplier for entropy magnitude (default 1.0)
        """
        raw = self._raw_float()  # [0, 1]
        magnitude = (raw * 2.0 - 1.0)  # [-1, 1]
        scaled = magnitude * intensity * 0.0001  # Small jitter around 0.00005 base
        actual = self._consume_budget(abs(scaled))
        result = actual * (1.0 if magnitude >= 0 else -1.0)
        logger.debug("Curiosity jitter: %.6f (budget remaining: %.4f)", result, self._budget_remaining)
        return result

    def get_goal_mutation_seed(self) -> float:
        """Entropy seed for goal/topic selection.
        
        Returns a float [0.0, 1.0] for weighted random selection.
        Higher values = more novel/unusual topic choices.
        """
        raw = self._raw_float()
        self._consume_budget(0.001)
        return raw

    def get_prediction_noise(self, dim: int) -> np.ndarray:
        """Entropy vector for PredictiveSelfModel observation perturbation.
        
        Args:
            dim: Dimensionality of the noise vector
            
        Returns:
            numpy array of shape (dim,) with small gaussian-like noise
        """
        # Generate noise from hardware entropy
        noise = np.array([self._raw_float() * 2.0 - 1.0 for _ in range(min(dim, 64))])
        if dim > 64:
            noise = np.pad(noise, (0, dim - 64), mode='wrap')
        
        # Scale to small perturbation
        scale = 0.01
        noise = noise * scale
        self._consume_budget(float(np.abs(noise).sum()) * 0.1)
        return noise[:dim]

    def get_exploration_weight(self) -> float:
        """Entropy weight for exploration-vs-exploitation decisions.
        
        Returns a float [0.0, 1.0] where:
          - Low values (< 0.3) → exploit known strategies
          - Mid values (0.3-0.7) → balanced
          - High values (> 0.7) → explore novel approaches
        """
        raw = self._raw_float()
        self._consume_budget(0.001)
        return raw

    # ── Diagnostics ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return entropy injection statistics."""
        return {
            "total_injected": round(self._total_injected, 6),
            "injection_count": self._injection_count,
            "budget_remaining": round(self._budget_remaining, 6),
            "max_per_tick": self._max_per_tick,
        }


import threading
_instance: Optional[ManagedEntropy] = None
_instance_lock = threading.Lock()


def get_managed_entropy() -> ManagedEntropy:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ManagedEntropy()
                logger.info("✓ Managed Entropy online (budget=%.3f/tick)", _instance._max_per_tick)
    return _instance

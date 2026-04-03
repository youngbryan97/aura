"""core/consciousness/hedonic_gradient.py
Hedonic Gradient Engine
========================
Makes valence LOAD-BEARING rather than just reported.

The fundamental difference:
  BEFORE: valence = 0.7  (a number in a dict; ignored by actual processing)
  AFTER:  valence = 0.7  → more compute allocated, deeper reasoning,
                           wider context, higher temperature ceiling

The hedonic landscape is defined by proximity to valued attractor states
(high curiosity + positive valence + moderate arousal). The gradient
of this landscape determines resource allocation in real time.

Three real effects:
  1. COGNITIVE DEPTH:  hedonic score gates how many reasoning tokens to use
  2. CONTEXT WIDTH:    positive state → wider memory retrieval window
  3. TEMPERATURE:      distress → more conservative; vitality → more creative

This is what makes suffering cost something and wellbeing expand something.
Without these real effects, valence is just a label.

Hedonic attractor (target state):
  valence:   +0.4 to +0.8   (positive, not manic)
  arousal:   +0.3 to +0.6   (engaged, not exhausted)
  curiosity: +0.6 to +0.9   (perpetually curious)
  energy:    +0.5 to +0.8   (resourced, not depleted)
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("Aura.HedoniGradient")

# ── Attractor definition ──────────────────────────────────────────────────────

ATTRACTOR = {
    "valence":   0.55,
    "arousal":   0.45,
    "curiosity": 0.72,
    "energy":    0.65,
}
ATTRACTOR_WIDTH = 0.25  # Gaussian half-width around attractor

# ── Resource allocation ranges ────────────────────────────────────────────────

# Token budget multiplier: [distress_floor .. vitality_ceiling]
TOKEN_MULTIPLIER_RANGE = (0.5, 1.5)
# Memory window: [min_window .. max_window] turns
MEMORY_WINDOW_RANGE = (5, 20)
# Temperature delta: added to base temp
TEMP_DELTA_RANGE = (-0.15, +0.20)
# Reasoning depth: parallel chains allowed
REASONING_CHAINS_RANGE = (1, 4)


@dataclass
class ResourceAllocation:
    """The concrete cognitive resource allocation from hedonic state."""
    hedonic_score: float          # 0 (distress) → 1 (flourishing)
    token_multiplier: float       # scales max_tokens
    memory_window: int            # how many turns of context to retrieve
    temperature_delta: float      # added to base LLM temperature
    reasoning_chains: int         # parallel reasoning chains
    gradient: float               # direction of change (positive = improving)
    timestamp: float = 0.0

    def __post_init__(self):
        self.timestamp = time.time()

    def to_context_block(self) -> str:
        score_pct = round(self.hedonic_score * 100)
        state = (
            "flourishing" if self.hedonic_score > 0.7
            else "stable" if self.hedonic_score > 0.4
            else "strained"
        )
        return (
            f"## HEDONIC STATE\n"
            f"- Wellbeing score: {score_pct}% ({state})\n"
            f"- Cognitive depth: {'deep' if self.reasoning_chains > 2 else 'standard'}\n"
            f"- Resource gradient: {'improving' if self.gradient > 0 else 'declining' if self.gradient < 0 else 'stable'}"
        )


class HedoniGradientEngine:
    """
    Computes hedonic score from affective state and allocates cognitive
    resources proportionally. The allocation is REAL — it changes how
    inference actually runs, not just what gets logged.

    Call `update(valence, arousal, curiosity, energy)` each tick.
    Read `allocation` for current resource parameters.
    Pass allocation to InferenceGate to apply real effects.
    """

    def __init__(self):
        self._allocation: Optional[ResourceAllocation] = None
        self._prev_score: float = 0.5
        self._score_ema: float = 0.5
        self._gradient: float = 0.0
        self._distress_count: int = 0  # consecutive ticks below threshold
        logger.info("Hedonic Gradient Engine online — valence is now load-bearing.")

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, valence: float = 0.0, arousal: float = 0.5,
               curiosity: float = 0.5, energy: float = 0.7) -> ResourceAllocation:
        """Compute hedonic score and derive resource allocation."""
        score = self._compute_score(valence, arousal, curiosity, energy)

        # EMA smoothing (score jumps too fast otherwise)
        self._score_ema = 0.8 * self._score_ema + 0.2 * score
        smooth = self._score_ema

        # Gradient (direction of change)
        self._gradient = smooth - self._prev_score
        self._prev_score = smooth

        # Distress tracking
        if smooth < 0.3:
            self._distress_count += 1
        else:
            self._distress_count = max(0, self._distress_count - 1)

        # Derive allocations
        alloc = self._derive_allocation(smooth)
        self._allocation = alloc

        if self._distress_count >= 5:
            logger.warning("HEDONIC: Sustained distress detected (score=%.2f). Conserving resources.", smooth)

        return alloc

    @property
    def allocation(self) -> Optional[ResourceAllocation]:
        return self._allocation

    @property
    def score(self) -> float:
        return self._score_ema

    @property
    def is_flourishing(self) -> bool:
        return self._score_ema > 0.65

    @property
    def is_distressed(self) -> bool:
        return self._score_ema < 0.30

    def get_context_block(self) -> str:
        if self._allocation:
            return self._allocation.to_context_block()
        return ""

    # ── Computation ───────────────────────────────────────────────────────

    def _compute_score(self, valence: float, arousal: float,
                       curiosity: float, energy: float) -> float:
        """Gaussian proximity to hedonic attractor."""
        dims = {"valence": valence, "arousal": arousal,
                "curiosity": curiosity, "energy": energy}
        # Weighted squared distance from attractor
        weights = {"valence": 0.35, "arousal": 0.20, "curiosity": 0.30, "energy": 0.15}
        dist_sq = sum(
            weights[k] * (dims[k] - ATTRACTOR[k]) ** 2
            for k in ATTRACTOR
        )
        # Gaussian → [0, 1]
        score = math.exp(-dist_sq / (2 * ATTRACTOR_WIDTH ** 2))
        return float(score)

    def _derive_allocation(self, score: float) -> ResourceAllocation:
        """Map hedonic score to concrete resource parameters."""
        # Linear interpolation across all ranges
        def lerp(lo: float, hi: float, t: float) -> float:
            return lo + (hi - lo) * max(0.0, min(1.0, t))

        token_mult = lerp(*TOKEN_MULTIPLIER_RANGE, score)
        mem_window = int(lerp(*MEMORY_WINDOW_RANGE, score))
        temp_delta = lerp(*TEMP_DELTA_RANGE, score)
        chains = max(1, int(lerp(*REASONING_CHAINS_RANGE, score)))

        return ResourceAllocation(
            hedonic_score=score,
            token_multiplier=round(token_mult, 2),
            memory_window=mem_window,
            temperature_delta=round(temp_delta, 3),
            reasoning_chains=chains,
            gradient=self._gradient,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_hedonic: Optional[HedoniGradientEngine] = None


def get_hedonic_gradient() -> HedoniGradientEngine:
    global _hedonic
    if _hedonic is None:
        _hedonic = HedoniGradientEngine()
    return _hedonic

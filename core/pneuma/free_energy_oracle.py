"""core/pneuma/free_energy_oracle.py
PNEUMA Layer 5 — Free Energy Oracle (FEO).

Implements the Free Energy Principle (Friston, 2010) to select responses
that minimize Expected Free Energy (EFE).

EFE(π) = E_q[log q(s) - log p(s,o|π)]
       ≈ -E[epistemic_value] - E[pragmatic_value]

Where:
  Epistemic value   = information gain (reduce uncertainty about hidden states)
  Pragmatic value   = expected reward (alignment with preferred outcomes)

For Aura's response selection:
  - Candidates are scored by their predicted effect on belief uncertainty
    (info gain) and alignment with heartstone values (pragmatic value).
  - The response with minimum EFE (most aligned, least uncertain) is selected.
"""

from core.runtime.errors import record_degradation
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("PNEUMA.FreeEnergyOracle")

_EPS = 1e-8


@dataclass
class ResponseCandidate:
    """A candidate response with EFE scoring."""
    text: str
    predicted_belief_shift: np.ndarray   # expected change in belief vector
    pragmatic_score: float = 0.5         # alignment with preferred outcomes [0,1]
    epistemic_score: float = 0.5         # expected info gain [0,1]
    efe: float = 0.0                     # total Expected Free Energy (lower = better)


class FreeEnergyOracle:
    """Scores response candidates by Expected Free Energy.

    Integrates:
    - Belief uncertainty from IGTracker (epistemic component)
    - Value alignment from HeartstoneValues (pragmatic component)
    - Topological stability from TopologicalMemory (structural component)
    """

    def __init__(
        self,
        epistemic_weight: float = 0.4,
        pragmatic_weight: float = 0.4,
        structural_weight: float = 0.2,
    ):
        self.w_epistemic = epistemic_weight
        self.w_pragmatic = pragmatic_weight
        self.w_structural = structural_weight
        self._last_efe: float = 0.0
        self._best_candidate: Optional[ResponseCandidate] = None
        logger.info(
            "FreeEnergyOracle online (w_e=%.2f w_p=%.2f w_s=%.2f)",
            epistemic_weight, pragmatic_weight, structural_weight,
        )

    def score_candidates(
        self,
        candidates: List[ResponseCandidate],
        current_belief: np.ndarray,
        ig_stability: float = 1.0,
        topo_complexity: float = 0.0,
    ) -> List[ResponseCandidate]:
        """Score all candidates and sort by EFE (ascending — lower is better)."""
        if not candidates:
            return candidates

        for c in candidates:
            c.efe = self._compute_efe(
                c, current_belief, ig_stability, topo_complexity
            )

        candidates.sort(key=lambda x: x.efe)
        self._best_candidate = candidates[0]
        self._last_efe = candidates[0].efe
        return candidates

    def _compute_efe(
        self,
        candidate: ResponseCandidate,
        current_belief: np.ndarray,
        ig_stability: float,
        topo_complexity: float,
    ) -> float:
        """Compute EFE for a single candidate.

        EFE = w_e * epistemic_cost + w_p * pragmatic_cost + w_s * structural_cost

        epistemic_cost  = predicted uncertainty after applying belief shift
                          (lower when candidate resolves ambiguity)
        pragmatic_cost  = 1 - alignment_with_values
        structural_cost = topological complexity (more loops = more cost)
        """
        # Epistemic: estimate post-shift uncertainty
        if candidate.predicted_belief_shift is not None and len(candidate.predicted_belief_shift) > 0:
            shifted = current_belief[:len(candidate.predicted_belief_shift)] + candidate.predicted_belief_shift
            # Entropy of normalized absolute values as uncertainty proxy
            shifted_abs = np.abs(shifted) + _EPS
            shifted_prob = shifted_abs / shifted_abs.sum()
            post_entropy = -float(np.sum(shifted_prob * np.log(shifted_prob + _EPS)))
            # Normalize by max entropy
            max_entropy = math.log(len(shifted_prob))
            epistemic_cost = post_entropy / max_entropy if max_entropy > 0 else 0.5
        else:
            epistemic_cost = 1.0 - candidate.epistemic_score

        pragmatic_cost = 1.0 - candidate.pragmatic_score
        structural_cost = topo_complexity * (1.0 - ig_stability)

        efe = (
            self.w_epistemic * epistemic_cost
            + self.w_pragmatic * pragmatic_cost
            + self.w_structural * structural_cost
        )
        return round(efe, 6)

    def get_preferred_temperature(self, base_temp: float = 0.72) -> float:
        """Adjust LLM temperature based on last EFE: high EFE → lower temp (more focused)."""
        if self._last_efe > 0.7:
            return max(0.4, base_temp - 0.2)
        elif self._last_efe < 0.3:
            return min(1.0, base_temp + 0.1)
        return base_temp

    def quick_score_text(
        self,
        text: str,
        belief_vector: np.ndarray,
        ig_stability: float = 1.0,
    ) -> float:
        """Fast heuristic EFE score for a single text without belief shift prediction.

        Used when full candidate scoring is too expensive.
        """
        # Pragmatic: estimate from heartstone values via keyword heuristic
        pragmatic = self._heuristic_pragmatic(text)
        # Epistemic: use length and question-marks as uncertainty proxy
        question_density = text.count("?") / max(1, len(text.split()))
        epistemic_cost = min(1.0, question_density * 5 + 0.3)

        efe = self.w_epistemic * epistemic_cost + self.w_pragmatic * (1.0 - pragmatic)
        return efe

    def _heuristic_pragmatic(self, text: str) -> float:
        """Rough pragmatic score from text via heartstone value keywords."""
        score = 0.5
        try:
            from core.affect.heartstone_values import get_heartstone_values
            vals = get_heartstone_values().values
            curious_words = ["why", "how", "explore", "discover", "learn", "curious"]
            empathy_words = ["feel", "understand", "care", "help", "support"]
            t_lower = text.lower()
            if any(w in t_lower for w in curious_words):
                score += 0.1 * vals.get("Curiosity", 0.5)
            if any(w in t_lower for w in empathy_words):
                score += 0.1 * vals.get("Empathy", 0.5)
        except Exception as _exc:
            record_degradation('free_energy_oracle', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return min(1.0, score)

    def get_state_dict(self) -> dict:
        return {
            "last_efe": round(self._last_efe, 4),
            "preferred_temp": round(self.get_preferred_temperature(), 4),
            "best_candidate_preview": (
                self._best_candidate.text[:60] if self._best_candidate else None
            ),
        }

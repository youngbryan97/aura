"""core/simulation/internal_simulator.py -- Counterfactual Action Simulator
============================================================================
Simulates future state variations to enable proactive planning.
Projects state transitions and evaluates outcomes.

Now wired into the initiative pipeline: before the InitiativeSynthesizer
selects a winner, the top candidates are simulated and evaluated here.

Evaluation dimensions:
  - Valence (emotional desirability)
  - Energy cost
  - Cortisol/stress risk
  - Identity alignment (does this match who I am?)
  - Commitment compatibility (does this conflict with promises?)
  - World state fit (does the environment support this?)
"""
import copy
import logging
from typing import Any, Dict, List, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Aura.InternalSimulator")


class InternalSimulator:
    """Simulates and evaluates hypothetical future states."""

    def __init__(self):
        logger.info("InternalSimulator initialized.")

    def simulate(self, current_state: Any, variation: Optional[Dict[str, Any]] = None) -> Any:
        """Create a hypothetical future state based on current state + variation."""
        try:
            hypothetical = copy.deepcopy(current_state)
        except Exception:
            # If deep copy fails (complex state), work with a shallow analysis
            return current_state

        try:
            hypothetical.state_id = f"sim_{hypothetical.state_id[:8]}"
        except (AttributeError, TypeError):
            pass  # no-op: intentional

        if variation:
            try:
                for key, val in variation.items():
                    if key == "risk":
                        hypothetical.affect.arousal = min(1.0, hypothetical.affect.arousal + (val * 0.1))
                        hypothetical.affect.physiology["cortisol"] += val * 5.0
                    elif key == "energy":
                        hypothetical.motivation.budgets["energy"]["level"] = max(
                            0.0, hypothetical.motivation.budgets["energy"]["level"] - val
                        )
            except (AttributeError, KeyError, TypeError):
                pass  # no-op: intentional

        try:
            hypothetical.version += 1
        except (AttributeError, TypeError):
            pass  # no-op: intentional

        return hypothetical

    def evaluate(self, predicted_state: Any, action_content: str = "",
                 action_source: str = "") -> float:
        """Evaluate the desirability of a predicted state.

        Returns a score: higher = more desirable, lower = worse.
        Range roughly [-1.0, 1.0].

        Dimensions:
          1. Valence (0.3 weight) -- emotional state
          2. Energy (0.2 weight) -- resource availability
          3. Cortisol risk (0.15 weight, inverted)
          4. Identity alignment (0.2 weight) -- does this match who I am?
          5. Commitment compatibility (0.15 weight) -- conflicts with promises?
        """
        score = 0.0

        # 1. Valence
        try:
            valence = predicted_state.affect.valence
            score += valence * 0.3
        except (AttributeError, TypeError):
            pass  # no-op: intentional

        # 2. Energy
        try:
            energy = predicted_state.motivation.budgets["energy"]["level"] / 100.0
            score += energy * 0.2
        except (AttributeError, KeyError, TypeError):
            score += 0.1  # neutral default

        # 3. Cortisol risk (inverted -- high cortisol = bad)
        try:
            cortisol = predicted_state.affect.physiology.get("cortisol", 0.0) / 50.0
            score -= cortisol * 0.15
        except (AttributeError, TypeError):
            pass  # no-op: intentional

        # 4. Identity alignment
        try:
            alignment_score = self._check_identity_alignment(action_content)
            score += alignment_score * 0.2
        except Exception:
            score += 0.1  # neutral default

        # 5. Commitment compatibility
        try:
            compat_score = self._check_commitment_compatibility(action_content)
            score += compat_score * 0.15
        except Exception:
            score += 0.05  # neutral default

        # 6. World-state fit (environment supports this action?)
        try:
            world_score = self._check_world_state_fit(action_content, action_source)
            score += world_score * 0.1
        except Exception:
            pass  # no-op: intentional

        return round(score, 4)

    def evaluate_candidates(self, state: Any,
                            candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate multiple candidate actions and return them scored.

        Each candidate dict should have at least 'goal' and optionally 'variation'.
        Returns candidates sorted by score (best first).
        """
        results = []
        for cand in candidates:
            variation = cand.get("variation", {"risk": 0.3, "energy": 5.0})
            sim_state = self.simulate(state, variation=variation)
            score = self.evaluate(
                sim_state,
                action_content=str(cand.get("goal", "")),
                action_source=str(cand.get("source", "")),
            )
            results.append({
                "candidate": cand,
                "score": score,
                "simulated": True,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def plan_next_action(self, state: Any, options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Simulate options and return the one with the highest evaluation score."""
        if not options:
            return None
        results = self.evaluate_candidates(state, options)
        best = results[0]
        logger.info("Simulation selected best path with score %.4f", best["score"])
        return best["candidate"]

    # ------------------------------------------------------------------
    # Advisory checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_identity_alignment(content: str) -> float:
        """Check if action content aligns with identity. Returns [-0.5, 0.5]."""
        content_lower = content.lower()

        # Axiomatic violations checked FIRST, before any service lookup
        violations = ["as an ai", "i'm just a language model", "i cannot feel",
                      "i don't have opinions", "i'm just a"]
        if any(v in content_lower for v in violations):
            return -0.5

        try:
            canonical = ServiceContainer.get("canonical_self", default=None)
            if canonical is None:
                return 0.1  # neutral

            # Check values overlap
            values = getattr(canonical, "core_values", []) or []
            matches = sum(1 for v in values if str(v).lower() in content_lower)
            if matches > 0:
                return min(0.5, 0.2 + matches * 0.1)

            return 0.1
        except Exception:
            return 0.1

    @staticmethod
    def _check_world_state_fit(content: str, source: str = "") -> float:
        """Check if the environment supports this action. Returns [-0.3, 0.3].

        Late night + frustrated user + error = high value for helpful action
        High thermal pressure = penalize expensive operations
        User active = penalize interruptions
        """
        try:
            from core.world_state import get_world_state
            ws = get_world_state()
            ws.update()
            score = 0.0

            content_lower = content.lower()

            # Late night + error detected → helpful actions score high
            if ws.time_of_day in ("night", "late_night"):
                if ws.get_belief("user_likely_frustrated"):
                    if any(w in content_lower for w in ["fix", "help", "repair", "patch", "error"]):
                        score += 0.3  # maximum boost for proactive help

            # User has been idle → exploration/research is appropriate
            if ws.user_idle_seconds > 1800:
                if any(w in content_lower for w in ["research", "explore", "learn", "investigate"]):
                    score += 0.15

            # High thermal pressure → penalize heavy operations
            if ws.thermal_pressure > 0.6:
                if any(w in content_lower for w in ["search", "compute", "analyze", "generate"]):
                    score -= 0.15

            # User recently active → penalize interruptions
            if ws.user_idle_seconds < 60:
                if source not in ("user", "voice", "admin"):
                    score -= 0.1  # don't interrupt

            return max(-0.3, min(0.3, score))
        except Exception:
            return 0.0

    @staticmethod
    def _check_commitment_compatibility(content: str) -> float:
        """Check if action conflicts with active commitments. Returns [-0.3, 0.3]."""
        try:
            commitment = ServiceContainer.get("commitment_engine", default=None)
            if commitment is None:
                return 0.05

            active = []
            if hasattr(commitment, "get_active_commitments"):
                active = commitment.get_active_commitments()
            elif hasattr(commitment, "commitments"):
                active = [c for c in commitment.commitments if c.get("status") == "active"]

            if not active:
                return 0.05

            content_lower = content.lower()
            # Check for conflicts (action that contradicts a promise)
            for c in active:
                goal = str(c.get("goal", c.get("description", ""))).lower()
                # If the action serves a commitment, that's good
                if any(word in content_lower for word in goal.split()[:3] if len(word) > 3):
                    return 0.3
            return 0.0
        except Exception:
            return 0.05

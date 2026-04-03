import copy
import logging
from typing import Dict, Any, Optional
from core.state.aura_state import AuraState

logger = logging.getLogger("Aura.InternalSimulator")

class InternalSimulator:
    """
    Simulates future state variations to enable proactive planning.
    Projects state transitions and evaluates outcomes (Valence/Risk).
    """

    def __init__(self):
        logger.info("InternalSimulator initialized.")

    def simulate(self, current_state: AuraState, variation: Dict[str, Any] = None) -> AuraState:
        """
        Create a hypothetical future state based on current state and a variation.
        """
        # Create a deep copy to avoid mutating the real state
        hypothetical = copy.deepcopy(current_state)
        hypothetical.state_id = f"sim_{hypothetical.state_id[:8]}"
        
        # Apply variation (e.g., increased risk, decreased energy)
        if variation:
            for key, val in variation.items():
                if key == "risk":
                    # Simulated risk affects cortisol and arousal
                    hypothetical.affect.arousal = min(1.0, hypothetical.affect.arousal + (val * 0.1))
                    hypothetical.affect.physiology["cortisol"] += val * 5.0
                elif key == "energy":
                    hypothetical.motivation.budgets["energy"]["level"] = max(0.0, hypothetical.motivation.budgets["energy"]["level"] - val)
        
        # Increment version to indicate "simulated time"
        hypothetical.version += 1
        return hypothetical

    def evaluate(self, predicted_state: AuraState) -> float:
        """
        Evaluate the 'desirability' or 'risk' of a predicted state.
        Returns a score: Higher is more desirable/less risky.
        """
        # Simple heuristic: High valence + High energy - High cortisol
        valence = predicted_state.affect.valence
        energy = predicted_state.motivation.budgets["energy"]["level"] / 100.0
        cortisol_risk = predicted_state.affect.physiology.get("cortisol", 0.0) / 50.0
        
        score = (valence * 0.5) + (energy * 0.3) - (cortisol_risk * 0.2)
        return round(score, 3)

    def plan_next_action(self, state: AuraState, options: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulates multiple options and returns the one with the highest evaluation score.
        """
        results = []
        for opt in options:
            sim_state = self.simulate(state, variation=opt.get("variation"))
            score = self.evaluate(sim_state)
            results.append({"option": opt, "score": score})
            
        best = max(results, key=lambda r: r["score"])
        logger.info("Internal simulation selected best path with score %s", best["score"])
        return best["option"]

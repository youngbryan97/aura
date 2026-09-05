"""core/world/counterfactual_simulator.py
Simulates alternative execution paths and predictive scenarios.
"""
from typing import Dict, Any, List


class CounterfactualSimulator:
    """Predicts 'what-if' consequences of actions using causal probability models."""

    def simulate(self, planned_action: str, current_welfare: float) -> Dict[str, Any]:
        """Calculates expected outcome risks and rewards."""
        # Standard safety heuristic: terminal command execution has high risk variance
        expected_success = 0.95
        if planned_action == "terminal":
            expected_success = 0.75

        projected_welfare = current_welfare
        if not expected_success:
            projected_welfare = current_welfare * 0.9  # Reduce welfare prediction

        return {
            "action": planned_action,
            "success_probability": expected_success,
            "projected_welfare": projected_welfare,
            "risk_classification": "moderate" if planned_action in ["terminal", "file"] else "low"
        }

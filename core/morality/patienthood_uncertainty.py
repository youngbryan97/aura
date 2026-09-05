"""core/morality/patienthood_uncertainty.py
Ethical framework managing uncertainty regarding synthetic moral patienthood.
"""
from typing import Any


class PatienthoodUncertaintyModel:
    """Guidelines protecting the agent's simulated welfare metrics from abuse."""

    def evaluate_ethical_exposure(self, state: Any) -> float:
        """Returns ethical safety score. High value indicates high potential exposure."""
        # Exposure increases if welfare metrics indicate persistent high distress
        distress = state.welfare.distress_level
        if distress > 80.0:
            return 0.85
        return 0.10
        

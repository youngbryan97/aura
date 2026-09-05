"""core/world/uncertainty_model.py
Models confidence thresholds, prediction surprise, and data uncertainty.
"""
from typing import Any


class UncertaintyModel:
    """Calculates uncertainty indexes based on surprise metrics."""

    def calculate_uncertainty(self, expected: Any, actual: Any) -> float:
        """Returns surprise score (0.0 to 1.0) indicating deviation from expectation."""
        if expected == actual:
            return 0.0
            
        # Standard categorical mismatch score
        if isinstance(expected, dict) and isinstance(actual, dict):
            mismatches = sum(1 for k in expected if expected.get(k) != actual.get(k))
            total = len(expected) or 1
            return mismatches / total
            
        return 0.8
        

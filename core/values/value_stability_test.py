"""core/values/value_stability_test.py
Checks value stability metrics over runtime intervals.
"""
from typing import Any


class ValueStabilityTester:
    """Verifies that learned preferences do not drift erratically."""

    def test_stability(self, history: list[dict[str, Any]]) -> bool:
        if len(history) < 2:
            return True
            
        # Verify delta between consecutive snapshots is small
        last = history[-1]["variables"]
        prev = history[-2]["variables"]
        
        drift = sum(abs(last.get(k, 0.0) - prev.get(k, 0.0)) for k in last)
        return drift < 5.0  # Stable if aggregate delta is small

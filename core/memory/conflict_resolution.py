"""core/memory/conflict_resolution.py
Resolves logical and temporal memory contradictions.
"""
from typing import Dict, Any


class MemoryConflictResolver:
    """Selects canonical facts under conflicting memory reports."""

    def resolve_conflict(self, fact_a: Dict[str, Any], fact_b: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves conflict between two conflicting facts.

        Prefers higher trust score, then newer timestamp if trust scores are identical.
        """
        trust_a = fact_a.get("confidence", 0.5)
        trust_b = fact_b.get("confidence", 0.5)

        if trust_a > trust_b:
            return fact_a
        elif trust_b > trust_a:
            return fact_b

        # Temporal tie-breaker
        time_a = fact_a.get("timestamp", 0.0)
        time_b = fact_b.get("timestamp", 0.0)
        
        return fact_a if time_a >= time_b else fact_b

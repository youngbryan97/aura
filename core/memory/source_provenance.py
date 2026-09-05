"""core/memory/source_provenance.py
Tracks memory source origin validation and confidence metrics.
"""
from typing import Dict, Any


class SourceProvenanceResolver:
    """Calculates confidence values for fact assertions based on source origin."""

    def resolve_confidence(self, origin: str) -> float:
        """Assign trust scores based on source types."""
        # 2026 Standards: direct user input and secure local files have highest trust.
        weights = {
            "user_direct": 0.99,
            "local_file": 0.90,
            "shell_execution": 0.85,
            "network_egress": 0.70,
            "hallucinated_inference": 0.20
        }
        return weights.get(origin, 0.50)

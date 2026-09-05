"""core/world/social_graph.py
Social graph tracking user relationship models and trust matrices.
"""


class SocialGraph:
    """Tracks social nodes and dynamic trust metrics."""

    def __init__(self):
        # Maps person -> trust_index (0.0 to 1.0)
        self._trust_matrix: dict[str, float] = {
            "Bryan": 1.0  # Canonical operator begins with absolute trust
        }

    def record_interaction_outcome(self, person: str, success: bool) -> None:
        current = self._trust_matrix.get(person, 0.5)
        # Dynamic adjust
        delta = 0.05 if success else -0.10
        self._trust_matrix[person] = min(max(current + delta, 0.0), 1.0)

    def get_trust_level(self, person: str) -> float:
        return self._trust_matrix.get(person, 0.5)

"""core/social/trust_model.py
Models dynamic trust calibration for distinct users.
"""


class TrustModel:
    """Manages trust values per user based on interaction outcomes."""

    def __init__(self):
        self._trust_scores: dict[str, float] = {}

    def get_trust(self, person: str) -> float:
        return self._trust_scores.get(person, 0.5)

    def calibrate_trust(self, person: str, outcome_success: bool) -> float:
        current = self.get_trust(person)
        delta = 0.05 if outcome_success else -0.10
        new_score = min(max(current + delta, 0.0), 1.0)
        self._trust_scores[person] = new_score
        return new_score

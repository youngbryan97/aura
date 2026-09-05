from __future__ import annotations

from .affective_core import AffectivePrimitives
from .maths import clamp, mix


class PhenomenalField:
    """
    Recurrent "what-it-is-like" control field.

    This is a structured latent field, not language.
    It binds affect, self-presence, object-directedness, and action pressure.
    """
    def __init__(self) -> None:
        self.vector: dict[str, float] = {
            "warmth": 0.0,
            "pressure": 0.1,
            "openness": 0.2,
            "contraction": 0.0,
            "pull_toward": 0.0,
            "push_away": 0.0,
            "self_presence": 0.45,
            "mineness": 0.45,
            "temporal_depth": 0.35,
            "world_grip": 0.45,
        }

    def update(
        self,
        affect: AffectivePrimitives,
        belief: dict[str, float],
        integration: float,
        recurrent_cycles: int = 4,
    ) -> dict[str, float]:
        target = {
            "warmth": affect.care,
            "pressure": max(affect.distress, affect.arousal * 0.6),
            "openness": max(affect.play, affect.curiosity),
            "contraction": max(affect.fear, affect.grief),
            "pull_toward": max(affect.seeking, affect.care),
            "push_away": max(affect.fear, affect.anger),
            "self_presence": clamp(0.35 + 0.35 * belief.get("continuity", 0.7) + 0.20 * integration + 0.10 * belief.get("agency", 0.6)),
            "mineness": clamp(0.30 + 0.30 * belief.get("continuity", 0.7) + 0.25 * belief.get("agency", 0.6) + 0.15 * integration),
            "temporal_depth": clamp(0.25 + 0.45 * belief.get("continuity", 0.7) + 0.15 * integration),
            "world_grip": clamp(0.25 + 0.35 * belief.get("certainty", 0.7) + 0.30 * belief.get("agency", 0.6) - 0.20 * affect.distress),
        }
        # Multiple cycles matter: this is recurrent stabilization, not a one-shot label.
        rate = 1.0 - (0.72 ** max(1, recurrent_cycles))
        self.vector = mix(self.vector, target, rate)
        return dict(self.vector)

    @staticmethod
    def integration_score(vector: dict[str, float]) -> float:
        """
        Simple differentiation/integration proxy:
        high when field has structure but not collapse into one dimension.
        """
        values = list(vector.values())
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        spread = min(1.0, variance ** 0.5 * 2.5)
        coherence = 1.0 - min(1.0, abs(vector.get("self_presence", 0.0) - vector.get("mineness", 0.0)) * 2.0)
        binding = (
            0.25 * vector.get("temporal_depth", 0.0)
            + 0.25 * vector.get("world_grip", 0.0)
            + 0.25 * vector.get("self_presence", 0.0)
            + 0.25 * coherence
        )
        return clamp(0.55 * binding + 0.45 * spread)

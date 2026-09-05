from __future__ import annotations

from dataclasses import dataclass, field

from .maths import Vector, bound01, clamp, l2, mix, weighted_error


@dataclass
class GenerativeModel:
    """
    Minimal active-inference layer.

    Beliefs predict body/runtime state.
    Prediction error updates beliefs.
    Precision controls how forcefully errors matter.
    Free energy is precision-weighted unresolved error.
    """
    belief: Vector = field(default_factory=lambda: {
        "energy": 0.70,
        "continuity": 0.80,
        "agency": 0.60,
        "safety": 0.80,
        "social": 0.50,
        "novelty": 0.25,
        "certainty": 0.75,
        "low_compute_pressure": 0.80,
        "low_memory_pressure": 0.80,
        "low_error_pressure": 0.80,
    })
    precision: Vector = field(default_factory=lambda: {
        "energy": 1.1,
        "continuity": 1.5,
        "agency": 1.2,
        "safety": 1.8,
        "social": 1.0,
        "novelty": 0.7,
        "certainty": 1.4,
        "low_compute_pressure": 0.9,
        "low_memory_pressure": 0.9,
        "low_error_pressure": 1.3,
    })
    learning_rate: float = 0.26

    def infer(self, observed: Vector, recurrent_cycles: int = 4) -> tuple[Vector, Vector, float]:
        recurrent_cycles = max(1, int(recurrent_cycles))
        local = dict(self.belief)
        for _ in range(recurrent_cycles):
            err = weighted_error(local, observed, self.precision)
            correction = {k: v * self.learning_rate for k, v in err.items()}
            local = bound01({k: local.get(k, 0.0) + correction.get(k, 0.0) for k in set(local) | set(correction)})
        final_error = weighted_error(local, observed, self.precision)
        free_energy = l2(final_error)
        self.belief = mix(self.belief, local, 0.55)
        return dict(self.belief), final_error, free_energy

    def expected_relief(self, action: str) -> float:
        """
        Prior estimate: how much this action should reduce free energy.
        Downstream planners can replace this with learned values.
        """
        table = {
            "seek_information": 0.22 * (1.0 - self.belief.get("certainty", 0.5)) + 0.10 * self.belief.get("novelty", 0.3),
            "ask_for_clarification": 0.20 * (1.0 - self.belief.get("certainty", 0.5)),
            "protect_boundary": 0.30 * (1.0 - self.belief.get("safety", 0.5)),
            "continue_goal": 0.25 * self.belief.get("agency", 0.5),
            "social_repair": 0.25 * (1.0 - self.belief.get("social", 0.5)),
            "restabilize": 0.20 * (1.0 - self.belief.get("continuity", 0.5)) + 0.10 * (1.0 - self.belief.get("energy", 0.5)),
            "play_explore": 0.18 * self.belief.get("novelty", 0.2) * self.belief.get("safety", 0.8),
        }
        return clamp(table.get(action, 0.05))

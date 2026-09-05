from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

from .maths import sigmoid, clamp, normalize_sum

@dataclass
class Coalition:
    name: str
    content: Dict[str, Any]
    salience: float
    affect_gain: float
    precision: float
    persistence: float = 0.0

    @property
    def ignition_score(self) -> float:
        return self.salience * 0.45 + self.affect_gain * 0.35 + self.precision * 0.20 + self.persistence * 0.15

@dataclass
class GlobalWorkspace:
    """
    Recurrent global workspace.

    Candidate coalitions compete. A winner becomes globally available when
    salience + precision + affective gain crosses ignition threshold.
    """
    threshold: float = 0.56
    recurrent_gain: float = 0.28
    last_winner: str = "none"
    ignition_trace: Dict[str, float] = field(default_factory=dict)

    def compete(self, coalitions: List[Coalition], cycles: int = 4) -> Dict[str, Any]:
        if not coalitions:
            return {
                "ignited": False,
                "winner": "none",
                "availability": 0.0,
                "content": {},
                "competition": {},
            }

        scores = {c.name: c.ignition_score + self.ignition_trace.get(c.name, 0.0) for c in coalitions}
        for _ in range(max(1, cycles)):
            weights = normalize_sum(scores)
            for c in coalitions:
                recurrence = self.recurrent_gain * weights.get(c.name, 0.0)
                scores[c.name] = scores[c.name] + recurrence - 0.05 * (1.0 - weights.get(c.name, 0.0))

        winner = max(coalitions, key=lambda c: scores[c.name])
        availability = sigmoid(scores[winner.name], gain=8.0, bias=self.threshold)
        ignited = availability >= 0.50
        for c in coalitions:
            old = self.ignition_trace.get(c.name, 0.0)
            self.ignition_trace[c.name] = clamp(old * 0.55 + (0.22 if c.name == winner.name and ignited else 0.0))
        self.last_winner = winner.name if ignited else "none"
        return {
            "ignited": ignited,
            "winner": winner.name if ignited else "none",
            "availability": availability,
            "content": winner.content if ignited else {},
            "competition": {k: round(v, 4) for k, v in scores.items()},
        }

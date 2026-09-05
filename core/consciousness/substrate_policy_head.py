"""Substrate policy head.

This makes the continuous substrate a decision participant, not merely a tone
source.  The head maps LTC state, phi, affect, prediction error, scar pressure,
and resource state into policy weights consumed by goal selection, memory
salience, tool affordance, risk thresholds, exploration, questioning, and
repair priority.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


POLICY_KEYS = (
    "goal_priority",
    "memory_salience",
    "tool_affordance",
    "risk_threshold",
    "exploration_budget",
    "question_threshold",
    "repair_priority",
)


@dataclass(frozen=True)
class SubstratePolicyInput:
    state64: Sequence[float]
    phi: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    prediction_error: float = 0.0
    scar_pressure: float = 0.0
    resource_headroom: float = 1.0
    continuity: float = 1.0


@dataclass(frozen=True)
class SubstratePolicyVector:
    weights: dict[str, float]
    mode: str
    generated_at: float = field(default_factory=time.time)
    evidence: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.weights[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {k: round(v, 5) for k, v in self.weights.items()},
            "mode": self.mode,
            "generated_at": self.generated_at,
            "evidence": {k: round(v, 5) for k, v in self.evidence.items()},
        }


class SubstratePolicyHead:
    """Deterministic lightweight MLP-style policy projection."""

    def __init__(self) -> None:
        self._centers = {
            "goal_priority": 0.10,
            "memory_salience": 0.25,
            "tool_affordance": 0.40,
            "risk_threshold": 0.55,
            "exploration_budget": 0.70,
            "question_threshold": 0.82,
            "repair_priority": 0.94,
        }

    def compute(self, inputs: SubstratePolicyInput, *, ablation: str = "full") -> SubstratePolicyVector:
        state = self._normalize_state(inputs.state64)
        features = self._features(inputs, state)
        if ablation == "prompt_only":
            features = {key: 0.0 for key in features}
            features["valence"] = inputs.valence * 0.15
            features["arousal"] = inputs.arousal * 0.10
        elif ablation == "policy_only":
            features["valence"] = 0.0
        elif ablation == "no_substrate_policy":
            return SubstratePolicyVector(
                weights={key: 0.5 for key in POLICY_KEYS},
                mode=ablation,
                evidence={"disabled": 1.0},
            )

        weights = {
            "goal_priority": self._sigmoid(0.8 * features["phi"] + 0.5 * features["dominance"] - 0.3 * features["scar"]),
            "memory_salience": self._sigmoid(0.7 * features["prediction"] + 0.4 * features["scar"] + 0.4 * features["continuity"]),
            "tool_affordance": self._sigmoid(0.8 * features["resource"] + 0.4 * features["dominance"] - 0.5 * features["scar"]),
            "risk_threshold": self._sigmoid(0.7 * features["continuity"] + 0.4 * features["resource"] - 0.9 * features["scar"]),
            "exploration_budget": self._sigmoid(0.8 * features["curiosity"] + 0.5 * features["resource"] - 0.6 * features["prediction"]),
            "question_threshold": self._sigmoid(0.9 * features["prediction"] + 0.3 * features["scar"] - 0.2 * features["dominance"]),
            "repair_priority": self._sigmoid(0.9 * features["prediction"] + 0.8 * (1.0 - features["continuity"]) + 0.4 * (1.0 - features["resource"])),
        }
        return SubstratePolicyVector(weights={k: self._clamp(v) for k, v in weights.items()}, mode=ablation, evidence=features)

    def choose_goal(self, priorities: Mapping[str, float], policy: SubstratePolicyVector) -> str:
        if not priorities:
            return ""
        risk = policy["risk_threshold"]
        exploration = policy["exploration_budget"]
        repair = policy["repair_priority"]
        scores: dict[str, float] = {}
        for goal, base in priorities.items():
            text = goal.lower()
            score = float(base) * (0.6 + policy["goal_priority"])
            if "repair" in text or "fix" in text:
                score *= 0.7 + repair
            if "explore" in text or "research" in text:
                score *= 0.7 + exploration
            if "risky" in text or "mutate" in text:
                score *= max(0.1, risk)
            scores[goal] = score
        return max(scores, key=scores.get)

    def ablation_report(self, inputs: SubstratePolicyInput) -> dict[str, Any]:
        reports = {
            mode: self.compute(inputs, ablation=mode).to_dict()
            for mode in ("full", "prompt_only", "policy_only", "no_substrate_policy")
        }
        full = reports["full"]["weights"]
        prompt = reports["prompt_only"]["weights"]
        distance = sum(abs(float(full[k]) - float(prompt[k])) for k in POLICY_KEYS) / len(POLICY_KEYS)
        return {"modes": reports, "full_vs_prompt_mean_abs_delta": round(distance, 5)}

    def _features(self, inputs: SubstratePolicyInput, state: list[float]) -> dict[str, float]:
        bands = self._band_means(state)
        return {
            "phi": self._clamp(inputs.phi / 2.0),
            "valence": self._scale_signed(inputs.valence),
            "arousal": self._clamp(inputs.arousal),
            "dominance": self._scale_signed(inputs.dominance),
            "prediction": self._clamp(inputs.prediction_error),
            "scar": self._clamp(inputs.scar_pressure),
            "resource": self._clamp(inputs.resource_headroom),
            "continuity": self._clamp(inputs.continuity),
            "curiosity": self._clamp((bands[4] + inputs.arousal + max(0.0, inputs.prediction_error)) / 3.0),
            "state_gradient": self._clamp(sum(abs(state[i] - state[i - 1]) for i in range(1, len(state))) / max(1, len(state) - 1)),
        }

    @staticmethod
    def _normalize_state(state64: Sequence[float]) -> list[float]:
        values = [float(x) for x in list(state64)[:64]]
        if len(values) < 64:
            values.extend([0.0] * (64 - len(values)))
        return [0.5 + 0.5 * math.tanh(v) for v in values]

    @staticmethod
    def _band_means(values: Sequence[float]) -> list[float]:
        return [sum(values[i : i + 8]) / 8.0 for i in range(0, 64, 8)]

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-3.0 * (float(x) - 0.5)))

    @staticmethod
    def _scale_signed(value: float) -> float:
        return 0.5 + 0.5 * math.tanh(float(value))

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))


_instance: SubstratePolicyHead | None = None


def get_substrate_policy_head() -> SubstratePolicyHead:
    global _instance
    if _instance is None:
        _instance = SubstratePolicyHead()
    return _instance


__all__ = [
    "POLICY_KEYS",
    "SubstratePolicyInput",
    "SubstratePolicyVector",
    "SubstratePolicyHead",
    "get_substrate_policy_head",
]

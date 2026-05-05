"""General resource and homeostasis assessment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .parsed_state import ParsedState

ResourceKind = Literal[
    "health",
    "energy",
    "nutrition",
    "time",
    "tokens",
    "memory",
    "vram",
    "disk",
    "api_quota",
    "money",
    "attention",
    "test_health",
    "trust",
    "unknown",
]


@dataclass
class Resource:
    name: str
    kind: ResourceKind
    value: float
    max_value: float | None = None
    normalized: float | None = None
    critical_low: float | None = None
    critical_high: float | None = None
    trend_per_step: float = 0.0
    volatility: float = 0.0

    def __post_init__(self) -> None:
        if self.normalized is None and self.max_value not in (None, 0):
            self.normalized = max(0.0, min(1.0, self.value / float(self.max_value)))


@dataclass
class HomeostaticAssessment:
    stability_score: float
    critical_resources: list[str]
    deteriorating_resources: list[str]
    recommended_goal: str | None
    action_biases: dict[str, float] = field(default_factory=dict)


class Homeostasis:
    def extract(self, parsed_state: ParsedState) -> list[Resource]:
        resources: list[Resource] = []
        for name, res in parsed_state.resources.items():
            resources.append(
                Resource(
                    name=name,
                    kind=res.kind if res.kind in ResourceKind.__args__ else "unknown",  # type: ignore[attr-defined]
                    value=float(res.value),
                    max_value=res.max_value,
                    normalized=res.normalized,
                    critical_low=res.critical_below,
                    critical_high=res.critical_above,
                    trend_per_step=res.trend,
                )
            )
        return resources

    def assess(self, resources: list[Resource]) -> HomeostaticAssessment:
        critical: list[str] = []
        deteriorating: list[str] = []
        scores: list[float] = []
        for res in resources:
            normalized = res.normalized if res.normalized is not None else res.value
            scores.append(max(0.0, min(1.0, normalized)))
            if res.critical_low is not None and normalized <= res.critical_low:
                critical.append(res.name)
            if res.critical_high is not None and normalized >= res.critical_high:
                critical.append(res.name)
            if res.trend_per_step < -0.01:
                deteriorating.append(res.name)
        stability = sum(scores) / len(scores) if scores else 1.0
        goal = "stabilize_resource" if critical else ("gather_resource_information" if not resources else None)
        biases = {"STABILIZE_RESOURCE": 0.95} if critical else {"MAKE_PROGRESS": 0.45}
        return HomeostaticAssessment(
            stability_score=max(0.0, min(1.0, stability)),
            critical_resources=critical,
            deteriorating_resources=deteriorating,
            recommended_goal=goal,
            action_biases=biases,
        )


__all__ = ["ResourceKind", "Resource", "HomeostaticAssessment", "Homeostasis"]

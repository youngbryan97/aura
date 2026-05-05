"""Model-tier routing for embodied runs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CognitionRequest:
    kind: str
    urgency: float
    risk: float
    uncertainty: float
    token_budget: int
    context: dict


@dataclass
class CognitionRoute:
    model_tier: str
    max_tokens: int
    temperature: float
    reason: str


class CognitionRouter:
    def route(self, request: CognitionRequest) -> CognitionRoute:
        if request.risk >= 0.85 or request.uncertainty >= 0.8:
            return CognitionRoute("cortex", min(request.token_budget, 2048), 0.2, "high risk or uncertainty")
        if request.urgency >= 0.8 and request.risk < 0.4:
            return CognitionRoute("reflex", min(request.token_budget, 128), 0.0, "urgent low-risk reflex")
        if request.kind in {"postmortem", "rule_extraction", "self_modification"}:
            return CognitionRoute("solver", min(request.token_budget, 4096), 0.1, "deep deliberation requested")
        return CognitionRoute("brainstem", min(request.token_budget, 512), 0.1, "routine embodied selection")


__all__ = ["CognitionRequest", "CognitionRoute", "CognitionRouter"]

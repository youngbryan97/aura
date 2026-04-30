"""Small society of specialist lanes with critic scoring."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Specialist:
    name: str
    domains: tuple[str, ...]
    adapter: str = ""
    toolset: tuple[str, ...] = ()


@dataclass(frozen=True)
class DelegationDecision:
    specialist: Specialist
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"specialist": self.specialist.__dict__, "score": self.score, "reasons": list(self.reasons)}


class SpecialistSocietyRouter:
    DEFAULTS = (
        Specialist("coder", ("code", "bug", "test", "repo", "patch"), adapter="coder", toolset=("shell", "pytest", "git")),
        Specialist("researcher", ("paper", "source", "evidence", "benchmark", "study"), adapter="researcher", toolset=("web", "memory", "summarizer")),
        Specialist("operator", ("deploy", "monitor", "incident", "process", "release"), adapter="operator", toolset=("shell", "logs", "health")),
    )

    def __init__(self, specialists: tuple[Specialist, ...] = DEFAULTS) -> None:
        self.specialists = specialists

    def route(self, task: str, substrate_policy: dict[str, float] | None = None) -> DelegationDecision:
        policy = substrate_policy or {}
        task_terms = set(re.findall(r"[a-zA-Z_]{3,}", task.lower()))
        scored: list[tuple[float, Specialist, list[str]]] = []
        for specialist in self.specialists:
            overlap = task_terms & set(specialist.domains)
            score = len(overlap) + 0.1
            if specialist.name == "coder":
                score *= 0.8 + float(policy.get("repair_priority", 0.5))
            if specialist.name == "researcher":
                score *= 0.8 + float(policy.get("exploration_budget", 0.5))
            if specialist.name == "operator":
                score *= 0.8 + float(policy.get("risk_threshold", 0.5))
            scored.append((score, specialist, sorted(overlap)))
        score, specialist, reasons = max(scored, key=lambda item: item[0])
        return DelegationDecision(specialist, float(score), tuple(reasons or ["default_prior"]))


__all__ = ["Specialist", "DelegationDecision", "SpecialistSocietyRouter"]

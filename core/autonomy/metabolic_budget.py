"""Bounded growth pressure for Aura autonomy.

Homeostasis keeps Aura viable; this scheduler adds a small, measured pressure
toward verified improvement when the system is healthy.  It reallocates effort
away from exploration during instability and toward repair when tests or health
signals degrade.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class MetabolicState:
    stability: float = 1.0
    resource_headroom: float = 1.0
    novelty_budget: float = 0.5
    benchmark_gap: float = 0.0
    diminishing_returns: float = 1.0
    tests_passing: bool = True
    external_usefulness: float = 0.5
    user_service_pressure: float = 0.1


@dataclass(frozen=True)
class BudgetAllocation:
    stability: float
    service: float
    improvement: float
    speculative: float
    repair: float
    growth_drive: float
    mode: str
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "stability": round(self.stability, 4),
            "service": round(self.service, 4),
            "improvement": round(self.improvement, 4),
            "speculative": round(self.speculative, 4),
            "repair": round(self.repair, 4),
            "growth_drive": round(self.growth_drive, 4),
            "mode": self.mode,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class CuriosityResearchQuestion:
    question_id: str
    anomaly_key: str
    prompt: str
    priority: float
    evidence_count: int
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "anomaly_key": self.anomaly_key,
            "prompt": self.prompt,
            "priority": round(self.priority, 4),
            "evidence_count": self.evidence_count,
            "generated_at": self.generated_at,
        }


class MetabolicBudgetScheduler:
    """Adaptive budget scheduler for service, repair, and improvement."""

    BASE = {
        "stability": 0.85,
        "service": 0.08,
        "improvement": 0.05,
        "speculative": 0.02,
        "repair": 0.0,
    }

    def compute_growth_drive(self, state: MetabolicState) -> float:
        if not state.tests_passing:
            return 0.0
        stability_gate = self._clamp(state.stability)
        resource_gate = self._clamp(state.resource_headroom)
        novelty = self._clamp(state.novelty_budget)
        benchmark_gap = self._clamp(state.benchmark_gap)
        diminishing = self._clamp(state.diminishing_returns)
        usefulness = 0.5 + (0.5 * self._clamp(state.external_usefulness))
        return self._clamp(stability_gate * resource_gate * novelty * benchmark_gap * diminishing * usefulness)

    def allocate(self, state: MetabolicState) -> BudgetAllocation:
        growth = self.compute_growth_drive(state)
        if not state.tests_passing or state.stability < 0.55:
            repair = 0.35 if not state.tests_passing else 0.20
            stability = min(0.92, self.BASE["stability"] + repair * 0.5)
            service = min(0.12, self.BASE["service"] + state.user_service_pressure * 0.05)
            improvement = 0.0
            speculative = 0.0
            return self._normalize(stability, service, improvement, speculative, repair, growth, "repair")

        service = self.BASE["service"] + self._clamp(state.user_service_pressure) * 0.08
        improvement = self.BASE["improvement"] + growth * 0.10
        speculative = self.BASE["speculative"] + max(0.0, growth - 0.55) * 0.03
        stability = max(0.72, self.BASE["stability"] - growth * 0.08)
        return self._normalize(stability, service, improvement, speculative, 0.0, growth, "growth" if growth > 0.05 else "stable")

    def anomaly_to_curiosity(
        self,
        anomalies: Iterable[dict[str, Any]],
        *,
        state: MetabolicState,
    ) -> list[CuriosityResearchQuestion]:
        growth = self.compute_growth_drive(state)
        questions: list[CuriosityResearchQuestion] = []
        for anomaly in anomalies:
            key = str(anomaly.get("key") or anomaly.get("fingerprint") or anomaly.get("type") or "unknown")
            count = int(anomaly.get("count", anomaly.get("evidence_count", 1)) or 1)
            if count < 2:
                continue
            priority = self._clamp((math.log1p(count) / 4.0) + growth * 0.4)
            questions.append(
                CuriosityResearchQuestion(
                    question_id=f"curiosity_{abs(hash((key, count))) & 0xffffffff:08x}",
                    anomaly_key=key,
                    prompt=(
                        "Characterize recurring anomaly "
                        f"{key!r}, identify causal boundaries, generate a small reproduction, "
                        "and extract one reusable reliability principle."
                    ),
                    priority=priority,
                    evidence_count=count,
                )
            )
        return sorted(questions, key=lambda q: q.priority, reverse=True)

    def _normalize(
        self,
        stability: float,
        service: float,
        improvement: float,
        speculative: float,
        repair: float,
        growth_drive: float,
        mode: str,
    ) -> BudgetAllocation:
        total = max(1e-9, stability + service + improvement + speculative + repair)
        return BudgetAllocation(
            stability=stability / total,
            service=service / total,
            improvement=improvement / total,
            speculative=speculative / total,
            repair=repair / total,
            growth_drive=growth_drive,
            mode=mode,
        )

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))


_instance: MetabolicBudgetScheduler | None = None


def get_metabolic_budget_scheduler() -> MetabolicBudgetScheduler:
    global _instance
    if _instance is None:
        _instance = MetabolicBudgetScheduler()
    return _instance


__all__ = [
    "MetabolicState",
    "BudgetAllocation",
    "CuriosityResearchQuestion",
    "MetabolicBudgetScheduler",
    "get_metabolic_budget_scheduler",
]

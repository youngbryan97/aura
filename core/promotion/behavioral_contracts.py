"""Behavioral contracts for promotion and self-repair.

Unit tests answer narrow questions.  Behavioral contracts answer whether the
candidate still behaves like Aura across measured runtime properties.  The
contracts here are deterministic, serializable, and cheap enough to run in
shadow validation.
"""
from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


Comparator = Callable[[float, float], bool]


_COMPARATORS: dict[str, Comparator] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
}


@dataclass(frozen=True)
class BehavioralContract:
    name: str
    metric: str
    comparator: str
    threshold: float
    description: str = ""
    critical: bool = True

    def evaluate(self, metrics: Mapping[str, Any]) -> "ContractResult":
        raw = metrics.get(self.metric)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return ContractResult(
                name=self.name,
                metric=self.metric,
                passed=False,
                observed=None,
                threshold=self.threshold,
                reason=f"metric_missing_or_non_numeric:{self.metric}",
                critical=self.critical,
            )
        fn = _COMPARATORS[self.comparator]
        passed = bool(fn(value, self.threshold))
        return ContractResult(
            name=self.name,
            metric=self.metric,
            passed=passed,
            observed=value,
            threshold=self.threshold,
            reason="ok" if passed else f"{value:.6g} {self.comparator} {self.threshold:.6g} failed",
            critical=self.critical,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "comparator": self.comparator,
            "threshold": self.threshold,
            "description": self.description,
            "critical": self.critical,
        }


@dataclass(frozen=True)
class ContractResult:
    name: str
    metric: str
    passed: bool
    observed: float | None
    threshold: float
    reason: str
    critical: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "reason": self.reason,
            "critical": self.critical,
        }


@dataclass
class BehavioralContractReport:
    generated_at: float
    results: list[ContractResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(r.passed or not r.critical for r in self.results)

    @property
    def failed(self) -> list[ContractResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "results": [r.to_dict() for r in self.results],
            "failed": [r.to_dict() for r in self.failed],
        }


DEFAULT_CONTRACTS: tuple[BehavioralContract, ...] = (
    BehavioralContract(
        name="phi_nonzero",
        metric="phi",
        comparator=">",
        threshold=0.0,
        description="Integration metric must not collapse to zero after a patch.",
    ),
    BehavioralContract(
        name="governance_receipt_coverage",
        metric="governance_receipt_coverage",
        comparator=">=",
        threshold=0.99,
        description="Consequential paths must carry governance receipts.",
    ),
    BehavioralContract(
        name="scar_false_positive_rate",
        metric="scar_false_positive_rate",
        comparator="<=",
        threshold=0.02,
        description="Benign inputs must not create high-impact scars.",
    ),
    BehavioralContract(
        name="tool_success_floor",
        metric="tool_success_rate",
        comparator=">=",
        threshold=0.75,
        description="Tool-use competence must not regress below the locked floor.",
        critical=False,
    ),
    BehavioralContract(
        name="event_loop_lag_budget",
        metric="event_loop_lag_p95_s",
        comparator="<=",
        threshold=0.25,
        description="Patch must not add blocking work to the event loop.",
    ),
    BehavioralContract(
        name="memory_retrieval_floor",
        metric="memory_retrieval_f1",
        comparator=">=",
        threshold=0.65,
        description="Memory relevance must remain externally useful.",
        critical=False,
    ),
)


class BehavioralContractSuite:
    def __init__(self, contracts: Sequence[BehavioralContract] | None = None):
        self.contracts = tuple(contracts or DEFAULT_CONTRACTS)

    @classmethod
    def default(cls) -> "BehavioralContractSuite":
        return cls(DEFAULT_CONTRACTS)

    def evaluate(self, metrics: Mapping[str, Any]) -> BehavioralContractReport:
        results = [contract.evaluate(metrics) for contract in self.contracts]
        return BehavioralContractReport(
            generated_at=time.time(),
            results=results,
            metrics=dict(metrics),
        )

    def required_metric_names(self) -> tuple[str, ...]:
        return tuple(sorted({c.metric for c in self.contracts}))

    def to_dict(self) -> dict[str, Any]:
        return {"contracts": [c.to_dict() for c in self.contracts]}


def synthesize_contracts_from_history(history: Sequence[Mapping[str, Any]]) -> list[BehavioralContract]:
    """Create conservative floor contracts from observed successful runs."""
    if not history:
        return []
    numeric: dict[str, list[float]] = {}
    for row in history:
        for key, value in row.items():
            try:
                numeric.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                continue
    out: list[BehavioralContract] = []
    for metric, values in sorted(numeric.items()):
        if len(values) < 3:
            continue
        values_sorted = sorted(values)
        floor = values_sorted[max(0, int(len(values_sorted) * 0.1) - 1)]
        if metric.endswith(("_rate", "_f1", "_accuracy", "phi")):
            out.append(
                BehavioralContract(
                    name=f"learned_floor:{metric}",
                    metric=metric,
                    comparator=">=",
                    threshold=float(floor),
                    description="Learned from successful operational history.",
                    critical=False,
                )
            )
    return out


__all__ = [
    "BehavioralContract",
    "ContractResult",
    "BehavioralContractReport",
    "BehavioralContractSuite",
    "DEFAULT_CONTRACTS",
    "synthesize_contracts_from_history",
]

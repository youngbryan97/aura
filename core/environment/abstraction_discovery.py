"""Autonomous abstraction discovery from repeated environment patterns."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from .command import ActionIntent
from .outcome_attribution import OutcomeAssessment
from .parsed_state import ParsedState


@dataclass(frozen=True)
class EmergentAbstraction:
    abstraction_id: str
    label: str
    predicate: dict[str, Any]
    evidence_count: int
    confidence: float
    transfer_hint: str
    examples: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AbstractionDiscoveryEngine:
    """Creates planner-usable categories when repeated patterns recur."""

    def __init__(self, min_evidence: int = 3) -> None:
        self.min_evidence = max(2, int(min_evidence))
        self._buckets: dict[str, list[str]] = {}
        self.abstractions: dict[str, EmergentAbstraction] = {}

    def observe_transition(
        self,
        *,
        environment_id: str,
        context_id: str,
        action: ActionIntent,
        outcome: OutcomeAssessment,
        observed_events: list[str],
        parsed_after: ParsedState,
    ) -> list[EmergentAbstraction]:
        signatures = self._signatures(environment_id, action, outcome, observed_events, parsed_after)
        created: list[EmergentAbstraction] = []
        for label, predicate in signatures:
            key = hashlib.sha256(repr((label, sorted(predicate.items()))).encode("utf-8")).hexdigest()[:16]
            examples = self._buckets.setdefault(key, [])
            examples.append(f"{environment_id}:{context_id}:{action.name}:{','.join(observed_events[:3])}")
            examples[:] = examples[-25:]
            if len(examples) >= self.min_evidence and key not in self.abstractions:
                abstraction = EmergentAbstraction(
                    abstraction_id=f"abs_{key}",
                    label=label,
                    predicate=predicate,
                    evidence_count=len(examples),
                    confidence=round(min(0.92, 0.4 + 0.12 * len(examples)), 3),
                    transfer_hint=self._transfer_hint(label),
                    examples=tuple(examples[-5:]),
                )
                self.abstractions[key] = abstraction
                created.append(abstraction)
        return created

    def _signatures(
        self,
        environment_id: str,
        action: ActionIntent,
        outcome: OutcomeAssessment,
        observed_events: list[str],
        parsed_after: ParsedState,
    ) -> list[tuple[str, dict[str, Any]]]:
        signatures: list[tuple[str, dict[str, Any]]] = []
        if outcome.success_score < 0.35 and "unknown" in action.tags:
            signatures.append((
                "unknown-asset-direct-interaction-risk",
                {"action_tags": ("unknown",), "outcome": "low_success", "domain_family": environment_id.split(":", 1)[0]},
            ))
        if outcome.is_death or any("death" in event.lower() or "fatal" in event.lower() for event in observed_events):
            signatures.append((
                "irreversible-failure-after-local-action",
                {"action": action.name, "terminal": True, "resource_state": sorted(parsed_after.resources.keys())},
            ))
        high_uncertainty = [name for name, value in parsed_after.uncertainty.items() if float(value) >= 0.7]
        if high_uncertainty:
            signatures.append((
                "high-uncertainty-state-needs-information",
                {"uncertainty_keys": tuple(sorted(high_uncertainty)), "recommended_mode": "observe_first"},
            ))
        if len(parsed_after.hazards) >= 2:
            signatures.append((
                "compound-hazard-local-policy",
                {"hazard_count": len(parsed_after.hazards), "recommended_mode": "stabilize_or_retreat"},
            ))
        return signatures

    @staticmethod
    def _transfer_hint(label: str) -> str:
        if "unknown-asset" in label:
            return "Gather cheap evidence before direct use of uncertain objects, tools, links, files, or commands."
        if "irreversible" in label:
            return "Add a reversible checkpoint before repeating actions that can end the episode or corrupt state."
        if "uncertainty" in label:
            return "Route to observation/search/inspection before executing effectful actions."
        return "Prefer policy changes that name the abstract predicate rather than an environment-specific object."


__all__ = ["AbstractionDiscoveryEngine", "EmergentAbstraction"]

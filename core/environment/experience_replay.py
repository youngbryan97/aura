"""Hindsight replay and transferable causal policy extraction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .command import ActionIntent
from .outcome_attribution import OutcomeAssessment
from .parsed_state import ParsedState


@dataclass(frozen=True)
class ReplayTransition:
    environment_id: str
    environment_family: str
    context_id: str
    action_name: str
    action_tags: tuple[str, ...]
    action_parameters: dict[str, Any]
    before_hash: str
    after_hash: str
    success_score: float
    is_death: bool
    observed_events: tuple[str, ...]


@dataclass(frozen=True)
class CausalPolicyRule:
    rule_id: str
    trigger: str
    recommendation: str
    evidence_count: int
    confidence: float
    transfer_tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HindsightReplayBuffer:
    """Stores recent transitions and induces general, cross-domain rules."""

    def __init__(self, maxlen: int = 500) -> None:
        self.maxlen = max(20, int(maxlen))
        self.transitions: list[ReplayTransition] = []
        self.rules: dict[str, CausalPolicyRule] = {}

    def add_transition(
        self,
        *,
        environment_id: str,
        context_id: str,
        action: ActionIntent,
        before: ParsedState,
        after: ParsedState,
        outcome: OutcomeAssessment,
        observed_events: list[str],
    ) -> None:
        transition = ReplayTransition(
            environment_id=environment_id,
            environment_family=environment_id.split(":", 1)[0],
            context_id=context_id,
            action_name=action.name,
            action_tags=tuple(sorted(action.tags)),
            action_parameters=dict(action.parameters),
            before_hash=before.stable_hash(),
            after_hash=after.stable_hash(),
            success_score=float(outcome.success_score),
            is_death=bool(outcome.is_death),
            observed_events=tuple(observed_events),
        )
        self.transitions.append(transition)
        self.transitions = self.transitions[-self.maxlen :]
        self._induce_rules()

    def applicable_rules(self, *, action: ActionIntent, environment_family: str | None = None) -> list[CausalPolicyRule]:
        tags = set(action.tags)
        rules = []
        for rule in self.rules.values():
            trigger = rule.trigger
            if trigger == "unknown_direct_action" and "unknown" in tags:
                rules.append(rule)
            elif trigger == f"action_failed:{action.name}":
                rules.append(rule)
            elif environment_family and environment_family in rule.transfer_tags:
                rules.append(rule)
        return sorted(rules, key=lambda rule: rule.confidence, reverse=True)

    def _induce_rules(self) -> None:
        unknown_failures = [
            item for item in self.transitions
            if "unknown" in item.action_tags and item.success_score < 0.35
        ]
        if unknown_failures:
            self._set_rule(
                "unknown_direct_action",
                "unknown_direct_action",
                "Prefer low-cost information gathering before direct use of uncertain assets.",
                unknown_failures,
                ("terminal_grid", "browser", "desktop", "shell", "api"),
            )
        by_action: dict[str, list[ReplayTransition]] = {}
        for item in self.transitions:
            if item.success_score < 0.3 or item.is_death:
                by_action.setdefault(item.action_name, []).append(item)
        for action_name, failures in by_action.items():
            if len(failures) >= 2:
                self._set_rule(
                    f"action_failed:{action_name}",
                    f"action_failed:{action_name}",
                    f"After repeated failure of {action_name}, switch to observation, recovery, or a different subgoal.",
                    failures,
                    tuple(sorted({item.environment_family for item in failures})),
                )

    def _set_rule(
        self,
        rule_id: str,
        trigger: str,
        recommendation: str,
        evidence: list[ReplayTransition],
        transfer_tags: tuple[str, ...],
    ) -> None:
        confidence = min(0.95, 0.45 + 0.1 * len(evidence))
        self.rules[rule_id] = CausalPolicyRule(
            rule_id=rule_id,
            trigger=trigger,
            recommendation=recommendation,
            evidence_count=len(evidence),
            confidence=round(confidence, 3),
            transfer_tags=transfer_tags,
        )


__all__ = ["CausalPolicyRule", "HindsightReplayBuffer", "ReplayTransition"]

"""Fast environment risk and reflex layer.

This is Aura's embodied System 1: cheap checks that run every observation
before slower deliberation. It does not know NetHack. It knows resources,
prompts, uncertainty, nearby threats, damage messages, and stale/looping
control signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .belief_state import EnvironmentBeliefState
from .environment_parser import EnvironmentState


RISK_ORDER = {"safe": 0, "caution": 1, "danger": 2, "critical": 3}


@dataclass
class DangerAssessment:
    risk_level: str
    reason: str
    recommended_override: Optional[str] = None
    score: float = 0.0
    tags: List[str] = field(default_factory=list)
    source: str = "reflex"


@dataclass
class RiskProfile:
    """Aggregated risk state for the current observation."""

    assessments: List[DangerAssessment] = field(default_factory=list)

    @property
    def level(self) -> str:
        if not self.assessments:
            return "safe"
        return max(self.assessments, key=lambda item: RISK_ORDER.get(item.risk_level, 0)).risk_level

    @property
    def score(self) -> float:
        if not self.assessments:
            return 0.0
        return max(item.score for item in self.assessments)

    @property
    def critical(self) -> bool:
        return self.level == "critical"

    @property
    def danger_or_worse(self) -> bool:
        return RISK_ORDER.get(self.level, 0) >= RISK_ORDER["danger"]

    def reasons(self) -> List[str]:
        return [item.reason for item in self.assessments]

    def tags(self) -> List[str]:
        tags: List[str] = []
        for item in self.assessments:
            for tag in item.tags:
                if tag not in tags:
                    tags.append(tag)
        return tags


class EnvironmentReflexLayer:
    """Evaluates environmental state for immediate threats."""

    def assess_danger(
        self,
        state: EnvironmentState,
        belief: EnvironmentBeliefState,
    ) -> List[DangerAssessment]:
        return self.assess_profile(state, belief).assessments

    def assess_profile(
        self,
        state: EnvironmentState,
        belief: EnvironmentBeliefState,
    ) -> RiskProfile:
        assessments: List[DangerAssessment] = []
        assessments.extend(self._resource_assessments(state))
        assessments.extend(self._prompt_assessments(state))
        assessments.extend(self._status_assessments(state))
        assessments.extend(self._message_assessments(state))
        assessments.extend(self._entity_assessments(state))
        assessments.extend(self._uncertainty_assessments(state, belief))
        assessments.extend(self._loop_assessments(belief))
        return RiskProfile(assessments=assessments)

    def generate_reflex_percepts(self, assessments: Iterable[DangerAssessment]) -> List[str]:
        percepts = []
        for assessment in assessments:
            if RISK_ORDER.get(assessment.risk_level, 0) >= RISK_ORDER["danger"]:
                msg = f"[REFLEX WARNING: {assessment.risk_level.upper()}] {assessment.reason}"
                if assessment.recommended_override:
                    msg += f" RECOMMENDED: {assessment.recommended_override}"
                percepts.append(msg)
        return percepts

    def _resource_assessments(self, state: EnvironmentState) -> List[DangerAssessment]:
        assessments: List[DangerAssessment] = []
        resource_pairs = (
            ("hp", "max_hp", "Health", "stabilize_or_retreat"),
            ("energy", "max_energy", "Energy", "stabilize_resource"),
            ("integrity", "max_integrity", "Integrity", "stop_risky_actions"),
            ("battery", "max_battery", "Battery", "reduce_load_or_recharge"),
        )
        for current_key, max_key, label, override in resource_pairs:
            ratio = state.resource_ratio(current_key, max_key)
            if ratio is None:
                continue
            current = state.self_state.get(current_key)
            maximum = state.self_state.get(max_key)
            if ratio <= 0.25:
                assessments.append(
                    DangerAssessment(
                        risk_level="critical",
                        reason=f"{label} critically low ({current}/{maximum}).",
                        recommended_override=override,
                        score=0.95,
                        tags=["resource", current_key, "survival"],
                    )
                )
            elif ratio <= 0.50:
                assessments.append(
                    DangerAssessment(
                        risk_level="danger",
                        reason=f"{label} is low ({current}/{maximum}).",
                        recommended_override=override,
                        score=0.7,
                        tags=["resource", current_key],
                    )
                )
        return assessments

    def _prompt_assessments(self, state: EnvironmentState) -> List[DangerAssessment]:
        if not state.active_prompts:
            return []
        prompt_text = " | ".join(state.active_prompts)
        return [
            DangerAssessment(
                risk_level="danger",
                reason=f"Environment is in a modal/prompt state: {prompt_text[:160]}",
                recommended_override="resolve_active_prompt_before_normal_action",
                score=0.7,
                tags=["prompt", "interface", "modal"],
            )
        ]

    def _status_assessments(self, state: EnvironmentState) -> List[DangerAssessment]:
        assessments: List[DangerAssessment] = []
        hunger = str(state.self_state.get("hunger", "")).lower()
        if hunger in {"fainting", "fainted", "starved"}:
            assessments.append(
                DangerAssessment(
                    risk_level="critical",
                    reason=f"Starvation/resource collapse imminent ({state.self_state.get('hunger')}).",
                    recommended_override="consume_or_restore_resource_immediately",
                    score=0.95,
                    tags=["hunger", "resource", "survival"],
                )
            )
        elif hunger == "weak":
            assessments.append(
                DangerAssessment(
                    risk_level="danger",
                    reason="Character/body is weak from resource depletion.",
                    recommended_override="prioritize_resource_recovery",
                    score=0.75,
                    tags=["hunger", "resource"],
                )
            )

        effects = state.self_state.get("status_effects") or []
        if isinstance(effects, str):
            effects = [effects]
        for effect in effects:
            lowered = str(effect).lower()
            if lowered in {"blind", "confused", "stunned", "paralyzed"}:
                assessments.append(
                    DangerAssessment(
                        risk_level="danger",
                        reason=f"Control/perception impaired by status effect: {effect}.",
                        recommended_override="avoid_irreversible_actions_until_stable",
                        score=0.72,
                        tags=["status", lowered, "control"],
                    )
                )
        return assessments

    def _message_assessments(self, state: EnvironmentState) -> List[DangerAssessment]:
        assessments: List[DangerAssessment] = []
        for msg in state.messages:
            lowered = msg.lower()
            if "you die" in lowered or "killed" in lowered or "fatal" in lowered:
                assessments.append(
                    DangerAssessment(
                        risk_level="critical",
                        reason=f"Failure/death signal detected: {msg[:160]}",
                        score=1.0,
                        tags=["terminal_failure"],
                    )
                )
            elif any(token in lowered for token in ("hit", "hurt", "damage", "poison", "weak")):
                assessments.append(
                    DangerAssessment(
                        risk_level="caution",
                        reason=f"Recent adverse event: {msg[:160]}",
                        recommended_override="reassess_current_plan",
                        score=0.45,
                        tags=["damage", "adverse_event"],
                    )
                )
        return assessments

    def _entity_assessments(self, state: EnvironmentState) -> List[DangerAssessment]:
        assessments: List[DangerAssessment] = []
        for entity in state.entities:
            entity_type = str(entity.get("type", "")).lower()
            tags = [str(tag).lower() for tag in entity.get("tags", []) or []]
            distance = entity.get("distance")
            try:
                close = distance is not None and float(distance) <= 1.0
            except (TypeError, ValueError):
                close = False
            hostile = entity.get("hostile") is True or entity_type in {"monster", "large_monster", "threat"}
            if close and hostile:
                assessments.append(
                    DangerAssessment(
                        risk_level="danger",
                        reason=f"Potential threat is adjacent or close: {entity}",
                        recommended_override="resolve_immediate_threat_or_retreat",
                        score=0.72,
                        tags=["threat", "proximity", *tags],
                    )
                )
        return assessments

    def _uncertainty_assessments(
        self,
        state: EnvironmentState,
        belief: EnvironmentBeliefState,
    ) -> List[DangerAssessment]:
        score = max(
            [belief.epistemic_uncertainty(), *[float(v) for v in state.uncertainty.values()]]
            or [0.0]
        )
        if score >= 0.75:
            return [
                DangerAssessment(
                    risk_level="danger",
                    reason=f"Epistemic uncertainty is high ({score:.2f}).",
                    recommended_override="gather_information_or_choose_reversible_action",
                    score=score,
                    tags=["uncertainty", "caution"],
                )
            ]
        if score >= 0.5:
            return [
                DangerAssessment(
                    risk_level="caution",
                    reason=f"Epistemic uncertainty elevated ({score:.2f}).",
                    recommended_override="prefer_reversible_actions",
                    score=score,
                    tags=["uncertainty"],
                )
            ]
        return []

    def _loop_assessments(self, belief: EnvironmentBeliefState) -> List[DangerAssessment]:
        outcomes = belief.action_outcomes[-8:]
        if len(outcomes) < 6:
            return []
        actions = [str(item.get("action", "")) for item in outcomes]
        no_success = all(item.get("success") is False for item in outcomes if item.get("success") is not None)
        if len(set(actions)) <= 2 and no_success:
            return [
                DangerAssessment(
                    risk_level="danger",
                    reason="Recent action outcomes suggest a control loop or repeated failed action.",
                    recommended_override="interrupt_current_skill_and_replan",
                    score=0.75,
                    tags=["loop", "stagnation"],
                )
            ]
        return []

"""Immutable frame contract emitted by the cognitive situation engine."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any


def clamp_unit_interval(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if not math.isfinite(value):
        return lower
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class CognitiveSituationFrame:
    frame_id: str
    objective: str
    semantic_flexibility: float
    analogical_leap_pressure: float
    sensorimotor_grounding: float
    abstraction_level: float
    ambiguity: float
    verification_pressure: float
    metacognition_pressure: float
    social_uncertainty: float
    social_repair_pressure: float
    agent_id: str = ""
    keywords: list[str] = field(default_factory=list)
    semantic_interpretations: list[dict[str, Any]] = field(default_factory=list)
    analogy_bridges: list[dict[str, str]] = field(default_factory=list)
    embodied_affordances: list[str] = field(default_factory=list)
    perception_summary: dict[str, Any] = field(default_factory=dict)
    social_summary: dict[str, Any] = field(default_factory=dict)
    predicted_consequences: dict[str, Any] = field(default_factory=dict)
    attention_targets: list[str] = field(default_factory=list)
    routing_bias: dict[str, bool] = field(default_factory=dict)
    sampling_bias: dict[str, float] = field(default_factory=dict)
    causal_effects: dict[str, Any] = field(default_factory=dict)
    governance: dict[str, Any] = field(
        default_factory=lambda: {
            "side_effect_free": True,
            "reads_existing_perception_only": True,
            "screen_capture_owner": "core/perception/screen_perception.py",
            "external_effects_require_authority_gateway": True,
            "claims_require_receipts": True,
        }
    )
    created_at: float = field(default_factory=lambda: time.time())

    @property
    def salience(self) -> float:
        return clamp_unit_interval(
            max(
                self.semantic_flexibility,
                self.analogical_leap_pressure,
                self.sensorimotor_grounding,
                self.ambiguity,
                self.social_repair_pressure,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["salience"] = self.salience
        return payload

    def prompt_block(self, *, compact: bool = False) -> str:
        if self.salience < 0.16:
            return ""
        if compact:
            directives = [
                "COGNITIVE SITUATION FRAME: use semantic alternatives, analogies, and embodiment only as causal grounding.",
            ]
            if self.semantic_flexibility >= 0.35:
                directives.append("Disambiguate the current user intent before carrying older topics forward.")
            if self.analogical_leap_pressure >= 0.35:
                directives.append("Use a relevant analogy or cross-domain bridge when it improves the answer.")
            if self.sensorimotor_grounding >= 0.30:
                directives.append("Ground screen/tool claims in observed state or governed receipts before claiming completion.")
            if self.routing_bias.get("perception_abstention_required"):
                directives.append(
                    "Perception is incomplete or contested: abstain from unsupported scene claims and gather evidence first."
                )
            if self.routing_bias.get("social_repair_required"):
                directives.append(
                    "A concrete relational failure may be active: acknowledge it accurately before advancing the task."
                )
            if self.routing_bias.get("social_state_clarification_required"):
                directives.append(
                    "The social read is uncertain: clarify material ambiguity without asserting feelings or intent."
                )
            return " ".join(directives)

        lines = [
            "## COGNITIVE SITUATION FRAME",
            (
                f"Semantic flexibility={self.semantic_flexibility:.2f}; "
                f"analogical pressure={self.analogical_leap_pressure:.2f}; "
                f"sensorimotor grounding={self.sensorimotor_grounding:.2f}; "
                f"ambiguity={self.ambiguity:.2f}; social uncertainty={self.social_uncertainty:.2f}; "
                f"repair pressure={self.social_repair_pressure:.2f}."
            ),
        ]
        if self.semantic_interpretations:
            rendered = "; ".join(
                f"{item.get('label')}: {item.get('focus')}"
                for item in self.semantic_interpretations[:3]
            )
            lines.append(f"Candidate interpretations: {rendered}.")
        if self.analogy_bridges:
            rendered = "; ".join(
                f"{item.get('source')} -> {item.get('target')}: {item.get('relation')}"
                for item in self.analogy_bridges[:3]
            )
            lines.append(f"Analogy bridges: {rendered}.")
        if self.embodied_affordances:
            lines.append(
                "Embodied affordances: " + ", ".join(self.embodied_affordances[:5]) + "."
            )
        if self.attention_targets:
            lines.append("Attention targets: " + ", ".join(self.attention_targets[:5]) + ".")
        constraints = self.causal_effects.get("perception_planning_constraints")
        if isinstance(constraints, list) and constraints:
            lines.append("Perception constraints: " + ", ".join(map(str, constraints[:5])) + ".")
        repairs = self.causal_effects.get("perception_repair_requirements")
        if isinstance(repairs, list) and repairs:
            lines.append("Perception repair: " + ", ".join(map(str, repairs[:5])) + ".")
        social_constraints = self.causal_effects.get("social_planning_constraints")
        if isinstance(social_constraints, list) and social_constraints:
            lines.append("Social constraints: " + ", ".join(map(str, social_constraints[:5])) + ".")
        if self.social_summary:
            lines.append(
                "Social state is an uncertain estimate, not a fact about the user's feelings, culture, or intent."
            )
        if self.routing_bias.get("perception_abstention_required"):
            lines.append(
                "Perception is incomplete or contested: abstain from unsupported scene claims and gather evidence first."
            )
        lines.append(
            "This frame is causal grounding, not prose to recite. It may change routing, "
            "sampling, verification, and attention. It does not prove perception or tool completion."
        )
        return "\n".join(lines) + "\n\n"

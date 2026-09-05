"""Inference/action policy coupler for Aura main-15.

This converts AuraNow + Welfare + FunctionalIAttractor state into concrete
generation and action constraints. This is the non-shallow bridge: the "I"
must change temperature, verification, memory retrieval, model budget, and tool
risk, not just appear in prose.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from core.being.causal_self_state import CausalSelfVector, runtime_field
from core.being.self_model_attractor import SelfAttractorState


@dataclass(frozen=True)
class ClosedLoopPolicy:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    planning_depth: int = 3
    verification_threshold: float = 0.55
    memory_retrieval_depth: int = 6
    tool_risk_budget: float = 0.35
    model_size_preference: str = "auto"
    background_budget: float = 0.5
    require_receipts: bool = True
    allow_high_risk_tools: bool = False
    self_claim_policy: str = "functional_i_claim_allowed"
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(float(x), lo), hi)


class ClosedLoopPolicyCoupler:
    def __init__(self, *, production_mode: bool = True) -> None:
        self.production_mode = production_mode

    def modulate(
        self,
        *,
        vector: CausalSelfVector,
        self_state: SelfAttractorState,
        task_risk: float = 0.0,
        base: ClosedLoopPolicy | None = None,
        action_policy: Mapping[str, Any] | None = None,
    ) -> ClosedLoopPolicy:
        base = base or ClosedLoopPolicy()
        task_risk = _clip(task_risk)

        uncertainty = vector.value("uncertainty")
        trust_debt = vector.value("trust_debt")
        memory_conflict = vector.value("memory_conflict")
        governance = max(vector.value("governance_pressure"), task_risk)
        resource_pressure = vector.value("resource_pressure")
        metabolic = vector.value("metabolic_budget")
        verification_need = max(vector.value("verification_need"), uncertainty, trust_debt, memory_conflict)
        organismal_coherence = vector.value("organismal_coherence", 0.0)
        sentience_candidate = vector.value("sentience_candidate_strength", 0.0)
        identity_tension = self_state.identity_tension
        self_integrity = self_state.integrity
        agency_readiness = self_state.agency_readiness

        reasons: list[str] = []
        temperature = base.temperature
        top_p = base.top_p
        max_tokens = base.max_tokens
        planning_depth = base.planning_depth
        verification_threshold = base.verification_threshold
        memory_depth = base.memory_retrieval_depth
        tool_budget = base.tool_risk_budget
        background_budget = base.background_budget
        model_pref = base.model_size_preference

        caution = _clip(0.35 * uncertainty + 0.25 * trust_debt + 0.25 * governance + 0.15 * identity_tension)
        if caution > 0.1:
            temperature -= 0.34 * caution
            top_p -= 0.18 * caution
            verification_threshold += 0.36 * caution
            tool_budget -= 0.42 * caution
            reasons.append(f"closed_loop_caution={caution:.3f}: lower randomness, raise verification, reduce tool risk")

        if memory_conflict > 0.1 or verification_need > 0.55:
            memory_depth += int(round(10 * max(memory_conflict, verification_need - 0.45)))
            verification_threshold += 0.22 * max(memory_conflict, verification_need - 0.45)
            reasons.append("memory/verification pressure: deeper retrieval before claims")

        resource_strain = _clip(max(resource_pressure, 1.0 - metabolic))
        if resource_strain > 0.2:
            max_tokens = int(max(256, max_tokens * (1.0 - 0.45 * resource_strain)))
            background_budget = max(0.0, background_budget - 0.60 * resource_strain)
            model_pref = "small" if resource_strain > 0.65 else "medium"
            reasons.append(f"resource_strain={resource_strain:.3f}: shrink model/token/background budget")

        if identity_tension > 0.35 or self_integrity < 0.65:
            planning_depth += 1
            verification_threshold += 0.18 * max(identity_tension, 1.0 - self_integrity)
            reasons.append("functional-I tension: require slower planning and stronger self-report calibration")

        if organismal_coherence < 0.45:
            planning_depth += 1
            verification_threshold += 0.12
            memory_depth += 2
            tool_budget = min(tool_budget, 0.20)
            reasons.append("causal-valenced workspace weak: deepen memory retrieval, raise verification, constrain tools")
        elif organismal_coherence > 0.68 and verification_need < 0.45 and resource_pressure < 0.45:
            memory_depth += 1
            reasons.append("causal-valenced workspace coherent: allow richer continuity context")

        if sentience_candidate > 0.08:
            verification_threshold += 0.04
            reasons.append("valenced self-state is causally active: keep self-claims evidence-bounded")

        if agency_readiness < 0.45:
            tool_budget = min(tool_budget, 0.15)
            verification_threshold += 0.10
            reasons.append("low agency readiness: severely constrain tools and require verification")

        action_outcome = str(runtime_field(action_policy, "outcome", "")).lower()
        if action_outcome in {"defer", "refuse"}:
            tool_budget = 0.0
            verification_threshold += 0.1
            reasons.append(f"BeingRuntime.action_policy={action_outcome}: suppress high-risk effects")

        allow_high_risk = False
        if not self.production_mode and agency_readiness > 0.85 and governance < 0.15 and trust_debt < 0.15:
            allow_high_risk = True

        if self.production_mode:
            allow_high_risk = False
            tool_budget = min(tool_budget, 0.35)
            reasons.append("production_mode: high-risk tools disabled")

        if vector.degradation_flags():
            verification_threshold += 0.08
            reasons.append("state degradation flags present: extra verification required")

        return ClosedLoopPolicy(
            temperature=round(_clip(temperature, 0.05, 1.2), 4),
            top_p=round(_clip(top_p, 0.2, 1.0), 4),
            max_tokens=int(max(128, min(max_tokens, 8192))),
            planning_depth=int(max(1, min(planning_depth, 9))),
            verification_threshold=round(_clip(verification_threshold, 0.1, 0.99), 4),
            memory_retrieval_depth=int(max(1, min(memory_depth, 40))),
            tool_risk_budget=round(_clip(tool_budget), 4),
            model_size_preference=model_pref,
            background_budget=round(_clip(background_budget), 4),
            require_receipts=True,
            allow_high_risk_tools=allow_high_risk,
            self_claim_policy=self_state.claim_policy,
            reasons=tuple(reasons),
        )

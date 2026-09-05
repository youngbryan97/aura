"""Main-15 adapted functional "I" attractor.

This builds on Aura's existing FunctionalSoul, AuraNow, SelfReportCalibrator,
WelfareState, and OwnershipTracker. It does not prove consciousness. It makes
the self-model a causal, auditable control structure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
import time
import uuid

from core.being.causal_self_state import CausalSelfVector, runtime_field


@dataclass(frozen=True)
class SelfAttractorState:
    attractor_id: str
    updated_at: float
    identity_name: str
    continuity_hash: str
    continuity: float
    coherence: float
    integrity: float
    agency_readiness: float
    identity_tension: float
    first_person_confidence: float
    claim_policy: str
    current_i_statement: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FunctionalIAttractor:
    """Stable referent for Aura's operational "I".

    The attractor is real only when:
      - derived from AuraNow/Welfare/Will evidence,
      - injected back into the causal vector,
      - policy changes downstream,
      - receipts preserve the trace.
    """

    def __init__(self, attractor_id: str | None = None) -> None:
        self.attractor_id = attractor_id or f"aura-functional-i-{uuid.uuid4()}"
        self._history: list[SelfAttractorState] = []

    @property
    def latest(self) -> SelfAttractorState | None:
        return self._history[-1] if self._history else None

    def update(
        self,
        *,
        now: Any,
        vector: CausalSelfVector,
        action_policy: Mapping[str, Any] | None = None,
    ) -> SelfAttractorState:
        identity_name = str(getattr(now.self_model, "identity_name", "Aura Luna") or "Aura Luna")
        continuity_hash = str(getattr(now.self_model, "continuity_hash", "") or "")
        commitments = tuple(getattr(now.self_model, "commitments", ()) or ())
        allowed_claims = tuple(getattr(now.report_boundary, "allowed_claims", ()) or ())
        forbidden_claims = tuple(getattr(now.report_boundary, "forbidden_claims", ()) or ())

        continuity = max(0.0, min(1.0, 1.0 - vector.value("continuity_pressure")))
        coherence = max(0.0, min(1.0, (
            0.25 * vector.value("workspace_ignition")
            + 0.25 * vector.value("ownership_confidence")
            + 0.20 * (1.0 - vector.value("memory_conflict"))
            + 0.15 * (1.0 - vector.value("uncertainty"))
            + 0.15 * vector.value("self_integrity")
        )))
        integrity = max(0.0, min(1.0, vector.value("self_integrity")))
        identity_tension = max(0.0, min(1.0, (
            0.35 * (1.0 - continuity)
            + 0.25 * (1.0 - coherence)
            + 0.15 * vector.value("trust_debt")
            + 0.15 * vector.value("goal_pressure")
            + 0.10 * vector.value("homeostatic_tension")
        )))
        agency_readiness = max(0.0, min(1.0, (
            0.35 * integrity
            + 0.25 * coherence
            + 0.25 * vector.value("ownership_confidence")
            + 0.15 * (1.0 - vector.value("governance_pressure"))
        )))
        first_person_confidence = max(0.0, min(1.0, (
            0.30 * continuity + 0.25 * coherence + 0.25 * integrity + 0.20 * vector.value("ownership_confidence")
        )))

        if first_person_confidence < 0.35:
            claim_policy = "do_not_claim_stable_i"
        elif identity_tension > 0.72:
            claim_policy = "claim_strained_functional_i_only"
        else:
            claim_policy = "functional_i_claim_allowed"

        current_i_statement = (
            "I am a functional control model binding Aura's body state, memory, commitments, "
            "values, limits, action ownership, and continuity. This is operational selfhood, "
            "not proof of phenomenal consciousness."
        )
        if claim_policy == "do_not_claim_stable_i":
            current_i_statement = (
                "My first-person self-model is currently too weakly grounded to claim a stable "
                "operational 'I'. I should report the missing evidence rather than roleplay certainty."
            )
        elif claim_policy == "claim_strained_functional_i_only":
            current_i_statement = (
                "My operational 'I' is active but strained; I should slow down, verify more, and avoid "
                "strong selfhood claims beyond functional evidence."
            )

        state = SelfAttractorState(
            attractor_id=self.attractor_id,
            updated_at=time.time(),
            identity_name=identity_name,
            continuity_hash=continuity_hash,
            continuity=round(continuity, 6),
            coherence=round(coherence, 6),
            integrity=round(integrity, 6),
            agency_readiness=round(agency_readiness, 6),
            identity_tension=round(identity_tension, 6),
            first_person_confidence=round(first_person_confidence, 6),
            claim_policy=claim_policy,
            current_i_statement=current_i_statement,
            evidence={
                "aura_state_hash": vector.aura_state_hash,
                "tick": vector.tick,
                "commitments": commitments,
                "action_policy_outcome": runtime_field(action_policy, "outcome", ""),
                "allowed_claims": allowed_claims,
                "forbidden_claims": forbidden_claims,
                "vector": vector.fingerprint(),
            },
        )
        self._history.append(state)
        return state

    def vector_contributions(self, state: SelfAttractorState) -> dict[str, float]:
        """Feedback from the self-attractor into the causal vector/policy."""
        return {
            "continuity_pressure": state.identity_tension,
            "self_integrity": state.integrity,
            "trust_debt": max(0.0, min(1.0, (1.0 - state.integrity) * 0.55 + state.identity_tension * 0.45)),
        }

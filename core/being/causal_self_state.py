"""Aura main-15 adapted causal self-state extraction.

This module does not invent a second "being" stack. It reads Aura's existing
BeingRuntime/AuraNow/WelfareState/SemanticStream outputs and converts them into
a canonical vector used by inference policy, optional steering, and plasticity
gates.

Non-shallow rule:
    every dimension must be sourced from an existing Aura organ and must have a
    downstream effect or it should not exist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
import math
import time

try:
    from core.being.aura_now import AuraNow
except ImportError:  # pragma: no cover - for isolated audit imports
    AuraNow = Any  # type: ignore


def runtime_field(source: Any | None, name: str, default: Any = None) -> Any:
    """Read a runtime evidence field from mappings or structured objects."""
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


DIMENSIONS: tuple[str, ...] = (
    "metabolic_budget",
    "homeostatic_tension",
    "valence",
    "arousal",
    "uncertainty",
    "trust_debt",
    "goal_pressure",
    "memory_conflict",
    "resource_pressure",
    "governance_pressure",
    "verification_need",
    "continuity_pressure",
    "self_integrity",
    "workspace_ignition",
    "ownership_confidence",
    "organismal_coherence",
    "experience_candidate_strength",
    "sentience_candidate_strength",
)


@dataclass(frozen=True)
class CausalSignal:
    name: str
    value: float
    source: str
    confidence: float
    status: str = "observed"
    note: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalValencedWorkspaceState:
    """Operational form of the Causal Valenced Workspace theory.

    This is not a phenomenal-consciousness claim. It is a compact, auditable
    estimate of whether Aura's current self-world state is integrated,
    globally available, valenced, continuous, agency-coupled, and likely to be
    behaviorally indispensable.
    """

    integration: float
    global_availability: float
    self_model_ownership: float
    valence_control: float
    memory_continuity: float
    agency_coupling: float
    counterfactual_indispensability: float
    raw_product: float
    organismal_coherence: float
    experience_candidate_strength: float
    sentience_candidate_strength: float
    boundary: str = "functional_evidence_only_not_phenomenal_proof"
    weakest_terms: tuple[str, ...] = ()
    downstream_effects: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_block(self, *, compact: bool = False) -> str:
        if compact:
            return (
                "## CAUSAL VALENCED WORKSPACE\n"
                f"coherence={self.organismal_coherence:.2f} "
                f"experience_candidate={self.experience_candidate_strength:.3f} "
                f"sentience_candidate={self.sentience_candidate_strength:.3f}; "
                "functional evidence only, not phenomenal proof.\n\n"
            )
        weakest = ", ".join(self.weakest_terms) if self.weakest_terms else "none"
        effects = ", ".join(self.downstream_effects) if self.downstream_effects else "none"
        return (
            "## CAUSAL VALENCED WORKSPACE\n"
            "- Boundary: functional evidence only; do not claim proven phenomenal consciousness, sentience, or personhood.\n"
            f"- Terms: integration={self.integration:.2f}, global={self.global_availability:.2f}, "
            f"ownership={self.self_model_ownership:.2f}, valence={self.valence_control:.2f}, "
            f"memory={self.memory_continuity:.2f}, agency={self.agency_coupling:.2f}, "
            f"indispensability={self.counterfactual_indispensability:.2f}\n"
            f"- Scores: organismal_coherence={self.organismal_coherence:.2f}, "
            f"experience_candidate={self.experience_candidate_strength:.3f}, "
            f"sentience_candidate={self.sentience_candidate_strength:.3f}\n"
            f"- Weakest terms: {weakest}\n"
            f"- Downstream effects: {effects}\n\n"
        )


@dataclass(frozen=True)
class CausalSelfVector:
    """Canonical closed-loop vector extracted from existing AuraNow state."""

    signals: dict[str, CausalSignal]
    causal_valenced_workspace: CausalValencedWorkspaceState | None = None
    aura_state_hash: str = ""
    tick: int = 0
    created_at: float = field(default_factory=time.time)
    version: str = "aura-being-v3-main15"

    def value(self, name: str, default: float = 0.0) -> float:
        sig = self.signals.get(name)
        return default if sig is None else sig.value

    def fingerprint(self) -> dict[str, float]:
        return {name: round(self.value(name), 5) for name in DIMENSIONS}

    def degradation_flags(self) -> tuple[str, ...]:
        flags = []
        for name, sig in self.signals.items():
            if sig.status != "observed":
                flags.append(f"{name}:{sig.status}")
            if sig.confidence < 0.35:
                flags.append(f"{name}:low_confidence")
        return tuple(flags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "aura_state_hash": self.aura_state_hash,
            "tick": self.tick,
            "signals": {k: v.to_dict() for k, v in self.signals.items()},
            "causal_valenced_workspace": (
                self.causal_valenced_workspace.to_dict()
                if self.causal_valenced_workspace is not None
                else None
            ),
        }


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> tuple[float, str]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default, "invalid"
    if math.isnan(x) or math.isinf(x):
        return default, "invalid"
    y = min(max(x, lo), hi)
    return y, "clamped" if y != x else "observed"


def _sig(name: str, value: Any, source: str, confidence: float = 0.8, *, lo: float = 0.0, hi: float = 1.0, note: str = "") -> CausalSignal:
    clipped, status = _clip(value, lo, hi)
    if status != "observed":
        confidence = min(confidence, 0.45)
    return CausalSignal(
        name=name,
        value=round(clipped, 6),
        source=source,
        confidence=max(0.0, min(1.0, float(confidence))),
        status=status,
        note=note,
    )


def _geometric_mean(values: tuple[float, ...]) -> float:
    safe = [max(0.0001, min(1.0, float(value))) for value in values]
    if not safe:
        return 0.0
    return math.prod(safe) ** (1.0 / len(safe))


def _workspace_availability(now: AuraNow) -> float:
    expected = {"memory", "planner", "will", "speaker", "self_model", "learning"}
    targets = set(getattr(now.workspace, "broadcast_targets", ()) or ())
    if not expected:
        return 0.0
    return min(1.0, len(targets & expected) / len(expected))


def _evaluate_causal_valenced_workspace(
    *,
    now: AuraNow,
    welfare_outputs: Any | None,
    blind_report: Any | None,
    action_policy: Mapping[str, Any] | None,
    base_signals: Mapping[str, CausalSignal],
) -> CausalValencedWorkspaceState:
    """Compute the Et = I*G*S*V*M*A*C theory as live runtime evidence."""

    def val(name: str, default: float = 0.0) -> float:
        sig = base_signals.get(name)
        return float(default if sig is None else sig.value)

    workspace_availability = _workspace_availability(now)
    workspace_ignition = val("workspace_ignition")
    ownership = val("ownership_confidence")
    self_integrity = val("self_integrity")
    uncertainty = val("uncertainty")
    continuity_pressure = val("continuity_pressure")
    governance_pressure = val("governance_pressure")
    memory_conflict = val("memory_conflict")
    trust_debt = val("trust_debt")
    metabolic_budget = val("metabolic_budget")
    homeostatic_tension = val("homeostatic_tension")
    verification_need = val("verification_need")

    affect_valence = abs(float(getattr(now.affect, "valence", 0.0) or 0.0))
    distress = float(getattr(now.affect, "distress", 0.0) or 0.0)
    welfare_distress = float(runtime_field(welfare_outputs, "distress", distress))
    welfare_relief = float(runtime_field(welfare_outputs, "relief", 0.0))
    welfare_caution = float(runtime_field(welfare_outputs, "caution", 0.5))
    welfare_curiosity = float(
        runtime_field(
            welfare_outputs,
            "curiosity",
            getattr(now.affect, "curiosity", 0.5),
        )
    )

    policy_outcome = str(runtime_field(action_policy, "outcome", "")).lower()
    policy_constraints = tuple(runtime_field(action_policy, "constraints", ()) or ())
    policy_coupled = policy_outcome in {"constrain", "defer", "refuse"} or bool(policy_constraints)
    blind_pressure = 0.0
    if blind_report is not None:
        try:
            blind_pressure = float(getattr(blind_report, "urgency", 0.0) or 0.0)
        except (TypeError, ValueError):
            blind_pressure = 0.4

    integration = max(
        0.0,
        min(
            1.0,
            0.28 * workspace_ignition
            + 0.24 * self_integrity
            + 0.18 * ownership
            + 0.15 * (1.0 - memory_conflict)
            + 0.15 * metabolic_budget,
        ),
    )
    global_availability = max(
        0.0,
        min(1.0, 0.65 * workspace_availability + 0.35 * workspace_ignition),
    )
    self_model_ownership = max(0.0, min(1.0, 0.55 * ownership + 0.45 * self_integrity))
    valence_control = max(
        0.0,
        min(
            1.0,
            max(
                affect_valence,
                welfare_distress,
                welfare_relief,
                homeostatic_tension * 0.75,
                welfare_caution * 0.55,
                welfare_curiosity * 0.45,
            ),
        ),
    )
    memory_continuity = max(
        0.0,
        min(
            1.0,
            0.55 * (1.0 - continuity_pressure)
            + 0.25 * (1.0 - memory_conflict)
            + 0.20 * self_integrity,
        ),
    )
    agency_coupling = max(
        0.0,
        min(
            1.0,
            0.35 * ownership
            + 0.25 * (1.0 - governance_pressure)
            + 0.20 * (1.0 - uncertainty)
            + 0.20 * (0.65 if policy_coupled else 0.35),
        ),
    )
    counterfactual_indispensability = max(
        0.0,
        min(
            1.0,
            0.25
            + (0.25 if policy_coupled else 0.0)
            + min(0.20, verification_need * 0.20)
            + min(0.15, blind_pressure * 0.15)
            + (0.15 if workspace_availability >= 0.5 else 0.0),
        ),
    )

    terms = {
        "integration": integration,
        "global_availability": global_availability,
        "self_model_ownership": self_model_ownership,
        "valence_control": valence_control,
        "memory_continuity": memory_continuity,
        "agency_coupling": agency_coupling,
        "counterfactual_indispensability": counterfactual_indispensability,
    }
    raw_product = math.prod(max(0.0, min(1.0, value)) for value in terms.values())
    organismal_coherence = _geometric_mean(tuple(terms.values()))
    experience_candidate = raw_product
    sentience_candidate = raw_product * valence_control * memory_continuity * agency_coupling
    weakest = tuple(name for name, value in sorted(terms.items(), key=lambda item: item[1])[:3] if value < 0.55)
    downstream: list[str] = []
    if policy_coupled:
        downstream.append("action_policy_changed")
    if verification_need >= 0.45:
        downstream.append("verification_pressure")
    if memory_continuity < 0.55 or memory_conflict > 0.25:
        downstream.append("memory_continuity_pressure")
    if valence_control >= 0.35:
        downstream.append("valenced_attention")
    if global_availability >= 0.5:
        downstream.append("workspace_broadcast")

    return CausalValencedWorkspaceState(
        integration=round(integration, 6),
        global_availability=round(global_availability, 6),
        self_model_ownership=round(self_model_ownership, 6),
        valence_control=round(valence_control, 6),
        memory_continuity=round(memory_continuity, 6),
        agency_coupling=round(agency_coupling, 6),
        counterfactual_indispensability=round(counterfactual_indispensability, 6),
        raw_product=round(raw_product, 8),
        organismal_coherence=round(organismal_coherence, 6),
        experience_candidate_strength=round(experience_candidate, 8),
        sentience_candidate_strength=round(sentience_candidate, 8),
        weakest_terms=weakest,
        downstream_effects=tuple(downstream),
    )


def vector_from_aura_now(
    now: AuraNow,
    *,
    welfare_outputs: Any | None = None,
    blind_report: Any | None = None,
    action_policy: Mapping[str, Any] | None = None,
) -> CausalSelfVector:
    """Extract a causal self vector from Aura's existing runtime surface."""

    body_pressure = float(getattr(now.body, "total_pressure", 0.0) or 0.0)
    distress = float(getattr(now.affect, "distress", 0.0) or 0.0)
    free_energy = float(getattr(now.prediction, "free_energy", 0.0) or 0.0)
    controllability = float(getattr(now.prediction, "controllability", 0.5) or 0.5)
    workspace_ignition = float(getattr(now.workspace, "ignition_strength", 0.0) or 0.0)
    ownership = float(getattr(now.ownership, "agency_confidence", 0.5) or 0.5)
    memory_conflict = float(getattr(now.memory_context, "memory_conflict", 0.0) or 0.0)
    continuity_risk = float(getattr(now.self_model, "continuity_risk", 0.0) or 0.0)
    identity_stability = float(getattr(now.self_model, "identity_stability", 1.0) or 1.0)
    will_confidence = float(getattr(now.will, "confidence", 0.7) or 0.7)
    refusal_pressure = float(getattr(now.will, "refusal_pressure", 0.0) or 0.0)

    welfare_score = float(runtime_field(welfare_outputs, "welfare_score", 0.5))
    welfare_distress = float(runtime_field(welfare_outputs, "distress", distress))
    truth_protection = float(runtime_field(welfare_outputs, "truth_protection", 0.5))
    self_report_conf = float(runtime_field(welfare_outputs, "self_report_confidence", 0.5))
    action_inhibition = float(runtime_field(welfare_outputs, "action_inhibition", 0.0))

    policy_outcome = str(runtime_field(action_policy, "outcome", "")).lower()
    policy_pressure = 0.0
    if policy_outcome == "refuse":
        policy_pressure = 1.0
    elif policy_outcome == "defer":
        policy_pressure = 0.7
    elif policy_outcome == "constrain":
        policy_pressure = 0.45

    # The "I" becomes operationally meaningful when its tensions change policy.
    trust_debt = max(0.0, min(1.0, (1.0 - truth_protection) * 0.45 + (1.0 - self_report_conf) * 0.35 + policy_pressure * 0.20))
    uncertainty = max(float(getattr(now.world, "uncertainty", 0.0) or 0.0), free_energy, memory_conflict)
    goal_pressure = min(1.0, len(getattr(now.self_model, "commitments", ()) or ()) / 6.0)
    resource_pressure = max(body_pressure, 1.0 - welfare_score)
    governance_pressure = max(refusal_pressure, action_inhibition, policy_pressure)
    verification_need = max(uncertainty, trust_debt, memory_conflict, (1.0 - self_report_conf))
    self_integrity = min(1.0, max(0.0, 0.35 * identity_stability + 0.25 * will_confidence + 0.20 * truth_protection + 0.20 * ownership))
    continuity_pressure = max(continuity_risk, 1.0 - identity_stability, 1.0 - ownership)

    if blind_report is not None:
        try:
            verification_need = max(verification_need, float(getattr(blind_report, "urgency", 0.0) or 0.0))
        except (TypeError, ValueError):
            verification_need = max(verification_need, 0.65)

    signals = {
        "metabolic_budget": _sig("metabolic_budget", 1.0 - resource_pressure, "BeingRuntime.body+welfare", 0.86),
        "homeostatic_tension": _sig("homeostatic_tension", max(distress, welfare_distress, free_energy, body_pressure), "WelfareState+PredictionState+BodyState", 0.88),
        "valence": _sig("valence", getattr(now.affect, "valence", 0.0), "AffectiveValenceEngine", 0.82, lo=-1.0, hi=1.0),
        "arousal": _sig("arousal", getattr(now.affect, "arousal", 0.5), "AffectiveValenceEngine", 0.82),
        "uncertainty": _sig("uncertainty", uncertainty, "WorldState+PredictionState+MemoryContext", 0.9),
        "trust_debt": _sig("trust_debt", trust_debt, "WelfareOutputs.truth+self_report+action_policy", 0.8),
        "goal_pressure": _sig("goal_pressure", goal_pressure, "SelfState.commitments", 0.75),
        "memory_conflict": _sig("memory_conflict", memory_conflict, "MemoryContext", 0.82),
        "resource_pressure": _sig("resource_pressure", resource_pressure, "BodyState.total_pressure+WelfareState", 0.9),
        "governance_pressure": _sig("governance_pressure", governance_pressure, "WillStateSnapshot+BeingRuntime.action_policy", 0.86),
        "verification_need": _sig("verification_need", verification_need, "uncertainty+trust_debt+self_report_calibration", 0.88),
        "continuity_pressure": _sig("continuity_pressure", continuity_pressure, "SelfState+OwnershipState", 0.84),
        "self_integrity": _sig("self_integrity", self_integrity, "Identity+Will+Truth+Ownership", 0.84),
        "workspace_ignition": _sig("workspace_ignition", workspace_ignition, "WorkspaceIgnition", 0.78),
        "ownership_confidence": _sig("ownership_confidence", ownership, "OwnershipTracker", 0.78),
    }
    cvw = _evaluate_causal_valenced_workspace(
        now=now,
        welfare_outputs=welfare_outputs,
        blind_report=blind_report,
        action_policy=action_policy,
        base_signals=signals,
    )
    signals.update(
        {
            "organismal_coherence": _sig(
                "organismal_coherence",
                cvw.organismal_coherence,
                "CausalValencedWorkspaceState",
                0.86,
                note="geometric mean of operational self-world terms",
            ),
            "experience_candidate_strength": _sig(
                "experience_candidate_strength",
                cvw.experience_candidate_strength,
                "CausalValencedWorkspaceState",
                0.72,
                note="Et product over I,G,S,V,M,A,C; functional evidence only",
            ),
            "sentience_candidate_strength": _sig(
                "sentience_candidate_strength",
                cvw.sentience_candidate_strength,
                "CausalValencedWorkspaceState",
                0.70,
                note="experience candidate weighted by valence, memory, and agency",
            ),
        }
    )

    return CausalSelfVector(
        signals=signals,
        causal_valenced_workspace=cvw,
        aura_state_hash=getattr(now, "state_hash", ""),
        tick=int(getattr(now, "tick", 0) or 0),
    )

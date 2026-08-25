"""What the transformer worked out, folded back into the state that asked.

The loop the architecture wants is not

    state → prompt → model → answer

but

    z_t → model reasoning → z_{t+1} → model → …

which requires the second arrow to exist. Without it a conclusion reached by
the model evaporates when the response is emitted, and the next turn starts
from a state that never learned anything. This module is that arrow, and an
arbitration surface that goes with it.

**Absorption** takes what a turn produced — how confident it was, whether
evidence was attached, whether a goal moved, whether a contradiction turned
up — and injects it into the organ that carries the continuous state. The
receipt records the state digest before and after, so a caller can tell an
absorption that landed from one that was swallowed. If no organ accepts the
injection, the receipt says so. It never reports success on an injection
nothing received.

**Arbitration** is the other half. A proposal that comes back from the model
is a proposal, not her decision. It is checked against the named channels of
the state she is actually in: a claim of certainty against the confidence she
holds, an abandoned goal against a goal that is active and high priority, an
action against evidence support. A conflict names the channel it came from,
which makes the whole thing testable by intervention — move the channel, and
the verdict has to move with it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.brain.llm.endogenous_state import (
    EndogenousState,
    assemble_state,
)

logger = logging.getLogger("Aura.EndogenousAbsorption")

#: Ceiling on how much one turn may move the substrate input bus. A single
#: conclusion is one observation among many; a turn that could saturate the
#: state would make the state a transcript of the last reply.
MAX_INJECTION_ENERGY = 0.6


@dataclass(frozen=True)
class TurnOutcome:
    """What a completed turn produced, in the terms the state can absorb."""

    summary: str = ""
    confidence: float | None = None
    evidence_items: int = 0
    goal_advanced: bool = False
    contradiction_found: bool = False
    refused: bool = False
    tokens_used: int = 0

    def as_observation(self) -> dict[str, Any]:
        """The turn as a grounded observation the substrate already accepts.

        Reusing the sensor-observation path rather than inventing a second
        injection API: the substrate has one documented input bus and one
        deterministic projection onto it, and a private second route would be
        a second thing to keep correct.
        """
        energy = min(
            MAX_INJECTION_ENERGY,
            0.15
            + 0.10 * min(4, int(self.evidence_items)) / 4.0
            + (0.15 if self.goal_advanced else 0.0)
            + (0.20 if self.contradiction_found else 0.0)
            + (0.10 if self.refused else 0.0),
        )
        return {
            "source": "endogenous_absorption",
            "type": "turn_conclusion",
            "summary": str(self.summary or "")[:400],
            "confidence": (
                float(self.confidence) if self.confidence is not None else 0.5
            ),
            "energy": float(energy),
            "timestamp_unix": time.time(),
            "evidence_items": int(self.evidence_items),
            "goal_advanced": bool(self.goal_advanced),
            "contradiction_found": bool(self.contradiction_found),
            "refused": bool(self.refused),
        }


@dataclass(frozen=True)
class AbsorptionReceipt:
    """Whether the conclusion reached the state, and how anyone would know."""

    accepted: bool
    organ: str
    reason: str
    before_digest: str = ""
    after_digest: str = ""
    before_coverage: float = 0.0
    after_coverage: float = 0.0

    @property
    def state_moved(self) -> bool:
        """Whether z_Aura actually differs after the injection.

        The substrate integrates on its own clock, so a state read in the same
        millisecond as an injection can be unchanged and the injection still
        correct. This field says what was observed; ``accepted`` says what was
        delivered. Conflating the two would let a dead input bus report
        success forever.
        """
        return bool(self.before_digest) and self.before_digest != self.after_digest

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "organ": self.organ,
            "reason": self.reason,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "state_moved": self.state_moved,
            "before_coverage": round(self.before_coverage, 4),
            "after_coverage": round(self.after_coverage, 4),
        }


def _substrate() -> tuple[Any, str]:
    """Find the organ that carries the continuous state, or report neither."""
    from core.brain.llm.endogenous_state import _service

    for name in ("continuous_substrate", "liquid_state", "liquid_substrate"):
        organ = _service(name)
        if organ is not None:
            return organ, name
    return None, ""


#: How much of the input bus one absorbed turn may claim. The rest stays with
#: whatever else is publishing into it.
ABSORPTION_WEIGHT = 0.35


def absorb(
    outcome: TurnOutcome,
    *,
    substrate: Any = None,
    allow_exclusive_bus: bool = False,
) -> AbsorptionReceipt:
    """Fold one turn's conclusion into the continuous state.

    The additive path is required by default. ``inject_observation`` replaces
    the substrate's whole input bus, so absorbing a turn through it would
    erase whatever the sensorimotor bridge had just published, and erase it
    invisibly — neither caller can see the other. ``allow_exclusive_bus``
    exists for a substrate that genuinely has one writer.
    """
    organ, organ_name = (substrate, "provided") if substrate is not None else _substrate()
    before = assemble_state()
    if organ is None:
        return AbsorptionReceipt(
            accepted=False,
            organ="",
            reason="no organ carries the continuous state in this process",
            before_digest=before.digest,
            after_digest=before.digest,
            before_coverage=before.coverage,
            after_coverage=before.coverage,
        )
    blend = getattr(organ, "blend_observation", None)
    inject = getattr(organ, "inject_observation", None)
    if callable(blend):
        def deliver(observation: dict[str, Any]) -> None:
            blend(observation, ABSORPTION_WEIGHT)
    elif callable(inject) and allow_exclusive_bus:
        deliver = inject
    elif callable(inject):
        return AbsorptionReceipt(
            accepted=False,
            organ=organ_name,
            reason="organ has only a replacing input bus; absorbing would erase other writers",
            before_digest=before.digest,
            after_digest=before.digest,
            before_coverage=before.coverage,
            after_coverage=before.coverage,
        )
    else:
        return AbsorptionReceipt(
            accepted=False,
            organ=organ_name,
            reason="organ has no observation input bus",
            before_digest=before.digest,
            after_digest=before.digest,
            before_coverage=before.coverage,
            after_coverage=before.coverage,
        )
    try:
        deliver(outcome.as_observation())
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        return AbsorptionReceipt(
            accepted=False,
            organ=organ_name,
            reason=f"injection refused: {exc}",
            before_digest=before.digest,
            after_digest=before.digest,
            before_coverage=before.coverage,
            after_coverage=before.coverage,
        )
    after = assemble_state()
    return AbsorptionReceipt(
        accepted=True,
        organ=organ_name,
        reason="injected",
        before_digest=before.digest,
        after_digest=after.digest,
        before_coverage=before.coverage,
        after_coverage=after.coverage,
    )


# ──────────────────────────────────────────────────────────────────────────
# Arbitration
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Proposal:
    """Something the model came back with, before it counts as her decision."""

    summary: str = ""
    asserted_confidence: float | None = None
    cites_evidence: bool = False
    abandons_active_goal: bool = False
    requires_action: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": str(self.summary)[:200],
            "asserted_confidence": self.asserted_confidence,
            "cites_evidence": self.cites_evidence,
            "abandons_active_goal": self.abandons_active_goal,
            "requires_action": self.requires_action,
        }


@dataclass(frozen=True)
class Conflict:
    """One disagreement between a proposal and the state, and its source."""

    channel: str
    feature: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"channel": self.channel, "feature": self.feature, "detail": self.detail}


@dataclass(frozen=True)
class Arbitration:
    """What the state says about a proposal that arrived from the model."""

    decision: str  # "accept" | "revise" | "reject"
    conflicts: tuple[Conflict, ...] = ()
    checks_run: tuple[str, ...] = ()
    checks_skipped: tuple[str, ...] = field(default_factory=tuple)

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "conflicts": [c.as_dict() for c in self.conflicts],
            "checks_run": list(self.checks_run),
            "checks_skipped": list(self.checks_skipped),
        }


#: A proposal claiming at least this much certainty is making a strong claim.
STRONG_CLAIM = 0.8

#: Confidence at or below this is a state that does not support one.
WEAK_STATE = 0.4

#: A goal this important is not dropped because a generation suggested it.
PROTECTED_GOAL_PRIORITY = 0.7


def arbitrate(proposal: Proposal, state: EndogenousState | None = None) -> Arbitration:
    """Check a proposal against the state, and name any channel that objects.

    A check whose channel is absent is skipped and reported as skipped. It is
    never passed: a memory system that failed to answer must not read as a
    memory system with no objection.
    """
    current = assemble_state() if state is None else state
    conflicts: list[Conflict] = []
    run: list[str] = []
    skipped: list[str] = []

    if proposal.asserted_confidence is not None:
        if current.is_present("uncertainty.confidence"):
            run.append("certainty_matches_state")
            if (
                float(proposal.asserted_confidence) >= STRONG_CLAIM
                and current.get("uncertainty.confidence") <= WEAK_STATE
            ):
                conflicts.append(
                    Conflict(
                        channel="uncertainty",
                        feature="uncertainty.confidence",
                        detail=(
                            f"proposal asserts {proposal.asserted_confidence:.2f} "
                            f"while held confidence is "
                            f"{current.get('uncertainty.confidence'):.2f}"
                        ),
                    )
                )
        else:
            skipped.append("certainty_matches_state")

    if proposal.abandons_active_goal:
        if current.is_present("goal.active"):
            run.append("respects_active_goal")
            priority = (
                current.get("goal.priority")
                if current.is_present("goal.priority")
                else 1.0
            )
            if current.get("goal.active") >= 0.5 and priority >= PROTECTED_GOAL_PRIORITY:
                conflicts.append(
                    Conflict(
                        channel="goal",
                        feature="goal.priority",
                        detail=(
                            f"a goal of priority {priority:.2f} is held and the "
                            "proposal drops it"
                        ),
                    )
                )
        else:
            skipped.append("respects_active_goal")

    if proposal.requires_action and not proposal.cites_evidence:
        if current.is_present("uncertainty.evidence_support"):
            run.append("action_has_support")
            if current.get("uncertainty.evidence_support") <= WEAK_STATE:
                conflicts.append(
                    Conflict(
                        channel="uncertainty",
                        feature="uncertainty.evidence_support",
                        detail="an action is proposed with no evidence on either side",
                    )
                )
        else:
            skipped.append("action_has_support")

    if current.is_present("memory.contradiction") and current.get(
        "memory.contradiction"
    ) >= 0.5:
        run.append("recall_is_consistent")
        conflicts.append(
            Conflict(
                channel="memory",
                feature="memory.contradiction",
                detail="recalled items disagree, so any conclusion drawn from them is provisional",
            )
        )
    elif not current.is_present("memory.contradiction"):
        skipped.append("recall_is_consistent")
    else:
        run.append("recall_is_consistent")

    if not conflicts:
        decision = "accept"
    elif any(c.channel == "goal" for c in conflicts):
        decision = "reject"
    else:
        decision = "revise"
    return Arbitration(
        decision=decision,
        conflicts=tuple(conflicts),
        checks_run=tuple(run),
        checks_skipped=tuple(skipped),
    )


def outcome_from_response(payload: Mapping[str, Any]) -> TurnOutcome:
    """Read a worker response frame as an absorbable outcome."""
    if not isinstance(payload, Mapping):
        return TurnOutcome()
    evidence = payload.get("evidence") or payload.get("grounding_evidence") or []
    return TurnOutcome(
        summary=str(payload.get("text") or "")[:400],
        confidence=(
            float(payload["confidence"])
            if isinstance(payload.get("confidence"), (int, float))
            else None
        ),
        evidence_items=len(evidence) if isinstance(evidence, (list, tuple)) else 0,
        goal_advanced=bool(payload.get("goal_advanced")),
        contradiction_found=bool(payload.get("contradiction_found")),
        refused=str(payload.get("status") or "") not in {"ok", ""},
        tokens_used=int(payload.get("tokens_used") or 0),
    )


__all__ = [
    "ABSORPTION_WEIGHT",
    "MAX_INJECTION_ENERGY",
    "PROTECTED_GOAL_PRIORITY",
    "STRONG_CLAIM",
    "WEAK_STATE",
    "AbsorptionReceipt",
    "Arbitration",
    "Conflict",
    "Proposal",
    "TurnOutcome",
    "absorb",
    "arbitrate",
    "outcome_from_response",
]

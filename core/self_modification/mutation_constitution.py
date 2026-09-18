"""One admission decision for every path that changes Aura's own source.

There were two, and they disagreed. `SafeSelfModification.validate_proposal`
classified the target with `classify_mutation_path` and refused a sealed
path outright. `self_code_improver.improve_function` — reached by the
`improve_own_code` skill, which declares `requires_approval = False` and
defaults `enact=True` — used its own containment check: inside the source
root, not on a substring denylist, ends in `.py`. Measured against the tier
scheme, that admitted `core/self_modification/safe_modification.py`, a
tier3_sealed path, for autonomous rewriting.

So the weaker of the two rules won, because it belonged to the path that
did not ask permission. A constitution with an alternate route around it is
not a constitution.

Both callers ask this module now. It answers with the tier, whether the
change may be enacted, and why — never with a bare boolean, because "no"
and "not without approval" and "not until the turn's inputs are known" are
three different answers and the caller has to be able to tell them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.self_modification.mutation_tiers import (
    MutationTier,
    MutationTierDecision,
    classify_mutation_path,
)

#: What a caller is allowed to do with a proposal.
ENACT = "enact"  # apply it to the file
PROPOSE = "propose"  # draft it, and stop
REFUSE = "refuse"  # not this path, not this way, not later
DEFER = "defer"  # not until something knowable becomes known


@dataclass(frozen=True)
class MutationAdmission:
    """The single answer, with its reason and its evidence obligations."""

    disposition: str
    tier: MutationTier
    reason: str
    required_gates: tuple[str, ...] = ()
    normalized_path: str = ""
    #: Evidence the caller must attach to the record it writes.
    receipt: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.disposition not in {ENACT, PROPOSE, REFUSE, DEFER}:
            raise ValueError(f"unknown mutation disposition: {self.disposition!r}")

    @property
    def may_enact(self) -> bool:
        return self.disposition == ENACT

    @property
    def may_propose(self) -> bool:
        """A refusal stops the draft too; a deferral does not."""
        return self.disposition in {ENACT, PROPOSE, DEFER}


def _receipt(
    decision: MutationTierDecision,
    *,
    disposition: str,
    owner_approved: bool,
    turn_trust: str,
) -> dict[str, Any]:
    return {
        "schema": "aura.mutation_constitution.v1",
        "path": decision.path,
        "tier": decision.tier.label,
        "tier_reason": decision.reason,
        "required_gates": list(decision.required_gates),
        "disposition": disposition,
        "owner_approved": bool(owner_approved),
        "turn_trust": turn_trust,
    }


def _mode_forbids_self_modification() -> str:
    """The reason this mode refuses source changes, or "" when it allows them.

    Fails OPEN on an unreadable mode: refusing every mutation because the
    mode module could not be imported would turn a bookkeeping fault into a
    frozen self-repair lane, and the tier gates below are still in force.
    """

    try:
        from core.runtime.mode import allows_self_modification, get_mode

        if allows_self_modification():
            return ""
        return (
            f"runtime mode {get_mode()} does not permit self-modification; "
            "the mode's capability manifest is the authority here, above the tier"
        )
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        return ""


def admit_mutation(
    target_path: str | Path,
    *,
    owner_approved: bool = False,
    turn_trust: str = "trusted",
) -> MutationAdmission:
    """Whether this source change may be enacted, drafted, deferred or refused.

    ``turn_trust`` is the verdict from `SafeSelfModification._turn_trust_verdict`:
    "trusted", "untrusted" or "unknown". Unknown defers rather than refusing,
    because the answer can become knowable on the next turn; untrusted refuses,
    because it already is known and it is no.
    """
    decision = classify_mutation_path(target_path)
    gates = tuple(decision.required_gates)

    # The runtime mode outranks the tier, because a mode that forbids
    # self-modification forbids all of it — safe mode is an emergency
    # lockdown and says so, and the two sandboxes must not touch source at
    # all. This gate consulted no mode until 2026-09-18: core/runtime/mode.py
    # declares a capability manifest per mode, its docstring says every
    # module needing to ask "am I in production?" must use its helpers, and
    # allows_self_modification() had ZERO callers in the tree. Safe mode
    # disabled autonomous behaviour in forty-one places and not in this one,
    # so an emergency lockdown still admitted a Tier 1 source rewrite.
    mode_refusal = _mode_forbids_self_modification()
    if mode_refusal:
        return MutationAdmission(
            disposition=REFUSE,
            tier=decision.tier,
            reason=mode_refusal,
            required_gates=gates,
            normalized_path=decision.path,
            receipt=_receipt(
                decision,
                disposition=REFUSE,
                owner_approved=owner_approved,
                turn_trust=turn_trust,
            ),
        )

    if decision.tier is MutationTier.SEALED:
        # Sealed outranks everything, including owner approval through this
        # route: the tier's own gates are external review, a manual patch and
        # a cold restart, none of which a running process can perform on
        # itself.
        return MutationAdmission(
            disposition=REFUSE,
            tier=decision.tier,
            reason=(
                f"{decision.path} is {decision.tier.label} ({decision.reason}); "
                "it changes by external review and manual patch, never from inside "
                "a running Aura"
            ),
            required_gates=gates,
            normalized_path=decision.path,
            receipt=_receipt(
                decision,
                disposition=REFUSE,
                owner_approved=owner_approved,
                turn_trust=turn_trust,
            ),
        )

    if turn_trust == "untrusted":
        return MutationAdmission(
            disposition=REFUSE,
            tier=decision.tier,
            reason=(
                "this turn read untrusted content, so the patch is not a trusted "
                "proposal however well it tests"
            ),
            required_gates=gates,
            normalized_path=decision.path,
            receipt=_receipt(
                decision,
                disposition=REFUSE,
                owner_approved=owner_approved,
                turn_trust=turn_trust,
            ),
        )

    if turn_trust == "unknown" and not owner_approved:
        return MutationAdmission(
            disposition=DEFER,
            tier=decision.tier,
            reason=(
                "nothing established what this turn read; the draft stands and can "
                "be enacted from a turn whose inputs are known, or with owner approval"
            ),
            required_gates=gates,
            normalized_path=decision.path,
            receipt=_receipt(
                decision,
                disposition=DEFER,
                owner_approved=owner_approved,
                turn_trust=turn_trust,
            ),
        )

    if decision.tier is MutationTier.PROPOSE_ONLY and not owner_approved:
        return MutationAdmission(
            disposition=PROPOSE,
            tier=decision.tier,
            reason=(
                f"{decision.path} is {decision.tier.label} ({decision.reason}); "
                "Aura may draft the patch, and applying it needs explicit owner approval"
            ),
            required_gates=gates,
            normalized_path=decision.path,
            receipt=_receipt(
                decision,
                disposition=PROPOSE,
                owner_approved=owner_approved,
                turn_trust=turn_trust,
            ),
        )

    return MutationAdmission(
        disposition=ENACT,
        tier=decision.tier,
        reason=f"{decision.path} is {decision.tier.label} ({decision.reason})",
        required_gates=gates,
        normalized_path=decision.path,
        receipt=_receipt(
            decision,
            disposition=ENACT,
            owner_approved=owner_approved,
            turn_trust=turn_trust,
        ),
    )


__all__ = [
    "DEFER",
    "ENACT",
    "PROPOSE",
    "REFUSE",
    "MutationAdmission",
    "admit_mutation",
]

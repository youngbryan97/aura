"""core/runtime/capability_maturity.py — registered is not the same as trusted.

Clean-room adoption of Home Assistant's integration quality scale. HA grades
every integration Bronze → Silver → Gold → Platinum against published rules, and
the grade is a first-class fact about the integration rather than a note in a
README. No upstream code is used here; the rules below are Aura's own.

The problem it solves for Aura is specific. Aura has a large capability surface
and she can act on it *autonomously* — a background initiative can reach a
connector nobody has exercised in months, with no human in the loop to notice it
misbehaving. Registration currently means "the import succeeded". It does not
mean the skill validates its inputs, bounds its timeouts, is safe to retry,
verifies its own postconditions, or reports a usable error. Those are different
claims, and collapsing them means the least mature connector in the registry is
reachable by the most consequential path in the system.

So maturity here is a **gate, not a badge**:

* an UNRATED or BRONZE capability may be used when a person asked for it and is
  present to see the result;
* autonomous, unattended, and self-initiated use requires SILVER or better;
* irreversible effects require GOLD.

The tier is derived from declared, checkable properties — not asserted. A
capability claiming GOLD without the properties that define GOLD is *demoted*,
and the demotion names the missing rules. That keeps this from becoming the
thing it is meant to prevent: a label that outruns the engineering.

This module is pure policy over declarations. It performs no I/O and imports
nothing above the runtime foundation, so the gate is available on every path
including boot and shutdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

__all__ = [
    "CapabilityMaturity",
    "MaturityTier",
    "UseContext",
    "admission_for",
    "derive_properties",
    "grade_capability",
]


class MaturityTier(IntEnum):
    """How much of the engineering behind a capability has actually been done.

    Ordered, so comparisons like ``tier >= MaturityTier.SILVER`` are the
    natural way to express a requirement.
    """

    UNRATED = 0   # Registered. Nothing else is claimed.
    BRONZE = 1    # Works, and fails in a way a caller can handle.
    SILVER = 2    # Safe to run unattended.
    GOLD = 3      # Safe to run unattended with irreversible effects.
    PLATINUM = 4  # All of the above, plus proven recovery and diagnostics.


class UseContext(IntEnum):
    """Who is asking, and who is watching.

    The same capability is a different risk depending on this, which is the
    whole reason the gate is contextual rather than a single global switch.
    """

    #: A person asked for this and is present to see what happens.
    ATTENDED = 0
    #: Triggered by a person, but they will not see the result immediately.
    DEFERRED = 1
    #: Aura decided to do this herself. Nobody is watching.
    AUTONOMOUS = 2
    #: Autonomous AND the effect cannot be undone.
    AUTONOMOUS_IRREVERSIBLE = 3


#: The properties a capability must actually have, and the tier each unlocks.
#: Each is phrased as a claim that can be checked rather than an intention.
TIER_REQUIREMENTS: dict[MaturityTier, tuple[str, ...]] = {
    MaturityTier.BRONZE: (
        # Without these, a caller cannot even tell what went wrong.
        "typed_request_schema",
        "typed_error_result",
    ),
    MaturityTier.SILVER: (
        # Without these, unattended use can hang forever, double-apply on
        # retry, or silently do nothing.
        "bounded_timeout",
        "idempotent_or_read_only",
        "postcondition_verification",
        "offline_or_failure_recovery",
    ),
    MaturityTier.GOLD: (
        # Without these, an irreversible mistake has no receipt and no undo.
        "effect_receipt",
        "reversible_or_confirmed",
        "authority_scoped",
    ),
    MaturityTier.PLATINUM: (
        "redacted_diagnostics",
        "integration_tests",
    ),
}

#: Minimum tier required to be reachable from each context.
CONTEXT_MINIMUM: dict[UseContext, MaturityTier] = {
    UseContext.ATTENDED: MaturityTier.UNRATED,
    UseContext.DEFERRED: MaturityTier.BRONZE,
    UseContext.AUTONOMOUS: MaturityTier.SILVER,
    UseContext.AUTONOMOUS_IRREVERSIBLE: MaturityTier.GOLD,
}


@dataclass(frozen=True)
class CapabilityMaturity:
    """A graded capability, with the reasons for its grade."""

    name: str
    tier: MaturityTier
    #: Properties the capability declared AND that count toward its tier.
    satisfied: tuple[str, ...] = ()
    #: What is missing for the next tier up, in order.
    missing_for_next: tuple[str, ...] = ()
    #: Rules explicitly waived, with the reason, e.g. a connector that cannot
    #: be integration-tested without a live third-party account.
    exemptions: dict[str, str] = field(default_factory=dict)
    #: Set when a capability claimed a tier it had not earned.
    demoted_from: MaturityTier | None = None

    @property
    def demoted(self) -> bool:
        return self.demoted_from is not None

    def permits(self, context: UseContext) -> bool:
        return self.tier >= CONTEXT_MINIMUM[context]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier.name.lower(),
            "tier_rank": int(self.tier),
            "satisfied": list(self.satisfied),
            "missing_for_next": list(self.missing_for_next),
            "exemptions": dict(self.exemptions),
            "demoted": self.demoted,
            "demoted_from": self.demoted_from.name.lower() if self.demoted_from else None,
            "permits": {
                context.name.lower(): self.permits(context) for context in UseContext
            },
        }


def _normalize_properties(declared: Any) -> set[str]:
    """Accept a set, list, or dict-of-flags without caring which."""
    if isinstance(declared, dict):
        return {str(k) for k, v in declared.items() if v}
    if isinstance(declared, (set, frozenset, list, tuple)):
        return {str(item) for item in declared}
    return set()


def grade_capability(
    name: str,
    declared_properties: Any = None,
    *,
    claimed_tier: MaturityTier | str | int | None = None,
    exemptions: dict[str, str] | None = None,
) -> CapabilityMaturity:
    """Derive a capability's tier from what it can actually do.

    The tier is EARNED, never taken at its word. A capability that claims GOLD
    without the properties defining GOLD is graded at what it actually
    satisfies and records the demotion, so an optimistic declaration cannot buy
    autonomous reach.

    Exemptions waive a specific rule with a stated reason — a connector that
    genuinely cannot be integration-tested without a live third-party account
    should not be barred from PLATINUM forever — but they are recorded, so an
    exemption is a visible decision rather than a silent gap.
    """
    have = _normalize_properties(declared_properties)
    waived = dict(exemptions or {})

    satisfied: list[str] = []
    earned = MaturityTier.UNRATED
    missing_for_next: tuple[str, ...] = ()

    # Tiers are cumulative: you cannot reach SILVER while failing BRONZE.
    for tier in (MaturityTier.BRONZE, MaturityTier.SILVER,
                 MaturityTier.GOLD, MaturityTier.PLATINUM):
        required = TIER_REQUIREMENTS[tier]
        missing = tuple(
            rule for rule in required
            if rule not in have and rule not in waived
        )
        if missing:
            missing_for_next = missing
            break
        satisfied.extend(rule for rule in required if rule in have)
        earned = tier

    demoted_from = None
    if claimed_tier is not None:
        claimed = _coerce_tier(claimed_tier)
        if claimed is not None and claimed > earned:
            demoted_from = claimed

    return CapabilityMaturity(
        name=str(name),
        tier=earned,
        satisfied=tuple(satisfied),
        missing_for_next=missing_for_next,
        exemptions=waived,
        demoted_from=demoted_from,
    )


def _coerce_tier(value: MaturityTier | str | int) -> MaturityTier | None:
    if isinstance(value, MaturityTier):
        return value
    if isinstance(value, int):
        try:
            return MaturityTier(value)
        # not a failure: being asked whether a number names a tier, and
        # answering that it does not, is this function's job.
        except ValueError:
            return None
    try:
        return MaturityTier[str(value).strip().upper()]
    # not a failure: the same question asked with a name instead of a number.
    except KeyError:
        return None


@dataclass(frozen=True)
class MaturityAdmission:
    """Whether this capability may be used in this context, and why not."""

    allowed: bool
    capability: str
    tier: MaturityTier
    context: UseContext
    required_tier: MaturityTier
    reason: str
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "capability": self.capability,
            "tier": self.tier.name.lower(),
            "context": self.context.name.lower(),
            "required_tier": self.required_tier.name.lower(),
            "reason": self.reason,
            "missing": list(self.missing),
        }


def admission_for(
    maturity: CapabilityMaturity, context: UseContext
) -> MaturityAdmission:
    """Decide whether a capability is mature enough for this kind of use.

    This is the gate. An UNRATED connector remains fully usable when a person
    asked for it and is watching — the point is not to restrict Aura's reach,
    it is to stop the least-exercised code in the registry from being reachable
    by the most consequential path.
    """
    required = CONTEXT_MINIMUM[context]
    if maturity.tier >= required:
        return MaturityAdmission(
            allowed=True,
            capability=maturity.name,
            tier=maturity.tier,
            context=context,
            required_tier=required,
            reason="tier satisfies the context",
        )
    return MaturityAdmission(
        allowed=False,
        capability=maturity.name,
        tier=maturity.tier,
        context=context,
        required_tier=required,
        reason=(
            f"{maturity.name} is {maturity.tier.name.lower()}; "
            f"{context.name.lower()} use requires {required.name.lower()}"
        ),
        missing=maturity.missing_for_next,
    )

# ── deriving maturity from what a capability ALREADY declares ───────────────
#
# Aura's skills already carry structured metadata: a typed input model, an
# effect scope, an authority class. Introducing a second, parallel declaration
# for maturity would mean two sources of truth that drift apart — and the one
# nobody updates would be the one gating autonomous action.
#
# So maturity is DERIVED from the existing declarations wherever possible, and
# explicit properties only supplement them. A skill that already declares a
# typed input model does not have to say so twice.

#: Effect scopes that cannot change anything outside the process, and are
#: therefore trivially safe to retry.
_READ_ONLY_SCOPES = frozenset({"read_only", "pure_compute", "status", "sandboxed_compute"})

#: Effect scopes whose actions reach outside Aura and may not be undoable.
_IRREVERSIBLE_SCOPES = frozenset({"external_io", "model_weight_mutation"})


def derive_properties(
    *,
    input_model: Any = None,
    schema_override: Any = None,
    effect_scope: str = "unknown",
    authority_class: str = "unclassified",
    declared: Any = None,
) -> set[str]:
    """Infer checkable maturity properties from a skill's existing metadata.

    Only claims that the metadata genuinely supports are inferred:

    * a typed input model or an explicit schema IS a typed request schema;
    * a read-only or pure-compute effect scope IS idempotent by construction —
      an operation that changes nothing outside the process is safe to repeat;
    * a classified authority IS authority scoping.

    Everything else — bounded timeouts, postcondition verification, effect
    receipts, recovery, diagnostics — has to be declared, because no existing
    field implies it and inferring it would be exactly the kind of unearned
    claim this module exists to prevent.
    """
    properties = _normalize_properties(declared)

    if input_model is not None or schema_override:
        properties.add("typed_request_schema")

    scope = str(effect_scope or "").strip().lower()
    if scope in _READ_ONLY_SCOPES:
        # Nothing outside the process changes, so a retry cannot double-apply.
        properties.add("idempotent_or_read_only")
        # There is nothing to undo, so reversibility is satisfied trivially.
        properties.add("reversible_or_confirmed")

    authority = str(authority_class or "").strip().lower()
    if authority and authority != "unclassified":
        properties.add("authority_scoped")

    return properties

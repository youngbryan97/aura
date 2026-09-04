"""Denied(m) ⇒ m does not run — held beside the ladder that denies.

The other two sentences of the one account of what a change is live in
``core/cognition/what_a_change_means.py``. This one is here because an
invariant belongs next to what it protects, and because core.cognition may
not import core.self_modification. The aggregator there reads all three
through the invariant registry, so neither package reaches for the other.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.verify.invariants import Severity, Violation, invariant

__all__ = ["MEANS_WHAT_IT_SAYS"]

#: The name this is registered under, for anything reading it by name.
MEANS_WHAT_IT_SAYS = "development.a_refused_proposal_does_not_run"


@invariant(
    "development.a_refused_proposal_does_not_run",
    scope="cognition",
    severity=Severity.ERROR,
    owner="core/self_modification/growth_ladder.py",
)
def _a_refused_proposal_does_not_run() -> Iterator[Violation]:
    """Denied(m) ⇒ m does not run, at every interface that asks.

    The ladder returns a proposal object, and the natural thing to write with
    one is ``if not granted:``. Every object is truthy, so for as long as the
    object did not answer that question, both callers read a refusal as a
    grant. The check is the caller's own expression, not the field it should
    have used.
    """
    try:
        from core.self_modification.growth_ladder import (
            ModificationLevel,
            ModificationProposal,
        )
    except ImportError:
        return

    refused = ModificationProposal(
        id="an-invariant",
        timestamp=0.0,
        level=ModificationLevel.SKILL_CREATION,
        domain="an invariant checking a refusal",
        description="",
        justification="",
        diff_patch=None,
        proposed_by="aura",
        status="rejected_user",
        decision=False,
    )
    if refused:
        yield Violation(
            invariant="development.a_refused_proposal_does_not_run",
            subject="ModificationProposal",
            message=(
                "a refused proposal reads as truthy, so `if not granted:` "
                "proceeds over a refusal at every caller"
            ),
            severity=Severity.ERROR,
        )
    unadjudicated = ModificationProposal(
        id="an-invariant",
        timestamp=0.0,
        level=ModificationLevel.OBSERVATION,
        domain="an invariant checking a default",
        description="",
        justification="",
        diff_patch=None,
        proposed_by="aura",
    )
    if unadjudicated:
        yield Violation(
            invariant="development.a_refused_proposal_does_not_run",
            subject="ModificationProposal",
            message=(
                "a proposal nobody has adjudicated reads as granted, so the "
                "default answer to a request to change herself is yes"
            ),
            severity=Severity.ERROR,
        )

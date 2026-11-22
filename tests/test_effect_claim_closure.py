"""The honesty layer's coverage is closed over what Aura can actually do.

These are ratchets, not examples. The criticism they answer is that honesty in
this codebase was a growing collection of linguistic detectors and therefore
could never be complete. The reply is that EFFECTS are finite even though
sentences are not, so coverage can be closed over the capability vocabulary —
and closure that is not gated is just a claim, which is the failure mode this
repository has had to name repeatedly.

Two properties are enforced here:

1. Every declared capability has a spec, with its act in words (scope closure).
2. Every observable action declares sentences that claim it and sentences that do not. Its claims are read by meaning
   against the turn's record (core/epistemics/what_her_record_refutes.py), and those sentences measure that reading
   (tests/test_what_her_record_refutes.py, REGISTRY_MISSED). The patterns that once found each claim are gone.
"""

from __future__ import annotations

from core.epistemics.effect_registry import (
    EFFECT_REGISTRY,
    coverage_gaps,
    observable_actions,
    registry_report,
)
from core.runtime.desktop_task_contract import DESKTOP_TASK_ALLOWED_ACTIONS


def test_every_declared_action_has_an_effect_spec() -> None:
    """A capability with no spec is a capability nothing can audit."""

    missing = [
        action for action in DESKTOP_TASK_ALLOWED_ACTIONS if action not in EFFECT_REGISTRY
    ]
    assert missing == [], (
        f"{missing} can be executed but have no EffectSpec. Add one to "
        "core/epistemics/effect_registry.py — an action Aura can perform and "
        "report without anything able to check the report is the exact failure "
        "the registry exists to make impossible."
    )


def test_coverage_is_closed() -> None:
    """The registry-level gate: no holes on either the render or audit side."""

    gaps = coverage_gaps()
    assert gaps == {}, f"honesty coverage has holes: {gaps}"
    assert registry_report()["closed"] is True


def test_registry_holds_nothing_aura_cannot_do() -> None:
    """The registry does not drift ahead of the executor either.

    A spec for an action the executor refuses would make the audit look
    broader than the capability surface actually is.
    """

    declared = set(DESKTOP_TASK_ALLOWED_ACTIONS)
    extra = sorted(action for action in EFFECT_REGISTRY if action not in declared)
    assert extra == [], f"{extra} are registered but not executable"


def test_every_observable_action_says_its_act_and_how_it_is_claimed() -> None:
    """What the record reading needs of each action: the act in words, and sentences to be measured on."""

    lacking = sorted(
        action for action in observable_actions()
        if not EFFECT_REGISTRY[action].claim_description
        or not EFFECT_REGISTRY[action].claim_examples
        or not EFFECT_REGISTRY[action].non_claim_examples
    )
    assert lacking == [], f"{lacking} cannot be read against the record or measured"

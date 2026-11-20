"""The honesty layer's coverage is closed over what Aura can actually do.

These are ratchets, not examples. The criticism they answer is that honesty in
this codebase was a growing collection of linguistic detectors and therefore
could never be complete. The reply is that EFFECTS are finite even though
sentences are not, so coverage can be closed over the capability vocabulary —
and closure that is not gated is just a claim, which is the failure mode this
repository has had to name repeatedly.

Three properties are enforced here:

1. Every declared capability has a recogniser (scope closure).
2. Every recogniser actually fires on a real phrasing of its effect — a
   recogniser that exists and never matches is the half-wired-signal defect,
   and it would satisfy property 1 while protecting nothing.
3. The zero-evidence case is read by meaning, without any per-effect pattern: that is
   core/epistemics/what_her_record_refutes.py, tested in tests/test_what_her_record_refutes.py.
"""

from __future__ import annotations

import pytest

from core.conversation.claimed_effect import (
    _CLAIMED_EFFECT_PATTERNS,
    unverified_effect_correction,
)
from core.conversation.effect_claim import _EFFECT_VOCABULARY
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


#: One phrasing per observable action, written the way a reply would say it.
#: These are the recall floor: if a recogniser stops matching its own effect,
#: property 1 still passes and nothing is protected, which is how a channel
#: goes half-wired without any test noticing.
#:
#: They are READ FROM THE REGISTRY, not written here. This was a second
#: hand-maintained list beside the specs, and the finding this file records —
#: "a second list that never learned what the first knew" — reproduced itself
#: in the fixture written to prevent it: adding pursue_on_screen to the
#: registry left this dict a phrasing short.
_LIVE_PHRASINGS: dict[str, str] = {
    action: spec.claim_examples[0]
    for action, spec in EFFECT_REGISTRY.items()
    if spec.claim_examples
}


def test_every_observable_action_has_a_live_phrasing_fixture() -> None:
    """The fixture set above cannot lag the registry silently."""

    uncovered = sorted(set(observable_actions()) - set(_LIVE_PHRASINGS))
    assert uncovered == [], (
        f"{uncovered} have a recogniser but no phrasing fixture, so nothing "
        "proves the recogniser fires. Declare claim_examples on the spec."
    )


@pytest.mark.parametrize("action", sorted(_LIVE_PHRASINGS))
def test_recognizer_fires_on_its_own_effect(action: str) -> None:
    """Property 2: a recogniser that never matches protects nothing."""

    spec = EFFECT_REGISTRY[action]
    assert spec.recognizer is not None
    sentence = _LIVE_PHRASINGS[action]
    assert spec.recognizer.search(sentence), (
        f"the {action} recogniser does not match {sentence!r}, so a reply "
        "phrased that way would claim the effect unaudited"
    )


@pytest.mark.parametrize("action", sorted(_LIVE_PHRASINGS))
def test_claim_without_receipt_is_corrected(action: str) -> None:
    """End to end: the phrasing, no receipts, a correction naming the effect."""

    correction = unverified_effect_correction(_LIVE_PHRASINGS[action], [])
    assert correction, f"{action} claim passed with no receipts"
    assert EFFECT_REGISTRY[action].claim_description in correction


@pytest.mark.parametrize("action", sorted(_LIVE_PHRASINGS))
def test_claim_with_a_verified_receipt_is_left_alone(action: str) -> None:
    """The guard corrects false completion, not completion."""

    receipts = [{"action": action, "ok": True, "index": 0}]
    correction = unverified_effect_correction(_LIVE_PHRASINGS[action], receipts)
    assert EFFECT_REGISTRY[action].claim_description not in correction


def test_recogniser_count_only_grows() -> None:
    """A hand-maintained list covered 6 of 23. Never go back below that.

    The number is the point of the finding: the gap was not a missing
    phrasing, it was a second list that never learned what the first knew.
    """

    assert len(_CLAIMED_EFFECT_PATTERNS) >= 23
    assert len(_EFFECT_VOCABULARY) >= 15


def test_open_app_recogniser_requires_a_real_capital() -> None:
    """IGNORECASE made ``[A-Z]`` match any letter, so the rule said nothing."""

    spec = EFFECT_REGISTRY["open_app"]
    assert spec.recognizer is not None
    assert not spec.recognizer.search("I opened your message and read it.")
    assert not spec.recognizer.search("I opened the file you mentioned.")
    assert spec.recognizer.search("I opened Notes.")

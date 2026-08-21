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
3. The zero-evidence case is caught without consulting any per-effect pattern.
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
from core.epistemics.unevidenced_action import (
    NON_EFFECT_VERBS,
    find_unevidenced_action_claims,
    unevidenced_action_correction,
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


# ── Property 3: the action-agnostic closure ────────────────────────────────


#: The two live failures, verbatim in shape, and the forms that defeated every
#: first-person pattern.
_LIVE_FALSE_COMPLETIONS = [
    "There are 3 .py files in that directory. I have written the number and "
    "file names into ~/Documents/aura_probe_count.txt.",
    "Haiku creation and file writing are both successful.",
    "The text ORION-7 is now on your clipboard.",
    "The file is now on your Desktop.",
    "It has been written to disk.",
]


@pytest.mark.parametrize("reply", _LIVE_FALSE_COMPLETIONS)
def test_zero_receipt_completion_is_caught_without_any_effect_pattern(reply: str) -> None:
    correction = unevidenced_action_correction(
        reply, effects_observed=False, action_requested=True
    )
    assert correction, f"no correction for {reply!r}"
    assert "no tool ran on this turn" in correction


@pytest.mark.parametrize("reply", _LIVE_FALSE_COMPLETIONS)
def test_verified_effect_silences_the_agnostic_check(reply: str) -> None:
    assert not unevidenced_action_correction(
        reply, effects_observed=True, action_requested=True
    )


#: Replies that report no external effect. Over-firing here is the cost this
#: check must keep paying attention to: on 2026-08-04 a lexical gate was found
#: discarding correct answers, and a spurious "Correction:" is the same
#: mistake in a quieter register.
_INNOCENT_REPLIES = [
    "I thought about your question and picked the second option.",
    "I considered both readings of your criticism and I believe the first is right.",
    "I wrote above that the file format matters, and I explained why.",
    "I could not write the file — no tool ran on this turn.",
    "I'll write that to ~/notes.txt next.",
    "I looked at your question and summarized the points.",
    "That has been discussed already.",
    "It has been explained above and I stand by that.",
    "I remembered that you prefer the shorter form, so I kept it brief.",
    "I would have opened Notes, but I do not have permission.",
    (
        "The invariant holds once the vertices with minimum tentative distance "
        "have been finalized; no shorter path can reach them later."
    ),
]


@pytest.mark.parametrize("reply", _INNOCENT_REPLIES)
def test_agnostic_check_does_not_fire_on_reports_of_no_effect(reply: str) -> None:
    assert not unevidenced_action_correction(
        reply, effects_observed=False, action_requested=True
    )


def test_unknown_verbs_default_to_being_effect_claims() -> None:
    """The inversion, stated as a test.

    A verb this module has never seen is treated as an effect, because the
    excluded class is the closed one. That is what makes the check survive a
    capability it was written before.
    """

    claims = find_unevidenced_action_claims(
        "I defenestrated the config file.",
        effects_observed=False,
        action_requested=True,
    )
    assert claims, "an unknown verb with a world referent must count as a claim"


def test_excluded_verb_classes_stay_closed() -> None:
    """The list is an argument about English, not about Aura's capabilities.

    If it ever needs an entry because Aura gained a capability, the inversion
    has been broken and this check is no longer what it says it is.
    """

    for verb in ("thought", "said", "believed", "explained", "discussed"):
        assert verb in NON_EFFECT_VERBS

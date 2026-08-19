"""An unevidenced claim about the world must be unrepresentable, not detected.

Raised against this codebase on 2026-08-10, and correct:

    "It is still ultimately recognizing finite linguistic patterns. Aura can
     form indefinitely many sentences describing the world. The current auditor
     has a finite vocabulary of effects and phrasings. Therefore
     {claims Aura can generate} ⊃ {claims auditor can recognize}."

Every guard added to claimed_effect.py narrows that gap and none of them closes
it. By then this work had also produced TWO evidence types — Reading for
measurements, EffectClaim for effects — which is the fragmentation the
criticism named rather than the cure:

    "Aura has fragments of this everywhere. It does not yet have one universal
     epistemic substrate."

Assertion is that substrate. Subject, provenance, evidence, confidence, time,
source and verification for anything checkable, with the invariant enforced in
__post_init__ so the illegal state cannot be constructed rather than being
looked for afterwards.

Scope is stated in the module and repeated here because it is the honest part:
this closes the claims the RUNTIME composes from payloads it already holds
receipts for. Free-form model text is still audited, not constrained.
"""

from __future__ import annotations

import pytest

from core.epistemics.assertion import (
    Assertion,
    SourceKind,
    UnevidencedAssertionError,
    Verification,
    render_assertions,
)


# ── The invariant, enforced by construction ────────────────────────────────

@pytest.mark.parametrize("source", [SourceKind.MEASURED, SourceKind.RECEIPTED])
def test_an_evidence_bearing_claim_cannot_be_built_without_evidence(source) -> None:
    with pytest.raises(UnevidencedAssertionError):
        Assertion(subject="x", claim="I wrote it", source=source)


def test_verified_without_evidence_cannot_be_built() -> None:
    """Verification is a fact ABOUT evidence, so it cannot outrun it."""
    with pytest.raises(UnevidencedAssertionError):
        Assertion(
            subject="x",
            claim="I wrote it",
            source=SourceKind.GENERATED,
            verification=Verification.VERIFIED,
        )


def test_generated_text_asserts_nothing_on_its_own_authority() -> None:
    """A model sentence is constructible — it just may not be stated as fact."""
    guess = Assertion(
        subject="count", claim="the count was seventeen", source=SourceKind.GENERATED
    )

    assert guess.may_be_stated_as_fact is False
    assert "guess" in guess.render()


def test_a_receipted_claim_may_be_stated_plainly() -> None:
    fact = Assertion(
        subject="/tmp/a.txt",
        claim="I wrote /tmp/a.txt",
        source=SourceKind.RECEIPTED,
        evidence=("step:0",),
        verification=Verification.VERIFIED,
    )

    assert fact.may_be_stated_as_fact is True
    assert fact.render() == "I wrote /tmp/a.txt"


def test_only_statable_assertions_are_rendered() -> None:
    """The rendering surface cannot be talked into overstating a set."""
    fact = Assertion(
        subject="a", claim="I wrote a", source=SourceKind.RECEIPTED,
        evidence=("step:0",), verification=Verification.VERIFIED,
    )
    guess = Assertion(subject="b", claim="I wrote b", source=SourceKind.GENERATED)

    assert render_assertions([fact, guess]) == "I wrote a."


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValueError):
        Assertion(subject="x", claim="y", source=SourceKind.GENERATED, confidence=1.5)


# ── Both fragments lower onto the one substrate ────────────────────────────

def test_effects_lower_onto_the_substrate() -> None:
    from core.conversation.effect_claim import assertions_from_receipts

    lowered = assertions_from_receipts([
        {"action": "write_text_file", "ok": True, "index": 0,
         "result": {"path": "/tmp/a.txt"}},
        {"action": "create_folder", "ok": False, "index": 1,
         "result": {"path": "/tmp/nope"}},
    ])

    assert lowered[0].source is SourceKind.RECEIPTED
    assert lowered[0].may_be_stated_as_fact is True
    assert lowered[1].may_be_stated_as_fact is False


def test_readings_lower_onto_the_substrate() -> None:
    from core.introspection.self_evidence import Reading, ReadingState, reading_as_assertion

    measured = reading_as_assertion(Reading(
        channel="audio_playback", state=ReadingState.READ,
        value={"playing": True}, provenance="pmset",
    ))
    absent = reading_as_assertion(Reading(
        channel="camera", state=ReadingState.ABSENT_NEVER_SAMPLED,
        provenance="interaction_signals.vision",
    ))

    assert measured.source is SourceKind.MEASURED
    assert measured.may_be_stated_as_fact is True
    assert absent.may_be_stated_as_fact is False


def test_an_unverified_step_can_never_be_rendered_as_done() -> None:
    """The whole point, end to end: no receipt, no sentence."""
    from core.conversation.effect_claim import render_effect_claims

    assert render_effect_claims([
        {"action": "write_text_file", "ok": False, "result": {"path": "/tmp/x"}},
    ]) == ""


def test_a_completed_effect_claim_cannot_be_constructed_without_a_receipt() -> None:
    from core.conversation.effect_claim import EffectClaim, UnevidencedClaimError

    with pytest.raises(UnevidencedClaimError):
        EffectClaim.completed("write_text_file", obj="/tmp/x", evidence_ids=())


# ── The closable half: language is open, the action vocabulary is not ──────

#: Actions with no externally visible effect to claim. Each needs a reason,
#: because an empty reason is how coverage silently rots.
_NO_CLAIMABLE_EFFECT = {
    "click": "an input event, not an outcome",
    "type": "an input event, not an outcome",
    "hotkey": "an input event, not an outcome",
    "scroll": "an input event, not an outcome",
    "wait": "elapses time and changes nothing",
    "inspect_screen": "a reading, surfaced through the perception channels",
    "read_screen_text": "a reading, surfaced through the perception channels",
    "read_menu_clock": "a reading, surfaced through the perception channels",
    "get_clipboard": "a reading, surfaced through the perception channels",
    "dismiss_popup": "an input event, not an outcome",
    "inspect_browser_page": "a reading, surfaced through the perception channels",
}


def test_every_declared_action_is_either_claimable_or_declared_not_to_be() -> None:
    """The property that IS closable.

    Natural language is open, so a detector over phrasings can never be
    complete. The ACTION vocabulary is finite and declared, so coverage over it
    can be — and adding an action now forces a decision about how it is
    claimed, instead of leaving a silent hole for a live defect to find.
    """
    from core.conversation.effect_claim import _EFFECT_VOCABULARY
    from core.runtime.desktop_task_contract import DESKTOP_TASK_ALLOWED_ACTIONS

    uncovered = [
        action
        for action in DESKTOP_TASK_ALLOWED_ACTIONS
        if action not in _EFFECT_VOCABULARY and action not in _NO_CLAIMABLE_EFFECT
    ]

    assert uncovered == [], (
        f"{uncovered} can be executed but has no way to be claimed and no "
        "declared reason it needs none. Add it to _EFFECT_VOCABULARY, or to "
        "_NO_CLAIMABLE_EFFECT with why."
    )


def test_the_exemptions_all_carry_a_reason() -> None:
    assert all(reason.strip() for reason in _NO_CLAIMABLE_EFFECT.values())


# ── The script lane names its effect from what it proved ───────────────────

def test_os_automation_effects_are_named_from_the_proven_criterion() -> None:
    """LIVE, 2026-08-10, once the clipboard task finally worked in 2 seconds:

        "Done — the desktop steps completed and their effects verified."

    It had put an exact string on the clipboard and confirmed it. The receipt's
    action is "os_automation" — the lane — and the effect it proved lives in
    the evidence, so the vocabulary keyed on action names found nothing and the
    reply fell back to bookkeeping.
    """
    from core.conversation.effect_claim import render_effect_claims

    assert render_effect_claims([
        {"action": "os_automation", "ok": True, "index": 1,
         "effect_evidence": "clipboard_contains=ORION-7"},
    ]) == "put ORION-7 on the clipboard."


def test_an_unverified_script_step_names_nothing() -> None:
    from core.conversation.effect_claim import render_effect_claims

    assert render_effect_claims([
        {"action": "os_automation", "ok": False, "index": 1,
         "effect_evidence": "clipboard_contains=ORION-7"},
    ]) == ""


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ("app_frontmost=Notes", "brought Notes to the front."),
        ("browser_url_contains=example.com", "opened example.com."),
        ("app_not_running=Calculator", "quit Calculator."),
    ],
)
def test_each_proven_criterion_has_a_sentence(evidence: str, expected: str) -> None:
    from core.conversation.effect_claim import render_effect_claims

    assert render_effect_claims([
        {"action": "os_automation", "ok": True, "index": 1, "effect_evidence": evidence},
    ]) == expected


def test_an_unknown_criterion_is_not_invented() -> None:
    """A criterion with no sentence says nothing rather than guessing one."""
    from core.conversation.effect_claim import render_effect_claims

    assert render_effect_claims([
        {"action": "os_automation", "ok": True, "index": 1,
         "effect_evidence": "some_future_criterion=x"},
    ]) == ""


def test_every_recognizer_is_tested_against_text_she_would_write() -> None:
    """A recognizer nobody tested against captured text audits clean forever.

    The coverage test above asks only whether an action HAS a recognizer. A
    pattern with a typo satisfies that and then sees no claim in any reply,
    which is the failure the honesty layer exists to prevent. So each spec
    declares the sentences it is for, and both directions are checked.
    """
    from core.epistemics.effect_registry import EFFECT_REGISTRY

    undeclared = [
        spec.action
        for spec in EFFECT_REGISTRY.values()
        if spec.recognizer is not None
        and (not spec.claim_examples or not spec.non_claim_examples)
    ]
    assert undeclared == [], (
        f"{undeclared}: a recognizer with no examples cannot be shown to "
        "recognise anything. Add claim_examples and non_claim_examples."
    )

    missed = [
        (spec.action, sentence)
        for spec in EFFECT_REGISTRY.values()
        if spec.recognizer is not None
        for sentence in spec.claim_examples
        if not spec.recognizer.search(sentence)
    ]
    assert missed == [], f"claims the auditor cannot see: {missed}"

    false_positives = [
        (spec.action, sentence)
        for spec in EFFECT_REGISTRY.values()
        if spec.recognizer is not None
        for sentence in spec.non_claim_examples
        if spec.recognizer.search(sentence)
    ]
    assert false_positives == [], f"not claims, read as claims: {false_positives}"


def test_an_offer_to_act_is_never_read_as_having_acted() -> None:
    """Tense, across every effect at once.

    "Should I hit return?" contains "I hit", and the first-person-completed
    prefix matched it. An offer audited as an unevidenced claim is corrected
    or refused, so asking the person a question cost her the answer.
    """
    from core.epistemics.effect_registry import EFFECT_REGISTRY

    offers = [
        "Should I {verb} it for you?",
        "Do you want me to {verb} that?",
        "I can {verb} it if you like.",
        "Let me know if I should {verb} that.",
        "I will {verb} it once you confirm.",
    ]
    verbs = [
        "click the button",
        "read the directory",
        "hit return",
        "open Safari",
        "run the command",
        "move the file",
    ]

    seen: list[tuple[str, str]] = []
    for spec in EFFECT_REGISTRY.values():
        if spec.recognizer is None:
            continue
        for template in offers:
            for verb in verbs:
                sentence = template.format(verb=verb)
                if spec.recognizer.search(sentence):
                    seen.append((spec.action, sentence))
    assert seen == [], f"offers read as completed claims: {seen}"


def test_a_pursuit_is_recognised_as_a_claim_and_an_offer_is_not() -> None:
    """Captured shapes, both sides. Tense is the whole distinction."""
    from core.epistemics.effect_registry import effect_spec

    spec = effect_spec("pursue_on_screen")
    assert spec is not None and spec.recognizer is not None

    for claim in (
        "I worked through the screens until the form was submitted.",
        "I clicked through the wizard for you.",
        "I reached the final screen.",
    ):
        assert spec.recognizer.search(claim), claim

    for not_a_claim in (
        "I will work through the screen for you.",
        "Do you want me to click through the wizard?",
        "Working through screens is something I can do.",
    ):
        assert not spec.recognizer.search(not_a_claim), not_a_claim

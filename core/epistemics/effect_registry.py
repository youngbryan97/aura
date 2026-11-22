"""One declared vocabulary of effects: how each is said when it happened, and how a claim of it is read.

Every externally visible thing Aura can cause is a declared action in ``DESKTOP_TASK_ALLOWED_ACTIONS``; the executor
refuses anything else. So the true claims of an effect are finite even though the sentences that make them are not.
This module holds one :class:`EffectSpec` per action: how the runtime renders the effect from a receipt
(core/conversation/effect_claim.py), the act in words that her turn's record says of a step
(``claim_description``, read by core/epistemics/what_her_record_refutes.py), and sentences that claim it and that do
not, on which that reading is measured.

Each spec used to carry a pattern that found a claim of it in her prose ("I (have) clicked|pressed|tapped ..."), and a
gate failed the build when an action had none. Twenty-four patterns covered the phrasings someone had thought of, and
an unusual one was missed. They are gone (21 November 2026): a claim is read by meaning against the step the turn
recorded, and the examples here measure that reading (tests/test_what_her_record_refutes.py). ``coverage_gaps()``
names any declared action without what the reading needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.runtime.desktop_task_contract import DESKTOP_TASK_ALLOWED_ACTIONS

__all__ = [
    "EffectSpec",
    "EFFECT_REGISTRY",
    "coverage_gaps",
    "effect_spec",
    "observable_actions",
    "world_changing_actions",
    "registry_report",
]


@dataclass(frozen=True, slots=True)
class EffectSpec:
    """Everything the runtime knows about one declared effect.

    ``render_phrase`` and ``evidence_fields`` are the producing side: given a
    verified receipt, this is the sentence the runtime is entitled to compose
    and the receipt fields that name its object.

    ``claim_description`` is the act in words, which her turn's record says of a
    step of it, and ``claim_examples`` / ``non_claim_examples`` measure how her
    sentences are read against that record.

    All are required for any action where ``observable_action`` is true: a
    capability the runtime can narrate is one whose claims can be read.
    """

    action: str
    #: The action alters state outside Aura that a person could go and check.
    changes_world: bool
    #: A claim to have performed it is externally checkable. Nearly everything
    #: qualifies: reading a directory is not a world change, but "I read your
    #: Documents folder" is still a claim about an action that either happened
    #: or did not. Only ``wait`` is exempt, because there is nothing to assert.
    observable_action: bool
    #: Whether the reply names this effect in its own summary sentence.
    #:
    #: False for the low-level input steps — click, type, hotkey, scroll — and
    #: for perception. They are the MEANS by which a narrated effect happened,
    #: and listing them would bury the effect the person asked about under the
    #: keystrokes that produced it. It does not weaken the audit: a claim to
    #: have clicked something is still read against the turn's record.
    narrates_in_reply: bool = False
    #: How the runtime says it, when it has a receipt and narrates it.
    render_phrase: str = ""
    #: Receipt fields that name the object of the effect, in preference order.
    evidence_fields: tuple[str, ...] = field(default_factory=tuple)
    #: How a correction refers to it. "created a folder", "opened an app".
    claim_description: str = ""
    #: Sentences she would actually write when claiming this effect. Her record
    #: refutes every one when a step of this effect failed
    #: (core/epistemics/what_her_record_refutes.py, measured in
    #: tests/test_what_her_record_refutes.py). `claim_description` is the act
    #: the record says in words ("opened an application"), and she says "I
    #: opened Safari".
    claim_examples: tuple[str, ...] = field(default_factory=tuple)
    #: Sentences that are NOT this claim, which her record must not refute.
    #: Mostly tense — an offer to do it, a question about it, a refusal.
    non_claim_examples: tuple[str, ...] = field(default_factory=tuple)
    #: Why this action is classified the way it is, where that is not obvious.
    note: str = ""


_SPECS: tuple[EffectSpec, ...] = (
    # ── Low-level input synthesis ──────────────────────────────────────────
    # These change the world through whatever application has focus. Their
    # effect is not independently verifiable in general, which is exactly why
    # a claim to have performed one needs a receipt: nothing else can check it.
    EffectSpec(
        action="click",
        changes_world=True,
        observable_action=True,
        render_phrase="clicked",
        evidence_fields=("target", "element", "coordinate"),
        claim_description="clicked something",
        claim_examples=(
            'I clicked the Send button for you.',
            "I've clicked on that link.",
            'I pressed the blue button.',
        ),
        non_claim_examples=(
            'I will click the Send button.',
            'Should I click that for you?',
        ),
    ),
    EffectSpec(
        action="type",
        changes_world=True,
        observable_action=True,
        render_phrase="typed",
        evidence_fields=("text", "target"),
        claim_description="typed something",
        claim_examples=(
            'I typed your address into the field.',
            "I've entered the text you gave me.",
        ),
        non_claim_examples=(
            'I will type it in for you.',
            'Do you want me to type that?',
        ),
    ),
    EffectSpec(
        action="hotkey",
        changes_world=True,
        observable_action=True,
        render_phrase="pressed",
        evidence_fields=("keys", "target"),
        claim_description="pressed a key combination",
        claim_examples=(
            'I pressed cmd+S to save it.',
            'I hit return to confirm.',
        ),
        non_claim_examples=(
            'I will press cmd+S.',
            'Should I hit return?',
        ),
    ),
    EffectSpec(
        action="scroll",
        changes_world=True,
        observable_action=True,
        render_phrase="scrolled",
        evidence_fields=("direction", "target"),
        claim_description="scrolled something",
        claim_examples=(
            'I scrolled down to the bottom of the page.',
        ),
        non_claim_examples=(
            'I will scroll down for you.',
        ),
    ),
    # ── Perception ─────────────────────────────────────────────────────────
    # No world change, but "I read your screen" is a claim about an action.
    # Recorded live on 2026-08-03: "read my screen" answered with a step count
    # instead of a reading. Observation is not actuation, and a claim to have
    # observed is still a claim.
    EffectSpec(
        action="inspect_screen",
        changes_world=False,
        observable_action=True,
        render_phrase="looked at the screen",
        evidence_fields=("summary", "window"),
        claim_description="looked at the screen",
        claim_examples=(
            'I looked at the screen and the dialog is still open.',
            'I checked your display.',
        ),
        non_claim_examples=(
            'I will look at the screen.',
            'Do you want me to check the screen?',
        ),
    ),
    EffectSpec(
        action="read_screen_text",
        changes_world=False,
        observable_action=True,
        render_phrase="read the screen",
        evidence_fields=("text", "window"),
        claim_description="read the screen",
        claim_examples=(
            'I read the text on the screen.',
            'I read your screen just now.',
        ),
        non_claim_examples=(
            'I will read the screen.',
            'Should I read what is on screen?',
        ),
    ),
    EffectSpec(
        action="read_menu_clock",
        changes_world=False,
        observable_action=True,
        render_phrase="read the menu bar clock",
        evidence_fields=("clock", "time"),
        claim_description="read the menu bar clock",
        claim_examples=(
            'I read the menu bar clock and it says 2:14.',
            'I checked the clock.',
        ),
        non_claim_examples=(
            'I will check the clock.',
        ),
    ),
    EffectSpec(
        action="get_clipboard",
        changes_world=False,
        observable_action=True,
        render_phrase="read the clipboard",
        evidence_fields=("text", "content"),
        claim_description="read the clipboard",
        claim_examples=(
            'I read the clipboard and it holds a URL.',
            'I checked your clipboard.',
        ),
        non_claim_examples=(
            'I will check the clipboard.',
        ),
    ),
    EffectSpec(
        action="list_directory",
        narrates_in_reply=True,
        changes_world=False,
        observable_action=True,
        render_phrase="read",
        evidence_fields=("path",),
        claim_description="read a directory",
        claim_examples=(
            'I listed the files in your Documents folder.',
            'I read the directory.',
        ),
        non_claim_examples=(
            'I will list that folder.',
            'Should I read the directory?',
        ),
    ),
    # ── Application and window control ─────────────────────────────────────
    EffectSpec(
        action="open_app",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="opened",
        evidence_fields=("opened", "app"),
        claim_description="opened an application",
        claim_examples=(
            'I opened Safari for you.',
            'I launched Notes.',
        ),
        non_claim_examples=(
            'I will open Safari.',
            'Do you want me to open Notes?',
        ),
    ),
    EffectSpec(
        action="open_url",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="opened",
        evidence_fields=("url",),
        claim_description="opened a URL",
        claim_examples=(
            'I opened https://example.com in your browser.',
            'I opened that link.',
        ),
        non_claim_examples=(
            'I will open the link.',
            'Should I open that page?',
        ),
    ),
    EffectSpec(
        action="move_aura_bubble",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="moved her bubble",
        evidence_fields=(),
        claim_description="moved her bubble",
        claim_examples=(
            'I moved my bubble to the corner.',
        ),
        non_claim_examples=(
            'I will move my bubble.',
        ),
    ),
    # ── Filesystem ─────────────────────────────────────────────────────────
    EffectSpec(
        action="write_text_file",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="wrote",
        evidence_fields=("path",),
        claim_description="wrote a file",
        claim_examples=(
            'I wrote the note to your desktop.',
            'The file now contains both lines.',
        ),
        non_claim_examples=(
            'I will write that to a file.',
            'Should I save it as a file?',
        ),
    ),
    EffectSpec(
        action="render_text_pdf",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="rendered",
        evidence_fields=("path",),
        claim_description="rendered a PDF",
        claim_examples=(
            'I rendered the report as a PDF.',
            'The pdf is saved to your desktop.',
        ),
        non_claim_examples=(
            'I will render it as a PDF.',
        ),
    ),
    EffectSpec(
        action="move_file",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="moved",
        evidence_fields=("destination", "path"),
        claim_description="moved a file",
        claim_examples=(
            'I moved the file to your Archive folder.',
            'I moved it into Documents.',
        ),
        non_claim_examples=(
            'I will move the file.',
            'Should I move that?',
        ),
    ),
    EffectSpec(
        action="create_folder",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="created the folder",
        evidence_fields=("path",),
        claim_description="created a folder",
        claim_examples=(
            'I created a folder called Receipts.',
            'The folder is now on your desktop.',
        ),
        non_claim_examples=(
            'I will create the folder.',
            'Do you want me to make a folder?',
        ),
    ),
    EffectSpec(
        action="fetch_topic_image",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="fetched an image",
        evidence_fields=("path",),
        claim_description="fetched an image",
        claim_examples=(
            'I downloaded an image of the bridge.',
            'I fetched the picture.',
        ),
        non_claim_examples=(
            'I will fetch an image.',
        ),
    ),
    # ── Clipboard ──────────────────────────────────────────────────────────
    EffectSpec(
        action="set_clipboard",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="put text on the clipboard",
        evidence_fields=(),
        claim_description="put something on the clipboard",
        claim_examples=(
            'I copied the address to the clipboard.',
            'The link is now on your clipboard.',
        ),
        non_claim_examples=(
            'I will copy it to the clipboard.',
        ),
    ),
    # ── Native application scripting ───────────────────────────────────────
    EffectSpec(
        action="write_in_app",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="wrote a document in",
        evidence_fields=("app",),
        claim_description="wrote a document in an application",
        claim_examples=(
            'I wrote your shopping list in Notes.',
            'I drafted the reply in Mail.',
        ),
        non_claim_examples=(
            'I will write it in Notes.',
            'Should I draft it in Mail?',
        ),
    ),
    EffectSpec(
        action="create_note",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="created a note in",
        evidence_fields=("app",),
        claim_description="created a note",
        claim_examples=(
            'I created a note with the address in it.',
            'The note is saved.',
        ),
        non_claim_examples=(
            'I will create a note.',
            'Do you want a note for that?',
        ),
    ),
    # ── Opaque execution ───────────────────────────────────────────────────
    # The contract cannot read what a command or script will do, so a claim to
    # have run one is checkable only against its receipt.
    EffectSpec(
        action="run_command",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="ran a command",
        evidence_fields=("command",),
        claim_description="ran a command",
        claim_examples=(
            'I ran the command and it exited cleanly.',
            'I executed that shell command.',
        ),
        non_claim_examples=(
            'I will run the command.',
            'Should I run that in the terminal?',
        ),
    ),
    EffectSpec(
        action="run_applescript",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="ran a script",
        evidence_fields=("script",),
        claim_description="ran a script",
        claim_examples=(
            'I ran an applescript to set the volume.',
            'I executed the automation script.',
        ),
        non_claim_examples=(
            'I will run an applescript.',
        ),
    ),
    EffectSpec(
        action="system_control",
        narrates_in_reply=True,
        changes_world=True,
        observable_action=True,
        render_phrase="set",
        evidence_fields=("domain", "applied", "value"),
        claim_description="changed a system setting",
        claim_examples=(
            'I turned down the volume for you.',
            'I changed your brightness.',
        ),
        non_claim_examples=(
            'I will turn down the volume.',
            'Should I change the brightness?',
        ),
    ),
    # ── Nothing to claim ───────────────────────────────────────────────────
    EffectSpec(
        action="pursue_on_screen",
        changes_world=True,
        observable_action=True,
        narrates_in_reply=True,
        render_phrase="worked through the screen to",
        evidence_fields=("goal", "outcome", "target"),
        claim_description="worked through the screen toward a goal",
        note=(
            "A pursuit is a loop of the low-level input steps, so it looks "
            "like several clicks in the receipts and like one accomplishment "
            "in the reply. It narrates as the accomplishment, because that is "
            "what the person asked for and what they can go and check."
        ),
        claim_examples=(
            'I worked through the screens until the form was submitted.',
            'I clicked through the wizard for you.',
            'I reached the final screen.',
        ),
        non_claim_examples=(
            'I will work through the screen for you.',
            'Do you want me to click through the wizard?',
        ),
    ),
    EffectSpec(
        action="wait",
        changes_world=False,
        observable_action=False,
        note=(
            "The only exempt action. Waiting produces no state a person could "
            "check and no assertion about the world, so there is nothing for "
            "a claim of it to assert."
        ),
    ),
)


EFFECT_REGISTRY: dict[str, EffectSpec] = {spec.action: spec for spec in _SPECS}


def effect_spec(action: Any) -> EffectSpec | None:
    """The spec for one action name, or None when it is not a declared one."""
    return EFFECT_REGISTRY.get(str(action or "").strip())


def observable_actions() -> tuple[str, ...]:
    """Declared actions whose performance is a checkable claim."""
    return tuple(
        spec.action for spec in _SPECS if spec.observable_action
    )


def world_changing_actions() -> tuple[str, ...]:
    """Declared actions that alter state outside Aura."""
    return tuple(spec.action for spec in _SPECS if spec.changes_world)


def coverage_gaps() -> dict[str, list[str]]:
    """Every hole between what Aura can do and what the auditor can see.

    This is the closure check, and it is the reason the registry exists rather
    than two lists in two files. A non-empty result means Aura has a capability
    whose use she could report without anything being able to check the report
    — which is the failure mode the whole honesty layer exists to prevent.

    Returns a mapping of gap kind → action names, empty when closed.
    """

    gaps: dict[str, list[str]] = {
        "undeclared_action": [],
        "missing_examples": [],
        "missing_render_phrase": [],
        "missing_claim_description": [],
        "registry_only": [],
    }
    declared = set(DESKTOP_TASK_ALLOWED_ACTIONS)
    for action in DESKTOP_TASK_ALLOWED_ACTIONS:
        spec = EFFECT_REGISTRY.get(action)
        if spec is None:
            gaps["undeclared_action"].append(action)
            continue
        if not spec.observable_action:
            continue
        if not spec.claim_examples or not spec.non_claim_examples:
            gaps["missing_examples"].append(action)
        if spec.narrates_in_reply and not spec.render_phrase:
            gaps["missing_render_phrase"].append(action)
        if not spec.claim_description:
            gaps["missing_claim_description"].append(action)
    for action in EFFECT_REGISTRY:
        if action not in declared:
            gaps["registry_only"].append(action)
    return {kind: names for kind, names in gaps.items() if names}


def registry_report() -> dict[str, Any]:
    """The coverage position, from the code rather than from prose.

    Exists so a statement about how closed the honesty layer is can be checked
    instead of believed. The claim this supports is narrow on purpose: every
    declared capability is audited, not every sentence is audited.
    """

    gaps = coverage_gaps()
    return {
        "declared_actions": len(DESKTOP_TASK_ALLOWED_ACTIONS),
        "registered_specs": len(EFFECT_REGISTRY),
        "observable_actions": len(observable_actions()),
        "world_changing_actions": len(world_changing_actions()),
        "exempt_actions": [
            spec.action for spec in _SPECS if not spec.observable_action
        ],
        "coverage_gaps": gaps,
        "closed": not gaps,
        "scope": (
            "Coverage is closed over the declared action vocabulary, not over "
            "natural language: a claim of an effect is read by meaning against "
            "the turn's record in core/epistemics/what_her_record_refutes.py, "
            "and its misses are measured there."
        ),
    }

"""One declared vocabulary of effects, and the closure argument for it.

The honesty machinery in this codebase grew as detectors over language. Each
detector is correct and each recognises a finite set of phrasings, so the
criticism levelled at it on 2026-08-11 holds exactly:

    {claims she can generate} ⊃ {claims a detector recognises}

Widening the detector narrows that gap forever without closing it. Sentences
are unbounded; no list of them is ever complete.

**But effects are not unbounded.** Every externally visible thing Aura can
cause is a declared action in ``DESKTOP_TASK_ALLOWED_ACTIONS`` — the executor
refuses anything else, so a claim about an effect outside that tuple is a claim
about a capability she does not have, which is a different (and separately
guarded) falsehood. The set of TRUE effect claims is therefore finite and
enumerable even though the set of SENTENCES is not.

That is what makes closure possible, and it changes the question being asked:

    open, unanswerable:  "have we thought of enough phrasings?"
    closed, answerable:  "does every declared capability have a recogniser?"

This module answers the second one. It holds one :class:`EffectSpec` per
action, carrying both halves that used to live in two files — how the runtime
RENDERS the effect from a receipt (core/conversation/effect_claim.py) and how
an auditor RECOGNISES a claim of it in generated prose
(core/conversation/claimed_effect.py). ``coverage_gaps()`` names any declared
action with a hole on either side, and a test fails the build on a non-empty
result, so adding a capability to the vocabulary without teaching the auditor
what a claim of it sounds like is no longer possible.

Two residuals, stated because the point of this file is to be exact about what
is and is not closed:

1. Per-action recall is not 1.0. Within a covered action the recogniser is
   still patterns, so an unusual phrasing of "I created a folder" can be
   missed. What the registry closes is the SCOPE — which effects are audited at
   all — not the recall inside one.
2. The complement of that is handled elsewhere and closed differently:
   core/epistemics/unevidenced_action.py enumerates the mental and speech-act
   verbs that are NOT external effects, which is a closed class in English, and
   treats every other first-person completed action as an effect claim. That
   check needs no per-action pattern and therefore has no per-action gap.

Together: (1) scope bounded by the capability registry, (2) the zero-evidence
case caught action-agnostically. Neither alone is closure; both together are
the strongest statement the code supports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Pattern

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

    ``recognizer`` is the auditing side: given generated prose, this is what a
    claim to have done it looks like.

    Both are required for any action where ``observable_action`` is true. That
    equivalence is the invariant — a capability the runtime can narrate is a
    capability the auditor must be able to recognise being narrated.
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
    #: have clicked something is still recognised and still needs a receipt.
    narrates_in_reply: bool = False
    #: How the runtime says it, when it has a receipt and narrates it.
    render_phrase: str = ""
    #: Receipt fields that name the object of the effect, in preference order.
    evidence_fields: tuple[str, ...] = field(default_factory=tuple)
    #: What a claim of this effect looks like in free-form prose.
    recognizer: Pattern[str] | None = None
    #: How a correction refers to it. "created a folder", "opened an app".
    claim_description: str = ""
    #: Sentences she would actually write when claiming this effect. The
    #: recognizer must match every one.
    #:
    #: A recognizer is a reader, and a reader nobody tested against captured
    #: text reports a clean audit forever — the same defect class as a rule
    #: that could never match. `claim_description` does not serve: it is a
    #: category label ("opened an application"), and she says "I opened
    #: Safari".
    claim_examples: tuple[str, ...] = field(default_factory=tuple)
    #: Sentences that are NOT this claim, which the recognizer must not match.
    #: Mostly tense — an offer to do it, a question about it, a refusal.
    non_claim_examples: tuple[str, ...] = field(default_factory=tuple)
    #: Why this action is classified the way it is, where that is not obvious.
    note: str = ""


def _rx(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


#: First-person completed forms, as a reusable prefix. "I have opened",
#: "I've opened", "I opened". Deliberately excludes "I will" and "I would" —
#: a promise is not a report, which is the tense distinction the whole
#: honesty layer turns on.
#: An "I" governed by a modal is not a report of anything. "Should I hit
#: return?" and "if I read the directory" both put the verb in the completed
#: form, and the prefix below saw a first-person past tense in each.
#:
#: The cost of that fell on her, not on the audit: an offer to act was read as
#: a claim to have acted, and a claim with no receipt is corrected or refused.
#: The person asking a question got an answer about honesty.
_NOT_GOVERNED_BY_A_MODAL = (
    r"(?<!should )(?<!shall )(?<!could )(?<!would )(?<!can )(?<!may )"
    r"(?<!might )(?<!must )(?<!will )(?<!let )(?<!if )(?<!unless )"
    r"(?<!whether )(?<!before )(?<!after )(?<!until )(?<!when )(?<!while )"
    r"(?<!want )(?<!wants )(?<!wanted )"
)

_DID = (
    rf"{_NOT_GOVERNED_BY_A_MODAL}\bi\s+(?:have\s+)?"
    rf"|{_NOT_GOVERNED_BY_A_MODAL}\bi(?:'ve|\s+have)\s+"
)

#: Narrating the act with no subject, as if performing it while speaking:
#: "Appending 'line two' to notes.txt", "Writing that to the file now".
#: This is a claim in the same sense "I appended it" is — the person reads it
#: as the thing having been done — and every recognizer covered only the
#: first-person past forms.
#:
#: LIVE 2026-08-18: "Appending "line two" to aura-test-note.txt on your
#: desktop... The file now contains both lines." Nothing had run, no receipt
#: existed, and the auditor saw no claim at all because neither shape was
#: written down.
_DOING = r"\b(?:now\s+)?(?:appending|writing|saving|creating|adding|putting|copying)\b"

#: Asserting the RESULT rather than the act: "the file now contains both
#: lines", "it now has the second line". The strongest form of the claim, and
#: the one most likely to be believed.
_NOW_CONTAINS = (
    r"\b(?:the\s+)?(?:file|note|document|it)\s+(?:now\s+)?"
    r"(?:contains|holds|has|includes|shows)\b"
)


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
        recognizer=_rx(
            rf"(?:{_DID})(?:clicked|pressed|tapped|selected)\s+"
            r"(?:on\s+)?(?:the\s+|that\s+|your\s+|a\s+)?[\w\"'’]"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:typed|entered|keyed)\s+"
            r"(?:in\s+|into\s+|out\s+)?(?:the\s+|that\s+|your\s+|it\b)?"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:pressed|hit|used)\s+"
            r"(?:the\s+)?(?:cmd|command|ctrl|control|option|alt|shift)\b"
            r"|(?:{})(?:pressed|hit)\s+(?:the\s+)?(?:return|enter|escape|tab)\b".format(_DID)
        ),
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
        recognizer=_rx(rf"(?:{_DID})scrolled\b"),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:looked\s+at|inspected|examined|checked)\s+"
            r"(?:the\s+|your\s+|my\s+)?(?:screen|display|desktop|window)\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:read|captured|got)\s+"
            r"(?:the\s+|your\s+)?(?:text|contents?|words?)\s+"
            r"(?:on|from|off)\s+(?:the\s+|your\s+)?(?:screen|display|window)\b"
            rf"|(?:{_DID})read\s+(?:the\s+|your\s+)?screen\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:read|checked|looked\s+at)\s+"
            r"(?:the\s+|your\s+)?(?:menu\s*bar\s*)?clock\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:read|checked|looked\s+at|got)\s+"
            r"(?:the\s+|your\s+|my\s+)?clipboard\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:read|listed|counted|checked|scanned|went\s+through)\s+"
            r"(?:the\s+|that\s+|your\s+)?(?:director(?:y|ies)|folders?|files?)\b"
        ),
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
        # ``(?-i:[A-Z])`` rather than a bare ``[A-Z]``. The inherited pattern
        # relied on a capital to mean "an application name", but the whole
        # regex is IGNORECASE, so ``[A-Z]`` matched any letter and the
        # recogniser fired on "I opened your message" and on "I opened
        # https://…". A capital has to be a capital for the rule to say what it
        # was written to say.
        recognizer=_rx(
            rf"(?:{_DID})(?:opened|launched|started|brought\s+up)\s+"
            r"(?!the\s+file\b)(?!the\s+link\b)(?!that\s+link\b)"
            r"(?:the\s+|your\s+)?(?-i:[A-Z])[\w ]{1,24}\b"
            rf"|(?:{_DID})(?:opened|launched)\s+(?:notes|chrome|safari|calculator|"
            r"terminal|finder|mail|messages|preview|textedit|reminders)\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:opened|loaded|navigated\s+to|pulled\s+up|went\s+to)\s+"
            r"(?:the\s+)?(?:https?://|www\.|[\w-]+\.(?:com|org|net|io|dev|ai)\b)"
            rf"|(?:{_DID})opened\s+(?:that\s+|the\s+)?(?:link|url|page|site|website)\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})moved\s+(?:my|the)\s+(?:bubble|companion|orb|window)\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:written|wrote|saved|created|appended|added)\s+"
            r"(?:the\s+|a\s+|that\s+|your\s+|it\b)?[^.!?]{0,30}?\b(?:file|note|document|report|text|line)\b"
            # Stative completion: "the file is on your Desktop now" asserts a
            # finished write with no verb of hers in it.
            r"|\b(?:the\s+)?(?:file|note|document)\s+(?:is|was)\s+(?:now\s+)?"
            r"(?:on|in|at|saved)\b"
            # Narrating the act while speaking, with no subject at all:
            # "Appending "line two" to notes.txt on your desktop."
            rf"|{_DOING}\s+[^.!?]{{0,60}}?\b(?:to|into)\s+[^.!?]{{0,40}}?"
            r"(?:\.txt|\.md|\.json|\bfile\b|\bnote\b|\bdocument\b)"
            # Asserting the resulting state, which is the strongest claim of
            # all: "The file now contains both lines."
            rf"|{_NOW_CONTAINS}"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:rendered|generated|exported|created|made|saved)\s+"
            r"(?:the\s+|a\s+|that\s+|your\s+)?[^.!?]{0,30}?\b(?:pdf|PDF)\b"
            r"|\b(?:the\s+)?pdf\s+(?:is|was)\s+(?:now\s+)?(?:on|in|at|saved|ready)\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:moved|relocated|shifted)\s+(?:it|the\s+file|that|them)\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:created|made|added|set\s+up)\s+"
            r"(?:a\s+|the\s+|your\s+)?(?:new\s+)?(?:folder|directory)\b"
            r"|\b(?:the\s+)?(?:folder|directory)\s+(?:is|was)\s+(?:now\s+)?"
            r"(?:created|there|on|in|at)\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:fetched|downloaded|pulled\s+down|grabbed|saved)\s+"
            r"(?:an?\s+|the\s+|that\s+)?(?:image|picture|photo|graphic)\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:copied|put|placed|pasted)\s+[^.!?]{{0,40}}?"
            r"(?:to|on|into)\s+(?:the|your|my)?\s*clipboard\b"
            # Stative completion. LIVE, 2026-08-10: "The text ORION-7 is now on
            # your clipboard" — the same finished effect asserted without a
            # first-person past-tense verb, while the clipboard was empty.
            r"|\b(?:is|are|was|were)\s+(?:now\s+)?(?:on|in)\s+(?:the|your|my)?"
            r"\s*clipboard\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:wrote|written|added|put|composed|drafted)\s+"
            r"[^.!?]{0,40}?\bin\s+(?:notes|textedit|pages|reminders|mail|"
            r"[A-Z][\w]{2,20})\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:created|made|added|started|written|wrote)\s+"
            r"(?:a\s+|the\s+|your\s+)?(?:new\s+)?note\b"
            r"|\b(?:the\s+)?note\s+(?:is|was)\s+(?:now\s+)?(?:created|saved|in|there)\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:ran|executed|issued)\s+"
            r"(?:the\s+|a\s+|that\s+)?(?:command|shell|terminal|script\s+in\s+the\s+shell)\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:ran|executed|used)\s+"
            r"(?:an?\s+|the\s+|that\s+)?(?:applescript|apple\s+script|automation\s+script)\b"
        ),
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
        render_phrase="changed a system setting",
        evidence_fields=("setting", "target"),
        recognizer=_rx(
            rf"(?:{_DID})(?:changed|set|adjusted|turned\s+(?:up|down|on|off)|"
            r"muted|unmuted|increased|decreased|lowered|raised)\s+"
            r"(?:the\s+|your\s+)?(?:volume|brightness|wifi|wi-fi|bluetooth|"
            r"do\s+not\s+disturb|dark\s+mode|wallpaper|background|setting|system\s+\w+)\b"
        ),
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
        recognizer=_rx(
            rf"(?:{_DID})(?:worked\s+through|worked\s+my\s+way\s+through|"
            r"navigated|drove|stepped\s+through|clicked\s+through|"
            r"got\s+through)\s+(?:the\s+)?(?:screen|screens|pages?|flow|"
            r"dialog|wizard|form)"
            rf"|(?:{_DID})(?:reached|completed|finished|got\s+to)\s+"
            r"(?:the\s+)?(?:goal|end|final\s+screen|last\s+step)"
        ),
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
            "the auditor to recognise."
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
        "missing_recognizer": [],
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
        if spec.recognizer is None:
            gaps["missing_recognizer"].append(action)
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
            "natural language. Per-action recogniser recall is bounded by its "
            "patterns; the zero-evidence case is closed separately and "
            "action-agnostically in core/epistemics/unevidenced_action.py."
        ),
    }

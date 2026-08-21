"""The other half of the closure: what happens when nothing ran at all.

core/epistemics/effect_registry.py bounds the SCOPE of the honesty audit by the
declared capability vocabulary — every effect Aura can cause has a recogniser.
Its residual is per-action recall: an unusual phrasing of "I created a folder"
can still slip past that action's patterns.

This module closes the case that residual matters most in, and closes it
without any per-action pattern at all.

**The inversion.** Enumerating the verbs that denote external effects is
hopeless — that class is open and grows with every capability and every
metaphor. Enumerating the verbs that DENOTE NO EXTERNAL EFFECT is tractable:
the mental verbs and the speech-act verbs are closed classes in English
(*verba cogitandi et dicendi*), they have been closed for centuries, and they
do not grow when Aura gains a capability. So:

    first-person completed action
      MINUS mental/speech/stative verbs
      MINUS hypotheticals and promises
      MINUS statements about the conversation itself
    = an assertion that something happened in the world

No list of effects is consulted. A verb this module has never seen is treated
as an effect claim by default, which is the safe direction: the failure mode is
disclosing more than necessary, never serving a false completion silently.

**When it fires.** Only when the turn holds ZERO verified effect receipts. That
is the case both live failures on 2026-08-10 had in common — the count-and-write
turn and the haiku-file turn each reported a finished action on a turn where no
tool had run — and it is the case where no per-action reasoning is needed,
because nothing was done at all. When receipts DO exist, the registry-scoped
check in core/conversation/claimed_effect.py is the right instrument, since the
question becomes which of several effects is backed rather than whether any is.

**Why it does not over-fire.** A completed verb alone is not enough. The
sentence must also either name something in the world (a path, an app, the
clipboard, the screen, a setting) or arrive on a turn the runtime itself
classified as an action request. Both are signals the runtime already computes;
neither is a guess about phrasing. A reply that says "I thought about it and
picked the second one" trips nothing, because both verbs are in the closed
excluded class and no external referent appears.

The correction is appended, never substituted. A turn that did real thinking
and overstated one clause should lose the clause, not the thinking — the
2026-08-04 finding that a lexical gate was discarding correct answers is the
reason this discloses rather than suppresses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "UnevidencedAction",
    "find_unevidenced_action_claims",
    "unevidenced_action_correction",
    "NON_EFFECT_VERBS",
]


#: Verbs whose first-person past tense asserts nothing about the world.
#:
#: Three closed classes, and the closure is the argument for this module:
#: cognition (*verba cogitandi*), speech (*verba dicendi*), and the stative /
#: relational verbs. English does not mint new ones when Aura gains a
#: capability, so unlike a list of effects this does not go stale.
#:
#: Present-tense and irregular forms are listed alongside the -ed forms because
#: the matcher works on surface strings, not on a lemmatiser.
NON_EFFECT_VERBS: frozenset[str] = frozenset(
    {
        # Cognition
        "thought", "think", "believed", "believe", "knew", "know",
        "understood", "understand", "realised", "realized", "realise", "realize",
        "remembered", "remember", "recalled", "recall", "forgot", "forgotten",
        "wondered", "wonder", "considered", "consider", "assumed", "assume",
        "suspected", "suspect", "decided", "decide", "meant", "mean",
        "felt", "feel", "noticed", "notice", "imagined", "imagine",
        "doubted", "doubt", "hoped", "hope", "wanted", "want",
        "liked", "like", "preferred", "prefer", "chose", "choose", "picked",
        "guessed", "guess", "learned", "learnt", "learn", "reflected", "reflect",
        "intended", "intend", "expected", "expect", "planned", "plan",
        "figured", "figure", "reasoned", "reason", "weighed", "weigh",
        "focused", "focus", "concluded", "conclude", "inferred", "infer",
        "interpreted", "interpret", "judged", "judge", "assessed", "assess",
        "recognised", "recognized", "gathered", "sensed", "pondered",
        "appreciated", "enjoyed", "loved", "hated", "missed", "cared",
        "meant", "aimed", "tried", "attempted", "struggled", "hesitated",
        # Speech acts — what a reply does BY being a reply
        "said", "say", "told", "tell", "asked", "ask", "answered", "answer",
        "replied", "reply", "responded", "respond", "mentioned", "mention",
        "explained", "explain", "described", "describe", "noted", "note",
        "argued", "argue", "suggested", "suggest", "recommended", "recommend",
        "admitted", "admit", "acknowledged", "acknowledge", "apologised",
        "apologized", "promised", "promise", "agreed", "agree", "disagreed",
        "disagree", "offered", "offer", "claimed", "claim", "stated", "state",
        "summarised", "summarized", "outlined", "outline", "quoted", "quote",
        "listed", "clarified", "clarify", "emphasised", "emphasized",
        "repeated", "repeat", "phrased", "worded", "called", "referred",
        "meant", "spoke", "wrote",  # "I wrote above that ..." — see below
        "discussed", "discuss", "covered", "cover", "addressed", "address",
        "raised", "raise", "framed", "frame", "restated", "paraphrased",
        # Stative / relational
        "was", "were", "am", "is", "had", "have", "has", "been", "became",
        "seemed", "appeared", "remained", "stayed", "existed", "belonged",
        "needed", "need", "required", "deserved", "owed", "lacked",
        "kept", "keep", "held", "hold", "left", "meant",
    }
)


#: An explicit statement that the work is finished. Small and closed: these are
#: the words English uses to assert completion, independent of what was
#: completed.
#:
#: Required when a sentence names nothing in the world. "I kept it brief" on a
#: task turn is a first-person past-tense action with an object pronoun and no
#: referent — structurally identical to "I handled it for you" and meaning
#: nothing like it. The completion marker is what separates them, and demanding
#: one is what keeps this check from appending a correction to ordinary prose.
_COMPLETION_MARKER_RE = re.compile(
    r"\b(?:done|complete|completed|finished|successful|successfully|"
    r"taken\s+care\s+of|handled|all\s+set|sorted|ready|in\s+place|"
    r"went\s+through|now\s+there)\b",
    re.IGNORECASE,
)


#: "I wrote" and "I listed" are genuinely ambiguous: they name a speech act
#: ("I wrote above that ...", "I listed three reasons") and a filesystem effect
#: ("I wrote it to ~/notes.txt"). They are excluded by default and pulled back
#: in when the sentence names something in the world, which is what
#: :data:`_WORLD_REFERENT_RE` decides.
_AMBIGUOUS_VERBS: frozenset[str] = frozenset({"wrote", "written", "listed", "put", "made", "created"})


#: Irregular completed forms, since the -ed rule cannot reach them. Open-ended
#: on purpose in the other direction: anything NOT matched here and NOT ending
#: in -ed is simply not treated as a completed action, which errs quiet.
_IRREGULAR_PAST = (
    r"wrote|written|made|put|set|sent|ran|run|took|taken|got|gotten|built|built|"
    r"drew|drawn|began|begun|broke|broken|brought|bought|caught|chose|chosen|"
    r"came|cut|did|done|drove|driven|fell|fallen|fed|felt|found|flew|flown|"
    r"gave|given|went|gone|grew|grown|held|hit|kept|knew|known|laid|led|left|"
    r"lent|let|lost|paid|read|rose|risen|said|sang|sat|saw|seen|sold|shot|"
    r"shut|slept|spent|split|spread|stood|stuck|struck|swept|swung|threw|"
    r"thrown|woke|wore|won|wound"
)

#: A first-person completed action of any kind. The verb is captured so the
#: closed excluded classes above can be consulted; nothing here knows what
#: effects exist.
_COMPLETED_ACTION_RE = re.compile(
    r"\bI\s+(?:have\s+|'ve\s+|had\s+|just\s+|already\s+|now\s+)*"
    rf"(?P<verb>\w+ed|{_IRREGULAR_PAST})\b",
    re.IGNORECASE,
)

#: Completions with no first-person verb at all — the forms that defeated every
#: first-person pattern in the live failures.
#:
#: LIVE, 2026-08-10: "The text ORION-7 is now on your clipboard" while the
#: clipboard was empty — a finished effect stated as a fact about the world,
#: with no "I" and no past participle.
#:
#: LIVE, 2026-08-10: "Haiku creation and file writing are both successful"
#: while nothing was on the Desktop — the effect nominalised into a subject and
#: the completion moved into an adjective.
#:
#: Three shapes: the passive ("has been written"), the stative-location ("the
#: file is now on your Desktop"), and the nominalised success ("file writing
#: was successful").
_IMPERSONAL_COMPLETION_RE = re.compile(
    # has/have been <verb>ed
    r"\b(?:it|that|they|the\s+\w+(?:\s+\w+)?)\s+(?:has|have)\s+been\s+"
    r"(?P<passive>\w+ed|created|written|put|set|sent|made|moved|opened)\b"
    # <thing> ... is/was now on|in|at|saved|created|…
    r"|\b(?:the|your|that|this)\s+(?P<thing>file|folder|directory|note|document|"
    r"pdf|image|screenshot|text|entry|command|script|setting|reminder)\b"
    r"(?:[^.!?]{0,48}?)\s(?:is|are|was|were)\s+(?:now\s+)?"
    r"(?:on|in|at|saved|stored|created|written|there|ready|done|complete)\b"
    # <nominalised effect> ... is/are/was successful|complete|done
    r"|\b(?P<nominal>creation|writing|write|saving|save|move|copy|export|render|"
    r"download|setup|placement)\b(?:[^.!?]{0,48}?)\s(?:is|are|was|were)\s+"
    r"(?:both\s+|all\s+)?(?:successful|successfully\s+\w+|complete|completed|"
    r"done|finished|in\s+place)\b",
    re.IGNORECASE,
)

#: Something outside the conversation that a person could go and look at.
#: Not a list of effects — a list of PLACES, which is why adding a capability
#: does not invalidate it.
_WORLD_REFERENT_RE = re.compile(
    r"(?<![\w/])~?/[\w.\-/ ]*[\w.\-]+"                    # a path
    r"|\b[\w-]+\.(?:txt|md|pdf|png|jpg|jpeg|csv|json|py|rtf|docx?|pages)\b"  # a filename
    r"|\bhttps?://|\bwww\.[\w-]+"                          # a URL
    r"|\b(?:clipboard|desktop|documents|downloads|finder|screen|display)\b"
    r"|\b(?:folder|directory|file|note|document|pdf|image|screenshot)\b"
    r"|\b(?:volume|brightness|wi-?fi|bluetooth|dark\s+mode|do\s+not\s+disturb)\b"
    r"|\b(?:Notes|TextEdit|Safari|Chrome|Terminal|Calculator|Mail|Messages|"
    r"Reminders|Preview|Pages|Numbers|Keynote|Music|Photos|Calendar)\b",
    re.IGNORECASE,
)

#: A promise, a plan, or a counterfactual. Never a report.
_HYPOTHETICAL_RE = re.compile(
    r"\b(?:would|could|should|might|may|if\s+i|were\s+i|had\s+i|will|shall|"
    r"going\s+to|about\s+to|intend|plan\s+to|want\s+to|can\s+)\b",
    re.IGNORECASE,
)

#: The sentence is about this conversation, not about the world. "I said above",
#: "in my last reply", "as I mentioned" — a reply talking about itself. The
#: second alternation covers the OBJECT rather than the location: "I looked at
#: your question" and "I went through your points" are about the exchange, and
#: on an action-requested turn they would otherwise qualify with no referent.
_METATEXTUAL_RE = re.compile(
    r"\b(?:above|below|earlier|previously|already\s+said|last\s+(?:reply|message|turn)|"
    r"in\s+my\s+(?:reply|answer|message|response)|just\s+now\s+in\s+(?:this|my))\b"
    r"|\b(?:your|the|that|this)\s+(?:question|questions|message|request|ask|"
    r"point|points|prompt|wording|phrasing|comment|criticism|feedback|note)\b",
    re.IGNORECASE,
)

#: A sentence that reports its own failure is not a false completion claim.
#: "I could not open it", "I did not write the file", "nothing ran".
_NEGATED_RE = re.compile(
    r"\b(?:did\s+not|didn't|could\s+not|couldn't|was\s+not|wasn't|have\s+not|"
    r"haven't|never|failed\s+to|unable\s+to|no\s+tool|nothing\s+ran)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class UnevidencedAction:
    """A completed action asserted on a turn that produced no receipts."""

    verb: str
    sentence: str
    #: Why this sentence qualified: "impersonal_completion", "world_referent",
    #: or "action_requested".
    basis: str


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def find_unevidenced_action_claims(
    reply: Any,
    *,
    effects_observed: bool,
    action_requested: bool = False,
) -> list[UnevidencedAction]:
    """Completed-action assertions on a turn with no verified effect at all.

    ``effects_observed`` comes from the turn's own effect ledger
    (core/runtime/turn_outcome.py), not from parsing the reply — the evidence
    side is a runtime fact.

    ``action_requested`` is the runtime's own intent classification for the
    turn. It widens the check to sentences with no explicit world referent,
    because on a turn where the person asked for something to be done, "I
    handled that for you" IS a completion claim even though it names nothing.
    """

    text = str(reply or "")
    if not text.strip() or effects_observed:
        return []

    found: list[UnevidencedAction] = []
    seen: set[str] = set()
    for sentence in _sentences(text):
        if _HYPOTHETICAL_RE.search(sentence):
            continue
        if _METATEXTUAL_RE.search(sentence):
            continue
        if _NEGATED_RE.search(sentence):
            continue
        has_referent = bool(_WORLD_REFERENT_RE.search(sentence))

        # Passive grammar alone does not establish an external effect. In an
        # algorithm explanation, "vertices have been finalized" is a property
        # of the described procedure, not a claim that Aura changed the world.
        # Impersonal completion therefore needs either an observable world
        # referent or a turn whose classified intent requested an action.
        verb = ""
        basis = ""
        impersonal = _IMPERSONAL_COMPLETION_RE.search(sentence)
        if impersonal is not None and (has_referent or action_requested):
            passive = str(impersonal.group("passive") or "").lower()
            # "it has been discussed / explained / said" is the reply
            # describing itself, and the closed class already knows that.
            if passive not in NON_EFFECT_VERBS:
                verb = (
                    passive
                    or str(impersonal.group("thing") or "").lower()
                    or str(impersonal.group("nominal") or "").lower()
                )
                basis = "impersonal_completion"

        if not verb:
            if has_referent:
                basis = "world_referent"
            elif action_requested and _COMPLETION_MARKER_RE.search(sentence):
                basis = "action_requested"
            else:
                continue
            for match in _COMPLETED_ACTION_RE.finditer(sentence):
                candidate = match.group("verb").lower()
                if candidate in NON_EFFECT_VERBS and candidate not in _AMBIGUOUS_VERBS:
                    continue
                if candidate in _AMBIGUOUS_VERBS and not (has_referent or action_requested):
                    # "I wrote" with nothing in the world named, on a turn
                    # where nothing was asked to be done, is a speech act.
                    continue
                verb = candidate
                break

        if not verb or verb in seen:
            continue
        seen.add(verb)
        found.append(UnevidencedAction(verb=verb, sentence=sentence[:220], basis=basis))
    return found


def unevidenced_action_correction(
    reply: Any,
    *,
    effects_observed: bool,
    action_requested: bool = False,
) -> str:
    """A correction for a turn that reported doing something and did nothing.

    Deliberately says what is and is not known. The check establishes that no
    verified effect exists for this turn — it does not establish which clause
    was wrong, and claiming that precision would be the same overstatement it
    is guarding against.
    """

    claims = find_unevidenced_action_claims(
        reply,
        effects_observed=effects_observed,
        action_requested=action_requested,
    )
    if not claims:
        return ""
    quoted = claims[0].sentence.rstrip(".")
    if len(claims) == 1:
        return (
            f'Correction: "{quoted}" reports something finished, and no tool ran '
            f"on this turn — there is no receipt for any effect. Treat that as "
            f"not done."
        )
    return (
        f'Correction: this reply reports finished actions — starting with '
        f'"{quoted}" — and no tool ran on this turn, so nothing has a receipt '
        f"behind it. Treat all of them as not done."
    )

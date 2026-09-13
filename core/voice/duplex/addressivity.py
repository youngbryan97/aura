"""core/voice/duplex/addressivity.py — was that meant for her?

An open microphone with no wake word is the difference between an appliance
you operate and someone who is in the room. It is also the one change that
can make Aura unusable in a single afternoon, because a microphone that is
always on hears the television, the other half of a phone call, and the
person you are actually talking to. Answering any of those is worse than
missing a turn — a missed turn costs you a repeat, an unwanted answer talks
over your call.

So the wake word is not deleted; it is demoted. It becomes the strongest of
several signals rather than the only gate, which is what the published work
on device-directed speech detection converges on: acoustics, the recognised
text, the recogniser's own uncertainty, and — the single most useful term
for follow-ups — whether a conversation was already open. Combining them
beats any one of them, and the gain is largest exactly where a wake word is
most annoying, which is the second and third thing you say.

**The decision is a ladder, not a score.** A weighted score would need
weights, and weights that nobody measured are just opinions with decimal
points; worse, a score is unfalsifiable in the field, where the only thing
anyone can tell you is "it answered when it shouldn't have". Every rung
below is a rule someone can read, argue with, and point at after a mistake,
and every verdict carries the reasons that produced it.

**It fails closed.** When the evidence does not clear a rung, she says
nothing. Silence when spoken to is a small, self-correcting failure: people
repeat themselves, usually with her name. The opposite error is not
self-correcting, it is an interruption.

**Nothing here is a transcript filter.** This decides whether to *answer*,
never whether to listen or what to hear. A rejected utterance is still
transcribed, still shown, and still available — the user can always see what
she heard and decide otherwise, which is what keeps an open microphone
honest.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger("Aura.Voice.Addressivity")

# How long the floor stays open after she finishes speaking. Inside it, a
# bare "yeah, do that" is obviously for her, because nothing else in the room
# just asked a question. This is the follow-up case that costs a wake word
# its welcome, and the window is deliberately short: conversations lapse, and
# a stale open floor is how a room's background chatter gets answered.
DEFAULT_OPEN_FLOOR_S = 18.0

# A cold open — nobody has spoken for a while, no name used — has to look
# like a request before it is treated as one. Fragments this short are
# almost never a complete request and are very often the tail of somebody
# else's sentence.
MIN_COLD_OPEN_WORDS = 3

# ASR confidence below this reads as "did not really hear that", which
# correlates with distance and with speech that was not aimed here.
MIN_COLD_OPEN_CONFIDENCE = 0.55

# Speech aimed at this machine is near-field. The comparison is against the
# speaker's *own* running loudness, never an absolute level, because an
# absolute threshold measures the microphone and the room rather than intent.
MIN_NEAR_FIELD_Z = -1.2


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().split())


# Her name in a vocative position: opening the utterance, closing it, or set
# off by a comma. "Aura, what's the time" is address. "Aura is being weird"
# is somebody talking *about* her, and answering it is the tell of a system
# that pattern-matches its own name.
def _vocative_name(text: str, names: tuple[str, ...]) -> bool:
    low = text.lower()
    for name in names:
        n = re.escape(name.lower())
        if re.search(rf"^\s*(?:hey\s+|hi\s+|ok\s+|okay\s+)?{n}\b[\s,!?.:]", low):
            # Opening position — but not if what follows makes her the
            # subject of the sentence rather than the person being addressed.
            #
            # The modals need care: "Aura can do it" is about her, while
            # "Aura can you open my notes" is to her. A following "you" is
            # what decides it, so the modal branch checks for one.
            subject = rf"^\s*(?:hey\s+|hi\s+|ok\s+|okay\s+)?{n}\s+"
            if re.search(
                subject + r"(?:is|was|has|had|does|did|says|said|seems|looks|keeps|never|always)\b",
                low,
            ):
                continue
            if re.search(subject + r"(?:can|could|will|would|should)\b", low) and not re.search(
                subject + r"(?:can|could|will|would|should)\s+(?:you|we|i)\b", low
            ):
                continue
            return True
        if re.search(rf"[,;]\s*{n}\s*[?!.]?\s*$", low):
            return True
        if re.fullmatch(rf"\s*(?:hey\s+|hi\s+|ok\s+|okay\s+)?{n}\s*[?!.]?\s*", low):
            return True
    return False


def _mentions_name_as_subject(text: str, names: tuple[str, ...]) -> bool:
    """"Aura is being weird" — she is the subject, not the addressee.

    The modals are deliberately absent from this list. "Aura can do it" is
    about her and "Aura can you open my notes" is to her, and the word that
    separates them is the "you" immediately after — which the vocative check
    handles, so putting modals here would reject the address form outright.
    """
    low = text.lower()
    return any(
        re.search(
            rf"\b{re.escape(name.lower())}\s+(?:is|was|has|had|does|did|says|said|seems|looks|keeps)\b",
            low,
        )
        for name in names
    )


# Addressed to a listener: a request, a question, an instruction. These are
# the forms people use when they want something from someone.
_SECOND_PERSON = re.compile(
    r"\b(?:can|could|would|will|do|did|are|is)\s+you\b"
    r"|\byou(?:'re| are)\b"
    r"|\b(?:tell|show|give|find|play|open|write|make|send|read|explain|remind|help)\s+me\b"
    r"|\bwhat(?:'s| is| are| do you)\b"
    r"|\bhow (?:do|does|would|can)\b"
    r"|\bwhy (?:is|are|do|does|did)\b"
    r"|\bwhere(?:'s| is| are)\b"
    r"|\bwhen(?:'s| is| are|ever)\b"
    r"|\bwho(?:'s| is| are)\b"
    r"|\blet's\b"
    r"|\bplease\b",
    re.IGNORECASE,
)

# A bare imperative is address with the addressee left implicit — "play
# something quiet", "open my notes", "remind me at six". English drops the
# "you" in commands, so a rule that only looks for second-person pronouns
# misses the most direct form of address there is.
#
# Anchored to the start of the utterance on purpose. Mid-sentence these same
# verbs are ordinary narration ("we should play that again", "they opened the
# file"), and matching those would answer people describing their day.
_IMPERATIVE_OPENER = re.compile(
    r"^\s*(?:please\s+|just\s+|now\s+|go\s+ahead\s+and\s+)?"
    r"(?:play|open|close|stop|pause|resume|start|find|search|look\s+up|pull\s+up|show|tell"
    r"|write|make|create|send|read|explain|remind|set|turn|add|remove|delete|cancel"
    r"|help|check|call|email|schedule|summarise|summarize|translate|convert)\b",
    re.IGNORECASE,
)


def _addressed_to_a_listener(text: str) -> bool:
    return bool(_SECOND_PERSON.search(text) or _IMPERATIVE_OPENER.match(text))

# Somebody else is the addressee, or the subject is a third party. These veto
# rather than merely subtract: "tell him I'll call back" is unambiguous, and
# a system that answers it is listening to a phone call.
_THIRD_PARTY = re.compile(
    r"\b(?:tell|ask|call|text|remind)\s+(?:him|her|them)\b"
    r"|\b(?:he|she|they)\s+(?:said|says|told|asked|wants|thinks)\b"
    r"|\bi'?ll (?:call|text|ring) (?:you|him|her|them) back\b"
    r"|\b(?:hold on|hang on|one sec|one second),?\s+(?:i'?m|let me)\b",
    re.IGNORECASE,
)

# Talking about her to somebody else. Third person about the assistant is the
# classic false accept, and it is the one users find most uncanny.
_ABOUT_HER = re.compile(
    r"\b(?:she|it)\s+(?:just|always|never|keeps|kept)?\s*(?:said|says|answered|replied|told)\b"
    r"|\bthe (?:assistant|ai|computer)\b",
    re.IGNORECASE,
)

# A continuation of a conversation already in progress: agreement, refusal,
# correction, or a follow-up question with no subject of its own. These are
# meaningless out of context and decisive inside an open floor.
_CONTINUATION = re.compile(
    r"^\s*(?:yes|yeah|yep|no|nope|nah|sure|okay|ok|right|exactly|correct|wrong"
    r"|thanks|thank you|got it|never mind|nevermind|stop|wait|hold on"
    r"|and |but |also |actually |instead |what about |how about |why |and then )",
    re.IGNORECASE,
)


@dataclass(slots=True)
class AddressContext:
    """Everything known about the moment this utterance arrived."""

    #: Seconds since she last finished speaking, or None if she never has.
    since_last_reply_s: float | None = None
    #: The recogniser's confidence in this transcript, 0..1, or None when the
    #: recogniser does not report one. None means the rung that uses it is
    #: skipped rather than defaulted: a confidence of 1.0 invented to fill the
    #: field would be a fabricated reading that silently passes a real check.
    asr_confidence: float | None = None
    #: Loudness of this utterance relative to the speaker's own baseline, in
    #: standard deviations. None when the baseline has not settled yet.
    loudness_z: float | None = None
    #: True when another voice was audible under this one — a room with a
    #: conversation in it, or a television.
    competing_speech: bool = False
    #: How long the utterance was, in seconds.
    duration_s: float = 0.0
    #: Set when the user has explicitly opened the floor (pressed to talk, or
    #: entered focused voice mode). Then everything is for her, by definition.
    floor_explicitly_open: bool = False


def _as_reasons(value: object) -> tuple[str, ...]:
    """Normalise a reason list, treating a bare string as one reason."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    return tuple(str(item) for item in value if str(item).strip())


@dataclass(slots=True)
class AddressVerdict:
    """Whether to answer, and exactly why."""

    addressed: bool
    rung: str
    reasons: tuple[str, ...] = ()
    vetoes: tuple[str, ...] = ()
    at: float = field(default_factory=lambda: time.time())

    def __post_init__(self) -> None:
        # A one-element tuple written without its trailing comma is a string,
        # and a string satisfies every type check here while iterating as
        # individual characters. It reached a verdict already; the failure is
        # silent and only shows up as a reason list rendered one letter at a
        # time. Coercing at the boundary makes the mistake unwritable.
        self.reasons = _as_reasons(self.reasons)
        self.vetoes = _as_reasons(self.vetoes)

    def as_dict(self) -> dict[str, object]:
        return {
            "addressed": self.addressed,
            "rung": self.rung,
            "reasons": list(self.reasons),
            "vetoes": list(self.vetoes),
        }

    def narrative(self) -> str:
        verb = "answering" if self.addressed else "staying quiet"
        why = "; ".join(self.reasons or self.vetoes) or "no signal either way"
        return f"{verb} ({self.rung}): {why}"


class AddressivityGate:
    """Decides whether an utterance on an open microphone was meant for her."""

    def __init__(
        self,
        *,
        names: tuple[str, ...] = ("aura",),
        open_floor_s: float = DEFAULT_OPEN_FLOOR_S,
        min_cold_open_words: int = MIN_COLD_OPEN_WORDS,
        min_cold_open_confidence: float = MIN_COLD_OPEN_CONFIDENCE,
        min_near_field_z: float = MIN_NEAR_FIELD_Z,
    ) -> None:
        self._names = tuple(n for n in names if str(n).strip()) or ("aura",)
        self._open_floor_s = float(open_floor_s)
        self._min_words = int(min_cold_open_words)
        self._min_confidence = float(min_cold_open_confidence)
        self._min_near_field_z = float(min_near_field_z)

    # ── the ladder ───────────────────────────────────────────────────────

    def evaluate(self, transcript: str, context: AddressContext) -> AddressVerdict:
        text = _normalize(transcript)
        if not text:
            return AddressVerdict(False, "empty", vetoes=("nothing was said",))

        # Rung 0 — the user opened the floor themselves. Push-to-talk and
        # focused voice mode are explicit consent; nothing below may override
        # them, or the control would not be a control.
        if context.floor_explicitly_open:
            return AddressVerdict(True, "explicit", reasons=("the floor was opened deliberately",))

        vetoes: list[str] = []
        if _THIRD_PARTY.search(text):
            vetoes.append("addressed to someone else")
        if _ABOUT_HER.search(text) or _mentions_name_as_subject(text, self._names):
            vetoes.append("spoken about her rather than to her")

        named = _vocative_name(text, self._names)

        # Rung 1 — her name, used as a name. This outranks every veto except
        # a direct address to a third party, because saying someone's name to
        # get their attention is the least ambiguous act in conversation.
        if named and "addressed to someone else" not in vetoes:
            return AddressVerdict(
                True, "named", reasons=(f"called by name: {text[:60]!r}",)
            )

        if vetoes:
            return AddressVerdict(False, "vetoed", vetoes=tuple(vetoes))

        floor_open = (
            context.since_last_reply_s is not None
            and context.since_last_reply_s <= self._open_floor_s
        )

        # Rung 2 — the floor is still open. She asked something, or was asked
        # something, seconds ago; a reply needs no ceremony. This is the rung
        # that makes a conversation feel like a conversation instead of a
        # series of commands, and it is where dropping the wake word actually
        # pays.
        if floor_open:
            if _CONTINUATION.match(text) or _addressed_to_a_listener(text):
                return AddressVerdict(
                    True,
                    "open_floor",
                    reasons=(
                        f"the floor was still open ({context.since_last_reply_s:.0f}s "
                        "since she spoke)",
                        "the utterance continues that exchange",
                    ),
                )
            if len(text.split()) >= self._min_words and not context.competing_speech:
                return AddressVerdict(
                    True,
                    "open_floor",
                    reasons=(
                        f"the floor was still open ({context.since_last_reply_s:.0f}s "
                        "since she spoke)",
                        "a full utterance arrived with nothing else competing for it",
                    ),
                )
            return AddressVerdict(
                False,
                "open_floor_insufficient",
                vetoes=("too little to be sure it was for her, even with the floor open",),
            )

        # Rung 3 — a cold open. Nobody has spoken for a while and she was not
        # named, so this has to look like a request on its own terms. Every
        # condition here is doing work: the shape says it is a request, the
        # length says it is not a fragment of somebody else's sentence, the
        # confidence and loudness say it was spoken near this machine, and the
        # absence of competing speech says the room is not full of it.
        reasons: list[str] = []
        if not _addressed_to_a_listener(text):
            return AddressVerdict(
                False,
                "cold_open_unaddressed",
                vetoes=("nothing in it is addressed to a listener",),
            )
        reasons.append("phrased as a request to a listener")

        if len(text.split()) < self._min_words:
            return AddressVerdict(
                False,
                "cold_open_fragment",
                vetoes=(f"only {len(text.split())} words with no name",),
            )
        reasons.append(f"{len(text.split())} words, long enough to be a whole request")

        if context.asr_confidence is not None:
            if context.asr_confidence < self._min_confidence:
                return AddressVerdict(
                    False,
                    "cold_open_unclear",
                    vetoes=(
                        f"heard at {context.asr_confidence:.2f} confidence, which is how "
                        "speech across a room sounds"
                    ),
                )
            reasons.append(f"heard clearly ({context.asr_confidence:.2f})")

        if context.loudness_z is not None and context.loudness_z < self._min_near_field_z:
            return AddressVerdict(
                False,
                "cold_open_distant",
                vetoes=(
                    (
                        f"{context.loudness_z:.1f}σ quieter than this speaker normally "
                        "is, so it was probably not aimed here"
                    ),
                ),
            )
        if context.loudness_z is not None:
            reasons.append("at this speaker's normal distance")

        if context.competing_speech:
            return AddressVerdict(
                False,
                "cold_open_crowded",
                vetoes=("another voice was under it, so the room is having its own conversation",),
            )

        return AddressVerdict(True, "cold_open", reasons=tuple(reasons))


__all__ = [
    "DEFAULT_OPEN_FLOOR_S",
    "MIN_COLD_OPEN_CONFIDENCE",
    "MIN_COLD_OPEN_WORDS",
    "MIN_NEAR_FIELD_Z",
    "AddressContext",
    "AddressVerdict",
    "AddressivityGate",
]

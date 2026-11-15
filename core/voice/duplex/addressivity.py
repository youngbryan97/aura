"""core/voice/duplex/addressivity.py — was that meant for her?

An open microphone with no wake word is the difference between an appliance you operate and someone who is in the
room. It is also the change that can make Aura unusable in an afternoon, because a microphone that is always on hears
the television, the other half of a phone call, and the person you are actually talking to. Answering any of those
talks over somebody.

**Which conversation the words continue decides.** Until 13 November 2026 this was a ladder of phrase lists: openers that count as
continuations, verbs that count as commands, pronouns that count as address. Measured that day on 121 declared
utterances from eleven kinds of situation (tools/voice/measure_address_margin.py), it answered 46 of the 70 that were
not said to her: "can you see how it thickens?" from a cooking show, "can you pick up some bread?" into a phone,
"sweetie, can you put your shoes on?". It also stayed silent to "Hello. Is anyone there?". Bryan, the same day: "a
voice system needs to account for anything. so it cant just be 'Some types of speech get responses. others dont.'"
Now a reader model reads which talk the words continue: hers, the room's, or none (address_reading.py), and the lead
of hers is held to a margin measured on that reader.

**What the room's acoustics measure stays.** Speech much quieter than this speaker usually is was probably not aimed
here, and that is a measurement of the person, not a list. It vetoes only speech that does not continue an exchange
with her that is still open.

**When the words cannot be read** (her address reader not running, or no margin measured for the reader now running),
she answers what is plainly hers by what is left: the floor opened on purpose, her name heard, or a reply while the
floor is still open and no other voice is under it. That is the old wake word, kept as the fallback it is.

**Nothing here is a transcript filter.** This decides whether to answer, never whether to listen or what to hear. A
rejected utterance is still transcribed and shown, and it becomes part of the room's talk the reader reads the next
utterance against.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from core.voice.duplex.address_reading import AddressReading

logger = logging.getLogger("Aura.Voice.Addressivity")

# How long after she finishes speaking the exchange counts as still open: inside it, quiet speech is not vetoed for
# being quiet, and while the words cannot be read a reply with no other voice under it is taken as hers.
DEFAULT_OPEN_FLOOR_S = 18.0

# ASR confidence below this reads as "did not really hear that", which correlates with distance and with speech that
# was not aimed here. The recogniser does not report one today, so this applies only when one is reported.
MIN_COLD_OPEN_CONFIDENCE = 0.55

# Speech aimed at this machine is near-field. The comparison is against the speaker's *own* running loudness, never an
# absolute level, because an absolute threshold measures the microphone and the room rather than intent.
MIN_NEAR_FIELD_Z = -1.2


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().split())


@dataclass(slots=True)
class AddressContext:
    """Everything known about the moment this utterance arrived."""

    #: Seconds since she last finished speaking, or None if she never has.
    since_last_reply_s: float | None = None
    #: The recogniser's confidence in this transcript, 0..1, or None when the recogniser does not report one. None
    #: means the check that uses it is skipped rather than defaulted: a confidence of 1.0 invented to fill the field
    #: would be a fabricated reading that silently passes a real check.
    asr_confidence: float | None = None
    #: Loudness of this utterance relative to the speaker's own baseline, in standard deviations. None when the
    #: baseline has not settled yet.
    loudness_z: float | None = None
    #: True when another voice was audible under this one.
    competing_speech: bool = False
    #: How long the utterance was, in seconds.
    duration_s: float = 0.0
    #: Set when the user has explicitly opened the floor (pressed to talk, or entered focused voice mode). Then
    #: everything is for her, by definition.
    floor_explicitly_open: bool = False
    #: The reading of which talk the words continue (address_reading.py), or None when it could not be read.
    reading: AddressReading | None = None
    #: The lead her conversation must have over the room's talk and nothing, measured on the model that read it.
    margin: float | None = None


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
    reading: dict[str, object] | None = None
    at: float = field(default_factory=lambda: time.time())

    def __post_init__(self) -> None:
        # A one-element tuple written without its trailing comma is a string, and a string satisfies every type check
        # here while iterating as individual characters. Coercing at the boundary makes the mistake unwritable.
        self.reasons = _as_reasons(self.reasons)
        self.vetoes = _as_reasons(self.vetoes)

    def as_dict(self) -> dict[str, object]:
        return {
            "addressed": self.addressed,
            "rung": self.rung,
            "reasons": list(self.reasons),
            "vetoes": list(self.vetoes),
            "reading": self.reading,
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
        min_cold_open_confidence: float = MIN_COLD_OPEN_CONFIDENCE,
        min_near_field_z: float = MIN_NEAR_FIELD_Z,
    ) -> None:
        self._names = tuple(n for n in names if str(n).strip()) or ("aura",)
        self._open_floor_s = float(open_floor_s)
        self._min_confidence = float(min_cold_open_confidence)
        self._min_near_field_z = float(min_near_field_z)

    def _floor_open(self, context: AddressContext) -> bool:
        return context.since_last_reply_s is not None and context.since_last_reply_s <= self._open_floor_s

    def evaluate(self, transcript: str, context: AddressContext) -> AddressVerdict:
        text = _normalize(transcript)
        if not text:
            return AddressVerdict(False, "empty", vetoes=("nothing was said",))

        # The user opened the floor themselves. Push-to-talk and focused voice mode are explicit consent; nothing
        # below may override them, or the control would not be a control.
        if context.floor_explicitly_open:
            return AddressVerdict(True, "explicit", reasons=("the floor was opened deliberately",))

        if context.reading is not None and context.margin is not None:
            return self._by_her_reading(context.reading, context.margin, context)
        return self._unread(text, context)

    def _by_her_reading(self, reading: AddressReading, margin: float, context: AddressContext) -> AddressVerdict:
        measured = {**reading.as_dict(), "margin": round(margin, 3)}
        if reading.lead <= margin:
            return AddressVerdict(
                False,
                "read_elsewhere",
                vetoes=(
                    f"it reads as continuing {reading.rival} more than her conversation "
                    f"(lead {reading.lead:+.2f} per token against a margin of {margin:.2f})",
                ),
                reading=measured,
            )
        reasons = [f"it reads as said to her (lead {reading.lead:+.2f} per token over {reading.rival}, "
                   f"margin {margin:.2f})"]
        if not self._floor_open(context):
            veto = self._acoustic_veto(context)
            if veto:
                return AddressVerdict(False, "read_but_not_aimed_here", vetoes=(veto,), reading=measured)
        return AddressVerdict(True, "read", reasons=tuple(reasons), reading=measured)

    def _acoustic_veto(self, context: AddressContext) -> str:
        if context.asr_confidence is not None and context.asr_confidence < self._min_confidence:
            return (f"heard at {context.asr_confidence:.2f} confidence, which is how speech across a room sounds")
        if context.loudness_z is not None and context.loudness_z < self._min_near_field_z:
            return f"{context.loudness_z:.1f}σ quieter than this speaker normally is, so it was probably not aimed here"
        return ""

    def _unread(self, text: str, context: AddressContext) -> AddressVerdict:
        """Her model could not read the words: answer only what is plainly hers without it."""
        lowered = text.casefold()
        for name in self._names:
            if re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", lowered):
                return AddressVerdict(True, "named_unread",
                                      reasons=(f"her name was heard ({name}) and the rest could not be read",))
        if self._floor_open(context) and not context.competing_speech:
            return AddressVerdict(
                True,
                "open_floor_unread",
                reasons=(f"she spoke {context.since_last_reply_s:.0f}s ago and no other voice was under this",),
            )
        return AddressVerdict(False, "unread",
                              vetoes=("it could not be read, and it neither named her nor answered her",))


__all__ = [
    "DEFAULT_OPEN_FLOOR_S",
    "MIN_COLD_OPEN_CONFIDENCE",
    "MIN_NEAR_FIELD_Z",
    "AddressContext",
    "AddressVerdict",
    "AddressivityGate",
]

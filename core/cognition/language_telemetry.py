"""What the language did, on channels, so a claim about it can decay.

A claim graded "measured live" is only as good as the measurement still
happening. Naming the channels a claim rests on is what makes the grade fall
back to unmeasured when they go quiet, rather than standing on a measurement
that stopped — which is the difference between a record and a slogan.

These say how far the language has actually grown in this process, and how
much of that growth happened while somebody was waiting for an answer.
"""

from __future__ import annotations

import logging

from core.runtime.errors import record_degradation

__all__ = [
    "CHANNEL_MEANINGS",
    "CHANNEL_WAYS",
    "CHANNEL_WORDS",
    "declare_language_channels",
    "note_the_language_grew",
]

logger = logging.getLogger("Aura.LanguageTelemetry")

CHANNEL_WORDS = "language.words_derived"
CHANNEL_WAYS = "language.ways_of_building"
CHANNEL_MEANINGS = "language.meanings_reachable"

_declared = False


def declare_language_channels() -> bool:
    """Declare the channels once. Ids are a contract and are never reused."""
    global _declared

    if _declared:
        return True
    try:
        from core.fsw.telemetry_dictionary import ChannelType, channel

        channel(
            identifier=0x1501, name=CHANNEL_WORDS, type=ChannelType.INT,
            unit="count", owner="cognition.widening_the_language",
            description=(
                "Words she derived and admitted to the language she makes "
                "rules out of, in this process. Zero is the honest reading "
                "for a process where nothing needed a word it did not have."
            ),
            group="language",
        )
        channel(
            identifier=0x1502, name=CHANNEL_WAYS, type=ChannelType.INT,
            unit="count", owner="cognition.a_constructor_she_built",
            description=(
                "Ways of BUILDING words in force. Each one applies to every "
                "word she has and every word she derives afterwards, so this "
                "is growth in what the language can grow into."
            ),
            group="language",
        )
        channel(
            identifier=0x1503, name=CHANNEL_MEANINGS, type=ChannelType.INT,
            unit="count", owner="cognition.an_invented_kind",
            description=(
                "Distinct meanings reachable now, counted over a bounded "
                "witness. The number that says whether admitting something "
                "was growth or a second spelling."
            ),
            group="language",
        )
        _declared = True
        return True
    except (ImportError, ValueError, TypeError, KeyError) as exc:
        record_degradation(
            "language_telemetry", exc, severity="debug",
            action="the language channels were not declared, so a claim "
                   "resting on them reads as unmeasured",
        )
        return False


def note_the_language_grew() -> None:
    """Record where the language stands. Called wherever it changes."""
    if not declare_language_channels():
        return
    try:
        from core.cognition.an_invented_kind import (
            WAYS_TO_BUILD,
            WHAT_OF_IT,
            WHERE_FROM,
            addressings,
        )
        from core.cognition.what_it_costs_to_say import everything_sayable
        from core.cognition.widening_the_language import (
            DerivedAddressing,
            DerivedOperation,
        )
        from core.fsw.telemetry_dictionary import write

        derived = sum(
            1 for word in WHERE_FROM.values() if isinstance(word, DerivedAddressing)
        ) + sum(
            1 for word in WHAT_OF_IT.values() if isinstance(word, DerivedOperation)
        )
        write(CHANNEL_WORDS, derived)
        write(CHANNEL_WAYS, len(WAYS_TO_BUILD))
        write(CHANNEL_MEANINGS, len(everything_sayable()))
        logger.info(
            "the language now stands at %d derived word(s), %d way(s) of "
            "building, %d addressing(s)",
            derived,
            len(WAYS_TO_BUILD),
            len(addressings()),
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "language_telemetry", exc, severity="debug",
            action="the language grew without being recorded",
        )

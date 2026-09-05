"""core/voice/duplex/endpointing.py — "Are they done talking?"

A fixed silence timeout is the loudest tell that you are talking to a
machine. Set it short and it interrupts you the moment you pause to think;
set it long and every answer feels sluggish. Humans do neither — we decide
someone is finished mostly from *syntax and prosody*, and only fall back on
the clock when those are ambiguous.

So the required silence is a function of how finished the utterance sounds:

    "What time is it?"                    -> complete    ->  340 ms
    "I was thinking we could maybe"       -> incomplete  -> 1100 ms
    "so the thing is, um…"                -> thinking    -> 1500 ms

Everything here is deterministic text analysis over the ASR stable prefix.
It runs in microseconds and adds no model to the hot path.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from core.voice.duplex.config import EndpointConfig

if TYPE_CHECKING:
    from core.voice.duplex.acoustic_endpoint import TerminalityReading

logger = logging.getLogger("Aura.Voice.Endpoint")


class Completeness(Enum):
    COMPLETE = "complete"
    NEUTRAL = "neutral"
    INCOMPLETE = "incomplete"
    THINKING = "thinking"


# Words that cannot end an English sentence. If the last word is one of
# these, the speaker is mid-thought no matter how long the pause.
_DANGLING = frozenset(
    """
    and but or nor so because although though while whereas since unless until
    if when whenever whether that which who whom whose where why how
    a an the my your his her its our their this that these those
    of in on at to for with from by about into onto upon over under between
    among through during before after above below across behind beyond
    is are was were be been being am do does did have has had
    can could will would shall should may might must
    i you he she it we they there here
    very really quite pretty rather just also too either neither both
    like as than then plus versus toward towards without within
    maybe perhaps probably possibly definitely basically actually literally
    kind sort bit lot more most less least some any every
    """.split()
)

# Audible thinking. These mean "I am still composing" and deserve the
# longest patience of all — cutting someone off here is the rudest failure.
_FILLERS = frozenset(
    "um uh erm er hmm hm mmm mm like well so anyway okay ok yeah".split()
)

# Openers that make a short utterance a complete turn on their own.
_QUESTION_WORDS = frozenset(
    "what when where who whom whose why how which is are do does did can could "
    "will would should may might have has had am was were".split()
)

# Complete-in-themselves regardless of length.
_STANDALONE = frozenset(
    """
    yes no yeah yep nope nah sure okay ok right correct exactly stop wait
    hello hi hey thanks goodbye bye please continue go
    """.split()
)


@dataclass(slots=True)
class EndpointDecision:
    should_end: bool
    completeness: Completeness
    required_silence_ms: float
    observed_silence_ms: float
    reason: str


def classify(text: str) -> Completeness:
    """Judge how finished an utterance sounds from its text alone."""
    stripped = (text or "").strip()
    if not stripped:
        return Completeness.NEUTRAL

    words = re.split(r"\s+", stripped)
    last = re.sub(r"[^\w']", "", words[-1]).lower()
    first = re.sub(r"[^\w']", "", words[0]).lower()

    # Order matters, and it is not obvious. Each rule below can contradict
    # the next, so the sequence encodes which evidence outranks which.

    # 1. Whisper writes an ellipsis or dash when it hears trailing-off
    #    prosody. That is a direct acoustic reading of "not finished".
    if stripped.endswith(("...", "…", "-", "—")):
        return Completeness.THINKING

    # 2. A bare "yeah" / "okay" / "no" is a complete turn even though those
    #    same words are fillers mid-sentence. Length disambiguates.
    if len(words) == 1 and first in _STANDALONE:
        return Completeness.COMPLETE

    # 3. Trailing filler: "so I was thinking, um…" is unfinished even though
    #    "thinking" would be a fine place to stop.
    if last in _FILLERS:
        return Completeness.THINKING

    # 4. A question or exclamation mark is trustworthy: Whisper rarely
    #    invents one, and questions legitimately end on pronouns ("what
    #    time is it?"), so this must outrank the word-class check below.
    if stripped.endswith(("?", "!")):
        return Completeness.COMPLETE

    # 5. A word that cannot end a sentence beats a *period*, though.
    #    Whisper's full stops are a language-model prior, not an acoustic
    #    reading — it happily writes "So I was thinking that maybe." for
    #    someone who is plainly still mid-sentence. Trusting that period
    #    cuts the speaker off, which is the failure this whole module
    #    exists to prevent, so the dangling word wins.
    if last in _DANGLING:
        return Completeness.INCOMPLETE

    # 6. A period with a plausible final word: treat as finished.
    if stripped.endswith("."):
        return Completeness.COMPLETE

    if first in _QUESTION_WORDS and len(words) >= 3:
        return Completeness.COMPLETE

    # A long clause that does not dangle is usually a finished thought that
    # Whisper simply did not punctuate.
    if len(words) >= 6:
        return Completeness.COMPLETE

    return Completeness.NEUTRAL


class Endpointer:
    """Decides when the user's turn is over."""

    def __init__(self, config: EndpointConfig | None = None) -> None:
        self._config = config or EndpointConfig()

    def required_silence_ms(self, completeness: Completeness) -> float:
        cfg = self._config
        return {
            Completeness.COMPLETE: cfg.complete_silence_ms,
            Completeness.NEUTRAL: cfg.neutral_silence_ms,
            Completeness.INCOMPLETE: cfg.incomplete_silence_ms,
            Completeness.THINKING: cfg.thinking_silence_ms,
        }[completeness]

    def evaluate(
        self,
        *,
        transcript: str,
        silence_ms: float,
        speech_ms: float,
        min_utterance_ms: float,
        terminality: TerminalityReading | None = None,
    ) -> EndpointDecision:
        """Should the turn end now?

        ``terminality`` is the pitch contour at the end of the utterance, when
        one could be read. It can only ever *extend* the wait — see
        ``acoustic_endpoint.patience_multiplier`` — so a bad reading costs a
        beat of latency and can never cut somebody off. It is what separates
        "that sentence is over" from "that sentence has a full stop because
        Whisper likes full stops, and the speaker is drawing breath".
        """
        completeness = classify(transcript)
        required = self.required_silence_ms(completeness)

        if terminality is not None:
            from core.voice.duplex.acoustic_endpoint import patience_multiplier

            multiplier = patience_multiplier(
                terminality, completeness is Completeness.COMPLETE
            )
            if multiplier > 1.0:
                # Still bounded by the ceiling below, so patience cannot become
                # a hang: max_silence_ms always ends the turn.
                required = min(required * multiplier, self._config.max_silence_ms)

        # A cough or a door is not a turn. Without this the lane fires on
        # every non-speech transient that clears the VAD threshold.
        if speech_ms < min_utterance_ms:
            return EndpointDecision(
                should_end=False,
                completeness=completeness,
                required_silence_ms=required,
                observed_silence_ms=silence_ms,
                reason="below_min_utterance",
            )

        # The ceiling guarantees the turn always ends: if someone trails off
        # mid-sentence and never comes back, waiting forever is its own bug.
        if silence_ms >= self._config.max_silence_ms:
            return EndpointDecision(
                should_end=True,
                completeness=completeness,
                required_silence_ms=required,
                observed_silence_ms=silence_ms,
                reason="max_silence",
            )

        if silence_ms >= required:
            return EndpointDecision(
                should_end=True,
                completeness=completeness,
                required_silence_ms=required,
                observed_silence_ms=silence_ms,
                reason=f"silence_met_{completeness.value}",
            )

        return EndpointDecision(
            should_end=False,
            completeness=completeness,
            required_silence_ms=required,
            observed_silence_ms=silence_ms,
            reason="waiting",
        )

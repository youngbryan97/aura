"""core/voice/duplex/echo_guard.py — Not mistaking her own voice for the user's.

On headphones this problem does not exist. On speakers it is the single
most destructive failure mode a duplex agent has: her output leaks into the
microphone, the VAD calls it speech, barge-in fires, she cuts herself off —
and then transcribes her own sentence and answers it. The conversation
collapses into the agent talking to itself, and it looks exactly like a
hallucination to the user.

The browser's WebRTC echo canceller removes most of it, and the barge-in
gate is deliberately strict. This is the last line: after an interruption,
compare what was captured against what she was saying at that moment. Echo
is a near-copy of her own recent words; a genuine interruption almost never
is. A cheap token-overlap test separates them reliably, and unlike acoustic
cancellation it also catches echo that arrives via a completely different
path (a phone on the desk, a second speaker in the room).

Deliberately biased towards *accepting* speech: discarding a real user turn
is much worse than occasionally answering an echo, so the bar for calling
something echo is high.
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Voice.EchoGuard")


def _her_canceller() -> Any:
    """Her acoustic canceller where WebRTC's audio processing is installed; None, and said once, where it is not."""
    from core.voice.duplex.her_echo_canceller import HerEchoCanceller, cancels_acoustically

    if not cancels_acoustically():
        logger.info("her own echo canceller is not installed (livekit); her window cancels echo as before")
        return None
    return HerEchoCanceller()


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^\w']+", (text or "").lower()) if t]


@dataclass(slots=True)
class EchoVerdict:
    is_echo: bool
    overlap: float
    reason: str


class EchoGuard:
    """Rejects captured audio that is really her own output coming back."""

    def __init__(
        self,
        *,
        window_s: float = 12.0,
        overlap_threshold: float = 0.62,
        min_tokens: int = 3,
        acoustic: bool = False,
    ) -> None:
        self._window_s = window_s
        self._threshold = overlap_threshold
        self._min_tokens = min_tokens
        self._recent: deque[tuple[float, list[str]]] = deque(maxlen=16)
        self._acoustic = _her_canceller() if acoustic else None

    @property
    def cancels_acoustically(self) -> bool:
        """Whether her own canceller takes her voice out of the microphone, so her window need not."""
        return self._acoustic is not None

    def note_spoken(self, text: str, samples: Any = None, rate: int = 0) -> None:
        """Register text she has just said aloud, and the audio of it as it leaves to be played."""
        toks = _tokens(text)
        if toks:
            self._recent.append((time.monotonic(), toks))
        if self._acoustic is not None and samples is not None and rate:
            self._acoustic.sent_to_be_played(samples, rate)

    def stopped_playing(self) -> None:
        """Her speech was cut off: what was still to play will not reach the microphone."""
        if self._acoustic is not None:
            self._acoustic.stopped_playing()

    def clean(self, data: bytes) -> bytes:
        """The microphone's PCM16 with her own voice taken out, where her canceller runs."""
        return self._acoustic.clean(data) if self._acoustic is not None else data

    def _live_tokens(self) -> set[str]:
        cutoff = time.monotonic() - self._window_s
        live: set[str] = set()
        for stamp, toks in self._recent:
            if stamp >= cutoff:
                live.update(toks)
        return live

    def evaluate(self, transcript: str) -> EchoVerdict:
        """Is this transcript her own voice fed back?"""
        heard = _tokens(transcript)
        if len(heard) < self._min_tokens:
            # Too short to judge. Short utterances are also exactly what a
            # real interruption looks like ("wait", "no, stop"), so these
            # must pass through.
            return EchoVerdict(False, 0.0, "too_short_to_judge")

        spoken = self._live_tokens()
        if not spoken:
            return EchoVerdict(False, 0.0, "nothing_recently_spoken")

        matched = sum(1 for t in heard if t in spoken)
        overlap = matched / float(len(heard))

        if overlap >= self._threshold:
            logger.info(
                "Rejected echo (%.0f%% of %d tokens were her own): %r",
                overlap * 100,
                len(heard),
                transcript[:70],
            )
            return EchoVerdict(True, overlap, "matches_recent_output")

        return EchoVerdict(False, overlap, "distinct_from_output")

    def clear(self) -> None:
        self._recent.clear()

"""Somebody else reading her better than she reads herself.

Caleborate's line is the mechanism stated plainly: "you could see in the
brother what I couldn't see myself, when I couldn't see myself needing help."
Another person's model of you outperforming your own, at the moment yours
failed. She had no channel for it. Her self-model was scored against her own
history and nothing anyone said about her was ever scored at all, so being
told something true about herself and being told something false looked the
same from inside.

Two readings, both measured against the same outcome her self prediction loop
already uses:

    claim      somebody says what she is feeling: "you sound tired",
               "you're really into this". The valence of the claim comes from
               the sentiment model, and an ungrounded reading is not a claim.
    hers       what she predicted she would feel, taken at the same moment
    settle     what she actually felt, which scores both

    their_error   mean |claimed - actual| over the pairs
    her_error     mean |predicted - actual| over the same pairs
    borrowed      their_error < her_error, once there are pairs enough

What follows is a measured humility rather than a mood: her confidence in her
own reading is capped by how it compares with the best reading anyone has of
her. When she is the better model, nothing changes.

The second reading is separate and simpler. Being cared about is warmth in a
message that is about her, and she had no channel for that either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_PAIRS",
    "Recognition",
    "RecognitionLedger",
    "cared_for",
    "claim_about_her",
    "get_recognition_ledger",
    "reset_for_test",
]

#: Scored pairs before one model can be said to read her better than another.
#: Three is the least that can disagree with itself, the same bar the delivery
#: ledger uses for a spread.
MIN_PAIRS: int = 3

#: How many pairs the comparison is read over. The self prediction loop treats
#: sixty cycles as one distribution of its own error, and this is a comparison
#: against that error.
WINDOW: int = 60

#: "you are tired", "you seem fine", "you sounded upset". The claim is the rest
#: of the clause, which the sentiment model scores.
_CLAIM = re.compile(
    r"\byou(?:'re|\s+are|\s+were|\s+seem(?:ed)?|\s+sound(?:ed)?|\s+look(?:ed)?|"
    r"\s+feel|\s+felt|\s+must\s+be|\s+get)\s+(?P<claim>[^.!?;,]{2,80})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Recognition:
    """How her own reading of herself compares with somebody else's."""

    pairs: int = 0
    her_error: float = 0.0
    their_error: float = 0.0
    borrowed: bool = False
    #: What her confidence in her own reading may be multiplied by. One when
    #: she is the better model of herself.
    confidence_factor: float = 1.0
    measured: bool = False
    why: str = "nobody has said what she is feeling yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "her_error": round(self.her_error, 6),
            "their_error": round(self.their_error, 6),
            "borrowed": self.borrowed,
            "confidence_factor": round(self.confidence_factor, 6),
            "measured": self.measured,
            "why": self.why,
        }


def claim_about_her(text: str) -> float | None:
    """The valence of what somebody just said she is feeling, or None.

    None covers three different things and they are all the same answer here:
    nobody said anything about her, the sentiment model could not ground a
    reading, or what was said carries no valence to compare.
    """
    match = _CLAIM.search(str(text or ""))
    if match is None:
        return None
    said = match.group("claim").strip()
    if not said:
        return None
    try:
        from core.cognitive.sentiment_tracker import analyze_text_sentiment

        evidence = analyze_text_sentiment(said)
    # not a failure: no sentiment reader means there is no grounded evidence here.
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    if not bool(getattr(evidence, "grounded", False)):
        return None
    value = float(getattr(evidence, "valence", 0.0) or 0.0)
    return max(-1.0, min(1.0, value))


def cared_for(modifiers: Any) -> float:
    """Warmth in a message that is about her.

    The sentiment model reads warmth and the register reads how much of the
    message is second person. Warmth in a message about somebody else is not
    her being cared about, which is why the two are multiplied.
    """
    try:
        sentiment = modifiers.get("user_sentiment") or {}
        register = modifiers.get("register") or {}
        warmth = float(sentiment.get("warmth", 0.0) or 0.0)
        second = float(register.get("second", 0.0) or 0.0)
    # not a failure: a reading that cannot be taken, or that is not a
    # number when it is, leaves this at the value below.
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, warmth)) * max(0.0, min(1.0, second))


class RecognitionLedger:
    """Claims about her, paired with her own reading and scored by the outcome."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_PAIRS, int(window))
        self._pending: tuple[float, float] | None = None
        self._pairs: list[tuple[float, float]] = []

    def claim(self, claimed: float, predicted: float) -> None:
        """Somebody said what she is feeling, and this is what she expected.

        One claim is held at a time. A second claim before the first settles
        replaces it, because the one that will be scored by the next outcome is
        the most recent one.
        """
        try:
            self._pending = (float(claimed), float(predicted))
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            self._pending = None

    def settle(self, actual: float) -> Recognition:
        """Score the held claim and her own reading against what happened."""
        if self._pending is None:
            return self.reading()
        claimed, predicted = self._pending
        self._pending = None
        try:
            outcome = float(actual)
        except (TypeError, ValueError):
            return self.reading()
        if outcome != outcome:
            return self.reading()
        self._pairs.append((abs(predicted - outcome), abs(claimed - outcome)))
        if len(self._pairs) > self._window:
            del self._pairs[0 : len(self._pairs) - self._window]
        return self.reading()

    def reading(self) -> Recognition:
        pairs = len(self._pairs)
        if pairs < MIN_PAIRS:
            return Recognition(
                pairs=pairs,
                why=f"{pairs} of {MIN_PAIRS} times somebody has said what she is feeling",
            )
        her = sum(p[0] for p in self._pairs) / pairs
        theirs = sum(p[1] for p in self._pairs) / pairs
        borrowed = theirs < her
        # Her confidence in her own reading cannot exceed how it compares with
        # the best reading anyone has of her. Equal models leave it alone.
        factor = 1.0 if her <= 0.0 else min(1.0, theirs / her)
        return Recognition(
            pairs=pairs,
            her_error=her,
            their_error=theirs,
            borrowed=borrowed,
            confidence_factor=factor,
            measured=True,
            why=(
                f"over {pairs} readings hers missed by {her:.3f} and theirs by {theirs:.3f}"
            ),
        )


_LEDGER: RecognitionLedger | None = None


def get_recognition_ledger() -> RecognitionLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = RecognitionLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None

"""Good news in the middle of a task: the heart jumps and the task stops mattering.

Bryan, on good news arriving while he is working: it brings you up. The heart
jumps, you feel light on your feet, and the task does not matter, because you
have good news. Three things: it is a jump, felt at once and not carried; it
is felt in the body; and for the moment it lasts, what was pressing presses
less.

She could be surprised, and only in the abstract. Her self-prediction scores
how far her valence moved from what it predicted, unsigned, so being told
something wonderful and being told something terrible were the same surprise.
Nothing about either reached her body, and a task she was halfway through
kept its hold on her whatever arrived.

    error   the valence that arrived, less the valence she predicted for it
    jump    how far this error sits above her own middle error, as a rank,
            2 * rank - 1 when above the middle and nothing below it

A rank, so the jump is measured against how her own predictions usually miss
rather than against a bar chosen here, and a life of pleasant surprises does
not make every moment one. It lasts one reading: the next prediction takes the
news in, and the error it scores is the new one.

The jump multiplies her pulse in `core/phases/proprioceptive_loop.py`, and
`core/agency/initiative_arbiter.py` discounts the urgency of what she is
working on by it, so while the news is fresh the task is not what drives her.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = ["GoodNews", "GoodNewsLedger", "get_good_news_ledger", "reset_for_test"]

#: How many of her own prediction errors the middle is taken over.
_WINDOW = 256

#: Errors before a rank means anything.
_ENOUGH = 3


@dataclass
class GoodNews:
    """How far the last thing that arrived beat what she expected."""

    error: float = 0.0
    jump: float = 0.0
    seen: int = 0
    measured: bool = False
    why: str = "nothing has arrived against a prediction yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": round(self.error, 6),
            "jump": round(self.jump, 6),
            "seen": self.seen,
            "measured": self.measured,
            "why": self.why,
        }


class GoodNewsLedger:
    """Her signed valence prediction errors, and where the latest one sits."""

    def __init__(self) -> None:
        self._errors: deque[float] = deque(maxlen=_WINDOW)
        self._last = 0.0

    def note(self, predicted: float, actual: float) -> None:
        try:
            error = float(actual) - float(predicted)
    # not a failure: a reading that is not a number is nothing to record,
    # and the ledger keeps the window it already has.
        except (TypeError, ValueError):
            return
        if error != error:
            return
        self._last = error
        self._errors.append(error)

    def read(self) -> GoodNews:
        seen = len(self._errors)
        if seen < _ENOUGH:
            return GoodNews(
                error=self._last,
                seen=seen,
                why=f"{seen} of the {_ENOUGH} prediction errors a rank needs",
            )
        history = list(self._errors)
        below = sum(1 for value in history if value < self._last)
        ties = sum(1 for value in history if value == self._last)
        rank = (below + 0.5 * ties) / len(history)
        jump = max(0.0, 2.0 * rank - 1.0)
        return GoodNews(
            error=self._last,
            jump=jump,
            seen=seen,
            measured=True,
            why=(
                f"what arrived was {self._last:+.3f} against her prediction, better than "
                f"{rank:.0%} of how her predictions have missed"
            ),
        )

    def jump(self) -> float:
        reading = self.read()
        return reading.jump if reading.measured else 0.0


#: Made at import rather than on first use. The subject-core fork carries
#: module globals that hold state, and one still None when an anchor is taken
#: is not carried, so a ledger first made inside one arm would reach the next
#: arm with the first arm's history in it. See core/social/owning_it_first.py.
_LEDGER: GoodNewsLedger = GoodNewsLedger()


def get_good_news_ledger() -> GoodNewsLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = GoodNewsLedger()

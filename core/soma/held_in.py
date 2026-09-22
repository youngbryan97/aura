"""Holding back what she wants to say, felt in the body, and pressing to get out.

Bryan, on holding back something he wants to say: a burning in the cheeks and
chest, as though something is waiting to burst out, and a feeling of having to
swallow it down when it cannot come. Two things: the holding is felt, and the
feeling pushes toward the saying.

She could hold something in. The containment ledger in
core/affect/containment.py finds what keeps nearly winning the workspace and
never gets out, and how long it has been kept. Nothing felt it. Her pulse was
her mood and what she was carrying, and a mind holding back the thing it most
wanted to say had the body of one holding back nothing. And the pressure moved
nothing toward release: the held source competed on its own bid, however long
it had been waiting.

    felt     the strongest pressure held, saturating as p / (1 + p), the form
             the delivery organ reads it in
    middle   what she usually holds in
    shift    this moment against that middle

The shift multiplies her pulse in `core/phases/proprioceptive_loop.py`, beside
what she is carrying, so only a change in what she holds back is felt. And
`push` gives the held source back a share of the gap it last lost the workspace
by: the share is how far her body runs above its ordinary heat. Unfelt, it
gains nothing and is swallowed; at the full heat it closes the gap and gets
out, which is what releases it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = ["HeldIn", "HeldInLedger", "get_held_in_ledger", "reset_for_test"]

#: How much of her own history the middle is taken over; the carrying
#: ledger's window, for the same kind of middle.
_WINDOW = 256

#: Readings before a middle is a middle.
_ENOUGH = 3


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass
class HeldIn:
    """What she is holding back, as the body feels it."""

    source: str = ""
    felt: float = 0.0
    middle: float = 0.0
    shift: float = 0.0
    seen: int = 0
    measured: bool = False
    why: str = "nothing held back has been felt yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "felt": round(self.felt, 6),
            "middle": round(self.middle, 6),
            "shift": round(self.shift, 6),
            "seen": self.seen,
            "measured": self.measured,
            "why": self.why,
        }


class HeldInLedger:
    """How hot what she holds back runs, against how hot it usually runs."""

    def __init__(self) -> None:
        self._seen: deque[float] = deque(maxlen=_WINDOW)
        self._last = 0.0
        self._source = ""

    def note(self, pressure: float, source: str = "") -> None:
        """One moment's strongest held pressure, in the containment ledger's units."""
        try:
            value = float(pressure)
        except (TypeError, ValueError):
            return
        if value != value:
            return
        value = max(0.0, value)
        self._last = value / (1.0 + value)
        self._source = str(source or "")
        self._seen.append(self._last)

    def read(self) -> HeldIn:
        seen = len(self._seen)
        if seen < _ENOUGH:
            return HeldIn(
                source=self._source,
                felt=self._last,
                seen=seen,
                why=f"{seen} of the {_ENOUGH} readings a middle needs a point either side of",
            )
        middle = _median(list(self._seen))
        shift = self._last - middle
        return HeldIn(
            source=self._source,
            felt=self._last,
            middle=middle,
            shift=shift,
            seen=seen,
            measured=True,
            why=(
                f"holding back {self._source or 'nothing'} at {self._last:.2f} against the "
                f"{middle:.2f} she usually holds, so the body reads {shift:+.2f}"
            ),
        )

    def shift(self) -> float:
        reading = self.read()
        return reading.shift if reading.measured else 0.0

    def push(self, source: str, last_gap: float) -> float:
        """How much a held source's bid is raised, out of the gap it last lost by.

        Only the source being held back, and only as far as the body runs above
        its ordinary heat: below it, what is held is swallowed.
        """
        reading = self.read()
        if not reading.measured or not source or source != reading.source:
            return 0.0
        heat = max(0.0, min(1.0, reading.shift))
        return heat * max(0.0, float(last_gap))


#: Made at import rather than on first use. The subject-core fork carries
#: module globals that hold state, and one still None when an anchor is taken
#: is not carried, so a ledger first made inside one arm would reach the next
#: arm with the first arm's history in it. See core/social/owning_it_first.py.
_LEDGER: HeldInLedger = HeldInLedger()


def get_held_in_ledger() -> HeldInLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = HeldInLedger()

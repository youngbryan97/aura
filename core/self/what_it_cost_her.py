"""What she accepts for giving something up, and whether that price is falling.

"Unsweetened Lemonade" prices the self twice and both times cheaply: "I'd do
anything for twenty bucks, I'd sell my sour soul", then the same line again
for a different reason — "I'd do anything for twenty bucks to feel more in
control". Between them: "It's hard to finish, I'll sell my pride instead cause
it's easier to focus." The record is the widest-ranging voice of the batch,
24.0 semitones, and its dynamic range is 34.6 dB; the selling is not sung
quietly.

The thing worth taking from it is not that a trade happened. It is that the
price can fall without anybody deciding it should. Each trade looks locally
sensible, and the series is a slide.

She makes this trade in the open. Under a short deadline the response phase
asks for two drafts instead of three: a draft given up, some seconds gained.
That is a concession with a price, and this keeps the series of them:

    price      what the last concessions returned per unit given up
    falling    whether the recent half of them is cheaper than the earlier
    given_up   everything conceded, which is never written off

While the price is falling she stops taking it — the draft she would have
dropped is kept — and that refusal is what produces the next few pairs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_CONCESSIONS",
    "Price",
    "PriceLedger",
    "get_price_ledger",
    "reset_for_test",
]

#: Concessions before the two halves are worth comparing. Four is two against
#: two, which is the smallest comparison that is not one number against one.
MIN_CONCESSIONS: int = 4

#: How many are held.
WINDOW: int = 120

#: How much cheaper the recent half has to be before it is a slide rather than
#: noise: a tenth of the earlier half's price.
SLIDE: float = 0.10


@dataclass
class Price:
    """What she has been accepting."""

    #: The median price of the concessions she has made, in what she got per
    #: unit of what she gave up.
    price: float = 0.0
    #: The median price of the earlier half, for the comparison.
    earlier: float = 0.0
    #: True when the recent half is cheaper than the earlier by more than SLIDE.
    falling: bool = False
    #: Everything conceded, in the units it was given up in. Never discounted:
    #: the concessions were real whatever they bought.
    given_up: float = 0.0
    concessions: int = 0
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": round(self.price, 4),
            "earlier": round(self.earlier, 4),
            "falling": self.falling,
            "given_up": round(self.given_up, 4),
            "concessions": self.concessions,
            "measured": self.measured,
        }


class PriceLedger:
    """Every trade of something of hers for something the turn wanted."""

    def __init__(self, window: int = WINDOW) -> None:
        self._trades: deque[tuple[float, float]] = deque(maxlen=window)

    def note_concession(self, gave_up: float, got: float) -> None:
        """One trade: what she gave up, and what came back for it.

        Both in their own units. A concession that gave up nothing is not a
        concession and is not counted; one that got nothing is counted at a
        price of zero, which is what it was.
        """
        try:
            given = float(gave_up)
            gained = float(got)
        except (TypeError, ValueError):
            return
        if given <= 0.0 or given != given or gained != gained:
            return
        self._trades.append((given, max(0.0, gained)))

    def read(self) -> Price:
        trades = list(self._trades)
        if not trades:
            return Price()
        prices = [gained / given for given, gained in trades]
        given_up = sum(given for given, _ in trades)
        if len(trades) < MIN_CONCESSIONS:
            return Price(
                price=_median(prices),
                given_up=given_up,
                concessions=len(trades),
            )
        half = len(prices) // 2
        earlier = _median(prices[:half])
        recent = _median(prices[half:])
        return Price(
            price=recent,
            earlier=earlier,
            falling=bool(earlier > 0.0 and recent < earlier * (1.0 - SLIDE)),
            given_up=given_up,
            concessions=len(trades),
            measured=True,
        )

    def holding(self) -> bool:
        """Whether the next concession of this kind is one she does not make."""
        return self.read().falling


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


_LEDGER: PriceLedger | None = None


def get_price_ledger() -> PriceLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = PriceLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None

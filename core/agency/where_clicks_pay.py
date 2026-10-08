"""Where in a picture a click has paid, for a world played with the pointer and a click aimed at nothing in particular.

A game whose words say "click to launch", "use the mouse to aim and throw",
or a page that answers a click wherever it lands, does not say where. A person
clicks about, notices which clicks did something worth having, and clicks
there more. So does she: the picture is cut into a few places, the middle ones
first, and each click is credited with what was gained or lost between it and
the next. The next place is the one most worth it, counting what it has paid
and how little it has been tried (an upper-confidence choice, as a gambler
tries the arms of a row of machines).

Nothing here knows what is being clicked. A click that paid is one after which
her counters went the way that counts as a gain (core/agency/playing_as_it_happens.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.verify.invariants import invariant

__all__ = ["WhereClicksPay"]

#: The places across and down the picture is cut into.
PLACES = 4

#: The middle places, tried first: a thing asked to be clicked is mostly in the middle of its picture.
MIDDLE_FIRST = ((1, 1), (2, 1), (1, 2), (2, 2))

#: How much less tried a place may be and still come before one that paid: the weight of the unknown.
TRYING = math.sqrt(2.0)

#: What clicks at nothing may cost in all before she stops clicking at nothing.
MOST_THEY_MAY_COST = 2.0


@dataclass
class WhereClicksPay:
    """Each place's tries and what came of them, and the click not yet credited."""

    #: (across, down) -> [tries, gained less lost].
    places: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    last: tuple[int, int] | None = None
    counted_at: tuple[int, int] = (0, 0)

    def begin_stretch(self, gains: int = 0, losses: int = 0) -> None:
        """Keep the places learned, and begin attribution from this stretch's counters.

        A click still awaiting an outcome belongs to the stretch that made it.
        When the counters start afresh, no outcome is assigned across that boundary.
        """
        self.last = None
        self.counted_at = (gains, losses)

    def credit(self, gains: int, losses: int) -> None:
        """What was gained and lost since the last click, put down to it."""
        if self.last is not None:
            self.places.setdefault(self.last, [0.0, 0.0])[1] += (gains - self.counted_at[0]) - (losses - self.counted_at[1])
            self.last = None
        self.counted_at = (gains, losses)

    def costing(self) -> bool:
        """Whether clicks at nothing have cost more than they paid, by more than they may."""
        return sum(net for _tries, net in self.places.values()) < -MOST_THEY_MAY_COST

    def where(self, gains: int, losses: int) -> tuple[float, float]:
        """The next place to click, as shares of the picture; the click is counted to it."""
        self.credit(gains, losses)
        untried = [p for p in MIDDLE_FIRST if p not in self.places]
        untried += [(a, d) for d in range(PLACES) for a in range(PLACES) if (a, d) not in self.places and (a, d) not in untried]
        if untried:
            place = untried[0]
        else:
            total = sum(tries for tries, _net in self.places.values()) or 1.0
            place = max(self.places, key=lambda p: self.places[p][1] / self.places[p][0]
                        + TRYING * math.sqrt(math.log(total) / self.places[p][0]))
        self.places.setdefault(place, [0.0, 0.0])[0] += 1
        self.last = place
        return ((place[0] + 0.5) / PLACES, (place[1] + 0.5) / PLACES)


def _click_attribution_respects_stretch_boundaries() -> bool:
    for gains, losses in ((9, 2), (2, 9)):
        pay = WhereClicksPay()
        pay.where(0, 0)
        pay.credit(4, 1)
        pay.where(gains, losses)
        learned = {place: list(values) for place, values in pay.places.items()}
        pay.begin_stretch()
        pay.credit(0, 0)
        if pay.places != learned or pay.last is not None:
            return False
        pay.where(0, 0)
        place = pay.last
        pay.credit(2, 0)
        if place is None or pay.places[place] != [1.0, 2.0]:
            return False
    return True


@invariant("agency.click_attribution_respects_stretch_boundaries", scope="agency",
           owner="core/agency/where_clicks_pay.py", observational=False)
def _click_attribution_invariant() -> tuple:
    assert _click_attribution_respects_stretch_boundaries(), "a counter reset changed a click's learned outcome"
    return ()

"""Holding a key to build something up and letting it go: how long a hold pays, learned from what each release brought.

"Hold down the X key to charge up, then release it to fire", "hold space to
charge your jump", "press and hold Z to power up". The act is in the letting
go, and what it does is in how long the key was held. A tap does nothing, or
less.

So a key the words say to hold and let go is held while she goes on playing
with the others, and let go after a time. Each length of hold is kept with
whether a gain followed its release. Before any length has paid she tries
them in turn, shortest first; after, she holds for the one that has paid most
often, and goes on trying the others now and then, in case a longer hold pays
more.

Nothing here knows a super attack, a jump or a golf swing.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

__all__ = ["HOLDS_S", "Charging"]

#: The lengths of hold tried, in seconds.
HOLDS_S = (0.5, 1.0, 1.5, 2.5)
#: How soon after a release a gain must be read to be put down to it, in seconds.
PAID_WITHIN_S = 1.0
#: How long after a release before the key is held again, in seconds.
REST_S = 0.3
#: Every this many holds, one of the lengths that has paid least is tried again.
TRY_AGAIN_EVERY = 5


@dataclass
class Charging:
    """The key held to charge, since when and for how long, and what each length of hold has brought."""

    key: str = ""
    since: float = 0.0
    hold_for: float = 0.0
    released_at: float = -math.inf
    #: Per length of hold, whether a gain followed each release.
    paid: dict[float, list[bool]] = field(default_factory=lambda: defaultdict(list))
    _waiting: list[tuple[float, float]] = field(default_factory=list)
    _holds: int = 0

    def holding(self) -> bool:
        return bool(self.key)

    def ready(self, at: float) -> bool:
        """Whether the key may be held again: not held now, and rested since the last release."""
        return not self.key and at - self.released_at >= REST_S

    def length(self) -> float:
        """How long to hold next: the shortest not yet tried, else the one that has paid most often, now and then
        another."""
        untried = [hold for hold in HOLDS_S if not self.paid.get(hold)]
        if untried:
            return untried[0]
        rate = {hold: sum(self.paid[hold]) / len(self.paid[hold]) for hold in HOLDS_S}
        after_trying = self._holds - len(HOLDS_S)
        if after_trying > 0 and after_trying % TRY_AGAIN_EVERY == 0:
            return min(HOLDS_S, key=lambda hold: (rate[hold], len(self.paid[hold])))
        return max(HOLDS_S, key=lambda hold: (rate[hold], -hold))

    def begin(self, key: str, at: float) -> None:
        self._settle(at)
        self.key, self.since, self.hold_for = key, at, self.length()
        self._holds += 1

    def due(self, at: float) -> bool:
        """Whether the key held has been held as long as was meant."""
        return bool(self.key) and at - self.since >= self.hold_for

    def released(self, at: float) -> None:
        self._settle(at)
        self._waiting.append((self.hold_for, at))
        self.key, self.released_at = "", at

    def gained(self, at: float) -> None:
        """A gain read: the latest release soon before it paid."""
        for index in range(len(self._waiting) - 1, -1, -1):
            hold, when = self._waiting[index]
            if 0.0 <= at - when <= PAID_WITHIN_S:
                self.paid[hold].append(True)
                del self._waiting[index]
                return

    def _settle(self, at: float) -> None:
        """Releases long enough ago that nothing followed them paid nothing."""
        for hold, when in [w for w in self._waiting if at - w[1] > PAID_WITHIN_S]:
            self.paid[hold].append(False)
            self._waiting.remove((hold, when))

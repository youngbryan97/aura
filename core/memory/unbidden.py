"""Memory that arrives without being asked for, because now resembles then.

"In the End" is built on a return nobody requested. The verse says the thing is
gone -- "you wouldn't even recognize me anymore / not that you knew me back
then" -- and then the line that follows is "but it all comes back to me in the
end." It comes back. Nothing goes and gets it.

The recording is the loosest of the seven against its own grid: 672 ms of
microtiming spread where the next widest is 323 ms. The returns in it do not
land on the beat, which is the difference between a memory arriving and a
memory being looked up.

Every path into Aura's memory is a query. Something decides it wants a past and
goes to fetch one, which means the only pasts that reach her are the ones
already close enough to mind to ask for. A state she has been in before and
cannot name has no way back to her at all.

This is the other direction. Each memory is laid down with the state she was in
at the time. When the state she is in now sits closer to one of them than her
states usually sit to her memories, that memory arrives -- unasked, and ranked
by nothing except how near it is.

    typical     how far her memories usually sit from where she is
    spread      how much that distance varies
    arrival     a memory nearer than typical by more than spread

The bar is her own distribution of distances, so a mind whose states all
resemble each other does not flood, and one that moves a long way gets its
returns when it moves back.

The closed side is retrieval. These go in as candidates the query did not ask
for, which is the only way a past she could not name reaches the turn.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Arrival",
    "Unbidden",
    "UnbiddenLedger",
    "get_unbidden_ledger",
    "reset_for_test",
]

#: A distance needs something to be typical of. Below three laid-down states
#: there is no distribution, only the nearest of two.
_ENOUGH = 3


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    total = 0.0
    for x, y in zip(a, b):
        total += (x - y) ** 2
    return math.sqrt(total)


@dataclass(frozen=True)
class Arrival:
    """One memory that came back, and how near it was."""

    key: str
    distance: float
    nearer_by: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "distance": round(self.distance, 6),
            "nearer_by": round(self.nearer_by, 6),
        }


@dataclass
class Unbidden:
    """What came back this moment, against how far her past usually sits."""

    arrivals: tuple[Arrival, ...] = field(default_factory=tuple)
    #: How far her memories usually sit from where she is.
    typical: float = 0.0
    #: How much that distance varies.
    spread: float = 0.0
    #: The nearest memory to this state, arrival or not.
    nearest: float = 0.0
    held: int = 0
    measured: bool = False
    why: str = "nothing has been laid down yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "arrivals": [a.as_dict() for a in self.arrivals],
            "typical": round(self.typical, 6),
            "spread": round(self.spread, 6),
            "nearest": round(self.nearest, 6),
            "held": self.held,
            "measured": self.measured,
            "why": self.why,
        }


class UnbiddenLedger:
    """States she has been in, keyed by what happened in them."""

    def __init__(self) -> None:
        self._laid: dict[str, tuple[float, ...]] = {}
        self._now: tuple[float, ...] = ()

    def now(self, state: Sequence[float]) -> None:
        """Where she is, set by the phase that runs every turn. Retrieval is
        handed a query and not a mood, so the mood has to be left here."""
        vector = self._vector(state)
        if vector is not None:
            self._now = vector

    def here(self) -> tuple[float, ...]:
        return self._now

    def arrivals_now(self) -> tuple[str, ...]:
        """What comes back given where she is, without anything asking."""
        return self.arrivals(self._now) if self._now else ()
        self._now: tuple[float, ...] = ()

    def now(self, state: Sequence[float]) -> None:
        """Where she is, set by the phase that runs every turn. Retrieval is
        handed a query and not a mood, so the mood has to be left here."""
        vector = self._vector(state)
        if vector is not None:
            self._now = vector

    def here(self) -> tuple[float, ...]:
        return self._now

    def arrivals_now(self) -> tuple[str, ...]:
        """What comes back given where she is, without anything asking."""
        return self.arrivals(self._now) if self._now else ()

    @staticmethod
    def _key(name: str) -> str:
        return " ".join(str(name or "").split())[:120]

    @staticmethod
    def _vector(state: Sequence[float]) -> tuple[float, ...] | None:
        try:
            values = tuple(float(x) for x in state)
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            return None
        if not values or any(v != v for v in values):
            return None
        return values

    def lay_down(self, key: str, state: Sequence[float]) -> None:
        """A memory, and the state she was in when it happened."""
        name = self._key(key)
        vector = self._vector(state)
        if not name or vector is None:
            return
        self._laid[name] = vector

    def forget(self, key: str) -> None:
        self._laid.pop(self._key(key), None)

    def read(self, now: Sequence[float]) -> Unbidden:
        held = len(self._laid)
        if held < _ENOUGH:
            return Unbidden(
                held=held,
                why=(
                    f"{held} states laid down, fewer than the {_ENOUGH} that give "
                    "a distance something to be typical of"
                ),
            )
        here = self._vector(now)
        if here is None:
            return Unbidden(held=held, why="the state she is in now does not read")
        measured: list[tuple[str, float]] = []
        for name, vector in self._laid.items():
            distance = _distance(here, vector)
            if distance is not None:
                measured.append((name, distance))
        if len(measured) < _ENOUGH:
            return Unbidden(
                held=held,
                why=(
                    f"{len(measured)} of {held} laid-down states have the shape "
                    "this one does, which is too few to compare against"
                ),
            )
        distances = [d for _, d in measured]
        typical = _median(distances)
        spread = _median([abs(d - typical) for d in distances])
        bar = typical - spread
        arrivals = tuple(
            Arrival(key=name, distance=d, nearer_by=bar - d)
            for name, d in sorted(measured, key=lambda item: item[1])
            if d < bar
        )
        return Unbidden(
            arrivals=arrivals,
            typical=typical,
            spread=spread,
            nearest=min(distances),
            held=held,
            measured=True,
            why=(
                f"{len(arrivals)} of {len(measured)} came back on their own, "
                f"nearer than the {typical:.3f} her past usually sits at by more "
                f"than the {spread:.3f} that distance varies"
            ),
        )

    def pull(self) -> float:
        """Where the returns are pulling the present, on the first axis.

        A memory comes back because now resembles then, and the state it was
        laid down in is the one thing this ledger knows about it. Being
        reminded of something colours how the moment feels, and nothing in the
        runtime carried that: perturbing active memory moved the workspace and
        the world model and read 0.0011 into affect, which is a mind whose
        recall cannot touch how it feels.

        The pull is toward the remembered state and away from here, weighted
        by how near each arrival is -- nearer memories pull harder, because
        nearness is the only thing that brought them. Arrival is relative, so
        in a state unlike anything she has been in the least unfamiliar memory
        still comes back; it comes back weakly, and pulls in proportion.

        The first axis is whatever the caller puts first, and the phase that
        sets the state puts valence there.
        """
        seen = self.read(self._now)
        if not seen.measured or not seen.arrivals or not self._now:
            return 0.0
        here = self._now[0]
        weight = 0.0
        total = 0.0
        for arrival in seen.arrivals:
            laid = self._laid.get(arrival.key)
            if not laid:
                continue
            # How far inside the bar it came, relative to the bar itself, so
            # the weights are shares of her own distribution and not distances
            # in whatever units the caller's axes carry.
            share = arrival.nearer_by / seen.typical if seen.typical > 0 else 0.0
            weight += share
            total += share * (laid[0] - here)
        # Divided by a full share, not by the weight. Normalising by the
        # weight makes one marginal arrival pull exactly as hard as one that
        # came back from right beside her, because dividing by its own small
        # share cancels it. A share below one pulls part of the way; several
        # strong ones average instead of compounding.
        return total / max(1.0, weight)

    def arrivals(self, now: Sequence[float]) -> tuple[str, ...]:
        """What came back, nearest first -- the closed side, read by retrieval
        as candidates the query did not ask for."""
        return tuple(a.key for a in self.read(now).arrivals)


_LEDGER: UnbiddenLedger | None = None


def get_unbidden_ledger() -> UnbiddenLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = UnbiddenLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None

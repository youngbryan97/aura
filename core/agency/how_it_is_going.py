"""What she is working toward, how far along she is, and whether it is going as it should.

A goal that only exists as a finishing condition is a goal she can only fail
or pass. Everything between — that she is a third of the way, that the last
two steps took thirty moves each and this one has taken ninety, that the
approach she adopted for this step has not moved her since she adopted it — is
the part somebody working on something actually reasons about, and the part
that tells her when to change what she is doing rather than keep pushing.

So this holds three things, and nothing else:

    the rungs she has passed   what she has reached on the way, each with the
                               move it was reached on and what it cost
    the rung she is on         projected from the ones behind it, because the
                               pattern of what a world gives back is a fact
                               about the world she can read off her own record
    how it is going            what a rung usually costs here, how long this
                               one is taking, and whether that is beyond what
                               her own history says to expect

None of it knows what any world is. A rung is a number she reached, a cost is
moves, and the projection is whichever of "times as much" or "more than" fits
the rungs she has actually passed. A board that doubles, a download that goes
by tens of megabytes, a form filled a field at a time and a route walked in
junctions all leave the same shape of record.

What it is for is the deciding. ``behind`` is the question a held approach
should be reconsidered on: not "have twelve moves gone by", which is a clock,
but "has this stopped working", which is the thing. And it survives a run, so
a second attempt starts knowing what the first reached and what that took.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import median
from typing import Any

logger = logging.getLogger("Aura.HowItIsGoing")

__all__ = ["HowItIsGoing", "Rung"]

#: How many rungs make a pattern. Two give a step or a ratio; three say
#: whether it held. Below this she has no projection and says so.
ENOUGH_TO_SEE_A_PATTERN = 2

#: How close two ratios have to be to count as the same ratio. A world that
#: doubles does not double exactly when a rung is missed, and a tenth is
#: nearer than any other pattern would be.
NEARLY_THE_SAME = 0.1

#: How far past what a rung usually costs counts as behind, as a share of
#: that cost. Her own spread where she has one; this is what stands in until
#: two rungs have been passed and the spread is a measurement.
HALF_AS_LONG_AGAIN = 0.5


@dataclass
class Rung:
    """One thing reached on the way, and what it cost to reach it."""

    reached: float
    at_move: int
    took: int
    #: The line she was holding when she reached it, where she was holding one.
    holding: str = ""

    def as_memory(self) -> dict[str, Any]:
        return {
            "reached": float(self.reached),
            "at_move": int(self.at_move),
            "took": int(self.took),
            "holding": str(self.holding),
        }


@dataclass
class HowItIsGoing:
    """The goal, the rungs behind her, and whether this one is going as it should."""

    #: What finishing looks like, where it is a number. Zero when the request
    #: named no number and the place said none.
    toward: float = 0.0
    #: The furthest she has got here, across attempts.
    best: float = 0.0
    rungs: list[Rung] = field(default_factory=list)
    #: The move she was on when the last rung was passed.
    last_rung_at: int = 0
    #: What she is holding while she works on this one.
    holding: str = ""
    #: How many times an approach was changed because this said so.
    reassessed: int = 0

    # ── watching it happen ───────────────────────────────────────────────

    def noticed(self, reached: float, at_move: int, *, holding: str = "") -> bool:
        """Take in what she has reached. True when that is a new rung."""
        self.holding = str(holding or self.holding)
        value = float(reached or 0.0)
        if value <= self.best:
            return False
        took = max(0, int(at_move) - int(self.last_rung_at))
        self.rungs.append(Rung(reached=value, at_move=int(at_move), took=took, holding=self.holding))
        self.best = value
        self.last_rung_at = int(at_move)
        logger.info(
            "a rung: %g after %d move(s)%s",
            value,
            took,
            f", holding {self.holding!r}" if self.holding else "",
        )
        return True

    # ── where it is going ────────────────────────────────────────────────

    def next_rung(self) -> float:
        """What she is working toward now, projected from what she has passed.

        Whichever of "times as much" or "more than" describes the rungs behind
        her. Neither, and the finish itself is the next thing — which is also
        what a first rung has to be.
        """
        reached = [rung.reached for rung in self.rungs]
        if len(reached) >= ENOUGH_TO_SEE_A_PATTERN:
            ratios = [
                later / earlier
                for earlier, later in zip(reached, reached[1:], strict=False)
                if earlier > 0
            ]
            steps = [later - earlier for earlier, later in zip(reached, reached[1:], strict=False)]
            if ratios and _all_nearly(ratios):
                projected = self.best * median(ratios)
            elif steps and _all_nearly(steps):
                projected = self.best + median(steps)
            else:
                projected = 0.0
            if projected > self.best and (not self.toward or projected <= self.toward):
                return projected
        return self.toward if self.toward > self.best else 0.0

    def usually_takes(self) -> float:
        """How many moves a rung costs here, from the ones she has paid for.

        Where the cost of a rung grows — and it does wherever a rung is worth
        twice the last one — the next one is projected by the same growth
        rather than by the middle of what is behind it, which would always be
        an underestimate.
        """
        costs = [float(rung.took) for rung in self.rungs if rung.took > 0]
        if not costs:
            return 0.0
        if len(costs) >= ENOUGH_TO_SEE_A_PATTERN + 1:
            growth = [
                later / earlier
                for earlier, later in zip(costs, costs[1:], strict=False)
                if earlier > 0
            ]
            if growth and _all_nearly(growth):
                return max(costs[-1] * median(growth), 1.0)
        return max(median(costs), 1.0)

    def how_long_this_one_has_taken(self, at_move: int) -> int:
        return max(0, int(at_move) - int(self.last_rung_at))

    def behind(self, at_move: int) -> bool:
        """Whether this rung is taking longer than her own record says to expect."""
        usual = self.usually_takes()
        if usual <= 0.0:
            return False
        costs = [float(rung.took) for rung in self.rungs if rung.took > 0]
        spread = (max(costs) / max(1.0, median(costs)) - 1.0) if len(costs) > 1 else HALF_AS_LONG_AGAIN
        allowed = usual * (1.0 + max(HALF_AS_LONG_AGAIN, min(2.0, spread)))
        return self.how_long_this_one_has_taken(at_move) > allowed

    # ── saying so ────────────────────────────────────────────────────────

    def where_it_stands(self, at_move: int) -> str:
        """How it is going, for whoever is watching and for her own deciding."""
        if not self.rungs and not self.best:
            return "nothing reached here yet"
        said = [f"{self.best:g} so far"]
        ahead = self.next_rung()
        if ahead:
            said.append(f"working on {ahead:g}")
        if self.toward and self.toward > self.best:
            said.append(f"{len(self.rungs)} rung(s) toward {self.toward:g}")
        usual = self.usually_takes()
        if usual > 0.0:
            said.append(
                f"{self.how_long_this_one_has_taken(at_move)} move(s) on this one, "
                f"about {usual:.0f} is usual"
            )
        return ", ".join(said)

    def why_reassess(self, at_move: int) -> str:
        """Why the approach should be looked at again, in her words. Empty when it should not."""
        if not self.behind(at_move):
            return ""
        return (
            f"{self.how_long_this_one_has_taken(at_move)} move(s) without reaching "
            f"{self.next_rung():g}, where {self.usually_takes():.0f} is usual here"
        )

    def it_was_reassessed(self) -> None:
        """Count a change of approach, and start this rung's clock from here.

        Without this the same stall asks for a new approach on every move
        after it: the rung has still not been reached, so it is still behind,
        so it asks again. What a change of approach earns is the time to work.
        """
        self.reassessed += 1
        self.last_rung_at = self.last_rung_at  # kept, so the rung's true cost is still true

    # ── keeping it ───────────────────────────────────────────────────────

    def as_memory(self) -> dict[str, Any]:
        return {
            "toward": float(self.toward),
            "best": float(self.best),
            "rungs": [rung.as_memory() for rung in self.rungs][-24:],
            "last_rung_at": int(self.last_rung_at),
            "holding": str(self.holding),
        }

    @classmethod
    def from_memory(cls, held: Any, *, toward: float = 0.0) -> "HowItIsGoing":
        """What she knew about this before, with the finish she was given now."""
        if not isinstance(held, dict):
            return cls(toward=float(toward or 0.0))
        rungs = []
        for one in held.get("rungs") or ():
            if not isinstance(one, dict):
                continue
            try:
                rungs.append(
                    Rung(
                        reached=float(one.get("reached") or 0.0),
                        at_move=int(one.get("at_move") or 0),
                        took=int(one.get("took") or 0),
                        holding=str(one.get("holding") or ""),
                    )
                )
            except (TypeError, ValueError):
                continue
        return cls(
            toward=float(toward or held.get("toward") or 0.0),
            best=float(held.get("best") or 0.0),
            rungs=rungs,
            # A new attempt starts its own clock; what the rungs cost stays.
            last_rung_at=0,
            holding=str(held.get("holding") or ""),
        )


def _all_nearly(values: list[float]) -> bool:
    """Whether these are all the same number, near enough to call it a pattern."""
    kept = [value for value in values if value > 0]
    if len(kept) < ENOUGH_TO_SEE_A_PATTERN - 1 or not kept:
        return False
    middle = median(kept)
    if middle <= 0:
        return False
    return all(abs(value - middle) <= NEARLY_THE_SAME * middle for value in kept)

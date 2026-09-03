"""What she has already started, and has not landed yet.

A Stellaris player, paused, with three fleets on screen and every one of them
reading "Estimated Arrival Date". None of them is anywhere yet. All three are
facts about the world they are planning in, and the plan is made against a
board that includes them — because sending a fourth fleet to a place three are
already arriving at is not caution, it is doing the thing twice.

She acts as though the world is what she can see. Everything she has started
and not finished — a build running, a message sent and unanswered, a page
loading, a task handed to something else — is invisible to the next decision,
so she can start it again, or plan around a gap something is already on its
way to fill, or give up on a thing that was about to arrive.

Three things follow from writing them down and none of them is clever. What
the world will be is what it is now plus what is already coming. Something
already on its way is not worth starting again. And a thing that should have
landed and has not is worth knowing about — that is not the same as a thing
still in flight, and telling them apart is the difference between waiting and
being stuck.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["OnItsWay", "WhatIsComing"]


@dataclass(frozen=True)
class OnItsWay:
    """One thing she started that has not landed."""

    what: str
    started_at: float
    lands_at: float
    #: What it will have done once it lands, in whatever terms the caller uses.
    brings: str = ""

    def late_by(self, now: float) -> float:
        return max(0.0, float(now) - self.lands_at)

    def describe(self, now: float) -> str:
        if self.late_by(now) > 0:
            return f"{self.what} should have landed {self.late_by(now):.0f}s ago"
        return f"{self.what} lands in {self.lands_at - now:.0f}s"


@dataclass
class WhatIsComing:
    """Everything she has set in motion and not seen the end of."""

    on_the_way: list[OnItsWay] = field(default_factory=list)

    def she_started(
        self, what: str, *, at: float, lands_at: float, brings: str = ""
    ) -> None:
        self.on_the_way.append(
            OnItsWay(what=str(what), started_at=float(at), lands_at=float(lands_at), brings=str(brings))
        )

    def it_landed(self, what: str) -> None:
        self.on_the_way = [one for one in self.on_the_way if one.what != what]

    def already_coming(self, what: str) -> bool:
        """Whether this is already on its way.

        Starting it again is not caution. It is doing the thing twice, and
        where the thing has an effect that is worse than wasting the time.
        """
        return any(one.what == what for one in self.on_the_way)

    def what_it_will_be(self, now_is: Iterable[str]) -> tuple[str, ...]:
        """The world as it will be: what is there, plus what is coming.

        This is what a plan should be made against. A gap something is already
        on its way to fill is not a gap.
        """
        coming = [one.brings for one in self.on_the_way if one.brings]
        return tuple(list(now_is) + coming)

    def overdue(self, now: float) -> tuple[OnItsWay, ...]:
        """Things that should have landed and have not.

        Not the same as things still in flight, and telling the two apart is
        the difference between waiting and being stuck. Nothing here decides
        what to do about it — being late is a fact, and what it means depends
        on what it was.
        """
        return tuple(one for one in self.on_the_way if one.late_by(now) > 0)

    def still_coming(self, now: float) -> tuple[OnItsWay, ...]:
        return tuple(one for one in self.on_the_way if one.late_by(now) <= 0)

    def describe(self, now: float) -> str:
        if not self.on_the_way:
            return "nothing in flight"
        late = self.overdue(now)
        said = f"{len(self.on_the_way)} in flight"
        return f"{said}, {len(late)} overdue" if late else said

    def as_memory(self) -> dict[str, Any]:
        return {
            "on_the_way": [
                {
                    "what": one.what,
                    "started_at": one.started_at,
                    "lands_at": one.lands_at,
                    "brings": one.brings,
                }
                for one in self.on_the_way
            ]
        }

    @classmethod
    def from_memory(cls, held: Any) -> "WhatIsComing":
        if not isinstance(held, dict):
            return cls()
        coming: list[OnItsWay] = []
        for one in held.get("on_the_way") or ():
            if not isinstance(one, dict):
                continue
            try:
                coming.append(
                    OnItsWay(
                        what=str(one.get("what") or ""),
                        started_at=float(one.get("started_at") or 0.0),
                        lands_at=float(one.get("lands_at") or 0.0),
                        brings=str(one.get("brings") or ""),
                    )
                )
            except (TypeError, ValueError):
                # not a failure: a time that is not a number is not a time.
                continue
        return cls(on_the_way=[one for one in coming if one.what])

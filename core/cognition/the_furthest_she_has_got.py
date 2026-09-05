"""Keeping the part that worked, and varying only where it stopped working.

Two recordings of somebody playing games that are famous for being unfair —
Ghosts 'n Goblins and Ninja Gaiden. The counter in the corner of the second
says P-6: this is his sixth go. Neither player is reacting his way through.
Both are replaying something they already know up to a point, and thinking
only at the place it went wrong last time — and when that place is passed, the
part they know grows by a few seconds and the thinking moves forward to the
next one.

Nothing about that is games. It is how anybody learns a route, a recipe, a
build that falls over at step nine, a form that rejects the fifth field. The
attempt is not repeated from curiosity. It is repeated because most of it was
right and the cost of re-deriving the right part is the whole reason a hard
thing feels impossible.

She does not do this. Every turn is decided afresh, so a sequence that was
nine tenths correct is worth nothing on the next attempt, and the tenth part
is re-approached with no more information than the first time.

Three things are wanted and they are all bookkeeping. The longest run of acts
that has ever led somewhere is worth replaying rather than rethinking. The
place it stopped is the only place worth thinking about. And what was tried
there and did not work is not worth trying there again — which is the
difference between practising and repeating yourself.

The frontier moves on its own. Get past it once, and the prefix is longer and
the thinking has somewhere new to be.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TheFurthestSheHasGot"]


@dataclass
class TheFurthestSheHasGot:
    """What has worked so far, and where it stopped working."""

    #: The longest run of acts that has led somewhere without failing.
    #: Replayed rather than rethought.
    got_through: tuple[str, ...] = ()
    #: What has been tried at the place it stops, and come to nothing.
    ruled_out_here: dict[int, set[str]] = field(default_factory=dict)
    attempts: int = 0

    @property
    def frontier(self) -> int:
        """How many acts in the thinking starts."""
        return len(self.got_through)

    def replay(self) -> tuple[str, ...]:
        """The part she does not have to think about."""
        return self.got_through

    def already_failed_here(self, at: int | None = None) -> frozenset[str]:
        """What has been tried at that place and led nowhere."""
        return frozenset(self.ruled_out_here.get(self.frontier if at is None else at, ()))

    def worth_trying_at_the_frontier(self, acts: Iterable[str]) -> list[str]:
        """The acts not yet ruled out where the thinking is.

        Empty is a real answer: everything available at that point has been
        tried and none of it works, so the place to think about is not this
        one. Something earlier has to change, and saying so is the finding.
        """
        failed = self.already_failed_here()
        return [one for one in acts if one not in failed]

    def stuck_at_the_frontier(self, acts: Sequence[str]) -> bool:
        """Whether everything possible here has been tried and failed."""
        return bool(acts) and not self.worth_trying_at_the_frontier(acts)

    def an_attempt_ended(self, took: Sequence[str], *, got_to: int) -> bool:
        """Take one attempt: the acts, and how many of them led somewhere.

        ``got_to`` is how far it worked, which the caller knows and this
        cannot. Returns whether the frontier moved, because that is the thing
        worth saying out loud — everything else is another go.
        """
        self.attempts += 1
        reached = max(0, min(int(got_to), len(took)))
        moved = reached > self.frontier
        if reached < len(took):
            # The act at the frontier is the one that did not work. Written
            # down against the place rather than against the act, because the
            # same act is right somewhere else and this is not a judgement
            # about the act.
            self.ruled_out_here.setdefault(reached, set()).add(str(took[reached]))
        if moved:
            self.got_through = tuple(str(one) for one in took[:reached])
            # What was ruled out further along was ruled out after a different
            # run-up, so it is no longer about this situation. Kept where it
            # still applies and dropped where it does not — otherwise she
            # inherits refusals from a route she is no longer taking.
            self.ruled_out_here = {
                where: what
                for where, what in self.ruled_out_here.items()
                if where <= reached
            }
        return moved

    def describe(self, acts: Sequence[str] = ()) -> str:
        if not self.attempts:
            return "nothing tried yet"
        if acts and self.stuck_at_the_frontier(acts):
            return (
                f"{self.frontier} in, and everything available there has failed — "
                "something earlier has to change"
            )
        return (
            f"{self.frontier} of them known, thinking at {self.frontier}, "
            f"{len(self.already_failed_here())} ruled out there, "
            f"{self.attempts} attempt(s)"
        )

    def as_memory(self) -> dict[str, Any]:
        return {
            "got_through": list(self.got_through),
            "ruled_out_here": {
                str(where): sorted(what) for where, what in self.ruled_out_here.items()
            },
            "attempts": self.attempts,
        }

    @classmethod
    def from_memory(cls, held: Any) -> TheFurthestSheHasGot:
        """What she got through last time. A route keeps between sittings."""
        if not isinstance(held, dict):
            return cls()
        ruled: dict[int, set[str]] = {}
        for where, what in (held.get("ruled_out_here") or {}).items():
            try:
                at = int(where)
            except (TypeError, ValueError):
                # not a failure: a place that is not a number is not a place.
                continue
            ruled[at] = {str(one) for one in (what or ())}
        try:
            attempts = int(held.get("attempts") or 0)
        except (TypeError, ValueError):
            attempts = 0
        return cls(
            got_through=tuple(str(one) for one in (held.get("got_through") or ())),
            ruled_out_here=ruled,
            attempts=max(0, attempts),
        )

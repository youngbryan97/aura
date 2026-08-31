"""How much either side can still do, and preferring that theirs be less.

Watching somebody beat a hard checkers engine: they finished five pieces to
eight, having given away three, with every piece on their back row untouched
since the first move. That is a losing position by every count of material,
and it was won. The other side's men could only move forward, forward was
their back row, their back row was full — so the other side ran out of legal
moves and forfeited.

Nothing about that is about checkers. They were not taking the other side's
pieces, they were taking the other side's OPTIONS, and they spent pieces to do
it because pieces were not what they had decided to win with. Meanwhile they
kept one piece that could go anywhere, so that their own options never ran
out. Win by having something to do when the other has nothing.

This is empowerment, in the sense Klyubin and Polani gave it: how much of its
own future a thing can still reach through its own acts. The usual use is to
keep your own high, which is where "keep your options open" comes from. The
other half is the interesting one here — driving somebody else's toward zero
is a goal in itself, reachable, and worth wanting even while losing on every
other count.

The measure is a count of the distinct places each side can still bring about.
Channel capacity is the logarithm of that under a flat distribution, so the
count orders states the same way and does not invent a precision the estimate
has not got.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "WhatIsStillOpen",
    "acts_by_what_they_leave_open",
    "what_is_still_open",
]


@dataclass(frozen=True)
class WhatIsStillOpen:
    """How many distinct places each side can still bring about."""

    hers: int
    theirs: int
    ahead: int

    @property
    def nothing_left_for_them(self) -> bool:
        """They cannot act and she can, which is a win without a capture."""
        return self.theirs == 0 and self.hers > 0

    @property
    def nothing_left_for_her(self) -> bool:
        return self.hers == 0 and self.theirs > 0

    @property
    def between(self) -> int:
        """How far ahead she is on room to move, which is what she is playing for."""
        return self.hers - self.theirs

    def describe(self) -> str:
        if self.nothing_left_for_them:
            return f"she has {self.hers} things she could do and they have none"
        return f"{self.hers} open to her, {self.theirs} to them, {self.ahead} ahead"


def _how_far_it_reaches(
    state: Any,
    acts: Iterable[Any],
    step: Callable[[Any, Any], Any | None],
    named: Callable[[Any], Hashable],
    ahead: int,
) -> int:
    """Count the distinct places these acts can bring about within ``ahead``.

    Distinct is what matters and not the number of acts: two acts that leave
    the world in the same place are one option, and a side with twenty moves
    that all lead to the same thing is not free.
    """
    here = {named(state): state}
    reached: set[Hashable] = set()
    for _ in range(max(1, ahead)):
        next_up: dict[Hashable, Any] = {}
        for was in here.values():
            for act in acts:
                got = step(was, act)
                if got is None:
                    continue
                key = named(got)
                reached.add(key)
                next_up.setdefault(key, got)
        if not next_up:
            break
        here = next_up
    return len(reached)


def what_is_still_open(
    state: Any,
    *,
    acts: Sequence[Any],
    step: Callable[[Any, Any], Any | None],
    their_acts: Sequence[Any] | None = None,
    their_step: Callable[[Any, Any], Any | None] | None = None,
    named: Callable[[Any], Hashable] = repr,
    ahead: int = 1,
) -> WhatIsStillOpen:
    """What each side could still bring about from here.

    ``step(state, act)`` returns where that act leaves things, or None when it
    is not allowed. When there is nobody on the other side the second count is
    zero, and ``nothing_left_for_them`` is then meaningless rather than a win,
    which is why it also asks that she has something.
    """
    hers = _how_far_it_reaches(state, acts, step, named, ahead)
    theirs = 0
    if their_acts is not None:
        theirs = _how_far_it_reaches(
            state, their_acts, their_step or step, named, ahead
        )
    return WhatIsStillOpen(hers=hers, theirs=theirs, ahead=ahead)


def acts_by_what_they_leave_open(
    state: Any,
    *,
    acts: Sequence[Any],
    step: Callable[[Any, Any], Any | None],
    their_acts: Sequence[Any] | None = None,
    their_step: Callable[[Any, Any], Any | None] | None = None,
    named: Callable[[Any], Hashable] = repr,
    ahead: int = 1,
    keeps: Callable[[Any], bool] | None = None,
) -> list[tuple[Any, WhatIsStillOpen]]:
    """Her acts, the one that leaves them least room first.

    ``keeps`` is something she has decided to hold true. Anything that breaks
    it goes last however good it looks, which is the whole of what a sacrifice
    is: a move is not bad because it loses something, it is bad because it
    loses the thing she is winning with, and those are different questions.
    """
    weighed: list[tuple[Any, WhatIsStillOpen, bool]] = []
    for act in acts:
        after = step(state, act)
        if after is None:
            continue
        held = True if keeps is None else bool(keeps(after))
        weighed.append(
            (
                act,
                what_is_still_open(
                    after,
                    acts=acts,
                    step=step,
                    their_acts=their_acts,
                    their_step=their_step,
                    named=named,
                    ahead=ahead,
                ),
                held,
            )
        )
    weighed.sort(key=lambda one: (not one[2], one[1].theirs, -one[1].hers))
    return [(act, open_) for act, open_, _ in weighed]

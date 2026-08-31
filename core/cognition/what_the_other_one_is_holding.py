"""Working out what somebody else is keeping true, and taking it away.

Everything else she has is about her own plan. This is about theirs.

There are two ways to be ahead of somebody. Advance what you are doing, or
stop what they are doing, and the second is usually cheaper because a plan
that has been broken has to be rebuilt from the start. Nimzowitsch called
playing against the other side's intentions prophylaxis and thought it was
most of positional play; in the recorded checkers game it looks like a near
back row that was never moved, which stops the other side promoting anything
without capturing a single piece.

Working out what they are holding needs no new machinery. It is the same
question she asks about herself — which property of a position predicts things
going well — asked with THEIR outcomes instead of hers. A thing that can model
its own commitments can model somebody else's with the same organ, and that is
worth more than a special one, because it improves whenever the first does.

Breaking it for one turn is worth nothing if they can put it back. So an act
only counts as taking it away when they cannot restore it on their reply.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.cognition.something_she_keeps_true import (
    SomethingTrue,
    the_one_worth_holding,
)

__all__ = [
    "WhatItTakesFromThem",
    "acts_that_take_it_away",
    "what_the_other_one_is_holding",
]


@dataclass(frozen=True)
class WhatItTakesFromThem:
    """What one of her acts does to what they are keeping true."""

    act: Any
    breaks_it: bool
    they_can_put_it_back: bool

    @property
    def takes_it_away(self) -> bool:
        return self.breaks_it and not self.they_can_put_it_back

    def describe(self) -> str:
        if not self.breaks_it:
            return "leaves what they are holding alone"
        if self.they_can_put_it_back:
            return "breaks it, and they put it straight back"
        return "takes it away and they cannot get it back"


def what_the_other_one_is_holding(
    watched: Sequence[tuple[Any, bool]], *, deepest: int = 2
) -> SomethingTrue | None:
    """The property that predicts things going well FOR THEM.

    ``watched`` is positions and whether the other side did well from them,
    which she can read off her own losses without being told anything. None
    when nothing she has seen predicts it — and that is worth having, because
    it says the other side is not holding anything she can play against, and
    she should get on with her own plan instead.
    """
    return the_one_worth_holding(watched, deepest=deepest)


def acts_that_take_it_away(
    state: Any,
    *,
    acts: Sequence[Any],
    step: Callable[[Any, Any], Any | None],
    theirs: SomethingTrue,
    their_acts: Sequence[Any] = (),
    their_step: Callable[[Any, Any], Any | None] | None = None,
    keeps: Callable[[Any], bool] | None = None,
) -> list[WhatItTakesFromThem]:
    """Her acts, the ones that take away what they are holding first.

    ``keeps`` is what SHE has decided to hold. Playing against their plan is
    not worth giving up her own for, so anything that breaks hers sorts below
    everything that does not — the same rule as everywhere else, because a
    move that wins the argument and loses the game is not a good move.
    """
    weighed: list[tuple[WhatItTakesFromThem, bool]] = []
    for act in acts:
        after = step(state, act)
        if after is None:
            continue
        try:
            standing = bool(theirs.holds(after))
        except (ArithmeticError, AttributeError, TypeError, ValueError):
            continue
        restored = False
        if not standing and their_acts and their_step is not None:
            for their_act in their_acts:
                back = their_step(after, their_act)
                if back is None:
                    continue
                try:
                    if theirs.holds(back):
                        restored = True
                        break
                except (ArithmeticError, AttributeError, TypeError, ValueError):
                    # not a failure: a place the property cannot be read of
                    # is not a place it was restored in.
                    continue
        held = True if keeps is None else bool(keeps(after))
        weighed.append(
            (
                WhatItTakesFromThem(
                    act=act,
                    breaks_it=not standing,
                    they_can_put_it_back=restored,
                ),
                held,
            )
        )
    weighed.sort(key=lambda one: (not one[1], not one[0].takes_it_away, not one[0].breaks_it))
    return [one for one, _ in weighed]

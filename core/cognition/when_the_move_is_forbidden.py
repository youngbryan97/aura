"""Wanting a move she may not make yet, and buying it somewhere else.

Go has a rule that you may not immediately recreate the position that was just
there. It exists to stop two players taking the same stone back and forth for
ever, and what it produces is one of the strangest and most useful shapes in
any game: when you cannot make the move you want, you go and make a DIFFERENT
move somewhere else that the other player has to answer — and while they are
answering it, the position moves on, and the move you wanted becomes legal.

Players call the elsewhere-move a threat, and choosing one is a real skill.
It has to be big enough that ignoring it costs more than the thing being
fought over, and no bigger, because a threat that is worth more than the fight
is a thing you should simply be doing.

Nothing of hers had any of this. Blocked meant blocked: an act refused was an
act crossed off, and she went and did the next best thing in the same place.
The idea that the way to get a move is to spend a turn making the world need
something else first was not expressible.

This is every situation where the thing she wants is unavailable NOW and would
be available after something moves: a lock held by somebody who will let go when their own work is
unblocked, a rate limit that clears, a person who will say yes after they have
been given something else to think about. The general shape is: find what is
in the way, find something that obliges it to change, spend a turn there, come
back.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["AWayRound", "a_way_round"]


@dataclass(frozen=True)
class AWayRound:
    """A move she cannot make, and what to do instead to get it.

    ``spend_a_turn_on`` is the thing to do now. ``worth`` is what ignoring it
    would cost them, and it is the number that makes a threat a threat.
    """

    wanted: Any
    spend_a_turn_on: Any
    worth: float
    then_it_is_free: bool

    @property
    def found(self) -> bool:
        return self.spend_a_turn_on is not None

    def describe(self) -> str:
        if not self.found:
            return "nothing she can do elsewhere makes it available"
        held = " and then it is free" if self.then_it_is_free else ""
        return f"do {self.spend_a_turn_on} first (worth {self.worth:.2f}){held}"


def a_way_round(
    wanted: Any,
    *,
    allowed: Callable[[Any], bool],
    elsewhere: Sequence[Any],
    they_must_answer: Callable[[Any], float],
    after_they_answer: Callable[[Any], Any],
    worth_of_the_fight: float,
) -> AWayRound:
    """Something to do elsewhere that makes the move she wants available.

    ``they_must_answer`` says what ignoring that act would cost them, which is
    what makes it a threat rather than a move. ``after_they_answer`` gives the
    world once they have, so that whether the wanted move is then allowed can
    be checked rather than hoped.

    Of the threats that work, the SMALLEST that is still big enough. A threat
    worth more than the thing being fought over is not a threat, it is a move
    she should be making anyway — and spending it here throws away the
    difference.
    """
    if allowed(wanted):
        return AWayRound(wanted, wanted, 0.0, True)
    good: list[tuple[float, Any, bool]] = []
    for one in elsewhere:
        try:
            costs = float(they_must_answer(one))
        except (ArithmeticError, TypeError, ValueError):
            # not a failure: an act whose weight cannot be read is not one she
            # can weigh, and guessing at it is how a threat gets bluffed.
            continue
        if costs < worth_of_the_fight:
            # Ignorable. They will simply take the thing instead.
            continue
        try:
            free = bool(allowed(wanted) or _allowed_after(wanted, one, allowed, after_they_answer))
        except (ArithmeticError, AttributeError, TypeError, ValueError):
            free = False
        good.append((costs, one, free))
    works = [one for one in good if one[2]]
    if not works:
        return AWayRound(wanted, None, 0.0, False)
    costs, act, free = min(works, key=lambda one: (one[0], repr(one[1])))
    return AWayRound(wanted=wanted, spend_a_turn_on=act, worth=costs, then_it_is_free=free)


def _allowed_after(
    wanted: Any,
    doing: Any,
    allowed: Callable[[Any], bool],
    after_they_answer: Callable[[Any], Any],
) -> bool:
    """Whether the wanted move is available once they have answered this."""
    world = after_they_answer(doing)
    if world is None:
        return False
    asks = getattr(world, "allows", None)
    if callable(asks):
        return bool(asks(wanted))
    return bool(allowed(wanted))

"""What can still change: which things in a world can never move again, where each of the rest could ever go, and
so whether two sides could ever meet.

A search judges a situation by looking ahead, and where the change that matters lies past every look it misjudges it:
a position may stand a piece up and be drawn, because nothing either side can do will ever bring a piece of one within
reach of a piece of the other. A person sees that without looking ahead at all, from what cannot move and where the
rest can go (suggestions of 23 November 2026, item 18: a plan reasoner that proves what cannot change, by reachability
over the moves, not by search).

Two questions, both over any world of things that move from place to place:

* ``staying``: the things that can never move again. A thing can move only where a place it could step to is free
  of things that stay; so the things that stay are the largest set in which each has nowhere to step but places held
  by others in the set. It is found by starting from every thing and letting go of each that has somewhere to step,
  until none does.
* ``could_reach``: every place a thing could ever stand, from where it is, stepping as it can, with the things that
  stay as walls and every other thing as if it might be gone. That is more than it will reach, never less; so what it
  cannot reach, it never will.

What these prove is proved for every future, and holds only as far as the steps given are all the steps there are.
Nothing here knows a game: a place is anything hashable, a step any function from a place to the places beyond it.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Iterable

__all__ = ["could_reach", "staying"]


def staying(things: Iterable[Hashable], has_somewhere_to_step: Callable[[Hashable, frozenset], bool]) -> frozenset:
    """The things that can never move again: the largest set in which no thing has somewhere to step while the rest
    of the set stays where it is (``has_somewhere_to_step(thing, staying)``)."""
    held = set(things)
    while True:
        free = {thing for thing in held if has_somewhere_to_step(thing, frozenset(held))}
        if not free:
            return frozenset(held)
        held -= free


def could_reach(start: Hashable, onward: Callable[[Hashable], Iterable[Hashable]]) -> frozenset:
    """Every place reachable from ``start`` by steps (``onward(place)``: the places one step goes to), ``start``
    included."""
    seen = {start}
    waiting = deque([start])
    while waiting:
        for there in onward(waiting.popleft()):
            if there not in seen:
                seen.add(there)
                waiting.append(there)
    return frozenset(seen)

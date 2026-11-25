"""Who can wait longer: each side's moves that change nothing that matters, counted, so that the side with fewer is
the one that will have to give something up.

Bryan, 24 November 2026, of a board game against the hardest computer opponent he plays: his pieces held the places
where theirs would grow stronger, so nothing of theirs could, and three of his moved freely behind their lines; theirs,
which only go one way, jammed in between. "Forcing an opponent into an unwinnable position ... uses the rules and physics of the game/environment
against an opponent until theyre in a position they physically cannot escape from or that forces them to address
whatever it is you're doing." What decided it was not what either side could take but who could keep moving without
changing anything: his free pieces could go back and forth for ever, and every move of theirs used up a step of a way
that only goes one way.

That is a fact about any world where some things can only go one way: a forward-only runner, a resource spent and
never regained, a fuse. Two counts, from how each mover goes (its reach) and nothing else:

- A mover with two free places it can go between and back has waiting moves without end (``ENDLESS``).
- A mover that cannot come back has as many as the longest way forward it can still take, with what stands in its way
  where it stands now.

``spare_moves`` gives each side's count, ENDLESS where any mover of it can shuttle; ``who_waits_longer`` says which
side can outlast the other, None where both can wait for ever. Nothing here knows a game, and nothing weighs it: what
outlasting is worth is what a world's own games teach a judgement that counts it.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence

from core.cognition.where_forces_can_meet import Mover, Reach

__all__ = ["ENDLESS", "spare_moves", "who_waits_longer"]

#: Waiting moves without end: a mover that can go back and forth.
ENDLESS = float("inf")


def _can_shuttle(mover: Mover, reach: Reach, standing: frozenset) -> bool:
    """Whether ``mover`` has a free place it can go to and come back from."""
    for there in reach(mover, mover.at, standing):
        if there in standing:
            continue
        moved = Mover(mover.name, mover.side, there, mover.worth)
        if mover.at in set(reach(moved, there, standing)):
            return True
    return False


def _longest_way(mover: Mover, reach: Reach, standing: frozenset, most: int) -> int:
    """The most moves ``mover`` can still make one after another without coming back, up to ``most``: the longest
    path over free places from where it stands."""
    best = 0
    path = {mover.at}

    def walk(here: Hashable, done: int) -> None:
        nonlocal best
        best = max(best, done)
        if done >= most or best >= most:
            return
        moved = Mover(mover.name, mover.side, here, mover.worth)
        for there in reach(moved, here, standing):
            if there in standing or there in path:
                continue
            path.add(there)
            walk(there, done + 1)
            path.discard(there)

    walk(mover.at, 0)
    return best


def spare_moves(movers: Sequence[Mover], reach: Reach, side: int, most: int = 64) -> float:
    """How many moves ``side`` can make that only use up its own ways forward: ENDLESS where a mover of it can go back
    and forth, else the sum of its movers' longest ways forward (each up to ``most``)."""
    occupied = frozenset(m.at for m in movers)
    total = 0
    for m in movers:
        if m.side != side:
            continue
        standing = occupied - {m.at}
        if _can_shuttle(m, reach, standing):
            return ENDLESS
        total += _longest_way(m, reach, standing, most)
    return float(total)


def who_waits_longer(movers: Sequence[Mover], reach: Reach, most: int = 64) -> int | None:
    """The side that can make more such moves than the other, None where they can make as many (both without end)."""
    sides = sorted({m.side for m in movers})
    if len(sides) != 2:
        return None
    first, second = (spare_moves(movers, reach, side, most) for side in sides)
    if first == second:
        return None
    return sides[0] if first > second else sides[1]

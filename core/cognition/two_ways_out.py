"""Whether a thing can survive, and who runs out first when two cannot.

Go decides whether a group lives by one question, and it is not how big the
group is or how strong it looks. It is whether the group has TWO separate
spaces the other player can never fill. One is not enough — one gets filled
and the group dies — and the difference between one and two is the difference
between a thing that is alive and a thing that is merely still there. Nothing
about size or effort changes it.

The second question is what happens when two things are eating each other and
only one can live. Go players call it a capturing race and they do not fight
it, they COUNT it: how many moves until this one dies, how many until that one
does, and whoever needs fewer has already won, so the loser should stop
spending moves there. And an asymmetry that decides those races is that a
group with an eye gets the use of the shared space and a group without gets
none of it.

Both are about anything. A service with one way of getting its input has one
eye. A plan with a single point of failure is a thing that is still there
rather than a thing that is alive. Two processes competing for the last of
something is a capturing race, and the answer is arrived at by counting rather
than by trying harder — which is the part she could not do, because nothing of
hers ever asked how many steps something had left.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["HowItStands", "how_long_it_holds", "who_wins_the_race"]


@dataclass(frozen=True)
class HowItStands:
    """Whether a thing can be killed, and what it rests on."""

    #: Ways it has of surviving that cannot be taken away separately.
    ways_out: int
    #: Steps somebody would need to finish it off.
    steps_left: int

    @property
    def alive(self) -> bool:
        """Two ways out and it cannot be killed. One is a thing still standing."""
        return self.ways_out >= 2

    def describe(self) -> str:
        if self.alive:
            return f"alive: {self.ways_out} separate ways out"
        if self.ways_out == 1:
            return f"one way out and {self.steps_left} step(s) of room — still standing, not alive"
        return f"no way out, {self.steps_left} step(s) left"


def how_long_it_holds(
    thing: Any,
    *,
    ways_out: Callable[[Any], Iterable[Any]],
    room: Callable[[Any], int],
) -> HowItStands:
    """How this thing stands: its separate ways out, and its room to move.

    ``ways_out`` yields the things that would each have to be taken away on
    their own — two of them and nothing can kill it, because closing one does
    not close the other. Counting them is the whole test, and it is a test
    about STRUCTURE rather than about size: a large thing with one way out is
    in more danger than a small thing with two.
    """
    separate = list(ways_out(thing))
    return HowItStands(ways_out=len(separate), steps_left=max(0, int(room(thing))))


def who_wins_the_race(
    mine: HowItStands,
    theirs: HowItStands,
    *,
    shared: int = 0,
    i_move_first: bool = True,
) -> tuple[str, int]:
    """Which of two things eating each other survives, and by how many steps.

    One name, one meaning. This said "who runs out first" while its early
    branches handed back the SURVIVOR and its counting branch handed back the
    casualty — two opposite answers under one name, which is a bug that reads
    as correct in either half on its own.

    Counted rather than fought. Whoever needs fewer steps has already won it,
    and the other should stop spending moves there — which is the finding, and
    the reason to count.

    ``shared`` is room they are both using up. It goes to whichever of them is
    alive, because a thing that cannot be killed can spend it and a thing that
    can cannot: that asymmetry decides most of these, and it is why one way
    out is worth so much less than two.
    """
    mine_left = mine.steps_left + (shared if mine.alive and not theirs.alive else 0)
    theirs_left = theirs.steps_left + (shared if theirs.alive and not mine.alive else 0)
    if mine.alive and not theirs.alive:
        return "mine", theirs_left
    if theirs.alive and not mine.alive:
        return "theirs", mine_left
    if mine.alive and theirs.alive:
        return "neither", 0
    # Neither can be killed outright, so it is a count — and moving first is
    # worth exactly one step of it.
    gap = mine_left - theirs_left + (1 if i_move_first else 0)
    if gap > 0:
        return "mine", gap
    if gap < 0:
        return "theirs", -gap
    return "neither", 0


def not_worth_another_move(
    mine: HowItStands, theirs: HowItStands, *, shared: int = 0, i_move_first: bool = True
) -> bool:
    """Whether spending anything more here is spending it on a settled thing.

    The point of counting. A race already lost is not made better by moves, and
    the moves are worth something somewhere else — which is the same reasoning
    as walking away from a group nobody is threatening, arrived at from the
    other end.
    """
    who, _by = who_wins_the_race(mine, theirs, shared=shared, i_move_first=i_move_first)
    return who == "theirs"


def the_ones_worth_defending(
    things: Sequence[Any],
    *,
    ways_out: Callable[[Any], Iterable[Any]],
    room: Callable[[Any], int],
) -> list[tuple[Any, HowItStands]]:
    """Things with one way out, nearest to dying first.

    Not the biggest and not the weakest — the ones where one more step takes
    away the only thing holding them up. A thing with two ways out needs
    nothing; a thing with none is already gone.
    """
    weighed = [
        (one, how_long_it_holds(one, ways_out=ways_out, room=room)) for one in things
    ]
    return sorted(
        (one for one in weighed if one[1].ways_out == 1),
        key=lambda one: one[1].steps_left,
    )

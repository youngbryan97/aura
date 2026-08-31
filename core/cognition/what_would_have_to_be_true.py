"""Working backward from something out of reach.

Watching somebody clear 2048: they never once searched for the winning tile.
It was twenty moves away and nothing they could hold in their head reached
that far. What they did instead was know that the thing they wanted is made of
two of the thing below it, and that of two of the thing below that — so they
stopped wanting the far thing and started wanting the near one, and every
merge they made was chosen because it built the next merge's precondition.

That is a different kind of reasoning from looking ahead, and looking ahead
cannot substitute for it at any depth she can afford. Forward search asks
"where do these moves get me" and dies at the horizon. This asks "what would
have to be true for the thing I want to be near" and walks back until the
answer is something she can do this turn. The horizon stops mattering, because
the distance is crossed in wants rather than in moves.

Nothing here can invert a rule; a learned rule maps a state to a state and
running it backwards is not generally possible. It does not need to. The
question "which states is this reachable from" is a question about which
states are different from which, and telling states apart by a property is
something she already does. So a subgoal is not derived symbolically — it is
the property that separates the places the want is close to from the places it
is not, learned the same way anything else here is learned, and checkable in
the same way: it either predicts or it does not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.cognition.something_she_keeps_true import (
    SomethingTrue,
    how_well_it_predicts,
)

__all__ = ["AWayThere", "a_way_to_get_there", "what_would_have_to_be_true"]


@dataclass(frozen=True)
class AWayThere:
    """A chain of wants, the nearest first, ending at the one she started with.

    ``steps`` read in order is what to want next, then after that, and so on.
    The last is the thing she actually wants. ``reached`` says whether the
    first of them is something she can bring about from where she is; when it
    is false the chain is still worth having, because it says how far short
    the walk got and what it got stuck wanting.
    """

    steps: tuple[SomethingTrue, ...]
    reached: bool
    why: str

    @property
    def want_first(self) -> SomethingTrue | None:
        """The one to go after now."""
        return self.steps[0] if self.steps else None

    def describe(self) -> str:
        if not self.steps:
            return self.why
        walk = " then ".join(one.name for one in self.steps)
        return f"{walk} ({self.why})"


def what_would_have_to_be_true(
    wanted: Callable[[Any], bool],
    somewhere_like: Sequence[Any],
    in_reach: Callable[[Any, Callable[[Any], bool]], bool],
    *,
    deepest: int = 2,
) -> SomethingTrue | None:
    """The property that separates the places it is near from the places it is not.

    Ask of each place she knows whether the want is close from there. That
    splits what she has seen in two, and what she does with a split like that
    is find the property that predicts it. The property is the subgoal: making
    it true is what brings the want within reach, and it says so with a number
    rather than with a story.

    None when nothing she has seen tells the near places from the far ones —
    which is a real answer, and means either that she has not been anywhere
    useful or that this want has no near places.
    """
    watched = [(place, in_reach(place, wanted)) for place in somewhere_like]
    if not any(near for _, near in watched):
        return None
    if all(near for _, near in watched):
        return None
    weighed = how_well_it_predicts(watched, deepest=deepest)
    if not weighed:
        return None
    best = weighed[0]
    return best if best.tells_them_apart > 0 else None


def a_way_to_get_there(
    wanted: Callable[[Any], bool],
    from_here: Any,
    *,
    somewhere_like: Sequence[Any],
    in_reach: Callable[[Any, Callable[[Any], bool]], bool],
    called: str = "the thing she wants",
    how_far_back: int = 4,
    deepest: int = 2,
) -> AWayThere:
    """Walk back from a want until it lands on something she can do now.

    ``in_reach(place, want)`` is her ordinary looking-ahead, asked of a place
    that is not necessarily where she is. Everything else is here: the walk
    keeps replacing the want with what would make the want near, and stops the
    moment one of them is near from where she actually stands.

    It stops for three other reasons and each is worth telling apart. Nothing
    separates the near places from the far ones, so there is no subgoal to
    find. The walk came back to a want it already had, which is a loop and
    would go round for ever. Or it ran out of how far back it was willing to
    go, which is not a failure of the want but a bound on the effort.
    """
    at_the_end = SomethingTrue(name=called, holds=wanted)
    chain: list[SomethingTrue] = [at_the_end]
    already = {called}
    want_now: Callable[[Any], bool] = wanted

    for _ in range(max(1, how_far_back)):
        if in_reach(from_here, want_now):
            return AWayThere(
                steps=tuple(reversed(chain)),
                reached=True,
                why="near enough to go after now",
            )
        nearer = what_would_have_to_be_true(
            want_now, somewhere_like, in_reach, deepest=deepest
        )
        if nearer is None:
            return AWayThere(
                steps=tuple(reversed(chain)),
                reached=False,
                why="nothing she has been through tells the near places from the far",
            )
        if nearer.name in already:
            return AWayThere(
                steps=tuple(reversed(chain)),
                reached=False,
                why=f"wanting {nearer.name} again, which is a circle",
            )
        already.add(nearer.name)
        chain.append(nearer)
        want_now = nearer.holds

    return AWayThere(
        steps=tuple(reversed(chain)),
        reached=False,
        why=f"still not near after walking back {how_far_back}",
    )

"""Acting against what they are about to do, and knowing who goes first.

Competitive Pokémon is mostly two things and neither is damage. The first is
that everything resolves in speed order, so before anything else you ask
whether you move first — and a single point either way is the difference
between landing a blow and being gone before you swing. The second is
prediction: strong players do not answer the thing in front of them, they
answer the thing that will be in front of them, because the opponent is about
to swap it out and the move that beats what is there loses to what is coming.

Both are absent from everything she has. She weighs acts against the situation
as it stands, which quietly assumes two things that are usually false — that
the world will still look like this when her act lands, and that she gets to
go first.

Predicting is not clairvoyance and does not need to be. It needs only that
people repeat themselves: what somebody did the last several times this
situation came up is the best guess about what they will do now, and a guess
held loosely with a number on it is worth much more than no guess. She keeps
that number, so a party who is predictable is treated as predictable and one
who is not is treated as unknown rather than as random.

And moving first turns one act into two different acts. The same move is worth
what it does if it lands first, and worth what is LEFT of it if it lands
second — which is nothing at all when going second means not going.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhatTheyTendToDo", "against_what_is_coming"]


@dataclass
class WhatTheyTendToDo:
    """What each party has done, in each kind of situation."""

    #: who -> kind of situation -> act -> how often.
    did: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)

    def they_did(self, who: str, *, facing: str, act: str) -> None:
        self.did.setdefault(str(who), {}).setdefault(str(facing), {})[str(act)] = (
            self.did.setdefault(str(who), {}).setdefault(str(facing), {}).get(str(act), 0)
            + 1
        )

    def likely_next(self, who: str, *, facing: str) -> tuple[str, float]:
        """What they will probably do, and how much to believe it.

        Laplace, so that one sighting is a hint rather than a law, and so that
        a party she has never seen in this situation comes back as unknown
        rather than as whatever they did somewhere else.
        """
        seen = (self.did.get(str(who)) or {}).get(str(facing)) or {}
        if not seen:
            return "", 0.0
        total = sum(seen.values())
        best = max(seen, key=lambda one: (seen[one], one))
        return best, (seen[best] + 1) / (total + len(seen) + 1)

    def how_predictable(self, who: str, *, facing: str) -> float:
        """How much of what they do is the one thing they usually do."""
        return self.likely_next(who, facing=facing)[1]


def against_what_is_coming(
    acts: Sequence[str],
    *,
    now: Any,
    they_will: str,
    how_likely: float,
    after_theirs: Callable[[Any, str], Any],
    how_good: Callable[[Any, str], float],
    she_moves_first: bool = True,
) -> list[tuple[str, float]]:
    """Her acts, weighed against the situation her act will actually meet.

    Where she moves first, her act meets the world as it stands. Where she does
    not, it meets the world after theirs — and those are different worlds, so
    the same act is two different acts and the order decides which.

    Weighed between the two by how much she believes the prediction. A party
    she cannot predict is not guessed at: the weight falls to the situation as
    it stands, which is exactly what she did before this existed, so a bad
    prediction costs her nothing she had.
    """
    belief = max(0.0, min(1.0, float(how_likely))) if they_will else 0.0
    coming = after_theirs(now, they_will) if they_will else None
    weighed: list[tuple[str, float]] = []
    for one in acts:
        as_it_stands = float(how_good(now, one))
        if coming is None:
            weighed.append((one, as_it_stands))
            continue
        as_it_will_be = float(how_good(coming, one))
        if she_moves_first:
            # Hers lands on the world in front of her, and what follows is
            # theirs — so the prediction shades it rather than replacing it.
            worth = as_it_stands * (1.0 - belief * 0.5) + as_it_will_be * belief * 0.5
        else:
            # Hers lands after theirs, so the predicted world is the one it
            # actually meets.
            worth = as_it_stands * (1.0 - belief) + as_it_will_be * belief
        weighed.append((one, worth))
    return sorted(weighed, key=lambda one: (-one[1], one[0]))


def worth_going_first(
    acts: Sequence[str],
    *,
    now: Any,
    how_good: Callable[[Any, str], float],
    if_it_lands_second: Callable[[Any, str], float],
) -> float:
    """What moving first is worth here, as a number rather than a feeling.

    The gap between her best act landing first and the same act landing
    second. Where going second means not going at all, that gap is the whole
    of the act — and it is why a single point of speed decides games.
    """
    if not acts:
        return 0.0
    first = max(float(how_good(now, one)) for one in acts)
    second = max(float(if_it_lands_second(now, one)) for one in acts)
    return first - second

"""Wanting enough of something rather than the most of it.

Strong Go players stop in the late game and count. Then, if they are ahead,
they play the simplifying move — the one that ends complications, even where a
sharper move is worth more. If they are behind they do the opposite, and go
looking for the complicated fight they might not win, because the quiet move
loses slowly and losing slowly is still losing.

The same thing shows up everywhere once it is named. Somebody beating a hard
checkers engine chose a way of winning that gives up pieces, and said outright
it was not the best strategy — it was the one they could actually reach
against that opponent. A player low on health in Ninja Gaiden stops taking the
fast route. Somebody clearing 2048 with a board nearly full takes the merge
that might not come off, because the tidy slide only postpones it.

None of that is a taste for risk. It falls out of what winning is. When what
she needs is a threshold — a win, a pass, a thing shipped by Friday — the
question is not which choice has the best average outcome, it is which is most
likely to clear the bar. Those come apart exactly when it matters: a long way
behind, the best average is the surest loss, and a small chance beats a good
average that is not good enough.

So there is no appetite for risk here, and no dial. There is a bar, there is
where she stands, and there is which of her options most often clears it. Far
ahead, the safe option clears the bar every time and wins on the tie-break, so
she simplifies without being told to. Far behind, only the wild one ever
clears it, so she takes it. Both from the same sentence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["HowLikelyItIs", "how_likely_each_is", "the_one_most_likely_to_do"]


@dataclass(frozen=True)
class HowLikelyItIs:
    """One option, and what it does about the bar."""

    name: str
    #: How often the outcomes she can foresee clear the bar.
    clears_it: float
    #: What it comes to on average, which is the thing this is NOT chosen by.
    on_average: float
    #: How much its outcomes differ from each other.
    how_spread: float

    def describe(self) -> str:
        return (
            f"{self.name}: clears it {self.clears_it:.0%} of the time, "
            f"averages {self.on_average:.2f}, spread {self.how_spread:.2f}"
        )


def _spread(outcomes: Sequence[float]) -> float:
    if len(outcomes) < 2:
        return 0.0
    middle = sum(outcomes) / len(outcomes)
    return (sum((one - middle) ** 2 for one in outcomes) / len(outcomes)) ** 0.5


def how_likely_each_is(
    options: Mapping[str, Sequence[float]],
    *,
    needs: float,
    has: float = 0.0,
) -> list[HowLikelyItIs]:
    """Each option, likeliest to clear the bar first.

    ``options`` maps a choice to the outcomes she can foresee from it —
    whatever she has, whether that is a search's futures or what happened the
    last several times she did this. They are counted rather than modelled: the
    share of them that clears the bar is the estimate, and it needs no
    assumption about the shape of anything.

    Ties are broken toward the narrower spread, which is where simplifying
    when ahead comes from. Where several are certain, the quiet one is the one
    that stays certain when she has foreseen something wrong.
    """
    wanted = float(needs) - float(has)
    weighed: list[HowLikelyItIs] = []
    for name, outcomes in options.items():
        got = [float(one) for one in outcomes]
        if not got:
            continue
        weighed.append(
            HowLikelyItIs(
                name=name,
                clears_it=sum(1 for one in got if one >= wanted) / len(got),
                on_average=sum(got) / len(got),
                how_spread=_spread(got),
            )
        )
    weighed.sort(key=lambda one: (-one.clears_it, one.how_spread, -one.on_average, one.name))
    return weighed


def the_one_most_likely_to_do(
    options: Mapping[str, Sequence[float]],
    *,
    needs: float,
    has: float = 0.0,
) -> HowLikelyItIs | None:
    """The option likeliest to get her there, or None when there is nothing.

    When nothing clears the bar at all, the best average is taken — not
    because it is good, but because a bar nothing reaches is not a bar she can
    steer by, and the ordinary measure is what is left.
    """
    weighed = how_likely_each_is(options, needs=needs, has=has)
    if not weighed:
        return None
    if weighed[0].clears_it > 0:
        return weighed[0]
    return max(weighed, key=lambda one: (one.on_average, -one.how_spread, one.name))

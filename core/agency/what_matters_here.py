"""core/agency/what_matters_here.py — what a situation is worth, in THIS world.

`how_good_is_this` scores a situation on six things — nearness, her line,
room, order, smoothness, freedom — and adds them up with weights. The weights
are a standing guess, said to be one in the file that holds them: "a good
enough one to start from and no more than that". Nothing moved them.

So entering an unfamiliar world, she judges it by what mattered in the last
one. In a world whose only signal is a number nobody explained, freedom is
weighted at one and nearness at a fifth, and the result is a policy that
keeps its options open forever and never goes anywhere — measured here, on
sealed worlds, at nought wins in six with a perfectly correct model of what
every act does. She knew exactly what she was doing and had no reason to do
anything.

What is learned, and from what
------------------------------
One number per term: how much higher that term ran on the runs that went well
than on the runs that did not. Nothing else. That is a difference of means,
it needs no gradient and no tuning, and it says the only thing weights can
honestly say — this mattered here, that did not.

Three things keep it from flattering itself.

**It moves on outcomes, never on scores.** A weight fitted to make the
scoring agree with itself would agree with itself. The evidence is whether
the run ended well, which the scoring does not get a vote on.

**It costs nothing until it has evidence.** Below a floor of finished runs
the standing guess is returned unchanged. A weight learned from two runs is
two runs written as a number.

**It is shrunk towards the guess by how much evidence there is.** With ten
runs it barely moves; with hundreds it is most of the way to what was
measured. Nothing chooses the rate: it is the count.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Agency.WhatMattersHere")

__all__ = [
    "ENOUGH_RUNS",
    "always_the_guess",
    "keep_nothing",
    "keeping_nothing",
    "WhatMattersHere",
    "for_this_world",
    "forget_what_mattered",
]

#: Finished runs before the standing guess moves at all. Below this the
#: difference between the runs that went well and the runs that did not is
#: one run and its noise.
ENOUGH_RUNS = 8

#: What she can read off a situation without having learned anything about
#: this world: how well the world says she is doing, and whether she has been
#: here before. Everything else in her measure is a claim about what matters,
#: and before the first outcome she has no grounds for one.
_WHAT_IS_IN_FRONT_OF_HER = frozenset({"nearness", "newness"})

#: How much evidence it takes to be halfway from the guess to what was
#: measured. Not a rate anybody set — it is the point where the count of runs
#: equals the weight the guess is being given, which is what shrinking
#: towards a prior means.
AS_GOOD_AS_THE_GUESS = 30.0


@dataclass
class WhatMattersHere:
    """What each term of a situation is worth in one world, learned."""

    world: str
    #: Sum of each term over the runs that ended well, and how many there were.
    well: dict[str, float] = field(default_factory=dict)
    badly: dict[str, float] = field(default_factory=dict)
    wins: int = 0
    losses: int = 0

    @property
    def runs(self) -> int:
        return self.wins + self.losses

    def watched(self, along: Sequence[Mapping[str, float]], *, went_well: bool) -> None:
        """One finished run: the terms it passed through, and how it ended.

        The average over the run rather than its last state, because the last
        state of a run that went badly is the state it went badly in, and
        every term reads the same there whatever the run was like.
        """

        if not along:
            return
        if _KEEP_NOTHING[0]:
            return
        into = self.well if went_well else self.badly
        for name in {key for one in along for key in one}:
            got = [float(one.get(name, 0.0)) for one in along]
            into[name] = into.get(name, 0.0) + sum(got) / len(got)
        if went_well:
            self.wins += 1
        else:
            self.losses += 1

    def weights(self, guess: Mapping[str, float]) -> dict[str, float]:
        """The standing guess, moved by what this world actually rewarded.

        With no run that went well there is nothing to learn from, and the
        standing guess is not neutral — it weights keeping your options open
        as highly as getting anywhere, which in a world whose good places are
        in a corner among places you cannot leave is a policy that never
        arrives. Measured: nought wins in six, with a perfectly correct model
        of every act.

        So before the first success she steers by what is actually in front of
        her and by nothing else: how well she is doing by whatever the world
        reports, and whether she has been here before. Neither is a guess
        about what matters here — the first is the world's own reading and
        the second is her own history — and they last exactly until there is
        an outcome to learn from.

        Leaving newness out of that pair cost the whole thing. Steering by
        the reading alone, she climbed to a ridge and paced along it for the
        rest of the budget, because a step back and a step forward read the
        same and nothing else was allowed to speak.
        """

        if _ALWAYS_THE_GUESS[0]:
            return dict(guess)
        if _KEEP_NOTHING[0] or not self.wins:
            return {
                name: (1.0 if name in _WHAT_IS_IN_FRONT_OF_HER else 0.0)
                for name in guess
            }
        if self.runs < ENOUGH_RUNS or not self.losses:
            return dict(guess)
        pull = self.runs / (self.runs + AS_GOOD_AS_THE_GUESS)
        found: dict[str, float] = {}
        for name in set(guess) | set(self.well) | set(self.badly):
            standing = float(guess.get(name, 0.0))
            here = (self.well.get(name, 0.0) / self.wins) - (
                self.badly.get(name, 0.0) / self.losses
            )
            # A term that ran higher on the runs that went well is worth
            # more here; one that ran higher on the runs that did not is
            # worth less, and may go negative, which is a finding rather
            # than an error — some things are worth avoiding.
            found[name] = standing + pull * here
        return found

    def what_it_learned(self) -> dict[str, Any]:
        return {
            "world": self.world,
            "runs": self.runs,
            "wins": self.wins,
            "losses": self.losses,
            "moved": self.runs >= ENOUGH_RUNS and bool(self.wins and self.losses),
            "shrunk_by": round(self.runs / (self.runs + AS_GOOD_AS_THE_GUESS), 3),
        }


_LEARNED: dict[str, WhatMattersHere] = {}

#: When set, nothing learned about a world outlives the run it was learned
#: in. A lesion, not a setting: it exists so the gap between keeping what she
#: worked out and throwing it away can be measured, and a part whose removal
#: changes nothing was not doing the work.
_KEEP_NOTHING = [False]


def keep_nothing(on: bool = True) -> None:
    """Throw away what each run taught, as an ablation."""

    _KEEP_NOTHING[0] = bool(on)
    if on:
        _LEARNED.clear()


def keeping_nothing() -> bool:
    return bool(_KEEP_NOTHING[0])


#: When set, the standing guess is used from the first move, as if entering
#: an unfamiliar world were the same as entering a familiar one. The other
#: lesion, and the one that says which half of this module does the work.
_ALWAYS_THE_GUESS = [False]


def always_the_guess(on: bool = True) -> None:
    """Judge an unfamiliar world by what mattered in the last one."""

    _ALWAYS_THE_GUESS[0] = bool(on)


def for_this_world(world: str) -> WhatMattersHere:
    """What she has learned matters here. Empty for a world she has not met."""

    name = str(world or "somewhere")
    held = _LEARNED.get(name)
    if held is None:
        held = WhatMattersHere(world=name)
        _LEARNED[name] = held
    return held


def forget_what_mattered(world: str = "") -> None:
    """Used by tests, and by anything measuring against its own null."""

    if world:
        _LEARNED.pop(str(world), None)
    else:
        _LEARNED.clear()

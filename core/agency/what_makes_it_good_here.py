"""What is worth having in THIS world, learned rather than declared.

A situation has several things to like about it: how near it is to what she
was asked for, whether her line still holds in it, how much room is left, how
much of it runs in order. What each of those is worth is set by three numbers
somebody picked, and picked once, for every world she will ever be in.

They cannot all be right. Room matters enormously in a world that fills up and
not at all in one that does not. Order is the whole game where things combine
by neighbour and meaningless where they do not. A weighting good for one world
is a handicap in the next, and no amount of care in choosing it fixes that,
because the right answer is a fact about the world rather than about her.

So she works it out the way she works everything else out here: by acting and
watching. Each move gives a situation before, a situation after, and whether
it left her better off by the measure she is currently using. Where a thing
was higher in the moves that went well than in the moves that went badly, it
is worth more than she thought. Where it was not, it is worth less.

Nothing about this knows what a world is. It reads the same terms any laid-out
thing produces and moves numbers toward what actually happened.

**Measured 2026-08-27, and it does not help yet.** Eight games each, run to a
dead board, depth held fixed so the arms differ only in the weights:

    not learning at all           median best 512   total 1144
    graded by what the world counts    "     512     "   1144
    graded by the measure being learned  "   512     "   1123

Two earlier runs appeared to show large gains in both directions. Both were
the depth confound: how deep she can afford tracks what a level costs, an arm
that does more work per move searches shallower, and the comparison becomes
one of speed. With depth fixed the difference disappears entirely.

The two graders tried both fail, for opposite reasons. Grading by the measure
being learned is a loop — a weight that grows makes more moves look good,
which grows it again, and `order` reached its ceiling inside a thousand moves.
Grading by what the world counts has no contrast, because something is dealt
after nearly every move, so almost everything reads as better and nothing
moves.

So this is not wired into the pursuit. It is kept, with its tests and this
record, because the machinery is right and the missing piece is named: a grade
that is dense enough to have contrast and independent of the weights it
teaches. Credit assigned over several moves rather than one would be that,
and is the next thing to try.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY

__all__ = [
    "ENOUGH_TO_REWEIGH",
    "HOW_FAST_SHE_CHANGES_HER_MIND",
    "LEAST_A_THING_CAN_BE_WORTH",
    "MOST_A_THING_CAN_BE_WORTH",
    "WhatMakesItGoodHere",
]

logger = logging.getLogger("Aura.WhatMakesItGoodHere")

#: How many moves she watches before the weights are hers rather than the
#: standing guess. Below this a run of luck decides what matters.
ENOUGH_TO_REWEIGH = 40

#: How far a weight moves on one move's evidence. Small, because a single move
#: is weak evidence about a whole world and the weights steer every decision
#: after it.
HOW_FAST_SHE_CHANGES_HER_MIND = 0.02

#: A thing cannot become worthless, however badly it reads: a term that is
#: pinned at zero can never earn its way back, and a world changes.
LEAST_A_THING_CAN_BE_WORTH = 0.05

#: Nor can one thing swamp everything else on the strength of a good spell.
MOST_A_THING_CAN_BE_WORTH = 3.0


@dataclass
class WhatMakesItGoodHere:
    """What each thing about a situation is worth in this world."""

    worth: dict[str, float] = field(
        default_factory=lambda: dict(AS_GOOD_A_GUESS_AS_ANY)
    )
    #: How many moves have been watched, so she knows when to trust it.
    seen: int = 0
    #: Where each thing stood in the moves that went well and badly.
    _when_it_went_well: dict[str, float] = field(default_factory=dict)
    _when_it_did_not: dict[str, float] = field(default_factory=dict)
    _went_well: int = 0
    _did_not: int = 0

    # ── learning ─────────────────────────────────────────────────────────

    def watched(self, after: Mapping[str, float], better: bool) -> None:
        """One move: what the situation it led to was like, and whether it helped.

        ``better`` must be judged by something these weights cannot move.
        Grading a move with the measure being learned is a loop: a weight that
        grows makes more moves look good, which grows it again. Measured
        2026-08-27 — order ran to its ceiling within a thousand moves and her
        play went from a best tile of 1024 down to 768. The goal she was given
        is the honest grader, because no weight of hers can shift it.
        """
        if not after:
            return
        self.seen += 1
        into = self._when_it_went_well if better else self._when_it_did_not
        for name, value in after.items():
            into[name] = into.get(name, 0.0) + float(value)
        if better:
            self._went_well += 1
        else:
            self._did_not += 1
        self._reweigh()

    def _reweigh(self) -> None:
        """Move each weight toward what the moves that went well had more of."""
        if self._went_well < 1 or self._did_not < 1:
            return
        for name in list(self.worth):
            good = self._when_it_went_well.get(name, 0.0) / self._went_well
            bad = self._when_it_did_not.get(name, 0.0) / self._did_not
            # How much more of this thing the good moments had. Positive means
            # it is worth more than she has been treating it as.
            told = good - bad
            moved = self.worth[name] + HOW_FAST_SHE_CHANGES_HER_MIND * told
            self.worth[name] = max(
                LEAST_A_THING_CAN_BE_WORTH, min(MOST_A_THING_CAN_BE_WORTH, moved)
            )

    # ── using it ─────────────────────────────────────────────────────────

    def worked_out(self) -> bool:
        """Whether she has watched enough for these to be hers."""
        return self.seen >= ENOUGH_TO_REWEIGH

    def weights(self) -> dict[str, float] | None:
        """What to weigh a situation by, or nothing if she has not worked it out."""
        return dict(self.worth) if self.worked_out() else None

    def says(self) -> str:
        """What matters here, for whoever has to answer for it."""
        if not self.worked_out():
            return (
                f"what matters here is not worked out yet ({self.seen} move(s) watched)"
            )
        ordered = sorted(self.worth.items(), key=lambda thing: -thing[1])
        said = ", ".join(f"{name} {value:.2f}" for name, value in ordered)
        return f"what matters here, worked out over {self.seen} move(s): {said}"

    # ── keeping it ───────────────────────────────────────────────────────

    def as_memory(self) -> dict[str, Any]:
        return {"worth": dict(self.worth), "seen": self.seen}

    @classmethod
    def from_memory(cls, held: Any, trust: float = 1.0) -> "WhatMakesItGoodHere":
        """What mattered here last time, discounted like anything carried over."""
        if not isinstance(held, dict):
            return cls()
        share = max(0.0, min(1.0, float(trust)))
        worth = dict(AS_GOOD_A_GUESS_AS_ANY)
        kept = held.get("worth")
        if isinstance(kept, dict):
            for name, value in kept.items():
                if name in worth and isinstance(value, (int, float)):
                    # Part of the way back toward the standing guess, so a
                    # weighting from yesterday is a starting point and not a
                    # verdict.
                    worth[name] = worth[name] * (1.0 - share) + float(value) * share
        return cls(worth=worth, seen=int(round(float(held.get("seen") or 0) * share)))

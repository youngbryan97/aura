"""Situations her own measure cannot tell apart, and what would.

Her measure of a good situation is a handful of properties added up. When two
situations score the same and one of them goes on to turn out much better, the
measure has nothing to say about the difference — and the difference is real,
because the outcomes differ. That gap is the only honest place a NEW property
can come from: not from somebody noticing one, but from her own failures to
account for what happened.

The loop:

    play, and remember each situation with how it eventually turned out
    find the pairs her measure calls equal whose outcomes were not
    search the space of measures she can compose for one that separates them
    check it separates HELD-BACK pairs it was not chosen on
    then, and only then, find out whether including it actually plays better

The last step is the one that matters and the one usually skipped. A property
that explains the past is a story; a property that improves the play is a
finding. Nothing is promoted on the strength of fitting what has already
happened.

Nothing here knows what a board is, or a tile, or a game.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.agency.inventing_a_measure import Measure, every_measure

__all__ = [
    "ENOUGH_PAIRS_TO_LOOK",
    "MUST_BEAT_CHANCE_BY",
    "Lived",
    "WhatICannotExplain",
]

logger = logging.getLogger("Aura.WhatICannotExplain")

#: How many unexplained pairs before searching for a reason is worth anything.
#: Below this, whichever measure happens to fit is fitting noise.
ENOUGH_PAIRS_TO_LOOK = 40

#: How much better than a coin a candidate has to be on pairs it was NOT
#: chosen on. A measure that explains the pairs it was fitted to explains
#: nothing; the number that counts is the one from the half held back.
MUST_BEAT_CHANCE_BY = 0.15

#: How many unexplained pairs are worth carrying. Past this the answer stops
#: changing and the search starts costing more than it tells her.
MOST_PAIRS_WORTH_LOOKING_AT = 4000

#: How many situations to remember. A long run must not grow without limit,
#: and the pairs that need explaining come from all over it rather than from
#: the end, so the oldest go first.
WORTH_REMEMBERING = 20000

#: How close two scores have to be before her measure is calling them equal.
#: A share of the range her measure produces, so it does not depend on the
#: scale anything happens to be on.
CALLED_EQUAL_WITHIN = 0.02

#: How differently two situations have to turn out before the difference is
#: worth explaining rather than being the ordinary spread of luck.
TURNED_OUT_DIFFERENTLY_BY = 0.25


@dataclass(frozen=True)
class Lived:
    """One situation she was in, and how things went from there."""

    situation: Any
    scored: float
    turned_out: float


@dataclass
class WhatICannotExplain:
    """Everything she has been in, and the properties that would account for it."""

    lived: list[Lived] = field(default_factory=list)

    def been_here(self, situation: Any, scored: float, turned_out: float) -> None:
        """One situation, what her measure said of it, and what came of it."""
        self.lived.append(Lived(situation, float(scored), float(turned_out)))
        if len(self.lived) > WORTH_REMEMBERING:
            del self.lived[: len(self.lived) - WORTH_REMEMBERING]

    # ── the pairs that need explaining ───────────────────────────────────

    def unexplained(self) -> list[tuple[Lived, Lived]]:
        """Pairs her measure called equal whose outcomes were not equal.

        Both halves are needed. Two situations that scored differently are
        already accounted for, however badly; two that turned out the same
        have nothing to account for.
        """
        if len(self.lived) < 2:
            return []
        scores = [one.scored for one in self.lived]
        spread = (max(scores) - min(scores)) or 1.0
        # In score order, so the ones her measure calls equal are next to each
        # other and the rest are never looked at. Comparing everything with
        # everything is the same answer and quadratic: on an ordinary run of
        # twenty thousand situations that is two hundred million comparisons
        # for a handful of pairs, which is a search nobody can afford to run
        # and so a capability nobody uses.
        in_order = sorted(self.lived, key=lambda one: one.scored)
        near = CALLED_EQUAL_WITHIN * spread
        pairs: list[tuple[Lived, Lived]] = []
        for index, one in enumerate(in_order):
            for other in in_order[index + 1 :]:
                if other.scored - one.scored > near:
                    break
                if abs(one.turned_out - other.turned_out) >= TURNED_OUT_DIFFERENTLY_BY:
                    pairs.append((one, other))
                if len(pairs) >= MOST_PAIRS_WORTH_LOOKING_AT:
                    return pairs
        return pairs

    # ── what would account for them ──────────────────────────────────────

    def worth_trying(
        self, most: int = 3, among: Sequence[Measure] | None = None
    ) -> list[tuple[Measure, float, int]]:
        """The properties best worth TESTING, best-explaining first.

        Not the answer — the shortlist. Explaining what already happened and
        improving what happens next are different things, and the first is
        only evidence about the second.

        Measured 2026-08-29: the single best explanation of her own unexplained
        pairs agreed with the outcome on 98% of the pairs it was never chosen
        on, and playing with it made her 10% WORSE. Nothing was wrong with the
        search. A property can correlate with how things turned out and still
        be a bad thing to steer by, because steering by it changes what
        happens — which is exactly why the causal test is the one that decides
        and this only proposes.
        """
        ranked = self._ranked(among)
        return ranked[: max(1, int(most))]

    def what_would_explain(
        self, among: Sequence[Measure] | None = None
    ) -> tuple[Measure, float, int] | None:
        """The single property that best separates what she cannot explain."""
        ranked = self._ranked(among)
        return ranked[0] if ranked else None

    def _ranked(
        self, among: Sequence[Measure] | None = None
    ) -> list[tuple[Measure, float, int]]:
        """Every candidate that clears the bar, best on held-back pairs first."""
        pairs = self.unexplained()
        if len(pairs) < ENOUGH_PAIRS_TO_LOOK:
            return []
        # Half to choose on, half to be judged on. Interleaved rather than
        # split down the middle, so a run that drifted does not put all of one
        # kind of situation in the half that decides.
        choosing = pairs[0::2]
        judging = pairs[1::2]
        found: list[tuple[Measure, float, int]] = []
        for measure in (among if among is not None else every_measure()):
            if _agrees_on(measure, choosing) <= 0.5:
                continue
            held_back = _agrees_on(measure, judging)
            if held_back - 0.5 < MUST_BEAT_CHANCE_BY:
                continue
            found.append((measure, held_back, len(pairs)))
        found.sort(key=lambda row: -row[1])
        if found:
            logger.info(
                "%d propert(ies) would explain what I could not, best %r at %.0f%% "
                "of %d pairs it was not chosen on",
                len(found), found[0][0].name, found[0][1] * 100.0, len(judging),
            )
        return found


def _agrees_on(measure: Measure, pairs: Sequence[tuple[Lived, Lived]]) -> float:
    """How often this property points the same way the outcomes did."""
    if not pairs:
        return 0.0
    agreed = 0
    counted = 0
    for one, other in pairs:
        mine = measure.read(one.situation) - measure.read(other.situation)
        theirs = one.turned_out - other.turned_out
        if abs(mine) < 1e-9:
            # Saying nothing is not agreeing. A measure that is flat across
            # the pairs explains them no better than the measure that missed
            # them, and counting its silence as a hit is how a constant wins.
            counted += 1
            continue
        counted += 1
        if (mine > 0) == (theirs > 0):
            agreed += 1
    return agreed / counted if counted else 0.0

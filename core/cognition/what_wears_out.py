"""Whether the world answers what she keeps doing.

Two fight breakdowns and a basketball one, watched 2 October 2026, teach one
thing between them. Cejudo against Walton: for a round, every time Cejudo
lunged in with his right hand Walton met him with the same check hook, until
Cejudo began ducking under it; later Walton, caught four times in a row by the
same short right, began slipping inside it. The commentator's summary was that
whoever kept lunging in kept getting caught. Rountree against Saki: one leg
kick thrown from punching range was answered the next time by a straight left
that landed while Saki stood on one leg. And a centre with one move scores all
season and is schemed out of it in the playoffs, because by then the other
side knows it is coming.

None of that is about fighting. It is about a world that learns. Some worlds
answer an act the same way however often she makes it: a board slides the same
way every time, and there the right thing to do with the best act is to keep
making it. Others watch, and an act that worked as a surprise stops working as
a habit. There the best act becomes the worst one by being the best one, and
the only defence is to be harder to read.

Which kind she is in is something she can measure, so nothing here assumes it.
For every act she makes she keeps how long it had been since she last made it,
and whether what she expected of it held. Two accounts of that record compete
on prediction, the way reference classes do in core/agency/reference_class.py:

    the act alone          each act has its own rate of holding
    the act and its rest   the rate also depends on whether she came back
                           to it soon

Each outcome is predicted by both before it is counted. Summed, those are the
two accounts' marginal likelihoods, so the second pays for its extra numbers
and gains weight only when the record really does depend on how soon she came
back to an act. The weight it has is the posterior chance it is the right
account, and an act she would be making again soon is marked down by that
chance times what coming back soon has cost it. In a world that does not
learn, the chance falls toward nothing as the record grows and so does every
mark.

The outcome is whether her expectation held, not whether the move went well.
A move that goes badly because the situation was bad is the situation's doing
and her model already says so; a move that goes worse than her model said,
more often when she has just made it, is the world answering her. Read on raw
results, pressing a key again straight after it slid everything would look
like wear on a board that never learns anything.

How soon is counted in her own acts, not in seconds, and in bands that double:
straight after, two or three acts later, four to seven, and so on. Doubling
bands have no scale of their own, so a world that remembers her last move and
one that remembers her last twenty are read by the same arithmetic. The cost
of coming back now is the best the act has done when she left it longer than
this, against what it has done at this distance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["REMEMBERED_OUTCOMES", "WhatWearsOut", "gap_before"]

#: The band of an act never made before: as rested as an act can be.
NEVER = 1 << 30

#: How many graded acts she keeps. A bound on memory and nothing more: enough
#: for a long run to settle which account is right, bounded so an indefinite
#: run does not grow without limit. Oldest go first, which also lets a world
#: that has stopped adapting be read as one.
REMEMBERED_OUTCOMES = 512


def gap_before(act: str, made: Sequence[str]) -> int | None:
    """How many acts back she last made this one; None if never.

    ``made`` is what she did before it, oldest first. One means she made it
    immediately before.
    """
    name = _name(act)
    for back, one in enumerate(reversed(made), start=1):
        if _name(one) == name:
            return back
    return None


def _band_of(gap: int | None) -> int:
    """Which doubling band a gap falls in: 1, 2-3, 4-7, ... and never."""
    if gap is None:
        return NEVER
    return max(0, int(gap)).bit_length() - 1


def _name(act: Any) -> str:
    return str(act or "").strip().lower()


def _rate(counts: Sequence[int] | None) -> float:
    """The chance of holding, from counts with nothing assumed: Laplace."""
    tried, held = (list(counts or ()) + [0, 0])[:2]
    return (held + 1.0) / (tried + 2.0)


def _rate_within(part: Sequence[int], whole: Sequence[int]) -> float:
    """The chance of holding in one part of an act's record, drawn toward the rest of it.

    Laplace's two pseudo-counts, centred on what the act did everywhere else
    instead of on a half. A part with three outcomes in it says little beyond
    the rest of the act's record, which is what three outcomes are worth; and a
    part that IS the whole record has no rest, so this is Laplace exactly and
    the two accounts cannot differ. Centred on the act's own rate instead, the
    second account came out a little sharper than the first on any record at
    all and led by about the log of its length on a board where nothing ever
    depended on rest.
    """
    tried, held = (list(part) + [0, 0])[:2]
    all_tried, all_held = (list(whole) + [0, 0])[:2]
    elsewhere = _rate([all_tried - tried, all_held - held])
    return (held + 2.0 * elsewhere) / (tried + 2.0)


def _loss(p: float, held: bool) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return -math.log(p if held else 1.0 - p)


@dataclass
class WhatWearsOut:
    """Her acts, how soon she came back to each, and whether that cost her."""

    #: (act, gap or None, its band, held), oldest first.
    record: list[tuple[str, int | None, int, bool]] = field(default_factory=list)
    #: Summed predictive loss of each account over the same outcomes.
    loss_alone: float = 0.0
    loss_with_rest: float = 0.0
    #: Said once, when she first finds the world answering her.
    noticed: bool = False

    # ── what the record holds ────────────────────────────────────────────

    def _counts(self, act: str, band: int | None = None, *, beyond: bool = False) -> list[int]:
        """Tried and held for an act after a gap: all of them, one band, or every band past one.

        First uses are left out of every count. Both accounts are about how the
        gap she left matters, and an outcome with no gap is evidence of neither.
        """
        rows = [
            held
            for one, gap, _band, held in self.record
            if one == act
            and gap is not None
            and (
                band is None
                or (_band_of(gap) > band if beyond else _band_of(gap) == band)
            )
        ]
        return [len(rows), sum(1 for held in rows if held)]

    # ── learning it ──────────────────────────────────────────────────────

    def it_went(self, act: str, held: bool, made_before: Sequence[str]) -> None:
        """One act graded: what she expected of it, and whether that held.

        Scored by both accounts before it is counted, so neither is ever
        credited with predicting something it had already been told.
        """
        name = _name(act)
        if name:
            self._replay(name, gap_before(name, made_before), bool(held))

    # ── using it ─────────────────────────────────────────────────────────

    def learns_her(self) -> float:
        """The posterior chance that coming back soon is what decides it.

        Even odds to begin with; the summed predictive losses are the two
        marginal likelihoods, so this is Bayes and nothing chosen.
        """
        lead = self.loss_alone - self.loss_with_rest
        if lead >= 0:
            return 1.0 / (1.0 + math.exp(-lead))
        return math.exp(lead) / (1.0 + math.exp(lead))

    def cost_of_going_again(self, act: str, made: Sequence[str]) -> float:
        """What making this act now is expected to cost in expectations that break.

        Nothing for an act she has rested, nothing for one whose soon and
        rested uses have held alike, and nothing in a world the record says
        does not learn her.
        """
        name = _name(act)
        band = _band_of(gap_before(name, made))
        if band == NEVER:
            return 0.0
        whole = self._counts(name)
        rested = [
            _rate_within(self._counts(name, longer), whole)
            for longer in {one for _act, _gap, one, _held in self.record if _act == name}
            if NEVER > longer > band
        ]
        if not rested:
            return 0.0
        # Against the best rest she has given it, not all rest pooled: where a
        # world forgets her after four acts, gaps of two and three are as bad
        # as one, and pooled with them the rest that works disappears. Each band
        # is drawn toward the act's other bands, so a band she has tried twice
        # cannot win on luck by much.
        worse = max(rested) - _rate_within(self._counts(name, band), whole)
        return self.learns_her() * max(0.0, worse)

    def worn(self, acts: Sequence[str], made: Sequence[str]) -> dict[str, float]:
        """Every act on offer, with what going to it again would cost."""
        return {str(act): self.cost_of_going_again(str(act), made) for act in acts}

    def says(self) -> str:
        """Whether this world answers what she keeps doing, for a log line."""
        if not self.record:
            return "nothing graded yet"
        chance = self.learns_her()
        acts = sorted({act for act, *_rest in self.record})
        rested_better = {
            act: _rate_within(self._counts(act, 0, beyond=True), self._counts(act))
            - _rate_within(self._counts(act, 0), self._counts(act))
            for act in acts
        }
        hurt = max(acts, key=lambda act: rested_better[act])
        if chance < 0.5 or rested_better[hurt] <= 0.0:
            return f"this does not answer what I keep doing ({len(self.record)} acts)"
        return (
            f"this answers what I keep doing ({chance:.2f} sure, {len(self.record)} acts); "
            f"going straight back to {hurt} has cost the most"
        )

    # ── keeping it ───────────────────────────────────────────────────────

    def as_memory(self) -> dict[str, Any]:
        return {
            "record": [[act, gap, band, held] for act, gap, band, held in self.record],
            "loss_alone": self.loss_alone,
            "loss_with_rest": self.loss_with_rest,
            "noticed": self.noticed,
        }

    @classmethod
    def from_memory(cls, held: Any, trust: float = 1.0) -> WhatWearsOut:
        """What she found last time, as evidence about today and not a fact about it.

        The newest share of the record that ``trust`` allows comes back, and the
        two accounts are scored again over it from nothing, so a world that has
        changed since can overturn it in fewer acts than it took to build.
        """
        if not isinstance(held, dict):
            return cls()
        rows = []
        for row in held.get("record") or ():
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                continue
            act, gap, _band, was_held = row
            rows.append((_name(act), None if gap is None else int(gap), bool(was_held)))
        keep = int(len(rows) * max(0.0, min(1.0, float(trust))))
        again = cls()
        for act, gap, was_held in rows[len(rows) - keep:] if keep else ():
            # The gap is what it was when she made it, not what this shortened
            # record would say, so it is replayed as it was recorded.
            again._replay(act, gap, was_held)
        again.noticed = bool(held.get("noticed")) and again.learns_her() >= 0.5
        return again

    def _replay(self, act: str, gap: int | None, held: bool) -> None:
        """Score one outcome under both accounts, then count it.

        The first time she makes an act there is no gap to read, so both
        accounts predict it alike and neither counts it. Counted by one of them,
        one miss on a first use and three hundred holds after it made the second
        account "explain" the miss and lead, on a board where nothing ever
        depended on rest.
        """
        band = _band_of(gap)
        whole = self._counts(act)
        alone = _rate(whole)
        with_rest = alone if band == NEVER else _rate_within(self._counts(act, band), whole)
        self.loss_alone += _loss(alone, held)
        self.loss_with_rest += _loss(with_rest, held)
        self.record.append((act, gap, band, held))
        if len(self.record) > REMEMBERED_OUTCOMES:
            self.record = self.record[len(self.record) - REMEMBERED_OUTCOMES:]

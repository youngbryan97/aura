"""Telling the thing she is acting on from the things that merely report it.

Five attempts at finding a board in a screenshot failed the same way, and the
reason is worth writing down: in ONE frame a board and a score panel are the
same object. Both are text in a box that sits inside the window. Density,
size, regular spacing, and being mostly full all pick out the advertisement as
readily as the game. There is nothing in a single picture that says which of
them is the thing.

Across two frames there is. When she acts, a board REARRANGES: the values in
it are the values that were in it, in other places. A score does not rearrange.
It shows a number that was nowhere on the screen a moment ago, computed from
the whole rather than moved from somewhere. So does a move counter, a clock,
and a line of commentary.

That is the difference and it is not about games. Anything she operates by
moving its contents around — a list of files, a hand of cards, a board — is a
place where what appears was already there. Anything that reports on what she
did shows her something new. The first is the thing. The second is about the
thing.

Nothing here needs to know which is which in advance. Each place keeps two
counts, and a place belongs to the thing when its changes have more often been
rearrangements than arrivals.

A place reports only when its arrivals are more than the thing's own places
show by chance. The thing has arrivals too: a tile appears in an empty square,
two tiles become one that was nowhere before. Called a readout on one arrival
against no rearrangement, a board square was cut out of her rule while a tile
sat in it, and every move through that square was scored as her rule being
wrong (live, 26 Sep: "1 place(s) only report", then "(0,2) said '2' saw None";
eight of the first nine misses she logged were on that square). Played out
over 200 games, the old test called some board square a readout on 12% of
moves with random play and on 89% with the tiles kept in a corner, where
merges land in place and nothing leaves. This one does on at most 0.2% of
moves, with or without a score on the screen, and finds the score in every
game, by the tenth move with random play and the fifteenth with the corner
kept (the middle game of 200).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import exp, lgamma
from typing import Any

__all__ = ["WRONG_BY_CHANCE", "MovesWithinItself", "at_least_as_many", "what_moved_within"]

#: How often a place of the thing may be called a readout by bad luck: one
#: time in twenty, shared across every place tested and every act watched,
#: because the question is asked of each of them again after every act.
WRONG_BY_CHANCE = 0.05


def at_least_as_many(times: int, out_of: int, others_times: int, others_not: int) -> float:
    """The chance of ``times`` or more in ``out_of``, at the rate the other places showed.

    That rate is not known, only what the others did: ``others_times`` of it
    and ``others_not`` of the rest, taken on Jeffreys' prior. So the chance is
    the beta-binomial tail, which carries the doubt about the rate as well as
    the luck of the draw. With nothing seen elsewhere yet, one time is no
    surprise at all.
    """
    a, b = others_times + 0.5, others_not + 0.5

    def log_beta(x: float, y: float) -> float:
        return lgamma(x) + lgamma(y) - lgamma(x + y)

    whole = log_beta(a, b)
    ways = lgamma(out_of + 1)
    return min(
        1.0,
        sum(
            exp(ways - lgamma(k + 1) - lgamma(out_of - k + 1) + log_beta(k + a, out_of - k + b) - whole)
            for k in range(times, out_of + 1)
        ),
    )


@dataclass
class MovesWithinItself:
    """Which places hold things that move about, and which report."""

    #: How often what appeared at a place was already somewhere in the region.
    rearranged: dict[tuple[int, int], int] = field(default_factory=dict)
    #: How often what appeared had not been anywhere in it.
    arrived: dict[tuple[int, int], int] = field(default_factory=dict)
    acts: int = 0
    #: For each reporting place, the last number it showed, how many times it
    #: went up, and how many times it went down. A reporter that never goes
    #: down is a measure of progress, which is a thing worth having when
    #: nobody has said what progress is.
    _last: dict[tuple[int, int], float] = field(default_factory=dict)
    _rose: dict[tuple[int, int], int] = field(default_factory=dict)
    _fell: dict[tuple[int, int], int] = field(default_factory=dict)
    #: Places she was told about rather than has seen — brought back from the
    #: last sitting. They have to be seen again before they count.
    _carried: frozenset[tuple[int, int]] = frozenset()
    #: Places that have rearranged since this sitting began.
    _seen_again: set[tuple[int, int]] = field(default_factory=set)
    #: The lowest each place has ever shown, so that going back to the
    #: beginning can be told from going backwards.
    _least: dict[tuple[int, int], float] = field(default_factory=dict)
    #: How many readings each place has held anything in. A place that turns
    #: up once is not one of the thing's own.
    _stood_in: dict[tuple[int, int], int] = field(default_factory=dict)

    def saw(
        self,
        before: Mapping[tuple[int, int], str],
        after: Mapping[tuple[int, int], str],
    ) -> None:
        """Take one act's worth of evidence.

        A value is counted as rearranged when it was somewhere else before and
        has LEFT that somewhere. Merely having been on the screen will not do,
        and the difference is not a nicety: early in a game of 2048 the score
        passes through 4, 8 and 16, which are also tile values, so a score that
        happens to read 8 while a tile reads 8 was being called a tile. Every
        reporting place on the screen was swallowed that way and she found no
        score at all.
        """
        if not before or not after:
            return
        self.acts += 1
        for where, text in after.items():
            if text:
                self._stood_in[where] = self._stood_in.get(where, 0) + 1
        for where, text in after.items():
            if not text or before.get(where) == text:
                continue
            came_from_somewhere = any(
                other != where
                and was_there == text
                and after.get(other) != text
                for other, was_there in before.items()
            )
            if came_from_somewhere:
                self.rearranged[where] = self.rearranged.get(where, 0) + 1
                self._seen_again.add(where)
            else:
                self.arrived[where] = self.arrived.get(where, 0) + 1
            self._watch_the_number(where, text)

    def _watch_the_number(self, where: tuple[int, int], text: str) -> None:
        """Note which way a place's number went, where it holds one."""
        try:
            now = float(str(text).replace(",", "").strip())
        except (TypeError, ValueError):
            # not a failure: a place holding words holds no measure.
            return
        was = self._last.get(where)
        self._last[where] = now
        least = self._least.get(where)
        self._least[where] = now if least is None else min(least, now)
        if was is None or now == was:
            return
        if now > was:
            self._rose[where] = self._rose.get(where, 0) + 1
            return
        if least is not None and now <= least:
            # Back to where it began rather than backwards. A score goes to
            # nought when a new game starts, and counting that as falling
            # makes every tally on the screen look like it goes both ways —
            # so she finds no measure of progress at all from the second game
            # onward, which is exactly when she has most use for one.
            return
        self._fell[where] = self._fell.get(where, 0) + 1

    def what_only_goes_up(self) -> frozenset[tuple[int, int]]:
        """Reporting places whose number has never gone down.

        A score. A move count. A total. Nobody had to say what progress is:
        something on the screen has been keeping the tally the whole time, and
        the thing that distinguishes it is that it only ever rises.

        Only places that report, because a value inside the thing itself rises
        and falls as she moves it about, and a tile that happens never to have
        been beaten is a coincidence rather than a tally.
        """
        reports = self.the_things_that_report()
        return frozenset(
            where
            for where in reports
            if self._rose.get(where, 0) > 0 and not self._fell.get(where, 0)
        )

    def what_measures_doing_well(self) -> frozenset[tuple[int, int]]:
        """The tally that rises when she does well, out of the ones that rise.

        A move counter only goes up too, and it goes up whatever she does, so
        it says nothing about whether what she did was any good. It is a clock.
        What separates a score from a clock is that a score sometimes does not
        move: it rises on the acts that achieved something and stands still on
        the ones that did not, and standing still is the whole of its value as
        a measure.
        """
        return frozenset(
            where
            for where in self.what_only_goes_up()
            if self._rose.get(where, 0) < self.acts
        )

    def the_thing_itself(self) -> frozenset[tuple[int, int]]:
        """The places whose contents move about rather than arrive.

        A place that has never changed is not in it, and that is deliberate:
        furniture inside a board's outline never changes, which is exactly how
        it got into the outline in the first place.

        Nor is a place that has only ever been seen once. A score's text is
        centred, so it SHIFTS as the number gets longer — 576 becomes 1024 a
        few hundredths to the left, which is a value appearing where it was
        not, having left where it was, and that is the exact signature of a
        tile sliding. LIVE 2026-08-31, playing the real game: the board she
        had settled on as four by four grew two more columns, and they were
        made of the places the score used to be.

        And a place she was told about rather than has seen has to be seen
        again before it counts. Where a thing is is only worth remembering
        while the thing is still there: the window is moved, the page reflows,
        the game restarts a little to the left, and what comes back from the
        last sitting is a second set of places laid over this one. Everything
        above is about telling places apart WITHIN a sitting and none of it
        helps, because a carried place arrives with its counts already made.
        LIVE 2026-09-02, two games back to back: four by EIGHT again, and the
        rule it had known at 83% fell to nothing.
        """
        return frozenset(
            where
            for where, moved in self.rearranged.items()
            if moved > self.arrived.get(where, 0)
            and self._stood_in.get(where, 0) > 1
            and (where not in self._carried or where in self._seen_again)
        )

    def the_things_that_report(self) -> frozenset[tuple[int, int]]:
        """The places that show her something new each time. A score, a clock.

        More arrivals than rearrangements, and more than a place of the thing
        would show by chance at the rate every other place has shown them.
        """
        changed = set(self.arrived) | set(self.rearranged)
        reporting: set[tuple[int, int]] = set()
        while True:
            # A readout already found is not evidence about the thing's own
            # rate. Left in, a score and a move counter each hide the other.
            others = changed - reporting
            came_all = sum(self.arrived.get(where, 0) for where in others)
            moved_all = sum(self.rearranged.get(where, 0) for where in others)
            found = set()
            for where in others:
                came, moved = self.arrived.get(where, 0), self.rearranged.get(where, 0)
                if came <= moved:
                    continue
                chance = at_least_as_many(came, came + moved, came_all - came, moved_all - moved)
                if chance < WRONG_BY_CHANCE / (len(changed) * max(1, self.acts)):
                    found.add(where)
            if not found:
                return frozenset(reporting)
            reporting |= found

    def settled(self) -> bool:
        """Whether she has watched enough acts for the split to mean anything.

        One act cannot separate them: everything on the screen either changed
        or did not. The two groups have to have disagreed at least once, which
        is a thing that happened rather than a number chosen.
        """
        return self.acts > 1 and bool(self.the_thing_itself())

    def as_memory(self) -> dict[str, Any]:
        return {
            "acts": self.acts,
            "rearranged": {f"{x},{y}": n for (x, y), n in self.rearranged.items()},
            "arrived": {f"{x},{y}": n for (x, y), n in self.arrived.items()},
            "stood_in": {f"{x},{y}": n for (x, y), n in self._stood_in.items()},
            "rose": {f"{x},{y}": n for (x, y), n in self._rose.items()},
            "fell": {f"{x},{y}": n for (x, y), n in self._fell.items()},
        }

    @classmethod
    def from_memory(cls, held: Any, trust: float = 1.0) -> "MovesWithinItself":
        """What she found last time, discounted. A page can be rebuilt."""
        if not isinstance(held, dict):
            return cls()
        share = max(0.0, min(1.0, float(trust)))

        def places(counts: Any) -> dict[tuple[int, int], int]:
            out: dict[tuple[int, int], int] = {}
            if not isinstance(counts, dict):
                return out
            for key, count in counts.items():
                try:
                    x, y = (int(part) for part in str(key).split(","))
                except (TypeError, ValueError):
                    # not a failure: a key that is not a place is not one.
                    continue
                kept = int(float(count) * share)
                if kept:
                    out[(x, y)] = kept
            return out

        rearranged = places(held.get("rearranged"))
        return cls(
            rearranged=rearranged,
            arrived=places(held.get("arrived")),
            acts=int(float(held.get("acts") or 0) * share),
            _stood_in=places(held.get("stood_in")),
            _rose=places(held.get("rose")),
            _fell=places(held.get("fell")),
            _carried=frozenset(rearranged),
        )


def what_moved_within(
    before: Mapping[tuple[int, int], str],
    after: Mapping[tuple[int, int], str],
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    """One act's worth: what rearranged, and what arrived from nowhere.

    Useful on its own where there is only one act to go on, and honest about
    what that is worth — one act separates nothing reliably, which is why the
    counts above are kept across several.
    """
    watching = MovesWithinItself()
    watching.saw(before, after)
    return watching.the_thing_itself(), watching.the_things_that_report()

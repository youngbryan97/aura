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

Nothing here needs to know which is which in advance, and nothing here has a
threshold in it. Each place keeps two counts, and a place belongs to the thing
when its changes have more often been rearrangements than arrivals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MovesWithinItself", "what_moved_within"]


@dataclass
class MovesWithinItself:
    """Which places hold things that move about, and which report."""

    #: How often what appeared at a place was already somewhere in the region.
    rearranged: dict[tuple[int, int], int] = field(default_factory=dict)
    #: How often what appeared had not been anywhere in it.
    arrived: dict[tuple[int, int], int] = field(default_factory=dict)
    acts: int = 0

    def saw(
        self,
        before: Mapping[tuple[int, int], str],
        after: Mapping[tuple[int, int], str],
    ) -> None:
        """Take one act's worth of evidence.

        A value is counted as rearranged when it was in the region before and
        is no longer where it was, which is what moving means. Counting mere
        presence would call an unchanged place a rearrangement and the whole
        screen would qualify.
        """
        if not before or not after:
            return
        self.acts += 1
        was = Counter(text for text in before.values() if text)
        for where, text in after.items():
            if not text or before.get(where) == text:
                continue
            if was.get(text):
                self.rearranged[where] = self.rearranged.get(where, 0) + 1
            else:
                self.arrived[where] = self.arrived.get(where, 0) + 1

    def the_thing_itself(self) -> frozenset[tuple[int, int]]:
        """The places whose contents move about rather than arrive.

        A place that has never changed is not in it, and that is deliberate:
        furniture inside a board's outline never changes, which is exactly how
        it got into the outline in the first place.
        """
        return frozenset(
            where
            for where, moved in self.rearranged.items()
            if moved > self.arrived.get(where, 0)
        )

    def the_things_that_report(self) -> frozenset[tuple[int, int]]:
        """The places that show her something new each time. A score, a clock."""
        return frozenset(
            where
            for where, came in self.arrived.items()
            if came > self.rearranged.get(where, 0)
        )

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

        return cls(
            rearranged=places(held.get("rearranged")),
            arrived=places(held.get("arrived")),
            acts=int(float(held.get("acts") or 0) * share),
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

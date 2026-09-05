"""The few acts she takes, out of all the ones she could.

Somebody clearing 2048 in 989 moves pressed two of the four keys almost
exclusively, and pressed a third only when the board left them nothing else.
Asked what they were doing they described a shape they were keeping — but the
shape is not what they were DOING. What they were doing was reaching for the
same two things every time.

Measured on the game itself, over forty runs, that distinction is most of the
difference between playing and flailing:

    taking any legal move                       mean best tile   91
    keeping the largest values on one edge                      136
    reaching for two of the four, forced off only when stuck    198

The state-side property is real and it helps. The act-side habit helps more,
and it is a different kind of thing: not a fact about the world but a
disposition of hers, which is why no amount of looking at the world finds it.
It is found by trying.

So each act is given a turn at being the one she reaches for, and how those
stretches went is what settles which ones she keeps reaching for. A stretch
rather than a move, because what a habit is worth does not show up in the step
that follows it — that is exactly what makes it a habit rather than a choice.
The ones that paid better than the middle are the ones she keeps, so there is
no number here saying how many to keep: the measurements say.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TheOnesSheReachesFor"]


@dataclass
class TheOnesSheReachesFor:
    """Which acts have paid when she leant on them."""

    #: How many stretches each way of leaning has had.
    tried: dict[tuple[str, ...], int] = field(default_factory=dict)
    #: What those stretches came to, added up.
    paid: dict[tuple[str, ...], float] = field(default_factory=dict)
    #: What this stretch is leaning on, and what it has come to so far.
    leaning_on: tuple[str, ...] = ()
    so_far: float = 0.0

    def worth(self, these: Sequence[str]) -> float:
        """What leaning on them has come to, per stretch, by Laplace's rule.

        Something nobody has leant on yet is worth the middle rather than
        nothing, so it gets its turn without anybody scheduling one.
        """
        key = tuple(sorted(these))
        return (self.paid.get(key, 0.0) + 1.0) / (self.tried.get(key, 0) + 2.0)

    def ways_of_leaning(self, acts: Sequence[str]) -> list[tuple[str, ...]]:
        """One act at a time, and then two.

        Two, because what pays is not always any of them on its own. In 2048,
        down is the best single thing to lean on and right is fourth of four,
        and down WITH right is better than down with anything — they keep the
        same corner, and neither of them says so alone. Stopping at pairs is
        not a cap on how many she can end up reaching for; it is a bound on
        how many she has to TRY, and pairs are the smallest number that can
        show a thing acts do together.
        """
        singles = [(one,) for one in sorted(acts)]
        pairs = [
            (a, b)
            for at, a in enumerate(sorted(acts))
            for b in sorted(acts)[at + 1 :]
        ]
        return singles + pairs

    def start_a_stretch(self, acts: Sequence[str]) -> tuple[str, ...]:
        """Take up whatever has least evidence, or the best when all are tried.

        Least evidence first, because something nobody has leant on could be
        the thing, and it costs one stretch to find out. After that the
        measurements lead.
        """
        if not acts:
            return ()
        ways = self.ways_of_leaning(acts)
        untried = [one for one in ways if not self.tried.get(tuple(sorted(one)))]
        self.leaning_on = (
            untried[0] if untried else max(ways, key=lambda one: (self.worth(one), one))
        )
        self.so_far = 0.0
        return self.leaning_on

    def went(self, how_much: float) -> None:
        """Add what the last act came to, to the stretch in progress."""
        self.so_far += float(how_much)

    def end_the_stretch(self) -> None:
        """Write down what leaning on that came to."""
        if not self.leaning_on:
            return
        key = tuple(sorted(self.leaning_on))
        self.tried[key] = self.tried.get(key, 0) + 1
        self.paid[key] = self.paid.get(key, 0.0) + self.so_far
        self.leaning_on, self.so_far = (), 0.0

    def settled(self, acts: Sequence[str]) -> bool:
        """Whether every way of leaning has had its turn."""
        ways = self.ways_of_leaning(acts)
        return bool(ways) and all(
            self.tried.get(tuple(sorted(one))) for one in ways
        )

    def the_ones_that_paid(self, acts: Sequence[str]) -> tuple[str, ...]:
        """What she reaches for now, once everything has been tried once.

        The best of them outright, rather than everything above some middle.
        Half of four ways of leaning is two of them and half of ten is five,
        and five is not a habit — it is the whole keyboard with extra steps.
        Nothing here says how many she ends up with: if one act on its own
        measured best, that is what she reaches for.
        """
        if not self.settled(acts):
            return ()
        ways = self.ways_of_leaning(acts)
        best = max(ways, key=lambda one: (self.worth(one), one))
        return () if len(best) >= len(acts) else tuple(sorted(best))

    def the_ones_to_consider(self, available: Sequence[str]) -> tuple[str, ...]:
        """The ones she is leaning on, out of what is on offer.

        Leaning on two acts means those two are the moves she looks at — not
        that she plays the first of them and never the other. Which of them
        the position calls for is what looking ahead is for, and a habit that
        picks the member as well as the set leaves nothing for it to do.

        LIVE 2026-09-04 on the real board: leaning on a pair, the first of the
        pair pressed every single move, the choice narrowed to it before
        anything looked ahead, and every move announced as "the only thing
        available". The pair was measured to beat any single act because the
        two keep the same corner — a fact neither of them can show alone, and
        one that never showed at all while only one of them was ever played.

        Empty when none of them is available, which is the caller's signal
        that the habit has no purchase here.
        """
        return tuple(one for one in available if one in self.leaning_on)

    def which_to_take(self, available: Sequence[str], acts: Sequence[str]) -> str:
        """What she reaches for, or whatever is going when none of it is.

        Being forced off them is not a failure of the habit. It is the habit
        working: the times she cannot have what she wants are exactly the times
        worth spending something on, and the rest of the time she is not
        deciding at all.
        """
        if not available:
            return ""
        leaning = [one for one in self.leaning_on if one in available]
        if leaning:
            return leaning[0]
        liked = [one for one in self.the_ones_that_paid(acts) if one in available]
        return liked[0] if liked else available[0]

    def as_memory(self) -> dict[str, Any]:
        return {
            "tried": {" ".join(k): v for k, v in self.tried.items()},
            "paid": {" ".join(k): v for k, v in self.paid.items()},
        }

    @classmethod
    def from_memory(cls, held: Any, trust: float = 1.0) -> TheOnesSheReachesFor:
        """What she found last time. A habit is worth carrying between sittings."""
        if not isinstance(held, dict):
            return cls()
        share = max(0.0, min(1.0, float(trust)))
        tried: dict[tuple[str, ...], int] = {}
        paid: dict[tuple[str, ...], float] = {}
        for way, count in (held.get("tried") or {}).items():
            try:
                kept = int(float(count) * share)
            except (TypeError, ValueError):
                # not a failure: a count that is not a number is not a count.
                continue
            if kept:
                tried[tuple(str(way).split())] = kept
        for way, total in (held.get("paid") or {}).items():
            try:
                paid[tuple(str(way).split())] = float(total) * share
            except (TypeError, ValueError):
                continue
        return cls(tried=tried, paid={k: v for k, v in paid.items() if k in tried})

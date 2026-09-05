"""Things that are impossible until she has something else.

Minecraft, and the first hour of it is one shape repeated. Bare hands get wood.
Wood makes a pickaxe. A wooden pickaxe gets stone and nothing harder. A stone
pickaxe gets iron. An iron pickaxe gets diamond. Nobody is told this. What
they learn is that trying to mine the blue block with a wooden pick does
nothing at all — and that the thing standing between them and it is not skill
or effort or a better plan, it is an object they do not have.

Pokémon has the same shape with a different face: a boulder in the path is not
a hard boulder, it is a boulder until Strength, and then it is not there.

She had nothing for this. An act that does not work is written down as an act
that does not work HERE, which is true and useless, because the interesting
thing is not that it failed — it is that it would succeed given one thing, and
that the one thing is gettable. Without that, a wall is a wall for ever and
the way round it is never a thing to go and fetch.

What is learned is which acts started working after she got something. Not
declared, and not a recipe book: the same act, the same kind of place, working
now and not before, with one thing different. That is the gate, and once it is
known the plan for an impossible thing is to go and get the key rather than to
try harder.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhatOpensWhat"]


@dataclass
class WhatOpensWhat:
    """Which acts started working once she had which things."""

    #: act -> thing she had -> [times it worked, times it was tried]
    with_it: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    #: act -> [times it worked, times it was tried] with nothing in particular.
    without: dict[str, list[int]] = field(default_factory=dict)

    def she_tried(self, act: str, *, holding: Iterable[str], it_worked: bool) -> None:
        """One act, what she had at the time, and whether it did anything."""
        name = str(act)
        had = {str(one) for one in holding}
        seen = self.without.setdefault(name, [0, 0])
        seen[1] += 1
        seen[0] += bool(it_worked)
        for thing in had:
            count = self.with_it.setdefault(name, {}).setdefault(thing, [0, 0])
            count[1] += 1
            count[0] += bool(it_worked)

    def what_opens(self, act: str) -> tuple[str, ...]:
        """Things this act works with and does not work without.

        The comparison is the whole of it. A thing she happens to be carrying
        every time she does anything looks like a key to everything, and is a
        key to nothing — so what counts is that the act works while holding it
        and has failed while not.
        """
        name = str(act)
        held = self.with_it.get(name) or {}
        opened: list[str] = []
        for thing, (worked, tried) in held.items():
            if not worked or worked != tried:
                continue
            everywhere, ever = self.without.get(name, [0, 0])
            # It has to have failed somewhere this thing was not.
            if ever - tried > 0 and everywhere - worked == 0:
                opened.append(thing)
        return tuple(sorted(opened))

    def why_it_will_not_work(self, act: str, *, holding: Iterable[str]) -> tuple[str, ...]:
        """What she is missing for this, or nothing when she is not missing anything.

        This is the answer that turns a wall into an errand.
        """
        had = {str(one) for one in holding}
        return tuple(one for one in self.what_opens(act) if one not in had)

    def what_to_go_and_get(
        self, wanting: Sequence[str], *, holding: Iterable[str]
    ) -> tuple[str, ...]:
        """Everything standing between her and these acts, commonest first.

        A thing that opens three of the acts she wants is worth more than one
        that opens one, and nothing here needs telling that — it comes out of
        counting.
        """
        had = {str(one) for one in holding}
        wanted: dict[str, int] = {}
        for act in wanting:
            for thing in self.why_it_will_not_work(act, holding=had):
                wanted[thing] = wanted.get(thing, 0) + 1
        return tuple(
            thing for thing, _ in sorted(wanted.items(), key=lambda one: (-one[1], one[0]))
        )

    def as_memory(self) -> dict[str, Any]:
        return {
            "with_it": {
                act: {thing: list(count) for thing, count in things.items()}
                for act, things in self.with_it.items()
            },
            "without": {act: list(count) for act, count in self.without.items()},
        }

    @classmethod
    def from_memory(cls, held: Any) -> WhatOpensWhat:
        if not isinstance(held, dict):
            return cls()

        def counts(raw: Any) -> list[int]:
            try:
                return [int(raw[0]), int(raw[1])]
            except (TypeError, ValueError, IndexError):
                # not a failure: a count that is not a pair of numbers is not
                # a count, and starting it fresh loses nothing that was real.
                return [0, 0]

        return cls(
            with_it={
                str(act): {str(t): counts(c) for t, c in (things or {}).items()}
                for act, things in (held.get("with_it") or {}).items()
            },
            without={
                str(act): counts(c) for act, c in (held.get("without") or {}).items()
            },
        )

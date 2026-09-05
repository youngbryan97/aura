"""Doing the work before the thing arrives, because it can be seen coming.

Three of the recordings are mostly this and none of it looks like the thing
being prepared for.

A Pokémon player stands outside a gym and does not go in. They go and buy
potions, and level a particular creature, and teach it a particular move —
half an hour of work aimed at a fight that has not started, whose shape they
know because gyms have a shape. In Minecraft the sun goes down every twenty
minutes and the work is done in the light: a door, a wall, a bed. A Stellaris
player builds a fleet a decade before the war.

None of them is reacting. They are all acting on a thing that has not happened
yet, and the reason they can is that it has happened before and left a mark:
these are the things that were needed, and this is roughly when it comes.

She reacts. Something arrives, she deals with it, and she deals with it out of
whatever she happens to be holding — so the same shortage is discovered at the
same moment every time, and discovered too late every time, because the moment
it is discovered is the moment it matters. Nothing in her turned "this went
badly for want of X" into "get X first".

Two things learned and one asked. What has been wanted when this kind of thing
came, which is her own record read the other way round. Roughly how long there
usually is between them, which is the same record read again. And then, at any
moment: is one due, and what is she short of.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhatUsuallyComes"]


@dataclass
class WhatUsuallyComes:
    """What arrives, when it tends to, and what it wanted."""

    #: kind of thing -> what was wanted -> how many times it was wanted.
    wanted: dict[str, dict[str, int]] = field(default_factory=dict)
    #: kind of thing -> how many times it has come.
    came: dict[str, int] = field(default_factory=dict)
    #: kind of thing -> when it last came, and the gaps between.
    _last: dict[str, float] = field(default_factory=dict)
    _gaps: dict[str, list[float]] = field(default_factory=dict)

    def it_came(self, kind: str, *, at: float, needing: Iterable[str] = ()) -> None:
        """One of these arrived, when, and what it turned out to want."""
        name = str(kind)
        self.came[name] = self.came.get(name, 0) + 1
        was = self._last.get(name)
        if was is not None and at > was:
            self._gaps.setdefault(name, []).append(float(at) - was)
        self._last[name] = float(at)
        for one in needing:
            self.wanted.setdefault(name, {})[str(one)] = (
                self.wanted.setdefault(name, {}).get(str(one), 0) + 1
            )

    def what_it_wants(self, kind: str) -> tuple[str, ...]:
        """What this kind of thing has wanted, commonest first.

        Everything it has ever wanted, not only what it always wants: a thing
        needed on half the occasions is worth having, and being short of it
        half the time is the same shortage discovered twice.
        """
        seen = self.wanted.get(str(kind)) or {}
        return tuple(
            one for one, _ in sorted(seen.items(), key=lambda one: (-one[1], one[0]))
        )

    def how_long_between(self, kind: str) -> float:
        """The usual gap between these, or nothing where she cannot say.

        The middling gap rather than the average, because one very long wait
        should not persuade her she has all day.
        """
        gaps = sorted(self._gaps.get(str(kind)) or [])
        return gaps[len(gaps) // 2] if gaps else 0.0

    def due_in(self, kind: str, *, now: float) -> float:
        """How long until one is due, negative when it is overdue.

        Nothing where she has never seen two of them, because one sighting is
        an event and two are a rhythm.
        """
        usual = self.how_long_between(kind)
        was = self._last.get(str(kind))
        if not usual or was is None:
            return 0.0
        return (was + usual) - float(now)

    def what_to_get_first(
        self, kind: str, *, holding: Iterable[str]
    ) -> tuple[str, ...]:
        """What she is short of for the next one of these."""
        had = {str(one) for one in holding}
        return tuple(one for one in self.what_it_wants(kind) if one not in had)

    def worth_getting_ready(
        self, kinds: Sequence[str], *, now: float, holding: Iterable[str]
    ) -> list[tuple[str, float, tuple[str, ...]]]:
        """What is coming, soonest first, and what she is short of for it.

        Only things she is short of something for. Being ready for a thing she
        is already ready for is not preparation, it is worrying.
        """
        got = []
        for kind in kinds:
            short = self.what_to_get_first(kind, holding=holding)
            if not short:
                continue
            got.append((kind, self.due_in(kind, now=now), short))
        return sorted(got, key=lambda one: (one[1], one[0]))

    def as_memory(self) -> dict[str, Any]:
        return {
            "wanted": {k: dict(v) for k, v in self.wanted.items()},
            "came": dict(self.came),
            "last": dict(self._last),
            "gaps": {k: list(v) for k, v in self._gaps.items()},
        }

    @classmethod
    def from_memory(cls, held: Any) -> WhatUsuallyComes:
        if not isinstance(held, dict):
            return cls()

        def numbers(raw: Any) -> list[float]:
            out = []
            for one in raw or ():
                try:
                    out.append(float(one))
                except (TypeError, ValueError):
                    # not a failure: a gap that is not a number is not a gap.
                    continue
            return out

        return cls(
            wanted={
                str(k): {str(a): int(b) for a, b in (v or {}).items() if isinstance(b, int)}
                for k, v in (held.get("wanted") or {}).items()
            },
            came={str(k): int(v) for k, v in (held.get("came") or {}).items() if isinstance(v, int)},
            _last={str(k): float(v) for k, v in (held.get("last") or {}).items() if isinstance(v, (int, float))},
            _gaps={str(k): numbers(v) for k, v in (held.get("gaps") or {}).items()},
        )

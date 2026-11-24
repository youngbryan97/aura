"""Where practice pays: time to practise spent on the kinds of practice that have lately changed what she will do,
without leaving the others untried.

A learner with several ways to practise (rehearsing, preparing, going over what went wrong, working a small case out
to the end) and a little time between turns does best to spend it where practice has been changing something, the way
a player drills the part of their game that is moving. Each kind is credited when a round of it changed what she will
do, and charged when it ran and changed nothing. A round with nothing to do says nothing of its kind, and is not
counted. The kind tried next is the one whose record promises most once a margin is added for how little it has been
tried: the upper confidence bound of its share of rounds that paid. A kind never tried comes first. Later rounds
count for more than old ones (each round's record fades by FADES), because what pays changes as she learns.

Nothing here knows what is practised: a kind is a name.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["FADES", "WherePracticePays"]

#: How much of a kind's record each later round of any kind leaves: what paid a hundred rounds ago counts for little.
FADES = 0.98


@dataclass
class WherePracticePays:
    """For each kind of practice, its rounds and the rounds that paid, each fading as rounds go by."""

    tried: dict[str, float] = field(default_factory=dict)
    paid: dict[str, float] = field(default_factory=dict)

    def ranked(self, kinds: Sequence[str]) -> list[str]:
        """``kinds`` in the order to try them now: those never tried first, in the order given, then by the upper
        confidence bound of the share of their rounds that paid."""
        total = sum(self.tried.get(kind, 0.0) for kind in kinds)

        def promise(kind: str) -> float:
            tried = self.tried.get(kind, 0.0)
            if tried < 1e-9:
                return math.inf
            return self.paid.get(kind, 0.0) / tried + math.sqrt(2.0 * math.log(max(total, 1.0) + 1.0) / tried)

        order = {kind: i for i, kind in enumerate(kinds)}
        return sorted(kinds, key=lambda kind: (-promise(kind), order[kind]))

    def came_to(self, kind: str, paid: bool) -> None:
        """A round of ``kind`` that had something to do, and whether it changed what she will do."""
        for name in list(self.tried):
            self.tried[name] *= FADES
            self.paid[name] = self.paid.get(name, 0.0) * FADES
        self.tried[kind] = self.tried.get(kind, 0.0) + 1.0
        self.paid[kind] = self.paid.get(kind, 0.0) + (1.0 if paid else 0.0)

    def as_memory(self) -> dict[str, Any]:
        return {"tried": dict(self.tried), "paid": dict(self.paid)}

    @classmethod
    def from_memory(cls, held: Any) -> WherePracticePays:
        if not isinstance(held, dict):
            return cls()
        return cls(tried={str(k): float(v) for k, v in (held.get("tried") or {}).items()},
                   paid={str(k): float(v) for k, v in (held.get("paid") or {}).items()})

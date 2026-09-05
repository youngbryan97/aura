"""What works against what, learned rather than looked up.

Pokémon runs on a table: water beats fire, fire beats grass, grass beats
water. Nobody playing it has the table. They have a hundred fights, and after
enough of them the table is in their head — and it is not really a table about
elements, it is a table about KINDS. What matters is that the thing in front
of you belongs to a kind, that your options belong to kinds, and that some
pairs of kinds go one way every time.

That shape is everywhere and she had no room for it. She learns which acts
work HERE, which is a fact about one place and dies with it. She learns which
act works generally, which averages over places that have nothing in common.
Neither can say the thing that is actually true: this act works on that kind
of thing, and that act does not, and the kind is visible before you commit.

So an act's worth is not one number. It is one number per kind of thing it is
used against, which is why a hundred fights teaches something a thousand
identical ones cannot: the table needs variety, and it is the variety that
carries the information.

What comes out is usable in the way that matters — given what is in front of
her, the acts that have gone well against its kind, best first, with the ones
she has never tried against it marked as untried rather than as bad. An act
that has never met this kind is not a bad act. It is an experiment, and
telling those apart is the difference between learning and superstition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhatBeatsWhat"]


@dataclass
class WhatBeatsWhat:
    """How each act has gone against each kind of thing."""

    #: kind -> act -> [times it went well, times tried]
    against: dict[str, dict[str, list[int]]] = field(default_factory=dict)

    def it_went(self, act: str, *, against: str, well: bool) -> None:
        count = self.against.setdefault(str(against), {}).setdefault(str(act), [0, 0])
        count[1] += 1
        count[0] += bool(well)

    def how_it_goes(self, act: str, *, against: str) -> float:
        """How that act goes against that kind, by Laplace's rule.

        A half where she has never tried it, which is neither a promise nor a
        warning — it is the honest middle, and it is what makes an untried act
        worth trying rather than worth avoiding.
        """
        well, tried = (self.against.get(str(against)) or {}).get(str(act), [0, 0])
        return (well + 1) / (tried + 2)

    def has_tried(self, act: str, *, against: str) -> int:
        return (self.against.get(str(against)) or {}).get(str(act), [0, 0])[1]

    def in_order(self, acts: Sequence[str], *, against: str) -> list[tuple[str, float, int]]:
        """Her acts against this kind, best first, with how much she knows.

        The third number is what stops this being superstition. An act at
        two thirds from three fights and one at two thirds from thirty are the
        same number and not the same claim.
        """
        return sorted(
            (
                (one, self.how_it_goes(one, against=against), self.has_tried(one, against=against))
                for one in acts
            ),
            key=lambda one: (-one[1], -one[2], one[0]),
        )

    def worth_finding_out(self, acts: Sequence[str], *, against: str) -> tuple[str, ...]:
        """Acts she has never used against this kind.

        An act that has never met this kind is not a bad act, it is an
        experiment — and a table with a hole in it stays holed until somebody
        goes and fills it.
        """
        return tuple(
            one for one in acts if not self.has_tried(one, against=against)
        )

    def what_it_knows_about(self, kind: str) -> str:
        seen = self.against.get(str(kind)) or {}
        if not seen:
            return f"nothing yet about {kind}"
        best = max(seen, key=lambda one: self.how_it_goes(one, against=kind))
        return (
            f"against {kind}, {best} goes best "
            f"({self.how_it_goes(best, against=kind):.0%} of {self.has_tried(best, against=kind)})"
        )

    def as_memory(self) -> dict[str, Any]:
        return {
            "against": {
                kind: {act: list(count) for act, count in acts.items()}
                for kind, acts in self.against.items()
            }
        }

    @classmethod
    def from_memory(cls, held: Any) -> WhatBeatsWhat:
        if not isinstance(held, dict):
            return cls()
        got: dict[str, dict[str, list[int]]] = {}
        for kind, acts in (held.get("against") or {}).items():
            if not isinstance(acts, dict):
                continue
            kept: dict[str, list[int]] = {}
            for act, count in acts.items():
                try:
                    kept[str(act)] = [int(count[0]), int(count[1])]
                except (TypeError, ValueError, IndexError):
                    # not a failure: a count that is not a pair is not a count.
                    continue
            if kept:
                got[str(kind)] = kept
        return cls(against=got)

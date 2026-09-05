"""Acts with their own limited supply, and acts that change what comes after.

Two things from competitive Pokémon that look like details and are not.

Every move has its own count of uses. Not a shared pool she draws from — each
one separately, so the strong move runs out while the weak one is still there,
and a fight can be lost by a party that never lost a turn. Anybody who has
watched a long game knows the shape: the thing that would win it has three
uses left and the fight needs four.

And the moves that decide most games do no damage at all. A burn halves what
they can hit for; paralysis halves how often they move. Neither changes the
position now. Both change what the other side is ABLE to do for the rest of
it, and that is worth more than a hit precisely because it does not wear off
when the turn ends.

She had neither. Every act cost the same nothing and could be taken for ever,
so an act that should have been saved was spent on whatever came first. And
every act was weighed by the state it produced, so an act whose entire point
is that the other side is worse at everything afterwards scored as though it
had done nothing — because in the state it produced, nothing had happened.

Both are the same correction from two sides: what an act is worth is not
contained in the situation immediately after it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["WhatEachActHasLeft", "WhatItDoesToThem"]


@dataclass
class WhatEachActHasLeft:
    """How many uses each act has of its own, and what running out costs."""

    left: dict[str, int] = field(default_factory=dict)
    #: How many she started with, so "nearly out" can be told from "few".
    started_with: dict[str, int] = field(default_factory=dict)

    def she_has(self, act: str, many: int) -> None:
        self.left[str(act)] = max(0, int(many))
        self.started_with.setdefault(str(act), max(0, int(many)))

    def she_used(self, act: str) -> None:
        name = str(act)
        if name in self.left:
            self.left[name] = max(0, self.left[name] - 1)

    def can_still(self, act: str) -> bool:
        """Whether it is available at all. An act with none left is not an act."""
        return self.left.get(str(act), 1) > 0

    def what_is_left(self, acts: Iterable[str]) -> tuple[str, ...]:
        return tuple(one for one in acts if self.can_still(one))

    def how_much_is_left(self, act: str) -> float:
        """The share of it remaining, so nearly-out is told from merely-few.

        Three of three hundred and three of four are the same number and not
        the same situation, and an act kept for the moment it is needed has to
        know which it is in.
        """
        name = str(act)
        began = self.started_with.get(name, 0)
        return (self.left.get(name, 0) / began) if began else 1.0

    def worth_saving(self, act: str, *, for_what: float, this_is_worth: float) -> bool:
        """Whether to keep this for later rather than spend it now.

        Spend it when what it is worth here is at least its share of what it
        is being kept for. Where plenty is left the share is small and she
        spends freely; where one is left the share is the whole of it and she
        will only spend it on the thing it was saved for.
        """
        left = self.left.get(str(act), 0)
        if left <= 0:
            return False
        return this_is_worth < (float(for_what) / left)

    def running_out(self, acts: Sequence[str]) -> tuple[str, ...]:
        """Acts with less than half of themselves left, emptiest first."""
        return tuple(
            one
            for one in sorted(acts, key=lambda a: (self.how_much_is_left(a), a))
            if self.how_much_is_left(one) < 0.5
        )


@dataclass
class WhatItDoesToThem:
    """What an act leaves behind in what somebody can do afterwards."""

    #: act -> what it changes about them -> by how much, as a share.
    lasting: dict[str, dict[str, float]] = field(default_factory=dict)

    def it_left(self, act: str, *, changing: str, by: float) -> None:
        """One act, and what it turned out to change about them lastingly."""
        self.lasting.setdefault(str(act), {})[str(changing)] = float(by)

    def what_it_leaves(self, act: str) -> Mapping[str, float]:
        return dict(self.lasting.get(str(act)) or {})

    def worth_beyond_now(self, act: str, *, turns_left: float) -> float:
        """What the lasting part is worth, given how long there is to use it.

        A thing that halves what they can do is worth half of everything they
        would have done — so it is worth more the earlier it lands, and worth
        nothing on the last turn. That is why it is spent early by people who
        know the game and hoarded by people who do not.
        """
        return sum(self.what_it_leaves(act).values()) * max(0.0, float(turns_left))

    def describe(self, act: str) -> str:
        leaves = self.what_it_leaves(act)
        if not leaves:
            return f"{act} changes nothing that outlasts the turn"
        said = ", ".join(f"{what} by {by:.0%}" for what, by in sorted(leaves.items()))
        return f"{act} leaves them worse at {said}"

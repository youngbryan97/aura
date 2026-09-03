"""Whether this is a world worth memorising, or one that must be played.

The two hard platformers are learned by rote. Ghosts 'n Goblins and Ninja
Gaiden put the same enemy in the same place every time, so the way through is
to remember it, and people who finish them are not reacting — they are
recalling. The same effort spent on 2048 buys nothing at all, because the
board is dealt fresh every time and there is nothing to remember.

Both are worth doing and each is a waste in the other's world. Memorising a
shuffled world fills her up with facts that will not recur. Playing a fixed
world by policy throws away the one thing that would have made it easy.

She had no way to ask which she was in. So which effort to spend was never a
question, and the answer she happened to have was the answer she used
everywhere.

It is a testable thing, not a judgement. Do the same acts from the same
situation and see whether the same thing happens. If it does, this world
repeats and what she learns about a place is worth keeping about that place.
If it does not, only what she learns about KINDS of place is worth anything,
and the rest is noise she would be storing at her own expense.

The answer is a matter of degree and is given as one. Most real worlds are
partly both — a fixed layout with wandering things in it — and a number
between says how much of each, which is more use than a verdict.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DoesItRepeat"]


@dataclass
class DoesItRepeat:
    """How often the same act from the same place has done the same thing."""

    #: (place, act) -> what happened -> how often.
    after: dict[tuple[str, str], dict[str, int]] = field(default_factory=dict)

    def she_saw(self, place: Hashable, act: str, became: Hashable) -> None:
        key = (repr(place), str(act))
        got = repr(became)
        self.after.setdefault(key, {})[got] = self.after.setdefault(key, {}).get(got, 0) + 1

    @property
    def tried_twice(self) -> int:
        """How many place-and-act pairs she has done more than once.

        Nothing can be said until something has been repeated, and this is the
        number that says whether the question has been asked at all.
        """
        return sum(1 for seen in self.after.values() if sum(seen.values()) > 1)

    def how_much_it_repeats(self) -> float:
        """Between nought and one: how often the same thing happened again.

        Weighted by how often each pair was tried, so a pair done thirty times
        counts for more than one done twice — and a half where nothing has
        been repeated, which is the honest answer to a question not yet asked.
        """
        same = 0
        total = 0
        for seen in self.after.values():
            times = sum(seen.values())
            if times < 2:
                continue
            # How often a second look agreed with a first, out of the pairs of
            # looks there were.
            agreed = sum(one * (one - 1) for one in seen.values())
            total += times * (times - 1)
            same += agreed
        return (same / total) if total else 0.5

    def worth_remembering_places(self) -> bool:
        """Whether what she learns about a PLACE is worth keeping about it.

        The bar is that it repeats more often than not, which is exactly the
        point at which remembering a place beats guessing about it — and not a
        level anybody chose.
        """
        return self.tried_twice > 0 and self.how_much_it_repeats() > 0.5

    def what_to_spend_it_on(self) -> str:
        if not self.tried_twice:
            return "nothing done twice yet, so there is nothing to say"
        if self.worth_remembering_places():
            return (
                f"this repeats ({self.how_much_it_repeats():.0%}) — worth learning "
                "the places themselves"
            )
        return (
            f"this is dealt fresh ({self.how_much_it_repeats():.0%} repeats) — only "
            "what holds across places is worth keeping"
        )

    def as_memory(self) -> dict[str, Any]:
        return {
            "after": {
                f"{place}\x1f{act}": dict(seen)
                for (place, act), seen in self.after.items()
            }
        }

    @classmethod
    def from_memory(cls, held: Any) -> "DoesItRepeat":
        if not isinstance(held, dict):
            return cls()
        after: dict[tuple[str, str], dict[str, int]] = {}
        for key, seen in (held.get("after") or {}).items():
            place, _, act = str(key).partition("\x1f")
            if not act or not isinstance(seen, dict):
                continue
            kept = {str(k): int(v) for k, v in seen.items() if isinstance(v, int)}
            if kept:
                after[(place, act)] = kept
        return cls(after=after)

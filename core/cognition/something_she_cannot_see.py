"""When the world stops being a function of what she can see.

Every rule she can form maps what is in front of her to what happens next. Such
a rule cannot account for a world where the same thing, done twice from the
same place, comes out differently — and there is no search long enough to fix
that, because the fault is not in the search. Something is there that she is
not reading.

The evidence is exact and needs no judgement: two occasions agreeing on
everything observed and disagreeing on the outcome. One such pair proves a
hidden quantity exists. Counting how many values it must take is then the
smallest number that makes the record a function again, which is the largest
number of different outcomes any one observed situation produced.

What she does next depends on something she can measure. If the hidden values
run in a cycle, the quantity is not hidden at all — it is the step number, and
she can compute it. A latent variable that turns out to be predictable becomes
a coordinate, and the world is a function again in the wider reading.

If they do not run in a cycle, the finding is the opposite and just as useful:
the world has something in it she does not control and cannot foresee. A tile
appears where she did not put one. Knowing that stops her rewriting a model
that was right, which is the failure this exists to prevent — every wrong
prediction after a random event looks exactly like a wrong rule.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "WhatSheCannotSee",
    "a_coordinate_she_can_compute",
    "what_she_cannot_see",
]

logger = logging.getLogger("Aura.SomethingSheCannotSee")


@dataclass(frozen=True)
class WhatSheCannotSee:
    """What the record proves about a quantity she is not reading."""

    #: Situations that came out more than one way, with the outcomes they gave.
    disagreements: tuple[tuple[Any, tuple[Any, ...]], ...] = ()
    #: How many values the hidden quantity must take. One means nothing hidden.
    how_many: int = 1
    #: Its value at each step of the record, in the order they happened.
    values: tuple[int, ...] = ()
    #: The cycle it runs in, if it runs in one. Nought means it does not.
    every: int = 0
    #: How many whole cycles that rests on, so a small number can be read.
    over: int = 0
    disagreeing_steps: tuple[int, ...] = field(default=())

    @property
    def anything(self) -> bool:
        return self.how_many > 1

    @property
    def she_can_compute_it(self) -> bool:
        """Whether the hidden quantity is the step number in disguise."""
        return self.anything and self.every > 0

    def __str__(self) -> str:
        if not self.anything:
            return "everything she saw follows from what she could see"
        if self.she_can_compute_it:
            return (
                f"one thing she was not reading, taking {self.how_many} values, "
                f"and it runs every {self.every} steps over {self.over} whole "
                "cycles — so she can compute it"
            )
        return (
            f"one thing she was not reading, taking {self.how_many} values, in "
            f"no cycle she can find over {len(self.values)} steps — the world "
            "has something in it she does not control"
        )


def _key(one: Any) -> Any:
    try:
        hash(one)
    except TypeError:
        return repr(one)
    return one


def what_she_cannot_see(
    history: Sequence[tuple[Any, Any]] | Sequence[tuple[Any, Any, Any]],
) -> WhatSheCannotSee:
    """Weigh a record for a quantity that is not in it.

    Each entry is what she saw and what happened, or what she saw, what she
    did, and what happened. Order matters: it is the only thing that can show a
    cycle.
    """
    steps: list[tuple[Any, Any]] = []
    for entry in history:
        if len(entry) == 3:
            seen, did, then = entry
            steps.append((_key((_key(seen), _key(did))), _key(then)))
        else:
            seen, then = entry
            steps.append((_key(seen), _key(then)))

    outcomes: dict[Any, list[Any]] = {}
    for where, then in steps:
        if then not in outcomes.setdefault(where, []):
            outcomes[where].append(then)

    how_many = max((len(one) for one in outcomes.values()), default=1)
    if how_many < 2:
        return WhatSheCannotSee(how_many=1, values=tuple([0] * len(steps)))

    # Its value at each step is which of that situation's outcomes came up. Any
    # other labelling needs more values to say the same thing.
    values = tuple(outcomes[where].index(then) for where, then in steps)
    disagreeing = tuple(
        at for at, (where, _then) in enumerate(steps) if len(outcomes[where]) > 1
    )
    every, over = _the_cycle_it_runs_in(values)
    return WhatSheCannotSee(
        disagreements=tuple(
            (where, tuple(gave)) for where, gave in outcomes.items() if len(gave) > 1
        ),
        how_many=how_many,
        values=values,
        every=every,
        over=over,
        disagreeing_steps=disagreeing,
    )


def _the_cycle_it_runs_in(values: Sequence[int]) -> tuple[int, int]:
    """The shortest cycle the hidden values run in, and how many whole ones.

    A cycle claimed on less than two whole turns of it is a claim about one
    turn, which every sequence satisfies. Two is not a threshold chosen for
    comfort; it is the smallest number for which the word repeat means
    anything.
    """
    found = list(values)
    if len(found) < 4:
        return 0, 0
    for every in range(1, len(found) // 2 + 1):
        if all(found[at] == found[at - every] for at in range(every, len(found))):
            turns = len(found) // every
            return (every, turns) if turns >= 2 else (0, 0)
    return 0, 0


def a_coordinate_she_can_compute(found: WhatSheCannotSee) -> Any:
    """The hidden quantity as something she reads, or nothing if it is not.

    Handing back a function she can call at any step, so the thing that was
    latent joins what she observes rather than staying a note about the record.
    """
    if not found.she_can_compute_it:
        return None
    cycle = found.values[: found.every]

    def what_it_is_at(step: int) -> int:
        return cycle[int(step) % len(cycle)]

    what_it_is_at.__doc__ = (
        f"The quantity she could not read, worked out: it runs {list(cycle)} and "
        f"repeats every {found.every} steps."
    )
    return what_it_is_at

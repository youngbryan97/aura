"""Whether the NAME bought anything, or only the body inside it did.

The control the whole literature on library learning turns on, and the one a
compression story usually skips: take the name away and leave the body, then
ask again. If nothing changes, what did the work was the term, and the term was
going to be there either way.

Where it does not apply, and why that is worth writing down
-----------------------------------------------------------
It does not apply to a head. Inline expansion means replacing every use of a
name with its body, and a positional term CANNOT contain a floor term — that is
what makes a head a head rather than an abbreviation. There is no expansion to
compare against, and a control with nothing on its other side is not a control.

The first version of this module made that mistake and got a number out of it.
It compared where a head appears in the positional enumeration against how many
candidates the head search walks, called sixteen worse than twelve, and refused
two heads that work. Two different things counted in two different units, and
the comparison read as evidence because both were integers.

What replaces it for a head is the control this codebase already had:
`which_kind_of_growth` takes the head out and looks for the behaviour in the
language without it, and a head that turns out to be a shorter name for
something already sayable comes back out. That is the same question — did the
name add anything — asked where an answer exists.

Where it does apply
-------------------
Forward, across families, in one unit. Having a term as a HEAD costs a shape at
every node of every term; having it as a library LEAF costs one more thing to
try in a hole. Which is worth more is a question about the next family, not
about the one it was written for, so it is measured by asking the next family
in both conditions and counting the candidates each walks.

That is a measurement rather than an admission gate, and it lives with the
campaigns for the same reason: at the moment a head is admitted there is no
next family to ask.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["WhatTheNameBought", "what_the_name_bought"]

logger = logging.getLogger("Aura.WhatTheNameBought")


@dataclass(frozen=True)
class WhatTheNameBought:
    """What a term is worth as a head, against what it is worth as a leaf."""

    name: str
    #: Families solved with the term in the grammar as a head, and the
    #: candidates that cost.
    solved_as_a_head: int
    walked_as_a_head: int
    #: The same, with the term in the library as a leaf instead.
    solved_as_a_leaf: int
    walked_as_a_leaf: int
    over: int

    @property
    def the_name_bought_something(self) -> bool:
        """Whether the shape at every node earned itself.

        More families solved is the first question and cheaper is the second,
        in that order: a head that solves the same number for less is worth
        having, and one that solves fewer is not, however cheap it was.
        """
        if self.solved_as_a_head != self.solved_as_a_leaf:
            return self.solved_as_a_head > self.solved_as_a_leaf
        return self.walked_as_a_head < self.walked_as_a_leaf

    def describes(self) -> str:
        verdict = (
            "the name bought something"
            if self.the_name_bought_something
            else "the body was doing the work"
        )
        return (
            f"{self.name!r} over {self.over} famil(ies): as a head "
            f"{self.solved_as_a_head} solved for {self.walked_as_a_head:,} "
            f"candidate(s); as a leaf {self.solved_as_a_leaf} for "
            f"{self.walked_as_a_leaf:,} — {verdict}"
        )


def what_the_name_bought(
    name: str,
    body: Any,
    families: Sequence[Any],
    *,
    ask: Callable[[Any, bool, Any], tuple[bool, int]],
) -> WhatTheNameBought:
    """Ask each family twice — once with the term as a head, once as a leaf.

    ``ask`` takes the family, whether the term is installed as a head, and the
    term, and gives back whether it was solved and what that cost in
    candidates. The caller owns what solving means; this owns the comparison.
    """
    as_a_head = [ask(one, True, body) for one in families]
    as_a_leaf = [ask(one, False, body) for one in families]
    found = WhatTheNameBought(
        name=str(name),
        solved_as_a_head=sum(1 for ok, _cost in as_a_head if ok),
        walked_as_a_head=sum(cost for ok, cost in as_a_head if ok),
        solved_as_a_leaf=sum(1 for ok, _cost in as_a_leaf if ok),
        walked_as_a_leaf=sum(cost for ok, cost in as_a_leaf if ok),
        over=len(families),
    )
    logger.info("what the name bought — %s", found.describes())
    return found

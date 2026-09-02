"""Several things going on at once, and one move to spend.

Watching a Go game between two very strong players: there are never fewer than
three or four separate fights on the board, each with its own state, and every
single move is an answer to the question of WHICH of them to spend it on.
Nothing in the position says. Both players can see all of it; what separates
them is where they put the next stone.

Everything built here so far has one situation in it. A board, a screen, a
task — she looks at the thing, works out what moves it, and moves it. That is
the whole shape of the loop, and it cannot express what most of a real problem
is: several things half done, none of them urgent enough to finish, all of
them getting worse at different rates.

What decides it is not which is biggest. It is what it costs to leave each one
alone, which is a different question with a different answer, and it is one
she can work out rather than be told: how good it would be if she acted here,
against how good it would be if she did not and somebody else did. The gap is
what her move is worth here. She spends it where the gap is widest.

That has two consequences worth naming because they fall out rather than being
put in. A large situation nobody is threatening is worth nothing to play in
this turn, which is why strong players walk away from their biggest group. And
a small situation about to be lost outright can be worth more than anything
else on the board, because the gap there is the whole of it.

The other half is what a move buys besides its own result. An act the other
side must answer leaves the answer predictable, so the move after it is hers
to place as well — the initiative, which is the thing Go players call sente
and spend whole games trading. It is not a separate mechanism: an act that
leaves the other side few ways to reply is exactly what her own counting of
what is still open already measures.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["WhatItCostsToLeaveIt", "what_it_costs_to_leave_them", "where_to_spend_it"]


@dataclass(frozen=True)
class WhatItCostsToLeaveIt:
    """One of the things going on, and what her next move is worth in it."""

    name: str
    #: How good it gets if she acts here.
    if_she_acts: float
    #: How good it stays if she does not, and the other side does.
    if_she_does_not: float
    #: Her best act here, when there is one.
    act: Any = None
    #: How many ways the other side could answer that act. Few means the
    #: answer is predictable and the move after it is hers to place.
    ways_they_could_answer: int = 0

    @property
    def worth(self) -> float:
        """What her move is worth spent here rather than anywhere else."""
        return self.if_she_acts - self.if_she_does_not

    @property
    def keeps_the_move(self) -> bool:
        """Whether acting here leaves the next choice with her.

        One way to answer is no choice at all, so what happens next is
        something she already knows. Nought means they cannot answer, which is
        stronger still.
        """
        return self.ways_they_could_answer <= 1

    def describe(self) -> str:
        held = ", and keeps the move" if self.keeps_the_move else ""
        return f"{self.name}: worth {self.worth:+.2f}{held}"


def what_it_costs_to_leave_them(
    situations: dict[str, Any],
    *,
    her_acts: Callable[[Any], Sequence[Any]],
    step: Callable[[Any, Any], Any | None],
    how_good: Callable[[Any], float],
    their_acts: Callable[[Any], Sequence[Any]] | None = None,
    their_step: Callable[[Any, Any], Any | None] | None = None,
) -> list[WhatItCostsToLeaveIt]:
    """What her next move is worth in each of them, the widest gap first.

    ``how_good`` is her own judgement of a situation, whatever that is here.
    Nothing in this file has an opinion about what makes one good; it only
    subtracts.

    Where there is nobody on the other side, leaving a situation alone leaves
    it as it is, and the gap is simply what she could improve it by. That is
    the right answer for a world that does not push back, and it is the same
    subtraction.
    """
    weighed: list[WhatItCostsToLeaveIt] = []
    for name, where in situations.items():
        acting: float | None = None
        best_act: Any = None
        for act in her_acts(where):
            after = step(where, act)
            if after is None:
                continue
            worth = how_good(after)
            if acting is None or worth > acting:
                acting, best_act = worth, act
        if acting is None:
            continue
        # What it comes to if she spends the move elsewhere. The other side
        # gets to choose, so it is their best rather than an average: hoping
        # they pick badly is not a plan.
        leaving = how_good(where)
        if their_acts is not None and their_step is not None:
            theirs = [
                their_step(where, one)
                for one in their_acts(where)
            ]
            got = [how_good(one) for one in theirs if one is not None]
            if got:
                leaving = min(got)
        answers = 0
        if best_act is not None and their_acts is not None and their_step is not None:
            after = step(where, best_act)
            if after is not None:
                answers = len(
                    {
                        repr(their_step(after, one))
                        for one in their_acts(after)
                        if their_step(after, one) is not None
                    }
                )
        weighed.append(
            WhatItCostsToLeaveIt(
                name=name,
                if_she_acts=acting,
                if_she_does_not=leaving,
                act=best_act,
                ways_they_could_answer=answers,
            )
        )
    weighed.sort(key=lambda one: (-one.worth, not one.keeps_the_move, one.name))
    return weighed


def where_to_spend_it(
    situations: dict[str, Any],
    *,
    her_acts: Callable[[Any], Sequence[Any]],
    step: Callable[[Any, Any], Any | None],
    how_good: Callable[[Any], float],
    their_acts: Callable[[Any], Sequence[Any]] | None = None,
    their_step: Callable[[Any, Any], Any | None] | None = None,
) -> WhatItCostsToLeaveIt | None:
    """The one to spend the next move on, or None when none of them is worth it.

    None is a real answer and not a failure. Where acting changes nothing
    anywhere, the move is better spent on something this does not know about,
    and saying so is more use than picking the least pointless of them.
    """
    weighed = what_it_costs_to_leave_them(
        situations,
        her_acts=her_acts,
        step=step,
        how_good=how_good,
        their_acts=their_acts,
        their_step=their_step,
    )
    if not weighed or weighed[0].worth <= 0:
        return None
    return weighed[0]

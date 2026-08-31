"""What is precious, what is spendable, and doing as little as possible.

The same recorded checkers game. Asked afterwards what they were doing:

    knowing i dont have to risk my king (unless i see easy pieces to grab that
    dont put me in danger) so i could pretty much just stall out the board to
    a win

Three things are in that and none of them is about kings.

The first is that the pieces were not worth the same. Not because a crowned
piece is worth more in general — it is worth more HERE, because it is the one
thing keeping their own options from running out, and running out is how the
other side was going to lose. Which things are precious follows from what she
has decided to win with, and it follows mechanically: take a part away and ask
whether the thing she is holding survives without it. What does not survive
its loss is what she cannot afford to lose. Everything else is spendable, and
three pieces were spent.

The second is stalling. Once the structure is winning, progress is not the
goal and looking for it is a way to lose. The best act is the one that changes
least — which is not a move she has, because looking ahead is always looking
for something better. Doing as little as possible while remaining able to act
is a strategy, and against something that must move it is often the whole one.

The third is that stalling is not passivity. A capture that costs nothing is
still taken. The condition is not "gain nothing", it is "expose nothing", and
those come apart.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "WhatAnActRisks",
    "how_to_hold_what_is_already_won",
    "what_an_act_risks",
    "what_she_cannot_afford_to_lose",
]


@dataclass(frozen=True)
class WhatAnActRisks:
    """What doing this would cost her, and what it would take from them."""

    act: Any
    keeps_it: bool
    exposes: tuple[Any, ...]
    takes_from_them: int
    changes: int

    @property
    def safe(self) -> bool:
        return self.keeps_it and not self.exposes

    def describe(self) -> str:
        if not self.keeps_it:
            return "gives up the thing she is holding"
        if self.exposes:
            return f"puts {len(self.exposes)} she cannot spare in reach"
        if self.takes_from_them:
            return f"takes {self.takes_from_them} and risks nothing"
        return f"changes {self.changes} and risks nothing"


def what_she_cannot_afford_to_lose(
    state: Any,
    *,
    holding: Callable[[Any], bool],
    parts_of: Callable[[Any], Iterable[Any]],
    without: Callable[[Any, Any], Any],
    acts: Sequence[Any] | None = None,
    step: Callable[[Any, Any], Any | None] | None = None,
) -> tuple[Any, ...]:
    """The parts the held thing does not survive the loss of.

    Asked by taking each away and looking, which needs no table of what things
    are worth and gives a different answer in a different position — as it
    should, because a thing is not precious in itself. It is precious given
    what she is trying to do with it.

    When she is also handed a way of acting, anything she would have no move
    without is precious too, whatever the held thing says. Being able to act at
    all underlies every plan that could want anything.
    """
    precious: list[Any] = []
    for part in parts_of(state):
        short = without(state, part)
        if not holding(short):
            precious.append(part)
            continue
        if acts is not None and step is not None:
            if not any(step(short, act) is not None for act in acts):
                precious.append(part)
    return tuple(precious)


def what_an_act_risks(
    state: Any,
    act: Any,
    *,
    holding: Callable[[Any], bool],
    parts_of: Callable[[Any], Iterable[Any]],
    without: Callable[[Any, Any], Any],
    step: Callable[[Any, Any], Any | None],
    acts: Sequence[Any] | None = None,
    their_acts: Sequence[Any] = (),
    their_step: Callable[[Any, Any], Any | None] | None = None,
    theirs: Callable[[Any], Iterable[Any]] | None = None,
    named: Callable[[Any], Hashable] = repr,
) -> WhatAnActRisks | None:
    """What this act would cost, once the other side has had its turn.

    Exposure is not a guess. It is asking what they could do next and looking
    for a precious part that is gone in any of those places. A move that leaves
    something she cannot spare where they can take it is not risky in some
    degree, it is a move she will not make while she has another.
    """
    after = step(state, act)
    if after is None:
        return None
    kept = bool(holding(after))
    was = set(map(named, parts_of(state)))
    now = set(map(named, parts_of(after)))
    precious = what_she_cannot_afford_to_lose(
        after,
        holding=holding,
        parts_of=parts_of,
        without=without,
        acts=acts,
        step=step,
    )
    exposed: list[Any] = []
    if kept and precious and their_acts and their_step is not None:
        by_name = {named(one): one for one in precious}
        for their_act in their_acts:
            theirs_after = their_step(after, their_act)
            if theirs_after is None:
                continue
            left = set(map(named, parts_of(theirs_after)))
            for gone in by_name.keys() - left:
                if by_name[gone] not in exposed:
                    exposed.append(by_name[gone])
    took = 0
    if theirs is not None:
        took = max(0, len(list(theirs(state))) - len(list(theirs(after))))
    return WhatAnActRisks(
        act=act,
        keeps_it=kept,
        exposes=tuple(exposed),
        takes_from_them=took,
        changes=len(was ^ now),
    )


def how_to_hold_what_is_already_won(
    state: Any,
    *,
    acts: Sequence[Any],
    holding: Callable[[Any], bool],
    parts_of: Callable[[Any], Iterable[Any]],
    without: Callable[[Any, Any], Any],
    step: Callable[[Any, Any], Any | None],
    their_acts: Sequence[Any] = (),
    their_step: Callable[[Any, Any], Any | None] | None = None,
    theirs: Callable[[Any], Iterable[Any]] | None = None,
    named: Callable[[Any], Hashable] = repr,
) -> list[WhatAnActRisks]:
    """Her acts in the order somebody sitting on a win would take them.

    Keeping the held thing comes before everything. Then risking nothing she
    cannot spare. Then a free taking, because refusing one is superstition
    rather than caution. Then the smallest change, which is where stalling
    comes from: with nothing to gain and nothing at risk, the move that
    disturbs least is the move, and against something that has to move every
    turn that is enough to win with.
    """
    weighed: list[WhatAnActRisks] = []
    for act in acts:
        risk = what_an_act_risks(
            state,
            act,
            holding=holding,
            parts_of=parts_of,
            without=without,
            step=step,
            acts=acts,
            their_acts=their_acts,
            their_step=their_step,
            theirs=theirs,
            named=named,
        )
        if risk is not None:
            weighed.append(risk)
    weighed.sort(
        key=lambda one: (
            not one.keeps_it,
            len(one.exposes),
            -one.takes_from_them,
            one.changes,
        )
    )
    return weighed

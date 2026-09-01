"""Improving the machinery she invents with, without moving the ruler it is judged by.

Her synthesis tries every term the grammar admits, shortest first, with the
holes filled from two dozen words ordered by how much each already agrees with
what she saw happen. Both of those are mine. The count is a number I wrote, and
the ordering is a rule of thumb I wrote, and neither came from her.

That is the last authored thing in a chain built to have none, and it is the
one worth taking seriously, because it is the machinery she extends herself
with. A system that can improve its own way of inventing is doing something
different from a system that invents.

What can be learned safely is exactly one half of it. Two questions look alike
and are not:

    what to try first    a guess, and a wrong one costs time and nothing else
    what to keep         a judgement, and a wrong one is a language that got
                         worse while every number said it got better

So the search order is learned from her own history and the ruler is not
touched. She proposes differently and is judged the same. A prior that learned
to propose rubbish loses time and keeps nothing, which is the whole safety
argument and needs no supervision to hold.

The estimate is Laplace's: a word that appeared in w of n winning terms is
worth (w+1)/(n+2). No weight balances it against the agreement this problem
shows, because none is needed — one is evidence from history and the other
evidence from the case, and the product of the two is what they say together.

What the history is worth, measured rather than assumed: nothing yet. Trained
on two families and tested on four it had not seen, the search checked
5,886,047 fillings cold and 5,872,688 warm. Agreement is computed from the case
in front of her and it separates her words cleanly — on a language of
twenty-five, one word held the top score and the rest spread over nine values —
so there is no tie for history to break. With no history Laplace gives every
word the same figure and the order is exactly the agreement order, which is why
this is a generalisation of what was there rather than a change to it. It will
start to matter when agreement ties, and until it does the honest thing is to
say it is worth nothing rather than to let a design argument stand in for a
measurement.

The count is the half that pays. Two dozen words was a number I wrote, so the
first attempt replaced it with widening until every word had been tried — which
is not a bound at all, and on a grown language the search stopped finishing.
What bounds a search is not a count of words but the time there is: the caller
says how long she has, the widening runs while that lasts, and the best few go
first. Over three families with the same allowance, that is 7,611,922 fillings
checked against 4,216,583, and the same two families solved.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.state_ownership import state_root

__all__ = [
    "WhatWorkedBefore",
    "forget_what_worked",
    "recall",
    "what_is_remembered",
    "how_often_it_worked",
    "in_the_order_worth_trying",
    "remember_what_worked",
    "widening_word_lists",
]

logger = logging.getLogger("Aura.HowSheLearnsToLook")

def _kept_at() -> Path:
    return state_root() / "state" / "what_worked_when_she_invented.json"

_WHAT_WORKED: dict[str, int] = {}
_HOW_MANY_TIMES = [0]


@dataclass(frozen=True)
class WhatWorkedBefore:
    """How often a word turned up inside a term that survived its own gate."""

    name: str
    won: int
    of: int

    @property
    def rate(self) -> float:
        """Laplace's estimate, which needs no number chosen to smooth it."""
        return (self.won + 1) / (self.of + 2)

    def __str__(self) -> str:
        return f"{self.name!r}: in {self.won} of {self.of} winning terms"


def how_often_it_worked(name: str) -> WhatWorkedBefore:
    return WhatWorkedBefore(
        name=name, won=_WHAT_WORKED.get(name, 0), of=_HOW_MANY_TIMES[0]
    )


def remember_what_worked(names: Iterable[str]) -> None:
    """Record the words inside a term that earned its place.

    Only terms that passed the gate are recorded. A term that computed the
    right thing and was then refused for costing more than it bought is not
    evidence about what to try first — it is evidence about what to try first
    and then throw away.
    """
    _HOW_MANY_TIMES[0] += 1
    for name in names:
        _WHAT_WORKED[name] = _WHAT_WORKED.get(name, 0) + 1
    _keep()


def forget_what_worked() -> None:
    _WHAT_WORKED.clear()
    _HOW_MANY_TIMES[0] = 0


def in_the_order_worth_trying(
    every: Mapping[str, Any],
    agrees: Any,
    wanted: dict[int, tuple[int, ...]],
    *,
    shortest: Any,
) -> list[str]:
    """Her words, likeliest first.

    Two pieces of evidence about one question. How much a word already puts
    where it belongs is what this family says; how often it appeared in a term
    that survived is what every family before said. Their product is what they
    say together, and length breaks the ties.
    """
    most = sum(max(0, len(found)) for found in wanted.values())

    # The rule that combines the two is a TERM, not this function.
    #
    # The counts here were learned and the rule combining them was mine, and a
    # Python expression is the next authored level up — the same shape of gap
    # growing_at_any_level left one level down when it collapsed the API for
    # makers and kept a Python callable. It lives in
    # core/cognition/the_order_she_tries_them_in.py now, where it can be
    # replaced by the code that replaces a head.
    #
    # Laplace on this side too, so that a word agreeing with nothing is
    # unlikely rather than impossible. A bare fraction makes it exactly zero,
    # the product is then zero whatever the history says, and the words
    # history could most help with — the ones that appear inside an undo,
    # contributing no direct agreement — are the ones it could never move.
    from core.cognition.the_order_she_tries_them_in import how_it_scores

    def worth(name: str) -> tuple[int, int, str]:
        try:
            agreed = max(0, int(agrees(every[name], wanted)))
        except (ArithmeticError, TypeError, ValueError):
            agreed = 0
        before = how_often_it_worked(name)
        symbols = shortest(name)
        return (
            -how_it_scores(
                agreed=agreed,
                places=most,
                won=before.won,
                of=before.of,
                symbols=symbols,
            ),
            symbols,
            name,
        )

    return sorted(every, key=worth)


def widening_word_lists(
    names: Sequence[str], *, holes: int, within: float, started: float | None = None
) -> Iterable[list[str]]:
    """The best few, then more, then more, while there is time.

    An easy family is one whose answer is near the front, and paying a hard
    family's price to find it buys nothing. Each round doubles, so reaching any
    depth costs no more than twice reaching it directly.

    The rounds stop when the time runs out rather than when the words do. A
    search over every word of a grown language is quadratic in hundreds and
    does not finish, and stopping on a word count would only put a different
    number of mine back where the last one was.
    """
    began = time.monotonic() if started is None else started
    start = max(2, int(holes) + 1)
    seen = 0
    while seen < len(names):
        if time.monotonic() - began >= within:
            return
        seen = min(len(names), max(start, seen * 2))
        yield list(names[:seen])


def _keep() -> None:
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        written = json.dumps({"won": _WHAT_WORKED, "of": _HOW_MANY_TIMES[0]})
        destination = _kept_at()
        with local_internal_governed_scope(
            "how_she_learns_to_look.keep", domain="state_mutation"
        ):
            get_file_write_gateway().ensure_directory(
                destination.parent, source="how_she_learns_to_look"
            )
            get_file_write_gateway().write_text(
                destination, written, source="how_she_learns_to_look"
            )
    except (OSError, RuntimeError, TypeError, ValueError):
        logger.debug("could not keep what worked", exc_info=True)


def recall() -> int:
    """What she learned about her own searching, from before this process."""
    try:
        row = json.loads(_kept_at().read_text())
    except (OSError, ValueError):
        return 0
    won = row.get("won")
    if not isinstance(won, dict):
        return 0
    _WHAT_WORKED.clear()
    _WHAT_WORKED.update({str(k): int(v) for k, v in won.items()})
    _HOW_MANY_TIMES[0] = int(row.get("of", 0))
    return len(_WHAT_WORKED)


def what_is_remembered() -> tuple[WhatWorkedBefore, ...]:
    return tuple(
        sorted(
            (how_often_it_worked(name) for name in _WHAT_WORKED),
            key=lambda one: (-one.won, one.name),
        )
    )

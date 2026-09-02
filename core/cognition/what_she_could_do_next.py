"""The things she could do about herself, as entries rather than as line numbers.

`sequence_induction` used to widen its language down a fixed ladder: try a new
word, and if that returns nothing try an operation, and if that returns nothing
try a way of building, and so on for eight rungs. Every rung ran because the
one above it failed. Nothing in that is a decision, and the ladder itself is a
hand-written taxonomy of what development can be — the exact thing this
question is about.

Here the rungs are entries in a registry. Each says what it changes, what it
costs to try, and what it admits when it works, and those three are what a
value needs. The order they run in is a consequence of what they are worth
rather than of where they sit in a file, and
`where_a_split_disagrees_with_the_whole` is what makes that difference
checkable rather than a claim about style.

A new kind of developmental action needs no edit here. A developmental action
is a place a term can go plus a shape of term to look for, and both are values:
the destinations are the six things she can already revise, and a shape is a
term. `the_action_she_wrote` takes those two and gives back an action, so an
action she invents is admitted by the same call that admits one that was
written down.

What is deliberately not here
-----------------------------
A category of opportunity. There is no list of the kinds of thing that could be
wrong with her, because such a list is the taxonomy again with a different
name. What is wrong is read off the record as a number, and an action is worth
doing when the number says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "ADevelopmentalAction",
    "WHAT_SHE_COULD_DO",
    "WHERE_A_TERM_CAN_GO",
    "forget_the_action",
    "the_action_she_wrote",
    "the_actions_she_has",
    "what_she_could_do",
]

logger = logging.getLogger("Aura.WhatSheCouldDoNext")


@dataclass(frozen=True, slots=True)
class ADevelopmentalAction:
    """One thing she could do about herself."""

    #: What it is called. Also the key, so admitting the same name twice
    #: replaces rather than duplicates.
    name: str
    #: What it changes about her. One of WHERE_A_TERM_CAN_GO, and the reason
    #: that set is closed is that it is the set of things a term can be
    #: installed as — which is a fact about the floor rather than a policy.
    over: str
    #: What it admits when it works. The key the record's estimators group by,
    #: so what a change of this kind has saved before is what it is estimated
    #: to save now.
    kind: str
    #: Doing it. Given whatever the caller is holding, gives back a note about
    #: what changed, or nothing where it changed nothing.
    do_it: Callable[..., Any]
    #: What trying costs, in candidates walked. Measured where the action
    #: knows, estimated from its own past where it does not.
    price: int = 0
    #: The term, where she wrote one. Nothing for the ones that were written
    #: down, and that difference is reported rather than hidden.
    written: Any = None
    #: Where it came from, for the trace.
    hers: bool = False


#: The places a term can be installed. Not a taxonomy of development — a list
#: of the things in this codebase that hold a term and can be handed a
#: different one. Each has an installer, a lesion, and a persistence path, and
#: those three are what makes something a destination rather than a wish.
WHERE_A_TERM_CAN_GO: tuple[str, ...] = (
    "the words",
    "the ways of building words",
    "the ways of computing",
    "the shapes a rule can have",
    "the order she tries them in",
    "the proposer",
    "what a change is worth",
)


WHAT_SHE_COULD_DO: dict[str, ADevelopmentalAction] = {}


def what_she_could_do(
    name: str,
    *,
    over: str,
    kind: str,
    do_it: Callable[..., Any],
    price: int = 0,
    written: Any = None,
    hers: bool = False,
) -> ADevelopmentalAction:
    """Put an action in the registry. The one call, for hers and for ours."""
    if over not in WHERE_A_TERM_CAN_GO:
        raise ValueError(f"a term cannot go to {over!r}")
    made = ADevelopmentalAction(
        name=str(name),
        over=over,
        kind=str(kind),
        do_it=do_it,
        price=max(0, int(price)),
        written=written,
        hers=bool(hers),
    )
    WHAT_SHE_COULD_DO[made.name] = made
    return made


def the_actions_she_has() -> tuple[ADevelopmentalAction, ...]:
    """Everything she could do, in the order they were admitted."""
    return tuple(WHAT_SHE_COULD_DO.values())


def the_action_she_wrote(
    name: str,
    *,
    over: str,
    look_for: Any,
    kind: str = "",
) -> ADevelopmentalAction:
    """An action she invented: a shape of term, and a place to put it.

    `look_for` is a term taking the situation and giving back a candidate to
    install, so the action is a value all the way down. The installer is
    whichever one holds `over`, and there is no second mechanism for it.
    """
    from core.cognition.the_floor_she_stands_on import run

    put_it = _WHERE_IT_GOES.get(over)
    if put_it is None:
        raise ValueError(f"nothing installs at {over!r}")

    def do_it(situation: Any = None) -> Any:
        try:
            made = run(look_for, fuel=200_000)
            if hasattr(made, "body"):
                made = run(made.body, (situation, *made.env), fuel=200_000)
        except Exception as exc:  # noqa: BLE001 - a refusal changes nothing
            logger.info("%s gave nothing: %s", name, exc)
            return None
        return put_it(made)

    return what_she_could_do(
        name,
        over=over,
        kind=kind or f"a term for {over}",
        do_it=do_it,
        written=look_for,
        hers=True,
    )


def forget_the_action(name: str) -> ADevelopmentalAction | None:
    """Take one out. The lesion."""
    return WHAT_SHE_COULD_DO.pop(str(name), None)


def _install_a_head(term: Any) -> Any:
    from core.cognition.a_way_of_computing_she_wrote import as_a_head
    from core.cognition.one_algebra import the_head_she_wrote

    return the_head_she_wrote(
        f"a way of computing ({len(WHAT_SHE_COULD_DO)})", 3, as_a_head(term)
    )


def _install_an_order(term: Any) -> Any:
    from core.cognition.the_order_she_tries_them_in import the_order_she_wrote

    return the_order_she_wrote(term)


def _install_a_proposer(term: Any) -> Any:
    from core.cognition.the_proposer_she_can_replace import the_proposer_she_wrote

    return the_proposer_she_wrote(term)


def _install_a_worth(term: Any) -> Any:
    from core.cognition.what_it_is_worth_doing import the_worth_she_wrote

    return the_worth_she_wrote(term)


#: Which installer holds each destination. The reason `the_action_she_wrote`
#: needs no edit for a new action is that this table is about the substrate and
#: not about development.
_WHERE_IT_GOES: dict[str, Callable[[Any], Any]] = {
    "the ways of computing": _install_a_head,
    "the order she tries them in": _install_an_order,
    "the proposer": _install_a_proposer,
    "what a change is worth": _install_a_worth,
}

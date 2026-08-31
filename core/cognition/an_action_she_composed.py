"""Making new actions the same way she makes new words.

Everything built so far grows the language she DESCRIBES with. A7 is the same
question on the other side: can she arrive at an action nobody gave her, out of
the ones she has?

The shape carries over exactly, which is the point of having one algebra. A
word was a term over places in a thing of some size; an action is a term over
states of a world. Composition, repetition, branching and undoing mean the same
thing in both, and the heads say so:

    do        one of the actions she was given
    then      this, and then that
    until     this, over and over, while it still changes anything
    if it     this when the world says so, otherwise that
    instead   this, and that only if this changed nothing

Two of those are worth saying out loud.

``until`` takes no count. "Press left three times" needs a three from
somewhere, and there is nowhere honest to get one; "press left while it still
moves" reads its own stopping point off the world. The world supplies the
number, which is the rule everywhere else here.

``instead`` exists because an action that does nothing is the commonest thing
on a screen and the hardest to see. A player who presses left into a wall and
carries on as though the board moved is not playing. Making "and if that did
nothing, do this" a piece of the grammar means she can BUILD the recovery
rather than have it written for her.

What earns an action its place is what earns a word its place, and the first
attempt got the unit wrong. "One right, while it still moves" costs the same
number of key presses as pressing right eight times, so counted in acts it
saves nothing and was refused. What it saves is eight DECISIONS. With b actions
and a plan L long there are b**L plans to walk, and a composed action turns
eight of those choices into one.

So it is weighed in plans-she-would-otherwise-walk. That is the unit a word is
weighed in too, but the word's own function cannot be borrowed for it: that one
charges the new word against plans of the OLD length, which is right for a
language searched to a fixed depth and wrong here, because the whole effect of
a composed action is that the depth collapses. Borrowed anyway, it made "one
right while it still moves" cost 9,330 plans to save 510, and refused it.

The comparison is between the two searches themselves:

    without it   b actions, plans as long as the primitives need   N(b, L)
    with it      b+1 actions, plans as long as they now need       N(b+1, L')

and the difference is the answer. For a line ten places long that is 510
against 3.

Weighed one state at a time this still missed half of what an action is for,
and "press left, and if that changed nothing, press right" was refused: for
each single state some one key does the job in one act, so state by state it
saves nothing. What it does is cover EVERY state with one action where the
primitives need a different key each time — and a different key each time is a
decision she can only make by already knowing which. So the weighing asks of
the whole set at once: is there one plan of primitives that takes every state
to where it should go? Where there is not, the composed action reaches
something no arrangement of primitives reaches, which is the whole search.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Doing",
    "World",
    "an_action_she_composed",
    "every_doing",
    "read_back",
    "what_it_does",
    "written_down",
]

logger = logging.getLogger("Aura.AnActionSheComposed")


@dataclass(frozen=True)
class World:
    """What she can do and what she can tell, in whatever world this is.

    Nothing here knows about boards, screens or files. A world is what answers
    "if I do this from here, what then" and "what does it say about here".
    """

    #: The actions she was given, by name.
    can_do: dict[str, Callable[[Any], Any]]
    #: What she can tell about a state, by name. Each answers yes or no.
    can_tell: dict[str, Callable[[Any], bool]]

    def act(self, state: Any, named: str) -> Any:
        return self.can_do[named](state)

    def tells(self, state: Any, named: str) -> bool:
        return bool(self.can_tell[named](state))


@dataclass(frozen=True)
class Doing:
    """One action, in the same shape a word is in."""

    head: str
    parts: tuple["Doing", ...] = ()
    value: Any = None

    @property
    def name(self) -> str:
        if self.head == "do":
            return str(self.value)
        if self.head == "then":
            return f"{self.parts[0].name}, then {self.parts[1].name}"
        if self.head == "until":
            return f"{self.parts[0].name} while it still changes anything"
        if self.head == "if it":
            return (
                f"if {self.value} then {self.parts[0].name} "
                f"else {self.parts[1].name}"
            )
        if self.head == "instead":
            return (
                f"{self.parts[0].name}, and if that changed nothing, "
                f"{self.parts[1].name}"
            )
        return self.head

    def how_long(self) -> int:
        return 1 + sum(part.how_long() for part in self.parts)


#: How many times a repetition may go round before she calls it a loop.
#:
#: Not a budget on how long she may try. It is the point past which "while it
#: still changes anything" has stopped being a description of a world that
#: settles: a state that keeps changing forever cannot be reached by waiting,
#: and treating it as though it could is how a plan hangs instead of failing.
_A_LOOP_RATHER_THAN_A_SETTLING = 64


def what_it_does(doing: Doing, state: Any, world: World, depth: int = 0) -> Any:
    """Where this action leaves the world, from here."""
    if depth > 32:
        raise ValueError("an action that will not settle")
    head = doing.head
    if head == "do":
        return world.act(state, str(doing.value))
    if head == "then":
        return what_it_does(
            doing.parts[1],
            what_it_does(doing.parts[0], state, world, depth + 1),
            world,
            depth + 1,
        )
    if head == "until":
        here = state
        for _turn in range(_A_LOOP_RATHER_THAN_A_SETTLING):
            went = what_it_does(doing.parts[0], here, world, depth + 1)
            if went == here:
                return here
            here = went
        raise ValueError("it never stopped changing")
    if head == "if it":
        which = 0 if world.tells(state, str(doing.value)) else 1
        return what_it_does(doing.parts[which], state, world, depth + 1)
    if head == "instead":
        went = what_it_does(doing.parts[0], state, world, depth + 1)
        if went != state:
            return went
        return what_it_does(doing.parts[1], state, world, depth + 1)
    raise ValueError(f"nothing in the grammar called {head!r}")


def how_many_acts(doing: Doing, state: Any, world: World, depth: int = 0) -> int:
    """Primitive acts this costs from here. The unit everything is weighed in."""
    if depth > 32:
        return 0
    head = doing.head
    if head == "do":
        return 1
    if head == "then":
        first = doing.parts[0]
        return how_many_acts(first, state, world, depth + 1) + how_many_acts(
            doing.parts[1],
            what_it_does(first, state, world, depth + 1),
            world,
            depth + 1,
        )
    if head == "until":
        here, spent = state, 0
        for _turn in range(_A_LOOP_RATHER_THAN_A_SETTLING):
            went = what_it_does(doing.parts[0], here, world, depth + 1)
            spent += how_many_acts(doing.parts[0], here, world, depth + 1)
            if went == here:
                return spent
            here = went
        return spent
    if head == "if it":
        which = 0 if world.tells(state, str(doing.value)) else 1
        return how_many_acts(doing.parts[which], state, world, depth + 1)
    if head == "instead":
        first = doing.parts[0]
        spent = how_many_acts(first, state, world, depth + 1)
        if what_it_does(first, state, world, depth + 1) != state:
            return spent
        return spent + how_many_acts(doing.parts[1], state, world, depth + 1)
    return 0


def every_doing(world: World, deepest: int = 2) -> Iterator[Doing]:
    """Every action the grammar admits, shortest first.

    Shortest first is not a preference. It is what makes the first thing found
    the cheapest thing that works, so nothing has to be compared afterwards.
    """
    given = [Doing(head="do", value=name) for name in sorted(world.can_do)]
    yield from given
    standing = list(given)
    for _round in range(max(0, int(deepest))):
        made: list[Doing] = []
        for one in standing:
            made.append(Doing(head="until", parts=(one,)))
            for other in standing:
                made.append(Doing(head="then", parts=(one, other)))
                if one != other:
                    made.append(Doing(head="instead", parts=(one, other)))
                    for telling in sorted(world.can_tell):
                        made.append(
                            Doing(head="if it", parts=(one, other), value=telling)
                        )
        fresh = [one for one in made if one not in standing]
        yield from fresh
        standing.extend(fresh)


@dataclass(frozen=True)
class WhatItWasWorth:
    """What a composed action buys, in plans, on states it was not built from."""

    doing: Doing
    #: Plans the primitives alone would have to walk to reach what it reaches.
    removes: int = 0
    #: Plans she walks with it in hand, one action richer and far shorter.
    adds: int = 0
    #: Decisions it stands in for, added up over the states it was weighed on.
    decisions_saved: int = 0
    tried: int = 0

    @property
    def keep_it(self) -> bool:
        return self.removes > self.adds

    def describes(self) -> str:
        verdict = "earns its place" if self.keep_it else "costs more than it buys"
        return (
            f"{verdict}: stands in for {self.decisions_saved} decision(s) over "
            f"{self.tried} state(s) it was not built from — {self.removes:,} "
            f"plan(s) to walk without it against {self.adds:,} with it"
        )


def an_action_she_composed(
    world: World,
    shown: Sequence[tuple[Any, Any]],
    *,
    held_out: Sequence[tuple[Any, Any]] = (),
    deepest: int = 2,
) -> tuple[Doing, WhatItWasWorth] | None:
    """An action that takes each of these states to the one beside it.

    ``shown`` is what she watched happen and wants to be able to do. Nothing is
    consulted but the grammar: what she can arrive at is not a list of useful
    macros but anything the algebra can say.
    """
    if not shown:
        return None
    for doing in every_doing(world, deepest=deepest):
        if doing.head == "do":
            # A primitive is not a new action. It is already hers.
            continue
        if not _takes_each_there(doing, shown, world):
            continue
        worth = _what_it_was_worth(doing, world, held_out or shown)
        if not worth.keep_it:
            logger.info("not keeping %r — %s", doing.name, worth.describes())
            continue
        logger.info("she composed an action: %s", doing.name)
        return doing, worth
    return None


def _takes_each_there(doing: Doing, shown: Sequence[tuple[Any, Any]], world: World) -> bool:
    for state, wanted in shown:
        try:
            if what_it_does(doing, state, world) != wanted:
                return False
        except (ArithmeticError, KeyError, RecursionError, TypeError, ValueError):
            return False
    return True


def _what_it_was_worth(
    doing: Doing, world: World, held_out: Sequence[tuple[Any, Any]]
) -> WhatItWasWorth:
    """Weighed on states it was not built from, in plans she does not walk.

    One composed action is one decision. Whatever the primitives would have
    needed to get to the same place is that many decisions, and the difference
    is what it stands in for. A state the primitives cannot reach at all inside
    that many acts is the whole search, which is why reaching and shortening
    are the same measurement here rather than two.
    """
    from core.cognition.what_it_costs_to_say import how_many_expressions

    fits: list[tuple[Any, Any, int]] = []
    for state, wanted in held_out:
        try:
            if what_it_does(doing, state, world) != wanted:
                continue
            fits.append((state, wanted, how_many_acts(doing, state, world)))
        except (ArithmeticError, KeyError, RecursionError, TypeError, ValueError):
            continue
    if not fits:
        return WhatItWasWorth(doing=doing, removes=0, adds=1, tried=len(held_out))

    actions = max(1, len(world.can_do))
    deepest = max(acts for _s, _w, acts in fits)

    # One plan of primitives that works for all of them, or none.
    uniform = _one_plan_for_all(
        world, [(state, wanted) for state, wanted, _a in fits], within=max(1, deepest)
    )
    if uniform is None:
        # Nothing the primitives can be arranged into covers this set, so what
        # it stands in for is every plan she would walk looking for one.
        saved = sum(max(0, acts - 1) for _s, _w, acts in fits) or len(fits)
        return WhatItWasWorth(
            doing=doing,
            removes=how_many_expressions(actions, deepest),
            adds=how_many_expressions(actions + 1, 1),
            decisions_saved=saved,
            tried=len(held_out),
        )

    # A single plan of primitives does cover it, so the only thing left to buy
    # is length: hers against theirs.
    saved = max(0, uniform - 1)
    return WhatItWasWorth(
        doing=doing,
        removes=how_many_expressions(actions, uniform),
        adds=how_many_expressions(actions + 1, 1),
        decisions_saved=saved,
        tried=len(held_out),
    )


def _one_plan_for_all(
    world: World, pairs: Sequence[tuple[Any, Any]], *, within: int
) -> int | None:
    """Length of the shortest primitive plan that works for EVERY pair.

    The same keys in the same order from each state. Where none exists, no
    arrangement of what she was given does uniformly what the composed action
    does, however well each state does on its own.
    """
    if not pairs:
        return None
    names = sorted(world.can_do)
    plans: list[tuple[str, ...]] = [()]
    for length in range(1, max(1, int(within)) + 1):
        plans = [(*plan, name) for plan in plans for name in names]
        for plan in plans:
            if all(_runs_to(world, state, plan) == wanted for state, wanted in pairs):
                return length
    return None


def _runs_to(world: World, state: Any, plan: Sequence[str]) -> Any:
    here = state
    for name in plan:
        try:
            here = world.act(here, name)
        except (ArithmeticError, KeyError, TypeError, ValueError):
            return None
    return here


def _shortest_way_there(
    world: World, state: Any, wanted: Any, *, within: int
) -> int | None:
    """Acts the primitives alone need, or None if they cannot inside that many."""
    seen = {_key(state)}
    edge = [state]
    for step in range(1, max(1, int(within)) + 1):
        nxt = []
        for here in edge:
            for named in sorted(world.can_do):
                try:
                    went = world.act(here, named)
                except (ArithmeticError, KeyError, TypeError, ValueError):
                    continue
                if went == wanted:
                    return step
                mark = _key(went)
                if mark not in seen:
                    seen.add(mark)
                    nxt.append(went)
        edge = nxt
        if not edge:
            return None
    return None


def _key(state: Any) -> Any:
    try:
        hash(state)
    except TypeError:
        return repr(state)
    return state


def written_down(doing: Doing) -> dict[str, Any]:
    """The action as data, so one she composed survives the process."""
    return {
        "head": doing.head,
        "value": doing.value,
        "parts": [written_down(one) for one in doing.parts],
    }


def read_back(row: Any) -> Doing | None:
    """An action she composed, back from what was written down."""
    if not isinstance(row, dict) or not isinstance(row.get("head"), str):
        return None
    parts = []
    for one in row.get("parts") or ():
        back = read_back(one)
        if back is None:
            return None
        parts.append(back)
    return Doing(head=row["head"], parts=tuple(parts), value=row.get("value"))

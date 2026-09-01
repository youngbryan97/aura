"""What kind of problem this is, and what kind of process suits it.

Reasoning through every low-level step is not what makes a mind general. It is
often the opposite: a general mind recognises the shape of what it is facing
and recruits or builds something specialised for that shape. Asked for the
shortest route through a hundred thousand cities, contemplating routes one by
one in language would be the stupid reading of generality; naming it a graph
problem and reaching for a graph algorithm is the intelligent one.

Nothing about that makes the mind narrower. The recognition is the general
part. What it recruits is allowed to be as specialised as the problem is.

This is the recognition step, and only that. It reads what she has already
established about a world — how many things she can do in it, whether she has
worked out what her acts do, whether anything changes without her, whether
what she wants can be counted — and names the shape those facts make. It
names the process that suits the shape, and says plainly whether she has one.

Nothing here knows about any particular world. The facts it reads are ones any
interactive thing produces, and the shapes it names are shapes, not subjects:
a small discrete world with a handful of acts and something arriving after
each of them is that shape whether it is a game, a queue, a market or a
factory floor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["Shape", "WhatSuitsIt", "recognise", "SMALL_ENOUGH_TO_SEARCH"]

logger = logging.getLogger("Aura.WhatKindOfProblem")

#: How many things she can do before trying all of them stops being cheap.
#: Above this the branching makes looking ahead a different proposition and
#: the shape is named differently.
FEW_ENOUGH_ACTS = 6

#: How many places a laid-out thing can have before searching over it is no
#: longer arithmetic. Chosen from what looking ahead actually costs rather
#: than from the size of anything in particular.
SMALL_ENOUGH_TO_SEARCH = 64


@dataclass(frozen=True)
class Shape:
    """What is true of this problem, in terms that are not about its subject."""

    acts: int = 0
    #: Whether she has worked out what her own acts do.
    transition_known: bool = False
    #: Whether the world adds things of its own between her acts.
    world_moves_too: bool = False
    #: Whether the state is small and countable rather than open-ended.
    discrete: bool = False
    #: Whether what she wants can be measured on a state.
    countable_goal: bool = False
    #: How big the thing is, across and down. Two worlds of the same size that
    #: take the same acts move the same way whoever drew them.
    across: int = 0
    down: int = 0

    def of_this_kind(self) -> str:
        """A name for the KIND of world this is, rather than for this one.

        What she learns is otherwise filed under the thing she learned it in —
        an application and an address — so a second world that moves in
        exactly the same way starts as ignorant as the first did, and the
        fortieth is no better off than the second. Two worlds are of a kind
        when they are the same size, take the same number of acts, and are
        countable in the same ways; and worlds of a kind move alike, which is
        the only claim this makes.

        Deliberately not part of it: what she has worked out about the
        transition, since that is the thing being carried and keying on it
        would mean only worlds she has already solved ever match.
        """
        if not self.acts or not (self.across and self.down):
            return ""
        size = "small" if self.discrete else "open"
        counted = "counted" if self.countable_goal else "uncounted"
        return f"a {self.across}x{self.down} {size} world, {self.acts} acts, {counted}"

    def named(self) -> str:
        """The shape, said in a line."""
        if not self.acts:
            return "nothing to do here"
        if not self.transition_known:
            return "an unmodelled world: act and look, because what her acts do is not worked out"
        stochastic = "stochastic" if self.world_moves_too else "deterministic"
        size = "small discrete" if self.discrete else "open"
        counted = "a countable objective" if self.countable_goal else "no measurable objective"
        return f"a {size} world, {self.acts} act(s), {stochastic} transition, {counted}"


@dataclass(frozen=True)
class WhatSuitsIt:
    """The kind of process this shape calls for, and whether she has one."""

    shape: Shape
    process: str
    have_it: bool
    because: str

    def says(self) -> str:
        held = "which she has" if self.have_it else "which she does not have"
        return f"{self.shape.named()} — that wants {self.process}, {held}. {self.because}"


def recognise(
    *,
    acts: Sequence[str] = (),
    knows_how_it_moves: Any = None,
    state: Any = None,
    toward: str = "",
) -> WhatSuitsIt:
    """Name the shape of this problem from what she has established about it.

    Every argument is something she worked out rather than something declared:
    the acts that do anything here, the model of what they do, the reading in
    front of her, and the goal in the words she was given.
    """
    shape = Shape(
        acts=len(list(acts)),
        transition_known=_has_a_transition(knows_how_it_moves),
        world_moves_too=_world_adds_things(knows_how_it_moves),
        discrete=_is_small_and_countable(state),
        countable_goal=_is_countable(toward),
        across=int(getattr(state, "columns", 0) or 0),
        down=int(getattr(state, "rows", 0) or 0),
    )
    suits = _what_suits(shape)
    logger.info("this is %s", suits.says())
    return suits


def _what_suits(shape: Shape) -> WhatSuitsIt:
    """Which kind of process a shape calls for."""
    if not shape.acts:
        return WhatSuitsIt(
            shape,
            "nothing",
            True,
            "There is nothing to choose between.",
        )
    if not shape.transition_known:
        return WhatSuitsIt(
            shape,
            "acting and looking",
            True,
            "Until what her acts do is worked out, the only way to find out is to do one.",
        )
    if shape.discrete and shape.acts <= FEW_ENOUGH_ACTS and shape.countable_goal:
        return WhatSuitsIt(
            shape,
            "looking ahead over the transition she worked out",
            True,
            (
                "Few enough acts and a small enough world that trying each of them in her "
                "head is arithmetic, and something to prefer one result over another by."
            ),
        )
    if shape.discrete and shape.acts <= FEW_ENOUGH_ACTS:
        return WhatSuitsIt(
            shape,
            "looking ahead, once there is something to prefer a result by",
            False,
            "She can try a move without making it and cannot yet say which result is better.",
        )
    if shape.acts > FEW_ENOUGH_ACTS:
        return WhatSuitsIt(
            shape,
            "a specialist built for this many acts, rather than trying all of them",
            False,
            (
                "Trying every act at every level stops being arithmetic here, and "
                "something that prunes is a different process from the one she has."
            ),
        )
    return WhatSuitsIt(
        shape,
        "acting and looking",
        True,
        "Nothing about this shape says a cheaper process would do better.",
    )


# ── reading what she established ─────────────────────────────────────────


def _has_a_transition(knows: Any) -> bool:
    rule = getattr(knows, "rule", None)
    return bool(callable(rule) and rule() is not None)


def _world_adds_things(knows: Any) -> bool:
    """Whether anything has ever arrived that her own acts did not put there.

    She already tolerates this when scoring a rule — a dealt tile is not a
    rule's mistake — so the fact is there to be read rather than assumed.
    """
    asked = getattr(knows, "world_adds_things", None)
    if callable(asked):
        return bool(asked())
    return bool(getattr(knows, "counters", None))


def _is_small_and_countable(state: Any) -> bool:
    places = getattr(state, "places", None)
    if not callable(places):
        return False
    room = int(places() or 0)
    return 0 < room <= SMALL_ENOUGH_TO_SEARCH


def _is_countable(toward: str) -> bool:
    try:
        from core.agency.how_good_is_this import worth_comparing

        return bool(worth_comparing(str(toward or ""), ""))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False

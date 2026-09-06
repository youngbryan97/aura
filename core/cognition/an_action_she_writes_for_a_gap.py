"""When nothing she has works on a family, she writes something that might.

This is the arrow that makes the developmental loop close over itself. Every
other action changes what she knows or how she computes; this one changes
what she can DO about either, by putting a new developmental action into the
set the next search reads.

The loop, and where each part already lived:

    a family beats everything she has     what_she_notices_about_herself
    write an operator for it              HERE
    install it where operators go         what_she_could_do_next
    try it, and undo it unless it pays    what_she_can_take_back
    promote it, or put it back exactly    how_a_change_is_promoted
    it is in the next search's choices    what_she_could_do_next

The term comes from enumeration over the floor, which is the honest
description and not a shortcoming. Nothing here decides that a term is good.
What decides is the same held-out test every other change faces: the new
action is written, and it earns its place only when taking it improves
families it was not chosen for. Enumerating and letting the measurement
refuse is the arrangement that cannot talk itself into a change.

What this deliberately does not do is choose the destination cleverly. It
takes the places she has no action for on this family, in the order the
substrate lists them, because a heuristic for where to look would be a
number nobody measured sitting in the middle of a loop whose whole point is
that measurement decides.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["a_gap_she_could_fill", "offer_writing_an_action_for_a_gap"]

#: How many enumerated terms are looked at before giving up on a gap. Small,
#: because writing an action is cheap and taking one is not: the search that
#: matters is the ranking that decides whether to take it, and spending here
#: buys nothing the ranking will not spend better.
_HOW_MANY_TERMS = 200

#: The shortest term worth installing. One symbol is a variable or a constant,
#: and an action built on either does the same thing everywhere.
_SHORTEST_WORTH_WRITING = 2


def a_gap_she_could_fill() -> tuple[str, str] | None:
    """The family that beat everything, and a place she has no action for.

    None where there is no gap, or where every destination already has an
    action written for this family — the second is not failure, it is the
    loop having nothing left to add before the ranking has judged what it
    already added.
    """
    from core.cognition.what_she_could_do_next import (
        WHERE_A_TERM_CAN_GO,
        the_actions_she_has,
    )
    from core.cognition.what_she_notices_about_herself import nothing_she_has

    gaps = sorted(nothing_she_has(), key=lambda one: -one.strength)
    if not gaps:
        return None
    family = gaps[0].about
    already = {
        one.over for one in the_actions_she_has() if family in one.name
    }
    for where in WHERE_A_TERM_CAN_GO:
        if where not in already:
            return family, where
    return None


def offer_writing_an_action_for_a_gap() -> None:
    """Put it in the registry. The action that enlarges the set of actions."""
    from core.cognition.what_she_could_do_next import (
        WHAT_SHE_COULD_DO,
        what_she_could_do,
    )

    def write_one(situation: Any = None) -> str | None:
        from core.cognition.the_floor_she_stands_on import (
            QUOTE,
            every_code,
            how_long,
        )
        from core.cognition.what_she_already_knows_how_to_say import (
            what_she_already_knows_how_to_say,
        )
        from core.cognition.what_she_could_do_next import the_action_she_wrote

        found = a_gap_she_could_fill()
        if found is None:
            return None
        family, where = found

        for at, body in enumerate(
            # Her own terms as leaves, for the reason `every_code` documents:
            # a library moves the horizon and a bigger budget does not.
            every_code(
                deepest=3,
                variables=1,
                constants=(0, 1, 2),
                also=what_she_already_knows_how_to_say(),
            )
        ):
            if at >= _HOW_MANY_TERMS:
                break
            if how_long(body) < _SHORTEST_WORTH_WRITING:
                continue
            name = f"one she wrote for {family} at {where} ({at})"
            if name in WHAT_SHE_COULD_DO:
                continue
            try:
                made = the_action_she_wrote(
                    name, over=where, look_for=QUOTE(body), kind="one she wrote"
                )
            except (ValueError, TypeError) as exc:
                logger.info("she could not write an action at %s: %s", where, exc)
                return None
            logger.info(
                "she wrote %s, because %s beat everything she had", made.name, family
            )
            # The sentence, not the action. What she did was enlarge what she
            # could do; whether the new action is any good is the ranking's
            # question and is answered the next time it is taken.
            return f"wrote an action for {family} at {where}"
        return None

    if "write an action for what nothing she has can do" not in WHAT_SHE_COULD_DO:
        what_she_could_do(
            "write an action for what nothing she has can do",
            over="the words",
            kind="an action she wrote",
            do_it=write_one,
            needs_a_case=False,
        )

"""The loop that widens her language, and the link it is starving at.

Six steps, each of which works, and the whole of which has never turned over:

1. the operator search enumerates terms over the floor;
2. one that pays becomes a term she has invented;
3. an invented term makes a part of her carry one;
4. two parts carrying terms can share something;
5. naming what they share writes a head;
6. a head is a leaf the next search stands on, which moves the horizon.

Step six is the only thing that widens a shortest-first search — depth three
over the bare floor is 380 terms and the enumeration cap is four thousand, so
the budget has never been the constraint. And the live record holds one induced
kind and no heads at all, so the horizon has never moved once.

The useful question is not "is the chain broken" — it is whole, and tracing it
is how that was settled. Running the head-writing action against the live
record shows every step firing: three pairs of her parts share something, the
action writes an 87-symbol head, the generic gate measures it on the held-out
families, it pays on none of them, and the change is put back exactly.

So nothing is starved of candidates. What is thin is the evidence. The gate
judges on families she has met that carry their cases, and the live record
holds 512 episodes across 76 families of which **three** carry cases. Three
binary observations cannot lift a posterior above the threshold the gate asks
for, so a change that genuinely helps and a change that does nothing produce
the same verdict, and the honest one is to refuse.

That is the binding constraint, and it is upstream of everything the review
asked about: the same three families are what the developmental evidence gate
weighs, so a thin probe makes every kept change read as unmeasured too.

This counts each step and the probe's width so the binding one is named rather
than guessed. Nothing here fixes it: a diagnostic that also intervened would be
measuring a world it had changed.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.WhereTheGrowingIsStarved")

__all__ = ["THE_STEPS", "where_it_is_starved", "how_the_growing_stands"]

#: The six, in the order they feed each other.
THE_STEPS: tuple[str, ...] = (
    "terms the search can reach",
    "terms she has invented",
    "parts of her carrying a term",
    "pairs that share something",
    "heads she has written",
    "leaves the next search stands on",
)

#: How many held-out families a change is judged on before the gate can say
#: yes. Below this the posterior cannot clear the threshold whatever the change
#: does, so the gate refuses everything and the refusal means nothing.
ENOUGH_TO_JUDGE_ON = 8


def _terms_the_search_can_reach() -> int:
    try:
        from core.cognition.how_far_the_search_reaches import how_many_up_to
        from core.cognition.what_she_already_knows_how_to_say import (
            what_she_already_knows_how_to_say,
        )

        return how_many_up_to(3, leaves=5 + len(what_she_already_knows_how_to_say()))
    except (ImportError, RuntimeError, TypeError, ValueError):
        return 0


def _parts_carrying_a_term() -> tuple[int, int]:
    try:
        from core.cognition.what_she_is_made_of import what_she_is_made_of

        parts = list(what_she_is_made_of())
    except (ImportError, RuntimeError, TypeError, ValueError):
        return 0, 0
    return len(parts), sum(1 for one in parts if getattr(one, "term", None) is not None)


def _pairs_that_share_something() -> int:
    """How many pairs of her parts have a term in common.

    Counted rather than short-circuited: `what_two_parts_share` returns the
    first pair it finds, which answers "is there one" and not "how close is
    this to being possible at all".
    """
    try:
        from core.cognition.what_she_is_made_of import (
            the_most_they_have_in_common,
            what_she_is_made_of,
        )

        terms = [one for one in what_she_is_made_of() if one.term is not None]
    except (ImportError, RuntimeError, TypeError, ValueError):
        return 0
    found = 0
    for at, first in enumerate(terms):
        for second in terms[at + 1 :]:
            try:
                if the_most_they_have_in_common(first.term, second.term) is not None:
                    found += 1
            except Exception:  # noqa: BLE001 — a pair that raises is not a pair
                continue
    return found


def _families_to_judge_on() -> int:
    """Held-out families carrying their cases. What the gate weighs."""
    try:
        from core.cognition.what_she_does_about_herself import (
            _probe,  # noqa: PLC2701
        )

        return len(_probe())
    except (ImportError, RuntimeError, TypeError, ValueError):
        return 0


def _counts() -> dict[str, int]:
    parts, carrying = _parts_carrying_a_term()
    try:
        from core.cognition.one_algebra import DERIVED_HEADS

        heads = len(DERIVED_HEADS)
    except (ImportError, AttributeError):
        heads = 0
    try:
        from core.cognition.what_she_already_knows_how_to_say import (
            what_she_already_knows_how_to_say,
        )

        leaves = len(what_she_already_knows_how_to_say())
    except (ImportError, RuntimeError, TypeError, ValueError):
        leaves = 0
    return {
        "terms the search can reach": _terms_the_search_can_reach(),
        "terms she has invented": carrying,
        "parts of her carrying a term": carrying,
        "pairs that share something": _pairs_that_share_something(),
        "heads she has written": heads,
        "leaves the next search stands on": leaves,
        "parts of her": parts,
        "families to judge on": _families_to_judge_on(),
    }


def where_it_is_starved() -> str:
    """The first step with nothing coming into it, or empty when it turns."""
    counted = _counts()
    for at, step in enumerate(THE_STEPS):
        if counted.get(step, 0) > 0:
            continue
        # A step is starved by what feeds it, not by itself. Step one has
        # nothing upstream, so an empty step one is the floor being empty,
        # which cannot happen while there is a floor.
        return step if at == 0 else THE_STEPS[at]
    return ""


def how_the_growing_stands() -> dict[str, Any]:
    """For the health report: the six counts and the link that is starving.

    Read from the live registries, so a process that has not recalled what she
    worked out reports zeros — which is true of that process and is why the
    conductor recalls before anything runs.
    """
    counted = _counts()
    starved = where_it_is_starved()
    judging_on = counted.get("families to judge on", 0)
    return {
        "schema": "aura.language_growth.v1",
        "steps": list(THE_STEPS),
        "counts": counted,
        "starved_at": starved,
        "turns_over": not starved,
        # The constraint that is upstream of the rest. A gate with too few
        # held-out families refuses a change that helps and a change that does
        # nothing alike, so its refusals carry no information — and the same
        # families are what the developmental evidence gate weighs.
        "families_to_judge_on": judging_on,
        "enough_to_judge_on": ENOUGH_TO_JUDGE_ON,
        "the_gate_can_say_yes": judging_on >= ENOUGH_TO_JUDGE_ON,
        "why_it_matters": (
            "a head is the only thing that widens a shortest-first search: "
            "depth three over the bare floor is 380 terms and the enumeration "
            "cap is four thousand, so the budget has never been the constraint"
        ),
    }

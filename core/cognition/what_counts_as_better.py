"""The thing a search is judged by, as a term rather than as a function.

`the_order_she_tries_them_in` made the rule deciding what to try first an
object. `the_proposer_she_can_replace` made the thing offering candidates an
object. `what_it_is_worth_doing` made the rule deciding what to change an
object. What stayed authored, through all of that, is the rule deciding whether
a change was any good.

That is the last one that matters, and it matters for a specific reason. A
search for a better order is a search against an objective, and if the
objective is a Python function then the space of orders she can look for is
whatever that function can see. Mean rank of the winner is a good objective and
it is not the only one: a rule that puts the winner second every time beats one
that puts it first half the time and last half the time on the mean, and which
of those is actually better depends on what happens next.

So the objective is a term over three numbers — where the winner sat, how many
there were, and how long the term is — and it gives a score where lower is
better. The default computes the rank, which is what was computed before, and a
test holds the two identical. Nothing has improved by making it a term. What
has changed is that an experiment she designs is now the same kind of thing as
a word she invents, which is what would have to be true before she could design
one.

What this is not
----------------
It is not the gate. What is KEPT is still decided outside the space, for the
reason `a_gate_inside_the_space_cannot_hold` executes: a rule that can rewrite
what judges it passes by changing what passing means. This decides what a
SEARCH aims at, which is a different thing — aiming at the wrong quantity
wastes a search, and rewriting the judge invalidates every result.
"""

from __future__ import annotations

import logging
from typing import Any

from core.cognition.the_floor_she_stands_on import (
    PLUS,
    TIMES,
    Code,
    L,
    N,
    V,
    build,
    how_long,
    read_back,
    run,
    written_down,
)

__all__ = [
    "THE_OBJECTIVE",
    "WHAT_THE_OBJECTIVE_IS_GIVEN",
    "forget_the_objective",
    "how_bad_that_is",
    "the_objective_read_back",
    "the_objective_she_uses",
    "the_objective_she_wrote",
    "written_objective",
]

logger = logging.getLogger("Aura.WhatCountsAsBetter")

#: What the objective is handed, outermost binder first.
WHAT_THE_OBJECTIVE_IS_GIVEN: tuple[str, ...] = (
    "where the winner sat",
    "how many there were",
    "how long the term is",
)

#: Where the winner sat, and nothing else. What the Python computed, written
#: where she can reach it. The other two are bound and unused on purpose: an
#: objective that cannot see how many there were cannot express "second of
#: three is worse than second of thirty", and one that cannot see the length
#: cannot prefer a short rule that ties.
THE_OBJECTIVE: Code = build(
    L(
        "sat",
        L(
            "of",
            L(
                "symbols",
                PLUS(V("sat"), TIMES(V("of"), N(0))),
            ),
        ),
    )
)

_IN_USE: list[Code] = [THE_OBJECTIVE]

#: What one judgement may spend. Judging happens once per occasion per
#: candidate rule, so it is small; a rule that cannot answer inside it judges
#: nothing, which sends the candidate to the back rather than stopping the
#: search.
_A_JUDGEMENT_MAY_SPEND = 20_000


def the_objective_she_uses() -> Code:
    return _IN_USE[0]


def the_objective_she_wrote(term: Code) -> Code:
    """Aim a search at something else. Same call shape as installing a head."""
    _IN_USE[0] = term
    logger.info("she is judging searches differently: %d symbols", how_long(term))
    return term


def forget_the_objective() -> Code:
    """Back to where the winner sat. The lesion."""
    _IN_USE[0] = THE_OBJECTIVE
    return THE_OBJECTIVE


def how_bad_that_is(*, sat: int, of: int, symbols: int) -> float:
    """Judge one occasion under the objective in force. Lower is better.

    A rule that raises or runs out of fuel judges the occasion as badly as it
    can be judged, which sends whatever produced it to the back rather than
    letting a broken objective look like a good result.
    """
    work: Any = _IN_USE[0]
    given = (int(sat), int(of), int(symbols))
    try:
        made = run(work, fuel=_A_JUDGEMENT_MAY_SPEND)
        for one in given:
            if not hasattr(made, "body"):
                return float(of)
            made = run(made.body, (one, *made.env), fuel=_A_JUDGEMENT_MAY_SPEND)
        return float(made)
    except Exception:  # noqa: BLE001 - a refusal is the worst score there is
        return float(of)


def written_objective() -> dict[str, Any]:
    return written_down(_IN_USE[0])


def the_objective_read_back(row: Any) -> Code | None:
    return read_back(row)

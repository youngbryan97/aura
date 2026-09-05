"""The rule that decides what to try first, as a term rather than as code.

`how_she_learns_to_look` says the honest thing about its own two halves. What
to try first is a guess, and a wrong one costs time; what to keep is a
judgement, and a wrong one is a language that got worse while every number said
it improved. So the order may be learned and the ruler may not, and that is the
whole safety argument.

What that module did not do is make the order an OBJECT. The counts it keeps
are learned; the rule combining them — Laplace on the history times agreement
from the case — is a Python expression, and a Python expression is the next
authored level up. `growing_at_any_level` made the same shape of mistake one
level down: it collapsed the API for adding makers and then accepted a Python
callable.

Here the rule is a floor term. It takes five numbers and gives a score, higher
first, and it is installed, kept, removed and replaced by exactly the code that
installs, keeps, removes and replaces a head. Nothing about the pipeline knows
which of the two it is holding.

Why this is a step and not the destination
------------------------------------------
The default term below computes what the Python expression computed, to the
precision integers allow, and a test holds the two orders identical on her real
vocabulary. So nothing has improved yet. What has changed is the KIND of thing
the rule is: it is now the same kind as the things she invents, which is what
would have to be true before she could ever write a better one.

Whether she does write a better one is experiment H, and it has not been run.
Nothing here claims recursive self-improvement.
"""

from __future__ import annotations

import logging
from typing import Any

from core.cognition.the_floor_she_stands_on import (
    Code,
    L,
    N,
    OVER,
    PLUS,
    TIMES,
    V,
    build,
    how_long,
    read_back,
    run,
    written_down,
)

__all__ = [
    "AS_FINE_AS_INTEGERS_ALLOW",
    "THE_ORDER",
    "WHAT_THE_ORDER_IS_GIVEN",
    "forget_the_order",
    "how_it_scores",
    "the_order_she_uses",
    "the_order_she_wrote",
    "written_order",
    "order_read_back",
]

logger = logging.getLogger("Aura.TheOrderSheTriesThemIn")

#: The scale integer division works at. Laplace gives a ratio and the floor
#: gives whole numbers, so the ratio is carried as a numerator over this. Large
#: enough that rounding never reorders two words her Python rule separated,
#: which the test beside this checks rather than assumes.
AS_FINE_AS_INTEGERS_ALLOW = 1_000_000_000

#: What the rule is handed, outermost binder first.
WHAT_THE_ORDER_IS_GIVEN: tuple[str, ...] = (
    "places this word already puts right",
    "places there are",
    "winning terms it appeared in",
    "winning terms there have been",
    "symbols it costs to say",
)

#: Laplace on both sides, multiplied, scaled. This is the expression
#: how_she_learns_to_look computed in Python, written where she can reach it:
#:
#:     (agreed + 1) / (places + 2)  ×  (won + 1) / (of + 2)
#:
#: The length is not in it. Length breaks ties after the score, the way it did
#: before, and a rule that could trade score against length would be a
#: different rule rather than the same one moved.
THE_ORDER: Code = build(
    L(
        "agreed",
        L(
            "places",
            L(
                "won",
                L(
                    "of",
                    L(
                        "symbols",
                        OVER(
                            TIMES(
                                TIMES(
                                    PLUS(V("agreed"), N(1)),
                                    PLUS(V("won"), N(1)),
                                ),
                                N(AS_FINE_AS_INTEGERS_ALLOW),
                            ),
                            TIMES(
                                PLUS(V("places"), N(2)),
                                PLUS(V("of"), N(2)),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
)

#: The one in use. Replaced by the same path that replaces a head, and there is
#: no second mechanism for it.
_IN_USE: list[Code] = [THE_ORDER]

#: What one scoring may spend. A search scores every word in the vocabulary
#: once per family, so this is small and a rule that cannot answer inside it is
#: a rule that does not score, which sends the word to the back rather than
#: stopping the search.
_A_SCORING_MAY_SPEND = 20_000


def the_order_she_uses() -> Code:
    """The rule in force."""
    return _IN_USE[0]


def the_order_she_wrote(term: Code) -> Code:
    """Put a different rule in force. Same call shape as installing a head."""
    _IN_USE[0] = term
    logger.info("she is trying them in a new order: %d symbols", how_long(term))
    return term


def forget_the_order() -> Code:
    """Back to the one she started with. The lesion, for experiment H."""
    _IN_USE[0] = THE_ORDER
    return THE_ORDER


def how_it_scores(
    *, agreed: int, places: int, won: int, of: int, symbols: int
) -> int:
    """Score one word under the rule in force, higher first.

    A rule that raises or runs out of fuel scores nothing, which puts the word
    at the back. A learned rule that scores rubbish loses time and keeps
    nothing, because what is kept is decided by a gate this cannot reach.
    """
    work: Any = _IN_USE[0]
    given = (int(agreed), int(places), int(won), int(of), int(symbols))
    try:
        made = run(work, fuel=_A_SCORING_MAY_SPEND)
        for one in given:
            if not isinstance(made, object) or not hasattr(made, "body"):
                return 0
            made = run(
                made.body, (one, *made.env), fuel=_A_SCORING_MAY_SPEND
            )
        return int(made)
    except Exception:  # noqa: BLE001 - any refusal is a score of nothing
        return 0


def written_order() -> dict[str, Any]:
    """The rule as plain data, so an order she wrote survives a restart."""
    return written_down(_IN_USE[0])


def order_read_back(row: Any) -> Code | None:
    """A rule from what was written down, or nothing where it does not read."""
    return read_back(row)

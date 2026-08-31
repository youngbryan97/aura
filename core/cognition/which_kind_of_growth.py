"""Which of the three things "the language grew" means, on this admission.

They are different, they want different evidence, and calling all of them
growth is how a useful abbreviation gets reported as a new concept.

    a shorter name     the word is a term over what she already had. Unfold it
                       and the meaning is unchanged, so the set of behaviours
                       is exactly as large as before. Worth having and not
                       growth of what can be said.

    a longer reach     the same set of behaviours, and one of them crossed the
                       length she can actually search to. Nothing became
                       sayable; something became findable, which for a mind
                       with a budget is the difference that matters.

    a new distinction  no term of the old language denotes it at all, at any
                       length the search could certify. Admitting it is not an
                       abbreviation, and the language now draws a line it could
                       not draw.

The third is the one worth being careful about, and the care is one rule: it
requires the search that found nothing to have FINISHED. A search that ran out
of time has said nothing about the language, only about the clock — which is
the difference this codebase already insists on between a search that went
badly and a language that cannot say it. Measured here: over a language of five
given words, six random behaviours were each unreachable after some four and a
half million terms, so the third kind is genuinely available and not a
formality.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "A_LONGER_REACH",
    "A_NEW_DISTINCTION",
    "A_SHORTER_NAME",
    "UNDECIDED",
    "WhichGrowth",
    "which_kind_of_growth",
]

logger = logging.getLogger("Aura.WhichKindOfGrowth")

#: It is a term over what she had. Unfolding it changes nothing.
A_SHORTER_NAME = "a shorter name for something already sayable"

#: The same meanings, one of them now inside the length she searches to.
A_LONGER_REACH = "a longer reach, at the same meanings"

#: No term of the old language denotes it, over a search that finished.
A_NEW_DISTINCTION = "a distinction the old language could not draw"

#: The search did not finish, so nothing has been established either way.
UNDECIDED = "not searched far enough to say"


@dataclass(frozen=True)
class WhichGrowth:
    """Which kind, and the numbers that decided it."""

    kind: str
    #: Length of the shortest saying in the OLD language, if one was found.
    without: int | None
    #: The longest term the search can reach.
    horizon: int
    #: How far this was certified to.
    certified_to: int
    #: Whether the search that found nothing actually finished.
    finished: bool
    #: How the old language said it, where it could.
    how: str = ""

    @property
    def is_a_new_distinction(self) -> bool:
        return self.kind == A_NEW_DISTINCTION

    def describes(self) -> str:
        if self.kind == A_SHORTER_NAME:
            return f"{self.kind}: the old language says it in {self.without} — {self.how}"
        if self.kind == A_LONGER_REACH:
            return (
                f"{self.kind}: the old language says it in {self.without}, past "
                f"the {self.horizon} she searches to"
            )
        if self.kind == A_NEW_DISTINCTION:
            return (
                f"{self.kind}: nothing up to {self.certified_to} says it, and the "
                "search finished"
            )
        return (
            f"{self.kind}: nothing up to {self.certified_to} said it, and the "
            "search did not finish — which is a fact about the clock"
        )


def which_kind_of_growth(
    says_it: Callable[[Any], bool],
    *,
    the_old_language: Mapping[str, Any],
    horizon: int,
    certify_to: int | None = None,
    holes: int = 2,
    within: float = 60.0,
) -> WhichGrowth:
    """Decide it, by looking for the behaviour in the language she already had.

    ``horizon`` is the longest term her search can reach — from
    :func:`core.cognition.what_an_invention_buys.the_horizon_of`, not chosen
    here. ``certify_to`` is how far past it to look before saying the old
    language cannot say it at all; further is a stronger claim and costs more.
    """
    from core.cognition.what_an_invention_buys import the_shortest_way_to_say

    look_to = int(certify_to if certify_to is not None else horizon * 2)
    without, how, finished = the_shortest_way_to_say(
        says_it, the_old_language, up_to=look_to, holes=holes, within=within
    )
    if without is not None:
        kind = A_SHORTER_NAME if without <= horizon else A_LONGER_REACH
    elif finished:
        kind = A_NEW_DISTINCTION
    else:
        kind = UNDECIDED
    found = WhichGrowth(
        kind=kind,
        without=without,
        horizon=int(horizon),
        certified_to=look_to,
        finished=finished,
        how=how,
    )
    logger.info("which kind of growth: %s", found.describes())
    return found

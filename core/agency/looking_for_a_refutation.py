"""Looking for a refutation: before she commits to the move her search chose, a look made only to break it.

A search spreads its time over every move. The move it chooses has been looked at as deeply as the rest, and the reply
that breaks it may lie just past where the search stopped. A look at the situation the move leaves, from the other
side, spends all its time on that one move, and so goes further down the replies to it. Where it finds the move worth
less than the search believed, by more than the search's own judgement moved while it looked, the move is refuted, and
the best move without it is looked for; the one of the two that holds up better is played.

The time is not added to her turn: the search, the look for a refutation and, where it is needed, the look for another
move share what the turn has (suggestions of 23 November 2026, item 12). Nothing here knows a game: a search is any
search over positions that answers with a move and a score from the mover's side.

Measured 23 November 2026, 40 games on each of two seeds against Stockfish limited to 2500 (300 ms a move; she at
0.3 s a move in all): without it she scored 0.300 and 0.350; with a third of each move's time given to it, 0.175 and
0.125; with half, 0.212 and 0.362. What it takes from the search costs more than the refutations it finds, so her play
does not call it; it is kept for a turn with time to spare, where the search has settled and a look of its own costs
the search nothing.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

logger = logging.getLogger("Aura.LookingForARefutation")

__all__ = ["tried_against_its_refutation"]


def tried_against_its_refutation(search: Any, family: Any, position: Any, found: Any, *, seconds: float,
                                 weights: Any = None, earlier: list | None = None, **how: Any) -> Any:
    """``found``, the move a search chose in ``position``, kept or replaced after a look for its refutation of half of
    ``seconds`` and, where it is refuted, a look for another move of the other half."""
    from core.agency.pieces_by_their_kind import after

    if found is None or found.move is None or seconds <= 0:
        return found
    earlier = list(earlier or [])
    left = after(family, position, found.move)
    against = search(family, left, seconds=seconds / 2, weights=weights, earlier=[*earlier, position], **how)
    if against.move is None:
        return found
    holds = -int(against.score)
    slack = max(int(found.unsteady), int(against.unsteady), 1)
    if holds >= int(found.score) - slack:
        return found
    aside = [*how.pop("set_aside", []), found.move]
    other = search(family, position, seconds=seconds / 2, weights=weights, earlier=earlier, set_aside=aside, **how)
    if other.move is None or int(other.score) <= holds:
        logger.info("her move held up worse than she thought (%+d against %+d), and nothing else held up better",
                    holds, int(found.score))
        return replace(found, score=holds)
    logger.info("her move was refuted (%+d against %+d): another holds up at %+d", holds, int(found.score),
                int(other.score))
    return replace(other, by="the first was refuted")

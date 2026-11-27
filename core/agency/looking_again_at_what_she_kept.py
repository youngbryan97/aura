"""Looking again at the positions she keeps for learning how the pieces stand together, with the judgement she has now.

Each kept position holds two judgements of it: her first look, and what her longer look found. Both were taken with the
weights she had when the position was gathered. How the pieces stand together is learned from the gap between them,
and the weights she plays by change whenever her practice keeps something: the first look on file is then no longer
hers, and the layer is fitted to a gap she no longer has. So before each fit every kept position is given her first
look as it is now, which costs little. And, a round at a time, the positions whose longer look is oldest are looked at
for longer again with the weights she has now, which costs a search each, so the longer looks they hold come from a
judgement as good as hers. Re-reading old positions with a newer judgement is what Schrittwieser and others (2021,
"Online and offline reinforcement learning by planning with a learned model") called reanalysis.

A position is kept as its pieces as the side to move sees them, so it is set out again with that side to move first
and its pieces standing where it saw them; which pieces never moved is read from where they began. Nothing here knows
a game.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("Aura.LookingAgain")

__all__ = ["a_position_from_its_parts", "first_looks_now", "looked_at_again"]


def a_position_from_its_parts(parts: Any, kinds: list[str], squares: int, columns: int, start: dict) -> Any:
    """A kept position set out again: the side that was to move first, its pieces where it saw them."""
    from core.agency.pieces_by_their_kind import FIRST, SECOND, Position

    cells: dict[tuple[int, int], tuple[int, str]] = {}
    for index in [int(i) for i in parts]:
        slot, rest = divmod(index, 2 * squares)
        theirs, square = divmod(rest, squares)
        cells[(square // columns, square % columns)] = (SECOND if theirs else FIRST, kinds[slot])
    unmoved = frozenset(square for square, held in cells.items() if start.get(square) == held)
    return Position(cells=cells, turn=FIRST, unmoved=unmoved)


def first_looks_now(store: Any, family: Any, weights: Any, start: dict) -> list[float] | None:
    """Her first look at every kept position, as the weights she has now judge it; None where it cannot be had."""
    from core.agency.searching_by_kind import _compiled, described, placed

    engine = _compiled()
    if engine is None or not hasattr(engine, "seen_as_practice_sees") or not store.parts:
        return None
    positions = [placed(a_position_from_its_parts(p, store.kinds, store.squares, store.columns, start))
                 for p in store.parts]
    try:
        _order, rows = engine.seen_as_practice_sees(described(family), positions, weights.as_json())
    # not a failure: a kept position her rules as they are now cannot set out is no first look to refresh.
    except ValueError as why:
        logger.info("her first looks at what she keeps could not be taken again: %s", why)
        return None
    return [float(first_look) for _cells, first_look, _quiet, _turn in rows]


def looked_at_again(store: Any, family: Any, weights: Any, start: dict, *, now: int, seconds: float,
                    nodes: int, threads: int = 1) -> int:
    """The kept positions whose longer look is oldest, looked at again for ``nodes`` with ``weights`` for about
    ``seconds``, each marked as looked at in round ``now``. How many were."""
    from core.agency.searching_by_kind import search

    if not store.parts:
        return 0
    labelled = store.rounds_labelled()
    store.labelled = labelled
    began, done = time.monotonic(), 0
    for index in sorted(range(len(labelled)), key=lambda i: labelled[i]):
        # One at least, however short the round: a round that looks at nothing is no round.
        if labelled[index] >= now or (done and time.monotonic() - began >= seconds):
            break
        position = a_position_from_its_parts(store.parts[index], store.kinds, store.squares, store.columns, start)
        found = search(family, position, seconds=max(seconds, 1.0), weights=weights, nodes=nodes, threads=threads)
        done += 1
        if found.move is None or getattr(found, "mate", False) or abs(int(found.score)) >= 3000:
            # A position whose look ends the game says nothing of how the pieces stand: left as it was, marked looked.
            store.labelled[index] = now
            continue
        store.slow[index] = float(found.score)
        store.labelled[index] = now
    return done

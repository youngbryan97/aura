"""A small win that costs no ground is taken, and one that lasts is worth more: money in the bank, for any world of
places and things that move.

Bryan, 24 November 2026: "putting money in the bank? little wins that add up. like in boxing or mma. you may rack up
kicks or shots to the body or the legs in opening rounds that arent necessarily meant to ko your opponent right there
and then. but add up and pay large dividends in the later rounds ... if you can get a small W and not lose much
ground, it is strategic to take it ... if it doesnt put you out of position and doesnt harm your strategy or give you
openings otherwise (unless deemed worth it), you take it. just to make the fight more annoying or slightly harder or to
put the opponent in a less advantageous position in the future."

Her judgement of a situation says which acts are best, and between acts it cannot tell apart (within its own
unsteadiness: how far its answer moves from one look to the next) it said nothing. Three questions settle those, over
positioning in time (core/cognition/where_forces_can_meet.py):

- What the act costs in ground: whether, after it, fewer of hers stand on what she must keep safe and around it, or
  more of theirs can gather there within t moves, or fewer of hers, for any t. An act that loses ground is not
  banked, however much it takes.
- What lasts of it: how much less the other side can reach a few moves out, toward what it must hold and over all: a
  mover of theirs gone, or boxed in, or kept from its own places. A gain that only blocks for a move is gone next move;
  one that holds at the horizon is still there in the later rounds.
- Which to take: among the acts her judgement puts within its unsteadiness of the best, the one that banks most and
  loses no ground; none where none banks anything.

Nothing here knows a game, and no weight is set here: what a lasting gain is worth beside a judged one is what her
games teach (a judgement that learns it takes ``lasting_gain`` as one more thing it counts).
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from core.cognition.where_forces_can_meet import Mover, Reach, balance, time_to_reach

__all__ = ["Act", "ground_lost", "lasting_gain", "what_to_bank"]


@dataclass(frozen=True)
class Act:
    """An act: what her judgement gives the situation after it, from her side, and the movers as they stand after."""

    name: Hashable
    judged: float
    after: tuple[Mover, ...]


def _reach_left(movers: Sequence[Mover], reach: Reach, side: int, horizon: int,
                places: Iterable[object] | None = None) -> int:
    """How many places the movers of ``side`` can reach within ``horizon``, together, counting each mover's own
    reach (or only ``places``, where given)."""
    occupied = frozenset(m.at for m in movers)
    wanted = frozenset(places) if places is not None else None
    total = 0
    for m in movers:
        if m.side != side:
            continue
        reached = time_to_reach(m, reach, occupied, horizon)
        total += sum(1 for place, t in reached.items() if t > 0 and (wanted is None or place in wanted))
    return total


def lasting_gain(before: Sequence[Mover], after: Sequence[Mover], reach: Reach, side: int,
                 matters: Mapping[int, Iterable[object]], horizon: int) -> int:
    """How much less the other side can reach at the horizon after the act: toward what it must hold, and over all."""
    other = 1 - side
    theirs = list(matters.get(other, ()))
    toward = _reach_left(before, reach, other, horizon, theirs) - _reach_left(after, reach, other, horizon, theirs)
    overall = _reach_left(before, reach, other, horizon) - _reach_left(after, reach, other, horizon)
    return toward + overall


def ground_lost(before: Sequence[Mover], after: Sequence[Mover], reach: Reach, side: int,
                matters: Mapping[int, Iterable[object]], horizon: int,
                matters_after: Mapping[int, Iterable[object]] | None = None) -> bool:
    """Whether after the act fewer of hers stand on what she must keep safe and around it, or, within t moves for some
    t up to the horizon, more of theirs can gather there or fewer of hers. ``matters_after`` is where what matters
    stands after the act, where the act moves it."""
    own = list(matters.get(side, ()))
    own_after = list((matters_after or matters).get(side, ()))
    if not own:
        return False
    standing = sum(1 for m in before if m.side == side and m.at in set(own))
    standing_after = sum(1 for m in after if m.side == side and m.at in set(own_after))
    if standing_after < standing:
        return True
    was, now = balance(before, reach, own, horizon), balance(after, reach, own_after, horizon)
    zero = [0] * horizon
    theirs_up = any(n > b for n, b in zip(now.get(1 - side, zero), was.get(1 - side, zero), strict=True))
    hers_down = any(n < b for n, b in zip(now.get(side, zero), was.get(side, zero), strict=True))
    return theirs_up or hers_down


def what_to_bank(acts: Sequence[Act], before: Sequence[Mover], reach: Reach, side: int,
                 matters: Mapping[int, Iterable[object]], horizon: int, within: float,
                 matters_after: Mapping[Hashable, Mapping[int, Iterable[object]]] | None = None) -> Act | None:
    """Among ``acts`` her judgement puts within ``within`` of the best, the one whose lasting gain is largest of those
    that lose no ground; None where none of them banks anything."""
    if not acts:
        return None
    best = max(act.judged for act in acts)
    banked: list[tuple[int, float, Act]] = []
    for act in acts:
        if best - act.judged > within:
            continue
        moved_to = (matters_after or {}).get(act.name)
        if ground_lost(before, act.after, reach, side, matters, horizon, moved_to):
            continue
        gain = lasting_gain(before, act.after, reach, side, moved_to or matters, horizon)
        if gain > 0:
            banked.append((gain, act.judged, act))
    if not banked:
        return None
    return max(banked, key=lambda item: (item[0], item[1]))[2]

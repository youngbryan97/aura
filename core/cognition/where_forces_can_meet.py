"""How soon each side can bring what it has to the places that matter: positioning in time, for any world of places and
things that move.

Bryan, 24 November 2026, on strategy: "If this goes here, then this takes this, which leads a spot open here, and from
there I can maneuver here"; "If I can bait this to move to this spot, then I can slowly inch here, and take it here"; "I
can draw defenses or offenses here and operate on the other side slowly"; "I want my offense to converge here". Her
search sees what happens within its horizon, and her judgement of a position counts what stands where now. Nothing of
hers held where things stand in time: how many of its own moves each mover needs to reach each place, so which side can
be stronger at a place before the other can answer there, which of theirs is the only one that can answer at two places
at once, and which of her own moves bring her nearer to where she means to be strong.

A world here is places, movers (each with a side and a worth, standing somewhere), and for a mover where it can go from
a place in one move given what stands where (its reach: learned, or given). Nothing here knows a game: a courier on a
road network, a unit on a map and a piece on a board are all movers, and a place that matters is whatever the caller
says matters (where a thing to be kept safe stands, a depot, a gate).

- ``time_to_reach``: the fewest of its own moves a mover needs to each place, within a horizon, breadth first over its
  reach, with what stands in its way where it stands now; ``time_to_reach_safely`` the same by places where it
  cannot be taken on arriving.
- ``arrivals``: for a place, each side's movers and how soon each can be there.
- ``balance``: for places that matter, how many of each side can be there within t moves, for each t up to the
  horizon. These are counts for a judgement to learn its weights for, not a judgement: how much a mover two moves away
  is worth beside one that is there now is what a world's own games teach.
- ``overloaded``: movers that are the only one of their side able to reach, in time, two or more places the other
  side converges on. Drawing it to one leaves the other: what "draw the defence here and operate on the other side"
  rests on.
- ``nearer``: a mover's next places by how much each shortens its way to places that matter: maneuvering.
- ``intended``: her moves that serve converging where she can be stronger, that reach for a defender with too much to
  hold, or that keep what she must keep safe where hers can gather sooner than theirs, each with a share; for a caller
  that looks at what it intends first.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

__all__ = ["Mover", "arrivals", "balance", "intended", "nearer", "overloaded", "time_to_reach",
           "time_to_reach_safely"]

Place = Hashable


@dataclass(frozen=True)
class Mover:
    """Something that moves: who it is, whose it is, where it stands, and what it is worth to its side."""

    name: Hashable
    side: int
    at: Place
    worth: float = 1.0


#: Where ``mover`` can go in one move from ``place``, given the places standing occupied (its own start left free).
Reach = Callable[[Mover, Place, frozenset], Iterable[Place]]


def time_to_reach(mover: Mover, reach: Reach, occupied: Iterable[Place], horizon: int) -> dict[Place, int]:
    """The fewest of its own moves ``mover`` needs to each place it can reach within ``horizon``, its own place at 0.

    Others are taken to stand where they stand now: a mover passes where its reach lets it, and may end a move where
    another stands (its reach says whether)."""
    standing = frozenset(occupied) - {mover.at}
    reached: dict[Place, int] = {mover.at: 0}
    frontier = deque([mover.at])
    while frontier:
        here = frontier.popleft()
        moves_so_far = reached[here]
        if moves_so_far >= horizon:
            continue
        for there in reach(mover, here, standing):
            if there not in reached:
                reached[there] = moves_so_far + 1
                # A place another stands on is reached, not passed through: what stands there is met there.
                if there not in standing:
                    frontier.append(there)
    return reached


def time_to_reach_safely(mover: Mover, reach: Reach, occupied: Iterable[Place], horizon: int,
                         safe: Callable[[Mover, Place, frozenset], bool]) -> dict[Place, int]:
    """As ``time_to_reach``, going only by places where ``safe`` says the mover cannot be taken on arriving, given
    what stands where (the world's own rules of taking say which): a way through on cover. Bryan, 24 November 2026: he
    "walked" pieces through the middle between pieces of both sides, knowing "the physics and rules of the game allow
    me to do it safely"."""
    standing = frozenset(occupied) - {mover.at}
    reached: dict[Place, int] = {mover.at: 0}
    frontier = deque([mover.at])
    while frontier:
        here = frontier.popleft()
        if reached[here] >= horizon:
            continue
        for there in reach(mover, here, standing):
            if there in reached or not safe(mover, there, standing):
                continue
            reached[there] = reached[here] + 1
            if there not in standing:
                frontier.append(there)
    return reached


def _times(movers: Sequence[Mover], reach: Reach, horizon: int) -> dict[Hashable, dict[Place, int]]:
    occupied = frozenset(m.at for m in movers)
    return {m.name: time_to_reach(m, reach, occupied, horizon) for m in movers}


def arrivals(movers: Sequence[Mover], reach: Reach, place: Place, horizon: int,
             times: Mapping[Hashable, Mapping[Place, int]] | None = None) -> dict[int, list[tuple[Mover, int]]]:
    """Each side's movers that can be at ``place`` within ``horizon``, soonest first."""
    times = times if times is not None else _times(movers, reach, horizon)
    found: dict[int, list[tuple[Mover, int]]] = {}
    for m in movers:
        t = times[m.name].get(place)
        if t is not None and m.at != place:
            found.setdefault(m.side, []).append((m, t))
    for side in found:
        found[side].sort(key=lambda pair: pair[1])
    return found


def balance(movers: Sequence[Mover], reach: Reach, places: Iterable[Place], horizon: int,
            times: Mapping[Hashable, Mapping[Place, int]] | None = None) -> dict[int, list[int]]:
    """For each side, how many of its movers can be at the places that matter within t moves, for t from 1 to
    ``horizon`` (each mover counted once, at the soonest it reaches any of them)."""
    times = times if times is not None else _times(movers, reach, horizon)
    wanted = frozenset(places)
    counts: dict[int, list[int]] = {}
    for m in movers:
        soonest = min((t for place, t in times[m.name].items() if place in wanted and place != m.at), default=None)
        row = counts.setdefault(m.side, [0] * horizon)
        if soonest is not None:
            for t in range(max(soonest, 1), horizon + 1):
                row[t - 1] += 1
    return counts


def overloaded(movers: Sequence[Mover], reach: Reach, contested: Iterable[Place], side: int, horizon: int,
               times: Mapping[Hashable, Mapping[Place, int]] | None = None) -> list[tuple[Mover, list[Place]]]:
    """Movers of ``side`` that are the only one of their side to reach, no later than the other side, two or more of
    the ``contested`` places: each with the places it alone holds."""
    times = times if times is not None else _times(movers, reach, horizon)
    holds: dict[Hashable, list[Place]] = {}
    by_name = {m.name: m for m in movers}
    for place in contested:
        came = arrivals(movers, reach, place, horizon, times)
        theirs = [t for _m, t in came.get(1 - side, [])]
        if not theirs:
            continue
        soonest_other = min(theirs)
        in_time = [m for m, t in came.get(side, []) if t <= soonest_other]
        if len(in_time) == 1:
            holds.setdefault(in_time[0].name, []).append(place)
    return [(by_name[name], places) for name, places in holds.items() if len(places) >= 2]


def nearer(mover: Mover, reach: Reach, occupied: Iterable[Place], places: Iterable[Place],
           horizon: int) -> list[tuple[Place, int]]:
    """The places ``mover`` can go next, each with how many moves it would then still need to the nearest place that
    matters, fewest first: how it closes in."""
    standing = frozenset(occupied) - {mover.at}
    wanted = frozenset(places)
    found: list[tuple[Place, int]] = []
    for there in reach(mover, mover.at, standing):
        moved = Mover(mover.name, mover.side, there, mover.worth)
        after = time_to_reach(moved, reach, standing | {there}, horizon)
        left = min((t for place, t in after.items() if place in wanted), default=None)
        if left is not None:
            found.append((there, left))
    return sorted(found, key=lambda pair: pair[1])


def _how_soon_they_come(movers: Sequence[Mover], times: Mapping[Hashable, Mapping[Place, int]], side: int,
                        place: Place, horizon: int) -> tuple[int, int]:
    """How soon the first of the other side can be at ``place`` (past the horizon where none can), and, as a second
    key, how few of them can be there by then: larger is safer."""
    coming = [times[m.name].get(place) for m in movers if m.side != side]
    coming = [t for t in coming if t is not None]
    if not coming:
        return horizon + 1, 0
    first = min(coming)
    return first, -sum(1 for t in coming if t == first)


def intended(movers: Sequence[Mover], reach: Reach, side: int, matters: Mapping[int, Iterable[Place]],
             horizon: int) -> dict[tuple[Place, Place], float]:
    """Her moves (``side``'s) that serve her position in time, each with a share: moves that bring a mover nearer to
    the places that matter on the other side where she can be stronger there than they can answer, and moves toward a
    place that a mover of theirs alone holds along with another. ``matters`` maps each side to the places that matter to
    it (where what it must keep safe stands, and around). Keyed by (from, to)."""
    times = _times(movers, reach, horizon)
    occupied = frozenset(m.at for m in movers)
    target = list(matters.get(1 - side, ()))
    if not target:
        return {}
    weighed = balance(movers, reach, target, horizon, times)
    mine, theirs = weighed.get(side, [0] * horizon), weighed.get(1 - side, [0] * horizon)
    scores: dict[tuple[Place, Place], float] = {}
    # Converging: worth doing where, at some time within the horizon, she can bring more there than they can.
    if any(m > t for m, t in zip(mine, theirs, strict=True)):
        for mover in movers:
            if mover.side != side:
                continue
            now = min((t for place, t in times[mover.name].items() if place in target and place != mover.at),
                      default=horizon + 1)
            for there, left in nearer(mover, reach, occupied, target, horizon):
                if left < now:
                    scores[(mover.at, there)] = scores.get((mover.at, there), 0.0) + (now - left)
    # Overload: a place one of theirs alone holds, among two or more it holds, is where to come.
    for _defender, places in overloaded(movers, reach, target, 1 - side, horizon, times):
        for mover in movers:
            if mover.side != side:
                continue
            for there, left in nearer(mover, reach, occupied, places, horizon):
                if left == 0:
                    scores[(mover.at, there)] = scores.get((mover.at, there), 0.0) + 1.0
    # Keeping safe: where the other side can bring more to what she must keep safe than she can answer with, the
    # moves that take what is kept safe toward where hers outnumber theirs, or bring one of hers nearer to it.
    own = list(matters.get(side, ()))
    if own:
        held = balance(movers, reach, own, horizon, times)
        against, for_it = held.get(1 - side, [0] * horizon), held.get(side, [0] * horizon)
        if any(a > f for a, f in zip(against, for_it, strict=True)):
            for mover in movers:
                if mover.side != side:
                    continue
                if mover.at in own:
                    # What is kept safe itself: toward a place the soonest of theirs comes to later, or as soon but
                    # fewer of them by then.
                    here = _how_soon_they_come(movers, times, side, mover.at, horizon)
                    for there in reach(mover, mover.at, occupied - {mover.at}):
                        later = _how_soon_they_come(movers, times, side, there, horizon)
                        if later > here:
                            scores[(mover.at, there)] = scores.get((mover.at, there), 0.0) + 1.0
                    continue
                now = min((t for place, t in times[mover.name].items() if place in own and place != mover.at),
                          default=horizon + 1)
                for there, left in nearer(mover, reach, occupied, own, horizon):
                    if left < now:
                        scores[(mover.at, there)] = scores.get((mover.at, there), 0.0) + (now - left)
    total = sum(scores.values())
    return {move: score / total for move, score in scores.items()} if total else {}

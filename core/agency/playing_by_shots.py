"""Playing by shots: pulling or holding a press and letting go, watching where it sends something, and doing better.

A world that waits until she sends something into it (a putt, a toss, a
launch, a top let go into a ring) is not played by steering. Each turn is one
act with a measure (core/agency/how_hard_and_which_way.py): she presses at a
place, pulls some way or holds some time, lets go, and watches what sets off,
where it goes and where it stops, and whether anything counted went up. Then
the next setting, from what the shots so far say.

What to send from: the place her own thing is, where she knows it, else the
shapes on the picture that stand out from what is around them, each tried in
turn until one sends something. What to send it to: the thing she has learned
is worth meeting, where there is one; else no place, and what pays is found by
searching the settings for what is counted.

Nothing here names a game, a key or a kind of thing.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from core.agency.how_hard_and_which_way import HOLD, PULL, Setting, Shot, Shots

logger = logging.getLogger("Aura.PlayingByShots")

__all__ = ["play_by_shots", "sends_by_letting_go"]

#: How long after letting go something must have set off for it to have been sent, and the longest a shot is
#: watched before what it sent is taken to have stopped where it is.
SETS_OFF_WITHIN_S = 1.0
WATCHED_FOR_S = 6.0

#: How slow, in working pixels a second, a thing must go for this long to have stopped.
STOPPED_BELOW = 6.0
STOPPED_FOR_S = 0.4

#: How many steps a pull is made in, so the page sees a drag and not a jump.
PULL_STEPS = 8

#: Words that say a thing is sent by a press pulled or held and let go. Not "drag": a thing dragged is carried to a
#: place and stays there (core/agency/putting_things_in_place.py); LIVE 2026-10-09 "click and drag it into place"
#: had her letting go of shots at a title screen for a minute. Not "aim" or "power" alone: a shooter is aimed with
#: the mouse and a platformer has power-ups, and "Mouse to aim" sent her into shots on a game's menu the same day.
_SENDING_WORDS = (
    "pull", "release", "let go", "launch", "fling", "sling", "toss", "putt", "power meter", "power bar",
    "hold the mouse", "click and hold", "hold down the mouse", "hold the button", "throw",
)


def sends_by_letting_go(told: str) -> bool:
    """Whether words a thing says of itself speak of sending by a press pulled or held and let go."""
    lowered = " ".join(told.lower().split())
    return any(word in lowered for word in _SENDING_WORDS)


async def _let_go(hands: Any, start: tuple[float, float], setting: Setting, way: str) -> None:
    """The act itself: press at ``start``, pull or hold, and let go."""
    import asyncio

    await hands.press(*start)
    try:
        if way == PULL:
            for step in range(1, PULL_STEPS + 1):
                share = step / PULL_STEPS
                await hands.point(start[0] + setting.across * share, start[1] + setting.down * share)
                await asyncio.sleep(0.015)
        else:
            await asyncio.sleep(setting.held_s)
    finally:
        await hands.release()


async def _where_it_went(look: Callable[[], Awaitable[Any]], moves: Any, let_go_at: float,
                         read_words: Callable[[Any], list[dict[str, Any]]] | None, counters: Any,
                         going_already: frozenset[int] = frozenset()) -> tuple[tuple[float, float] | None, int]:
    """Watch after a shot: the thing that set off first after the letting go, followed until it stops or goes; where,
    as shares of the picture, and what was counted meanwhile.

    A thing already going before she pressed was not sent by her: LIVE 2026-10-09 a title's word bobbing on its
    own was taken for what her press sent, and she went on shooting at it.
    """
    sent: int | None = None
    last: tuple[float, float] | None = None
    slow_since: float | None = None
    gained = 0
    read_at = 0.0
    while True:
        seen = await look()
        if seen is None:
            break
        picture, at = seen
        happened = moves.see(picture, at)
        tall, wide = moves.shape
        if sent is None and at - let_go_at <= SETS_OFF_WITHIN_S:
            setting_off = [thing for thing in moves.things.values()
                           if math.hypot(thing.vx, thing.vy) > 2 * STOPPED_BELOW and thing.number not in going_already]
            if setting_off:
                sent = max(setting_off, key=lambda thing: math.hypot(thing.vx, thing.vy)).number
        if sent is not None:
            thing = moves.things.get(sent)
            if thing is None:
                gone = next((h for h in happened if h.get("what") == "gone" and h.get("thing") == sent), None)
                if gone is not None:
                    last = (float(gone["last_visible"][0]) / max(1, wide), float(gone["last_visible"][1]) / max(1, tall))
                break
            last = (thing.x / max(1, wide), thing.y / max(1, tall))
            if math.hypot(thing.vx, thing.vy) < STOPPED_BELOW:
                slow_since = slow_since if slow_since is not None else at
                if at - slow_since >= STOPPED_FOR_S:
                    break
            else:
                slow_since = None
        elif at - let_go_at > SETS_OFF_WITHIN_S:
            break
        if read_words is not None and counters is not None and at - read_at >= 0.5:
            read_at = at
            for verdict in counters.read(read_words(picture), at, None):
                gained += int(verdict["by"]) if verdict["what"] == "gain" else 0
        if at - let_go_at > WATCHED_FOR_S:
            break
    return last, gained


def _places_to_send_from(picture: Any, keep: dict[str, Any]) -> list[tuple[float, float]]:
    """Where a press might send something from: where it did before, else what stands out on the picture, largest first."""
    from core.perception.shapes_that_look_pressable import pressable_shapes

    known = keep.get("sends_from")
    places: list[tuple[float, float]] = [tuple(known)] if known else []
    for shape in pressable_shapes(picture):
        place = (float(shape.get("center_x", 0.5)), float(shape.get("center_y", 0.5)))
        if all(math.dist(place, other) > 0.05 for other in places):
            places.append(place)
    if all(math.dist((0.5, 0.5), other) > 0.05 for other in places):
        places.append((0.5, 0.5))
    return places


async def play_by_shots(
    look: Callable[[], Awaitable[Any]],
    hands: Any,
    *,
    seconds: float,
    keep: dict[str, Any] | None = None,
    say: Callable[[str], Any] | None = None,
    read_words: Callable[[Any], list[dict[str, Any]]] | None = None,
    aim_at: Callable[[Any], tuple[float, float] | None] | None = None,
    ways: Sequence[str] = (PULL, HOLD),
) -> dict[str, Any]:
    """Send things by letting go of a press, shot after shot, for ``seconds``; doing better each time.

    ``hands`` has ``press(x, y)``, ``point(x, y)`` and ``release()``, places as shares of the picture. ``aim_at``
    says, of what ``look`` shows, where a thing sent ought to end up, or None where nothing says. ``keep`` carries what
    she found out (where a press sends from, which way of letting go sends, the shots) into the next stretch.
    """
    from core.agency.what_meeting_things_does import Readouts
    from core.perception.what_moves_in_the_picture import WhatMoves

    keep = keep if keep is not None else {}
    began = time.monotonic()
    counters = Readouts()
    shots_by_way: dict[str, Shots] = keep.setdefault("shots", {})
    taken = 0
    said: set[str] = set()

    def tell(line: str) -> None:
        if say is not None and line not in said:
            said.add(line)
            say(line)

    seen = await look()
    if seen is None:
        return {"shots": 0, "ended": "the picture could not be taken"}
    places = _places_to_send_from(seen[0], keep)
    order = [way for way in ways if way in shots_by_way and shots_by_way[way].sends_anything] or list(ways)
    # One watch of the picture for all the shots, so what stays put is known before anything is sent.
    moves = WhatMoves(kinds=keep.get("kinds"))
    moves.see(seen[0], seen[1])
    gains = 0
    ended = "out of time"
    while time.monotonic() - began < seconds:
        if keep.get("sends_from") is None and not places:
            ended = "nothing sends anything"
            break
        start = tuple(keep.get("sends_from") or places[0])
        way = order[0]
        shots = shots_by_way.setdefault(way, Shots(way=way))
        seen = await look()
        if seen is None:
            ended = "the picture could not be taken"
            break
        moves.see(seen[0], seen[1])
        while not moves.has_looked:
            seen = await look()
            if seen is None:
                break
            moves.see(*seen)
        if seen is None:
            ended = "the picture could not be taken"
            break
        aim = aim_at(seen[0]) if aim_at is not None else None
        setting = shots.next_setting(aim, start)
        going = frozenset(n for n, thing in moves.things.items() if math.hypot(thing.vx, thing.vy) > STOPPED_BELOW)
        await _let_go(hands, start, setting, way)
        # When it was let go, on the pictures' own clock: the time of the last picture before it, and the act's length.
        let_go_at = seen[1] + (setting.held_s if way == HOLD else PULL_STEPS * 0.015)
        ended_at, gained = await _where_it_went(look, moves, let_go_at, read_words, counters, going_already=going)
        keep["kinds"] = moves.kinds
        shot = Shot(setting=setting, ended_at=ended_at, gained=gained, aimed_at=aim)
        shots.took(shot)
        taken += 1
        gains += max(0, gained)
        logger.info("shot %d (%s from %s, %s): ended at %s, gained %d; %s", taken, way, start, setting, ended_at, gained, shots.how_it_goes())
        if ended_at is None and not shots.sends_anything and len(shots.tried) >= 2:
            # Two tries of one way from one place sent nothing: the other way, then the next place.
            if len(order) > 1:
                order = order[1:] + order[:1]
                if all(shots_by_way.get(w) is not None and len(shots_by_way[w].tried) >= 2 and not shots_by_way[w].sends_anything for w in order):
                    places = places[1:]
                    shots_by_way.clear()
            else:
                places = places[1:]
                shots_by_way.clear()
            continue
        if ended_at is not None and keep.get("sends_from") is None:
            keep["sends_from"] = start
            tell("Pressing there and letting go sends it; now finding how hard and which way.")
        if shot.got_there:
            tell("That's the measure of it.")
    return {"shots": taken, "gains": gains, "ended": ended, "sends_from": keep.get("sends_from"),
            "how_it_goes": {way: shots.how_it_goes() for way, shots in shots_by_way.items()}}

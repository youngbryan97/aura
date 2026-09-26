"""Acting in a world seen through a camera: learn the body, then go where she is asked.

The screen loop plays boards, where an act moves things between places. A
game seen through a camera is a different kind of world, and SIMA 2's whole
evaluation is in worlds of that kind. This is the loop for them, made of
parts that each know nothing about any game:

    her eyes           the window's pixels, and the words on it with their boxes
    her body           what each act does to the view (`how_the_view_moves`)
    her hands          chunks of keys and mouse, played slot by slot (`hands`)
    a trip             find, face, walk, use (`going_to_what_she_sees`)

A world is met the way a person picks up a controller they have not held:
move the mouse a little and watch which way the view goes, press each likely
key and watch what happens. The first mouse move is one point, doubled until
the view slides by something the measurement can tell from nothing, so how
far to move is found rather than chosen. The key that grows the middle of the
view is the key that walks.

What she says as she goes is composed from what happened, never asked for:
which way she turned, what she walked toward, what the screen told her to
press, and why the trip ended.

The machine side is passed in, so the whole loop runs against a simulated
world in tests and against a real window live.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.agency.going_to_what_she_sees import GoingTo, seen_named
from core.agency.what_hands_do import Chunk, Slot
from core.perception.how_the_view_moves import WhatMyHandsDoToTheView, grey

__all__ = ["ACameraWorld", "Trip", "go_to", "learn_the_body", "live_camera_world"]


@dataclass
class ACameraWorld:
    """What the loop needs from the machine."""

    #: The window's pixels and the words on it, with their boxes.
    look: Callable[[], tuple[Any, list[dict[str, Any]]]]
    #: Play a chunk; returns once it has been played.
    play: Callable[[Chunk], Awaitable[Any]]


@dataclass
class Trip:
    """One trip, as it went."""

    named: str
    said: list[str] = field(default_factory=list)
    chunks: int = 0
    ended: str = ""
    done: bool = False


async def learn_the_body(
    world: ACameraWorld, *, keys: Sequence[str], slot_s: float, most_travel: int = 4096
) -> WhatMyHandsDoToTheView:
    """What the mouse and each key do to the view, measured by trying them.

    ``most_travel`` only stops a mouse that never moves the view from being
    doubled for ever; a world where nothing a mouse does shows is a world
    this learns nothing about the mouse in, and says so by a gain of nought.
    """
    body = WhatMyHandsDoToTheView()
    travel = 1
    while travel <= most_travel:
        before, _ = world.look()
        await world.play(Chunk((Slot(moved=(travel, 0)),), slot_s))
        after, _ = world.look()
        change = body.watched("mouse", before, after, mouse=(travel, 0))
        # Back where she was, so the next thing tried starts from the same view.
        await world.play(Chunk((Slot(moved=(-travel, 0)),), slot_s))
        if abs(change.across) >= 2.0:
            break
        travel *= 2
    for key in keys:
        before, _ = world.look()
        await world.play(Chunk((Slot(frozenset({key})),), slot_s))
        after, _ = world.look()
        body.watched(key, before, after)
    return body


async def go_to(
    world: ACameraWorld,
    named: str,
    body: WhatMyHandsDoToTheView,
    *,
    slot_s: float,
    most_chunks: int,
    tell: Callable[[str], None] | None = None,
) -> Trip:
    """One trip to the thing named, told as it goes."""
    trip = Trip(named)
    walks = body.walks_forward()
    if not walks:
        trip.ended = "nothing she pressed moved her forward, so there is no walking here yet"
        trip.said.append(trip.ended)
        return trip
    frame, layout = world.look()
    small_wide = grey(frame).shape[1]
    going = GoingTo(named, turn_for=lambda share: body.turn_for(share * small_wide), walks=walks)
    last_said = ""
    for _ in range(most_chunks):
        chunk = going.next_chunk(layout, slot_s=slot_s, slots=1)
        said = _what_this_does(chunk, named, going)
        if said and said != last_said:
            trip.said.append(said)
            if tell is not None:
                tell(said)
            last_said = said
        if chunk.slots:
            await world.play(chunk)
            trip.chunks += 1
        if chunk.done or chunk.think:
            trip.done = chunk.done
            break
        frame, layout = world.look()
    trip.ended = going.ended or "the time for this trip ran out"
    if trip.ended not in trip.said:
        trip.said.append(trip.ended)
    return trip


def _what_this_does(chunk: Chunk, named: str, going: GoingTo) -> str:
    """A sentence about a chunk, from what the chunk is."""
    if chunk.done or chunk.think:
        return going.ended
    moved = sum(slot.moved[0] for slot in chunk.slots)
    if moved:
        return f"turning {'right' if moved > 0 else 'left'} toward the {named}"
    if chunk.slots and all(going.walks in slot.held for slot in chunk.slots):
        return f"walking toward the {named}"
    return ""


def live_camera_world(app: str, *, still_ours: Callable[[], bool] | None = None) -> ACameraWorld | None:
    """The real thing: an application's window, her eyes on it and her hands on the machine."""
    from core.capabilities import window_server
    from core.capabilities.hands import QuartzHands, play
    from core.perception.what_the_pixels_show import recognize_text

    window = window_server.window_of(app, on_screen_only=True)
    if window is None:
        return None
    hands = QuartzHands()
    ours = still_ours or (lambda: window_server.owns_the_front(app))

    def look() -> tuple[Any, list[dict[str, Any]]]:
        frame = window_server.capture(window)
        return frame, (recognize_text(frame) if frame is not None else [])

    async def played(chunk: Chunk) -> Any:
        return await play(chunk, sink=hands, window=tuple(window.bounds), still_ours=ours)

    return ACameraWorld(look=look, play=played)


def a_look_takes(world: ACameraWorld, times: int = 3) -> float:
    """How long one look takes here, measured: the natural length of a slot."""
    began = time.monotonic()
    for _ in range(max(1, times)):
        world.look()
    return (time.monotonic() - began) / max(1, times)


def what_answers_to(world: ACameraWorld, named: str) -> list[str]:
    """What on screen answers to a name right now, for asking the person which."""
    _frame, layout = world.look()
    return [sight.text for sight in seen_named(layout, named)]

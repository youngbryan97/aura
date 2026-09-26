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

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.agency.going_to_what_she_sees import GoingTo, seen_named
from core.agency.what_hands_do import Chunk, Slot
from core.perception.how_the_view_moves import (
    ViewChange,
    WhatMyHandsDoToTheView,
    grey,
    how_alike,
    how_it_moved,
)

__all__ = ["ACameraWorld", "NotInFront", "Trip", "go_to", "learn_the_body", "live_camera_world"]


class NotInFront(RuntimeError):  # noqa: N818 - named for what happened
    """A chunk her hands refused, because the window she meant was not the one in front."""


@dataclass
class ACameraWorld:
    """What the loop needs from the machine."""

    #: The window's pixels and the words on it, with their boxes.
    look: Callable[[], tuple[Any, list[dict[str, Any]]]]
    #: Play a chunk; returns once it has been played, saying why it stopped if it did.
    play: Callable[[Chunk], Awaitable[Any]]
    #: Bring the window forward before acting in it. True when it is in front.
    bring_forward: Callable[[], Awaitable[bool]] | None = None


async def _played(world: ACameraWorld, chunk: Chunk) -> Any:
    """Play a chunk, and refuse to carry on as though a refused chunk had been played.

    A chunk her hands would not play because another window was in front did
    nothing, and reading that as "this key does nothing" teaches her a body
    she does not have. Live, the first time: the chat window came in front,
    every test move was refused, and she concluded no key walked here.
    """
    result = await world.play(chunk)
    stopped = str(getattr(result, "stopped", "") or "")
    if stopped:
        raise NotInFront(stopped)
    return result

@dataclass
class Trip:
    """One trip, as it went."""

    named: str
    said: list[str] = field(default_factory=list)
    chunks: int = 0
    ended: str = ""
    done: bool = False
    #: Whether the world answered what she did at the end, judged by what it
    #: showed afterwards; empty where nothing was there to answer.
    answered: str = ""
    #: Every chunk she played, as written, in order.
    played: list[str] = field(default_factory=list)
    #: The small grey view she had before each of them.
    looks: list[Any] = field(default_factory=list)


def look_settled(world: ACameraWorld, *, most: int = 3) -> tuple[Any, list[dict[str, Any]]]:
    """A look taken once the view has stopped changing, or the last of ``most`` looks.

    A world draws an act's effect when it gets round to it. On a loaded
    machine the look right after a key came up could still be the picture
    from before it, and an act whose effect had not been drawn yet read as an
    act that did nothing: live, on a machine running four other jobs, a key
    that walked was measured as one that did not. Two looks alike in a row
    means the drawing has caught up; a world that never stops moving is taken
    at its last look.
    """
    frame, layout = world.look()
    for _ in range(max(0, most - 1)):
        again, layout_again = world.look()
        if frame is not None and again is not None and np.array_equal(np.asarray(frame), np.asarray(again)):
            return again, layout_again
        frame, layout = again, layout_again
    return frame, layout


async def learn_the_body(
    world: ACameraWorld, *, keys: Sequence[str], slot_s: float, most_travel: int = 4096
) -> WhatMyHandsDoToTheView:
    """What the mouse and each key do to the view, measured by trying them.

    ``most_travel`` only stops a mouse that never moves the view from being
    doubled for ever; a world where nothing a mouse does shows is a world
    this learns nothing about the mouse in, and says so by a gain of nought.
    """
    body = WhatMyHandsDoToTheView()
    if world.bring_forward is not None and not await world.bring_forward():
        raise NotInFront("the window she was asked to act in would not come to the front")
    travel = 1
    while travel <= most_travel:
        before, _ = world.look()
        await _played(world, Chunk((Slot(moved=(travel, 0)),), slot_s))
        after, _ = look_settled(world)
        change = body.watched("mouse", before, after, mouse=(travel, 0))
        # Back where she was, so the next thing tried starts from the same view.
        await _played(world, Chunk((Slot(moved=(-travel, 0)),), slot_s))
        if abs(change.across) >= 2.0:
            break
        travel *= 2
    for key in keys:
        before, _ = world.look()
        await _played(world, Chunk((Slot(frozenset({key})),), slot_s))
        after, _ = look_settled(world)
        body.watched(key, before, after)
    if not body.walks_forward():
        # Up against something, walking forward changes nothing, and a key
        # that walks looks like a key that does nothing. Live, standing at the
        # door she had just opened, she concluded no key walked. A key that
        # shrank the view backed her away; from there, with room ahead, the
        # others are tried again.
        backs = min(
            (act for act in body.seen if act != "mouse"),
            key=lambda act: (body.what_it_does(act) or ViewChange(0, 0, 1.0, 0)).grew,
            default="",
        )
        backed = body.what_it_does(backs) if backs else None
        if backed is not None and backed.grew < 1.0:
            await _played(world, Chunk((Slot(frozenset({backs})),) * 2, slot_s))
            retry = [key for key in keys if key != backs]
        else:
            # Nothing moved her either way: facing something she cannot walk
            # into or back out of. Half a view round, and try again.
            frame, _ = world.look()
            travel = body.turn_for(0.5 * grey(frame).shape[1])
            if travel:
                await _played(world, Chunk((Slot(moved=(travel, 0)),), slot_s))
            retry = list(keys)
        for key in retry:
            before, _ = world.look()
            await _played(world, Chunk((Slot(frozenset({key})),), slot_s))
            after, _ = look_settled(world)
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
    leads: bool = False,
) -> Trip:
    """One trip to the thing named, told as it goes. ``leads`` aims where a moving thing will be."""
    trip = Trip(named)
    walks = body.walks_forward()
    if not walks:
        trip.ended = "nothing she pressed moved her forward, so there is no walking here yet"
        trip.said.append(trip.ended)
        return trip
    frame, layout = world.look()
    small_wide = grey(frame).shape[1]
    by_look = _what_it_looks_like(named)

    def with_what_it_looks_like(frame: Any, layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Where no words on screen name it, a thing called by its colour is
        # looked for by that colour.
        if by_look is None or frame is None or seen_named(layout, named):
            return layout
        found = by_look(frame)
        return [*layout, found] if found else layout

    layout = with_what_it_looks_like(frame, layout)
    going = GoingTo(named, turn_for=lambda share: body.turn_for(share * small_wide), walks=walks, leads=leads)
    last_said = ""
    sweep: list[Any] = []
    walked_since_press = True
    for _ in range(most_chunks):
        chunk = going.next_chunk(layout, slot_s=slot_s, slots=1)
        looking = False
        if chunk.think and not seen_named(layout, named):
            # Not in view: look around for it, the way a person turns on the
            # spot. Half a view a step, so every step overlaps the last and
            # nothing narrower than half the view can fall between two looks.
            sweep.append(frame)
            if _back_where_she_started(sweep):
                going.ended = f"looked all the way round and nothing answers to {named!r}"
                chunk = Chunk((), slot_s, think=True)
            else:
                # Toward the side it was last seen on, when it was seen at
                # all: a thing that walked out of view is most likely still
                # going that way.
                side = -1.0 if (going._last is not None and going._last < 0.5) else 1.0
                chunk = Chunk((Slot(moved=(body.turn_for(-0.5 * side * small_wide), 0)),), slot_s)
                going.ended = ""
                looking = True
        said = f"looking around for the {named}" if looking else _what_this_does(chunk, named, going)
        if said and said != last_said:
            trip.said.append(said)
            if tell is not None:
                tell(said)
            last_said = said
        before = frame
        if chunk.slots:
            try:
                await _played(world, chunk)
            except NotInFront as why:
                trip.ended = f"stopped: {why}"
                trip.said.append(trip.ended)
                return trip
            trip.chunks += 1
            trip.played.append(chunk.as_text())
            if before is not None:
                small = grey(before)
                trip.looks.append(np.clip(small - small.min(), 0, 255).astype(np.uint8))
        if chunk.done or chunk.think:
            trip.done = chunk.done
            if chunk.done and chunk.slots:
                frame, after = look_settled(world)
                trip.answered = _how_the_world_answered(layout, after, named)
                if trip.answered:
                    trip.said.append(trip.answered)
                    if tell is not None:
                        tell(trip.answered)
                # Pressed and nothing answered, with the thing still there: a
                # thing that moves can be out of reach by the time a key lands.
                # She goes on and presses again, within the trip she was given.
                # Once: a second press nothing answered, with no walking since
                # the first, would be the same press again, and a locked door
                # stays locked however often she tries it.
                if trip.answered == UNANSWERED and seen_named(after, named) and walked_since_press:
                    layout, going.ended, trip.done = after, "", False
                    walked_since_press = False
                    continue
            break
        frame, layout = look_settled(world)
        layout = with_what_it_looks_like(frame, layout)
        if before is not None and frame is not None:
            across, _down, grew, _sure = how_it_moved(grey(before), grey(frame))
            going.slid(across / small_wide)
            if chunk.slots and all(walks in slot.held for slot in chunk.slots):
                going.walked(grew, chunk.slot_s * len(chunk.slots))
                walked_since_press = True
    trip.ended = going.ended or "the time for this trip ran out"
    if trip.ended not in trip.said:
        trip.said.append(trip.ended)
    return trip


def _what_it_looks_like(named: str) -> Callable[[Any], dict[str, Any] | None] | None:
    """A way to find ``named`` by how it looks, where its name says how: a colour it is called by."""
    from core.perception.finding_it_by_look import HUES, of_colour

    colours = [word for word in named.lower().split() if word in HUES]
    if not colours:
        return None

    def finds(frame: Any) -> dict[str, Any] | None:
        pixels = np.asarray(frame)
        if pixels.ndim != 3:
            return None
        return of_colour(pixels, colours[0], name=named)

    return finds


def _how_the_world_answered(
    before: list[dict[str, Any]], after: list[dict[str, Any]], named: str = ""
) -> str:
    """What the screen says to the last thing she did: words that went, and words that came.

    Pressing what a prompt names is not success; the world answering is. A
    prompt that is still there after the press, with nothing new beside it,
    is a press nothing heard, and she says so rather than calling it done.

    A prompt that only went away is an answer when the thing is still in
    front of her, no smaller: a prompt also goes when its thing moves out of
    reach, and chasing one, that read as the thing having been used.
    """
    was = {str(region.get("text", "")).strip() for region in before or ()}
    now = {str(region.get("text", "")).strip() for region in after or ()}
    came = sorted(text for text in now - was if text)
    went = sorted(text for text in was - now if text)
    if came:
        return f"the screen now says {came[0]!r}"
    if went:
        if named:
            then, still = seen_named(before, named), seen_named(after, named)
            if not still or (then and still[0].high < then[0].high):
                return UNANSWERED
        return f"{went[0]!r} went away"
    return UNANSWERED


#: What she says when she pressed what a prompt named and nothing answered.
UNANSWERED = "she pressed it and nothing on screen answered"


def _back_where_she_started(sweep: list[Any]) -> bool:
    """Whether the latest look of a sweep is the view it began with.

    Measured against the sweep itself rather than a threshold: back at the
    start means the latest view, lined up with the first, is more like it
    than halfway between "the same view" and how alike the first is to the
    views in between. The look right after the first overlaps it by half, so
    it is not one of the views in between.
    """
    if len(sweep) < 4:
        return False
    first = grey(sweep[0])
    between = [how_alike(first, grey(frame)) for frame in sweep[2:-1]]
    return how_alike(first, grey(sweep[-1])) > (1.0 + sum(between) / len(between)) / 2.0


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


def live_camera_world(
    app: str, *, still_ours: Callable[[], bool] | None = None, wait_s: float = 60.0
) -> ACameraWorld | None:
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

    async def bring_forward() -> bool:
        if ours():
            return True
        # Someone is at the keyboard. Taking the front from a person halfway
        # through typing sends her keys into their work and theirs into her
        # world; live, the first time, the front came straight back to the
        # window being typed in and every chunk was refused. She waits for a
        # pause as long as one of her looks, and gives up when the time she
        # was given for this is gone.
        waited_until = time.monotonic() + max(1.0, float(wait_s))
        pause = a_look_takes(ACameraWorld(look=look, play=played))
        while window_server.seconds_since_someone_touched_it() < pause:
            if time.monotonic() > waited_until:
                return False
            await asyncio.sleep(pause)
        from core.capabilities.host_automation import get_host_automation

        await get_host_automation().focus_app(app)
        return ours()

    return ACameraWorld(look=look, play=played, bring_forward=bring_forward)


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


async def a_trip_for_the_pursuit(
    named: str,
    app: str,
    move_keys: Sequence[str],
    *,
    tell: Callable[[str], None] | None,
    within_s: float,
) -> dict[str, Any]:
    """A trip asked for in a request, run live and answered the way a pursuit answers.

    The keys tried for walking are the request's own, then w and s, which is
    what a person tries first in a game they have not played.
    """
    world = live_camera_world(app, wait_s=within_s)
    if world is None:
        return {"ok": False, "completed": False, "outcome": "cannot_see",
                "cannot_see": f"there is no {app} window on screen", "moves": [], "attempts": []}
    began = time.monotonic()
    slot = a_look_takes(world)
    keys = tuple(dict.fromkeys([*move_keys, "w", "s"]))
    try:
        body = await learn_the_body(world, keys=keys, slot_s=slot)
        trip = await go_to(
            world, named, body, slot_s=slot,
            most_chunks=max(1, int((within_s - (time.monotonic() - began)) / max(slot, 1e-3))),
            tell=tell,
        )
    except NotInFront as why:
        return {"ok": False, "completed": False, "outcome": "navigated_away",
                "needs_person": str(why), "moves": [], "attempts": []}
    heard = trip.done and trip.answered != "she pressed it and nothing on screen answered"
    from core.agency.what_she_tried import ASKED, Episode, keep

    await asyncio.to_thread(
        keep,
        Episode(
            world=app, task=f"go to the {named}", set_by=ASKED, succeeded=heard,
            ended=trip.ended, answered=trip.answered, played=list(trip.played),
            skill="going to a thing",
        ),
        list(trip.looks),
    )
    return {
        "ok": heard,
        "completed": heard,
        "outcome": "reached" if heard else "no_move_available",
        "cannot_decide": "" if heard else (trip.answered or trip.ended),
        "moves": trip.said,
        "attempts": [],
        "said": trip.said,
        "trip": {"to": named, "ended": trip.ended, "chunks": trip.chunks},
    }


async def what_is_around(
    world: ACameraWorld,
    body: WhatMyHandsDoToTheView,
    *,
    slot_s: float,
    most_looks: int,
) -> list[tuple[str, float]]:
    """Everything she can read by turning on the spot once, with the way she turned to see it.

    Answering a question about a world by going and looking is one of the
    things SIMA 2 shows (a scan it walked to and read, arXiv 2512.04797,
    Fig. 1). The nearest version of that needs no walking: turn half a view
    at a time until the view is the one she began with, and keep every name
    she read and how far round she had turned when she read it, in shares of
    a view. She ends facing the way she started.
    """
    from core.agency.going_to_what_she_sees import reads_as_a_thing

    seen: dict[str, float] = {}
    sweep: list[Any] = []
    frame, layout = look_settled(world)
    small_wide = grey(frame).shape[1] if frame is not None else 0
    turned = 0.0
    for _ in range(max(1, most_looks)):
        for region in layout or ():
            if reads_as_a_thing(region):
                name = " ".join(str(region.get("text", "")).split())
                seen.setdefault(name, turned + float(region.get("center_x", 0.5)) - 0.5)
        sweep.append(frame)
        if _back_where_she_started(sweep) or not small_wide:
            break
        travel = body.turn_for(-0.5 * small_wide)
        if not travel:
            break
        await _played(world, Chunk((Slot(moved=(travel, 0)),), slot_s))
        turned += 0.5
        frame, layout = look_settled(world)
    return sorted(seen.items(), key=lambda item: item[1])


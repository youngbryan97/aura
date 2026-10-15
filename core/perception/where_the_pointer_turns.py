"""Where the pointer turns into a hand: the places on a screen that answer to a click, found as a person finds them,
by moving the mouse over it.

Much of what can be clicked on a game's screen looks like nothing in particular: the doorway to the next room, the
drawer with something in it, the edge of a scene that leads on, a word drawn into the picture. A person playing a
point-and-click game moves the mouse about and watches the pointer. Where it turns into a hand, something answers.
LIVE-like 2026-10-11 a search through a house's rooms was clicked at shapes picked out by colour for two minutes,
and the way to the next room was found only at the end.

So, once a screen, the pointer is swept over the part of the page that draws, on a grid, and where it is shown as a
hand is noted. Neighbouring places where it is a hand are one place, and the middle of each is offered as a thing to
click that stands out. A screen that is a hand everywhere (a screen clicked anywhere to go on) or nowhere says
nothing about where.

What the pointer is shown as is read from the page, as a person reads it off the screen. Nothing here knows a game.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import math
import weakref
from typing import Any

from core.verify import invariant

__all__ = ["places_the_pointer_turns", "shown_as_a_hand", "the_thing_there", "where_the_hand_is_now", "where_the_pointer_turns"]

logger = logging.getLogger("Aura.WhereThePointerTurns")

# A sweep can exceed its look while the browser delivers physical moves.
# Keep only finished physical queries when its time expires.
SWEEP_WITHIN_S = 2.0
_CURSOR_SAMPLES: contextvars.ContextVar[list | None] = contextvars.ContextVar('physical_cursor_samples', default=None)


def _sampled(x: float, y: float, shown: str) -> None:
    samples = _CURSOR_SAMPLES.get()
    if samples is not None and len(samples) < 1024:
        samples.append((x, y, shown))


def _spread(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    remaining, selected = list(dict.fromkeys(points)), []
    while remaining and len(selected) < MOST:
        point = max(remaining, key=lambda p: min(((p[0]-x)**2+(p[1]-y)**2 for x, y in selected), default=0.))
        selected.append(point)
        remaining.remove(point)
    return selected


def _geometry(page: Any, over: Any) -> tuple | None:
    size = getattr(page, 'viewport_size', None) or {}
    try:
        width, height = float(size.get('width') or 0), float(size.get('height') or 0)
        left, top, right, bottom = over or (0., 0., 1., 1.)
        if (not all(type(v) in (int, float) and math.isfinite(v) for v in (width, height, left, top, right, bottom))
                or width <= 0 or height <= 0 or not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1):
            return None
        return width, height, left, top, right, bottom
    except (TypeError, ValueError, AttributeError):
        return None


def _partial_places(samples: list, geometry: tuple) -> list[tuple[float, float]]:
    width, height, left, top, right, bottom = geometry
    local = [((x/width-left)/(right-left), (y/height-top)/(bottom-top), cursor) for x, y, cursor in samples
             if left <= x/width <= right and top <= y/height <= bottom]
    return _spread([(x, y) for x, y, cursor in local if cursor in _A_HAND]) if any(
        cursor not in _A_HAND for _x, _y, cursor in local) else []


@invariant('perception.interrupted_cursor_sweeps_keep_only_physical_evidence', scope='perception',
           owner='core/perception/where_the_pointer_turns.py', observational=False)
def _partial_cursor_invariant() -> tuple:
    geometry = (100., 100., .2, .2, .8, .8)
    points = _partial_places([(50, 50, 'pointer'), (30, 30, 'auto')], geometry)
    assert len(points) == 1 and all(abs(v-.5) < 1e-12 for v in points[0])
    assert _partial_places([(50, 50, 'pointer')], geometry) == []
    assert _partial_places([(50, 50, 'pointer'), (0, 0, 'auto')], geometry) == []
    return ()


def corners_to_query(seen: Any) -> tuple[tuple[float, float], ...]:
    """Current image corners propose pointer queries, without click authority."""
    from core.perception.observed_transfer import _pixels, freeze_screen
    from core.perception.points_that_keep_their_motion import _grey, _new_points

    frame = freeze_screen(seen)
    picture = _pixels(frame.picture) if frame.picture is not None else None
    if picture is None or min(picture.shape[:2]) < 16:
        return ()
    height, width = picture.shape[:2]
    return tuple(((x+.5)/width, (y+.5)/height) for x, y in _new_points(_grey(picture), [], 32))


async def _local_hints(page: Any, hints: tuple, over: tuple, size: tuple) -> list[tuple[float, float]]:
    """A sparse proposal needs both current positive and negative cursor evidence."""
    if (not hints or len(hints) > 32 or any(len(p) != 2 or any(type(v) not in (int, float)
            or not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in hints)):
        return []
    left, top, right, bottom = over
    wide, tall = size
    controls = tuple((x, y) for x in (.125, .375, .625, .875) for y in (.2, .8))
    points = (*hints, *controls)
    shown = await _shown_at_each(page, [((left+(right-left)*x)*wide, (top+(bottom-top)*y)*tall) for x, y in points])
    if len(shown) != len(points) or not any(cursor not in _A_HAND for cursor in shown[len(hints):]):
        return []
    positive = list(dict.fromkeys(p for p, cursor in zip(hints, shown[:len(hints)], strict=True) if cursor in _A_HAND))
    # Keep a positive sample, then spread the finite returned set. Averaging
    # points can put a proposed click in a hole between real hit regions.
    return _spread(positive)


@invariant('perception.picture_corners_propose_cursor_queries', scope='perception',
           owner='core/perception/where_the_pointer_turns.py', observational=False)
def _corner_queries_invariant() -> tuple:
    import numpy as np

    from core.perception.observed_transfer import ScreenReading, make_snapshot

    picture = np.zeros((80, 100, 3), dtype=np.uint8)
    picture[20:45, 30:65] = 240
    hints = corners_to_query(ScreenReading({}, picture_snapshot=make_snapshot(picture)))
    assert 0 < len(hints) <= 32 and all(0 < x < 1 and 0 < y < 1 for x, y in hints)
    assert corners_to_query({}) == ()
    return ()

#: The grid the pointer is swept over, across and down.
ACROSS, DOWN = 12, 9
#: The most places kept for one screen.
MOST = 8
#: How the pointer is shown over a thing that answers to a click.
_A_HAND = frozenset({"pointer", "hand"})

#: What the pointer is shown as at a place of the page, looking into the player the game is drawn in.
_SHOWN_AS = """(([x, y]) => {
  let e = document.elementFromPoint(x, y), shown = '';
  for (let i = 0; e && i < 6; i++) {
    shown = getComputedStyle(e).cursor;
    const inner = e.shadowRoot && e.shadowRoot.elementFromPoint(x, y);
    if (!inner || inner === e) break;
    e = inner;
  }
  return shown;
})"""


def the_thing_there(place: tuple[float, float], regions: list[dict[str, Any]], across: int = ACROSS,
                    down: int = DOWN) -> dict[str, Any] | None:
    """The thing already read that a place where the pointer turned stands on, if any: the nearest whose outline, or
    middle where it has no outline, lies within one cell of the swept grid. A place found on a grid is known no more
    closely than a cell of it: offline 2026-10-12 a carved "back" was found half a cell beside its word, taken for an
    unnamed thing that stands out, and clicked as a way on."""
    x, y = place
    best, nearest, ambiguous = None, None, False
    for region in regions:
        try:
            if region.get("width") is not None and region.get("x") is not None:
                left, top = float(region["x"]), float(region["y"])
                right, bottom = left + float(region["width"]), top + float(region["height"])
            else:
                left = right = float(region["center_x"])
                top = bottom = float(region["center_y"])
        except (KeyError, TypeError, ValueError):
            continue
        off_x = max(left - x, 0.0, x - right) * across
        off_y = max(top - y, 0.0, y - bottom) * down
        if off_x <= 1.0 and off_y <= 1.0:
            distance = off_x + off_y
            if nearest is None or distance < nearest - 1e-9:
                best, nearest, ambiguous = region, distance, False
            elif abs(distance - nearest) <= 1e-9:
                ambiguous = True
    return None if ambiguous else best


#: The pointer moved over each of a list of places inside the page, as the page's own pointer moves, and what it is
#: shown as at each: the whole sweep in one asking of the page.
_SWEPT_IN_THE_PAGE = """(([points]) => {
  const at = (x, y) => {
    let e = document.elementFromPoint(x, y);
    for (let i = 0; e && i < 6; i++) {
      const inner = e.shadowRoot && e.shadowRoot.elementFromPoint(x, y);
      if (!inner || inner === e) break;
      e = inner;
    }
    return e;
  };
  const shown = [];
  for (const [x, y] of points) {
    const e = at(x, y);
    if (e) {
      const how = {clientX: x, clientY: y, bubbles: true, composed: true, pointerId: 1, pointerType: 'mouse', isPrimary: true};
      e.dispatchEvent(new PointerEvent('pointermove', how));
      e.dispatchEvent(new MouseEvent('mousemove', how));
    }
    shown.push(""" + _SHOWN_AS + """([x, y]));
  }
  return shown;
})"""


async def _shown_at_each(page: Any, points: list[tuple[float, float]]) -> list[str]:
    """What the pointer is shown as at each of ``points`` (the page's pixels), the pointer taken over them in turn.

    Asked of the page in one go: the pointer's moves are handed to what is drawn under each place as the page hands it
    the mouse's, and what the pointer is shown as is read after each. One move and one question to the browser a place
    took most of a second for a screen (108 places), and was most of the time of a look at a screen that had something
    to find under the pointer; in one go it takes a few thousandths. Offline 2026-10-12 the two found the same places
    on sixteen screens of eight games. Where the page cannot be asked so, the mouse is moved over each place as before.
    The mouse is left where the last move left the pointer, as it was."""
    if not points:
        return []
    if page not in _NEEDS_THE_MOUSE:
        try:
            shown = await page.evaluate(_SWEPT_IN_THE_PAGE, [[list(point) for point in points]])
            if isinstance(shown, list) and len(shown) == len(points):
                shown = [str(one or "") for one in shown]
                if await _the_mouse_agrees(page, points, shown):
                    await page.mouse.move(*points[-1])
                    return shown
                # A page whose drawing answers only the mouse itself: swept with it, now and from now on.
                _NEEDS_THE_MOUSE.add(page)
                logger.info("the pointer swept inside the page did not show what the mouse shows: swept with the mouse")
        except Exception as why:  # noqa: BLE001 - a page that will not be asked in one go is asked place by place
            logger.debug("the pointer could not be swept inside the page: %s", str(why)[:120])
    out = []
    for x, y in points:
        await page.mouse.move(x, y)
        shown = str(await page.evaluate(_SHOWN_AS, [x, y]) or "")
        _sampled(x, y, shown)
        out.append(shown)
    return out


#: Pages where the pointer's moves handed inside the page were seen not to do what the mouse does.
_NEEDS_THE_MOUSE: weakref.WeakSet[Any] = weakref.WeakSet()
#: How many places, besides every place found to be a hand, the mouse itself is taken to after a sweep inside the page.
CHECKED_BY_THE_MOUSE = 8


async def _the_mouse_agrees(page: Any, points: list[tuple[float, float]], shown: list[str]) -> bool:
    """Whether the mouse itself shows what the sweep inside the page showed, at every place it found to be a hand and
    at places spread over the rest. A drawing may answer only moves the browser makes from a real mouse (a report on
    a test page, 2026-10-13); on such a page the two part, and the sweep is not trusted."""
    hands = [i for i, one in enumerate(shown) if one in _A_HAND][:CHECKED_BY_THE_MOUSE]
    rest = [i for i, one in enumerate(shown) if one not in _A_HAND]
    spread = rest[:: max(1, len(rest) // CHECKED_BY_THE_MOUSE)][:CHECKED_BY_THE_MOUSE] if rest else []
    for i in [*hands, *spread]:
        x, y = points[i]
        await page.mouse.move(x, y)
        physical = str(await page.evaluate(_SHOWN_AS, [x, y]) or "")
        _sampled(x, y, physical)
        if (physical in _A_HAND) != (shown[i] in _A_HAND):
            return False
    return True


def _groups(hands: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """The cells where the pointer was a hand, neighbours taken together."""
    left, groups = set(hands), []
    while left:
        start = left.pop()
        group, frontier = [start], [start]
        while frontier:
            column, row = frontier.pop()
            for near in ((column + 1, row), (column - 1, row), (column, row + 1), (column, row - 1)):
                if near in left:
                    left.discard(near)
                    group.append(near)
                    frontier.append(near)
        groups.append(group)
    return groups


def places_the_pointer_turns(hands: set[tuple[int, int]], across: int = ACROSS, down: int = DOWN) -> list[tuple[float, float]]:
    """The places of a swept grid where the pointer was a hand, neighbours taken together, as the middle of each in
    shares of what was swept; none where it was a hand nearly everywhere or nowhere."""
    if not hands or len(hands) >= 0.8 * across * down:
        return []
    left, places = set(hands), []
    while left:
        start = left.pop()
        group, frontier = [start], [start]
        while frontier:
            column, row = frontier.pop()
            for near in ((column + 1, row), (column - 1, row), (column, row + 1), (column, row - 1)):
                if near in left:
                    left.discard(near)
                    group.append(near)
                    frontier.append(near)
        places.append(((sum(c for c, _ in group) / len(group) + 0.5) / across,
                       (sum(r for _, r in group) / len(group) + 0.5) / down, len(group)))
    places.sort(key=lambda place: -place[2])
    return [(x, y) for x, y, _size in places[:MOST]]


async def where_the_pointer_turns(page: Any, over: tuple[float, float, float, float] | None = None, *,
                                hints: tuple[tuple[float, float], ...] = ()) -> list[tuple[float, float]]:
    """Bound pointer inspection; an interrupted pass retains only physical samples.

    A partial result needs a positive and a negative in the requested region.
    It proposes actual hit points and cannot establish a complete control list.
    External cancellation still cancels the look.
    """
    geometry, address = _geometry(page, over), getattr(page, 'url', None)
    if geometry is None:
        return []
    samples: list = []
    token = _CURSOR_SAMPLES.set(samples)
    try:
        try:
            return await asyncio.wait_for(_sweep(page, over, hints=hints), SWEEP_WITHIN_S)
        except TimeoutError:
            if _geometry(page, over) != geometry or getattr(page, 'url', None) != address:
                return []
            points = _partial_places(samples, geometry)
            logger.info('cursor inspection reached %.1fs: %d physical samples, %d local proposals',
                        SWEEP_WITHIN_S, len(samples), len(points))
            return points
    finally:
        _CURSOR_SAMPLES.reset(token)


async def _sweep(page: Any, over: tuple[float, float, float, float] | None = None, *,
                 hints: tuple[tuple[float, float], ...] = ()) -> list[tuple[float, float]]:
    """The places in ``over`` (shares of the page's view, left, top, right, bottom) where the pointer turns into a
    hand, as shares of ``over``; none where the page cannot be asked."""
    size = getattr(page, "viewport_size", None) or {}
    width, height = float(size.get("width") or 0), float(size.get("height") or 0)
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        return []
    left, top, right, bottom = over or (0.0, 0.0, 1.0, 1.0)
    if not all(math.isfinite(value) for value in (left, top, right, bottom)) or not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        return []
    if hints:
        try:
            found = await _local_hints(page, hints, (left, top, right, bottom), (width, height))
            if found:
                logger.info('where current picture corners answer the pointer: %s', [(round(x, 3), round(y, 3)) for x, y in found])
                return found
        except Exception as why:  # noqa: BLE001 - an unqualified sparse probe keeps the complete existing sweep
            logger.info('picture-corner cursor queries did not qualify: %s', str(why)[:120])
    cells = [(column, row) for row in range(DOWN) for column in range(ACROSS)]
    try:
        shown = await _shown_at_each(page, [((left + (right - left) * (column + 0.5) / ACROSS) * width,
                                             (top + (bottom - top) * (row + 0.5) / DOWN) * height)
                                            for column, row in cells])
        hands = {cell for cell, one in zip(cells, shown, strict=True) if one in _A_HAND}
    except Exception as why:  # noqa: BLE001 - a page that cannot be asked is a screen with no such places found
        logger.info("the pointer could not be swept over the page: %s", str(why)[:120])
        return []
    places = places_the_pointer_turns(hands)
    if len(hands) >= .8 * ACROSS * DOWN:
        return []
    if places and len(hands) < 0.8 * ACROSS * DOWN:
        places = await _split_finer(page, hands, (left, top, right, bottom), (width, height)) or places
    # Cell centres leave an unprobed strip along every boundary. Probe its
    # half-cell centres too, retaining actual positive samples as witnesses.
    border = [((c+.5)/ACROSS, y) for c in range(ACROSS) for y in (.25/DOWN, 1-.25/DOWN)]
    border += [(x, (r+.5)/DOWN) for r in range(DOWN) for x in (.25/ACROSS, 1-.25/ACROSS)]
    try:
        shown = await _shown_at_each(page, [((left+(right-left)*x)*width, (top+(bottom-top)*y)*height) for x, y in border])
        for point, cursor in zip(border, shown, strict=True):
            if len(places) >= MOST*3:
                break
            if cursor in _A_HAND and not any(abs(point[0]-x) <= .5/ACROSS and abs(point[1]-y) <= .5/DOWN for x, y in places):
                places.append(point)
    except Exception as why:  # noqa: BLE001 - retained interior samples remain measurements
        logger.info('the pointer boundary could not be swept: %s', str(why)[:120])
    logger.info("where the pointer turns into a hand: %s", [(round(x, 2), round(y, 2)) for x, y in places])
    return places


#: How much finer a place the pointer turned over is swept again, and the least cells a place covers to be swept again;
#: and the longest the finer sweeps of one screen may take, all together, in seconds: a sweep is inside a reading of
#: the screen, and a reading has seconds of its own (offline 2026-10-11 a screen of hidden things, a hand nearly all
#: over, was swept finer for longer than its reading was allowed three times running, and the game was left).
FINER = 3
SPLIT_FROM_CELLS = 3
FINER_AT_MOST_S = 1.5


async def _split_finer(page: Any, hands: set[tuple[int, int]], over: tuple[float, float, float, float],
                       size: tuple[float, float]) -> list[tuple[float, float]]:
    """The places again, where a place covers several cells of the sweep, swept again finer over that place, so that
    things side by side with a little between them are told apart: offline 2026-10-11 a corridor of doors, a row of six
    at a time, was found as one place a row, and the middle of each row was clicked over and over. [] where nothing
    is split."""
    import time

    left, top, right, bottom = over
    width, height = size
    split, places, began = False, [], time.monotonic()
    for group in sorted(_groups(hands), key=len, reverse=True)[:MOST]:
        if len(group) < SPLIT_FROM_CELLS or time.monotonic() - began >= FINER_AT_MOST_S:
            places.append(((sum(c for c, _ in group) / len(group) + 0.5) / ACROSS, (sum(r for _, r in group) / len(group) + 0.5) / DOWN))
            continue
        columns = [c for c, _ in group]
        rows = [r for _, r in group]
        c0, c1, r0, r1 = min(columns), max(columns) + 1, min(rows), max(rows) + 1
        wide, high = (c1 - c0) * FINER, (r1 - r0) * FINER
        cells = [(column, row) for row in range(high) for column in range(wide)]
        try:
            shown = await _shown_at_each(page, [((left + (right - left) * (c0 + (column + 0.5) / FINER) / ACROSS) * width,
                                                 (top + (bottom - top) * (r0 + (row + 0.5) / FINER) / DOWN) * height)
                                                for column, row in cells])
            finer = {cell for cell, one in zip(cells, shown, strict=True) if one in _A_HAND}
        except Exception as why:  # noqa: BLE001 - a place that cannot be swept again is kept as it was
            logger.info("the pointer could not be swept finer: %s", str(why)[:120])
            return []
        parts = [part for part in _groups(finer) if len(part) >= 2]
        split = split or len(parts) > 1
        for part in parts or [[(column, row) for column in range(wide) for row in range(high)]]:
            places.append(((c0 + (sum(c for c, _ in part) / len(part) + 0.5) / FINER) / ACROSS,
                           (r0 + (sum(r for _, r in part) / len(part) + 0.5) / FINER) / DOWN))
    return places[:MOST * 3] if split else []


async def shown_as_a_hand(page: Any, x: float, y: float) -> bool:
    """Whether the pointer, moved to ``x``, ``y`` in the page's pixels, is shown as a hand there now."""
    await page.mouse.move(x, y)
    return str(await page.evaluate(_SHOWN_AS, [x, y]) or "") in _A_HAND


async def where_the_hand_is_now(page: Any, x: float, y: float, left: float, top: float, wide: float,
                                tall: float) -> tuple[float, float] | None:
    """Where a place found to answer the pointer answers now, as shares of the part swept: where it was, where the pointer
    is still a hand there; else the nearest place within a grid step or so where it is; else None. A button
    that moves (a top spinning about a title, a door that slides) is not where it was when the screen was swept: offline
    2026-10-11 a title's start was a spinning top, and every click went to where it had been."""
    try:
        if await shown_as_a_hand(page, left + x * wide, top + y * tall):
            return x, y
        found = []
        for row in range(-NEAR, NEAR + 1):
            for column in range(-NEAR, NEAR + 1):
                there = (x + column / (2 * ACROSS), y + row / (2 * DOWN))
                if not (0.0 <= there[0] <= 1.0 and 0.0 <= there[1] <= 1.0) or (row, column) == (0, 0):
                    continue
                if await shown_as_a_hand(page, left + there[0] * wide, top + there[1] * tall):
                    found.append(there)
    except Exception as why:  # noqa: BLE001 - no current hand witness authorizes this pointer-derived click
        logger.info("where the hand is now could not be asked: %s", str(why)[:120])
        return None
    if not found:
        return None
    near = min(found, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
    return near


#: How many half grid steps either way a place found to answer the pointer is looked for where it has gone.
NEAR = 3


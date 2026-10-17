"""The screen pursuit's eyes and hands on a page in her own browser.

The screen pursuit reaches the world through nine things: read the screen,
press a key, press several, click a point, and five questions about windows —
is hers in front, what page is it, what is in front, what is on top of it, and
bring it back. Every test of it replaces exactly those nine with a world of its
own. A page in her own browser is one more world, and this is those nine
answered by the page: a picture the browser takes of what it draws, keys and
clicks the browser delivers to the page itself, and the page's own address.

Nothing reaches the desktop while a page is the surface. The keys go to the
page whatever the person has in front of them, and no window is ever guessed
at: LIVE 2 October, a hand-over that took the front window to be hers would
have played into the person's own Chrome.

The surface is set for the length of one hand-over (``HER_OWN_PAGE``) and each
of the nine checks it first, so a pursuit over the desktop is exactly what it
was.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.perception.presses_on_the_same_material import MaterialPressHistory, bind_material_presses
from core.perception.recording_the_frames_and_inputs import (
    begin_navigation_input,
    finish_navigation_input,
)
from core.runtime.what_stops_it import current
from core.runtime.errors import record_degradation
from core.verify import invariant

__all__ = ["HER_OWN_PAGE", "OnAPage", "it_was_answered", "on_her_page"]

#: Where keys go when a page is the surface: the pursuit's names to the
#: browser's.
_KEY_NAMES = {
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "return": "Enter",
    "enter": "Enter",
    "tab": "Tab",
    "space": "Space",
    "escape": "Escape",
    "esc": "Escape",
    "backspace": "Backspace",
    "delete": "Delete",
    "shift": "Shift",
    "control": "Control",
    "ctrl": "Control",
    "alt": "Alt",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "pagedown": "PageDown",
}

#: Focus the thing the page draws, across shadow roots, so keys reach it. A
#: player in a shadow root takes the keyboard only once it is focused (measured
#: in the pane 2 October: after "Play Game" the focus stayed on the body).
_FOCUS_THE_DRAWING = """
(function () {
  var best = null, area = 0;
  var walk = function (root, host) {
    var all = root.querySelectorAll('*');
    for (var i = 0; i < all.length; i++) {
      var el = all[i];
      if (el.tagName === 'CANVAS' || el.tagName === 'EMBED' || el.tagName === 'OBJECT') {
        var r = el.getBoundingClientRect();
        if (r.width * r.height > area) { area = r.width * r.height; best = host || el; }
      }
      if (el.shadowRoot) walk(el.shadowRoot, host || el);
    }
  };
  walk(document, null);
  if (!best) return false;
  if (!best.hasAttribute('tabindex')) best.setAttribute('tabindex', '0');
  best.focus();
  return document.activeElement === best;
})()
"""


async def focus_drawing_for_input(page: Any) -> bool:
    """Measure the drawing's current keyboard target before a key is sent.

    A missing drawing leaves document-level keyboard handling available.
    The receipt describes focus only, separately from input transport and
    any effect the application may subsequently show.
    """
    from core.perception.recording_the_frames_and_inputs import record_keyboard_target

    began = time.monotonic()
    error = ''
    try:
        reply = await page.evaluate(_FOCUS_THE_DRAWING)
        status = 'focused' if reply is True else 'not_focused' if reply is False else 'unmeasured'
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as why:
        status, error = 'unavailable', type(why).__name__
    record_keyboard_target(page, status=status, began=began, ended=time.monotonic(), error=error)
    return status == 'focused'



#: How long the pointer rests over a thing before it is pressed, and how long the button is held, as a hand's are: a
#: button drawn by some engines answers only a pointer that came over it and a press that lasted. Offline 2026-10-12 a
#: title's NEW GAME, clicked in an instant from nowhere, never started its game; arrived at, rested on and pressed, it did.
HAND_ARRIVES_S = 0.12
HAND_PRESSES_S = 0.08


async def as_a_hand_clicks(page: Any, x: float, y: float, *, arrives_s: float = HAND_ARRIVES_S,
                           presses_s: float = HAND_PRESSES_S) -> None:
    """A click at page pixels (x, y) made as a hand makes one: the pointer comes over the place from just beside it,
    rests there a moment, and the button goes down for a moment and up. The moments are the browser's own, not this
    process's clock."""
    mouse = page.mouse
    if hasattr(mouse, "move"):
        await mouse.move(x - 6, y - 4)
        try:
            await mouse.move(x, y, steps=3)
        except TypeError:
            await mouse.move(x, y)
        rest = getattr(page, "wait_for_timeout", None)
        if arrives_s and callable(rest):
            await rest(int(arrives_s * 1000))
    try:
        await mouse.click(x, y, delay=int(presses_s * 1000))
    except TypeError:
        # Something that clicks only in an instant, clicks so.
        await mouse.click(x, y)


@dataclass(frozen=True)
class _CursorSweep:
    key: tuple
    places: tuple[tuple[float, float], ...]
    materials: tuple = ()


def _cursor_scope(page: Any, seen: dict, over: Any) -> tuple | None:
    """The screenshot and CSS coordinate frames belonging to this sweep."""
    from core.perception.observed_transfer import freeze_screen

    try:
        size = getattr(page, 'viewport_size', None) or {}
        if not isinstance(size, Mapping) or any(isinstance(value, bool)
                for value in (size.get('width'), size.get('height'), *(over or ()))):
            return None
        wide, tall = float(size['width']), float(size['height'])
        left, top, right, bottom = over or (0., 0., 1., 1.)
        if (not all(math.isfinite(v) for v in (wide, tall, left, top, right, bottom))
                or min(wide, tall) <= 0 or not (0 <= left < right <= 1 and 0 <= top < bottom <= 1)):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    scan = (left*wide, top*tall, (right-left)*wide, (bottom-top)*tall)
    frame = freeze_screen(seen)
    if 'bounds' in seen and frame.bounds is None:
        return None
    bounds = frame.bounds or scan
    picture = frame.picture
    digest = ((picture.height, picture.width, picture.channels, hashlib.sha256(picture.data).hexdigest())
              if picture and frame.bounds is not None else None)
    key = (id(page), str(getattr(page, 'url', '') or ''), frame.surface, scan, bounds, digest)
    return key, scan, bounds


@invariant('agency.cursor_cache_keeps_pixels_and_coordinate_frames', scope='agency',
           owner='core/skills/screen_pursuit_on_a_page.py', observational=False)
def _cursor_cache_custody_invariant() -> tuple:
    from types import SimpleNamespace

    from core.perception.observed_transfer import PixelSnapshot, ScreenReading

    page = SimpleNamespace(viewport_size={'width': 100, 'height': 100}, url='one')
    first = ScreenReading(dict(text='', bounds=[0, 0, 100, 100]), picture_snapshot=PixelSnapshot(bytes(768), 16, 16))
    other = ScreenReading(first, picture_snapshot=PixelSnapshot(bytes([1])*768, 16, 16))
    assert _cursor_scope(page, first, None)[0] != _cursor_scope(page, other, None)[0]
    assert _cursor_scope(page, first, None)[0] != _cursor_scope(page, first, (.1, .1, .9, .9))[0]
    assert _cursor_scope(page, {}, None)[0][-1] is None
    return ()


@dataclass
class OnAPage:
    """One page of her own browser, as the surface a screen pursuit acts on."""

    page: Any
    name: str
    _focused: bool = field(default=False, init=False)
    #: The page watched as one stream while she uses it (core/perception/watching_it_happen.py).
    watching: Any = field(default=None, init=False)
    _watching_over: Any = field(default=None, init=False)
    _pictures: Any = field(default=None, init=False)
    #: The words of the screen the pointer was last swept over, and where it turned into a hand there.
    _swept: Any = field(default=None, init=False)
    _swept_bounds: Any = field(default=None, init=False)
    _cursor_sweeps: dict = field(default_factory=dict, init=False)
    _material_presses: MaterialPressHistory = field(default_factory=MaterialPressHistory, init=False)

    #: The longest she waits on what plays by itself inside one look, in seconds.
    WATCH_IN_A_LOOK_S = 5.0

    async def read(self, over: tuple[float, float, float, float] | None = None) -> dict[str, Any]:
        from core.perception.what_her_page_shows import look_at_a_page

        await self._watch(over)
        seen = await look_at_a_page(self.page, over, name=self.name)
        if seen is None:
            return {"ok": False, "text": "", "layout": [], "error": "her page could not be photographed"}
        seen.setdefault("ok", True)
        _what_the_stream_shows(seen, self.watching)
        # Whether what is read is a screen of play in a place played in time, for whoever waits on it to come to rest.
        from .screen_pursuit_as_it_happens import (
            AS_IT_HAPPENS,
            a_hand_among_things,
            a_screen_of_play_in_time,
            stuck_on_one_screen_of_play,
        )

        reflexes = AS_IT_HAPPENS.get()
        labels = [str(r.get("text") or "") for r in seen.get("layout") or () if isinstance(r, dict)]
        if reflexes is not None:
            reflexes.play_in_sight = a_screen_of_play_in_time(reflexes, labels)
        # Where she is a hand among things, what answers the pointer is what she plays with, clock or none; and where
        # one screen of play has stood a while, the same, she looks it over with the pointer.
        if (not getattr(reflexes, "play_in_sight", False) or a_hand_among_things()
                or stuck_on_one_screen_of_play(reflexes, str(seen.get("text") or ""))):
            await self._where_the_pointer_turns(seen, over)
            _what_the_stream_shows(seen, self.watching)
        return seen

    async def _where_the_pointer_turns(self, seen: dict[str, Any], over: tuple[float, float, float, float] | None) -> None:
        """Once a screen, the places on it where the pointer turns into a hand (core/perception/where_the_pointer_turns.py),
        among the shapes she may click: a doorway or a hidden thing looks like nothing in particular."""
        from core.perception.where_the_pointer_turns import (
            bind_cursor_samples,
            corners_to_query,
            the_thing_there,
            where_the_pointer_turns,
        )

        scope = _cursor_scope(self.page, seen, over)
        if scope is None:
            self._invalidate_cursor_samples()
            return
        key, scan, bounds = scope
        bounds_key = tuple(round(v, 6) for v in bounds)
        known = self._cursor_sweeps.get(bounds_key)
        if known is not None and key[-1] is not None and known.key == key and known.places:
            places, materials = known.places, known.materials
        else:
            from core.perception.cursor_materials_after_inspection import (
                collector_for_reading,
                cursor_materials_after_inspection,
            )

            collector = collector_for_reading(self.page, seen, bounds)
            hinted = tuple(((bounds[0]+x*bounds[2]-scan[0])/scan[2],
                            (bounds[1]+y*bounds[3]-scan[1])/scan[3]) for x, y in corners_to_query(seen))
            hinted = tuple((x, y) for x, y in hinted if 0 <= x <= 1 and 0 <= y <= 1)
            options = {'hints': hinted} if hinted else {}
            if collector is not None:
                options['on_sample'] = collector.sampled
            sampled = await where_the_pointer_turns(self.page, over, **options)
            places = tuple(((scan[0]+x*scan[2]-bounds[0])/bounds[2],
                            (scan[1]+y*scan[3]-bounds[1])/bounds[3]) for x, y in sampled)
            places = tuple((x, y) for x, y in places if 0 <= x <= 1 and 0 <= y <= 1)
            captured = tuple(collector.targets) if collector is not None and places else () if collector is not None else None
            places, materials = await cursor_materials_after_inspection(
                self.page, seen, places, bounds, over=over, name=self.name, sampled_targets=captured)
            current_scope = _cursor_scope(self.page, seen, over)
            if current_scope is None:
                self._invalidate_cursor_samples()
                return
            key = current_scope[0]
            self._cursor_sweeps.pop(bounds_key, None)
            if places:
                self._cursor_sweeps[bounds_key] = _CursorSweep(key, places, materials)
                while len(self._cursor_sweeps) > 4:
                    self._cursor_sweeps.pop(next(iter(self._cursor_sweeps)))
        words = frozenset(str(seen.get('text') or '').casefold().split())
        self._swept, self._swept_bounds = (words, places), bounds_key
        regions = [r for r in [*(seen.get("shapes") or []), *(seen.get("layout") or [])] if isinstance(r, dict)]
        linked = {}
        unnamed = []
        for x, y in places:
            # Where a thing already read stands there, it is that thing which answers the pointer.
            standing = the_thing_there((x, y), regions)
            if standing is not None:
                linked.setdefault(id(standing), (standing, []))[1].append((x, y))
            else:
                unnamed.append((x, y))
        for standing, hits in linked.values():
            standing['answers_the_pointer'] = True
            if len(hits) == 1:
                standing['pointer_x'], standing['pointer_y'] = hits[0]
            else:
                standing['pointer_ambiguous'] = True
                unnamed.extend(hits)
        counts = {}
        for x, y in unnamed:
            label = f"the one that stands out at {round(x * 100)}% across, {round(y * 100)}% down"
            seen.setdefault("shapes", []).append({"text": label, "center_x": x, "center_y": y,
                                                  "shape": True, "answers_the_pointer": True})
            if materials:
                target = next((target for point, target in zip(places, materials, strict=True) if point == (x, y)), None)
                if target is not None:
                    try:
                        count = await asyncio.to_thread(self._material_presses.count, target)
                    except (ValueError, TypeError, RuntimeError, OverflowError) as exc:
                        record_degradation('cursor_material_exploration', exc, action='dropping optional press costs')
                        self._material_presses.entries = ()
                        count = None
                    if count:
                        counts[label] = count
        bind_cursor_samples(seen, places)
        bind_material_presses(seen, counts)

    async def _watch(self, over: tuple[float, float, float, float] | None) -> None:
        """Watch the part being read as a stream from the first look; and while something plays by itself, wait for it."""
        from core.perception.watching_it_happen import Watching

        if self.watching is not None and over is None:
            # A look at the whole page does not move what she watches: the stream stays on the part she acts on, and her
            # acts on it stay hers to be answered. Measured offline 2026-10-11: each act was followed by a whole-page
            # look that began the stream again, every act's answer was lost, and every click was judged by whether the
            # words changed; a choice that only lit up ("1 Player") was taken to do nothing, and she went back and on
            # between two screens for four minutes.
            return
        if self.watching is None or over != self._watching_over:
            await self.stop_watching()
            self._pictures = await _pictures_of(self.page, over)
            self.watching, self._watching_over = Watching(take=self._pictures), over
            self.watching.start()
            return
        from .screen_pursuit import _tell
        from .screen_pursuit_as_it_happens import AS_IT_HAPPENS, a_screen_of_play_in_time

        # Play that goes on without her is not waited out, it is played: offline 2026-10-11 a game's eggs fell while she
        # watched them "play out on its own" twice, and it was over before she played.
        reflexes = AS_IT_HAPPENS.get()
        if getattr(reflexes, "play_in_sight", False) or a_screen_of_play_in_time(reflexes):
            return
        # Inside one look, which has its own few seconds: a wait longer than the look is a look that never comes back.
        await self.watching.watch(at_most_s=self.WATCH_IN_A_LOOK_S, tell=_tell)

    async def stop_watching(self) -> None:
        if self.watching is not None:
            await self.watching.stop()
            self.watching = None
        pictures, self._pictures = self._pictures, None
        if pictures is not None:
            await pictures.close()

    async def _focus(self) -> None:
        self._focused = await focus_drawing_for_input(self.page)

    def _invalidate_cursor_samples(self) -> None:
        self._swept = self._swept_bounds = None
        self._cursor_sweeps.clear()

    async def press(self, key: str) -> bool:
        context = current(whose="page_input.press")
        context.check()
        from core.skills.screen_pursuit_surface import PRESSABLE_KEYS

        name = str(key or "").strip().lower()
        if name not in PRESSABLE_KEYS:
            return False
        await self._focus()
        context.check()
        from .screen_pursuit_as_it_happens import AS_IT_HAPPENS

        reflexes = AS_IT_HAPPENS.get()
        watching = self.watching
        window = watching.begin_act(name) if watching is not None else None
        delivered = False
        recorded = begin_navigation_input(self.page, 'press', (name,))
        try:
            if reflexes is not None:
                # In a game, a key is held a moment and watched, the way a person tries one.
                await reflexes.pressed(name)
            else:
                from core.skills.keys_tapped_as_a_hand_taps import TAP_HELD_S

                # Held a moment, as a hand presses one: a page that asks each frame which keys are down sees it.
                await self.page.keyboard.press(_KEY_NAMES.get(name, name), delay=TAP_HELD_S * 1000.0)
            delivered = True
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return False
        finally:
            finish_navigation_input(recorded, delivered)
            self._invalidate_cursor_samples()
            if window is not None:
                watching.finish_act(window, delivered)
        return True

    async def press_many(self, keys: Sequence[str]) -> int:
        landed = 0
        for key in keys:
            if not await self.press(key):
                break
            landed += 1
        return landed

    async def click(self, x: float, y: float, bounds: Sequence[int] | None) -> bool:
        """A click at shares of ``bounds``, in the page's own pixels; never outside them."""
        context = current(whose="page_input.click")
        context.check()
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
        if bounds and len(bounds) >= 4:
            left, top, wide, tall = (float(value) for value in bounds[:4])
        else:
            from core.perception.what_her_page_shows import the_page_size

            wide, tall = await the_page_size(self.page)
            left = top = 0.0
        bounds_key = tuple(round(v, 6) for v in (left, top, wide, tall))
        swept = self._cursor_sweeps.get(bounds_key)
        places = swept.places if swept is not None else self._swept[1] if self._swept is not None and self._swept_bounds is None else ()
        chosen = next((i for i, (a, b) in enumerate(places) if abs(x-a) < 1e-6 and abs(y-b) < 1e-6), None)
        arrived = None
        if chosen is not None and swept is not None and swept.materials:
            from core.perception.a_cursor_target_in_its_pixels import arrive_at_cursor_target

            arrived = await arrive_at_cursor_target(self.page, swept.materials[chosen], (x, y), check=context.check)
            if arrived is None:
                self._invalidate_cursor_samples()
                return False
            x, y = (arrived[0]-left)/wide, (arrived[1]-top)/tall
        elif chosen is not None:
            # A place found by the pointer turning into a hand is clicked where it answers now: it may move.
            from core.perception.where_the_pointer_turns import where_the_hand_is_now

            found = await where_the_hand_is_now(self.page, x, y, left, top, wide, tall)
            if found is None:
                self._invalidate_cursor_samples()
                return False
            x, y = found
        watching = self.watching
        window = None
        delivered = False
        recorded = begin_navigation_input(self.page, 'click', (left+x*wide, top+y*tall), bounds=(left, top, wide, tall))
        try:
            context.check()
            window = watching.begin_act(f"click at {x:.2f}, {y:.2f}") if watching is not None else None
            if arrived is not None:
                # Feedback already placed and rested the pointer on this
                # material. Another approach movement would invalidate it.
                await self.page.mouse.down()
                try:
                    await asyncio.sleep(HAND_PRESSES_S)
                finally:
                    await self.page.mouse.up()
            else:
                await as_a_hand_clicks(self.page, left + x * wide, top + y * tall)
            delivered = True
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return False
        finally:
            finish_navigation_input(recorded, delivered)
            self._invalidate_cursor_samples()
            if window is not None:
                watching.finish_act(window, delivered)
        # A click on the drawing is also what gives it the keyboard.
        self._focused = True
        if arrived is not None:
            try:
                self._material_presses.delivered(swept.materials[chosen])
            except (ValueError, TypeError, RuntimeError, OverflowError) as exc:
                record_degradation('cursor_material_exploration', exc, action='forgetting history after completed input')
                self._material_presses.entries = ()
        return True

    #: The steps a carry is made in, and the pause between them: a page sees the pointer travel with the button down,
    #: as a hand carries it, and not a jump from the one place to the other.
    CARRY_STEPS = 12
    CARRY_STEP_S = 0.02

    async def carry(self, start: tuple[float, float], end: tuple[float, float], bounds: Sequence[int] | None) -> bool:
        """A press at ``start`` carried with the button held to ``end`` and let go, at shares of ``bounds``."""
        import asyncio

        context = current(whose="page_input.carry")
        context.check()

        if not all(0.0 <= v <= 1.0 for v in (*start, *end)):
            return False
        if bounds and len(bounds) >= 4:
            left, top, wide, tall = (float(value) for value in bounds[:4])
        else:
            from core.perception.what_her_page_shows import the_page_size

            wide, tall = await the_page_size(self.page)
            left = top = 0.0
        (x0, y0), (x1, y1) = ((left + x * wide, top + y * tall) for x, y in (start, end))
        watching = self.watching
        window = None
        delivered = False
        recorded = begin_navigation_input(self.page, 'carry', (x0, y0, x1, y1), bounds=(left, top, wide, tall))
        try:
            context.check()
            await self.page.mouse.move(x0, y0)
            context.check()
            window = watching.begin_act(
                f"carry from {start[0]:.2f}, {start[1]:.2f} to {end[0]:.2f}, {end[1]:.2f}"
            ) if watching is not None else None
            try:
                await self.page.mouse.down()
                for step in range(1, self.CARRY_STEPS + 1):
                    context.check()
                    share = step / self.CARRY_STEPS
                    await self.page.mouse.move(x0 + (x1 - x0) * share, y0 + (y1 - y0) * share)
                    await asyncio.sleep(self.CARRY_STEP_S)
            finally:
                await self.page.mouse.up()
            delivered = True
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return False
        finally:
            finish_navigation_input(recorded, delivered)
            self._invalidate_cursor_samples()
            if window is not None:
                watching.finish_act(window, delivered)
        self._focused = True
        return True

    async def identity(self) -> dict[str, str]:
        try:
            title = str(await self.page.title() or "")
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            title = ""
        return {"url": str(getattr(self.page, "url", "") or ""), "title": title, "error": ""}


async def _pictures_of(page: Any, over: tuple[float, float, float, float] | None) -> Any:
    """Where the stream's pictures come from: the page's own frames as it draws them, else screenshots of the part."""
    from core.perception.frames_as_they_are_drawn import CanvasFrames, PageFrames
    from core.perception.what_her_page_shows import the_page_size

    wide, tall = await the_page_size(page)
    left, top, right, bottom = over if over is not None else (0.0, 0.0, 1.0, 1.0)
    clip = {"x": left * wide, "y": top * tall, "width": max(1.0, (right - left) * wide), "height": max(1.0, (bottom - top) * tall)}
    frames = PageFrames(page, every_nth_frame=4)

    # Reading a drawing directly avoids streaming an entire window at the
    # display's refresh rate just to sample its changes eight times a second.
    canvas = CanvasFrames(page) if over is not None else None
    return _PagePictures(page, clip, frames, canvas)


@dataclass
class _PagePictures:
    """The pictures and the stream that produces them have the same owner."""

    page: Any
    clip: dict[str, float]
    frames: Any
    canvas: Any = None

    async def __call__(self) -> Any:

        if self.canvas is not None and not self.canvas.unavailable:
            drawn = await self.canvas.look(self.clip)
            if drawn is not None:
                return drawn
        if not self.frames.unavailable:
            streamed = await self.frames.look(self.clip)
            if streamed is not None:
                return streamed
        from core.perception.a_picture_of_her_page import picture_of

        # The whole window, cut down here: a picture of part of the page made a visible window flicker (LIVE 2026-10-09).
        picture = await picture_of(self.page, self.clip, css=False, kind="png")
        return None if picture is None else picture[:, :, ::-1].copy()

    async def close(self) -> None:
        if self.canvas is not None:
            await self.canvas.close()
        await self.frames.close()


def _what_the_stream_shows(seen: dict[str, Any], watching: Any) -> None:
    """Mark the writing that keeps changing by itself, a scene's lines, which are read and not pressed; and whether her last act was answered."""
    if watching is None:
        return
    for region in seen.get("layout") or []:
        try:
            box = (float(region["x"]), float(region["y"]), float(region["width"]), float(region["height"]))
        except (KeyError, TypeError, ValueError):
            continue
        if watching.keeps_changing_on_its_own(box):
            region["of_its_own"] = True
    if watching.acts:
        when, _act = watching.acts[-1]
        seen["answered"] = watching.answered(when)
        seen["answer_window"] = watching.answer_window()
        # Where what her last act set going came to rest: where a machine she built stopped when tested.
        seen["came_to_rest_at"] = watching.where_it_came_to_rest(when)


def it_was_answered(changed: bool | None, observation: dict[str, Any]) -> bool | None:
    """Use the owned pixel response, or still comparisons when no stream owns it.

    Identical words can accompany a visual response. An incomplete stream
    supplies None; a measured autonomous change keeps its False veto.
    """
    if changed is None:
        return None
    window = observation.get("answer_window")
    if isinstance(window, dict):
        if window.get("delivered") is not True:
            return None
        measured = observation.get("answered")
        return measured if type(measured) is bool else None
    return bool(changed) and observation.get("answered") is not False


#: The page a hand-over is playing on, for as long as it plays.
HER_OWN_PAGE: contextvars.ContextVar[OnAPage | None] = contextvars.ContextVar(
    "aura_her_own_page", default=None
)


def on_her_page() -> OnAPage | None:
    """The page the pursuit is acting on, when it is acting on one."""
    return HER_OWN_PAGE.get()

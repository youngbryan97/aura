"""Watching something with a person where they watch it: in their own browser, on their screen, as they see and hear it.

Watching together is watching the same thing in the same place. When a person asks her to watch with them, what she
found in her own browser is opened in theirs (their default browser, as any link is opened), and she watches that:
she looks at its window every few seconds with her eyes and hears its sound with her ear (core/senses/her_ear.py),
the sound of that application alone, so neither her own voice nor anything else on the machine. Nothing in their
browser is pressed or read but the window's picture, its title, and its sound.

Her own browser is where she looks things up, and a site may decline to play to it what it plays to a person (LIVE
19 October 2026: a video site stopped her browser's playback after a minute, "Something went wrong", with its
captions "unavailable"). Watching with a person in their place is also what watching together is.

A watching here ends when the window stops showing what was opened (its title changes when the next video plays, or
the window closes), or when it has played as long as it lasts and two minutes more, the room the page-side watching
gives adverts (core/skills/sovereign_browser_on_the_page.py).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.WatchingTheScreen")

__all__ = ["OnTheirScreen", "on_their_screen"]

#: How long a window is waited for once the link is opened, and the room after its length that a watching gives.
WINDOW_WAITED_S = 30.0
ADVERTS_ROOM_S = 120.0
#: The longest side, in pixels, of the window's picture her eyes are given. LIVE 20 October 2026: the whole window at
#: the screen's resolution was 6,869 image tokens, which her eyes were still reading (580-850 tokens a second) when a
#: look's eight seconds ran out, every look. At this size a picture is about 460.
PICTURE_LONGEST = 768
#: The least share of its window's width and height the part where a video plays takes. LIVE 20 October 2026: on the
#: six-second film, whose colour bars hold still, what changed in more looks than not was a button of the player's own
#: controls, and her eyes were given the button. A player is wider than this in any page made for watching (a video
#: site's default player is about two thirds of a laptop's window).
SMALLEST_PLAYER = 0.25


def _smaller(picture: bytes, longest: int = PICTURE_LONGEST, *, within: tuple[float, ...] | None = None) -> bytes:
    """A JPEG at most ``longest`` pixels on its longer side, of the part ``within`` (fractions of its width and
    height: left, top, right, bottom) when given."""
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(picture))
    if within:
        width, height = image.size
        image = image.crop((round(within[0] * width), round(within[1] * height),
                            round(within[2] * width), round(within[3] * height)))
    image.thumbnail((longest, longest))
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=80)
    return out.getvalue()


class _WhereItPlays:
    """Where in a window its video plays: the part that differs from one look to the next in more looks than not.

    Pictures of a window are lossless, so the page around a video is the same to the pixel from one look to the next,
    and what changes once (a picture loading below it, a menu closing) changes in fewer looks than what plays. LIVE 20
    October 2026: given the whole window, her eyes described the browser ("a macOS media player", "the Apple TV+
    app") and not the film in it."""

    def __init__(self) -> None:
        self.last: Any = None
        self.changed: Any = None
        self.pairs = 0

    def took(self, picture: bytes) -> tuple[float, float, float, float] | None:
        """The part where it plays, as fractions of the window (left, top, right, bottom), with this picture taken in;
        None until something has changed in more looks than not, and again when the window changes size."""
        import io

        import numpy as np
        from PIL import Image

        image = Image.open(io.BytesIO(picture)).convert("RGB")
        image.thumbnail((PICTURE_LONGEST, PICTURE_LONGEST))
        now = np.asarray(image)
        last, self.last = self.last, now
        if last is None or last.shape != now.shape:
            self.changed, self.pairs = None, 0
            return None
        differs = (last != now).any(axis=2).astype(int)
        self.changed = differs if self.changed is None else self.changed + differs
        self.pairs += 1
        rows, columns = np.nonzero(self.changed * 2 > self.pairs)
        if not len(rows):
            return None
        height, width = now.shape[:2]
        box = (columns.min() / width, rows.min() / height, (columns.max() + 1) / width, (rows.max() + 1) / height)
        # Too small to be where it plays: a control, a cursor, a clock. The whole window is looked at instead.
        return box if box[2] - box[0] >= SMALLEST_PLAYER and box[3] - box[1] >= SMALLEST_PLAYER else None


def _their_browser(url: str) -> tuple[str, str]:
    """The bundle identifier and name of the application that opens ``url`` on this machine."""
    from AppKit import NSWorkspace
    from Foundation import NSURL, NSBundle

    found = NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(NSURL.URLWithString_(url))
    if found is None:
        return "", ""
    bundle = NSBundle.bundleWithURL_(found)
    name = str(bundle.objectForInfoDictionaryKey_("CFBundleName") or found.lastPathComponent().replace(".app", ""))
    return str(bundle.bundleIdentifier() or ""), name


def _words_of(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", title.lower())}


def side_by_side(visible: tuple[float, float, float, float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """A screen's usable area (left, top, width, height, from its top left) as two equal halves side by side: hers on
    the left, the video's on the right."""
    left, top, width, height = visible
    half = round(width / 2)
    return (left, top, half, height), (left + half, top, width - half, height)


def _usable_area_of_the_screen_holding(bounds: dict[str, Any]) -> tuple[float, float, float, float]:
    """The usable area (less the menu bar and the Dock) of the screen a window's centre is on, from the top left."""
    from AppKit import NSScreen

    screens = list(NSScreen.screens())
    tallest = screens[0].frame().size.height  # the screen whose bottom left is the origin of the others
    middle_x = float(bounds.get("X", 0)) + float(bounds.get("Width", 0)) / 2
    middle_y = float(bounds.get("Y", 0)) + float(bounds.get("Height", 0)) / 2
    for screen in screens:
        frame = screen.frame()
        top = tallest - (frame.origin.y + frame.size.height)
        if frame.origin.x <= middle_x < frame.origin.x + frame.size.width and top <= middle_y < top + frame.size.height:
            usable = screen.visibleFrame()
            return (usable.origin.x, tallest - (usable.origin.y + usable.size.height), usable.size.width, usable.size.height)
    usable = screens[0].visibleFrame()
    return (usable.origin.x, tallest - (usable.origin.y + usable.size.height), usable.size.width, usable.size.height)


def _her_window() -> dict[str, Any] | None:
    """Her own app's window on screen: the one titled with her name, of the process that is not a browser."""
    import Quartz

    for window in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID) or []:
        if str(window.get("kCGWindowName") or "") == "Aura" and int(window.get("kCGWindowLayer") or 0) == 0 \
                and str(window.get("kCGWindowOwnerName") or "") in ("Python", "Aura"):
            return dict(window)
    return None


def _put(window: dict[str, Any], area: tuple[float, ...]) -> bool:
    """Move and size one window to ``area`` through her Accessibility access, and raise it; False where it could not
    be moved."""
    from ApplicationServices import (
        AXUIElementCopyAttributeValue,
        AXUIElementCreateApplication,
        AXUIElementPerformAction,
        AXUIElementSetAttributeValue,
        AXValueCreate,
        kAXPositionAttribute,
        kAXRaiseAction,
        kAXSizeAttribute,
        kAXTitleAttribute,
        kAXValueCGPointType,
        kAXValueCGSizeType,
        kAXWindowsAttribute,
    )
    from Quartz import CGPoint, CGSize

    error, windows = AXUIElementCopyAttributeValue(AXUIElementCreateApplication(int(window["kCGWindowOwnerPID"])),
                                                   kAXWindowsAttribute, None)
    if error or not windows:
        return False
    named = str(window.get("kCGWindowName") or "")
    for candidate in windows:
        _error, title = AXUIElementCopyAttributeValue(candidate, kAXTitleAttribute, None)
        if named and str(title or "") != named:
            continue
        left, top, width, height = area
        AXUIElementSetAttributeValue(candidate, kAXPositionAttribute, AXValueCreate(kAXValueCGPointType, CGPoint(left, top)))
        AXUIElementSetAttributeValue(candidate, kAXSizeAttribute, AXValueCreate(kAXValueCGSizeType, CGSize(width, height)))
        AXUIElementPerformAction(candidate, kAXRaiseAction)
        return True
    return False


def _both_in_view(video: dict[str, Any]) -> bool:
    """Her window and the one the video plays in, each half of the screen hers is on, both raised. Bryan, 20 October
    2026: "she should be able to bring the window up, and adjust it to the screen so that both she and the window fit
    in frame"."""
    hers = _her_window()
    area = _usable_area_of_the_screen_holding((hers or video).get("kCGWindowBounds") or {})
    her_half, its_half = side_by_side(area)
    placed = _put(video, its_half)
    if hers is not None:
        placed = _put(hers, her_half) and placed
    return placed


def _window_showing(owner: str, title: str) -> dict[str, Any] | None:
    """The window of ``owner`` whose title shares most of the words of ``title``, on screen; None when none does."""
    import Quartz

    wanted = _words_of(title)
    best, share = None, 0.0
    for window in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID) or []:
        if str(window.get("kCGWindowOwnerName") or "") != owner or int(window.get("kCGWindowLayer") or 0) != 0:
            continue
        named = _words_of(str(window.get("kCGWindowName") or ""))
        found = len(wanted & named) / len(wanted) if wanted else 0.0
        if found > share:
            best, share = dict(window), found
    # More than half of the title's words: a browser window's title is the page's title and its own name.
    return best if share > 0.5 else None


def on_their_desktop(effect: str) -> Any:
    """The scope a step of watching on the person's screen runs in: an action on their desktop, taken at their asking.

    The watching goes on beside the conversation, in a task begun from inside her browser's step, and carries that
    step's governance (a network call), under which nothing may be started on their desktop: LIVE 20 October 2026,
    "could not be opened on their screen: ... requires governance domain [...] but active token domain is
    network_call". Each such step has its own receipt saying what it is.
    """
    from core.governance_context import local_internal_governed_scope

    return local_internal_governed_scope("core.skills.watching_the_screen", domain="environment_action",
                                         constraints={"effect": effect, "user_visible": True, "asked_by_the_person": True})


class OnTheirScreen:
    """What plays in the person's own browser: its window looked at, its sound heard."""

    def __init__(self, url: str, title: str, duration_s: float, words: str = "") -> None:
        self.url, self.title, self.duration_s, self.words = url, title, float(duration_s or 0.0), words
        self.bundle, self.owner = _their_browser(url)
        self.window: dict[str, Any] | None = None
        self.ear: Any = None
        self.began = 0.0
        #: A digest of each picture of the window, to tell a video still playing from one that has stopped.
        self.pictures: list[str] = []
        #: Where in the window it plays, from what changes between pictures of it.
        self.where_it_plays = _WhereItPlays()

    async def open(self) -> bool:
        """Open the link in their browser and find its window; False when it is not shown."""
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        if not self.bundle:
            return False
        with on_their_desktop("the video they asked to watch, opened in their browser"):
            opened = await get_subprocess_gateway().spawn_async(["/usr/bin/open", "-b", self.bundle, self.url],
                                                                source="core.skills.watching_the_screen",
                                                                accelerator_capability="none")
        await opened.wait()
        deadline = time.monotonic() + WINDOW_WAITED_S
        while time.monotonic() < deadline:
            self.window = await asyncio.to_thread(_window_showing, self.owner, self.title)
            if self.window is not None:
                try:
                    if not await asyncio.to_thread(_both_in_view, self.window):
                        logger.info("her window and the video's could not both be put in view")
                except (ImportError, RuntimeError, OSError, ValueError, TypeError, KeyError, AttributeError) as why:
                    logger.info("her window and the video's were left where they were: %s", why)
                self.window = await asyncio.to_thread(_window_showing, self.owner, self.title) or self.window
                self.began = time.monotonic()
                return True
            await asyncio.sleep(1.0)
        return False

    async def says(self) -> dict[str, Any]:
        return {"title": self.title, "words": self.words}

    async def start(self) -> None:
        from core.senses.her_ear import HerEar

        self.ear = HerEar(self.bundle)
        if not await self.ear.start():
            logger.info("watching on their screen without hearing it: %s", self.ear.why_not)

    async def state(self) -> dict[str, Any] | None:
        played = time.monotonic() - self.began
        showing = await asyncio.to_thread(_window_showing, self.owner, self.title)
        # Past its length, a window whose picture has not changed since the look before has stopped playing: a page
        # keeps its title when its video ends (LIVE 20 October 2026, a six-second film watched for two minutes).
        stopped = bool(self.duration_s) and played > self.duration_s and len(self.pictures) > 1 and \
            self.pictures[-1] == self.pictures[-2]
        if showing is None or stopped or (self.duration_s and played > self.duration_s + ADVERTS_ROOM_S):
            return {"ended": True, "at": played, "duration": self.duration_s}
        self.window = showing
        # Their player's own clock is not read here, only the time since its window opened: at most its length, so a
        # video that has ended is not said to be at 0:25 of 0:06 (LIVE 20 October 2026, she wondered whether it looped).
        at = min(played, self.duration_s) if self.duration_s else played
        return {"at": at, "duration": self.duration_s, "ended": False, "paused": False}

    async def tend(self) -> str:
        return ""

    async def heard(self) -> list[dict[str, Any]] | None:
        return self.ear.since() if self.ear is not None and self.ear.listening else None

    def eyes(self, *, seconds: float = 8.0) -> Any:
        from core.perception.her_eyes import look_with_her_eyes, plainly
        from core.skills.watching_along import LOOK_TOKENS

        async def look(_when: str) -> str:
            if self.window is None:
                return ""
            shot = Path(f"/tmp/aura-watching-{int(self.window.get('kCGWindowNumber') or 0)}.png")
            try:
                from core.runtime.subprocess_gateway import get_subprocess_gateway

                with on_their_desktop("a picture of the window the video plays in"):
                    taken = await get_subprocess_gateway().spawn_async(
                        ["/usr/sbin/screencapture", "-x", "-o", "-t", "png", "-l", str(int(self.window["kCGWindowNumber"])), str(shot)],
                        source="core.skills.watching_the_screen", accelerator_capability="none")
                await taken.wait()
                taken_bytes = await asyncio.to_thread(shot.read_bytes)
                await asyncio.to_thread(shot.unlink, missing_ok=True)
                self.pictures.append(hashlib.sha1(taken_bytes).hexdigest())
                within = await asyncio.to_thread(self.where_it_plays.took, taken_bytes)
                picture = base64.b64encode(await asyncio.to_thread(_smaller, taken_bytes, within=within)).decode("ascii")
                asked = "What is happening in this picture?" if within else "What is happening in the video playing in this window?"
                return plainly(await look_with_her_eyes(asked, picture, max_tokens=LOOK_TOKENS, timeout_s=seconds))
            except (RuntimeError, OSError, ValueError, TypeError, KeyError, TimeoutError) as why:
                logger.info("a look did not come back: %s", why)
                return ""

        return look

    async def done(self) -> None:
        if self.ear is not None:
            await self.ear.stop()


async def on_their_screen(url: str, title: str, duration_s: float, words: str = "") -> OnTheirScreen | None:
    """What she found, opened where the person watches; None when it could not be shown there."""
    try:
        screen = OnTheirScreen(url, title, duration_s, words)
        return screen if await screen.open() else None
    except (ImportError, RuntimeError, OSError, ValueError, AttributeError) as why:
        logger.info("it could not be opened on their screen: %s", why)
        return None

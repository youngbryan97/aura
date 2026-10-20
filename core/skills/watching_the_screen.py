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

    async def open(self) -> bool:
        """Open the link in their browser and find its window; False when it is not shown."""
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        if not self.bundle:
            return False
        with on_their_desktop("the video they asked to watch, opened in their browser"):
            opened = await get_subprocess_gateway().spawn_async(["/usr/bin/open", "-b", self.bundle, self.url],
                                                                source="core.skills.watching_the_screen")
        await opened.wait()
        deadline = time.monotonic() + WINDOW_WAITED_S
        while time.monotonic() < deadline:
            self.window = await asyncio.to_thread(_window_showing, self.owner, self.title)
            if self.window is not None:
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
        if showing is None or (self.duration_s and played > self.duration_s + ADVERTS_ROOM_S):
            return {"ended": True, "at": played, "duration": self.duration_s}
        self.window = showing
        return {"at": played, "duration": self.duration_s, "ended": False, "paused": False}

    async def tend(self) -> str:
        return ""

    async def heard(self) -> list[dict[str, Any]] | None:
        return self.ear.since() if self.ear is not None and self.ear.listening else None

    def eyes(self, *, seconds: float = 8.0) -> Any:
        from core.perception.her_eyes import look_with_her_eyes

        async def look(_when: str) -> str:
            if self.window is None:
                return ""
            shot = Path(f"/tmp/aura-watching-{int(self.window.get('kCGWindowNumber') or 0)}.jpg")
            try:
                from core.runtime.subprocess_gateway import get_subprocess_gateway

                with on_their_desktop("a picture of the window the video plays in"):
                    taken = await get_subprocess_gateway().spawn_async(
                        ["/usr/sbin/screencapture", "-x", "-o", "-t", "jpg", "-l", str(int(self.window["kCGWindowNumber"])), str(shot)],
                        source="core.skills.watching_the_screen")
                await taken.wait()
                picture = base64.b64encode(await asyncio.to_thread(shot.read_bytes)).decode("ascii")
                await asyncio.to_thread(shot.unlink, missing_ok=True)
                return (await look_with_her_eyes("What is happening in the video playing in this window?", picture,
                                                 max_tokens=140, timeout_s=seconds)).strip()
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

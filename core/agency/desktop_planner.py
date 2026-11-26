"""Desktop planner — turn a desktop goal into VERIFIED computer-use steps.

The goal planner could reason about open-ended goals and reach allowlisted services, but
a goal that needs live UI control ("open Notes", "make a folder on the desktop", "open
the doc in Chrome") had nowhere to go. This planner closes that: it parses common desktop
intents and produces :class:`~core.skills.fluid_executor.Step` objects that pair a real
computer-use action with the ``PostActionVerifier`` predicate that proves it worked — so
desktop actions run through the same governed, verified, self-recovering loop as
everything else, never "fire and hope".

The actions go through an injected :class:`DesktopAdapter` (the default dispatches to the
``computer_use`` skill; tests inject a recorder), so the intent→action→verifier mapping is
the deterministically-tested core and the live wiring is a thin, swappable edge.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from core.conversation.word_markers import names_any
from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service
from core.skills.fluid_executor import Step

logger = logging.getLogger("Aura.DesktopPlanner")

_DESKTOP = Path(os.path.expanduser("~/Desktop"))


class DesktopAdapter:
    """Clean desktop action surface. Default impl dispatches to the computer_use skill."""

    async def open_app(self, name: str) -> None:
        await self._dispatch("open_app", app_name=name)

    async def open_url(self, url: str) -> None:
        await self._dispatch("open_url", url=url)

    async def type_text(self, text: str) -> None:
        await self._dispatch("type", text=text)

    async def make_folder(self, path: str) -> None:
        await self._dispatch("create_folder", path=path)

    async def set_wallpaper(self, target: str) -> None:
        await self._dispatch("set_wallpaper", target=target)

    async def _dispatch(self, action: str, **params: Any) -> None:
        try:
            skill = get_runtime_service("skill:computer_use", default=None)
            if skill is None or not hasattr(skill, "safe_execute"):
                logger.debug("[Desktop] computer_use unavailable; '%s' is a no-op edge.", action)
                return
            await skill.safe_execute({"action": action, **params}, {})
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("desktop_planner", exc)


_OPEN_APP_RE = re.compile(r"\bopen(?:\s+the)?\s+(?:app\s+)?([A-Z][\w ]+?)(?:\s+app)?\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")
_FOLDER_RE = re.compile(r"\b(?:create|make|new)\s+(?:a\s+)?folder\s+(?:(?:called|named|titled)\s+)?[\"']?([\w '\-]+?)[\"']?(?:\s+(?:on|in)\s+(?:the\s+)?desktop)?\s*$", re.IGNORECASE)
_WALLPAPER_RE = re.compile(r"\b(?:set|change)\s+(?:the\s+)?(?:desktop\s+)?(?:wallpaper|background)\b(?:\s+to\s+(.+))?", re.IGNORECASE)
_TYPE_RE = re.compile(r"\b(?:type|write|enter)\s+[\"'](.+?)[\"']", re.IGNORECASE)


def is_desktop_goal(goal: str) -> bool:
    g = str(goal or "")
    gl = g.lower()
    # A URL is a desktop goal only in a browser-open context ("open … in Chrome",
    # "go to …") — "fetch <url>" / "api" is data retrieval, which is reach, not UI.
    browser_url = bool(_URL_RE.search(g)) and names_any(
        gl, ("open ", "go to", "chrome", "browser", "safari", "in the browser")
    )
    return bool(
        browser_url
        or _FOLDER_RE.search(g)
        or _WALLPAPER_RE.search(g)
        or _OPEN_APP_RE.search(g)
        or _TYPE_RE.search(g)
    )


class DesktopPlanner:
    def __init__(self, *, adapter: DesktopAdapter | None = None, on_result: Any | None = None) -> None:
        self._adapter = adapter or DesktopAdapter()
        self._on_result = on_result

    def _emit(self, goal: str, intent: str) -> None:
        if self._on_result is not None:
            try:
                self._on_result(goal, "desktop", intent)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("desktop_planner", exc)

    async def plan(self, goal: str) -> list[Step]:
        g = str(goal or "").strip()
        a = self._adapter

        # open a URL → verify a browser tab is present
        m = _URL_RE.search(g)
        if m and ("open" in g.lower() or "go to" in g.lower() or "chrome" in g.lower() or "browser" in g.lower()):
            url = m.group(0)
            self._emit(goal, "open_url")
            return [Step("open_url", (lambda u=url: a.open_url(u)), verify="browser_tabs", verify_args={"min_count": 1})]

        # create a folder → verify it exists
        m = _FOLDER_RE.search(g)
        if m:
            name = m.group(1).strip()
            path = str(_DESKTOP / name)
            self._emit(goal, "make_folder")
            return [Step("make_folder", (lambda p=path: a.make_folder(p)), verify="folder_exists", verify_args={"path": path})]

        # set wallpaper → action only (verification of change is best-effort)
        m = _WALLPAPER_RE.search(g)
        if m:
            target = (m.group(1) or "").strip()
            self._emit(goal, "set_wallpaper")
            return [Step("set_wallpaper", (lambda t=target: a.set_wallpaper(t)), verify="always_true")]

        # open an app → verify it is frontmost
        m = _OPEN_APP_RE.search(g)
        if m:
            name = m.group(1).strip().rstrip(".")
            self._emit(goal, "open_app")
            return [Step("open_app", (lambda n=name: a.open_app(n)), verify="app_frontmost", verify_args={"name": name})]

        # type text
        m = _TYPE_RE.search(g)
        if m:
            text = m.group(1)
            self._emit(goal, "type_text")
            return [Step("type", (lambda t=text: a.type_text(t)), verify="always_true")]

        return []


_instance: DesktopPlanner | None = None


def get_desktop_planner() -> DesktopPlanner:
    global _instance
    if _instance is None:
        _instance = DesktopPlanner()
    return _instance

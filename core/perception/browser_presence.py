"""Show where her browser acts, without putting the mark into her observations."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from core.runtime.errors import record_degradation
from core.verify.invariants import Violation, invariant

logger = logging.getLogger("Aura.BrowserPresence")
BOOTSTRAP = Path(__file__).with_suffix(".js").read_text(encoding="utf-8").replace(
    "__AURA_NEURON__",
    json.dumps((Path(__file__).resolve().parents[2] / "interface/static/neuron_glyph.svg").read_text(encoding="utf-8")),
)
_ERRORS = (RuntimeError, AttributeError, TypeError, ValueError, OSError, TimeoutError)
_VISIBLE_UNTIL: weakref.WeakKeyDictionary[Any, float] = weakref.WeakKeyDictionary()
try:
    from playwright.async_api import Error as _PageError
except ImportError:
    _PageError = RuntimeError


def activity_payload(action: str, stage: str, raw_url: str, visible: bool) -> dict[str, Any]:
    """Activity names and a site origin; page text and account paths stay local."""
    if action not in {"navigating", "reading", "clicking", "typing", "scrolling", "closed"}:
        raise ValueError("unknown browser activity")
    if stage not in {"started", "completed", "failed"}:
        raise ValueError("unknown browser activity stage")
    try:
        parts = urlsplit(raw_url)
        url = urlunsplit((parts.scheme, parts.hostname or "", "", "", "")) if parts.scheme in {"http", "https"} else ""
    except ValueError:
        url = ""
    return {"type": "browser_activity", "action": action, "stage": stage,
            "url": url, "visible": bool(visible), "at": time.time()}


@invariant("browser_presence.site_origin_only", scope="browser_presence", owner=__name__)
def activity_keeps_page_secrets_local():
    payload = activity_payload("reading", "started", "https://user:secret@example.org/account/private?token=hidden#message", False)
    if payload["url"] != "https://example.org":
        yield Violation(subject="browser activity", message="The activity address includes more than its site origin.")


async def show_activity(browser: Any, action: str, *, stage: str = "started", box: dict | None = None) -> None:
    """A cosmetic failure cannot discard an otherwise valid browser action."""
    if not getattr(browser, "_presence_available", False):
        return
    page = getattr(browser, "page", None)
    raw_url = str(getattr(page, "url", "") or "")
    data = activity_payload(action, stage, raw_url, getattr(browser, "visible", False))
    try:
        from core.event_bus import get_event_bus

        await asyncio.wait_for(get_event_bus().publish("telemetry", data), timeout=.25)
    except _ERRORS as why:
        logger.debug("Browser activity did not reach the HUD: %s", why)
    if page is None:
        return
    try:
        await asyncio.wait_for(page.evaluate(
            "data => window.__auraBrowserPresence?.show(data)",
            {"action": action, "stage": stage, "box": box},
        ), timeout=.5)
        _VISIBLE_UNTIL[page] = time.monotonic() + (5.0 if stage == "started" else 1.1)
    except (*_ERRORS, _PageError) as why:
        record_degradation("browser_presence", why, severity="debug", action="continued without the browser activity overlay")


@asynccontextmanager
async def without_presence(page: Any):
    """Reference-counted suspension, including when a screenshot fails."""
    # Most play frames have no mark to hide. They need no extra browser round trips.
    if _VISIBLE_UNTIL.get(page, 0.0) < time.monotonic():
        yield
        return
    suspended = False
    try:
        suspended = bool(await asyncio.wait_for(page.evaluate(
            "() => window.__auraBrowserPresence?.suspend() || false"
        ), timeout=.5))
    except (*_ERRORS, _PageError) as why:
        logger.debug("No browser overlay to suspend: %s", why)
    try:
        yield
    finally:
        if suspended:
            try:
                await asyncio.wait_for(page.evaluate(
                    "() => window.__auraBrowserPresence?.resume()"
                ), timeout=.5)
            except (*_ERRORS, _PageError) as why:
                logger.debug("Browser overlay was not restored: %s", why)

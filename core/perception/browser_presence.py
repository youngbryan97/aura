"""Show where her browser acts, without putting the mark into her observations."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree

from core.runtime.errors import record_degradation
from core.verify.invariants import Violation, invariant

logger = logging.getLogger("Aura.BrowserPresence")


def as_nodes(svg: str) -> list:
    """The glyph as [tag, attributes, children], for a page that refuses markup.

    Comments are left out; namespaced attributes are left out too, since the
    script sets every attribute without a namespace.
    """
    def node(element: ElementTree.Element) -> list:
        children: list = [element.text] if (element.text or "").strip() else []
        for child in element:
            children.append(node(child))
            if (child.tail or "").strip():
                children.append(child.tail)
        attributes = {name: value for name, value in element.attrib.items() if not name.startswith("{")}
        return [element.tag.rpartition("}")[2], attributes, children]

    return node(ElementTree.fromstring(svg))


BOOTSTRAP = Path(__file__).with_suffix(".js").read_text(encoding="utf-8").replace(
    "__AURA_NEURON__",
    json.dumps(as_nodes((Path(__file__).resolve().parents[2] / "interface/static/neuron_glyph.svg").read_text(encoding="utf-8"))),
)
_ERRORS = (RuntimeError, AttributeError, TypeError, ValueError, OSError, TimeoutError)
#: Until when each page's mark is on screen: from her first action on it until her browser closes (math.inf).
_VISIBLE_UNTIL: weakref.WeakKeyDictionary[Any, float] = weakref.WeakKeyDictionary()
#: Where her pointer last was in each browser, so a page she opens next shows it where it was.
_LAST_POINT: weakref.WeakKeyDictionary[Any, dict[str, float]] = weakref.WeakKeyDictionary()
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
        point = await asyncio.wait_for(page.evaluate(
            "data => window.__auraBrowserPresence?.show(data)",
            {"action": action, "stage": stage, "box": box, "point": _last_point_of(browser)},
        ), timeout=.5)
        if isinstance(point, dict) and {"x", "y"} <= set(point):
            try:
                _LAST_POINT[browser] = {"x": float(point["x"]), "y": float(point["y"])}
            except TypeError:
                pass  # not a failure: a browser object that cannot be weakly referenced keeps no pointer
        _VISIBLE_UNTIL[page] = 0.0 if action == "closed" else math.inf
    except (*_ERRORS, _PageError) as why:
        record_degradation("browser_presence", why, severity="debug", action="continued without the browser activity overlay")


def _last_point_of(browser: Any) -> dict[str, float] | None:
    try:
        return _LAST_POINT.get(browser)
    except TypeError:
        return None


@asynccontextmanager
async def without_presence(page: Any, clip: dict | None = None):
    """Reference-counted suspension, including when a screenshot fails.

    ``clip`` is the part of the window the picture takes, in CSS pixels; only what of the mark falls inside it is
    hidden, so a picture of the video she is watching does not make her hue blink. None hides the whole mark."""
    # Most play frames have no mark to hide. They need no extra browser round trips. A page that cannot be weakly
    # referenced never had a mark set on it (the same map holds the marks), so it has none to hide; asked anyway, the
    # lookup raised and took the screenshot with it.
    try:
        visible_until = _VISIBLE_UNTIL.get(page, 0.0)
    except TypeError:
        visible_until = 0.0
    if visible_until < time.monotonic():
        yield
        return
    suspended = False
    try:
        suspended = bool(await asyncio.wait_for(page.evaluate(
            "clip => window.__auraBrowserPresence?.suspend(clip) || false", clip
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

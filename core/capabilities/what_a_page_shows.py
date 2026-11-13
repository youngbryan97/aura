"""Everything a page shows once its scripts have run, as her browser renders it, read whole.

Her browser's `read_content` keeps the best-scoring article container and falls back to the page's body only below
200 characters, and her research pipeline renders a page only when its plain fetch came back under 1,000 characters.
Both are right for an article and wrong for a page of readings: LIVE 12 November 2026, "Can you look up what the
weather is in Paris right now?" found Weather.com, whose fetch was 1,152 characters of navigation (read, by that rule)
with no temperature, and whose rendered article container held none either; the values are in the page's widgets.
Bryan: she should absolutely be able to pull the values.

This reads the whole visible text of the rendered page, under the same browser authority as any read.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.Capabilities.WhatAPageShows")


#: A page's text is read again every half second until two reads agree. A page that never stops changing (ads, live
#: tickers) is read as it stands after five seconds. LIVE 12 November 2026, Weather.com: 1,310 characters right after
#: parsing, then 3,266 with "Now 46°, Fair, Feels Like 40°" within three seconds; waiting for its network to go quiet
#: ran out 15 s and never did.
_SETTLE_EVERY_S, _SETTLE_READS = 0.5, 10


async def _once_it_settles(page: Any, gateway: Any, page_script: Any) -> str:
    """The page's visible text once it has stopped changing, or as it stands when it never does."""
    import asyncio

    script = page_script("() => document.body ? document.body.innerText : ''")
    text = str(await gateway.read(page, script) or "")
    for _ in range(_SETTLE_READS):
        await asyncio.sleep(_SETTLE_EVERY_S)
        later = str(await gateway.read(page, script) or "")
        if later and later == text:
            return later
        text = later or text
    return text


#: Her browser's cleaner drops lines under this many characters as noise (core/capabilities/phantom_browser.py),
#: which is right for an article and drops a page of readings whole: "Now", "46°", "Fair", "Feels Like", "High".
_PROSE_LINE_CHARS = 18


def _as_rows(text: str) -> str:
    """The page's lines, with each run of short ones (labels and their readings) kept together on one row."""
    rows: list[str] = []
    short: list[str] = []
    for raw in str(text or "").splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if len(line) < _PROSE_LINE_CHARS:
            short.append(line)
            continue
        if short:
            rows.append(" · ".join(short))
            short = []
        rows.append(line)
    if short:
        rows.append(" · ".join(short))
    return "\n".join(rows)


async def everything_a_page_shows(url: str, *, principal: str = "world_evidence") -> dict[str, Any] | None:
    """The page at ``url`` rendered, and all of its visible text; None when it would not load or may not be read."""
    from core.capabilities.browser_authority import BrowserAction, authorize_browser_action
    from core.capabilities.phantom_browser import PhantomBrowser
    from core.runtime.page_gateway import get_page_gateway, page_script

    browser = PhantomBrowser(visible=False, principal=principal)
    try:
        if not await browser.ensure_ready() or not await browser.browse(url) or browser.page is None:
            return None
        shown_at = str(getattr(browser.page, "url", "") or url)
        if not authorize_browser_action(BrowserAction.READ, principal=principal, url=shown_at).allowed:
            return None
        text = _as_rows(await _once_it_settles(browser.page, get_page_gateway(), page_script))
        title = str(await browser.page.title() or url)
        return {"url": shown_at, "title": title, "text": text} if text.strip() else None
    except Exception as exc:  # noqa: BLE001 - one page is optional, and playwright's errors are its own class
        logger.info("The page did not render to be read (%s): %s", url, exc)
        return None
    finally:
        await browser.close()

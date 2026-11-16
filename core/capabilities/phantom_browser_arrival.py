"""Where her browser arrived: the page it was sent to, a block page in its place, or a document it became.

Lifted out of core/capabilities/phantom_browser.py, which passed its 2,000-line ceiling when a navigation answered
with a file began to be read as the page asked for (27bafd1eb, 16 November 2026). PhantomBrowser carries these as a
mixin, as it carries reading its frames and acting on the page.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from core.capabilities.browser_authority import BrowserAction, authorize_browser_action
from core.runtime.errors import record_degradation
from core.runtime.page_gateway import get_page_gateway, page_script

try:
    from playwright.async_api import Error as PlaywrightError
except ImportError:  # the browser is optional; its errors are runtime errors without it
    PlaywrightError = RuntimeError

logger = logging.getLogger("PhantomBrowser")

#: How long a navigation may take, and so how long the file a navigation became may take to arrive.
_NAVIGATION_MS = 30000


class _ArrivesWhereItWasSent:
    """Whether a navigation reached what it was sent to, and what it is when it became a document."""

    async def is_blocked(self) -> bool:
        """Detect if we are hitting a bot-detection page or CAPTCHA."""
        if not self.page:
            return False

        # What the page shows, as a person reads it, not its source: a page with a
        # CAPTCHA widget somewhere in its code for a comment form is not a block.
        # LIVE 2026-10-06 the Internet Archive's page for a game was refused as
        # "captcha" for a script it carries for reviews.
        try:
            content = str(await get_page_gateway().read(self.page, page_script("document.body ? document.body.innerText.slice(0, 4000) : ''"))).lower()
        except (PlaywrightError, RuntimeError, AttributeError) as exc:
            logger.debug("Could not read what the page shows: %s", exc)
            content = ""
        title = (await self.page.title()).lower()

        block_signals = [
            "unusual traffic from your computer network",
            "not a robot",
            "captcha",
            "verify you are a human",
            "verifying you are human",
            "access to this page has been denied",
            "security check",
            "bot detection",
            "automated requests",
            "bots use duckduckgo too",
            "complete the following challenge",
            "you have been blocked",
            "you've been blocked",
        ]
        if title.strip() in ("just a moment...", "attention required! | cloudflare"):
            logger.warning("🚨 Browser Blocked Detected: %s", title)
            return True

        for signal in block_signals:
            if signal in content or signal in title:
                logger.warning("🚨 Browser Blocked Detected: %s", signal)
                return True
        return False

    async def _arrive_at_a_document(self, requested: str, download: Any, *, principal: str) -> bool:
        """A navigation the browser answered with a file: read as the page asked for, when it is a document.

        The file is read as her attachments are (core/conversation/attachments.read_text), within their size
        limit, and its address passes the same policy a redirect does.
        """
        from core.conversation.attachments import MOST_BYTES, AttachmentRefused, read_text

        final_url = str(getattr(download, "url", "") or requested)
        name = str(getattr(download, "suggested_filename", "") or Path(final_url).name or "download")
        record: dict[str, Any] = {
            "ok": False, "reason": "", "requested_url": requested, "final_url": final_url, "status": None,
            "redirected": final_url.rstrip("/") != requested.rstrip("/"), "blocked": False, "at": time.time(),
            "document": name,
        }
        self._last_navigation = record
        try:
            if record["redirected"]:
                revalidated = authorize_browser_action(BrowserAction.NAVIGATE, principal=principal, url=final_url)
                if not revalidated.allowed:
                    record["reason"] = f"redirect refused: {revalidated.reason}"
                    return False
            path = Path(await asyncio.wait_for(download.path(), timeout=_NAVIGATION_MS / 1000.0))
            if await asyncio.to_thread(lambda: path.stat().st_size) > MOST_BYTES:
                record["reason"] = "document_larger_than_she_reads"
                return False
            kind, text = await asyncio.to_thread(read_text, name, await asyncio.to_thread(path.read_bytes))
        except (TimeoutError, OSError, RuntimeError, ValueError, AttachmentRefused, PlaywrightError) as exc:
            record_degradation('phantom_browser', exc, severity="info",
                               action=f"a download from {final_url} was not read as a document")
            record["reason"] = f"document_unread:{type(exc).__name__}"
            return False
        if not text.strip():
            record["reason"] = f"download_without_text:{kind}"
            return False
        self._document = {"url": final_url, "kind": kind, "title": name, "text": text,
                          "page_url": str(getattr(self.page, "url", "") or "")}
        record.update(ok=True, reason="arrived_at_a_document", kind=kind)
        logger.info("🌐 %s is a %s; read %d characters of it", final_url, kind, len(text))
        return True

    def _document_on_the_page(self) -> dict[str, Any] | None:
        """The document the last navigation became, while the page has not gone anywhere since."""
        document = self._document
        if document and document.get("page_url") == str(getattr(self.page, "url", "") or ""):
            return document
        return None

    async def _verify_arrival(
        self, requested: str, response: Any, *, principal: str
    ) -> dict[str, Any]:
        """Did we land where we asked, on a page that will answer?"""
        record: dict[str, Any] = {
            "ok": False,
            "reason": "",
            "requested_url": requested,
            "final_url": "",
            "status": None,
            "redirected": False,
            "blocked": False,
            "at": time.time(),
        }
        try:
            final_url = str(getattr(self.page, "url", "") or "")
            record["final_url"] = final_url
            status = getattr(response, "status", None)
            record["status"] = int(status) if isinstance(status, int) else None
            record["redirected"] = bool(final_url and final_url.rstrip("/") != requested.rstrip("/"))

            if record["redirected"] and final_url:
                # A redirect is a NEW destination and gets the same policy
                # the original did. Without this, a permitted URL that 302s
                # to a private address defeats the whole check.
                revalidated = authorize_browser_action(
                    BrowserAction.NAVIGATE, principal=principal, url=final_url
                )
                if not revalidated.allowed:
                    record["reason"] = f"redirect refused: {revalidated.reason}"
                    return record

            if record["status"] is not None and record["status"] >= 400:
                record["reason"] = f"http_{record['status']}"
                return record

            if await self.is_blocked():
                record["blocked"] = True
                record["reason"] = "bot_block_or_captcha"
                return record

            record["ok"] = True
            record["reason"] = "arrived"
            return record
        except (RuntimeError, AttributeError, TypeError, ValueError, PlaywrightError) as exc:
            record_degradation('phantom_browser', exc, severity="info")
            record["reason"] = f"arrival_check_failed:{type(exc).__name__}"
            return record

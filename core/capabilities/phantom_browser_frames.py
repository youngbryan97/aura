"""The words in the frames of a page, read as part of the page a person reads.

Lifted from PhantomBrowser (core/capabilities/phantom_browser.py), which had grown past the size a module may have.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from core.capabilities.browser_authority import BrowserAction, authorize_browser_action
from core.runtime.errors import record_degradation

# The same optional-dependency guard the module this was lifted from carries.
try:
    from playwright.async_api import Error as PlaywrightError
except ImportError:  # pragma: no cover - exercised where playwright is absent
    PlaywrightError = RuntimeError


class _ReadsItsFrames:
    """Lifted whole from PhantomBrowser; see phantom_browser.py."""

    async def _what_its_frames_say(self, principal: str) -> list[dict[str, Any]]:
        """The words in each frame a person can see on this page, top to bottom.

        LIVE 2026-10-02 23:40: the OEJTS results page says "A detailed
        description of this personality type is below", and the description is
        a text file shown in a frame. Only the main document was read, so she
        pressed "more" and "less" and scrolled for fifteen minutes looking for
        words that were on her screen the whole time. A frame is part of the
        page a person reads. Who serves it does not change that, so each one is
        read under the verdict for its own address.
        """
        page = self.page
        found: list[dict[str, Any]] = []
        for frame in list(getattr(page, "frames", None) or []):
            try:
                if frame is page.main_frame or frame.is_detached():
                    continue
                holder = await frame.frame_element()
                box = await holder.bounding_box() if await holder.is_visible() else None
                verdict = authorize_browser_action(
                    BrowserAction.READ, principal=principal, url=str(frame.url or "")
                )
                if not box or not verdict.allowed:
                    continue
                said = await frame.evaluate("() => document.body ? document.body.innerText : ''")
            except (PlaywrightError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                # A frame that went away while it was read: an advert
                # reloading, a widget replaced. The rest of the page stands.
                record_degradation("phantom_browser.frames", exc, severity="info")
                continue
            said = re.sub(r"\n{3,}", "\n\n", str(said or "")).strip()
            if said:
                where = urlparse(str(frame.url or ""))
                found.append({**box, "said": said, "where": f"{where.netloc}{where.path}"})
        return sorted(found, key=lambda shown: (shown["y"], shown["x"]))

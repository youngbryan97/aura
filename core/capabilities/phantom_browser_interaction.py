"""Acting on a page rather than reading one.

A click, a typed field, a scroll, and the pause between them that keeps the
rhythm human. Each one writes a receipt of what it did, because an action
nobody recorded is an action she cannot later say she took.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time

from core.capabilities.browser_authority import (
    BrowserAction,
    authorize_browser_action,
)
from core.runtime.errors import (
    record_degradation,
)
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("PhantomBrowser")

# The same optional-dependency guard the module this was lifted from carries.
# Importing the names from there instead would be a cycle: that module imports
# this one to build the class.
try:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
except ImportError:  # pragma: no cover - exercised where playwright is absent
    PlaywrightError = RuntimeError
    PlaywrightTimeoutError = TimeoutError



class _ActsOnThePage:
    """Lifted whole from PhantomBrowser; see phantom_browser.py."""

    async def click(
        self,
        selector: str | None = None,
        text_match: str | None = None,
        *,
        principal: str = "",
        lease_id: str = "",
    ) -> bool:
        """Click an element, under a lease, with the effect recorded.

        A click can buy something, send something, or delete something,
        and this took no principal, no lease and no policy — the same
        boolean-returning interface as reading a page (CP126
        ``a66d2e59``). Success also meant only that Playwright accepted
        the call: no before/after URL, no target identity, no receipt
        (``ed96f557``).
        """
        principal = self._effective_principal(principal)
        verdict = authorize_browser_action(
            BrowserAction.CLICK,
            principal=principal,
            url=str(getattr(self.page, "url", "") or ""),
            lease_id=lease_id,
            target=str(selector or text_match or ""),
        )
        self._last_verdict = verdict.to_dict()
        if not verdict.allowed:
            logger.warning("🖱️ Click refused: %s", verdict.reason)
            return False
        before_url = str(getattr(self.page, "url", "") or "")
        try:
            element = None
            if text_match:
                # Try multiple ways to find text (case-insensitive, contains)
                # Caller text was interpolated into selector strings and then
                # compiled as a regular expression, so a quote, a selector
                # metacharacter or a regex operator changed the matching
                # scope and could click a broader element than the literal
                # label asked for (CP126 ``ae66231a``). Playwright's own
                # quoting handles the literal case; the fuzzy fallback
                # escapes before compiling.
                safe = text_match.replace("\\", "\\\\").replace('"', '\\"')
                selectors = [
                    f'text="{safe}"',
                    f'a:has-text("{safe}")',
                    f'button:has-text("{safe}")',
                    f'*[role="button"]:has-text("{safe}")',
                ]
                for s in selectors:
                    try:
                        loc = self.page.locator(s).first
                        if await loc.is_visible(timeout=2000):
                            element = loc
                            break
                    except PlaywrightError:
                        continue
                
                if not element:
                    # Fallback to get_by_text with regex for fuzzy match
                    import re
                    try:
                        loc = self.page.get_by_text(
                            re.compile(re.escape(text_match), re.IGNORECASE)
                        ).first
                        if await loc.is_visible(timeout=2000):
                            element = loc
                    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                        record_degradation('phantom_browser', exc, severity="debug", action="fuzzy text selector failed")
                        logger.debug("Fuzzy text selector failed: %s", exc)
            elif selector:
                element = self.page.locator(selector).first

            if element and await element.is_visible():
                # Scroll into view if needed
                await element.scroll_into_view_if_needed()
                await self._human_delay(0.2, 0.5)
                
                # Human-like mouse movement
                box = await element.bounding_box()
                if box:
                    x = box['x'] + box['width'] / 2 + random.randint(-5, 5)
                    y = box['y'] + box['height'] / 2 + random.randint(-3, 3)
                    await self.page.mouse.move(x, y, steps=15)
                    await self._human_delay(0.1, 0.3)
                
                # A styled-invisible control still has to be clickable.
                #
                # Playwright's actionability check includes hit-target: the
                # point being clicked must actually receive the event. Sites
                # hide the native input (opacity 0) and paint a custom graphic
                # over it, so the overlay receives the click and the check
                # times out — the element is visible, in view, and unclickable
                # by the ordinary path.
                #
                # MEASURED live 2026-08-18: a questionnaire ran five rounds of
                # observe-decide-act and the page never changed, because every
                # answer click was silently landing on nothing. The loop's
                # stall detector correctly stopped it and reported no_progress,
                # which named the symptom, not this.
                #
                # So an ordinary click first — it respects overlays, consent
                # banners and anything genuinely in the way — and a forced
                # click only when that times out, which is exactly the case
                # where the "obstruction" is the control's own decoration.
                try:
                    await element.click(timeout=4000)
                except PlaywrightError as click_exc:
                    # `force=True` is NOT the answer here: it only skips the
                    # actionability checks and still clicks at a POINT, so the
                    # decoration on top receives it just the same. Measured
                    # against a covered radio: forced click, still unchecked.
                    #
                    # Dispatching to the element itself is what reaches the
                    # control, because it bypasses hit-testing rather than
                    # ignoring it.
                    logger.info(
                        "🖱️ Ordinary click did not land on %s (%s); "
                        "dispatching to the element itself.",
                        selector or text_match,
                        type(click_exc).__name__,
                    )
                    await element.dispatch_event("click")
                logger.info("🖱️ Clicked: %s", selector or text_match)
                await self._human_delay(0.5, 1.5)
                await self._record_interaction(
                    "click", before_url, target=str(selector or text_match or "")
                )
                return True
            else:
                logger.warning("Element not found or not visible: %s", selector or text_match)
                return False
        except (ImportError, AttributeError, RuntimeError, PlaywrightError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Click failed: %s", e)
            return False

    async def type(
        self, selector: str, text: str, *, principal: str = "", lease_id: str = ""
    ) -> bool:
        """Type into a field, under a lease, and read the value back.

        Typing puts the person's words into someone else's form. It took
        no principal and no lease, and "success" meant the keystrokes were
        accepted — there was no read-back proving the field holds what was
        typed (CP126 ``a66d2e59``, ``ed96f557``).
        """
        principal = self._effective_principal(principal)
        verdict = authorize_browser_action(
            BrowserAction.TYPE,
            principal=principal,
            url=str(getattr(self.page, "url", "") or ""),
            lease_id=lease_id,
            target=str(selector or ""),
        )
        self._last_verdict = verdict.to_dict()
        if not verdict.allowed:
            logger.warning("⌨️ Typing refused: %s", verdict.reason)
            return False
        before_url = str(getattr(self.page, "url", "") or "")
        try:
            if not await self.click(selector, principal=principal, lease_id=lease_id):
                return False

            logger.info("Typing %d characters into selector %s", len(text), selector)
            for char in text:
                await self.page.keyboard.type(char)
                # Random typing delay between keystrokes
                await asyncio.sleep(random.uniform(0.05, 0.15))

            await self._human_delay(0.5, 1.0)
            landed = await self._field_value(selector)
            if landed is not None and landed != text:
                # The keystrokes were accepted and the field holds
                # something else — a masked input, a reformatter, a
                # controlled component. Reporting success here is the
                # finding; the caller needs to know.
                await self._record_interaction(
                    "type", before_url, target=selector, verified=False,
                    detail="field value differs from what was typed",
                )
                return False
            await self._record_interaction(
                "type", before_url, target=selector, verified=landed is not None
            )
            return True
        except (RuntimeError, AttributeError, TypeError, ValueError, PlaywrightError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Typing failed: %s", e)
            return False

    async def _field_value(self, selector: str) -> str | None:
        """What the field holds now, or None when it cannot be read."""
        try:
            return str(await self.page.locator(selector).first.input_value(timeout=2000))
        except (RuntimeError, AttributeError, TypeError, ValueError, PlaywrightError):
            return None

    async def _record_interaction(
        self,
        action: str,
        before_url: str,
        *,
        target: str = "",
        verified: bool = True,
        detail: str = "",
    ) -> None:
        """A receipt for one interaction: what changed, and whether it was seen."""
        after_url = str(getattr(self.page, "url", "") or "")
        self._last_interaction = {
            "schema": "aura.capabilities.phantom_browser.interaction.v1",
            "action": action,
            "target": str(target)[:200],
            "url_before": before_url,
            "url_after": after_url,
            "navigated": bool(before_url and after_url and before_url != after_url),
            "postcondition_verified": bool(verified),
            "detail": detail,
            "at": time.time(),
        }

    async def scroll(
        self, direction: str = "down", amount: int = 500, *, principal: str = ""
    ) -> bool:
        """Scroll the page in a human-like manner."""
        principal = self._effective_principal(principal)
        verdict = authorize_browser_action(
            BrowserAction.SCROLL,
            principal=principal,
            url=str(getattr(self.page, "url", "") or ""),
        )
        self._last_verdict = verdict.to_dict()
        if not verdict.allowed:
            logger.warning("Scroll refused: %s", verdict.reason)
            return False
        try:
            steps = 5
            step_amount = amount // steps
            for _ in range(steps):
                if direction == "down":
                    await self.page.mouse.wheel(0, step_amount)
                else:
                    await self.page.mouse.wheel(0, -step_amount)
                await asyncio.sleep(random.uniform(0.1, 0.3))
            await self._human_delay(0.5, 1.0)
            return True
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Scroll failed: %s", e)
            return False

    async def _human_delay(self, min_s=0.5, max_s=1.5):
        """Random delay to simulate human pause, modulated by homeostasis."""
        if self._homeostasis is None:
            try:
                from core.container import ServiceContainer
                self._homeostasis = ServiceContainer.get("homeostatic_coupling", default=None)
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('phantom_browser', e)
                capture_and_log(e, {'module': __name__})
        
        delay_mod = 1.0
        if self._homeostasis:
            mods = self._homeostasis.get_modifiers()
            # Exhaustion (low vitality) makes her move SLOWER
            if mods.overall_vitality < 0.4:
                delay_mod = 2.5 # Significant fatigue delay
            elif mods.overall_vitality < 0.7:
                delay_mod = 1.5 # Mild fatigue delay
            
            # Urgency makes her move FASTER
            if mods.urgency_flag:
                delay_mod *= 0.6
                
        await asyncio.sleep(random.uniform(min_s, max_s) * delay_mod)

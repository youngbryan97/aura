#!/usr/bin/env python3
"""Dynamic Web-Browsing Task Runner.

This script executes a live browser navigation task using Aura's PhantomBrowser
and verifies that she can dynamic-browse, navigate links, click elements,
and extract content/facts from webpages or local fixture services.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from core.capabilities.browser_authority import (
    BrowserAction,
    issue_browser_lease,
    origin_of,
    revoke_browser_lease,
)
from core.capabilities.phantom_browser import PhantomBrowser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DynamicBrowsingRunner")

_BROWSING_RECOVERABLE_ERRORS = (
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
)


async def run_browsing_task(
    start_url: str,
    target_link_text: str | None = None,
    expected_content_keywords: list[str] | None = None,
    click_selector: str | None = None,
    *,
    principal: str = "browsing_task",
) -> dict[str, Any]:
    """Execute a dynamic browsing session using the live PhantomBrowser.

    Names a principal and takes an interaction lease. Every browser method
    now requires both — a click can buy something, and this harness used
    to drive one anonymously (CP126 ``a66d2e59``).
    """
    logger.info("Starting dynamic browsing task for URL: %s", start_url)
    browser = PhantomBrowser(visible=False)
    lease = issue_browser_lease(
        principal=principal,
        origin=origin_of(start_url if start_url.lower().startswith("http") else f"https://{start_url}"),
        actions={BrowserAction.CLICK, BrowserAction.TYPE},
        purpose="dynamic browsing task",
    )
    
    try:
        # Initialize browser
        ready = await browser.ensure_ready()
        if not ready:
            status = browser.get_status()
            logger.error("Failed to initialize browser. Status: %s", status)
            return {
                "ok": False,
                "error": "Browser initialization failed",
                "status": status,
            }
        
        # 1. Browse start URL
        success = await browser.browse(start_url, principal=principal)
        if not success:
            return {"ok": False, "error": f"Failed to navigate to {start_url}"}
        
        # 2. Extract initial content
        initial_content = await browser.read_content()
        logger.info("Successfully navigated to start page. Content length: %d", len(initial_content))
        
        # 3. Handle optional dynamic interactions
        if target_link_text:
            logger.info("Attempting to click link with text: '%s'", target_link_text)
            clicked = await browser.click(text_match=target_link_text, principal=principal, lease_id=lease.lease_id)
            if not clicked:
                logger.warning("Failed to click link using text match. Attempting link traversal via extraction.")
                links = await browser.get_links(principal=principal)
                for link in links:
                    if target_link_text.lower() in link.get("text", "").lower():
                        logger.info("Found matching link URL: %s. Direct traversing.", link["url"])
                        await browser.browse(link["url"], principal=principal)
                        clicked = True
                        break
            if not clicked:
                return {"ok": False, "error": f"Could not navigate to target link: '{target_link_text}'"}
                
        elif click_selector:
            logger.info("Attempting to click selector: '%s'", click_selector)
            clicked = await browser.click(selector=click_selector, principal=principal, lease_id=lease.lease_id)
            if not clicked:
                return {"ok": False, "error": f"Could not click selector: '{click_selector}'"}

        # 4. Extract final content
        #
        # Named, like every other call in this task. Without a principal the
        # read is refused, and a refusal comes back as an empty string — the
        # same value a blank page gives. The task then reported that its
        # keywords were missing from the page, which was true and had nothing
        # to do with the page.
        final_content = await browser.read_content(principal=principal)
        if not final_content:
            refused = browser.last_verdict
            if refused and not refused.get("allowed", True):
                return {
                    "ok": False,
                    "start_url": start_url,
                    "error": f"the page could not be read: {refused.get('reason', '')}",
                    "refused": refused,
                    "verification": {},
                    "content_snippet": "",
                }
        logger.info("Final page content length: %d", len(final_content))
        
        # 5. Verify keywords
        verification_results = {}
        all_matched = True
        if expected_content_keywords:
            for kw in expected_content_keywords:
                matched = kw.lower() in final_content.lower()
                verification_results[kw] = matched
                if not matched:
                    all_matched = False
                    logger.warning("Verification failed: keyword '%s' not found in final content.", kw)
        
        return {
            "ok": all_matched,
            "start_url": start_url,
            "verification": verification_results,
            "content_snippet": final_content[:800],
        }
        
    except _BROWSING_RECOVERABLE_ERRORS as e:
        logger.exception("An error occurred during dynamic browsing: %s", e)
        return {"ok": False, "error": str(e)}
    finally:
        revoke_browser_lease(lease.lease_id)
        await browser.close()
        logger.info("Phantom Browser closed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./run_dynamic_browsing_task.py <url> [target_link_text] [expected_keywords_comma_separated]")
        sys.exit(1)
        
    url = sys.argv[1]
    link_text = sys.argv[2] if len(sys.argv) > 2 else None
    kws = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    
    res = asyncio.run(run_browsing_task(url, link_text, kws))
    print("\nResult:")
    print(res)
    sys.exit(0 if res["ok"] else 1)

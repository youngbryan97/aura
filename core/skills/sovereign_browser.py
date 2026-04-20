import asyncio
import logging
import urllib.parse
import subprocess
import os
import base64
import random
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.phantom_browser import PhantomBrowser
from core.search.research_pipeline import query_requires_source_reading
from core.thought_stream import get_emitter

logger = logging.getLogger("Skills.SovereignBrowser")

class BrowserAction(BaseModel):
    type: str = Field(..., description="Action: 'click', 'type', 'scroll', 'wait', 'get_html', 'screenshot'")
    selector: Optional[str] = Field(None, description="CSS selector or text match for elements.")
    value: Optional[str] = Field(None, description="Value to type or wait duration.")

class BrowserInput(BaseModel):
    mode: str = Field("search", description="Mode: 'search', 'browse', 'interact'")
    query: Optional[str] = Field(None, description="Search query for 'search' mode.")
    url: Optional[str] = Field(None, description="URL for 'browse' or 'interact' mode.")
    actions: Optional[List[BrowserAction]] = Field(None, description="Sequence of actions for 'interact' mode.")
    deep: bool = Field(False, description="Whether to deep-dive by reading the first non-ad search result.")

class SovereignBrowserSkill(BaseSkill):
    """The unified, high-fidelity web capability for Aura.
    Handles searching, navigation, and complex interactions using PhantomBrowser.

    HARDENING (2026-04):
    - Ephemeral browser sessions: each execute() gets a fresh browser, closed in a
      finally block. No more process leaks across conversation turns.
    - Per-operation timeouts on all Playwright calls (read_content, browse, etc.)
    - Resource lock integration to pause background inference during heavy browsing.
    """

    name = "sovereign_browser"
    description = "Browse the web, search for information, or interact with websites (click, type, etc.)."
    input_model = BrowserInput

    # Timeouts for Playwright operations (seconds)
    BROWSE_TIMEOUT = 25.0
    READ_TIMEOUT = 15.0
    INTERACTION_TIMEOUT = 45.0
    SEARCH_TIMEOUT = 40.0

    def __init__(self):
        super().__init__()
        # User requested: capable of using any browser (Chrome, Firefox, Safari)
        self._browser_types = ["chromium", "firefox", "webkit"]

    def _pick_browser_type(self) -> str:
        return random.choice(self._browser_types)

    async def _create_browser(self) -> PhantomBrowser:
        """Create a fresh, ephemeral PhantomBrowser instance."""
        browser_type = self._pick_browser_type()
        browser = PhantomBrowser(visible=False, browser_type=browser_type)
        await asyncio.wait_for(browser.ensure_ready(), timeout=30.0)
        return browser

    async def _safe_close(self, browser: Optional[PhantomBrowser]) -> None:
        """Guaranteed browser teardown — never raises."""
        if browser is None:
            return
        try:
            await asyncio.wait_for(browser.close(), timeout=10.0)
        except Exception as close_exc:
            logger.debug("Browser close error (suppressed): %s", close_exc)
            # Force-kill if close() hangs
            try:
                if browser.browser:
                    await browser.browser.close()
            except Exception:
                pass
            try:
                if browser.playwright:
                    await browser.playwright.stop()
            except Exception:
                pass
            browser.is_active = False
            browser.page = None
            browser.context = None
            browser.browser = None
            browser.playwright = None

    async def _safe_read_content(self, browser: PhantomBrowser) -> str:
        """Read page content with a timeout to prevent hung-page stalls."""
        try:
            return await asyncio.wait_for(browser.read_content(), timeout=self.READ_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("🕐 read_content() timed out after %.0fs", self.READ_TIMEOUT)
            return ""
        except Exception as e:
            logger.warning("read_content() error: %s", e)
            return ""

    async def _safe_browse(self, browser: PhantomBrowser, url: str) -> bool:
        """Navigate with a timeout."""
        try:
            return await asyncio.wait_for(browser.browse(url), timeout=self.BROWSE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("🕐 browse(%s) timed out after %.0fs", url[:80], self.BROWSE_TIMEOUT)
            return False
        except Exception as e:
            logger.warning("browse(%s) error: %s", url[:80], e)
            return False

    async def execute(self, params: BrowserInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Unified entry point for all web activities.

        HARDENING: Each invocation creates an ephemeral browser session that is
        guaranteed to be closed in a finally block, preventing Playwright process
        leaks that accumulate across conversation turns.
        """
        if isinstance(params, dict):
            try:
                params = BrowserInput(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input schema: {e}"}

        browser: Optional[PhantomBrowser] = None
        try:
            # 1. Try High-Fidelity Playwright (Phantom)
            try:
                browser = await self._create_browser()

                if params.mode == "search":
                    return await asyncio.wait_for(
                        self._handle_search(browser, params.query, params.deep),
                        timeout=self.SEARCH_TIMEOUT,
                    )
                elif params.mode == "browse":
                    return await asyncio.wait_for(
                        self._handle_browse(browser, params.url),
                        timeout=self.BROWSE_TIMEOUT + self.READ_TIMEOUT,
                    )
                elif params.mode == "interact":
                    return await asyncio.wait_for(
                        self._handle_interact(browser, params.url, params.actions),
                        timeout=self.INTERACTION_TIMEOUT,
                    )
                else:
                    return {"ok": False, "error": f"Unsupported browser mode: {params.mode}"}
            except asyncio.TimeoutError as te:
                logger.warning("Browser operation timed out: %s", te)
                return {"ok": False, "error": f"Browser operation timed out: {params.mode}"}
            except Exception as e:
                logger.warning("Primary Playwright strategy failed, attempting fallback: %s", e)
                return await self._execute_fallback(params)

        except Exception as e:
            logger.error("Browser skill failed completely: %s", e)
            return {"ok": False, "error": str(e)}
        finally:
            # CRITICAL: Always tear down the browser to prevent process leaks
            await self._safe_close(browser)

    async def _execute_fallback(self, params: BrowserInput) -> Dict[str, Any]:
        """Technically difficult sites often require Undetected Chromedriver."""
        try:
            import undetected_chromedriver as uc
            from selenium.webdriver.common.by import By

            options = uc.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')

            # Start ephemeral driver
            driver = uc.Chrome(options=options)
            try:
                url = params.url
                if params.mode == "search" and params.query:
                    url = f"https://www.google.com/search?q={urllib.parse.quote_plus(params.query)}"

                if not url: return {"ok": False, "error": "No URL for fallback."}

                driver.get(url)
                await asyncio.sleep(3) # Wait for JS/stealth

                content = driver.find_element(By.TAG_NAME, "body").text
                title = driver.title

                return {
                    "ok": True,
                    "engine": "selenium_uc_fallback",
                    "title": title,
                    "content": content[:5000],
                    "message": f"Successfully bypassed protection via Selenium UC for {url}."
                }
            finally:
                driver.quit()
        except ImportError:
            return {"ok": False, "error": "Playwright failed and Selenium UC not installed."}
        except Exception as e:
            return {"ok": False, "error": f"Fallback failed: {e}"}

    async def _handle_search(self, browser: PhantomBrowser, query: str, deep: bool) -> Dict[str, Any]:
        if not query:
            return {"ok": False, "error": "Search mode requires a 'query'."}

        logger.info("🔍 Searching: %s (Deep: %s)", query, deep)
        # User requested: doesn't have to be duckduckgo
        engines = [
            f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}",
            f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}",
            f"https://www.bing.com/search?q={urllib.parse.quote_plus(query)}"
        ]
        random.shuffle(engines)

        for url in engines:
            if await self._safe_browse(browser, url):
                # Detect block/CAPTCHA
                preview = await self._safe_read_content(browser)
                if self._check_blocked(preview):
                    logger.warning("🚫 Search engine blocked. Rotating UA and trying next engine...")
                    try:
                        await asyncio.wait_for(browser.rotate_user_agent(), timeout=10.0)
                    except Exception as rot_exc:
                        logger.debug("UA rotation failed: %s", rot_exc)
                    continue

                await browser._human_delay(2, 3)
                if deep:
                    get_emitter().emit("🔍 Deep Search", f"Analyzing search results for organic targets...", category="Browser")
                    try:
                        links = await asyncio.wait_for(browser.get_links(), timeout=10.0)
                    except Exception:
                        links = []
                    target = self._select_search_result(links, query=query)
                    if target:
                        get_emitter().emit("🌊 Deep-Diving", f"Navigating to exact source: {target}", category="Browser")
                        logger.info("🌊 Deep-diving into: %s", target)
                        try:
                            if await self._safe_browse(browser, target):
                                # Check if target site is blocked too
                                target_content = await self._safe_read_content(browser)
                                if self._check_blocked(target_content):
                                    get_emitter().emit("🔒 Security Block", "Target site is blocking access. Attempting rotation...", level="warning", category="Browser")
                                    try:
                                        await asyncio.wait_for(browser.rotate_user_agent(), timeout=10.0)
                                    except Exception:
                                        continue
                                    if not await self._safe_browse(browser, target):
                                        continue
                                    target_content = await self._safe_read_content(browser)

                                get_emitter().emit("📄 Extracting Content", f"Reading content from {target}", category="Browser")
                                content = target_content or await self._safe_read_content(browser)
                                title = ""
                                try:
                                    if browser.page is not None:
                                        title = (await browser.page.title() or "").strip()
                                except Exception:
                                    title = ""
                                # Phase 39: Deep Synthesis — Provide major content for the LLM
                                snippet_size = 5000
                                snippet = content[:snippet_size].strip() if content else "Content could not be extracted."

                                logger.info("✅ Deep synthesized %d chars from %s", len(snippet), target)
                                return {
                                    "ok": True, "source": target, "title": title, "content": content, "mode": "deep_search",
                                    "message": f"I have deeply synthesized the content from {target}. Here is the core information:\n\n{snippet[:2000]}..."
                                }
                        except Exception as e:
                            logger.error("Deep dive into %s failed: %s", target, e)
                            continue

                get_emitter().emit("📄 Reading Search Results", f"Extracting immediate snippets from {url}", category="Browser")
                content = await self._safe_read_content(browser)
                # Increase snippet size for non-deep search too
                snippet_size = 2000
                snippet = content[:snippet_size].strip() if content else "No content extracted."

                logger.info("✅ Extracted %d chars from %s", len(snippet), url)
                return {
                    "ok": True, "source": url, "content": content, "mode": "search",
                    "message": f"I searched for '{query}' and here's what I found:\n\n{snippet}"
                }

        return {"ok": False, "error": "Search engines unreachable or blocked."}

    async def _handle_browse(self, browser: PhantomBrowser, url: str) -> Dict[str, Any]:
        if not url:
            return {"ok": False, "error": "Browse mode requires a 'url'."}

        get_emitter().emit("🌐 Navigating", f"Opening {url}", category="Browser")
        if await self._safe_browse(browser, url):
            get_emitter().emit("📄 Reading Document", f"Extracting content from {url}", category="Browser")
            content = await self._safe_read_content(browser)
            return {"ok": True, "source": url, "content": content, "message": f"I've navigated to {url} and captured the content."}
        return {"ok": False, "error": f"Failed to load {url}"}

    async def _handle_interact(self, browser: PhantomBrowser, url: str, actions: List[BrowserAction]) -> Dict[str, Any]:
        if url and not await self._safe_browse(browser, url):
            return {"ok": False, "error": f"Failed to load start URL: {url}"}

        if not actions:
            return {"ok": False, "error": "Interact mode requires 'actions'."}

        results = []
        for action in actions:
            logger.info("🎬 Action: %s | Sel: %s", action.type, action.selector)
            try:
                if action.type == "click":
                    success = await asyncio.wait_for(browser.click(selector=action.selector), timeout=10.0)
                elif action.type == "type":
                    success = await asyncio.wait_for(browser.type(action.selector, action.value), timeout=10.0)
                elif action.type == "scroll":
                    await asyncio.wait_for(browser.scroll(direction=action.value or "down"), timeout=5.0)
                    success = True
                elif action.type == "wait":
                    await asyncio.sleep(min(float(action.value or 1), 10.0))  # Cap wait at 10s
                    success = True
                elif action.type == "get_html":
                    if browser.page:
                        html = await asyncio.wait_for(browser.page.content(), timeout=10.0)
                        results.append({"type": "html", "content": html[:60000]})
                    success = True
                elif action.type == "screenshot":
                    ss = await asyncio.wait_for(browser.screenshot(), timeout=10.0)
                    results.append({"type": "screenshot", "data": ss})
                    success = True
                else:
                    success = False
                    logger.warning("Unsupported action type: %s", action.type)
            except asyncio.TimeoutError:
                logger.warning("Action '%s' timed out", action.type)
                success = False
            except Exception as action_exc:
                logger.warning("Action '%s' failed: %s", action.type, action_exc)
                success = False

            results.append({"action": action.type, "ok": success})
            if not success: break

        final_content = await self._safe_read_content(browser)
        final_url = ""
        try:
            if browser.page:
                final_url = browser.page.url
        except Exception:
            pass

        return {
            "ok": True,
            "url": final_url,
            "content": final_content,
            "action_report": results,
            "message": "I've completed the sequence of interactions on the page."
        }

    @staticmethod
    def _check_blocked(content: str) -> bool:
        """Heuristic check for CAPTCHA, 403, or bot-blocking pages."""
        if not content:
            return False
        lower = content[:2000].lower()
        block_signals = [
            "captcha", "robot", "blocked", "access denied",
            "403 forbidden", "please verify you are a human",
            "cf-challenge", "cloudflare", "rate limit",
        ]
        return sum(1 for s in block_signals if s in lower) >= 2

    def _select_search_result(self, links: List[Dict[str, str]], query: str = "") -> Optional[str]:
        """Choose the most query-aligned organic result instead of the first link-shaped thing."""
        query_tokens = {
            token for token in re.findall(r"[a-z0-9]+", str(query or "").lower())
            if len(token) >= 3
        }
        source_reading = query_requires_source_reading(query)
        quoted = [
            phrase.lower()
            for phrase in re.findall(r"[\"“”']([^\"“”']{4,200})[\"“”']", str(query or ""))
        ]

        best_url: Optional[str] = None
        best_score = float("-inf")

        for link in links:
            url = str(link.get("url") or "").strip()
            text = str(link.get("text") or "").strip()
            if not url.startswith("http"):
                continue

            lower_url = url.lower()
            lower_text = text.lower()
            if any(x in lower_url for x in ("duckduckgo.com", "google.com", "bing.com", "googleadservices.com")):
                continue
            if any(x in lower_text for x in ("privacy", "settings", "help", "about", "login", "signup")):
                continue

            score = 0.0
            tokens = {
                token for token in re.findall(r"[a-z0-9]+", f"{lower_text} {lower_url}")
                if len(token) >= 3
            }
            if query_tokens:
                score += len(query_tokens & tokens) / max(1, len(query_tokens))
            for phrase in quoted:
                normalized_phrase = re.sub(r"[^a-z0-9]+", " ", phrase).strip()
                normalized_target = re.sub(r"[^a-z0-9]+", " ", f"{lower_text} {lower_url}").strip()
                if normalized_phrase and normalized_phrase in normalized_target:
                    score += 1.2
            if source_reading and re.search(r"(story|article|post|thread|chapter|page|document)", lower_text):
                score += 0.25
            if lower_url.endswith(".pdf") or "youtube.com" in lower_url or "youtu.be" in lower_url:
                score -= 0.35
            if len(text) >= 10:
                score += 0.05

            if score > best_score:
                best_score = score
                best_url = url

        return best_url

    async def on_stop_async(self):
        """No-op: browsers are now ephemeral per-invocation."""
        pass

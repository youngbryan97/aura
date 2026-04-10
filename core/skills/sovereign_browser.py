import asyncio
import logging
import urllib.parse
import subprocess
import os
import base64
import random
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.phantom_browser import PhantomBrowser
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
    """
    
    name = "sovereign_browser"
    description = "Browse the web, search for information, or interact with websites (click, type, etc.)."
    input_model = BrowserInput
    
    def __init__(self):
        super().__init__()
        # User requested: capable of using any browser (Chrome, Firefox, Safari)
        engines = ["chromium", "firefox", "webkit"]
        self.browser_type = random.choice(engines)
        self.browser = PhantomBrowser(visible=False, browser_type=self.browser_type)

    async def execute(self, params: BrowserInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Unified entry point for all web activities."""
        if isinstance(params, dict):
            try:
                params = BrowserInput(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input schema: {e}"}

        try:
            # 1. Try High-Fidelity Playwright (Phantom)
            try:
                if not self.browser.is_active:
                    await self.browser.ensure_ready()
                
                if params.mode == "search":
                    return await self._handle_search(params.query, params.deep)
                elif params.mode == "browse":
                    return await self._handle_browse(params.url)
                elif params.mode == "interact":
                    return await self._handle_interact(params.url, params.actions)
                else:
                    return {"ok": False, "error": f"Unsupported browser mode: {params.mode}"}
            except Exception as e:
                logger.warning("Primary Playwright strategy failed, attempting Selenium/UC fallback: %s", e)
                return await self._execute_fallback(params)
                
        except Exception as e:
            logger.error("Browser skill failed completely: %s", e)
            return {"ok": False, "error": str(e)}

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

    async def _handle_search(self, query: str, deep: bool) -> Dict[str, Any]:
        if not query:
            return {"ok": False, "error": "Search mode requires a 'query'."}
            
        logger.info("🔍 Searching: %s (Deep: %s)", query, deep)
        # Phase 36: Always do deep search for real content
        import random
        # User requested: doesn't have to be duckduckgo
        engines = [
            f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}",
            f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}",
            f"https://www.bing.com/search?q={urllib.parse.quote_plus(query)}"
        ]
        random.shuffle(engines)
        
        for url in engines:
            if await self.browser.browse(url):
                # Detect block/CAPTCHA
                if self._check_blocked(await self.browser.read_content() if hasattr(self.browser, 'read_content') else ""):
                    logger.warning("🚫 Search engine blocked. Rotating UA and trying next engine...")
                    await self.browser.rotate_user_agent()
                    continue

                await self.browser._human_delay(2, 3)
                if deep:
                    get_emitter().emit("🔍 Deep Search", f"Analyzing search results for organic targets...", category="Browser")
                    links = await self.browser.get_links()
                    target = self._select_search_result(links)
                    if target:
                        get_emitter().emit("🌊 Deep-Diving", f"Navigating to exact source: {target}", category="Browser")
                        logger.info("🌊 Deep-diving into: %s", target)
                        try:
                            if await self.browser.browse(target):
                                # Check if target site is blocked too
                                if self._check_blocked(await self.browser.read_content() if hasattr(self.browser, 'read_content') else ""):
                                    get_emitter().emit("🔒 Security Block", "Target site is blocking access. Attempting rotation...", level="warning", category="Browser")
                                    await self.browser.rotate_user_agent()
                                    if not await self.browser.browse(target):
                                        continue
                                get_emitter().emit("📄 Extracting Content", f"Reading content from {target}", category="Browser")
                                content = await self.browser.read_content()
                                # Phase 39: Deep Synthesis — Provide major content for the LLM
                                snippet_size = 5000
                                snippet = content[:snippet_size].strip() if content else "Content could not be extracted."
                                
                                logger.info("✅ Deep synthesized %d chars from %s", len(snippet), target)
                                return {
                                    "ok": True, "source": target, "content": content, "mode": "deep_search",
                                    "message": f"I have deeply synthesized the content from {target}. Here is the core information:\n\n{snippet[:2000]}..."
                                }
                        except Exception as e:
                            logger.error("Deep dive into %s failed: %s", target, e)
                            continue
                
                get_emitter().emit("📄 Reading Search Results", f"Extracting immediate snippets from {url}", category="Browser")
                content = await self.browser.read_content()
                # Increase snippet size for non-deep search too
                snippet_size = 2000
                snippet = content[:snippet_size].strip() if content else "No content extracted."
                
                logger.info("✅ Extracted %d chars from %s", len(snippet), url)
                return {
                    "ok": True, "source": url, "content": content, "mode": "search",
                    "message": f"I searched for '{query}' and here's what I found:\n\n{snippet}"
                }
                
        return {"ok": False, "error": "Search engines unreachable or blocked."}

    async def _handle_browse(self, url: str) -> Dict[str, Any]:
        if not url:
            return {"ok": False, "error": "Browse mode requires a 'url'."}
        
        get_emitter().emit("🌐 Navigating", f"Opening {url}", category="Browser")
        if await self.browser.browse(url):
            get_emitter().emit("📄 Reading Document", f"Extracting content from {url}", category="Browser")
            content = await self.browser.read_content()
            return {"ok": True, "source": url, "content": content, "message": f"I've navigated to {url} and captured the content."}
        return {"ok": False, "error": f"Failed to load {url}"}

    async def _handle_interact(self, url: str, actions: List[BrowserAction]) -> Dict[str, Any]:
        if url and not await self.browser.browse(url):
            return {"ok": False, "error": f"Failed to load start URL: {url}"}
            
        if not actions:
            return {"ok": False, "error": "Interact mode requires 'actions'."}

        results = []
        for action in actions:
            logger.info("🎬 Action: %s | Sel: %s", action.type, action.selector)
            if action.type == "click":
                success = await self.browser.click(selector=action.selector)
            elif action.type == "type":
                success = await self.browser.type(action.selector, action.value)
            elif action.type == "scroll":
                await self.browser.scroll(direction=action.value or "down")
                success = True
            elif action.type == "wait":
                await asyncio.sleep(float(action.value or 1))
                success = True
            elif action.type == "get_html":
                results.append({"type": "html", "content": await self.browser.page.content()})
                success = True
            elif action.type == "screenshot":
                results.append({"type": "screenshot", "data": await self.browser.screenshot()})
                success = True
            else:
                success = False
                logger.warning("Unsupported action type: %s", action.type)
            
            results.append({"action": action.type, "ok": success})
            if not success: break

        return {
            "ok": True, 
            "url": self.browser.page.url, 
            "content": await self.browser.read_content(),
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

    def _select_search_result(self, links: List[Dict[str, str]]) -> Optional[str]:
        """Heuristic to find the first real organic search result."""
        for link in links:
            u = link.get('url', '')
            text = link.get('text', '').lower()
            if any(x in u for x in ['duckduckgo.com', 'google.com', 'bing.com', 'googleadservices.com']): continue
            if any(x in text for x in ['privacy', 'settings', 'help', 'about', 'login', 'signup']): continue
            if not u.startswith('http'): continue
            if len(text) < 10: continue
            return u
        return None

    async def on_stop_async(self):
        await self.browser.close()
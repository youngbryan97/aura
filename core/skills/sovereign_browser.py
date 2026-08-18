import asyncio
import hashlib
import logging
import random
import re
import time
import urllib.parse
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from core.capabilities.browser_authority import (
    BrowserAction as AuthorityAction,
    issue_browser_lease,
    origin_of,
    revoke_browser_lease,
)
from core.capabilities.phantom_browser import PhantomBrowser
from core.governance_context import get_active_governance
from core.governance.will import ActionDomain
from core.runtime.action_executor import ActionExecutor
from core.runtime.errors import record_degradation
from core.runtime.skill_contract import ActionExpectation
from core.search.research_pipeline import query_requires_source_reading
from core.skills.base_skill import BaseSkill
from core.thought_stream import get_emitter

logger = logging.getLogger("Skills.SovereignBrowser")


def _read_comprehension(*, url: str, title: str, text: str) -> dict[str, Any]:
    """What this page claims, judged — merged into the browse result.

    A browse Bryan asked for by name went through the one read path that
    returned raw characters and nothing about them, so the turn after it had
    nothing to say about the page.
    """

    try:
        from core.knowledge.source_comprehension import comprehension_payload
    except ImportError:
        return {}
    return comprehension_payload(url=url, title=title, text=text)

class BrowserAction(BaseModel):
    type: Literal["click", "type", "scroll", "wait", "get_html", "screenshot"] = Field(
        ...,
        description="Browser action type.",
    )
    selector: str | None = Field(None, description="CSS selector or text match for elements.")
    value: str | None = Field(None, description="Value to type or wait duration.")

    @model_validator(mode="after")
    def validate_action_contract(self) -> Self:
        if self.type in {"click", "type"} and not str(self.selector or "").strip():
            raise ValueError(f"browser action {self.type} requires a selector")
        if self.type == "type" and not str(self.value or ""):
            raise ValueError("browser action type requires a non-empty value")
        if self.type == "wait":
            try:
                delay = float(self.value or "")
            except (TypeError, ValueError) as exc:
                raise ValueError("browser wait action requires a numeric value") from exc
            if not 0.0 <= delay <= 10.0:
                raise ValueError("browser wait duration must be between 0 and 10 seconds")
        if self.type == "scroll" and str(self.value or "down").lower() not in {"up", "down"}:
            raise ValueError("browser scroll direction must be 'up' or 'down'")
        return self

class BrowserInput(BaseModel):
    mode: Literal["search", "browse", "interact"] = Field(
        "search",
        description="Browser operation mode.",
    )
    query: str | None = Field(None, description="Search query for 'search' mode.")
    url: str | None = Field(None, description="URL for 'browse' or 'interact' mode.")
    actions: list[BrowserAction] | None = Field(None, description="Sequence of actions for 'interact' mode.")
    deep: bool = Field(False, description="Whether to deep-dive by reading the first non-ad search result.")
    browser_type: Literal["auto", "chromium", "firefox", "webkit"] = "auto"

    @model_validator(mode="after")
    def validate_mode_contract(self) -> Self:
        if self.mode == "search":
            self.query = str(self.query or "").strip()
            if not self.query:
                raise ValueError("search mode requires a non-empty query")
        if self.mode in {"browse", "interact"}:
            url = str(self.url or "").strip()
            if not url:
                raise ValueError(f"{self.mode} mode requires a URL")
            if "://" not in url:
                url = "https://" + url
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("browser URL must be an absolute HTTP(S) URL")
            self.url = url
        if self.mode == "interact" and not self.actions:
            raise ValueError("interact mode requires at least one action")
        return self

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

    def __init__(self) -> None:
        super().__init__()
        self._browser_types = ["chromium", "firefox", "webkit"]

    def _pick_browser_type(self, preference: str = "auto") -> str:
        return preference if preference in self._browser_types else "chromium"

    async def _create_browser(self, preference: str = "auto") -> PhantomBrowser:
        """Create a fresh, ephemeral PhantomBrowser instance."""
        browser_type = self._pick_browser_type(preference)
        browser = PhantomBrowser(
            visible=False,
            browser_type=browser_type,
            principal="sovereign_browser",
        )
        ready = await asyncio.wait_for(browser.ensure_ready(), timeout=30.0)
        if not ready:
            status = browser.get_status()
            raise RuntimeError(
                "browser startup failed: " + str(status.get("startup_error") or "not active")
            )
        return browser

    async def _safe_close(self, browser: PhantomBrowser | None) -> None:
        """Guaranteed browser teardown — never raises."""
        if browser is None:
            return
        try:
            await asyncio.wait_for(browser.close(), timeout=10.0)
        except (RuntimeError, TimeoutError, AttributeError) as close_exc:
            record_degradation('sovereign_browser', close_exc)
            logger.debug("Browser close error (suppressed): %s", close_exc)
            # Force-kill if close() hangs
            try:
                if browser.browser:
                    await browser.browser.close()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass  # no-op: intentional
            try:
                if browser.playwright:
                    await browser.playwright.stop()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass  # no-op: intentional
            browser.is_active = False
            browser.page = None
            browser.context = None
            browser.browser = None
            browser.playwright = None

    async def _safe_read_content(self, browser: PhantomBrowser) -> str:
        """Read page content with a timeout to prevent hung-page stalls."""
        try:
            return await asyncio.wait_for(browser.read_content(), timeout=self.READ_TIMEOUT)
        except TimeoutError:
            logger.warning("🕐 read_content() timed out after %.0fs", self.READ_TIMEOUT)
            return ""
        except (RuntimeError, AttributeError) as e:
            record_degradation('sovereign_browser', e)
            logger.warning("read_content() error: %s", e)
            return ""

    async def _safe_browse(self, browser: PhantomBrowser, url: str) -> bool:
        """Navigate with a timeout."""
        try:
            return await asyncio.wait_for(browser.browse(url), timeout=self.BROWSE_TIMEOUT)
        except TimeoutError:
            logger.warning("🕐 browse(%s) timed out after %.0fs", url[:80], self.BROWSE_TIMEOUT)
            return False
        except (RuntimeError, AttributeError) as e:
            record_degradation('sovereign_browser', e)
            logger.warning("browse(%s) error: %s", url[:80], e)
            return False

    def _execution_timeout(self, mode: str) -> float:
        operation_timeout = {
            "search": self.SEARCH_TIMEOUT,
            "browse": self.BROWSE_TIMEOUT + self.READ_TIMEOUT,
            "interact": self.INTERACTION_TIMEOUT,
        }.get(mode, self.INTERACTION_TIMEOUT)
        return 30.0 + operation_timeout + 15.0

    @staticmethod
    def _observed_url(browser: PhantomBrowser) -> str:
        try:
            if browser.page is not None:
                return str(browser.page.url or "")[:2048]
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return ""
        return ""

    @staticmethod
    def _verify_browser_effect(context: Mapping[str, Any]) -> dict[str, Any]:
        raw_result = context.get("result")
        raw_params = context.get("params")
        result = dict(raw_result) if isinstance(raw_result, Mapping) else {}
        params = dict(raw_params) if isinstance(raw_params, Mapping) else {}
        mode = str(params.get("mode") or "").strip().lower()
        observed_url = str(result.get("observed_url") or result.get("url") or "")
        try:
            parsed = urllib.parse.urlsplit(observed_url)
            url_observed = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        except ValueError:
            url_observed = False

        content = result.get("content")
        content_text = content if isinstance(content, str) else ""
        action_report = result.get("action_report")
        action_rows = (
            [
                row
                for row in action_report
                if isinstance(row, Mapping) and row.get("action") is not None
            ]
            if isinstance(action_report, list)
            else []
        )
        expected_actions = params.get("actions")
        expected_action_count = len(expected_actions) if isinstance(expected_actions, list) else 0
        actions_verified = bool(
            expected_action_count > 0
            and len(action_rows) == expected_action_count
            and all(row.get("ok") is True for row in action_rows)
        )
        navigation_confirmed = result.get("navigation_confirmed") is True
        if mode in {"search", "browse"}:
            effect_verified = bool(
                result.get("ok") is True
                and navigation_confirmed
                and url_observed
                and content_text.strip()
            )
        elif mode == "interact":
            requested_url = bool(str(params.get("url") or "").strip())
            navigation_ok = not requested_url or (navigation_confirmed and url_observed)
            effect_verified = bool(result.get("ok") is True and actions_verified and navigation_ok)
        else:
            effect_verified = False

        content_hash = (
            hashlib.sha256(content_text.encode("utf-8", errors="replace")).hexdigest()
            if content_text
            else ""
        )
        return {
            "effect_verified": effect_verified,
            "observation": {
                "kind": "browser_state_readback",
                "mode": mode,
                "observed_url": observed_url[:512],
                "navigation_confirmed": navigation_confirmed,
                "content_chars": len(content_text),
                "content_sha256": content_hash,
                "expected_action_count": expected_action_count,
                "observed_action_count": len(action_rows),
                "actions_verified": actions_verified,
            },
        }

    async def execute(self, params: BrowserInput, context: dict[str, Any]) -> dict[str, Any]:
        """Run browser work through the canonical consequential-action transaction."""
        if isinstance(params, dict):
            try:
                params = BrowserInput(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('sovereign_browser', e)
                return {"ok": False, "error": f"Invalid input schema: {e}"}
        context = dict(context or {})
        if context.get("action_executor_managed_welfare_transaction"):
            return await self._execute_browser(params, action_context=context)

        async def perform_browser_action(action_context: Mapping[str, Any]) -> dict[str, Any]:
            return await self._execute_browser(params, action_context=action_context)

        source = str(context.get("source") or "sovereign_browser.direct")[:240]
        return await ActionExecutor.execute(
            domain=ActionDomain.NETWORK_CALL,
            action_name=f"sovereign_browser.{params.mode}",
            params=params.model_dump(mode="json"),
            source=source,
            predicted_welfare_delta={
                "curiosity": 0.03 if params.mode in {"search", "browse"} else 0.0,
                "caution": 0.04 if params.mode == "interact" else 0.01,
            },
            expectation=ActionExpectation(
                objective=f"complete the requested browser {params.mode} operation",
                acceptance_criteria=["effect_verified"],
                required_evidence=["custom_verifier.observation"],
                user_visible_effect="the requested page state or browser interaction is observed",
                repair_hint="retry with a supported browser engine or a more specific selector",
                rollback_hint="browser interactions may require site-specific compensation",
                allow_partial=True,
            ),
            effect_handler=perform_browser_action,
            effect_verifier=self._verify_browser_effect,
            execution_timeout_s=self._execution_timeout(params.mode),
            verification_timeout_s=5.0,
        )

    async def _execute_browser(
        self,
        params: BrowserInput,
        *,
        action_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one ephemeral browser session inside an established transaction."""

        browser: PhantomBrowser | None = None
        try:
            # 1. Try High-Fidelity Playwright (Phantom)
            try:
                browser = await self._create_browser(params.browser_type)

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
                        self._handle_interact(
                            browser,
                            params.url,
                            params.actions,
                            action_context=action_context,
                        ),
                        timeout=self.INTERACTION_TIMEOUT,
                    )
                else:
                    return {"ok": False, "error": f"Unsupported browser mode: {params.mode}"}
            except TimeoutError as te:
                logger.warning("Browser operation timed out: %s", te)
                return {"ok": False, "error": f"Browser operation timed out: {params.mode}"}
            except (RuntimeError, AttributeError) as e:
                record_degradation('sovereign_browser', e)
                logger.warning("Primary Playwright strategy failed, attempting fallback: %s", e)
                return await self._execute_fallback(params)

        except (RuntimeError, TimeoutError, AttributeError) as e:
            record_degradation('sovereign_browser', e)
            logger.error("Browser skill failed completely: %s", e)
            return {"ok": False, "error": str(e)}
        finally:
            # CRITICAL: Always tear down the browser to prevent process leaks
            await self._safe_close(browser)

    async def _execute_fallback(self, params: BrowserInput) -> dict[str, Any]:
        """Technically difficult sites often require Undetected Chromedriver."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._execute_fallback_sync, params),
                timeout=self.BROWSE_TIMEOUT + 15.0,
            )
        except TimeoutError as exc:
            record_degradation("sovereign_browser", exc)
            return {"ok": False, "error": "Selenium fallback timed out"}

    def _execute_fallback_sync(self, params: BrowserInput) -> dict[str, Any]:
        """Run Selenium's blocking API off the event loop with bounded page loads."""
        try:
            import undetected_chromedriver as uc
            from selenium.common.exceptions import WebDriverException
            from selenium.webdriver.common.by import By
        except ImportError:
            return {"ok": False, "error": "Playwright failed and Selenium UC not installed."}

        driver: Any = None
        try:
            options = uc.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')

            # Start ephemeral driver
            driver = uc.Chrome(options=options)
            driver.set_page_load_timeout(self.BROWSE_TIMEOUT)
            url = params.url
            if params.mode == "search" and params.query:
                url = f"https://www.google.com/search?q={urllib.parse.quote_plus(params.query)}"

            if not url:
                return {"ok": False, "error": "No URL for fallback."}

            driver.get(url)
            time.sleep(1.5)

            content = driver.find_element(By.TAG_NAME, "body").text
            title = driver.title
            observed_url = str(driver.current_url or "")

            return {
                "ok": True,
                "engine": "selenium_uc_fallback",
                "title": title,
                "content": content[:5000],
                "observed_url": observed_url,
                "navigation_confirmed": bool(observed_url),
                "message": f"Successfully bypassed protection via Selenium UC for {url}.",
            }
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, WebDriverException) as e:
            record_degradation('sovereign_browser', e)
            return {"ok": False, "error": f"Fallback failed: {e}"}
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except (AttributeError, OSError, RuntimeError, WebDriverException) as close_error:
                    record_degradation("sovereign_browser", close_error)

    async def _handle_search(
        self,
        browser: PhantomBrowser,
        query: str | None,
        deep: bool,
    ) -> dict[str, Any]:
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
                    except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError) as rot_exc:
                        record_degradation('sovereign_browser', rot_exc)
                        logger.debug("UA rotation failed: %s", rot_exc)
                    continue

                await browser._human_delay(2, 3)
                if deep:
                    get_emitter().emit("🔍 Deep Search", "Analyzing search results for organic targets...", category="Browser")
                    try:
                        links = await asyncio.wait_for(browser.get_links(), timeout=10.0)
                    except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError):
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
                                    except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError):
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
                                except (RuntimeError, AttributeError, TypeError, ValueError):
                                    title = ""
                                # Phase 39: Deep Synthesis — Provide major content for the LLM
                                snippet_size = 5000
                                snippet = content[:snippet_size].strip() if content else "Content could not be extracted."
                                observed_url = self._observed_url(browser)

                                logger.info("✅ Deep synthesized %d chars from %s", len(snippet), target)
                                return {
                                    "ok": True, "source": target, "title": title, "content": content, "mode": "deep_search",
                                    "observed_url": observed_url,
                                    "navigation_confirmed": bool(observed_url),
                                    "message": f"I have deeply synthesized the content from {target}. Here is the core information:\n\n{snippet[:2000]}..."
                                }
                        except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError) as e:
                            record_degradation('sovereign_browser', e)
                            logger.error("Deep dive into %s failed: %s", target, e)
                            continue

                get_emitter().emit("📄 Reading Search Results", f"Extracting immediate snippets from {url}", category="Browser")
                content = await self._safe_read_content(browser)
                # Increase snippet size for non-deep search too
                snippet_size = 2000
                snippet = content[:snippet_size].strip() if content else "No content extracted."
                observed_url = self._observed_url(browser)

                logger.info("✅ Extracted %d chars from %s", len(snippet), url)
                return {
                    "ok": True, "source": url, "content": content, "mode": "search",
                    "observed_url": observed_url,
                    "navigation_confirmed": bool(observed_url),
                    "message": f"I searched for '{query}' and here's what I found:\n\n{snippet}"
                }

        return {"ok": False, "error": "Search engines unreachable or blocked."}

    async def _handle_browse(
        self,
        browser: PhantomBrowser,
        url: str | None,
    ) -> dict[str, Any]:
        if not url:
            return {"ok": False, "error": "Browse mode requires a 'url'."}

        get_emitter().emit("🌐 Navigating", f"Opening {url}", category="Browser")
        if await self._safe_browse(browser, url):
            get_emitter().emit("📄 Reading Document", f"Extracting content from {url}", category="Browser")
            content = await self._safe_read_content(browser)
            observed_url = self._observed_url(browser)
            payload = {
                "ok": True,
                "source": url,
                "observed_url": observed_url,
                "navigation_confirmed": bool(observed_url),
                "content": content,
                "message": f"I've navigated to {url} and captured the content.",
            }
            payload.update(
                _read_comprehension(url=observed_url or url, title="", text=content)
            )
            return payload
        return {"ok": False, "error": f"Failed to load {url}"}

    async def _handle_interact(
        self,
        browser: PhantomBrowser,
        url: str | None,
        actions: list[BrowserAction] | None,
        *,
        action_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if url and not await self._safe_browse(browser, url):
            return {"ok": False, "error": f"Failed to load start URL: {url}"}

        if not actions:
            return {"ok": False, "error": "Interact mode requires 'actions'."}

        authority_actions: set[AuthorityAction] = set()
        interaction_count = 0
        for action in actions:
            if action.type == "click":
                authority_actions.add(AuthorityAction.CLICK)
                interaction_count += 1
            elif action.type == "type":
                # ``PhantomBrowser.type`` first focuses the target through its
                # governed click boundary, then authorizes the actual typing.
                authority_actions.update({AuthorityAction.CLICK, AuthorityAction.TYPE})
                interaction_count += 2

        browser_lease = None
        authority_receipt = str((action_context or {}).get("will_receipt_id") or "")
        if authority_actions:
            token = get_active_governance()
            authorized_domains = {
                ActionDomain.NETWORK_CALL.value,
                ActionDomain.TOOL_EXECUTION.value,
            }
            if (
                token is None
                or not token.authorizes
                or token.receipt_id != authority_receipt
                or token.domain not in authorized_domains
            ):
                return {
                    "ok": False,
                    "error": "browser_interaction_authority_unavailable",
                    "action_report": [],
                    "browser_authority": {
                        "issued": False,
                        "reason": "no matching live ActionExecutor receipt",
                    },
                }
            interaction_origin = origin_of(self._observed_url(browser))
            if not interaction_origin:
                return {
                    "ok": False,
                    "error": "browser_interaction_origin_unobserved",
                    "action_report": [],
                    "browser_authority": {
                        "issued": False,
                        "reason": "the current page origin was not observable",
                    },
                }
            browser_lease = issue_browser_lease(
                principal="sovereign_browser",
                origin=interaction_origin,
                actions=authority_actions,
                ttl_s=self.INTERACTION_TIMEOUT,
                interactions=interaction_count,
                purpose="complete one governed sovereign_browser interaction sequence",
            )

        results = []
        lease_revoked = browser_lease is None
        try:
            for action in actions:
                logger.info("🎬 Action: %s | Sel: %s", action.type, action.selector)
                try:
                    if action.type == "click":
                        success = await asyncio.wait_for(
                            browser.click(
                                selector=action.selector or "",
                                lease_id=browser_lease.lease_id if browser_lease else "",
                            ),
                            timeout=10.0,
                        )
                    elif action.type == "type":
                        success = await asyncio.wait_for(
                            browser.type(
                                action.selector or "",
                                action.value or "",
                                lease_id=browser_lease.lease_id if browser_lease else "",
                            ),
                            timeout=10.0,
                        )
                    elif action.type == "scroll":
                        success = await asyncio.wait_for(
                            browser.scroll(direction=action.value or "down"),
                            timeout=5.0,
                        )
                    elif action.type == "wait":
                        await asyncio.sleep(min(float(action.value or 1), 10.0))  # Cap wait at 10s
                        success = True
                    elif action.type == "get_html":
                        if browser.page:
                            html = await asyncio.wait_for(browser.page.content(), timeout=10.0)
                            results.append({"type": "html", "content": html[:60000]})
                            success = bool(html)
                        else:
                            success = False
                    elif action.type == "screenshot":
                        ss = await asyncio.wait_for(browser.screenshot(), timeout=10.0)
                        if ss:
                            results.append({"type": "screenshot", "data": ss})
                        success = bool(ss)
                    else:
                        success = False
                        logger.warning("Unsupported action type: %s", action.type)
                except TimeoutError:
                    logger.warning("Action '%s' timed out", action.type)
                    success = False
                except (RuntimeError, AttributeError) as action_exc:
                    record_degradation('sovereign_browser', action_exc)
                    logger.warning("Action '%s' failed: %s", action.type, action_exc)
                    success = False

                results.append({"action": action.type, "ok": success})
                if not success:
                    break
        finally:
            if browser_lease is not None:
                lease_revoked = revoke_browser_lease(browser_lease.lease_id)

        final_content = await self._safe_read_content(browser)
        final_url = ""
        try:
            if browser.page:
                final_url = browser.page.url
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass  # no-op: intentional

        action_rows = [row for row in results if "action" in row]
        completed = len(action_rows) == len(actions) and all(
            row.get("ok") is True for row in action_rows
        )
        return {
            "ok": completed,
            "url": final_url,
            "observed_url": final_url,
            "navigation_confirmed": bool(final_url),
            "content": final_content,
            "action_report": results,
            "browser_authority": {
                "issued": browser_lease is not None,
                "lease_id": browser_lease.lease_id if browser_lease else "",
                "origin": browser_lease.origin if browser_lease else "",
                "interactions": interaction_count,
                "revoked": lease_revoked,
            },
            "message": (
                "I've completed the sequence of interactions on the page."
                if completed
                else "The browser interaction sequence stopped before every action completed."
            ),
            "error": "" if completed else "browser_interaction_incomplete",
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

    def _select_search_result(self, links: list[dict[str, str]], query: str = "") -> str | None:
        """Choose the most query-aligned organic result instead of the first link-shaped thing."""
        query_tokens = {
            token for token in re.findall(r"[a-z0-9]+", str(query or "").lower())
            if len(token) >= 3
        }
        source_reading = query_requires_source_reading(query)
        quoted = [
            phrase.lower()
            for phrase in re.findall(r"[\"“”]([^\"“”]{4,200})[\"“”]", str(query or ""))
        ]

        best_url: str | None = None
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
        return None

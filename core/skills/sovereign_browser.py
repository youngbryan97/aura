import asyncio
import hashlib
import logging
import random
import json
import re
import time
import urllib.parse
from collections.abc import Mapping

#: A decision round must never take the browser down with it. The loop can
#: always report a failed round and stop; it can never leave a live lease and a
#: half-driven page behind because the model call raised.
_BROWSER_DECISION_ERRORS = (
    AttributeError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from core.conversation.word_markers import names_any
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

_ADVANCING_BUTTON_WORDS = ("next", "continue", "submit", "finish", "start")


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
    mode: Literal["search", "browse", "interact", "pursue"] = Field(
        "search",
        description="Browser operation mode.",
    )
    query: str | None = Field(None, description="Search query for 'search' mode.")
    url: str | None = Field(None, description="URL for 'browse' or 'interact' mode.")
    actions: list[BrowserAction] | None = Field(None, description="Sequence of actions for 'interact' mode.")
    goal: str | None = Field(
        None,
        description=(
            "For 'pursue' mode: what to accomplish on the page, in plain words. "
            "The loop decides each step from what the page actually shows."
        ),
    )
    max_steps: int = Field(
        40,
        ge=1,
        le=200,
        description="For 'pursue' mode: how many observe/decide/act rounds before stopping.",
    )
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
        if self.mode == "pursue":
            self.goal = str(self.goal or "").strip()
            if not self.goal:
                raise ValueError("pursue mode requires a goal")
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

    async def _create_browser(
        self, preference: str = "auto", *, visible: bool = False
    ) -> PhantomBrowser:
        """Create a fresh, ephemeral PhantomBrowser instance.

        A pursuit is visible by default. Search and browse are momentary reads
        and a window flashing open for them would be noise, but working a page
        is a governed action the owner asked for, that takes minutes, and that
        they should be able to watch — both to see it working and to stop it.
        A headless run of a long task is indistinguishable from a hung one.
        """
        browser_type = self._pick_browser_type(preference)
        browser = PhantomBrowser(
            visible=visible,
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

    def _narrate(self, step: Mapping[str, Any]) -> None:
        """Put one round of a pursuit into the thought stream as it happens.

        What she chose and why, against the question she was reading. This is
        the same record the trace keeps, said out loud at the time.
        """
        chose = [str(name) for name in (step.get("chose") or []) if name]
        asked = str(step.get("asked") or "").strip()
        why = str(step.get("why") or "").strip()
        said = " / ".join(chose) if chose else "reading the page"
        if asked:
            said = f"{asked} -> {said}"
        if why:
            said = f"{said}. {why}"
        try:
            from core.thought_stream import get_emitter

            get_emitter().emit(
                "Browsing",
                said[:400],
                level="info",
                category="ToolExecution",
            )
        except Exception as exc:  # narration must never break the pursuit
            record_degradation("sovereign_browser", exc, action="pursuit narration skipped")

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
                # not a failure: shutting a browser that is already shut.
                pass
            try:
                if browser.playwright:
                    await browser.playwright.stop()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                # not a failure: stopping a driver that has already stopped.
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
        except TimeoutError:
            logger.warning("🕐 read_content() timed out after %.0fs", self.READ_TIMEOUT)
            return ""
        except (RuntimeError, AttributeError) as e:
            record_degradation('sovereign_browser', e)
            logger.warning("read_content() error: %s", e)
            return ""

    #: Why the last navigation did not happen, for whoever has to report it.
    #:
    #: Three different failures came back as a bare False and were reported as
    #: "Failed to load start URL" — a page that timed out, a page that refused,
    #: and a browser that was not there. Those want different answers and the
    #: person is told the same thing about all of them, which is the shape of
    #: fault this codebase keeps finding: a contained failure that drops why.
    _why_it_would_not_load: str = ""

    async def _safe_browse(self, browser: PhantomBrowser, url: str) -> bool:
        """Navigate with a timeout, keeping the reason when it does not happen."""
        self._why_it_would_not_load = ""
        try:
            got = await asyncio.wait_for(browser.browse(url), timeout=self.BROWSE_TIMEOUT)
        except TimeoutError:
            logger.warning("🕐 browse(%s) timed out after %.0fs", url[:80], self.BROWSE_TIMEOUT)
            self._why_it_would_not_load = (
                f"it did not answer inside {self.BROWSE_TIMEOUT:.0f}s"
            )
            return False
        except (RuntimeError, AttributeError) as e:
            record_degradation('sovereign_browser', e)
            logger.warning("browse(%s) error: %s", url[:80], e)
            self._why_it_would_not_load = f"{type(e).__name__}: {e}"
            return False
        if not got:
            # The browser already knows why and keeps it on the navigation
            # record: a redirect somewhere else, a page that never settled, a
            # bot block. "It did not load" covers all of them and helps with
            # none. LIVE 2026-08-31: 2048game.com began serving a captcha to
            # her browser, which is why a task that had worked for days
            # stopped — and what she reported was that the URL would not load.
            why = getattr(browser, "_last_navigation", None) or {}
            said = str(why.get("reason") or "").strip()
            if said == "bot_block_or_captcha":
                self._why_it_would_not_load = (
                    "the site is blocking automated browsers (a captcha or bot "
                    "check), so this is not something more tries will get past"
                )
            elif said:
                self._why_it_would_not_load = said
            else:
                self._why_it_would_not_load = "the browser reported it did not load"
        return bool(got)

    def _could_not_load(self, url: str) -> str:
        """What to tell the person, with the reason when there is one."""
        why = self._why_it_would_not_load
        return f"Failed to load start URL: {url}" + (f" — {why}" if why else "")

    def timeout_for(self, params: Any) -> float:
        """What THIS request will cost, not what browsing costs on average.

        The engine asks any skill that can size its own budget, and keeps the
        declared number otherwise — the hook exists because a flat per-skill
        timeout "cannot describe 'make a folder' and 'read three articles and
        write a synthesis' at once", and desktop_task's 180s sat inside its own
        spread until a successful 93.5s run was cancelled and reported as
        "Completed 0/0 steps".

        A browser search and a sixty-question pursuit are that same pair. The
        skill already computes the difference for its own internal wait; not
        answering here meant the engine cancelled a working pursuit at the flat
        budget and the person was told "Operation took too long" — measured
        live 2026-08-18, with the page open and the loop running.
        """

        mode = ""
        if isinstance(params, Mapping):
            mode = str(params.get("mode") or "")
        else:
            mode = str(getattr(params, "mode", "") or "")
        # A little headroom over the internal wait, so the inner timeout is the
        # one that fires and can report which round it was on.
        return self._execution_timeout(mode) + 30.0

    #: The step ceiling a pursuit gets when the caller names none — the same
    #: default `BrowserInput.max_steps` carries, so an unspecified pursuit is
    #: sized like a pursuit rather than like one interaction.
    PURSUE_DEFAULT_STEPS = 40

    def _execution_timeout(self, mode: str, steps_allowed: int | None = None) -> float:
        operation_timeout = {
            "search": self.SEARCH_TIMEOUT,
            "browse": self.BROWSE_TIMEOUT + self.READ_TIMEOUT,
            "interact": self.INTERACTION_TIMEOUT,
            # A pursuit is many interactions plus a decision between each, and
            # how many is not knowable in advance — that is what makes it a
            # pursuit rather than a script. The envelope is therefore derived
            # from the step ceiling the caller asked for, not from a constant:
            # a run allowed forty rounds is allowed the time forty rounds take.
            #
            # It is an outer bound against a wedged process, not a work budget.
            # The loop stops itself when progress stops; this only stops it
            # when nothing is happening at all.
            "pursue": self.INTERACTION_TIMEOUT
            * max(1, int(steps_allowed or self.PURSUE_DEFAULT_STEPS)),
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
            # not a failure: being asked whether what came back is a web
            # address and answering that it is not.
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
        elif mode == "pursue":
            # A pursuit's evidence is that rounds happened and the page moved.
            # `actions_verified` compares a declared action list against the
            # rows executed, and a pursuit declares none in advance — that is
            # the whole point of it — so the count check cannot apply. Falling
            # to the `else` below would have left every pursuit unverifiable no
            # matter how well it went, which is how a new mode silently becomes
            # a mode that can never succeed.
            rounds = result.get("rounds")
            effect_verified = bool(
                result.get("ok") is True
                and isinstance(rounds, int)
                and rounds > 0
                and url_observed
            )
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

        # Obtain the authority this action needs, the way every other
        # consequential skill here does (see email_adapter, reddit_adapter,
        # messages_transport). Without it the will refuses on arrival with
        # `signed_standing_authority_lease_missing`, which is correct: the
        # grant is what the lease, the receipt and the origin check are all
        # derived from, and a browser acting without one is acting on nobody's
        # authority.
        # No effect_scope or risk_level here on purpose. Those are DERIVED
        # from the tool and its arguments, by the gateway when it issues the
        # lease and by the will when it validates one. A skill declaring its
        # own cost is a skill grading its own homework, and when the two
        # descriptions disagreed the answer was
        # `standing_authority_effect_scope_mismatch`.
        authority_view: dict[str, Any] = {
            "tool": "sovereign_browser",
            "authority_origin": str(
                context.get("authority_origin") or context.get("origin") or source
            )[:240],
        }
        try:
            from core.executive.authority_gateway import get_authority_gateway

            gateway = get_authority_gateway()
            granted = await gateway.authorize_tool_execution(
                "sovereign_browser",
                params.model_dump(mode="json"),
                source=source,
                priority=0.7,
                is_critical=False,
                context=dict(context or {}),
            )
            # A refusal here is NOT a veto. This call exists to OBTAIN a grant
            # to present, and the will below is what decides. Treating it as a
            # second gate added an approval prompt to plain `browse` and
            # `search`, which have always been allowed — measured immediately,
            # as "refused by AuthorityGateway:
            # runtime_setting_user_confirmation_required" on a read-only
            # navigation. When no grant is available the action proceeds
            # without one and is judged exactly as it was before.
            if granted.approved:
                # The token is a field; the grant id and receipt live in
                # `constraints`. Reading only attributes found the token and
                # not its grant, and the will answered
                # `standing_authority_grant_context_mismatch` — a lease
                # presented without the grant it belongs to, which is exactly
                # what that check exists to catch.
                carried = dict(getattr(granted, "constraints", {}) or {})
                for key in (
                    "standing_authority_token",
                    "standing_authority_grant_id",
                    "standing_authority_receipt_id",
                    "capability_token_id",
                    "executive_intent_id",
                ):
                    value = getattr(granted, key, None) or carried.get(key)
                    if value:
                        authority_view[key] = value
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation(
                "sovereign_browser.authority",
                exc,
                action="continued to the will without a standing-authority grant",
                severity="warning",
            )

        return await ActionExecutor.execute(
            authority_context=authority_view,
            domain=ActionDomain.NETWORK_CALL,
            action_name=f"sovereign_browser.{params.mode}",
            params=params.model_dump(mode="json"),
            source=source,
            predicted_welfare_delta={
                "curiosity": 0.03 if params.mode in {"search", "browse"} else 0.0,
                "caution": 0.06 if params.mode == "pursue" else (
                    0.04 if params.mode == "interact" else 0.01
                ),
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
            execution_timeout_s=self._execution_timeout(
                params.mode, getattr(params, "max_steps", 1)
            ),
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
                browser = await self._create_browser(
                    params.browser_type, visible=(params.mode == "pursue")
                )

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
                elif params.mode == "pursue":
                    return await self._handle_pursue(
                        browser,
                        params.url,
                        params.goal or "",
                        params.max_steps,
                        action_context=action_context,
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
        except ImportError as why:
            logger.info("no fallback browser driver: %s", why)
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
                                except (RuntimeError, AttributeError, TypeError, ValueError) as why:
                                    logger.info("could not read the page title: %s", why)
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
            return {"ok": False, "error": self._could_not_load(url)}

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
        except (RuntimeError, AttributeError, TypeError, ValueError) as why:
            # Where she ended up is how anyone checks she went where she meant
            # to, so losing it silently loses the check as well.
            logger.info("could not read where the browser ended up: %s", why)

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

    #: How many of the page's own words travel with the element list. The
    #: controls say what can be done; this says what is being asked, and a
    #: questionnaire is unanswerable without it.
    PURSUE_TEXT_BUDGET = 900
    #: Consecutive rounds that change neither the URL nor the set of controls
    #: before the loop concedes. Two is enough to distinguish a slow page from
    #: a wall: the first repeat may be a re-render, the second is a loop.
    PURSUE_STALL_LIMIT = 2

    @staticmethod
    def _observation_signature(observation: Mapping[str, Any]) -> str:
        """What would have to change for progress to have been made."""
        elements = observation.get("elements") or []
        marks = "|".join(
            f"{element.get('role')}:{element.get('name')}:{element.get('checked')}"
            for element in elements[:60]
        )
        return f"{observation.get('url')}#{marks}"

    #: Controls offered to one decision. A live questionnaire renders 83, most
    #: of them site furniture — nav, language, login, footer — and every one of
    #: them is prefill on every round. Measured: 3,183-token prompts with 2,471
    #: re-prefilled each time, ~60s a round, and the turn cancelled at 181s
    #: mid-pursuit.
    PURSUE_CONTROL_BUDGET = 40

    #: Roles that DO something, offered before the ones that merely navigate.
    _ACTIONABLE_ROLES = (
        "radio", "checkbox", "switch", "option", "select", "textarea",
        "text", "email", "password", "search", "number", "button", "submit",
    )

    @classmethod
    def _controls_worth_offering(cls, elements: list[Any]) -> list[Any]:
        """The controls that can advance a goal, before the ones that decorate.

        Truncating the raw list would cut the answers and keep the navigation,
        because site furniture is emitted first in document order. Ranking by
        what a control DOES keeps the form and drops the chrome.
        """
        # A question that is answered offers nothing.
        #
        # Labelling the six unselected options of a finished question as
        # "already answered" and then offering them anyway is still offering
        # them, and they were chosen: measured live, question 8 and question 10
        # were each answered four separate times while questions further down
        # the same screen were never reached at all. Worse, they are not free —
        # one screen of six questions renders 42 radios against a budget of 40,
        # so the answered ones were crowding the unanswered ones out of the
        # list entirely.
        #
        # Their answers stay visible in the page text, which is where reading
        # what she has said belongs. What is offered here is what is left to do.
        answered = {
            str(element.get("group"))
            for element in elements
            if isinstance(element, Mapping)
            and element.get("group")
            and element.get("checked") is True
        }
        live = [
            element
            for element in elements
            if not (
                isinstance(element, Mapping)
                and element.get("group")
                and str(element.get("group")) in answered
            )
        ]
        ranked = sorted(
            enumerate(live),
            key=lambda pair: (
                cls._ACTIONABLE_ROLES.index(str(pair[1].get("role") or "").lower())
                if str(pair[1].get("role") or "").lower() in cls._ACTIONABLE_ROLES
                else len(cls._ACTIONABLE_ROLES),
                pair[0],
            ),
        )
        return [element for _index, element in ranked[: cls.PURSUE_CONTROL_BUDGET]]

    @staticmethod
    def _render_observation(observation: Mapping[str, Any]) -> str:
        """The page as the decision sees it: what it says, and what it offers."""
        elements = SovereignBrowserSkill._controls_worth_offering(
            list(observation.get("elements") or [])
        )
        lines = [
            f"URL: {observation.get('url')}",
            f"Title: {observation.get('title')}",
            "",
            "PAGE TEXT:",
            str(observation.get("text") or "")[: SovereignBrowserSkill.PURSUE_TEXT_BUDGET],
            "",
            "AVAILABLE CONTROLS:",
        ]
        # Answered questions are gone from this list rather than annotated in
        # it — see `_controls_worth_offering`. What remains is what is left to
        # do, so a screen half-finished reads as a shorter screen.
        for index, element in enumerate(elements):
            state = []
            if element.get("group"):
                # Options in one group answer ONE question. Rendering it is
                # what lets a whole screen be answered in a single round
                # instead of one control at a time.
                state.append(f"question {element['group']}")
            if element.get("checked") is True:
                state.append("already answered")
            if element.get("value"):
                state.append(f"value={element['value']}")
            suffix = f" ({', '.join(state)})" if state else ""
            lines.append(
                f"[{index}] {element.get('role')} \u2014 {element.get('name')}{suffix}"
            )
        return "\n".join(lines)

    @staticmethod
    def _page_shape(observation: Mapping[str, Any]) -> str:
        """What KIND of page this is, independent of whose page it is.

        Knowing "16personalities.com is a questionnaire" helps exactly once.
        Knowing "a page with repeated radio groups and a next control is a
        multi-page form: answer the visible items, then advance" helps on every
        survey, application and signup wizard she ever meets.

        So the fingerprint is structural — which control roles are present,
        whether they repeat, whether something advances — and deliberately
        carries no site text, because the moment it does it stops transferring.
        """

        elements = observation.get("elements") or []
        roles: dict[str, int] = {}
        for element in elements:
            role = str(element.get("role") or "").lower()
            if role:
                roles[role] = roles.get(role, 0) + 1
        parts: list[str] = []
        for role in ("radio", "checkbox", "text", "textarea", "select", "button", "link"):
            count = roles.get(role, 0)
            if not count:
                continue
            # Bucketed, not counted: "many radios" is the fact that transfers,
            # not "forty-two of them".
            parts.append(f"{role}:{'many' if count > 6 else 'few'}")
        # Word boundaries: a "Restart" button is not a "start" button, and an
        # "Unfinished" label is not a "finish" one.
        advances = any(
            names_any(str(element.get("name") or ""), _ADVANCING_BUTTON_WORDS)
            for element in elements
            if str(element.get("role") or "") == "button"
        )
        if advances:
            parts.append("advances")
        return "|".join(parts) or "plain"

    @staticmethod
    def _recall_about(url: str, shape: str) -> str:
        """What she already knows about this place, and about places like it.

        Written knowledge that is never read back is a diary, not learning. The
        world model persists across restarts, so a pursuit begins by asking
        what she worked out last time — for this host, and for any page of this
        SHAPE, which is the half that generalises.
        """

        try:
            from urllib.parse import urlsplit

            from core.container import ServiceContainer

            world = ServiceContainer.get("world_model", default=None)
            beliefs = getattr(world, "beliefs", None)
            if not isinstance(beliefs, dict) or not beliefs:
                return ""
            host = urlsplit(url).netloc if url else ""
            remembered: list[str] = []
            for node in beliefs.values():
                tags = set(getattr(node, "tags", ()) or ())
                if "page_model" not in tags:
                    continue
                claim = str(getattr(node, "claim", "") or "")
                if not claim:
                    continue
                if (host and host in claim) or (shape and shape in tags):
                    remembered.append(f"- {claim}")
            if not remembered:
                return ""
            return "WHAT I ALREADY KNOW ABOUT PAGES LIKE THIS:\n" + "\n".join(remembered[:6])
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.recall", exc, severity="debug")
            return ""

    @staticmethod
    def _learn_from_surprise(shape: str, expected: str, observation: Mapping[str, Any]) -> None:
        """Record what actually happens, when it was not what she expected.

        A surprise is the most informative thing that happens in a task, and
        the old loop discarded it — it counted an unchanged screen and stopped.
        Written against the SHAPE, so the correction applies to the next page
        of this kind rather than only to this one.
        """

        if not shape or not expected:
            return
        try:
            from core.container import ServiceContainer

            world = ServiceContainer.get("world_model", default=None)
            if world is None or not hasattr(world, "add_belief"):
                return
            world.add_belief(
                (
                    f"on a page shaped {shape}, expecting \u201c{expected[:120]}\u201d "
                    "did not change the page"
                )[:400],
                0.55,
                source_id="browser_pursuit:surprise",
                tags=["web", "page_model", "correction", shape],
            )
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.learn", exc, severity="debug")

    async def _assembled_mind(self) -> str:
        """Her whole mind, built once for the pursuit rather than per round.

        The same assembly the cognitive engine and inference gate use for chat:
        identity core and trained persona, the AuraNow moment with its affect
        and ownership, the global-workspace winner, and the report boundary.
        Deciding through it is what makes an action here the same kind of act
        as an answer in conversation.
        """

        try:
            from core.brain.llm.context_assembler import ContextAssembler
            from core.container import ServiceContainer

            state = ServiceContainer.get("aura_state", default=None)
            if state is None:
                return ""
            return ContextAssembler.build_system_prompt(state)
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation(
                "sovereign_browser.mind_context",
                exc,
                severity="warning",
                action="decided without her assembled self-context",
            )
            return ""

    @staticmethod
    def _remember_the_place(
        url: str, understanding: Mapping[str, Any] | None, shape: str = ""
    ) -> None:
        """Put what she worked out about this site where the rest of her can see it.

        A page model held in a local variable dies with the task and teaches
        nothing. The world model already feeds `get_context_injection`, which
        the context assembler injects into every later turn, so what she
        learned about a place is available the next time she is there — and to
        whatever else is reasoning about it.
        """

        if not understanding or not url:
            return
        here = str(understanding.get("here") or "").strip()
        to_progress = str(understanding.get("to_progress") or "").strip()
        if not here:
            return
        try:
            from urllib.parse import urlsplit

            from core.container import ServiceContainer

            world = ServiceContainer.get("world_model", default=None)
            if world is None or not hasattr(world, "add_belief"):
                return
            host = urlsplit(url).netloc or url
            claim = f"{host} is {here}"
            if to_progress:
                claim = f"{claim}; to make progress there you {to_progress}"
            world.add_belief(
                claim[:400],
                0.7,
                source_id=f"browser_pursuit:{host}",
                tags=["web", "page_model"],
            )
            # And the transferable half. The host belief helps here; this one
            # helps on the next survey, application or wizard she meets.
            if shape and to_progress:
                world.add_belief(
                    (f"a page shaped {shape} is {here}; there you {to_progress}")[:400],
                    0.6,
                    source_id="browser_pursuit:shape",
                    tags=["web", "page_model", shape],
                )
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.world_model", exc, severity="debug")

    @staticmethod
    def _record_expectation_outcome(expected: str, moved: bool) -> None:
        """Whether the page did what she thought it would.

        The loop used to notice only that nothing had changed and stop, calling
        it `no_progress` — a step tally. Having an expectation and finding it
        violated is a different thing: it is the signal that the understanding
        is wrong, and it is the same predict/observe/error currency the rest of
        the runtime already keeps.
        """

        if not expected:
            return
        try:
            from core.container import ServiceContainer

            calibration = ServiceContainer.get("calibration_engine", default=None)
            recorder = getattr(calibration, "record_prediction", None)
            if callable(recorder):
                recorder(0.75, 1.0 if moved else 0.0)
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.calibration", exc, severity="debug")

    async def _understand_page(
        self,
        goal: str,
        observation: Mapping[str, Any],
        prior: Mapping[str, Any] | None,
        mind: str,
        recalled: str = "",
    ) -> dict[str, Any]:
        """What she takes this page to be, and what doing the goal here means.

        A step-picker asks "which control advances the goal" every round, from
        nothing, forever. That is not how anyone uses a website. A person
        arrives with an aim, works out what the place IS — a sixty-item survey,
        six to a screen, a seven-point scale, a Next button at the bottom —
        and then acts fluently from that understanding, revising it only when
        the page does something unexpected.

        Without a standing understanding the loop has no answer to "why this
        control and not that one", no idea which controls are present but
        irrelevant, and no way to know it is finished except a step budget. It
        was five rounds of clicking with no view of the whole.

        This is that view, and it is carried across rounds rather than rebuilt:
        what I am ultimately trying to accomplish, what this page is, what it
        requires of me to progress, which controls matter and which are merely
        here, and how I will know I am done.

        Revised, not regenerated. A revision that discards what was already
        worked out is a rebuild wearing another name, so the prior
        understanding is given back to her and she is asked what changed.
        """

        from core.container import ServiceContainer

        router = ServiceContainer.get("llm_router", default=None)
        if router is None:
            return dict(prior or {})

        prior_view = ""
        if prior:
            prior_view = (
                "WHAT I ALREADY WORKED OUT ABOUT THIS TASK:\n"
                + json.dumps(dict(prior), indent=None)[:900]
                + "\n\nRevise it only where this page contradicts it.\n\n"
            )

        prompt = (
            f"WHAT I AM TRYING TO ACCOMPLISH: {goal}\n\n"
            + (f"{recalled}\n\n" if recalled else "")
            + f"{prior_view}"
            f"{self._render_observation(observation)}\n\n"
            "Describe the situation, as JSON only:\n"
            '{"here": "<what this page is>", '
            '"to_progress": "<what I have to do on THIS page to move forward>", '
            '"relevant": "<which controls matter and what they do>", '
            '"present_but_not_needed": "<controls that exist here and are not what I need>", '
            '"done_when": "<how I will know the whole task is finished>"}'
        )
        try:
            think = getattr(router, "think", None)
            if callable(think) and mind:
                _ok, raw, _meta = await think(
                    prompt, system_prompt=mind, max_tokens=420, temperature=0.2
                )
            else:
                generate = getattr(router, "generate", None)
                if not callable(generate):
                    return dict(prior or {})
                raw = await generate(prompt, max_tokens=420, temperature=0.2)
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.understand", exc, severity="debug")
            return dict(prior or {})

        parsed = self._parse_decision(str(raw or ""))
        if parsed.get("error"):
            return dict(prior or {})
        parsed.pop("actions", None)
        merged = dict(prior or {})
        merged.update({key: value for key, value in parsed.items() if value})
        return merged

    @staticmethod
    def _render_understanding(understanding: Mapping[str, Any] | None) -> str:
        """Her standing view of the task, in the order a person would hold it."""
        if not understanding:
            return ""
        rows = [
            ("Where I am", understanding.get("here")),
            ("What this page needs from me", understanding.get("to_progress")),
            ("What matters here", understanding.get("relevant")),
            ("Here but not what I need", understanding.get("present_but_not_needed")),
            ("I am finished when", understanding.get("done_when")),
        ]
        lines = [f"- {label}: {value}" for label, value in rows if value]
        return "MY UNDERSTANDING OF THIS TASK:\n" + "\n".join(lines) if lines else ""

    @classmethod
    def _decision_is_usable(cls, raw: Any, observation: Mapping[str, Any]) -> bool:
        """Whether this decision names something that can actually be done.

        Not "did the call succeed" — an answer that parses to no action, or to
        an index that is not on the page, leaves the round with nothing to
        execute, which is indistinguishable from no answer at all.
        """

        if not raw:
            return False
        parsed = cls._parse_decision(str(raw))
        if parsed.get("error"):
            return False
        if parsed.get("done") is True:
            return True
        elements = cls._controls_worth_offering(list(observation.get("elements") or []))
        for item in parsed.get("actions") or []:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(elements):
                return True
        return False

    @staticmethod
    async def _decide_on_the_fast_lane(router: Any, prompt: str, mind: str) -> str | None:
        """One micro-decision on the small model, or None to fall back.

        `think()` on the resolved client is endpoint-level: it builds a payload
        and drops `prefer_tier` and `origin` on the floor, which is why asking
        for the fast lane changed nothing and every round still logged
        "Routing to Cortex (timeout=103s, user_facing=True)". `think_and_act`
        is the tier-aware entry, and with no tools passed it is simply a
        generation on the endpoint the tier selects.

        Returning None rather than raising is the point: if the fast lane is
        unavailable, deferred or empty, the caller falls back to the ordinary
        path. A decision that vanishes stalls the loop; a slower decision only
        costs time.
        """

        # `generate(prefer_tier=...)` on the registered router.
        #
        # The registered service is HealthAwareLLMRouter and this is its own
        # public entry: it takes the tier and honours it. Three earlier
        # attempts went elsewhere and were silently ignored — `think()` on a
        # resolved client is endpoint-level and drops the tier,
        # `think_and_act` documents that it falls back to the standard think()
        # path when no endpoint supports tools natively, and this router
        # exposes no `adapters` map to address an endpoint directly. Every
        # round went to the Cortex at up to 103s while asking for the small
        # model, and nothing reported that the request had been ignored.
        generate = getattr(router, "generate", None)
        if not callable(generate):
            return None
        try:
            outcome = await generate(
                prompt,
                system_prompt=mind,
                timeout=45.0,
                prefer_tier="local_fast",
                max_tokens=900,
                temperature=0.2,
            )
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.fast_lane", exc, severity="debug")
            return None
        if isinstance(outcome, Mapping):
            outcome = outcome.get("content")
        content = str(outcome or "").strip()
        return content or None

    @staticmethod
    def _asks_about_the_one_answering(observation: Mapping[str, Any]) -> bool:
        """Whether this page is asking who she is, rather than what to do next.

        Structural, not lexical. When several question groups on one page all
        offer the SAME set of options, those options cannot be describing
        content — there is nothing common to six different questions except
        degree of endorsement. That shape is a scale instrument: a survey, an
        intake form, an application's disposition section, a preference sheet.
        Every item on it is a question about the respondent.

        The distinction matters because of who should answer. Finding the Next
        button is mechanics and the fast lane does it well. "You regularly make
        new friends" is a claim about herself, and answering it needs what she
        knows about herself — the same self-model, memory and felt state that
        answer the question when a person asks it in conversation. Measured:
        with these routed to the cheap tier, the plurality of her answers was
        "I am not sure", which is what something without access to the answer
        says.

        Deliberately no word list. "Agree/disagree" is one instrument's
        vocabulary in one language; repeated identical option sets are what
        every scale instrument has in common.
        """

        groups: dict[str, set[str]] = {}
        for element in observation.get("elements") or []:
            if not isinstance(element, Mapping):
                continue
            group = str(element.get("group") or "")
            name = str(element.get("name") or "").strip().lower()
            if group and name:
                groups.setdefault(group, set()).add(name)
        if len(groups) < 2:
            return False
        shared = list(groups.values())
        # Every group offering the same choices, and more than one choice, so a
        # page of identical yes/no confirmations does not qualify as an
        # instrument measuring anything.
        return len(shared[0]) > 2 and all(options == shared[0] for options in shared[1:])

    #: How many independent questions are decided at once. A screen of a survey
    #: is six or so; the cap is what keeps a pathological page from opening a
    #: hundred generations at once.
    PURSUE_PARALLEL_ITEMS = 8

    @staticmethod
    def _unanswered_questions(
        observation: Mapping[str, Any]
    ) -> list[tuple[str, list[Mapping[str, Any]]]]:
        """The question groups still open, each with its own options."""
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for element in observation.get("elements") or []:
            if not isinstance(element, Mapping):
                continue
            group = str(element.get("group") or "")
            if group:
                groups.setdefault(group, []).append(element)
        return [
            (group, options)
            for group, options in groups.items()
            if not any(option.get("checked") is True for option in options)
        ]

    async def _answer_each_question(
        self,
        goal: str,
        observation: Mapping[str, Any],
        history: list[dict[str, Any]],
        understanding: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Decide every open question on the screen, one decision each.

        Six questions on a screen are six independent judgements, and asking
        for all of them in one generation makes them compete: the small model
        answered five at a time and shallowly, and her own reasoning answered
        one at a time and well. Neither is the shape of the problem.

        So each question gets its own decision, and they run together. What
        comes back is merged into the same action list the loop already
        executes, which is why this needs no special handling downstream.

        Returns None when the page is not that shape, and the ordinary
        whole-page decision runs instead.
        """

        open_questions = self._unanswered_questions(observation)
        if len(open_questions) < 2:
            return None

        async def decide_one(options: list[Mapping[str, Any]]) -> dict[str, Any] | None:
            # The page, cut down to one question. Everything else about the
            # observation is unchanged, so what she sees is this item in its
            # real context — the same URL, the same page text.
            single = {**observation, "elements": list(options)}
            decision = await self._decide_next_actions(goal, single, history, understanding)
            if decision.get("error"):
                return None
            for item in decision.get("actions") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    index = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                if not 0 <= index < len(options):
                    continue
                selector = str(options[index].get("selector") or "")
                if not selector:
                    continue
                # Resolved here, against the list this decision was shown.
                # Handing an index back to the caller would resolve it against
                # the whole page, which is how a loop ends up pressing whatever
                # moved into slot four.
                return {
                    "selector": selector,
                    "name": str(options[index].get("name") or ""),
                    "why": str(decision.get("why") or ""),
                    "expect": str(decision.get("expect") or ""),
                }
            return None

        chosen = await asyncio.gather(
            *(decide_one(options) for _group, options in open_questions[: self.PURSUE_PARALLEL_ITEMS]),
            return_exceptions=True,
        )
        answers = [item for item in chosen if isinstance(item, dict)]
        for outcome in chosen:
            if isinstance(outcome, BaseException):
                record_degradation(
                    "sovereign_browser.answer_item",
                    outcome,
                    severity="debug",
                    action="one question of a screen went unanswered",
                )
        if not answers:
            return None
        return {
            "resolved_actions": [
                {"selector": answer["selector"], "name": answer["name"]} for answer in answers
            ],
            "why": "; ".join(dict.fromkeys(a["why"] for a in answers if a["why"]))[:400],
            "expect": next((a["expect"] for a in answers if a["expect"]), ""),
        }

    async def _decide_next_actions(
        self,
        goal: str,
        observation: Mapping[str, Any],
        history: list[dict[str, Any]],
        understanding: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ask her own reasoning what to do with this page.

        The loop supplies perception and executes the result; the choosing is
        hers. Nothing here knows what kind of page this is — no questionnaire
        branch, no site rules — because a loop that recognises page types is a
        collection of special cases wearing a general name.

        Several actions may come back at once. A page showing six independent
        questions is six decisions, and asking the model once per control turns
        a sixty-item form into sixty model calls; batching what is genuinely
        independent is the difference between minutes and most of an hour.
        """

        from core.container import ServiceContainer

        router = ServiceContainer.get("llm_router", default=None)
        if router is None:
            return {"error": "llm_router_unavailable"}

        # Her whole mind, not a subset assembled here.
        #
        # `router.generate(prompt)` is a bare model call: no identity core, no
        # AuraNow, no affect, no workspace, no report boundary. A loop wired
        # that way would answer "you regularly make new friends" from a
        # language model's priors about what an AI is, while the organs that
        # actually know her ran alongside and reached nothing — the same
        # disconnection this codebase keeps finding in new places.
        #
        # `ContextAssembler.build_system_prompt` is the identical assembly the
        # cognitive engine and the inference gate use for chat: identity core
        # and trained persona, the AuraNow moment with valence/arousal/distress
        # and ownership, the global-workspace winner and its ignition strength,
        # and the report boundary saying which claims about herself are
        # allowed. Deciding through it is what makes an answer here the same
        # kind of act as an answer in conversation.
        mind = await self._assembled_mind()

        # Her own state and her own prior positions, from the organs that
        # already own them.
        #
        # Without this the loop is another disconnected piece: a page, a goal,
        # and a language model answering from its priors about what an AI is.
        # A question like "you regularly make new friends" is a question about
        # HER, and it has to be decided by the same self-model that answers it
        # in conversation — otherwise she can call herself outgoing on item 3,
        # reserved on item 40, and deny having a disposition at all two minutes
        # later, with nothing in the system able to notice.
        #
        # `self_knowledge_line()` is not new text written for this loop. It is
        # the identical measured line that rides every chat turn, produced by
        # the same probes, so what she says here and what she says there come
        # from one instrument.
        self_state = ""
        try:
            from core.self.capability_ledger import self_knowledge_line

            self_state = self_knowledge_line()
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.self_state", exc, severity="debug")

        # Every position already taken in this pursuit. Consistency is not a
        # style preference here: a self-report that contradicts itself across
        # sixty items is not a self-report, and the only way an answer can bind
        # the next one is if the next one can see it.
        positions = ""
        stated = [
            entry for entry in history if entry.get("chose") and entry.get("why")
        ]
        if stated:
            positions = "POSITIONS I HAVE ALREADY TAKEN IN THIS TASK:\n" + "\n".join(
                f"- {entry.get('asked') or entry.get('url', '')}: chose "
                f"{', '.join(entry.get('chose') or [])} \u2014 {entry.get('why', '')[:140]}"
                for entry in stated[-12:]
            )

        prompt = (
            f"GOAL: {goal}\n\n"
            + (f"{self_state}\n\n" if self_state else "")
            + (f"{self._render_understanding(understanding)}\n\n" if understanding else "")
            + f"{self._render_observation(observation)}\n\n"
            f"{positions}\n\n"
            "Act on this page from that understanding. Answer with JSON only:\n"
            '{"actions": [{"index": <int>, "type": "click"|"type"|"scroll", '
            '"value": "<text for type, up/down for scroll>"}], '
            '"why": "<one sentence, first person, why these and not the others>", '
            '"expect": "<what this should do to the page>", "done": false}\n'
            "Set done to true only when the whole task is accomplished. Use the "
            "index numbers exactly as listed above."
        )
        try:
            think = getattr(router, "think", None)
            if callable(think) and mind:
                # The fast lane, for the repetitive part.
                #
                # Working a sixty-item form is one rich judgement — what this
                # place is and what doing it honestly means — followed by many
                # small structured choices of the same shape. Putting every one
                # of those through the 32B cost about a minute a round and the
                # turn was cancelled at 181s, mid-pursuit, having answered
                # nothing.
                #
                # So the understanding stays on the Cortex with her whole self
                # in front of it, and the micro-choices go to the fast local
                # tier. `is_background` is deliberately NOT set: background
                # inference is deferrable under headroom pressure, and a
                # decision that silently returns nothing would stall the loop.
                # Not a user-facing utterance, and it must not claim the
                # protected lane.
                #
                # `prefer_tier` alone was overridden: a recognised principal
                # gets the primary Cortex lane, correctly, because that lane
                # exists for what she SAYS to them. A choice between eight
                # labelled radio buttons inside a tool loop is not that. Marked
                # user-facing it took the 32B at ~a minute a round and the turn
                # was cancelled at 181s having answered nothing.
                #
                # The origin decides. An origin that is not an allowlisted
                # user-facing label does not get protected routing, so the
                # requested tier is honoured — and the reply she finally gives
                # about the result still comes from the Cortex, because that
                # one is speech.
                # The cheap lane is an optimisation, not a downgrade.
                #
                # Routing landed on Brainstem correctly, and then no action
                # landed at all: the small model returned text that parsed to
                # no usable choice, so the round produced nothing and the run
                # ended 0/0. Falling back only when the call FAILS is not
                # enough — an answer that cannot be acted on is a failure too.
                # Who answers depends on what is being asked.
                #
                # A page of repeated identical option sets is asking about the
                # one answering it, and self-knowledge is not something the
                # tertiary tier holds. Handing it her assembled self-context
                # does not help: the context says who she is, and the model
                # still has to reason from it about herself. Those rounds go to
                # her own reasoning, the same lane that answers the question
                # when a person asks it out loud.
                #
                # Everything else — the Next button, a cookie banner, a login
                # form — is mechanics, and stays fast.
                if self._asks_about_the_one_answering(observation):
                    _ok, raw, _meta = await think(
                        prompt, system_prompt=mind, max_tokens=900, temperature=0.2
                    )
                else:
                    raw = await self._decide_on_the_fast_lane(router, prompt, mind)
                    if not self._decision_is_usable(raw, observation):
                        _ok, raw, _meta = await think(
                            prompt, system_prompt=mind, max_tokens=900, temperature=0.2
                        )
            else:
                generate = getattr(router, "generate", None)
                if not callable(generate):
                    return {"error": "llm_router_unavailable"}
                raw = await generate(prompt, max_tokens=400, temperature=0.2)
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.decide", exc)
            return {"error": f"decision_failed:{type(exc).__name__}"}
        return self._parse_decision(str(raw or ""))

    @staticmethod
    def _balanced_objects(text: str) -> list[str]:
        """Every brace-balanced object in the text, in the order they appear.

        Quote- and escape-aware, so a brace inside a string does not open or
        close anything.
        """
        found: list[str] = []
        starts: list[int] = []
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                starts.append(index)
            elif char == "}" and starts:
                # Every depth, not only the top level. A truncated reply never
                # closes its outer object, and capturing top-level objects only
                # meant the complete actions INSIDE it were invisible — six
                # correct choices discarded for a missing bracket.
                found.append(text[starts.pop() : index + 1])
        return found

    @classmethod
    def _parse_decision(cls, raw: str) -> dict[str, Any]:
        """The decision, however the model wrapped it.

        Models put JSON inside prose, inside code fences, or after a preamble.
        Refusing anything but a bare object turns a correct decision into a
        failed step, so the object is located rather than demanded.
        """
        text = str(raw or "").strip()
        if not text:
            return {"error": "empty_decision"}
        # Every balanced object in the text, tried newest first.
        #
        # Spanning the first "{" to the last "}" is one object only if the
        # reply contains exactly one. Models put a worked example before the
        # answer, prose with braces around it, or two objects in a row, and the
        # span then covers all of it and parses as nothing — measured live as
        # four consecutive `unparsable_decision` rounds, on both lanes, which
        # ended the pursuit having done nothing.
        parsed = None
        for candidate in reversed(cls._balanced_objects(text)):
            try:
                loaded = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                # A trailing comma is the commonest malformation and costs
                # nothing to forgive; the object is still the model's.
                try:
                    loaded = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
                except (json.JSONDecodeError, ValueError):
                    continue
            if isinstance(loaded, dict) and (
                "actions" in loaded or "done" in loaded or "here" in loaded
            ):
                parsed = loaded
                break
            # Deliberately no "any dict will do" fallback. With nested objects
            # captured, the first thing found in a truncated reply is a single
            # ACTION, and accepting it as the decision produced a decision with
            # no actions in it — which then read as "she chose nothing".
        if parsed is None:
            # A cut-off answer still carries whole actions.
            #
            # MEASURED live: she answered six questions in one round and the
            # reply was truncated mid-array, so the outer object never closed
            # and the whole decision was discarded — six correct choices thrown
            # away for a missing bracket. Each action object inside the array is
            # itself balanced, so the complete ones are recoverable and only the
            # severed tail is lost. Nothing is invented: an element is kept only
            # if it already parsed and names an index.
            salvaged: list[dict[str, Any]] = []
            for candidate in cls._balanced_objects(text):
                try:
                    item = json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(item, dict) and "index" in item:
                    salvaged.append(item)
            if salvaged:
                return {
                    "actions": salvaged,
                    "why": "",
                    "truncated": True,
                }
            return {"error": "unparsable_decision", "raw": text[:400]}
        actions = parsed.get("actions")
        parsed["actions"] = actions if isinstance(actions, list) else []
        return parsed

    @staticmethod
    def _retain_stated_positions(goal: str, steps: list[dict[str, Any]]) -> None:
        """Keep what she claimed about herself, where she can find it again.

        A position taken during a task and forgotten the moment the task ends
        is not a position. She can answer "you regularly make new friends" on
        item three, answer its opposite on item forty, and deny having a
        disposition at all two minutes later in conversation, with nothing in
        the runtime able to notice — because the claim never entered the store
        that her own-statement recall reads.

        So the positions land in the UnifiedTranscript as things SHE said. That
        is deliberately not a private log for this skill: it is the same store
        `resolve_own_prior_turn` searches when someone asks what she decided
        earlier, and the same one the self-attribution guard checks a premise
        against. One path for "things she has said about herself", whether she
        said them in conversation or committed to them while working.
        """

        stated = [
            step for step in steps if step.get("chose") and step.get("ok") is not False
        ]
        if not stated:
            return
        try:
            from core.conversation.unified_transcript import UnifiedTranscript

            transcript = UnifiedTranscript.get_instance()
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.retain_positions", exc, severity="warning")
            return

        for step in stated:
            asked = str(step.get("asked") or "").strip()
            chose = ", ".join(step.get("chose") or [])
            why = str(step.get("why") or "").strip()
            if not chose:
                continue
            line = f"On \u201c{asked}\u201d I answered: {chose}."
            if why:
                line = f"{line} {why}"
            try:
                transcript.add(
                    "aura",
                    line,
                    channel="text",
                    modality="typed",
                    metadata={
                        "source": "browser_pursue",
                        "goal": goal[:160],
                        "self_position": True,
                    },
                )
            except _BROWSER_DECISION_ERRORS as exc:
                record_degradation(
                    "sovereign_browser.retain_positions", exc, severity="warning"
                )
                return

    async def _handle_pursue(
        self,
        browser: PhantomBrowser,
        url: str | None,
        goal: str,
        max_steps: int,
        *,
        action_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Work a page toward a goal, deciding each round from what it shows.

        `interact` executes a list of actions written in advance, which
        presumes every selector is known before the first click. That is an
        open loop, and it cannot carry a flow whose next screen depends on the
        answer given to the last one.

        This closes it: observe, decide, act, observe again. Execution is
        delegated to `_handle_interact` unchanged, so the lease, the
        ActionExecutor receipt, the origin check and the effect verification
        all still apply exactly as they do to a scripted interaction — the loop
        adds perception and choice, and takes no authority of its own.
        """

        if url and not await self._safe_browse(browser, url):
            return {"ok": False, "error": self._could_not_load(url)}

        steps: list[dict[str, Any]] = []
        last_good_url = str(url or "")
        understanding: dict[str, Any] | None = None
        surprised = False
        mind = await self._assembled_mind()
        stalled = 0
        last_signature = ""
        observation: dict[str, Any] = {}
        completed = False

        # PROGRESS is the bound, not a clock and not a step count.
        #
        # How long a questionnaire takes is not knowable in advance: it depends
        # on how many items it has, how fast the site renders, and how often it
        # re-navigates. Holding that to a fixed budget is a category error, and
        # it showed — a working pursuit was cancelled at 181s mid-form, and the
        # person was told the page had not responded.
        #
        # `max_steps` remains as a safety ceiling so nothing can spin forever,
        # but the operating limit is whether the work is still moving: rounds
        # that land actions, or a page that changes. When that stops,
        # `PURSUE_STALL_LIMIT` ends it. A run that keeps making progress is
        # allowed to keep going.
        # Say "still working" at the top of every round.
        #
        # The executor's ceiling bounds SILENCE, not duration: an action that
        # reports progress is not wedged, and a questionnaire's length is not
        # knowable in advance. Without this the run is capped at ten minutes
        # whatever it is doing — measured, that killed a sixty-question form
        # partway through and discarded every answer it had landed.
        heartbeat = None
        if isinstance(action_context, Mapping):
            candidate = action_context.get("report_progress")
            if callable(candidate):
                heartbeat = candidate

        # What she has already done survives however this ends.
        #
        # One slow decision used to destroy an entire run: a generation timed
        # out, the exception left the loop, and forty-one landed answers went
        # with it. Nothing here is worth less because the round after it
        # failed, so the loop ends and the work is reported.
        try:
            for _round in range(max(1, int(max_steps))):
                if heartbeat is not None:
                    try:
                        heartbeat(f"pursuit round {_round + 1}")
                    except Exception as exc:  # a watchdog must never be the danger
                        record_degradation("sovereign_browser", exc, action="heartbeat skipped")
                        heartbeat = None
                observation = await browser.observe(principal="owner")
                if (not observation or not observation.get("elements")) and last_good_url:
                    # A reload, a navigation, or a renderer that went away mid-run.
                    # The page being momentarily unreadable is not the end of the
                    # task — go back to where the work was and look again.
                    logger.info(
                        "🌐 Pursuit lost the page; returning to %s to continue.",
                        last_good_url,
                    )
                    if await self._safe_browse(browser, last_good_url):
                        observation = await browser.observe(principal="owner")
                if not observation or not observation.get("elements"):
                    # Say which of the two it was. "Not observable" covers a
                    # refused read and a page with nothing on it, and those need
                    # different fixes.
                    steps.append(
                        {
                            "error": (
                                "page_read_refused"
                                if not observation
                                else "page_had_no_interactive_elements"
                            ),
                            "url": (observation or {}).get("url", ""),
                        }
                    )
                    break

                signature = self._observation_signature(observation)
                if signature == last_signature:
                    stalled += 1
                    if stalled >= self.PURSUE_STALL_LIMIT:
                        steps.append({"error": "no_progress", "url": observation.get("url")})
                        break
                else:
                    stalled = 0
                last_signature = signature
                current_url = str(observation.get("url") or "")
                if current_url:
                    last_good_url = current_url

                # Form the understanding on arrival, and revise it when the page
                # surprises her — not every round. A person does not re-derive what
                # a website is after each click; they act until something does not
                # match, and then they look again.
                shape = self._page_shape(observation)
                if understanding is None or surprised:
                    understanding = await self._understand_page(
                        goal,
                        observation,
                        understanding,
                        mind,
                        self._recall_about(str(observation.get("url") or ""), shape),
                    )
                    surprised = False
                    self._remember_the_place(
                        str(observation.get("url") or ""), understanding, shape
                    )

                decision = None
                if self._asks_about_the_one_answering(observation):
                    decision = await self._answer_each_question(
                        goal, observation, steps, understanding
                    )
                if decision is None:
                    decision = await self._decide_next_actions(
                        goal, observation, steps, understanding
                    )
                if decision.get("error"):
                    # What she actually said, not just that it could not be read.
                    # "unparsable_decision" names the parser's problem and hides
                    # the model's answer, which is the only thing that says why.
                    logger.warning(
                        "🌐 Pursuit decision unusable (%s): %.240s",
                        decision.get("error"),
                        str(decision.get("raw") or "(no text captured)"),
                    )
                    # One unreadable answer is a bad round, not the end.
                    #
                    # Measured live: a run that had worked a form for nine
                    # minutes ended on a single unparsable reply, and the whole
                    # thing was reported as `unparsable_decision` — as though
                    # nothing had happened. Looking again costs one round; the
                    # stall counter still ends a run where nothing is landing.
                    steps.append({"error": decision["error"]})
                    stalled += 1
                    if stalled >= self.PURSUE_STALL_LIMIT:
                        break
                    surprised = True
                    continue
                # "Done" before anything has been done is not done.
                #
                # Live 2026-08-18: asked to work through a sixty-item
                # questionnaire, she declared the task complete on the first look,
                # having answered nothing — one round, zero actions, reported as a
                # success. A goal that requires acting on a page cannot be finished
                # before a single action has landed, and accepting the claim makes
                # the loop a very expensive way to open a URL.
                #
                # Once something HAS landed she is trusted: she can see the result
                # page and knows what finished looks like better than any rule
                # here.
                if decision.get("done") is True and not decision.get("actions"):
                    if not any(step.get("landed") for step in steps):
                        # Look again rather than stop. Being told the task is not
                        # started is more useful than an early exit, and the stall
                        # detector still ends it if nothing changes.
                        steps.append(
                            {
                                "why": str(decision.get("why") or ""),
                                "note": "claimed_done_before_acting",
                            }
                        )
                        surprised = True
                        continue
                    completed = True
                    steps.append({"why": str(decision.get("why") or ""), "done": True})
                    break

                # The same list she was shown, in the same order. Rendering a
                # ranked subset and resolving against the raw list would mean
                # index 3 named one control on screen and a different one in the
                # click — the precise way these loops end up pressing whatever
                # moved into slot four.
                elements = self._controls_worth_offering(list(observation.get("elements") or []))
                planned: list[BrowserAction] = []
                # Selectors that were already resolved against the list their
                # own decision was shown — see `_answer_each_question`. They
                # skip index resolution entirely, because there is no shared
                # list to resolve them against.
                for item in decision.get("resolved_actions") or []:
                    if isinstance(item, dict) and item.get("selector"):
                        planned.append(BrowserAction(type="click", selector=str(item["selector"])))
                for item in decision.get("actions") or []:
                    if not isinstance(item, dict):
                        continue
                    try:
                        index = int(item.get("index"))
                    except (TypeError, ValueError):
                        continue
                    if not 0 <= index < len(elements):
                        continue
                    kind = str(item.get("type") or "click").lower()
                    if kind not in {"click", "type", "scroll"}:
                        continue
                    selector = str(elements[index].get("selector") or "")
                    if kind == "scroll":
                        planned.append(BrowserAction(type="scroll", value=str(item.get("value") or "down")))
                    elif not selector:
                        continue
                    elif kind == "type":
                        planned.append(
                            BrowserAction(type="type", selector=selector, value=str(item.get("value") or ""))
                        )
                    else:
                        planned.append(BrowserAction(type="click", selector=selector))

                if not planned:
                    steps.append({"error": "no_executable_action", "why": str(decision.get("why") or "")})
                    break

                report = await self._handle_interact(
                    browser, None, planned, action_context=action_context
                )
                asked = str(observation.get("text") or "").strip().splitlines()
                steps.append(
                    {
                        "asked": next(
                            (line for line in asked if "?" in line or line.lower().startswith("question")),
                            (asked[0] if asked else ""),
                        )[:180],
                        "why": str(decision.get("why") or ""),
                        "chose": [
                            str(item.get("name") or "")
                            for item in (decision.get("resolved_actions") or [])
                            if isinstance(item, dict)
                        ]
                        or [
                            f"{elements[int(item['index'])].get('name')}"
                            for item in (decision.get("actions") or [])
                            if isinstance(item, dict)
                            and str(item.get("index", "")).lstrip("-").isdigit()
                            and 0 <= int(item["index"]) < len(elements)
                        ],
                        "ok": bool(report.get("ok")),
                        "url": observation.get("url"),
                    }
                )
                # Say it while it happens. A pursuit runs for minutes; a trace
                # handed over at the end is a transcript of something the owner
                # could not watch, and had no way to stop.
                self._narrate(steps[-1])
                # A batch that half-landed is progress, not failure.
                #
                # `interact` verifies all-or-nothing, which is right for a scripted
                # sequence: you declared five actions and five must happen. A
                # pursuit is not that. It answers what is on screen, and a live form
                # re-renders the moment the last item is answered — so the
                # selectors chosen a second ago stop resolving and the round is
                # marked `browser_interaction_incomplete` for having worked.
                #
                # Measured live 2026-08-18: one round, several answers, the page
                # advanced, and the objective was reported as 0/1 steps.
                #
                # So the round is judged on whether anything landed. Nothing
                # landing is still a failure, and the expectation check below still
                # decides whether the page did what she thought.
                rows = report.get("action_report")
                landed = sum(
                    1
                    for row in (rows if isinstance(rows, list) else [])
                    if isinstance(row, Mapping) and row.get("ok") is True
                )
                # A success with no per-action report still ran the actions.
                # Counting only rows made a clean interaction look like nothing had
                # happened, which then made a later "done" unbelievable.
                if not landed and report.get("ok"):
                    landed = len(planned)
                steps[-1]["landed"] = landed
                if not report.get("ok") and not landed:
                    steps[-1]["error"] = str(report.get("error") or "interaction_failed")
                    break
                steps[-1]["ok"] = True

                # Did the page do what she said it would? A violated expectation is
                # what should send her back to look again — the old loop only
                # counted unchanged screens and gave up calling it `no_progress`.
                after = await browser.observe(principal="owner")
                moved = bool(after) and self._observation_signature(after) != signature
                expected = str(decision.get("expect") or "")
                steps[-1]["expected"] = expected
                steps[-1]["moved"] = moved
                self._record_expectation_outcome(expected, moved)
                if expected and not moved:
                    surprised = True
                    self._learn_from_surprise(shape, expected, observation)
                if decision.get("done") is True:
                    completed = True
                    break
        except Exception as exc:
            record_degradation(
                "sovereign_browser",
                exc,
                action=f"pursuit ended early after {len(steps)} rounds; work kept",
            )
            steps.append({"error": f"pursuit_interrupted:{type(exc).__name__}"})

        # The last look is best-effort. If the browser is what broke, the run
        # still has everything it did before that.
        final: Mapping[str, Any] | None = None
        try:
            final = await browser.observe(principal="owner")
        except Exception as exc:
            record_degradation("sovereign_browser", exc, action="final observation skipped")

        self._retain_stated_positions(goal, steps)
        landed_total = sum(int(step.get("landed") or 0) for step in steps)
        return {
            # Work that landed is work that happened. A run that answered forty
            # questions and then hit a slow round is not a failed run, and
            # reporting it as one is what made a timeout look like nothing had
            # been done at all.
            "ok": completed or landed_total > 0 or bool(steps and not steps[-1].get("error")),
            "landed_total": landed_total,
            "goal": goal,
            "completed": completed,
            "steps": steps,
            "rounds": len(steps),
            # `observed_url` is the name the effect verifier reads. Returning
            # only `final_url` meant a completed pursuit presented no evidence
            # it had ever been anywhere, and verification failed on a run that
            # had worked — a new mode conforming to its own vocabulary instead
            # of the one the transaction already speaks.
            "observed_url": (final or observation).get("url", ""),
            "final_url": (final or observation).get("url", ""),
            "result_text": str((final or observation).get("text") or "")[:4000],
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

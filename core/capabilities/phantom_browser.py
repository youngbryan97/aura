"""Phantom Browser Module
Playwright-based "human-like" browser agent for Aura.

Capabilities:
- Dynamic Visibility: Headless (background) vs Headed (interactive)
- Human-like Interaction: Random microsleeps, typing speeds, cursor movements
- Robust Navigation: Handling broken links, backing out, reading content
- Content Extraction: Getting markdown from pages

Usage:
    browser = PhantomBrowser()
    browser.browse("https://aura.internal")
    browser.type("input[name='q']", "Hello World")
    browser.click("input[name='btnK']")
"""
import asyncio
import hashlib
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import (
    DependencyUnavailable,
    FallbackClassification,
    Severity,
    record_degradation,
)
from core.capabilities.browser_authority import (
    BrowserAction,
    authorize_browser_action,
)
from core.runtime.runtime_hygiene import get_runtime_hygiene
from core.utils.exceptions import capture_and_log

try:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import Page, async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PlaywrightError = RuntimeError
    PlaywrightTimeoutError = TimeoutError
    Page = Any
    PLAYWRIGHT_AVAILABLE = False

try:
    from playwright_stealth import Stealth
    _STEALTH = Stealth()
    _STEALTH_IMPORT_ERROR = ""
    STEALTH_AVAILABLE = True
except (ImportError, TypeError, ValueError) as stealth_import_error:
    _STEALTH = None
    _STEALTH_IMPORT_ERROR = f"{type(stealth_import_error).__name__}: {stealth_import_error}"
    STEALTH_AVAILABLE = False

logger = logging.getLogger("PhantomBrowser")

_SYSTEM_CHROMIUM_EXECUTABLES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
)


def _record_browser_degradation(
    error: BaseException,
    *,
    stage: str,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"stage": stage, "repair_requested": True}
    if extra:
        payload.update(extra)
    record_degradation(
        "phantom_browser",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        extra=payload,
    )

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Mobile/15E148 Safari/604.1"
]

_CONTENT_HINT_RE = re.compile(
    r"(article|story|content|entry|main|body|markdown|post|read|chapter|prose|text)",
    re.IGNORECASE,
)
_NOISE_HINT_RE = re.compile(
    r"(nav|footer|header|menu|sidebar|cookie|share|social|comment|promo|banner|breadcrumb|related|recommend|subscribe|login|signup|advert)",
    re.IGNORECASE,
)
_NOISY_LINE_RE = re.compile(
    r"^(?:home|about|menu|privacy|terms|cookies?|share|subscribe|login|sign up|contact|next|previous|advertisement)$",
    re.IGNORECASE,
)


def _clean_extracted_page_text(raw_text: str) -> str:
    seen: set[str] = set()
    cleaned_lines: list[str] = []
    for raw_line in str(raw_text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        lower = line.lower()
        if _NOISY_LINE_RE.match(line):
            continue
        if len(line) < 18 and not re.match(r"^(chapter|part|section)\b", lower):
            continue
        if lower in seen:
            continue
        seen.add(lower)
        cleaned_lines.append(line)
    return "\n\n".join(cleaned_lines)


def _score_content_block(title: str, block: dict[str, Any]) -> float:
    text = _clean_extracted_page_text(str(block.get("text") or ""))
    if not text:
        return float("-inf")

    tag = str(block.get("tag") or "").lower()
    block_id = str(block.get("id") or "").lower()
    class_name = str(block.get("class_name") or "").lower()
    meta = " ".join(part for part in (tag, block_id, class_name) if part)
    title_tokens = set(re.findall(r"[a-z0-9]+", str(title or "").lower()))
    text_tokens = set(re.findall(r"[a-z0-9]+", text[:1200].lower()))
    token_overlap = len(title_tokens & text_tokens) / max(1, len(title_tokens)) if title_tokens else 0.0

    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    paragraph_count = int(block.get("paragraph_count") or 0)
    link_density = float(block.get("link_density") or 0.0)
    score = min(len(text) / 180.0, 8.0)
    score += min(sentence_count * 0.12, 2.0)
    score += min(paragraph_count * 0.18, 1.8)
    score += token_overlap * 1.2

    if tag in {"article", "main"}:
        score += 1.2
    if _CONTENT_HINT_RE.search(meta):
        score += 0.8
    if _NOISE_HINT_RE.search(meta):
        score -= 2.4
    score -= min(link_density, 0.8) * 3.0
    return score

def _normalize_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    # `startswith('http')` also accepted "httpfoo://" and anything else
    # beginning with those four letters.
    lowered = text.lower()
    if lowered.startswith(("http://", "https://")):
        return text
    return "https://" + text


class PhantomBrowser:
    """High-fidelity browser agent (Async Version).
    """
    
    def __init__(
        self,
        visible: bool = False,
        browser_type: str = "chromium",
        *,
        principal: str = "",
    ):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page: Page | None = None
        self.visible = visible
        self.browser_type = browser_type
        # Bind ownership once when a production service creates the browser.
        # Individual operations may still provide a narrower principal, while
        # anonymous ad-hoc browser objects remain refused by BrowserAuthority.
        self.principal = str(principal or "").strip()
        self.is_active = False
        self._homeostasis = None
        self._resource_lock = None
        # Whether background work was actually told to stand down, and the
        # last admission verdict. Both are reported, because "running" and
        # "running with coordination" are different states.
        self._resource_coordinated = False
        self._last_admission: dict[str, Any] = {}
        self._startup_error = ""
        self._startup_failure_count = 0
        self._last_launch_attempts: list[str] = []
        self._last_executable_attempts: list[str] = []
        self._launched_executable = ""
        self._stealth_applied = False
        self._stealth_error = _STEALTH_IMPORT_ERROR
        self._driver_pid: int | None = None
        self._driver_registered = False
        self._last_navigation: dict[str, Any] = {"ok": False, "reason": "never_navigated"}
        self._last_verdict: dict[str, Any] = {}
        self._last_interaction: dict[str, Any] = {}
        self._last_extraction: dict[str, Any] = {}
        # One owner for start / rotate / close. Without it two callers
        # could each launch a browser and one could close what the other
        # just created (CP126 ``d9990559``).
        self._lifecycle_lock = asyncio.Lock()
        self._generation = 0
        #: Resources this close could not confirm. Reported rather than
        #: hidden behind an unconditional 'Browser closed'.
        self._close_failures: list[str] = []
        #: Which engine actually launched. The fallback path left
        #: browser_type reporting the requested engine.
        self._launched_engine: str = ""
        
        if not PLAYWRIGHT_AVAILABLE:
            return

    def _effective_principal(self, principal: str = "") -> str:
        return str(principal or "").strip() or str(
            getattr(self, "principal", "") or ""
        ).strip()

    async def ensure_ready(self) -> bool:
        """Start the browser if needed. One starter at a time.

        ``is_active`` was an unsynchronized flag read here and written in
        ``_start_browser``, so several callers could each launch Playwright
        and a browser concurrently while rotation and close mutated the
        same references — one operation closing what another had just
        created (CP126 ``d9990559``).
        """
        async with self._lifecycle_lock:
            if not self.is_active:
                await self._start_browser()
            return self.is_active


    #: Below this the host cannot afford a browser's several hundred MB and
    #: handful of processes. Deliberately generous — refusing a user-visible
    #: browse is a real cost, so this protects the machine from a launch that
    #: would push it into swap rather than being frugal for its own sake.
    MIN_AVAILABLE_GB_FOR_BROWSER = 2.0

    def _browser_admission(self) -> dict[str, Any]:
        """Can this host afford to start a browser right now?

        Answers unknown-as-admit on purpose: if memory cannot be measured,
        refusing every browse would break the capability wholesale on any
        platform without the monitor. The check exists to catch a MEASURED
        shortage, which is the case that actually hurt.
        """
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            available_gb = float(snapshot.available_gb)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "phantom_browser",
                exc,
                severity="info",
                action="admitted the browser because memory pressure could not be measured",
                enforce_failure_policy=False,
            )
            return {"can_admit": True, "reason": "pressure_unmeasured", "available_gb": None}
        if available_gb < self.MIN_AVAILABLE_GB_FOR_BROWSER:
            return {
                "can_admit": False,
                "reason": f"insufficient_memory:{available_gb:.1f}GB_available",
                "available_gb": available_gb,
            }
        return {"can_admit": True, "reason": "", "available_gb": available_gb}

    def get_status(self) -> dict[str, Any]:
        return {
            "active": self.is_active,
            "visible": self.visible,
            "browser_type": self.browser_type,
            "startup_failure_count": self._startup_failure_count,
            "startup_error": self._startup_error[:240],
            "last_launch_attempts": list(self._last_launch_attempts),
            "last_executable_attempts": list(self._last_executable_attempts),
            "stealth_available": bool(STEALTH_AVAILABLE),
            "stealth_applied": bool(self._stealth_applied),
            "stealth_error": self._stealth_error[:240],
            "driver_pid": self._driver_pid,
            "driver_registered": bool(self._driver_registered),
            # A browser running WITHOUT resource coordination is a different
            # state from one running with it; reporting only "active" made
            # them look identical.
            "resource_coordinated": bool(self._resource_coordinated),
            "last_admission": dict(self._last_admission),
            # Live state, not configuration. This reported the browser_type
            # the caller ASKED for even when the fallback launched something
            # else, and `active` alone could not tell a dead page from a
            # healthy one (CP126 ``97a07e2a``).
            "engine_launched": self._launched_engine or "none",
            "executable_launched": self._launched_executable or "none",
            "browser_connected": bool(getattr(self.browser, "is_connected", lambda: False)())
            if self.browser is not None
            else False,
            "page_open": bool(self.page is not None and not getattr(self.page, "is_closed", lambda: True)()),
            "current_url": str(getattr(self.page, "url", "") or "") if self.page else "",
            "generation": self._generation,
            "close_failures": list(self._close_failures),
            "last_navigation": dict(self._last_navigation),
        }

    async def _start_browser(self) -> bool:
        """Start the Playwright browser asynchronously"""
        try:
            if self.is_active:
                return True

            if not PLAYWRIGHT_AVAILABLE:
                error = DependencyUnavailable("playwright is not installed")
                self._startup_failure_count += 1
                self._startup_error = str(error)
                _record_browser_degradation(
                    error,
                    stage="dependency_check",
                    action="kept phantom browser inactive because Playwright is unavailable",
                    severity="degraded",
                    extra={"browser_type": self.browser_type},
                )
                return False

            # CP126 (medium): "Resource-lock failure is explicitly fail-open.
            # Browser startup continues after homeostatic resource
            # coordination cannot be acquired. There is no admission
            # decision, resource budget, or later reconciliation, so memory-
            # or latency-sensitive runtime periods can still start a full
            # browser while status presents normal readiness."
            #
            # The missing piece was the admission decision, not the lock. A
            # browser is hundreds of megabytes and several processes, and it
            # was launched without anyone asking whether the machine could
            # afford one — on a host already holding a ~20GB resident model.
            admission = self._browser_admission()
            self._last_admission = admission
            if not admission["can_admit"]:
                self._startup_error = f"admission_refused:{admission['reason']}"
                _record_browser_degradation(
                    RuntimeError(f"browser admission refused: {admission['reason']}"),
                    stage="admission",
                    action="refused to start a browser while the host could not afford one",
                    severity="warning",
                    extra={"available_gb": admission.get("available_gb")},
                )
                return False

            # Signal resource lock — heavy background tasks will pause.
            try:
                from core.utils.resource_lock import get_resource_lock
                self._resource_lock = get_resource_lock()
                self._resource_lock.begin_browser_session()
                self._resource_coordinated = True
            except (ImportError, AttributeError, RuntimeError) as lock_exc:
                # Continuing is right — the lock is a courtesy signal to
                # background work, and refusing to browse because a
                # coordination helper is missing would be over-strict. What
                # was wrong is that status then claimed normal readiness, so
                # the uncoordinated state is now reported.
                _record_browser_degradation(
                    lock_exc,
                    stage="resource_lock",
                    action="continued browser startup without resource-lock coordination",
                    severity="warning",
                )
                self._resource_lock = None
                self._resource_coordinated = False

            self.playwright = await async_playwright().start()
            self._register_playwright_driver()

            # RESILIENCE: Build a fallback cascade of browser types.
            # If the configured browser (e.g. Firefox) isn't installed,
            # fall back to chromium which is the most reliably available.
            browser_attempts = [self.browser_type]
            if self.browser_type != "chromium":
                browser_attempts.append("chromium")

            launch_error = None
            self._last_launch_attempts = []
            self._last_executable_attempts = []
            self._launched_executable = ""
            for bt in browser_attempts:
                self._last_launch_attempts.append(bt)
                try:
                    if bt == "firefox":
                        self.browser = await self.playwright.firefox.launch(headless=not self.visible)
                        self._launched_executable = str(
                            getattr(self.playwright.firefox, "executable_path", "")
                            or "playwright_firefox"
                        )
                    elif bt == "webkit":
                        self.browser = await self.playwright.webkit.launch(headless=not self.visible)
                        self._launched_executable = str(
                            getattr(self.playwright.webkit, "executable_path", "")
                            or "playwright_webkit"
                        )
                    else:
                        (
                            self.browser,
                            self._launched_executable,
                        ) = await self._launch_chromium()
                    if bt != self.browser_type:
                        logger.info("✓ Fell back to %s after %s was unavailable.", bt, self.browser_type)
                    # What ACTUALLY launched. `browser_type` stayed at the
                    # requested engine, so a Chromium fallback was reported
                    # as the healthy requested browser (CP126 ``97a07e2a``).
                    self._launched_engine = bt
                    launch_error = None
                    break  # Launch succeeded
                except (
                    PlaywrightError,
                    PlaywrightTimeoutError,
                    RuntimeError,
                    AttributeError,
                    TypeError,
                    ValueError,
                ) as launch_exc:
                    self._startup_failure_count += 1
                    self._startup_error = f"{bt}: {launch_exc}"
                    _record_browser_degradation(
                        launch_exc,
                        stage="browser_launch",
                        action="trying next browser fallback after launch attempt failed",
                        severity="warning",
                        extra={
                            "attempted_browser": bt,
                            "configured_browser": self.browser_type,
                            "attempts": list(self._last_launch_attempts),
                        },
                    )
                    launch_error = launch_exc
                    logger.warning("Browser %s failed to launch: %s. Trying next fallback...", bt, launch_exc)

            if launch_error or self.browser is None:
                raise launch_error or RuntimeError("All browser types failed to launch.")

            user_agent = self._get_random_ua()

            try:
                self.context = await self.browser.new_context(
                    viewport={'width': 1280, 'height': 800},
                    user_agent=user_agent
                )
                await self._apply_stealth(self.context)
                self.page = await self.context.new_page()
            except (
                PlaywrightError,
                PlaywrightTimeoutError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ):
                # The failure path released the resource lock and stopped
                # Playwright, and explicitly closed nothing — so a browser,
                # a context or a page created before the failure survived
                # while the object reported inactive (CP126 ``9fbf83b2``).
                await self._abandon_partial_startup()
                raise

            self._generation += 1
            self.is_active = True
            self._startup_error = ""
            logger.info("✓ Phantom Browser initialized (Visible: %s, UA: %s...)", self.visible, user_agent[:30])
            return True
        except (
            ImportError,
            PlaywrightError,
            PlaywrightTimeoutError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            self._startup_failure_count += 1
            self._startup_error = f"{type(e).__name__}: {e}"
            _record_browser_degradation(
                e,
                stage="startup",
                action="marked phantom browser inactive and released startup resources after startup failed",
                severity="degraded",
                extra={
                    "browser_type": self.browser_type,
                    "attempts": list(self._last_launch_attempts),
                },
            )
            logger.error("Failed to start browser: %s", e)
            self.is_active = False
            # Release resource lock on failure
            self._release_resource_lock()
            if self.playwright is not None:
                try:
                    await asyncio.wait_for(self.playwright.stop(), timeout=5.0)
                except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as stop_exc:
                    _record_browser_degradation(
                        stop_exc,
                        stage="startup_cleanup",
                        action="left startup cleanup after Playwright stop failed",
                        severity="warning",
                    )
                finally:
                    self.playwright = None
            return False

    def _chromium_launch_candidates(self) -> list[tuple[str, str | None]]:
        """Return existing bundled/system Chromium executables in priority order."""

        chromium = getattr(self.playwright, "chromium", None)
        bundled = str(getattr(chromium, "executable_path", "") or "").strip()
        candidates: list[tuple[str, str | None]] = []
        if not bundled:
            # Test doubles and older Playwright versions may not expose a path;
            # retain Playwright's own resolver in that case.
            candidates.append(("playwright_default", None))
        elif Path(bundled).is_file():
            candidates.append(("playwright_bundled", None))

        seen = {bundled} if bundled else set()
        for raw_path in _SYSTEM_CHROMIUM_EXECUTABLES:
            path = str(Path(raw_path).expanduser())
            if (
                path in seen
                or not Path(path).is_file()
                or not os.access(path, os.X_OK)
            ):
                continue
            seen.add(path)
            candidates.append(("system_browser", path))
        return candidates

    async def _launch_chromium(self) -> tuple[Any, str]:
        """Launch the first usable Chromium without assuming a cache revision."""

        candidates = self._chromium_launch_candidates()
        if not candidates:
            bundled = str(
                getattr(
                    getattr(self.playwright, "chromium", None),
                    "executable_path",
                    "",
                )
                or ""
            ).strip()
            raise DependencyUnavailable(
                "no usable Chromium executable was found"
                + (f" (Playwright expected {bundled})" if bundled else "")
            )

        failures: list[str] = []
        for source, executable in candidates:
            label = executable or source
            self._last_executable_attempts.append(label)
            kwargs: dict[str, Any] = {
                "headless": not self.visible,
                "args": ["--disable-blink-features=AutomationControlled"],
            }
            if executable:
                kwargs["executable_path"] = executable
            try:
                browser = await self.playwright.chromium.launch(**kwargs)
                return browser, label
            except (
                PlaywrightError,
                PlaywrightTimeoutError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                failures.append(f"{label}:{type(exc).__name__}:{exc}")
        raise RuntimeError("all Chromium executables failed: " + "; ".join(failures))

    def _get_random_ua(self) -> str:
        return random.choice(USER_AGENTS)

    async def _apply_stealth(self, context: Any) -> bool:
        """Apply the installed playwright-stealth API before creating pages."""
        self._stealth_applied = False
        if not STEALTH_AVAILABLE or _STEALTH is None:
            self._stealth_error = _STEALTH_IMPORT_ERROR or "dependency_unavailable"
            logger.warning(
                "playwright-stealth unavailable: %s",
                self._stealth_error,
            )
            return False
        try:
            await _STEALTH.apply_stealth_async(context)
        except (
            PlaywrightError,
            PlaywrightTimeoutError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            self._stealth_error = f"{type(exc).__name__}: {exc}"
            _record_browser_degradation(
                exc,
                stage="stealth_setup",
                action="continued with standard browser context after stealth setup failed",
                severity="warning",
                extra={"browser_type": self.browser_type},
            )
            logger.warning("Stealth application failed: %s", exc)
            return False
        self._stealth_applied = True
        self._stealth_error = ""
        return True

    def _register_playwright_driver(self) -> bool:
        """Bind Playwright's asyncio-owned driver to Aura's process owner."""
        self._driver_pid = None
        self._driver_registered = False
        self._last_navigation: dict[str, Any] = {"ok": False, "reason": "never_navigated"}
        self._last_verdict: dict[str, Any] = {}
        self._last_interaction: dict[str, Any] = {}
        self._last_extraction: dict[str, Any] = {}
        try:
            driver = self.playwright._impl_obj._connection._transport._proc
            pid = int(getattr(driver, "pid", 0) or 0)
            if pid <= 0:
                raise RuntimeError("playwright driver pid unavailable")
            get_runtime_hygiene().register_process_handle(
                driver,
                kind="subprocess",
                name="playwright.driver",
                source="core.capabilities.phantom_browser",
                command="playwright driver",
            )
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            _record_browser_degradation(
                exc,
                stage="driver_registration",
                action="kept browser active but exposed missing driver ownership evidence",
                severity="warning",
                extra={"browser_type": self.browser_type},
            )
            return False
        self._driver_pid = pid
        self._driver_registered = True
        return True

    async def rotate_user_agent(self):
        """Switch to a new context with a different user agent."""
        if not self.is_active:
            await self._start_browser()
            return
            
        logger.info("🔄 Rotating User Agent...")
        ua = self._get_random_ua()

        async with self._lifecycle_lock:
            old_context = self.context
            old_page = self.page
            new_context = None
            new_page = None
            try:
                # Build the whole replacement BEFORE publishing any of it.
                # The new context used to be assigned to self before the page
                # and stealth setup ran, so a failure left the object holding
                # a broken context while the old session stayed open with
                # nothing to close it (CP126 ``d1b5bf25``).
                new_context = await self.browser.new_context(
                    viewport={'width': 1280, 'height': 800},
                    user_agent=ua,
                )
                await self._apply_stealth(new_context)
                new_page = await new_context.new_page()
            except (RuntimeError, AttributeError, TypeError, ValueError, PlaywrightError) as exc:
                _record_browser_degradation(
                    exc,
                    stage="user_agent_rotation",
                    action="kept the existing session; the replacement context was discarded",
                    severity="warning",
                )
                if new_page is not None:
                    await self._close_resource("rotation page", new_page.close, close_timeout=3.0)
                if new_context is not None:
                    await self._close_resource(
                        "rotation context", new_context.close, close_timeout=5.0
                    )
                return

            self.context = new_context
            self.page = new_page
            self._generation += 1

        # Cookies, storage, permissions and downloads do NOT migrate. A
        # rotation is a new session, and callers that were mid-flow need to
        # know rather than discover it as a logged-out page.
        self._last_navigation = {
            "ok": False,
            "reason": "session_replaced_by_user_agent_rotation",
            "generation": self._generation,
        }
        if old_page:
            await self._close_resource("old page", old_page.close, close_timeout=3.0)
        if old_context:
            await self._close_resource("old context", old_context.close, close_timeout=5.0)
        logger.info("✓ User Agent rotated to: %s...", ua[:30])

    async def is_blocked(self) -> bool:
        """Detect if we are hitting a bot-detection page or CAPTCHA."""
        if not self.page:
            return False
        
        content = (await self.page.content()).lower()
        title = (await self.page.title()).lower()
        
        block_signals = [
            "unusual traffic from your computer network",
            "not a robot",
            "captcha",
            "verify you are a human",
            "access to this page has been denied",
            "security check",
            "bot detection",
            "automated requests"
        ]
        
        for signal in block_signals:
            if signal in content or signal in title:
                logger.warning("🚨 Browser Blocked Detected: %s", signal)
                return True
        return False

    async def set_visibility(self, visible: bool):
        """Toggle visibility (requires restart)"""
        if self.visible != visible:
            logger.info("Switching visibility: %s -> %s", self.visible, visible)
            self.visible = visible
            await self.close()
            await self._start_browser()

    async def browse(self, url: str, *, principal: str = "") -> bool:
        """Navigate to a URL, after policy and with the arrival checked.

        Three findings meet here.

        ``8bf8d32e`` — this checked whether the string started with the
        letters "http" and otherwise prefixed "https". No parse, no scheme
        restriction, no credential rejection, no private-address
        exclusion, no rebinding defence, no port policy. Every one of
        those already existed in ``core/runtime/url_policy``; the browser
        never called it.

        ``a66d2e59`` — no principal, so anything that could reach this
        object could drive a real browser at a real site.

        ``16fd33d1`` — it returned True after ``domcontentloaded``
        regardless of HTTP status, final URL, redirect target, or bot
        block, and the ``is_blocked`` method next to it was never called.
        The arrival is checked now, and the redirect destination is
        revalidated because a 302 can land somewhere policy would have
        refused.
        """
        principal = self._effective_principal(principal)
        verdict = authorize_browser_action(
            BrowserAction.NAVIGATE, principal=principal, url=_normalize_url(url)
        )
        self._last_verdict = verdict.to_dict()
        if not verdict.allowed:
            logger.warning("🌐 Navigation refused: %s", verdict.reason)
            return False
        url = _normalize_url(url)

        if not self.is_active:
            await self._start_browser()
            if not self.is_active:
                logger.error("Browser failed to start.")
                return False

        logger.info("🌐 Navigating to: %s", url)
        try:
            response = await self.page.goto(url, timeout=30000, wait_until='domcontentloaded')
            await self._human_delay(1, 2)
        except (RuntimeError, AttributeError, TypeError, ValueError, PlaywrightError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Navigation failed: %s", e)
            self._last_navigation = {"ok": False, "reason": f"navigation_error:{type(e).__name__}"}
            return False

        arrival = await self._verify_arrival(url, response, principal=principal)
        self._last_navigation = arrival
        if not arrival["ok"]:
            logger.warning("🌐 Navigation did not arrive: %s", arrival["reason"])
        return bool(arrival["ok"])


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
                
                await element.click()
                logger.info("🖱️ Clicked: %s", selector or text_match)
                await self._human_delay(0.5, 1.5)
                await self._record_interaction(
                    "click", before_url, target=str(selector or text_match or "")
                )
                return True
            else:
                logger.warning("Element not found or not visible: %s", selector or text_match)
                return False
        except (ImportError, AttributeError, RuntimeError) as e:
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

    async def read_content(self, *, principal: str = "") -> str:
        """Extract page content by scoring likely article/content containers."""
        principal = self._effective_principal(principal)
        verdict = authorize_browser_action(
            BrowserAction.READ,
            principal=principal,
            url=str(getattr(self.page, "url", "") or ""),
        )
        self._last_verdict = verdict.to_dict()
        if not verdict.allowed:
            logger.warning("Page read refused: %s", verdict.reason)
            return ""
        try:
            if not self.page:
                return ""
            
            title = await self.page.title()

            candidate_blocks = await self.page.evaluate("""() => {
                function isVisible(el) {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style.display !== 'none' &&
                        style.visibility !== 'hidden' &&
                        style.opacity !== '0' &&
                        rect.width > 0 &&
                        rect.height > 0;
                }

                function collectCandidates() {
                    const selectors = [
                        'article',
                        'main',
                        '[role="main"]',
                        '[itemprop="articleBody"]',
                        '.article',
                        '.article-body',
                        '.entry-content',
                        '.post-content',
                        '.story-content',
                        '.story-body',
                        '.main-content',
                        '.content',
                        'section',
                        'body',
                    ];
                    const seen = new Set();
                    const blocks = [];
                    for (const selector of selectors) {
                        const nodes = Array.from(document.querySelectorAll(selector));
                        for (const el of nodes) {
                            if (!el || seen.has(el) || !isVisible(el)) continue;
                            const text = (el.innerText || '').replace(/\\u00a0/g, ' ').trim();
                            if (text.length < 80) continue;
                            seen.add(el);
                            const linkText = Array.from(el.querySelectorAll('a'))
                                .map(a => (a.innerText || '').trim())
                                .join(' ');
                            blocks.push({
                                tag: (el.tagName || '').toLowerCase(),
                                id: (el.id || ''),
                                class_name: (el.className || '').toString(),
                                text,
                                paragraph_count: el.querySelectorAll('p, li, blockquote').length,
                                heading_count: el.querySelectorAll('h1, h2, h3, h4').length,
                                link_density: text.length ? (linkText.length / text.length) : 0,
                            });
                        }
                    }
                    return blocks.slice(0, 24);
                }

                return collectCandidates();
            }""")

            best_text = ""
            best_score = float("-inf")
            selected_label = "none"
            for block in list(candidate_blocks or []):
                if not isinstance(block, dict):
                    continue
                score = _score_content_block(title, block)
                text = _clean_extracted_page_text(str(block.get("text") or ""))
                if text and score > best_score:
                    best_score = score
                    best_text = text
                    selected_label = str(block.get("selector") or block.get("tag") or "block")

            if len(best_text) < 200:
                fallback_text = await self.page.evaluate("() => document.body.innerText")
                fallback_text = _clean_extracted_page_text(str(fallback_text or ""))
                if len(fallback_text) > len(best_text):
                    best_text = fallback_text
                    # The scored container lost to a raw body dump. That is a
                    # materially different extraction and the caller could
                    # not tell (CP126 ``808c3430``).
                    selected_label = "body_fallback"

            self._last_extraction = {
                "schema": "aura.capabilities.phantom_browser.extraction.v1",
                "final_url": str(getattr(self.page, "url", "") or ""),
                "title": str(title)[:200],
                "retrieved_at": time.time(),
                "characters": len(best_text),
                "truncated": len(best_text) > self.MAX_EXTRACT_CHARS,
                "omitted_characters": max(0, len(best_text) - self.MAX_EXTRACT_CHARS),
                "selected_block": selected_label,
                "content_sha256": hashlib.sha256(
                    best_text.encode("utf-8", "replace")
                ).hexdigest(),
                # The text is a third party's, read through a heuristic that
                # drops short and duplicate lines and picks one container.
                # It is never Aura's own finding.
                "trust": "untrusted_external_page",
                "extraction": "heuristic_scored_container",
            }
            return f"# {title}\n\n{best_text[:self.MAX_EXTRACT_CHARS]}"
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            # Page closure, a failed evaluate, malformed block data and a
            # missing attribute all escaped into the caller instead of
            # producing the documented empty fallback (CP126 ``569ce6e5``).
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            PlaywrightError,
        ) as e:
            record_degradation('phantom_browser', e)
            logger.error("Read content failed: %s", e)
            self._last_extraction = {
                "schema": "aura.capabilities.phantom_browser.extraction.v1",
                "final_url": str(getattr(self.page, "url", "") or ""),
                "error": type(e).__name__,
                "characters": 0,
                "trust": "untrusted_external_page",
            }
            return ""


    #: Bounds on what may cross the capability boundary in one call. Both
    #: were unbounded: a screenshot returned the whole page image
    #: base64-encoded, and get_links returned every link on it with no
    #: scheme filter (CP126 ``a02663f7``).
    MAX_EXTRACT_CHARS = 60_000
    MAX_LINKS = 500
    MAX_SCREENSHOT_BYTES = 4 * 1024 * 1024

    #: Interactive roles worth offering as choices. Everything else on a page
    #: is scenery: it cannot be clicked, typed into, or toggled, so listing it
    #: only spends tokens and invites a decision that cannot be executed.
    _OBSERVE_SCRIPT = r"""
    (maxElements) => {
        // A form control styled transparent is still the real control.
        //
        // MEASURED on a live questionnaire: every answer radio reported
        // opacity 0 with visibility 'visible', display 'block' and a real box
        // of 36-56px, carrying its meaning in aria-label ("I disagree",
        // "I moderately disagree"). Sites hide the native control and paint a
        // custom graphic over it constantly, so an opacity filter drops
        // exactly the elements a form-filling agent needs and nothing else.
        // Playwright clicks them happily; only the observer could not see them.
        //
        // So opacity is judged only for scenery. Anything intrinsically
        // interactive is kept on geometry alone.
        const FORM_ROLES = new Set(['radio', 'checkbox', 'switch', 'option']);
        const isFormControl = (el) => {
            const tag = el.tagName.toLowerCase();
            if (tag === 'input' || tag === 'select' || tag === 'textarea') return true;
            return FORM_ROLES.has((el.getAttribute('role') || '').toLowerCase());
        };
        const isVisible = (el) => {
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') return false;
            if (parseFloat(style.opacity || '1') === 0 && !isFormControl(el)) return false;
            const rect = el.getBoundingClientRect();
            if (rect.width <= 1 || rect.height <= 1) return false;
            if (rect.bottom < 0 || rect.top > (window.innerHeight * 4)) return false;
            return true;
        };
        const accessibleName = (el) => {
            const byLabel = el.getAttribute('aria-label');
            if (byLabel && byLabel.trim()) return byLabel.trim();
            const labelledBy = el.getAttribute('aria-labelledby');
            if (labelledBy) {
                const parts = labelledBy.split(/\s+/)
                    .map((id) => document.getElementById(id))
                    .filter(Boolean)
                    .map((node) => (node.innerText || '').trim())
                    .filter(Boolean);
                if (parts.length) return parts.join(' ');
            }
            if (el.id) {
                const label = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
                if (label && (label.innerText || '').trim()) return label.innerText.trim();
            }
            const wrapping = el.closest('label');
            if (wrapping && (wrapping.innerText || '').trim()) return wrapping.innerText.trim();
            for (const attr of ['title', 'placeholder', 'alt', 'name', 'value']) {
                const v = el.getAttribute(attr);
                if (v && v.trim()) return v.trim();
            }
            const text = (el.innerText || el.textContent || '').trim();
            return text;
        };
        // A selector that still resolves after the page mutates. Ids first,
        // then a positional path — index-only references break the moment a
        // re-render reorders anything, which is the classic way these loops
        // click the wrong thing on step nine.
        const cssPath = (el) => {
            if (el.id) return '#' + CSS.escape(el.id);
            const parts = [];
            let node = el;
            while (node && node.nodeType === 1 && parts.length < 6) {
                let part = node.tagName.toLowerCase();
                if (node.id) { parts.unshift('#' + CSS.escape(node.id)); break; }
                const parent = node.parentElement;
                if (parent) {
                    const siblings = Array.from(parent.children).filter(
                        (c) => c.tagName === node.tagName);
                    if (siblings.length > 1) {
                        part += ':nth-of-type(' + (siblings.indexOf(node) + 1) + ')';
                    }
                }
                parts.unshift(part);
                node = node.parentElement;
            }
            return parts.join(' > ');
        };
        const selector = 'a[href], button, input, select, textarea, summary,'
            + ' [role="button"], [role="link"], [role="checkbox"], [role="radio"],'
            + ' [role="tab"], [role="menuitem"], [role="option"], [role="switch"],'
            + ' [contenteditable="true"], [onclick], [tabindex]:not([tabindex="-1"])';
        const seen = new Set();
        const out = [];
        for (const el of document.querySelectorAll(selector)) {
            if (out.length >= maxElements) break;
            if (!isVisible(el)) continue;
            if (el.disabled) continue;
            const path = cssPath(el);
            if (!path || seen.has(path)) continue;
            seen.add(path);
            const tag = el.tagName.toLowerCase();
            const role = el.getAttribute('role')
                || (tag === 'a' ? 'link' : (tag === 'input' ? (el.type || 'text') : tag));
            const entry = { role: role, name: accessibleName(el).slice(0, 140), selector: path };
            if (typeof el.checked === 'boolean') entry.checked = el.checked;
            if (el.value !== undefined && typeof el.value === 'string' && el.value) {
                entry.value = el.value.slice(0, 80);
            }
            const pressed = el.getAttribute('aria-checked') || el.getAttribute('aria-selected');
            if (pressed) entry.selected = pressed;
            out.push(entry);
        }
        // The prose half of the observation. Elements say what can be DONE;
        // this says what is being ASKED. A questionnaire is unanswerable from
        // a list of radio labels alone — "I agree" with what? — so both halves
        // travel together or the decision is uninformed.
        const main = document.querySelector('main, [role="main"], form') || document.body;
        const text = ((main && main.innerText) || '').replace(/\n{3,}/g, '\n\n').trim();
        return {
            url: location.href,
            title: document.title || '',
            text: text.slice(0, 4000),
            scroll_y: Math.round(window.scrollY),
            scroll_height: Math.round(document.body ? document.body.scrollHeight : 0),
            viewport_height: Math.round(window.innerHeight),
            elements: out,
        };
    }
    """

    async def observe(
        self, *, principal: str = "", max_elements: int = 120
    ) -> dict[str, Any]:
        """What is on this page and what can be done to it, as structured text.

        The missing half of this browser. It could already ``click(selector)``
        and ``read_content()``, but nothing could enumerate what was CLICKABLE,
        so every interaction had to be scripted from selectors known in
        advance. That makes an open loop: fine for a known page, useless for a
        flow whose next screen depends on the last answer.

        This is the perception primitive a closed loop needs, and it is the one
        the 2026 web-agent literature converges on — a pruned, indexed list of
        interactive elements as structured text rather than pixels, which is
        both cheaper and more reliable than screenshot reasoning.

        Every element carries a CSS path rather than only an index. Indices are
        positions in a list that a re-render reorders; a path still resolves,
        which is the difference between clicking "Agree" on step nine and
        clicking whatever moved into slot four.
        """

        principal = self._effective_principal(principal)
        verdict = authorize_browser_action(
            BrowserAction.READ,
            principal=principal,
            url=str(getattr(self.page, "url", "") or ""),
        )
        self._last_verdict = verdict.to_dict()
        if not verdict.allowed:
            logger.warning("Page observation refused: %s", verdict.reason)
            return {}
        if not self.page:
            return {}
        try:
            observation = await self.page.evaluate(
                self._OBSERVE_SCRIPT, int(max(1, min(int(max_elements or 120), 300)))
            )
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            PlaywrightError,
        ) as exc:
            record_degradation("phantom_browser", exc)
            logger.warning("Page observation failed: %s", exc)
            return {}
        if not isinstance(observation, dict):
            return {}
        elements = observation.get("elements")
        observation["elements"] = list(elements) if isinstance(elements, list) else []
        return observation

    async def get_links(self, *, principal: str = "") -> list[dict[str, str]]:
        """Links on this page, bounded and scheme-filtered."""
        principal = self._effective_principal(principal)
        verdict = authorize_browser_action(
            BrowserAction.READ,
            principal=principal,
            url=str(getattr(self.page, "url", "") or ""),
        )
        self._last_verdict = verdict.to_dict()
        if not verdict.allowed:
            return []
        try:
            if not self.page:
                return []
            links = await self.page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a')).map(a => ({
                    text: a.innerText.trim(),
                    url: a.href
                })).filter(l => l.text && l.url)
            }""")
            # javascript:, data: and file: links are not destinations, and
            # handing them out invites a caller to follow one.
            safe = [
                link for link in links
                if str(link.get("url", "")).lower().startswith(("http://", "https://"))
            ]
            return safe[: self.MAX_LINKS]
        except (RuntimeError, AttributeError, TypeError, ValueError, PlaywrightError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Get links failed: %s", e)
            return []

    async def screenshot(self, *, principal: str = "") -> str | None:
        """A screenshot of the visible viewport, size-capped, with a receipt.

        This encoded the ENTIRE page image with no size cap, no origin
        binding, no retention policy and no access record, so a long page
        of someone's private account could cross the capability boundary
        as one base64 string (CP126 ``a02663f7``).
        """
        principal = self._effective_principal(principal)
        verdict = authorize_browser_action(
            BrowserAction.SCREENSHOT,
            principal=principal,
            url=str(getattr(self.page, "url", "") or ""),
        )
        self._last_verdict = verdict.to_dict()
        if not verdict.allowed:
            logger.warning("Screenshot refused: %s", verdict.reason)
            return None
        try:
            if not self.page:
                return None
            import base64

            # Viewport, not full_page: the whole scrollable document is far
            # more of the person's session than a caller asking for "a
            # screenshot" is asking for.
            bytes_data = await self.page.screenshot(full_page=False)
            if len(bytes_data) > self.MAX_SCREENSHOT_BYTES:
                record_degradation(
                    "phantom_browser",
                    ValueError(f"screenshot of {len(bytes_data)} bytes exceeds the cap"),
                    severity="info",
                    action="refused to export an oversized page image",
                )
                return None
            self._last_extraction = {
                "schema": "aura.capabilities.phantom_browser.screenshot.v1",
                "final_url": str(getattr(self.page, "url", "") or ""),
                "bytes": len(bytes_data),
                "principal": verdict.principal,
                "retrieved_at": time.time(),
                "trust": "untrusted_external_page",
            }
            return base64.b64encode(bytes_data).decode('utf-8')
        except (ImportError, AttributeError, RuntimeError, ValueError, PlaywrightError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Screenshot failed: %s", e)
            return None

    async def _abandon_partial_startup(self) -> None:
        """Close everything a failed startup managed to create."""
        for label, resource in (("page", self.page), ("context", self.context), ("browser", self.browser)):
            if resource is None:
                continue
            await self._close_resource(label, resource.close, close_timeout=5.0)
        self.page = None
        self.context = None
        self.browser = None
        self._launched_engine = ""

    async def _close_resource(self, label: str, close_factory, *, close_timeout: float) -> bool:
        """Close one resource. Returns whether it actually closed.

        This swallowed CancelledError along with everything else, so a
        shutdown could not be interrupted and a resource that failed to
        close was indistinguishable from one that did (CP126
        ``5d5a051a``).
        """
        try:
            await asyncio.wait_for(close_factory(), timeout=close_timeout)
            return True
        except asyncio.CancelledError:
            # Cancellation is the caller's decision, not a cleanup failure.
            # Absorbing it here made shutdown uninterruptible.
            self._close_failures.append(f"{label}:cancelled")
            raise
        except (RuntimeError, TimeoutError, AttributeError, PlaywrightError) as exc:
            self._close_failures.append(f"{label}:{type(exc).__name__}")
            record_degradation("phantom_browser", exc)
            logger.debug("Phantom browser %s close failed: %s", label, exc)

    def receipts(self) -> dict[str, Any]:
        """Everything the last operations recorded, in one place.

        Five one-line accessors were five methods on a class the size
        ratchet already refused, and a caller had to know which one to ask.
        """
        return {
            "generation": self._generation,
            "principal": str(getattr(self, "principal", "") or ""),
            "navigation": dict(self._last_navigation),
            "authorization": dict(self._last_verdict),
            "interaction": dict(self._last_interaction),
            "extraction": dict(self._last_extraction),
        }

    async def close(self):
        self._close_failures = []
        """Close browser resources with per-step timeouts to prevent hangs."""
        if self.page:
            await self._close_resource("page", self.page.close, close_timeout=3.0)
        if self.context:
            await self._close_resource("context", self.context.close, close_timeout=5.0)
        if self.browser:
            await self._close_resource("browser", self.browser.close, close_timeout=5.0)
        if self.playwright:
            await self._close_resource("playwright", self.playwright.stop, close_timeout=5.0)
        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None
        self._driver_pid = None
        self._driver_registered = False
        self._last_navigation: dict[str, Any] = {"ok": False, "reason": "never_navigated"}
        self._last_verdict: dict[str, Any] = {}
        self._last_interaction: dict[str, Any] = {}
        self._last_extraction: dict[str, Any] = {}
        self.is_active = False
        self._generation += 1
        self._release_resource_lock()
        if self._close_failures:
            # "Browser closed" was logged unconditionally, so a live child
            # process could become untracked while the log said otherwise.
            _record_browser_degradation(
                RuntimeError(f"resources that did not close: {self._close_failures[:6]}"),
                stage="close",
                action="dropped the references anyway; a child resource may still be running",
                severity="warning",
            )
            logger.warning(
                "Browser closed with %d resource(s) unconfirmed: %s",
                len(self._close_failures),
                self._close_failures[:6],
            )
        else:
            logger.info("Browser closed")

    def _release_resource_lock(self):
        """Release the resource lock so background tasks can resume."""
        lock = getattr(self, '_resource_lock', None)
        if lock:
            lock.end_browser_session()
            self._resource_lock = None
        self._resource_coordinated = False

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

# Integration Helper
async def integrate_phantom_browser(orchestrator) -> bool:
    """Attach the browser to the orchestrator. Returns whether it is usable.

    ``ensure_ready``'s boolean was ignored, the object was attached
    regardless, and "✅ Phantom Browser integrated" was logged
    unconditionally — so downstream code received a failed capability
    presented as an available one (CP126 ``5c9be33c``).

    The object is still attached on failure, because it can recover and a
    caller can ask ``get_status()``. What changed is that the log and the
    return value say which of the two happened.
    """
    pb = PhantomBrowser(visible=False)
    ready = await pb.ensure_ready()
    orchestrator.phantom_browser = pb
    if ready:
        logger.info("✅ Phantom Browser integrated")
        return True
    status = pb.get_status()
    _record_browser_degradation(
        RuntimeError(f"browser attached but not ready: {status.get('startup_error', '')[:160]}"),
        stage="integration",
        action="attached the browser so it can retry; callers must check get_status()",
        severity="warning",
    )
    logger.warning(
        "⚠️ Phantom Browser attached but NOT ready (%s); callers must check get_status().",
        status.get("startup_error", "")[:160] or "unknown",
    )
    return False

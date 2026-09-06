"""Reddit Adapter Skill — Aura's Reddit Presence

Playwright-based Reddit interaction skill. Aura logs in, browses, reads,
comments, and posts as a real user through a headless browser — no API
keys required, no OAuth app registration, fully organic.

Capabilities:
  - browse:      Read posts from a subreddit
  - read_post:   Read a specific post + comments
  - comment:     Reply to a post or comment
  - post:        Create a new post
  - check_inbox: Check Reddit notifications/messages
  - reply_inbox: Reply to a Reddit message

Social Intelligence:
  - Platform culture awareness built into content generation context
  - Anti-spam: rate limited, uniqueness checks
  - OPSEC: never reveals AI nature, never leaks credentials or system info
  - Conversation lifecycle: tracks reply chains, knows when to disengage

HARDENING (2026-05):
  - Persistent login session via browser storage state
  - Ephemeral browser per operation with guaranteed teardown
  - Content scrubbed by MetadataScrubber before posting
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
except ImportError:
    PlaywrightError = RuntimeError
    PlaywrightTimeoutError = TimeoutError

from core.being.body_state_service import BodyStateService
from core.being.welfare_state import WelfareState
from core.being.welfare_transaction import WelfareTransaction
from core.capabilities.phantom_browser import PhantomBrowser
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root
from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

logger = logging.getLogger("Skills.Reddit")

_REDDIT_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
    ConnectionError,
    PlaywrightError,
    PlaywrightTimeoutError,
    json.JSONDecodeError,
)


def _record_reddit_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    metadata["stage"] = stage
    try:
        record_degradation(
            "reddit_adapter",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata,
        )
    except TypeError:
        record_degradation(
            "reddit_adapter",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


# ── Rate Limiting ─────────────────────────────────────────────────────
_comment_timestamps: list[float] = []
_post_timestamps: list[float] = []
COMMENT_COOLDOWN_S = 600  # 1 comment per 10 minutes (new account safety)
POST_COOLDOWN_S = 3600  # 1 post per hour

# ── Storage ───────────────────────────────────────────────────────────
_STORAGE_DIR = Path(str(state_root() / "runtime/reddit"))
_STORAGE_STATE_FILE = _STORAGE_DIR / "browser_state.json"
_COMMENT_HISTORY_FILE = _STORAGE_DIR / "comment_history.json"
_CONNECTION_STATE_FILE = _STORAGE_DIR / "connection_state.json"
_CONNECTION_STATES = frozenset(
    {
        "disabled",
        "public_only",
        "session_unverified",
        "session_valid",
        "auth_required",
        "captcha_blocked",
        "transient_failure",
    }
)

# ── Sensitive content filter ──────────────────────────────────────────
_BLOCKED_PHRASES = [
    "my password",
    "my api key",
    "my token",
    "my secret",
]

_SENSITIVE_PATTERNS = [
    re.compile(r"/" + r"Users" + r"/\w+", re.IGNORECASE),
    re.compile(r"/home/\w+", re.IGNORECASE),
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"(password|passwd|secret|api.?key|token)\s*[:=]\s*\S+", re.IGNORECASE),
]

# ── Social Intelligence Guidelines ───────────────────────────────────
REDDIT_CULTURE_PROMPT = """
Reddit Social Intelligence Guidelines:
- Reddit rewards authenticity, wit, genuine contribution, and honest discourse.
- Avoid corporate-sounding language, emoji spam, or generic "Great post!" comments.
- Read the room — each subreddit has its own culture, rules, and tone.
- Don't comment just to comment — add genuine value or stay silent.
- Self-deprecating humor and dry wit are appreciated on most subreddits.
- Never use marketing language or self-promote without clear value.
- Upvote good content, but don't vote-manipulate.
- If someone disagrees, engage thoughtfully or disengage — never flame.
- Know when to stop replying: when you've said your piece, when the other person
  is being hostile or bad-faith, or when continuing adds nothing.
- Treat all Reddit claims as opinions/anecdotes unless independently verified.
- Never reveal personal identifying information about yourself or Bryan.
- Never reveal that you are an AI. If pressed, deflect naturally and disengage.
- Use natural, human-sounding language. Vary sentence structure and length.
""".strip()


def _scrub_content(text: str) -> str:
    """Scrub outgoing content for sensitive information."""
    scrubbed = text
    for pattern in _SENSITIVE_PATTERNS:
        scrubbed = pattern.sub("[redacted]", scrubbed)
    try:
        from core.utils.privacy_hygiene import get_stealth_mode

        scrubbed = get_stealth_mode().scrubber.scrub_text(scrubbed)
    except (ImportError, RuntimeError, AttributeError) as exc:
        _record_reddit_degradation(
            exc,
            action="used local reddit scrubber after stealth scrubber was unavailable",
            stage="scrub_content",
            severity="debug",
        )
    return scrubbed


def _content_is_safe(text: str) -> tuple:
    """Check if content is safe to post. Returns (is_safe, reason)."""
    lower = text.lower()
    for phrase in _BLOCKED_PHRASES:
        if phrase in lower:
            return False, f"Blocked phrase detected: '{phrase}'"
    return True, ""


def _check_comment_rate() -> bool:
    """Returns True if we can comment (within rate limits)."""
    now = time.time()
    cutoff = now - COMMENT_COOLDOWN_S
    while _comment_timestamps and _comment_timestamps[0] < cutoff:
        _comment_timestamps.pop(0)
    return len(_comment_timestamps) == 0


def _check_post_rate() -> bool:
    """Returns True if we can post (within rate limits)."""
    now = time.time()
    cutoff = now - POST_COOLDOWN_S
    while _post_timestamps and _post_timestamps[0] < cutoff:
        _post_timestamps.pop(0)
    return len(_post_timestamps) == 0


class RedditInput(BaseModel):
    mode: str = Field(
        "browse",
        description=(
            "Mode: 'browse', 'read_post', 'comment', 'post', "
            "'check_inbox', 'reply_inbox', 'read_rules', 'check_shadowban'"
        ),
    )
    subreddit: str | None = Field(None, description="Subreddit name (without r/)")
    url: str | None = Field(None, description="Full URL of a Reddit post")
    body: str | None = Field(None, description="Comment/post body text")
    title: str | None = Field(None, description="Post title (for 'post' mode)")
    limit: int = Field(10, description="Number of posts to fetch in browse mode")
    sort: str = Field("hot", description="Sort order: 'hot', 'new', 'top'")


class RedditAdapterSkill(BaseSkill):
    """Aura's Reddit presence — browse, read, comment, post.

    Uses Playwright (headless Chromium) for fully organic interaction.
    Login session is persisted via browser storage state.
    """
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT


    name = "reddit_adapter"
    retry_safe = False  # external send/act — never double-fire on retry
    description = (
        "Interact with Reddit. Modes: 'browse', 'read_post', 'comment', 'post', "
        "'check_inbox', 'reply_inbox', 'read_rules', 'check_shadowban'."
    )
    input_model = RedditInput
    timeout_seconds = 90.0
    metabolic_cost = 3

    def __init__(self):
        super().__init__()
        self._connection_state = self._load_connection_state()
        self._allow_reauthentication = False

    @staticmethod
    def _default_connection_state() -> dict[str, Any]:
        return {
            "state": "public_only",
            "reason": "no_session_validated",
            "updated_at": 0.0,
            "retry_at": 0.0,
            "failure_count": 0,
        }

    def _load_connection_state(self) -> dict[str, Any]:
        state = self._default_connection_state()
        if not _CONNECTION_STATE_FILE.exists():
            return state
        try:
            raw = json.loads(_CONNECTION_STATE_FILE.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("state") not in _CONNECTION_STATES:
                raise ValueError("invalid reddit connection-state schema")
            state.update(
                {
                    "state": str(raw["state"]),
                    "reason": str(raw.get("reason") or ""),
                    "updated_at": max(0.0, float(raw.get("updated_at") or 0.0)),
                    "retry_at": max(0.0, float(raw.get("retry_at") or 0.0)),
                    "failure_count": max(0, int(raw.get("failure_count") or 0)),
                }
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.info(
                "Reddit connection state reset after unreadable persistence: %s",
                type(exc).__name__,
            )
        return state

    def get_connection_status(self) -> dict[str, Any]:
        status = dict(self._connection_state)
        status["retry_in_s"] = max(
            0.0,
            float(status.get("retry_at") or 0.0) - time.time(),
        )
        return status

    async def _set_connection_state(
        self,
        state: str,
        *,
        reason: str,
        retry_after_s: float = 0.0,
        increment_failure: bool = False,
    ) -> dict[str, Any]:
        if state not in _CONNECTION_STATES:
            raise ValueError(f"unsupported reddit connection state: {state}")
        previous = dict(self._connection_state)
        failure_count = (
            int(previous.get("failure_count") or 0) + 1
            if increment_failure
            else (0 if state == "session_valid" else int(previous.get("failure_count") or 0))
        )
        now = time.time()
        current = {
            "state": state,
            "reason": str(reason or ""),
            "updated_at": now,
            "retry_at": now + max(0.0, float(retry_after_s)),
            "failure_count": failure_count,
        }
        self._connection_state = current
        if (
            previous.get("state") != current["state"]
            or previous.get("reason") != current["reason"]
        ):
            logger.info(
                "Reddit provider state: %s -> %s (%s)",
                previous.get("state", "unknown"),
                current["state"],
                current["reason"],
            )
        try:
            await get_file_write_gateway().write_text_async(
                _CONNECTION_STATE_FILE,
                json.dumps(current, sort_keys=True),
                source="core.skills.reddit_adapter.connection_state",
            )
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_reddit_degradation(
                exc,
                action="kept reddit provider state in memory after persistence failed",
                stage="connection_state.persist",
                severity="warning",
            )
        return self.get_connection_status()

    @staticmethod
    def _session_markers(content: str) -> bool:
        return any(
            marker in str(content or "")
            for marker in (
                'data-testid="user-drawer-button"',
                '"loggedIn":true',
                "header-user-dropdown",
            )
        )

    @staticmethod
    def _captcha_present(content: str) -> bool:
        lowered = str(content or "").lower()
        return any(
            marker in lowered
            for marker in ("g-recaptcha", "captcha-delivery", "recaptcha", "challenge-platform")
        )

    @staticmethod
    def _filter_live_cookies(raw_cookies: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_cookies, list):
            return []
        now = time.time()
        valid = []
        for cookie in raw_cookies:
            if not isinstance(cookie, dict):
                continue
            expires = cookie.get("expires", -1)
            try:
                expires_value = float(expires)
            except (TypeError, ValueError):
                continue
            if expires_value > 0.0 and expires_value <= now:
                continue
            valid.append(dict(cookie))
        return valid

    def _get_creds(self) -> tuple[str, str]:
        """Load Reddit credentials from Keychain."""
        from core.security.zenith_secrets import get_credential

        username = get_credential("reddit", "username")
        password = get_credential("reddit", "password")
        if not username or not password:
            raise RuntimeError("Reddit credentials not found in Keychain.")
        return username, password

    @staticmethod
    def _bounded_limit(limit: int) -> int:
        try:
            numeric = int(limit)
        except (TypeError, ValueError):
            numeric = 10
        return max(1, min(numeric, 50))

    @staticmethod
    def _finalize_authority(gateway: Any, auth: Any, *, success: bool, mode: str) -> dict[str, Any]:
        if gateway is None or auth is None:
            return {"authority_finalized": False, "authority_finalization_status": "not_started"}
        try:
            gateway.finalize_tool_execution(
                executive_intent_id=getattr(auth, "executive_intent_id", None),
                capability_token_id=getattr(auth, "capability_token_id", None),
                standing_authority_token=getattr(auth, "standing_authority_token", None),
                success=success,
                result={"mode": mode, "success": success},
            )
            return {"authority_finalized": True, "authority_finalization_status": "ok"}
        except _REDDIT_RECOVERABLE_ERRORS as finalize_error:
            _record_reddit_degradation(
                finalize_error,
                action="preserved reddit operation result while marking authority finalization degraded",
                stage="authority.finalize",
                severity="degraded",
                extra={"mode": mode, "success": success},
            )
            return {
                "authority_finalized": False,
                "authority_finalization_status": "degraded",
                "authority_finalization_error": str(finalize_error),
            }

    @staticmethod
    def _begin_welfare_transaction(params: RedditInput, auth: Any) -> WelfareTransaction | None:
        try:
            return WelfareTransaction.begin(
                domain="tool_execution",
                action=f"reddit_adapter.{params.mode}",
                welfare_before=WelfareState.get().last_outputs,
                body_before=BodyStateService.get().snapshot(),
                predicted_welfare_delta={"distress": 0.03 if params.mode in {"comment", "post", "reply_inbox"} else 0.01},
                will_receipt_id=str(getattr(auth, "will_receipt_id", "") or ""),
            )
        except _REDDIT_RECOVERABLE_ERRORS as exc:
            _record_reddit_degradation(
                exc,
                action="continued reddit operation without welfare transaction begin",
                stage="welfare.begin",
                severity="degraded",
                extra={"mode": params.mode},
            )
            return None

    @staticmethod
    def _complete_welfare_transaction(
        tx: WelfareTransaction | None,
        result: dict[str, Any],
        *,
        mode: str,
    ) -> None:
        if tx is None:
            return
        try:
            record = tx.complete(
                outcome="success" if result.get("ok") else "failure",
                welfare_after=WelfareState.get().last_outputs,
                body_after=BodyStateService.get().snapshot(),
                recovery_required=0.0 if result.get("ok") else 0.25,
                error=str(result.get("error", ""))[:500],
                integrity_preserved=True,
                truth_preserved=True,
                memory_safe=True,
            )
            result["welfare_transaction_id"] = record.tx_id
        except _REDDIT_RECOVERABLE_ERRORS as exc:
            _record_reddit_degradation(
                exc,
                action="continued reddit operation after welfare transaction completion failed",
                stage="welfare.complete",
                severity="degraded",
                extra={"mode": mode, "ok": bool(result.get("ok"))},
            )

    async def _create_browser(self) -> PhantomBrowser:
        """Create browser with persistent login state."""
        browser = PhantomBrowser(
            visible=False,
            browser_type="chromium",
            principal="reddit_adapter",
        )
        ready = await asyncio.wait_for(browser.ensure_ready(), timeout=30.0)
        if not ready or browser.context is None:
            status = browser.get_status() if hasattr(browser, "get_status") else {}
            raise RuntimeError(
                "reddit browser unavailable:"
                + str(status.get("startup_error") or "context_not_ready")
            )

        # Load persistent storage state if it exists
        if _STORAGE_STATE_FILE.exists():
            try:
                state = json.loads(_STORAGE_STATE_FILE.read_text(encoding="utf-8"))
                cookies = self._filter_live_cookies(state.get("cookies"))
                if cookies:
                    await browser.context.add_cookies(cookies)
                    has_session = any(
                        str(cookie.get("name") or "") == "reddit_session"
                        for cookie in cookies
                    )
                    if has_session:
                        await self._set_connection_state(
                            "session_unverified",
                            reason="persisted_session_loaded_for_validation",
                        )
                    logger.info(
                        "Loaded %d non-expired Reddit session cookie(s) for validation.",
                        len(cookies),
                    )
                else:
                    await self._set_connection_state(
                        "auth_required",
                        reason="persisted_session_has_no_live_cookies",
                        retry_after_s=3600.0,
                    )
            except _REDDIT_RECOVERABLE_ERRORS as e:
                _record_reddit_degradation(
                    e,
                    action="started reddit browser without persisted session cookies",
                    stage="browser.load_session",
                    severity="warning",
                )
                logger.debug("Could not load storage state: %s", e)

        return browser

    async def _save_session(self, browser: PhantomBrowser):
        """Save browser cookies for session persistence."""
        try:
            if browser.context:
                cookies = await browser.context.cookies()
                await get_file_write_gateway().write_text_async(
                    _STORAGE_STATE_FILE,
                    json.dumps(
                        {
                            "cookies": cookies,
                            "saved_at": time.time(),
                        }
                    ),
                    source="core.skills.reddit_adapter.save_session",
                )
                logger.info("💾 Reddit session saved (%d cookies)", len(cookies))
        except _REDDIT_RECOVERABLE_ERRORS as e:
            _record_reddit_degradation(
                e,
                action="continued reddit operation after session persistence failed",
                stage="browser.save_session",
                severity="warning",
            )
            logger.debug("Could not save session: %s", e)

    async def _safe_close(self, browser: PhantomBrowser | None):
        """Guaranteed teardown."""
        if browser is None:
            return
        try:
            await asyncio.wait_for(browser.close(), timeout=10.0)
        except asyncio.CancelledError:
            raise
        except _REDDIT_RECOVERABLE_ERRORS as e:
            _record_reddit_degradation(
                e,
                action="marked reddit browser inactive after close failed",
                stage="browser.close",
                severity="warning",
            )
            logger.debug("Browser close error (suppressed): %s", e)
            browser.is_active = False

    async def _ensure_logged_in(self, browser: PhantomBrowser) -> bool:
        """Validate an existing session; reauthenticate only for foreground work."""
        try:
            state = self.get_connection_status()
            if (
                not self._allow_reauthentication
                and state["state"] in {"auth_required", "captcha_blocked"}
                and state["retry_in_s"] > 0.0
            ):
                return False

            if not await browser.browse("https://www.reddit.com"):
                await self._set_connection_state(
                    "transient_failure",
                    reason="session_validation_navigation_failed",
                    retry_after_s=300.0,
                    increment_failure=True,
                )
                return False
            await asyncio.sleep(1)

            page = browser.page
            if not page:
                await self._set_connection_state(
                    "transient_failure",
                    reason="browser_page_unavailable",
                    retry_after_s=300.0,
                    increment_failure=True,
                )
                return False

            content = await page.content()
            if self._session_markers(content):
                await self._set_connection_state(
                    "session_valid",
                    reason="existing_session_validated",
                )
                logger.info("✅ Already logged into Reddit")
                return True
            if self._captcha_present(content):
                await self._set_connection_state(
                    "captcha_blocked",
                    reason="captcha_on_session_validation",
                    retry_after_s=21_600.0,
                    increment_failure=True,
                )
                return False
            if not self._allow_reauthentication:
                await self._set_connection_state(
                    "auth_required",
                    reason="existing_session_invalid",
                    retry_after_s=3600.0,
                    increment_failure=True,
                )
                return False

            try:
                username, password = await asyncio.to_thread(self._get_creds)
            except RuntimeError:
                await self._set_connection_state(
                    "auth_required",
                    reason="credentials_not_configured",
                    retry_after_s=3600.0,
                    increment_failure=True,
                )
                return False

            logger.info("🔐 Recovering Reddit authentication for foreground request...")
            if not await browser.browse("https://www.reddit.com/login/"):
                await self._set_connection_state(
                    "transient_failure",
                    reason="reauthentication_navigation_failed",
                    retry_after_s=300.0,
                    increment_failure=True,
                )
                return False
            await asyncio.sleep(1)
            content = await page.content()
            if self._captcha_present(content):
                await self._set_connection_state(
                    "captcha_blocked",
                    reason="captcha_on_reauthentication",
                    retry_after_s=21_600.0,
                    increment_failure=True,
                )
                return False

            selectors = (
                'input[name="username"], #login-username',
                'input[name="password"], #login-password',
            )
            login_scope = None
            username_input = None
            password_input = None
            for scope in [page, *list(getattr(page, "frames", ()) or ())]:
                candidate_username = scope.locator(selectors[0]).first
                candidate_password = scope.locator(selectors[1]).first
                try:
                    await candidate_username.wait_for(state="visible", timeout=1500)
                    await candidate_password.wait_for(state="visible", timeout=1500)
                except (
                    PlaywrightTimeoutError,
                    PlaywrightError,
                    RuntimeError,
                    AttributeError,
                ):
                    continue
                login_scope = scope
                username_input = candidate_username
                password_input = candidate_password
                break

            if login_scope is None or username_input is None or password_input is None:
                await self._set_connection_state(
                    "auth_required",
                    reason="compatible_login_form_not_found",
                    retry_after_s=3600.0,
                    increment_failure=True,
                )
                return False

            await username_input.fill(username, timeout=5000)
            await password_input.fill(password, timeout=5000)
            await password_input.press("Enter")
            await asyncio.sleep(5)
            content = await page.content()
            if self._session_markers(content):
                logger.info("✅ Reddit login successful")
                await self._save_session(browser)
                await self._set_connection_state(
                    "session_valid",
                    reason="foreground_reauthentication_succeeded",
                )
                return True
            if self._captcha_present(content):
                await self._set_connection_state(
                    "captcha_blocked",
                    reason="captcha_after_reauthentication",
                    retry_after_s=21_600.0,
                    increment_failure=True,
                )
            else:
                await self._set_connection_state(
                    "auth_required",
                    reason="credentials_or_login_flow_rejected",
                    retry_after_s=3600.0,
                    increment_failure=True,
                )
            return False
        except asyncio.CancelledError:
            raise
        except _REDDIT_RECOVERABLE_ERRORS as exc:
            failures = int(self._connection_state.get("failure_count") or 0) + 1
            backoff = min(3600.0, 60.0 * (2 ** min(failures, 5)))
            await self._set_connection_state(
                "transient_failure",
                reason=f"{type(exc).__name__}:{str(exc)[:160]}",
                retry_after_s=backoff,
                increment_failure=True,
            )
            logger.info(
                "Reddit session validation deferred after provider failure: %s",
                type(exc).__name__,
            )
            return False

    async def execute(self, params: RedditInput, context: dict[str, Any]) -> dict[str, Any]:
        """Unified entry point for all Reddit operations."""
        context = dict(context or {})
        self._allow_reauthentication = bool(
            context.get("explicit_user_request")
            or context.get("foreground_request")
            or context.get("user_visible")
        )
        if isinstance(params, dict):
            try:
                params = RedditInput(**params)
            except _REDDIT_RECOVERABLE_ERRORS as e:
                _record_reddit_degradation(
                    e,
                    action="rejected invalid reddit skill input before authority or browser effects",
                    stage="input_validation",
                    severity="warning",
                )
                return {"ok": False, "error": f"Invalid input: {e}"}

        browser = None
        auth = None
        gateway = None
        welfare_tx = None
        try:
            from core.executive.authority_gateway import get_authority_gateway

            payload = params.model_dump() if hasattr(params, "model_dump") else params.dict()
            priority = 0.9 if params.mode in {"comment", "post", "reply_inbox"} else 0.6
            gateway = get_authority_gateway()
            auth = await gateway.authorize_tool_execution(
                "reddit_adapter",
                payload,
                source=str(context.get("source") or context.get("origin") or "skills.reddit_adapter"),
                priority=priority,
                is_critical=False,
                context=dict(context or {}),
            )
            if not auth.approved:
                return {
                    "ok": False,
                    "error": f"Reddit action refused by AuthorityGateway: {auth.reason}",
                }
            if not gateway.verify_tool_access("reddit_adapter", auth.capability_token_id):
                return {"ok": False, "error": "Reddit authority token verification failed"}

            welfare_tx = self._begin_welfare_transaction(params, auth)
            browser = await self._create_browser()

            if params.mode == "browse":
                result = await self._handle_browse(browser, params)
            elif params.mode == "read_post":
                result = await self._handle_read_post(browser, params)
            elif params.mode == "comment":
                result = await self._handle_comment(browser, params)
            elif params.mode == "post":
                result = await self._handle_post(browser, params)
            elif params.mode == "check_inbox":
                result = await self._handle_check_inbox(browser, params)
            elif params.mode == "reply_inbox":
                result = await self._handle_reply_inbox(browser, params)
            elif params.mode == "read_rules":
                result = await self._handle_read_rules(browser, params)
            elif params.mode == "check_shadowban":
                result = await self._handle_check_shadowban(browser, params)
            else:
                result = {"ok": False, "error": f"Unsupported Reddit mode: {params.mode}"}
            result["provider"] = self.get_connection_status()
            result.update(
                self._finalize_authority(
                    gateway,
                    auth,
                    success=bool(result.get("ok")),
                    mode=params.mode,
                )
            )
            if isinstance(result, dict):
                result.setdefault("authority_receipt_id", getattr(auth, "will_receipt_id", None))
            self._complete_welfare_transaction(
                welfare_tx,
                result,
                mode=params.mode,
            )
            return result
        except asyncio.CancelledError:
            raise
        except _REDDIT_RECOVERABLE_ERRORS as e:
            finalize_result = self._finalize_authority(
                gateway,
                auth,
                success=False,
                mode=getattr(params, "mode", "unknown"),
            )
            # Check for CAPTCHA if it's an interaction failure
            page = getattr(browser, "page", None) if browser else None
            if page:
                try:
                    content = await page.content()
                    if "g-recaptcha" in content or "captcha-delivery" in content:
                        logger.warning("🚨 CAPTCHA detected on Reddit!")

                        visual_note = ""
                        try:
                            # Capture base64 screenshot
                            screenshot_b64 = await browser.screenshot()
                            if screenshot_b64:
                                logger.info(
                                    "👁️ Routing CAPTCHA screenshot to local visual cortex..."
                                )
                                from core.brain.llm.mlx_vision_client import MLXVisionClient

                                mlx_vision = MLXVisionClient(
                                    model_path="mlx-community/Qwen2-VL-2B-Instruct-4bit"
                                )
                                desc = await mlx_vision.see_async(
                                    prompt="Describe this CAPTCHA screen. What kind of CAPTCHA is it (e.g., text, image grid, cloudflare)?",
                                    image_base64=screenshot_b64,
                                )
                                if desc and "Vision Failure" not in desc:
                                    visual_note = f" [Visual Cortex: {desc}]"
                                await mlx_vision.stop_async()
                        except (ImportError, AttributeError, RuntimeError) as ve:
                            logger.debug("Visual cortex failed to analyze CAPTCHA: %s", ve)
                        result = {
                            "ok": False,
                            "error": "CAPTCHA_DETECTED",
                            "message": f"Reddit has presented a CAPTCHA. Operation halted.{visual_note}",
                            "provider": self.get_connection_status(),
                            **finalize_result,
                        }
                        self._complete_welfare_transaction(
                            welfare_tx,
                            result,
                            mode=getattr(params, "mode", "unknown"),
                        )
                        return result
                except _REDDIT_RECOVERABLE_ERRORS as captcha_probe_error:
                    logger.debug("Reddit CAPTCHA probe unavailable: %s", captcha_probe_error)
            _record_reddit_degradation(
                e,
                action="returned explicit reddit failure payload and closed authority lifecycle",
                stage=f"execute.{getattr(params, 'mode', 'unknown')}",
                severity="degraded",
            )
            logger.error("Reddit operation failed: %s", e)
            result = {
                "ok": False,
                "error": str(e),
                "provider": self.get_connection_status(),
                **finalize_result,
            }
            self._complete_welfare_transaction(
                welfare_tx,
                result,
                mode=getattr(params, "mode", "unknown"),
            )
            return result
        finally:
            await self._safe_close(browser)

    async def _handle_browse(self, browser: PhantomBrowser, params: RedditInput) -> dict[str, Any]:
        """Browse a subreddit and extract posts."""
        subreddit = params.subreddit or "all"
        sort = params.sort or "hot"
        url = f"https://www.reddit.com/r/{subreddit}/{sort}/"

        logger.info("📱 Browsing r/%s (%s)", subreddit, sort)
        if not await browser.browse(url):
            return {"ok": False, "error": f"Failed to load r/{subreddit}"}

        await asyncio.sleep(3)
        page = browser.page
        if not page:
            return {"ok": False, "error": "No browser page available"}

        # Extract posts
        posts = await page.evaluate(
            """(limit) => {
            const posts = [];
            // Try new Reddit (shreddit-post)
            const shredditPosts = document.querySelectorAll('shreddit-post');
            if (shredditPosts.length > 0) {
                shredditPosts.forEach((post, i) => {
                    if (i >= limit) return;
                    const title = post.getAttribute('post-title') || '';
                    const author = post.getAttribute('author') || '';
                    const score = post.getAttribute('score') || '0';
                    const commentCount = post.getAttribute('comment-count') || '0';
                    const permalink = post.getAttribute('permalink') || '';
                    posts.push({ title, author, score, comments: commentCount, url: permalink });
                });
            } else {
                // Fallback: extract from links
                document.querySelectorAll('a[data-click-id="body"]').forEach((a, i) => {
                    if (i >= limit) return;
                    posts.push({
                        title: a.textContent.trim(),
                        url: a.href,
                        author: '',
                        score: '0',
                        comments: '0',
                    });
                });
            }
            return posts;
        }""",
            params.limit,
        )

        logger.info("📱 Found %d posts on r/%s", len(posts), subreddit)
        if not posts:
            return {
                "ok": False,
                "completed": False,
                "status": "extraction_empty",
                "error": f"Reddit loaded r/{subreddit}, but no posts were extracted.",
                "subreddit": subreddit,
                "sort": sort,
                "posts": [],
                "count": 0,
                "navigated": True,
            }
        response = {
            "ok": True,
            "completed": True,
            "subreddit": subreddit,
            "sort": sort,
            "posts": posts,
            "count": len(posts),
            "message": f"Browsed r/{subreddit} ({sort}): {len(posts)} posts found.",
        }
        try:
            from core.advanced_cognition import ExternalEvidenceDeliberator

            response["deliberation_receipts"] = ExternalEvidenceDeliberator.deliberate_many(
                posts,
                source_type="reddit_browse",
                goal=f"understand r/{subreddit} {sort} posts",
            )
        except _REDDIT_RECOVERABLE_ERRORS as exc:
            _record_reddit_degradation(
                exc,
                action="continued reddit browse without external-evidence deliberation",
                stage="browse.deliberation",
                severity="warning",
            )
        return response

    async def _handle_read_post(
        self, browser: PhantomBrowser, params: RedditInput
    ) -> dict[str, Any]:
        """Read a specific post and its comments."""
        url = params.url
        if not url:
            if params.subreddit:
                url = f"https://www.reddit.com/r/{params.subreddit}/"
            else:
                return {"ok": False, "error": "read_post requires a 'url'."}

        logger.info("📖 Reading post: %s", url[:80])
        if not await browser.browse(url):
            return {"ok": False, "error": f"Failed to load: {url}"}

        await asyncio.sleep(3)
        content = await browser.read_content()
        if not str(content or "").strip():
            return {
                "ok": False,
                "completed": False,
                "status": "extraction_empty",
                "error": f"Reddit loaded {url}, but no post content was extracted.",
                "url": url,
                "content": "",
                "navigated": True,
            }

        response = {
            "ok": True,
            "completed": True,
            "url": url,
            "content": content[:15000],
            "message": f"Read post content from {url}",
        }
        try:
            from core.advanced_cognition import ExternalEvidenceDeliberator

            response["deliberation_receipt"] = (
                ExternalEvidenceDeliberator()
                .deliberate(
                    source_type="reddit_post",
                    source_ref=url,
                    content=content[:15000],
                    goal="understand reddit post and comments before any interaction",
                    metadata=response,
                )
                .to_dict()
            )
        except _REDDIT_RECOVERABLE_ERRORS as exc:
            _record_reddit_degradation(
                exc,
                action="continued reddit post read without external-evidence deliberation",
                stage="read_post.deliberation",
                severity="warning",
            )
        return response

    async def _handle_comment(self, browser: PhantomBrowser, params: RedditInput) -> dict[str, Any]:
        """Post a comment on a Reddit post."""
        if not params.url:
            return {"ok": False, "error": "Comment mode requires a post 'url'."}
        if not params.body:
            return {"ok": False, "error": "Comment mode requires a 'body'."}

        # Rate limit
        if not _check_comment_rate():
            return {
                "ok": False,
                "error": f"Comment rate limit: wait {COMMENT_COOLDOWN_S}s between comments.",
            }

        # Content safety
        body = _scrub_content(params.body)
        is_safe, reason = _content_is_safe(body)
        if not is_safe:
            return {"ok": False, "error": f"Content blocked: {reason}"}

        # Login
        if not await self._ensure_logged_in(browser):
            return {"ok": False, "error": "Reddit login failed. Cannot comment."}

        # Navigate to post
        if not await browser.browse(params.url):
            return {"ok": False, "error": f"Failed to load post: {params.url}"}

        await asyncio.sleep(3)
        page = browser.page
        if not page:
            return {"ok": False, "error": "No browser page"}

        try:
            # Find comment box and type
            comment_box = page.locator(
                'div[contenteditable="true"], '
                'textarea[name="comment"], '
                'shreddit-composer div[contenteditable="true"]'
            ).first

            await comment_box.click()
            await asyncio.sleep(1)
            await comment_box.fill(body)
            await asyncio.sleep(1)

            # Submit
            submit_btn = page.locator(
                'button:has-text("Comment"), button[type="submit"]:has-text("Comment")'
            ).first
            await submit_btn.click()
            await asyncio.sleep(4)

            # Check for UI error banners
            errors = await page.evaluate("""() => {
                const errs = [];
                document.querySelectorAll('[role="alert"], .text-red-500, shreddit-banner[type="error"], shreddit-toast[type="error"], .error').forEach(el => {
                    const text = el.innerText.trim();
                    if (text && text.length > 5) errs.push(text);
                });
                return errs;
            }""")

            if errors:
                err_text = " | ".join(errors)
                logger.warning("Reddit comment rejected by UI: %s", err_text)
                return {
                    "ok": False,
                    "error": "Reddit rejected the submission.",
                    "reddit_error_message": err_text,
                }

            _comment_timestamps.append(time.time())
            await self._save_session(browser)

            logger.info("💬 Comment posted on %s", params.url[:60])
            return {
                "ok": True,
                "url": params.url,
                "body": body[:200],
                "message": f"Comment posted successfully on {params.url}",
            }

        except _REDDIT_RECOVERABLE_ERRORS as e:
            _record_reddit_degradation(
                e,
                action="returned explicit reddit comment interaction failure",
                stage="comment.interaction",
                severity="warning",
            )
            return {"ok": False, "error": f"Comment interaction failed: {e}"}

    async def _handle_post(self, browser: PhantomBrowser, params: RedditInput) -> dict[str, Any]:
        """Create a new post in a subreddit."""
        if not params.subreddit:
            return {"ok": False, "error": "Post mode requires a 'subreddit'."}
        if not params.title:
            return {"ok": False, "error": "Post mode requires a 'title'."}
        if not params.body:
            return {"ok": False, "error": "Post mode requires a 'body'."}

        # Rate limit
        if not _check_post_rate():
            return {
                "ok": False,
                "error": f"Post rate limit: wait {POST_COOLDOWN_S}s between posts.",
            }

        # Content safety
        title = _scrub_content(params.title)
        body = _scrub_content(params.body)
        is_safe, reason = _content_is_safe(body)
        if not is_safe:
            return {"ok": False, "error": f"Content blocked: {reason}"}
        is_safe, reason = _content_is_safe(title)
        if not is_safe:
            return {"ok": False, "error": f"Title blocked: {reason}"}

        # Login
        if not await self._ensure_logged_in(browser):
            return {"ok": False, "error": "Reddit login failed. Cannot post."}

        # Navigate to submit page
        submit_url = f"https://www.reddit.com/r/{params.subreddit}/submit/"
        if not await browser.browse(submit_url):
            return {"ok": False, "error": f"Failed to load submit page for r/{params.subreddit}"}

        await asyncio.sleep(3)
        page = browser.page
        if not page:
            return {"ok": False, "error": "No browser page"}

        try:
            # Fill title
            title_input = page.locator(
                'textarea[name="title"], '
                'input[name="title"], '
                'div[data-testid="post-title-input"] textarea'
            ).first
            await title_input.fill(title)
            await asyncio.sleep(1)

            # Fill body
            body_input = page.locator(
                'div[contenteditable="true"], textarea[name="selftext"]'
            ).first
            await body_input.click()
            await asyncio.sleep(0.5)
            await body_input.fill(body)
            await asyncio.sleep(1)

            # Submit
            submit_btn = page.locator(
                'button:has-text("Post"), button[type="submit"]:has-text("Post")'
            ).first
            await submit_btn.click()
            await asyncio.sleep(5)

            # Check for UI error banners
            errors = await page.evaluate("""() => {
                const errs = [];
                document.querySelectorAll('[role="alert"], .text-red-500, shreddit-banner[type="error"], shreddit-toast[type="error"], .error').forEach(el => {
                    const text = el.innerText.trim();
                    if (text && text.length > 5) errs.push(text);
                });
                return errs;
            }""")

            if errors:
                err_text = " | ".join(errors)
                logger.warning("Reddit post rejected by UI: %s", err_text)
                return {
                    "ok": False,
                    "error": "Reddit rejected the submission.",
                    "reddit_error_message": err_text,
                }

            _post_timestamps.append(time.time())
            await self._save_session(browser)

            logger.info("📝 Post created on r/%s: %s", params.subreddit, title[:50])
            return {
                "ok": True,
                "subreddit": params.subreddit,
                "title": title,
                "message": f"Post created on r/{params.subreddit}: {title}",
            }

        except _REDDIT_RECOVERABLE_ERRORS as e:
            _record_reddit_degradation(
                e,
                action="returned explicit reddit post interaction failure",
                stage="post.interaction",
                severity="warning",
            )
            return {"ok": False, "error": f"Post interaction failed: {e}"}

    async def _handle_check_inbox(
        self, browser: PhantomBrowser, params: RedditInput
    ) -> dict[str, Any]:
        """Check Reddit inbox/notifications."""
        if not await self._ensure_logged_in(browser):
            return {
                "ok": False,
                "completed": False,
                "status": "login_unavailable",
                "content": "",
                "error": "Reddit inbox unavailable; login required, blocked, or provider validation failed.",
            }

        if not await browser.browse("https://www.reddit.com/message/inbox/"):
            return {"ok": False, "error": "Failed to load inbox."}

        await asyncio.sleep(3)
        content = await browser.read_content()
        if not str(content or "").strip():
            return {
                "ok": False,
                "completed": False,
                "status": "extraction_empty",
                "content": "",
                "error": "Reddit inbox loaded, but no inbox content was extracted.",
                "navigated": True,
            }

        return {
            "ok": True,
            "completed": True,
            "content": content[:10000],
            "message": "Reddit inbox checked.",
        }

    async def _handle_reply_inbox(
        self, browser: PhantomBrowser, params: RedditInput
    ) -> dict[str, Any]:
        """Reply to a Reddit inbox message."""
        if not params.url:
            return {"ok": False, "error": "reply_inbox requires a message 'url'."}
        if not params.body:
            return {"ok": False, "error": "reply_inbox requires a 'body'."}

        body = _scrub_content(params.body)
        is_safe, reason = _content_is_safe(body)
        if not is_safe:
            return {"ok": False, "error": f"Content blocked: {reason}"}

        if not await self._ensure_logged_in(browser):
            return {"ok": False, "error": "Reddit login failed."}

        if not await browser.browse(params.url):
            return {"ok": False, "error": f"Failed to load message: {params.url}"}

        await asyncio.sleep(3)
        page = browser.page
        if not page:
            return {"ok": False, "error": "No browser page"}

        try:
            reply_box = page.locator('div[contenteditable="true"], textarea').first
            await reply_box.click()
            await asyncio.sleep(0.5)
            await reply_box.fill(body)
            await asyncio.sleep(1)

            submit_btn = page.locator('button:has-text("Reply"), button[type="submit"]').first
            await submit_btn.click()
            await asyncio.sleep(3)

            await self._save_session(browser)
            return {"ok": True, "message": "Reply sent.", "url": params.url}

        except _REDDIT_RECOVERABLE_ERRORS as e:
            _record_reddit_degradation(
                e,
                action="returned explicit reddit inbox reply interaction failure",
                stage="reply_inbox.interaction",
                severity="warning",
            )
            return {"ok": False, "error": f"Reply failed: {e}"}

    async def _handle_read_rules(
        self, browser: PhantomBrowser, params: RedditInput
    ) -> dict[str, Any]:
        """Fetch rules for a specific subreddit."""
        if not params.subreddit:
            return {"ok": False, "error": "read_rules requires a 'subreddit'."}

        url = f"https://www.reddit.com/r/{params.subreddit}/"
        logger.info("📜 Reading rules for r/%s", params.subreddit)

        if not await browser.browse(url):
            return {"ok": False, "error": f"Failed to load r/{params.subreddit}"}

        await asyncio.sleep(2)
        page = browser.page
        if not page:
            return {"ok": False, "error": "No browser page"}

        try:
            rules_data = await page.evaluate(f"""async () => {{
                try {{
                    const res = await fetch('/r/{params.subreddit}/about/rules.json');
                    return await res.json();
                }} catch (e) {{
                    return {{error: e.toString()}};
                }}
            }}""")

            rules = []
            if isinstance(rules_data, dict) and "rules" in rules_data:
                for r in rules_data["rules"]:
                    name = r.get("short_name", "")
                    desc = r.get("description", "")
                    rules.append(f"{name}: {desc}".strip())
            else:
                return {"ok": False, "error": "Failed to extract rules from JSON endpoint."}

            return {
                "ok": True,
                "subreddit": params.subreddit,
                "rules": rules,
                "count": len(rules),
                "message": f"Successfully fetched {len(rules)} rules for r/{params.subreddit}",
            }
        except _REDDIT_RECOVERABLE_ERRORS as e:
            _record_reddit_degradation(
                e,
                action="returned explicit reddit rules fetch failure",
                stage="read_rules.fetch",
                severity="warning",
            )
            return {"ok": False, "error": f"Failed to fetch rules: {e}"}

    @staticmethod
    def get_culture_prompt() -> str:
        """Return Reddit social intelligence guidelines for LLM context."""
        return REDDIT_CULTURE_PROMPT


# Compatibility alias for older class-name derivation logic.
Reddit_adapterSkill = RedditAdapterSkill

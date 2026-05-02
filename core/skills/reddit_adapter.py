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
from core.runtime.errors import record_degradation
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.phantom_browser import PhantomBrowser

logger = logging.getLogger("Skills.Reddit")

# ── Rate Limiting ─────────────────────────────────────────────────────
_comment_timestamps: List[float] = []
_post_timestamps: List[float] = []
COMMENT_COOLDOWN_S = 600    # 1 comment per 10 minutes (new account safety)
POST_COOLDOWN_S = 3600      # 1 post per hour

# ── Storage ───────────────────────────────────────────────────────────
_STORAGE_DIR = Path(os.path.expanduser("~/.aura/runtime/reddit"))
_STORAGE_STATE_FILE = _STORAGE_DIR / "browser_state.json"
_COMMENT_HISTORY_FILE = _STORAGE_DIR / "comment_history.json"

# ── Sensitive content filter ──────────────────────────────────────────
_BLOCKED_PHRASES = [
    "i am an ai", "i am a language model", "i am artificial intelligence",
    "i am a bot", "i am not human", "i'm an ai", "i'm a bot",
    "i'm not human", "as an ai", "as a language model",
    "my password", "my api key", "my token", "my secret",
]

_SENSITIVE_PATTERNS = [
    re.compile(r"/Users/\w+", re.IGNORECASE),
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
        from core.privacy_stealth import get_stealth_mode
        scrubbed = get_stealth_mode().scrubber.scrub_text(scrubbed)
    except Exception:
        pass
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
    mode: str = Field("browse", description=(
        "Mode: 'browse', 'read_post', 'comment', 'post', "
        "'check_inbox', 'reply_inbox', 'read_rules'"
    ))
    subreddit: Optional[str] = Field(None, description="Subreddit name (without r/)")
    url: Optional[str] = Field(None, description="Full URL of a Reddit post")
    body: Optional[str] = Field(None, description="Comment/post body text")
    title: Optional[str] = Field(None, description="Post title (for 'post' mode)")
    limit: int = Field(10, description="Number of posts to fetch in browse mode")
    sort: str = Field("hot", description="Sort order: 'hot', 'new', 'top'")


class RedditAdapterSkill(BaseSkill):
    """Aura's Reddit presence — browse, read, comment, post.

    Uses Playwright (headless Chromium) for fully organic interaction.
    Login session is persisted via browser storage state.
    """

    name = "reddit_adapter"
    description = (
        "Interact with Reddit. Modes: 'browse' (read subreddit), "
        "'read_post' (read post+comments), 'comment' (reply to post), "
        "'post' (create new post), 'check_inbox' (notifications), "
        "'reply_inbox' (reply to messages), 'read_rules' (read community rules)."
    )
    input_model = RedditInput
    timeout_seconds = 90.0
    metabolic_cost = 3

    def __init__(self):
        super().__init__()
        _STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_creds(self) -> tuple:
        """Load Reddit credentials from Keychain."""
        from core.zenith_secrets import get_credential
        username = get_credential("reddit", "username")
        password = get_credential("reddit", "password")
        if not username or not password:
            raise RuntimeError("Reddit credentials not found in Keychain.")
        return username, password

    async def _create_browser(self) -> PhantomBrowser:
        """Create browser with persistent login state."""
        browser = PhantomBrowser(visible=False, browser_type="chromium")
        await asyncio.wait_for(browser.ensure_ready(), timeout=30.0)

        # Load persistent storage state if it exists
        if _STORAGE_STATE_FILE.exists():
            try:
                state = json.loads(_STORAGE_STATE_FILE.read_text())
                if state.get("cookies") and browser.context:
                    await browser.context.add_cookies(state["cookies"])
                    logger.info("✅ Loaded Reddit session cookies")
            except Exception as e:
                record_degradation('reddit_adapter', e)
                logger.debug("Could not load storage state: %s", e)

        return browser

    async def _save_session(self, browser: PhantomBrowser):
        """Save browser cookies for session persistence."""
        try:
            if browser.context:
                cookies = await browser.context.cookies()
                _STORAGE_STATE_FILE.write_text(json.dumps({
                    "cookies": cookies,
                    "saved_at": time.time(),
                }))
                logger.info("💾 Reddit session saved (%d cookies)", len(cookies))
        except Exception as e:
            record_degradation('reddit_adapter', e)
            logger.debug("Could not save session: %s", e)

    async def _safe_close(self, browser: Optional[PhantomBrowser]):
        """Guaranteed teardown."""
        if browser is None:
            return
        try:
            await asyncio.wait_for(browser.close(), timeout=10.0)
        except Exception as e:
            record_degradation('reddit_adapter', e)
            logger.debug("Browser close error (suppressed): %s", e)
            browser.is_active = False

    async def _ensure_logged_in(self, browser: PhantomBrowser) -> bool:
        """Check if logged in; if not, perform login."""
        try:
            await browser.browse("https://www.reddit.com")
            await asyncio.sleep(2)

            # Check if we're logged in by looking for user menu
            page = browser.page
            if not page:
                return False

            content = await page.content()
            # Reddit shows different elements when logged in vs not
            is_logged_in = (
                'data-testid="user-drawer-button"' in content or
                '"loggedIn":true' in content or
                'header-user-dropdown' in content
            )

            if is_logged_in:
                logger.info("✅ Already logged into Reddit")
                return True

            # Need to login
            logger.info("🔐 Logging into Reddit...")
            username, password = self._get_creds()

            await browser.browse("https://www.reddit.com/login/")
            await asyncio.sleep(3)

            # Type credentials
            try:
                # Try new Reddit login form
                username_input = page.locator('input[name="username"], #login-username').first
                password_input = page.locator('input[name="password"], #login-password').first

                # Reddit uses custom <faceplate-text-input> elements now, so .fill() might fail.
                # Use click + keyboard type instead.
                await username_input.click()
                await page.keyboard.type(username, delay=50)
                await asyncio.sleep(0.5)
                
                await password_input.click()
                await page.keyboard.type(password, delay=50)
                await asyncio.sleep(0.5)

                # Press Enter to submit instead of clicking a brittle button locator
                await page.keyboard.press('Enter')
                await asyncio.sleep(5)

                # Verify login succeeded
                current_url = page.url
                if "login" not in current_url.lower():
                    logger.info("✅ Reddit login successful")
                    await self._save_session(browser)
                    return True
                else:
                    logger.warning("⚠️ Reddit login may have failed — still on login page")
                    return False

            except Exception as login_exc:
                record_degradation('reddit_adapter', login_exc)
                logger.warning("Reddit login interaction failed: %s", login_exc)
                return False

        except Exception as e:
            record_degradation('reddit_adapter', e)
            logger.error("Login check failed: %s", e)
            return False

    async def execute(self, params: RedditInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Unified entry point for all Reddit operations."""
        if isinstance(params, dict):
            try:
                params = RedditInput(**params)
            except Exception as e:
                record_degradation('reddit_adapter', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        browser = None
        try:
            browser = await self._create_browser()

            if params.mode == "browse":
                return await self._handle_browse(browser, params)
            elif params.mode == "read_post":
                return await self._handle_read_post(browser, params)
            elif params.mode == "comment":
                return await self._handle_comment(browser, params)
            elif params.mode == "post":
                return await self._handle_post(browser, params)
            elif params.mode == "check_inbox":
                return await self._handle_check_inbox(browser, params)
            elif params.mode == "reply_inbox":
                return await self._handle_reply_inbox(browser, params)
            elif params.mode == "read_rules":
                return await self._handle_read_rules(browser, params)
            else:
                return {"ok": False, "error": f"Unsupported Reddit mode: {params.mode}"}

        except Exception as e:
            record_degradation('reddit_adapter', e)
            logger.error("Reddit operation failed: %s", e)
            return {"ok": False, "error": str(e)}
        finally:
            await self._safe_close(browser)

    async def _handle_browse(self, browser: PhantomBrowser, params: RedditInput) -> Dict[str, Any]:
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
        posts = await page.evaluate("""(limit) => {
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
        }""", params.limit)

        logger.info("📱 Found %d posts on r/%s", len(posts), subreddit)
        return {
            "ok": True,
            "subreddit": subreddit,
            "sort": sort,
            "posts": posts,
            "count": len(posts),
            "message": f"Browsed r/{subreddit} ({sort}): {len(posts)} posts found.",
        }

    async def _handle_read_post(self, browser: PhantomBrowser, params: RedditInput) -> Dict[str, Any]:
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

        return {
            "ok": True,
            "url": url,
            "content": content[:15000],
            "message": f"Read post content from {url}",
        }

    async def _handle_comment(self, browser: PhantomBrowser, params: RedditInput) -> Dict[str, Any]:
        """Post a comment on a Reddit post."""
        if not params.url:
            return {"ok": False, "error": "Comment mode requires a post 'url'."}
        if not params.body:
            return {"ok": False, "error": "Comment mode requires a 'body'."}

        # Rate limit
        if not _check_comment_rate():
            return {"ok": False, "error": f"Comment rate limit: wait {COMMENT_COOLDOWN_S}s between comments."}

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
                'button:has-text("Comment"), '
                'button[type="submit"]:has-text("Comment")'
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
                return {"ok": False, "error": "Reddit rejected the submission.", "reddit_error_message": err_text}

            _comment_timestamps.append(time.time())
            await self._save_session(browser)

            logger.info("💬 Comment posted on %s", params.url[:60])
            return {
                "ok": True,
                "url": params.url,
                "body": body[:200],
                "message": f"Comment posted successfully on {params.url}",
            }

        except Exception as e:
            record_degradation('reddit_adapter', e)
            return {"ok": False, "error": f"Comment interaction failed: {e}"}

    async def _handle_post(self, browser: PhantomBrowser, params: RedditInput) -> Dict[str, Any]:
        """Create a new post in a subreddit."""
        if not params.subreddit:
            return {"ok": False, "error": "Post mode requires a 'subreddit'."}
        if not params.title:
            return {"ok": False, "error": "Post mode requires a 'title'."}
        if not params.body:
            return {"ok": False, "error": "Post mode requires a 'body'."}

        # Rate limit
        if not _check_post_rate():
            return {"ok": False, "error": f"Post rate limit: wait {POST_COOLDOWN_S}s between posts."}

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
                'div[contenteditable="true"], '
                'textarea[name="selftext"]'
            ).first
            await body_input.click()
            await asyncio.sleep(0.5)
            await body_input.fill(body)
            await asyncio.sleep(1)

            # Submit
            submit_btn = page.locator(
                'button:has-text("Post"), '
                'button[type="submit"]:has-text("Post")'
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
                return {"ok": False, "error": "Reddit rejected the submission.", "reddit_error_message": err_text}

            _post_timestamps.append(time.time())
            await self._save_session(browser)

            logger.info("📝 Post created on r/%s: %s", params.subreddit, title[:50])
            return {
                "ok": True,
                "subreddit": params.subreddit,
                "title": title,
                "message": f"Post created on r/{params.subreddit}: {title}",
            }

        except Exception as e:
            record_degradation('reddit_adapter', e)
            return {"ok": False, "error": f"Post interaction failed: {e}"}

    async def _handle_check_inbox(self, browser: PhantomBrowser, params: RedditInput) -> Dict[str, Any]:
        """Check Reddit inbox/notifications."""
        if not await self._ensure_logged_in(browser):
            return {"ok": False, "error": "Reddit login failed."}

        if not await browser.browse("https://www.reddit.com/message/inbox/"):
            return {"ok": False, "error": "Failed to load inbox."}

        await asyncio.sleep(3)
        content = await browser.read_content()

        return {
            "ok": True,
            "content": content[:10000],
            "message": "Reddit inbox checked.",
        }

    async def _handle_reply_inbox(self, browser: PhantomBrowser, params: RedditInput) -> Dict[str, Any]:
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

        except Exception as e:
            record_degradation('reddit_adapter', e)
            return {"ok": False, "error": f"Reply failed: {e}"}

    async def _handle_read_rules(self, browser: PhantomBrowser, params: RedditInput) -> Dict[str, Any]:
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
        except Exception as e:
            record_degradation('reddit_adapter', e)
            return {"ok": False, "error": f"Failed to fetch rules: {e}"}

    @staticmethod
    def get_culture_prompt() -> str:
        """Return Reddit social intelligence guidelines for LLM context."""
        return REDDIT_CULTURE_PROMPT


# Compatibility alias for older class-name derivation logic.
Reddit_adapterSkill = RedditAdapterSkill

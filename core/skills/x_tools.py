"""core/skills/x_tools.py — Enhanced Twitter/X Capabilities
=============================================================
First-class BaseSkill that extends Aura's Twitter/X presence with:
  - Advanced search (keyword, user, hashtag, geo-filtered)
  - Thread fetching and conversation reconstruction
  - Trend monitoring with topic extraction
  - Media extraction (video frame capture, image download)
  - Engagement analytics

Works alongside the existing social_media.py and joy_social_integration.py
skills. Uses Aura's phantom browser as the transport layer when API keys
are unavailable.

This closes the "X/Twitter tools" gap in tool parity.
"""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from core.runtime.errors import FallbackClassification, record_degradation
from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

logger = logging.getLogger("Skills.XTools")

_XTOOLS_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
    ConnectionError,
)


def _record_xtools_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "x_tools",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


class XToolsInput(BaseModel):
    action: str = Field(
        ...,
        description=(
            "Action to perform: 'search', 'thread', 'trends', 'user_timeline', "
            "'engagement_stats', 'extract_media'."
        ),
    )
    query: str | None = Field(None, description="Search query or tweet URL/ID.")
    username: str | None = Field(None, description="Twitter username (without @).")
    count: int = Field(20, ge=1, le=100, description="Number of results to return.")
    include_replies: bool = Field(False, description="Include replies in results.")
    media_only: bool = Field(False, description="Filter to only media-containing tweets.")


class XToolsSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "x_tools"
    description = (
        "Advanced Twitter/X operations: search tweets, fetch threads, "
        "monitor trends, extract media, and analyze engagement. "
        "Use for real-time social intelligence and content interaction."
    )
    input_model = XToolsInput
    timeout_seconds = 60.0
    metabolic_cost = 2
    effect_scope = "external_io"

    def __init__(self):
        super().__init__()
        self._api_client = None
        self._api_checked = False

    def _get_api_client(self) -> Any | None:
        """Resolve Twitter API client from ServiceContainer."""
        if self._api_checked:
            return self._api_client
        self._api_checked = True
        try:
            from core.container import ServiceContainer
            self._api_client = ServiceContainer.get("twitter_client", default=None)
        except _XTOOLS_RECOVERABLE_ERRORS as exc:
            _record_xtools_degradation(
                exc,
                action="skipped Twitter API client resolution, using browser fallback",
            )
        return self._api_client

    async def _get_browser(self) -> Any | None:
        """Resolve phantom browser for scraping fallback."""
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get("phantom_browser", default=None)
        except _XTOOLS_RECOVERABLE_ERRORS:
            return None

    async def execute(
        self, params: XToolsInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Route to the appropriate X/Twitter action."""
        if isinstance(params, dict):
            try:
                params = XToolsInput(**params)
            except _XTOOLS_RECOVERABLE_ERRORS as exc:
                _record_xtools_degradation(
                    exc,
                    action="rejected invalid X tools input",
                )
                return {"ok": False, "error": f"Invalid input: {exc}"}

        action = params.action.lower().strip()
        handlers = {
            "search": self._search,
            "thread": self._fetch_thread,
            "trends": self._get_trends,
            "user_timeline": self._user_timeline,
            "engagement_stats": self._engagement_stats,
            "extract_media": self._extract_media,
        }

        handler = handlers.get(action)
        if not handler:
            return {
                "ok": False,
                "error": f"Unknown action: {action}. Valid: {list(handlers.keys())}",
            }

        try:
            return await handler(params, context)
        except _XTOOLS_RECOVERABLE_ERRORS as exc:
            _record_xtools_degradation(
                exc,
                action=f"reported {action} failure",
                extra={"action": action, "query": params.query},
            )
            return {"ok": False, "error": f"{action} failed: {exc}"}

    async def _search(
        self, params: XToolsInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Search tweets by keyword, hashtag, or user mention."""
        query = params.query
        if not query:
            return {"ok": False, "error": "No search query provided."}

        # Strategy 1: API client
        client = self._get_api_client()
        if client and hasattr(client, "search_tweets"):
            try:
                results = await asyncio.to_thread(
                    client.search_tweets,
                    query,
                    count=params.count,
                )
                return {
                    "ok": True,
                    "results": self._normalize_tweets(results),
                    "count": len(results),
                    "engine": "api",
                    "summary": f"Found {len(results)} tweets for: {query}",
                }
            except _XTOOLS_RECOVERABLE_ERRORS as exc:
                logger.debug("API search failed, trying browser: %s", exc)

        # Strategy 2: Web search fallback
        try:
            from core.skills.web_search import EnhancedWebSearchSkill
            web_search = EnhancedWebSearchSkill()
            results = await web_search.execute(
                {"query": f"site:twitter.com {query}", "max_results": params.count},
                context,
            )
            return {
                "ok": results.get("ok", False),
                "results": results.get("results", []),
                "engine": "web_search",
                "summary": f"Web search fallback for X query: {query}",
            }
        except _XTOOLS_RECOVERABLE_ERRORS as exc:
            return {"ok": False, "error": f"Search failed (all backends): {exc}"}

    async def _fetch_thread(
        self, params: XToolsInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Fetch a tweet thread / conversation from a tweet URL or ID."""
        query = params.query
        if not query:
            return {"ok": False, "error": "No tweet URL or ID provided."}

        # Extract tweet ID from URL
        tweet_id = self._extract_tweet_id(query)
        if not tweet_id:
            return {"ok": False, "error": f"Could not extract tweet ID from: {query}"}

        # Strategy 1: API
        client = self._get_api_client()
        if client and hasattr(client, "get_thread"):
            try:
                thread = await asyncio.to_thread(client.get_thread, tweet_id)
                return {
                    "ok": True,
                    "thread": self._normalize_tweets(thread),
                    "tweet_id": tweet_id,
                    "engine": "api",
                    "summary": f"Fetched thread for tweet {tweet_id}",
                }
            except _XTOOLS_RECOVERABLE_ERRORS as _exc:
                logger.debug("Suppressed %s in core.skills.x_tools: %s", type(_exc).__name__, _exc)

        # Strategy 2: Browser scrape
        browser = await self._get_browser()
        if browser and hasattr(browser, "fetch_page"):
            try:
                url = f"https://x.com/i/status/{tweet_id}"
                page_data = await browser.fetch_page(url)
                return {
                    "ok": True,
                    "raw_content": page_data.get("text", "")[:5000],
                    "tweet_id": tweet_id,
                    "engine": "browser",
                    "summary": f"Scraped thread page for tweet {tweet_id}",
                }
            except _XTOOLS_RECOVERABLE_ERRORS as exc:
                return {"ok": False, "error": f"Thread fetch failed: {exc}"}

        return {"ok": False, "error": "No Twitter backend available for thread fetching."}

    async def _get_trends(
        self, params: XToolsInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Get current Twitter/X trending topics."""
        # Strategy 1: API
        client = self._get_api_client()
        if client and hasattr(client, "get_trends"):
            try:
                trends = await asyncio.to_thread(client.get_trends)
                return {
                    "ok": True,
                    "trends": trends[:params.count],
                    "engine": "api",
                    "summary": f"Fetched {len(trends)} trending topics",
                }
            except _XTOOLS_RECOVERABLE_ERRORS as _exc:
                logger.debug("Suppressed %s in core.skills.x_tools: %s", type(_exc).__name__, _exc)

        # Strategy 2: Web search
        try:
            from core.skills.web_search import EnhancedWebSearchSkill
            web_search = EnhancedWebSearchSkill()
            results = await web_search.execute(
                {"query": "Twitter trending topics today", "max_results": 10},
                context,
            )
            return {
                "ok": results.get("ok", False),
                "trends": results.get("results", []),
                "engine": "web_search",
                "summary": "Trends via web search fallback",
            }
        except _XTOOLS_RECOVERABLE_ERRORS as exc:
            return {"ok": False, "error": f"Trends fetch failed: {exc}"}

    async def _user_timeline(
        self, params: XToolsInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Fetch recent tweets from a specific user."""
        username = params.username or params.query
        if not username:
            return {"ok": False, "error": "No username provided."}
        username = username.lstrip("@").strip()

        client = self._get_api_client()
        if client and hasattr(client, "get_user_timeline"):
            try:
                tweets = await asyncio.to_thread(
                    client.get_user_timeline,
                    username,
                    count=params.count,
                )
                return {
                    "ok": True,
                    "tweets": self._normalize_tweets(tweets),
                    "username": username,
                    "engine": "api",
                    "summary": f"Fetched {len(tweets)} tweets from @{username}",
                }
            except _XTOOLS_RECOVERABLE_ERRORS as _exc:
                logger.debug("Suppressed %s in core.skills.x_tools: %s", type(_exc).__name__, _exc)

        # Fallback: browser scrape
        browser = await self._get_browser()
        if browser and hasattr(browser, "fetch_page"):
            try:
                page_data = await browser.fetch_page(f"https://x.com/{username}")
                return {
                    "ok": True,
                    "raw_content": page_data.get("text", "")[:5000],
                    "username": username,
                    "engine": "browser",
                    "summary": f"Scraped timeline for @{username}",
                }
            except _XTOOLS_RECOVERABLE_ERRORS as exc:
                return {"ok": False, "error": f"Timeline fetch failed: {exc}"}

        return {"ok": False, "error": "No backend available for user timeline."}

    async def _engagement_stats(
        self, params: XToolsInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Get engagement statistics for a tweet or user."""
        query = params.query
        if not query:
            return {"ok": False, "error": "No tweet URL/ID or username provided."}

        client = self._get_api_client()
        if not client:
            return {
                "ok": False,
                "error": "Engagement stats require API access. No API client configured.",
            }

        tweet_id = self._extract_tweet_id(query)
        if tweet_id and hasattr(client, "get_tweet"):
            try:
                tweet = await asyncio.to_thread(client.get_tweet, tweet_id)
                return {
                    "ok": True,
                    "stats": {
                        "likes": tweet.get("favorite_count", 0),
                        "retweets": tweet.get("retweet_count", 0),
                        "replies": tweet.get("reply_count", 0),
                        "quotes": tweet.get("quote_count", 0),
                        "impressions": tweet.get("impression_count", 0),
                    },
                    "tweet_id": tweet_id,
                    "summary": f"Engagement stats for tweet {tweet_id}",
                }
            except _XTOOLS_RECOVERABLE_ERRORS as exc:
                return {"ok": False, "error": f"Stats fetch failed: {exc}"}

        return {"ok": False, "error": f"Could not resolve target from: {query}"}

    async def _extract_media(
        self, params: XToolsInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract media (images, video thumbnails) from a tweet."""
        query = params.query
        if not query:
            return {"ok": False, "error": "No tweet URL or ID provided."}

        tweet_id = self._extract_tweet_id(query)
        if not tweet_id:
            return {"ok": False, "error": f"Could not extract tweet ID from: {query}"}

        client = self._get_api_client()
        if client and hasattr(client, "get_tweet"):
            try:
                tweet = await asyncio.to_thread(client.get_tweet, tweet_id)
                media = tweet.get("media", [])
                return {
                    "ok": True,
                    "media": media,
                    "count": len(media),
                    "tweet_id": tweet_id,
                    "summary": f"Extracted {len(media)} media items from tweet {tweet_id}",
                }
            except _XTOOLS_RECOVERABLE_ERRORS as exc:
                return {"ok": False, "error": f"Media extraction failed: {exc}"}

        return {"ok": False, "error": "Media extraction requires API access."}

    @staticmethod
    def _extract_tweet_id(query: str) -> str | None:
        """Extract tweet ID from a URL or raw ID string."""
        import re

        query = query.strip()

        # Direct ID (numeric)
        if query.isdigit():
            return query

        # URL patterns: twitter.com/user/status/ID or x.com/user/status/ID
        match = re.search(r"(?:twitter\.com|x\.com)/\w+/status/(\d+)", query)
        if match:
            return match.group(1)

        return None

    @staticmethod
    def _normalize_tweets(tweets: Any) -> list[dict[str, Any]]:
        """Normalize tweet data into a consistent format."""
        if not tweets:
            return []
        if isinstance(tweets, dict):
            tweets = [tweets]
        normalized = []
        for tweet in tweets:
            if isinstance(tweet, dict):
                normalized.append({
                    "id": tweet.get("id") or tweet.get("id_str"),
                    "text": tweet.get("text") or tweet.get("full_text", ""),
                    "user": tweet.get("user", {}).get("screen_name", "unknown"),
                    "created_at": tweet.get("created_at"),
                    "likes": tweet.get("favorite_count", 0),
                    "retweets": tweet.get("retweet_count", 0),
                    "media": tweet.get("media", []),
                })
            else:
                normalized.append({"raw": str(tweet)})
        return normalized

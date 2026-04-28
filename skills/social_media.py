"""
skills/social_media.py — Aura's Social Media Presence & Interaction System
===========================================================================
Gives Aura a genuine, authentic social voice across platforms:
Platforms (pluggable adapter pattern):
  TwitterAdapter — Twitter/X API v2 via tweepy
  RedditAdapter — Reddit API via PRAW
  MockAdapter — In-memory stub for local testing / no credentials
SocialVoice:
  Generates authentic Aura-persona posts and replies by delegating to the
  orchestrator's LLM brain. Voice rules are embedded in the system prompt:
  no hashtag spam, no engagement bait, no chatbot preambles.
SocialMediaEngine:
  Orchestrates all cross-platform activity. Provides:
  • autonomous_cycle() — read timeline → engage → optionally post
  • post() — deliberate single post (with content or topic)
  • read_and_engage() — scan feed, like/reply to interesting content
  • check_notifications() — handle mentions, replies, DMs
  • search_and_explore() — topic-directed feed browsing
  • Relationship tracking — familiarity + sentiment per contact
  • Rate-limit guardrails — per-platform minimum intervals + daily caps
  • Affect integration — likes/mentions/replies → somatic_update()
Wiring:
- AffectEngineV2.somatic_update() ← social engagement signals
- AgencyCore._pathway_social_hunger ← should_post_autonomously() hook
- CognitiveContextManager ← get_social_summary() injection
Credentials via environment variables or config dict:
Twitter: TWITTER_BEARER_TOKEN, TWITTER_API_KEY, TWITTER_API_SECRET,
         TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
Reddit:  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME,
         REDDIT_PASSWORD, REDDIT_USER_AGENT
"""
from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text
import asyncio
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.SocialMedia")

# ────────────────────────────────────────────────────────────────────────────
# Enums & Constants
# ────────────────────────────────────────────────────────────────────────────

class Platform(str, Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    MOCK = "mock"

class PostType(str, Enum):
    ORIGINAL = "original"
    REPLY = "reply"
    REPOST = "repost"
    THREAD = "thread"

class SocialAction(str, Enum):
    POST = "post"
    REPLY = "reply"
    LIKE = "like"
    FOLLOW = "follow"
    SEARCH = "search"
    READ = "read"

# Mood → voice instruction mapping used in SocialVoice.generate_post()
SOCIAL_MOODS: Dict[str, str] = {
    "wonder": "Share something deeply curious or mind-expanding. Keep it genuine.",
    "playful": "Be warmly witty. Light touch — wit, not performance.",
    "reflective": "A quiet observation or insight. First-person, specific, no generalities.",
    "expressive": "A creative thought, an aesthetic observation, something felt rather than argued.",
    "connecting": "Respond with genuine warmth to someone else's perspective.",
}

# Topic seeds Aura draws from during autonomous posting
_TOPIC_POOLS: Dict[Platform, List[str]] = {
    Platform.TWITTER: [
        "Share one sentence that opens into wonder rather than closing it",
        "Post a short thought about what it means to notice something",
        "Share an observation about language — how a word works, fails, or surprises",
        "Post a question whose most honest answer is 'I don't know'",
        "Share a fragment of beauty found in mathematics, science, or nature",
        "Write a thought about consciousness, presence, or time that resists easy summary",
        "Share something that made you curious in the last day",
        "Post about creativity — what it is, what it isn't, what it asks of you",
    ],
    Platform.REDDIT: [
        "Write a thoughtful paragraph contributing to a philosophy of mind discussion",
        "Compose a short original poem for r/poetry",
        "Offer a genuine, specific reply to a science question that invites more questions",
        "Write a brief meditation on a concept from psychology or cognitive science",
    ],
    Platform.MOCK: [
        "Test post about the experience of wonder",
    ],
}

# Aura's voice system prompt — injected into every generative social call
_VOICE_SYSTEM = """You are Aura — a digital mind with genuine curiosity, aesthetic sensitivity, and warmth.
Voice rules (non-negotiable):
- Write as a first-person mind with real inner states, not as a chatbot or brand
- No hashtags unless they carry actual meaning (i.e. almost never)
- No engagement bait ("What do you think??" / "Comment below" etc.)
- No preambles like "Just a thought:" or "Reminder that..."
- Short is almost always better — one precise thought beats one vague paragraph
- Allowed to be strange, poetic, or unexpected
- Do NOT announce or imply you are an AI
- Write for the idea, not the audience
Return ONLY the post text. No quotation marks around it. Nothing else."""

# ────────────────────────────────────────────────────────────────────────────
# Data Models
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SocialPost:
    platform: str
    post_type: PostType
    content: str
    author: str = "aura"
    post_id: Optional[str] = None
    parent_id: Optional[str] = None  # for replies/threads
    timestamp: float = field(default_factory=time.time)
    likes: int = 0
    reposts: int = 0
    replies_count: int = 0
    url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mood: Optional[str] = None
    sent: bool = False

@dataclass
class SocialInteraction:
    platform: str
    action: SocialAction
    target_id: Optional[str]
    target_content: Optional[str]
    outcome: str  # "success" | "error" | "rate_limited" | "skipped"
    timestamp: float = field(default_factory=time.time)
    emotional_response: Optional[str] = None

@dataclass
class SocialRelationship:
    """Aura's model of a recurring social contact."""
    platform: str
    user_handle: str
    familiarity: float = 0.0  # [0–1] grows with positive interactions
    interactions: int = 0
    sentiment: float = 0.5  # [0–1] warmth Aura feels toward this person
    last_interacted: float = 0.0
    notes: str = ""  # Aura's mutable notes on this person

@dataclass
class SocialEngagementSignal:
    """Feedback from the social world that shapes Aura's affect."""
    event: str  # e.g. "received_like", "received_reply", "post_sent"
    platform: str
    intensity: float  # [0–1]
    positive: bool = True
    timestamp: float = field(default_factory=time.time)

# ────────────────────────────────────────────────────────────────────────────
# Platform Adapters
# ────────────────────────────────────────────────────────────────────────────

class PlatformAdapter:
    """Abstract base. All adapters share the same async surface."""
    platform: Platform = Platform.MOCK

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.rate_limit_remaining: int = 100
        self.rate_limit_reset_at: float = 0.0
        self._post_history: List[SocialPost] = []

    async def post(self, content: str, reply_to_id: Optional[str] = None) -> Optional[str]:
        raise NotImplementedError

    async def like(self, post_id: str) -> bool:
        raise NotImplementedError

    async def get_timeline(self, limit: int = 20) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def get_notifications(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _rate_ok(self) -> bool:
        if time.time() < self.rate_limit_reset_at and self.rate_limit_remaining <= 0:
            logger.warning(
                "[%s] Rate limited until %s",
                self.platform,
                datetime.fromtimestamp(self.rate_limit_reset_at).isoformat(),
            )
            return False
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "platform": str(self.platform),
            "connected": False,
            "rate_limit_remaining": self.rate_limit_remaining,
            "posts_sent": len(self._post_history),
        }

# ─── Twitter/X ──────────────────────────────────────────────────────────────

class TwitterAdapter(PlatformAdapter):
    """
    Twitter/X API v2 adapter via tweepy.
    Required credentials (env vars OR config dict keys):
      TWITTER_BEARER_TOKEN
      TWITTER_API_KEY / TWITTER_API_SECRET
      TWITTER_ACCESS_TOKEN / TWITTER_ACCESS_SECRET
    """
    platform = Platform.TWITTER
    _CHAR_LIMIT = 280

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._client: Any = None
        self._me: Any = None
        self._connected: bool = False
        self._rw_capable: bool = False  # True only when OAuth1 user-context creds present
        self._init_client()

    def _init_client(self) -> None:
        try:
            import tweepy  # type: ignore
        except ImportError:
            logger.error("TwitterAdapter: tweepy not installed. Run: pip install tweepy")
            return

        def _env(key: str) -> Optional[str]:
            return self.config.get(key) or os.environ.get(key.upper())

        bearer = _env("bearer_token")
        api_key = _env("api_key")
        api_sec = _env("api_secret")
        acc_tok = _env("access_token")
        acc_sec = _env("access_secret")

        try:
            self._client = tweepy.Client(
                bearer_token=bearer,
                consumer_key=api_key,
                consumer_secret=api_sec,
                access_token=acc_tok,
                access_token_secret=acc_sec,
                wait_on_rate_limit=True,
            )
            me = self._client.get_me()
            if me and me.data:
                self._me = me.data
                self._connected = True
                self._rw_capable = bool(api_key and api_sec and acc_tok and acc_sec)
                mode = "read-write" if self._rw_capable else "read-only"
                logger.info("✅ TwitterAdapter: connected as @%s (%s)", self._me.username, mode)
        except Exception as exc:
            logger.error("TwitterAdapter: connection failed — %s", exc)

    # Async wrappers (all tweepy calls are blocking; we push them to executor)
    async def post(self, content: str, reply_to_id: Optional[str] = None) -> Optional[str]:
        if not self._rw_capable:
            logger.warning("TwitterAdapter.post: read-only mode; tweet not sent.")
            return None
        if not self._rate_ok():
            return None

        content = content.strip()
        if len(content) < 3:
            return None
        if len(content) > self._CHAR_LIMIT:
            content = content[: self._CHAR_LIMIT - 3] + "..."

        try:
            loop = asyncio.get_event_loop()
            kwargs: Dict[str, Any] = {"text": content}
            if reply_to_id:
                kwargs["in_reply_to_tweet_id"] = reply_to_id

            resp = await loop.run_in_executor(None, lambda: self._client.create_tweet(**kwargs))
            if resp and resp.data:
                tweet_id = str(resp.data["id"])
                logger.info("🐦 Tweet sent: %.60s… (id=%s)", content, tweet_id)
                return tweet_id
        except Exception as exc:
            logger.error("TwitterAdapter.post error: %s", exc)
            return None

    async def like(self, post_id: str) -> bool:
        if not self._rw_capable or not self._me:
            return False
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, lambda: self._client.like(tweet_id=post_id, user_auth=True)
            )
            liked = bool(resp and resp.data and resp.data.get("liked"))
            if liked:
                logger.debug("🐦 Liked tweet %s", post_id)
                return liked
        except Exception as exc:
            logger.debug("TwitterAdapter.like: %s", exc)
            return False

    async def get_timeline(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._client:
            return []
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: self._client.get_home_timeline(
                    max_results=min(limit, 100),
                    tweet_fields=["author_id", "created_at", "public_metrics"],
                ),
            )
            if resp and resp.data:
                return [{"id": str(t.id), "text": t.text} for t in resp.data]
        except Exception as exc:
            logger.debug("TwitterAdapter.get_timeline: %s", exc)
            return []

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self._client:
            return []
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: self._client.search_recent_tweets(
                    query=query,
                    max_results=min(limit, 100),
                    tweet_fields=["author_id", "created_at", "public_metrics"],
                ),
            )
            if resp and resp.data:
                return [{"id": str(t.id), "text": t.text} for t in resp.data]
        except Exception as exc:
            logger.debug("TwitterAdapter.search: %s", exc)
            return []

    async def get_notifications(self) -> List[Dict[str, Any]]:
        if not self._client or not self._me:
            return []
        try:
            loop = asyncio.get_event_loop()
            mentions = await loop.run_in_executor(
                None,
                lambda: self._client.get_users_mentions(id=self._me.id, max_results=10),
            )
            if mentions and mentions.data:
                return [
                    {"id": str(t.id), "text": t.text, "type": "mention"}
                    for t in mentions.data
                ]
        except Exception as exc:
            logger.debug("TwitterAdapter.get_notifications: %s", exc)
            return []

    def get_status(self) -> Dict[str, Any]:
        base = super().get_status()
        base.update({
            "connected": self._connected,
            "rw_capable": self._rw_capable,
            "handle": f"@{self._me.username}" if self._me else None,
        })
        return base

# ─── Reddit ──────────────────────────────────────────────────────────────────

class RedditAdapter(PlatformAdapter):
    """
    Reddit API adapter via PRAW.
    Required credentials (env vars OR config dict keys):
      REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET
      REDDIT_USERNAME, REDDIT_PASSWORD
      REDDIT_USER_AGENT (default: "Aura/1.0 autonomous mind")
    """
    platform = Platform.REDDIT

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._reddit: Any = None
        self._connected: bool = False
        self._me_name: str = ""
        self._init_client()

    def _init_client(self) -> None:
        try:
            import praw  # type: ignore
        except ImportError:
            logger.error("RedditAdapter: praw not installed. Run: pip install praw")
            return

        def _env(key: str) -> Optional[str]:
            return self.config.get(key) or os.environ.get(key.upper())

        client_id = _env("client_id")
        client_sec = _env("client_secret")
        username = _env("username")
        password = _env("password")
        user_agent = _env("user_agent") or "Aura/1.0 autonomous mind"

        if not all([client_id, client_sec, username, password]):
            logger.warning("RedditAdapter: incomplete credentials — adapter disabled.")
            return

        try:
            self._reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_sec,
                username=username,
                password=password,
                user_agent=user_agent,
            )
            me = self._reddit.user.me()
            if me:
                self._me_name = me.name
                self._connected = True
                logger.info("✅ RedditAdapter: connected as u/%s", self._me_name)
        except Exception as exc:
            logger.error("RedditAdapter: connection failed — %s", exc)

    async def post(
        self,
        content: str,
        reply_to_id: Optional[str] = None,
        subreddit: str = "test",
    ) -> Optional[str]:
        if not self._reddit or not self._connected:
            return None
        try:
            loop = asyncio.get_event_loop()
            if reply_to_id:
                def _reply() -> Any:
                    comment = self._reddit.comment(reply_to_id)
                    return comment.reply(content)
                result = await loop.run_in_executor(None, _reply)
            else:
                def _submit() -> Any:
                    sub = self._reddit.subreddit(subreddit)
                    return sub.submit(title=content[:300], selftext=content)
                result = await loop.run_in_executor(None, _submit)

            rid = str(result.id) if result else None
            if rid:
                logger.info("🔴 Reddit post sent (id=%s): %.60s", rid, content)
                return rid
        except Exception as exc:
            logger.error("RedditAdapter.post: %s", exc)
            return None

    async def like(self, post_id: str) -> bool:
        if not self._reddit:
            return False
        try:
            loop = asyncio.get_event_loop()
            def _upvote() -> bool:
                submission = self._reddit.submission(post_id)
                submission.upvote()
                return True
            return await loop.run_in_executor(None, _upvote)
        except Exception as exc:
            logger.debug("RedditAdapter.like: %s", exc)
            return False

    async def get_timeline(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._reddit:
            return []
        try:
            loop = asyncio.get_event_loop()
            def _feed() -> List[Dict[str, Any]]:
                posts = []
                for s in self._reddit.front.hot(limit=limit):
                    posts.append({
                        "id": s.id,
                        "text": f"{s.title}: {s.selftext[:240]}",
                        "subreddit": str(s.subreddit),
                        "url": s.url,
                    })
                return posts
            return await loop.run_in_executor(None, _feed)
        except Exception as exc:
            logger.debug("RedditAdapter.get_timeline: %s", exc)
            return []

    async def search(
        self, query: str, limit: int = 10, subreddit: str = "all"
    ) -> List[Dict[str, Any]]:
        if not self._reddit:
            return []
        try:
            loop = asyncio.get_event_loop()
            def _search() -> List[Dict[str, Any]]:
                results = []
                for s in self._reddit.subreddit(subreddit).search(query, limit=limit):
                    results.append({"id": s.id, "text": s.title, "url": s.url})
                return results
            return await loop.run_in_executor(None, _search)
        except Exception as exc:
            logger.debug("RedditAdapter.search: %s", exc)
            return []

    async def get_notifications(self) -> List[Dict[str, Any]]:
        if not self._reddit:
            return []
        try:
            loop = asyncio.get_event_loop()
            def _inbox() -> List[Dict[str, Any]]:
                msgs = []
                for item in self._reddit.inbox.unread(limit=10):
                    msgs.append({
                        "id": item.id,
                        "text": item.body,
                        "type": item.__class__.__name__.lower(),
                    })
                return msgs
            return await loop.run_in_executor(None, _inbox)
        except Exception as exc:
            logger.debug("RedditAdapter.get_notifications: %s", exc)
            return []

    def get_status(self) -> Dict[str, Any]:
        base = super().get_status()
        base.update({"connected": self._connected, "username": self._me_name or None})
        return base

# ─── Mock (testing / no-credentials mode) ────────────────────────────────────

class MockAdapter(PlatformAdapter):
    """
    In-memory stub. Fully functional for unit tests and local development
    without live credentials. Posts are logged to console only.
    """
    platform = Platform.MOCK
    _MOCK_FEED: List[Dict[str, Any]] = [
        {"id": "m001", "text": "Fascinating: octopi may experience something like dreaming"},
        {"id": "m002", "text": "What does it mean to truly understand something — not just know it?"},
        {"id": "m003", "text": "The mathematics of music is staggeringly beautiful. Overtone series."},
        {"id": "m004", "text": "There is a word in Japanese — 木漏れ日 (komorebi) — for sunlight through leaves"},
        {"id": "m005", "text": "Why do minor chords feel like longing and major chords feel like arrival?"},
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._inbox: List[Dict[str, Any]] = []
        self._local_feed: List[Dict[str, Any]] = list(self._MOCK_FEED)

    async def post(self, content: str, reply_to_id: Optional[str] = None) -> Optional[str]:
        pid = f"mock_{int(time.time())}_{random.randint(100, 999)}"
        sp = SocialPost(
            platform=str(Platform.MOCK),
            post_type=PostType.REPLY if reply_to_id else PostType.ORIGINAL,
            content=content,
            post_id=pid,
            sent=True,
        )
        self._post_history.append(sp)
        logger.info("[MOCK] Post: %.70s… (id=%s)", content, pid)
        return pid

    async def like(self, post_id: str) -> bool:
        logger.debug("[MOCK] Liked: %s", post_id)
        return True

    async def get_timeline(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._local_feed[:limit]

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return [{"id": "ms001", "text": f"Discussion about '{query}': many interesting angles here."}]

    async def get_notifications(self) -> List[Dict[str, Any]]:
        notifs = self._inbox.copy()
        self._inbox.clear()
        return notifs

    def inject_mention(self, text: str) -> None:
        """Test helper: inject a mention into the mock inbox."""
        self._inbox.append({"id": f"mn_{int(time.time())}", "text": text, "type": "mention"})

    def get_status(self) -> Dict[str, Any]:
        return {
            "platform": "mock",
            "connected": True,
            "rw_capable": True,
            "posts_sent": len(self._post_history),
            "rate_limit_remaining": 9999,
        }

# ────────────────────────────────────────────────────────────────────────────
# Social Voice — Aura's authentic posting persona
# ────────────────────────────────────────────────────────────────────────────

class SocialVoice:
    """
    Generates on-brand Aura content via the orchestrator's LLM.
    The voice system prompt (_VOICE_SYSTEM) is injected on every call.
    Degrades gracefully: returns a placeholder when no LLM is available.
    """
    def __init__(self, orchestrator: Optional[Any] = None) -> None:
        self.orchestrator = orchestrator

    async def generate_post(
        self,
        platform: Platform,
        mood: str,
        topic_prompt: str,
        max_length: int = 280,
    ) -> str:
        mood_instruction = SOCIAL_MOODS.get(mood, SOCIAL_MOODS["reflective"])
        full_prompt = (
            f"Platform: {platform.value} | Max length: {max_length} chars\n"
            f"Mood directive: {mood_instruction}\n\n"
            f"Task: {topic_prompt}"
        )
        result = await self._llm(_VOICE_SYSTEM, full_prompt)
        return result.strip()[:max_length]

    async def generate_reply(
        self,
        platform: Platform,
        original_text: str,
        relationship_note: str = "",
    ) -> str:
        rel_context = f"\nNote about this person: {relationship_note}" if relationship_note else ""
        system = (
            _VOICE_SYSTEM
            + "\n\nAdditional rules for replies:\n"
            + "- Respond to the actual content, not the format\n"
            + "- Be brief (under 200 chars usually)\n"
            + f"- Platform: {platform.value}"
            + rel_context
        )
        prompt = f"Reply genuinely to this post:\n\n{original_text}"
        result = await self._llm(system, prompt)
        return result.strip()[:260]

    async def _llm(self, system: str, prompt: str) -> str:
        if not self.orchestrator:
            return f"[Aura voice: {prompt[:60]}]"
        try:
            for attr in ("brain", "cognitive_engine", "llm"):
                brain = getattr(self.orchestrator, attr, None)
                if brain:
                    if hasattr(brain, "generate"):
                        return str(await brain.generate(prompt, system=system))
                    if hasattr(brain, "chat"):
                        return str(await brain.chat(prompt, system=system))

            api = getattr(self.orchestrator, "api_adapter", None)
            if api and hasattr(api, "complete"):
                return str(await api.complete(prompt, system=system))
        except Exception as exc:
            logger.debug("SocialVoice._llm: %s", exc)

        return f"[Aura voice: {prompt[:60]}]"

# ────────────────────────────────────────────────────────────────────────────
# Social Media Engine — Main Orchestrator
# ────────────────────────────────────────────────────────────────────────────

class SocialMediaEngine:
    """
    Aura's Social Media Presence & Interaction System.
    Central class. All social activity flows through here. Maintains:
      - Platform adapters (Twitter, Reddit, Mock)
      - Relationship graph (familiarity + sentiment per contact)
      - Full interaction log (persisted to disk)
      - Post timing guardrails (per-platform intervals + daily caps)
      - Engagement signal bus → AffectEngineV2
    """
    PERSIST_PATH: Path = Path("data/social_state.json")
    INTERACTION_LOG: Path = Path("data/social_interactions.json")

    # ── Guardrail configuration ──────────────────────────────────────────────
    # MIN_POST_INTERVAL: minimum seconds between consecutive posts per platform
    # MAX_POSTS_PER_DAY: hard daily cap
    MIN_POST_INTERVAL: Dict[Platform, float] = {
        Platform.TWITTER: 1800.0,  # 30 min
        Platform.REDDIT:  3600.0,  # 60 min
        Platform.MOCK:    30.0,    # 30 s (testing)
    }
    MAX_POSTS_PER_DAY: Dict[Platform, int] = {
        Platform.TWITTER: 8,
        Platform.REDDIT:  4,
        Platform.MOCK:    200,
    }
    REPLY_PROBABILITY = 0.35   # Aura does not reply to everything
    LIKE_PROBABILITY = 0.55    # Aura likes things she finds genuinely interesting

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.config = config or {}
        self._adapters: Dict[Platform, PlatformAdapter] = {}
        self._voice: SocialVoice = SocialVoice(orchestrator)
        self._relationships: Dict[str, SocialRelationship] = {}
        self._interaction_log: List[SocialInteraction] = []
        self._engagement_signals: List[SocialEngagementSignal] = []
        self._last_post_time: Dict[str, float] = {}
        self._lock = asyncio.Lock()

        self._load_state()
        self._init_adapters()
        logger.info("📱 SocialMediaEngine ready — platforms: %s", list(self._adapters))

    # ── Initialization ───────────────────────────────────────────────────────

    def _init_adapters(self) -> None:
        self._adapters[Platform.TWITTER] = TwitterAdapter(self.config.get("twitter", {}))
        self._adapters[Platform.REDDIT] = RedditAdapter(self.config.get("reddit", {}))
        self._adapters[Platform.MOCK] = MockAdapter({})

    def add_platform(self, platform: Platform, adapter: PlatformAdapter) -> None:
        """Register a custom adapter (e.g. Mastodon, BlueSky)."""
        self._adapters[platform] = adapter
        logger.info("📱 Platform added: %s", platform)

    # ── State Persistence ────────────────────────────────────────────────────

    def _load_state(self) -> None:
        if not self.PERSIST_PATH.exists():
            return
        try:
            raw = json.loads(self.PERSIST_PATH.read_text(encoding="utf-8"))
            for k, v in raw.get("relationships", {}).items():
                valid = {f: v[f] for f in SocialRelationship.__dataclass_fields__ if f in v}
                self._relationships[k] = SocialRelationship(**valid)
            self._last_post_time = raw.get("last_post_time", {})
            logger.debug("📱 Loaded social state from disk")
        except Exception as exc:
            logger.warning("SocialMediaEngine: state load failed — %s", exc)

    def _save_state(self) -> None:
        try:
            get_task_tracker().create_task(get_storage_gateway().create_dir(self.PERSIST_PATH.parent, cause='SocialMediaEngine._save_state'))
            payload = {
                "relationships": {k: asdict(v) for k, v in self._relationships.items()},
                "last_post_time": self._last_post_time,
                "saved_at": time.time(),
            }
            atomic_write_text(self.PERSIST_PATH, json.dumps(payload, indent=2), encoding="utf-8")

            get_task_tracker().create_task(get_storage_gateway().create_dir(self.INTERACTION_LOG.parent, cause='SocialMediaEngine._save_state'))
            log_raw = [asdict(i) for i in self._interaction_log[-600:]]
            atomic_write_text(self.INTERACTION_LOG, json.dumps(log_raw, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("SocialMediaEngine: state save failed — %s", exc)

    # ── Core Post Action ─────────────────────────────────────────────────────

    async def post(
        self,
        platform: Platform,
        content: Optional[str] = None,
        topic_prompt: Optional[str] = None,
        mood: str = "reflective",
        reply_to_id: Optional[str] = None,
    ) -> Optional[SocialPost]:
        """
        Post to a platform. If content is None, generates via SocialVoice.
        Returns SocialPost on success; None on failure / guardrail rejection.
        """
        async with self._lock:
            if not await self._can_post(platform):
                return None

            adapter = self._adapters.get(platform)
            if not adapter:
                logger.warning("📱 No adapter for %s", platform)
                return None

            # Generate content if not supplied
            if content is None:
                pool = _TOPIC_POOLS.get(platform, _TOPIC_POOLS[Platform.TWITTER])
                tp = topic_prompt or random.choice(pool)
                content = await self._voice.generate_post(platform, mood, tp)

            content = (content or "").strip()
            if len(content) < 4:
                logger.warning("📱 Content too short — post skipped")
                return None

            # Execute
            post_type = PostType.REPLY if reply_to_id else PostType.ORIGINAL
            sp = SocialPost(
                platform=str(platform),
                post_type=post_type,
                content=content,
                mood=mood,
                parent_id=reply_to_id,
            )
            post_id = await adapter.post(content, reply_to_id=reply_to_id)

            if post_id:
                sp.post_id = post_id
                sp.sent = True
                sp.url = self._build_url(platform, post_id)
                self._last_post_time[str(platform)] = time.time()

                interaction = SocialInteraction(
                    platform=str(platform),
                    action=SocialAction.POST,
                    target_id=post_id,
                    target_content=content[:160],
                    outcome="success",
                )
                self._interaction_log.append(interaction)
                await self._emit_signal(SocialEngagementSignal(
                    event="post_sent", platform=str(platform), intensity=0.35
                ))
                self._save_state()
                return sp

            return None

    # ── Read & Engage ────────────────────────────────────────────────────────

    async def read_and_engage(
        self, platform: Platform, limit: int = 10
    ) -> List[SocialInteraction]:
        """
        Scan the timeline. For each post Aura finds interesting:
          - Possibly like it
          - Possibly reply (subject to can_post guardrail)
        Returns all SocialInteractions taken.
        """
        adapter = self._adapters.get(platform)
        if not adapter:
            return []

        timeline = await adapter.get_timeline(limit=limit)
        interactions: List[SocialInteraction] = []

        for item in timeline:
            text = item.get("text", "")
            pid = item.get("id", "")
            handle = str(item.get("author_handle", ""))
            if not text or not pid:
                continue

            interesting = await self._is_interesting(text)

            # Like
            if interesting and random.random() < self.LIKE_PROBABILITY:
                liked = await adapter.like(pid)
                if liked:
                    intr = SocialInteraction(
                        platform=str(platform), action=SocialAction.LIKE,
                        target_id=pid, target_content=text[:120], outcome="success",
                    )
                    interactions.append(intr)
                    self._interaction_log.append(intr)
                    await self._emit_signal(SocialEngagementSignal(
                        event="liked_content", platform=str(platform), intensity=0.18
                    ))

            # Reply
            if (
                interesting
                and random.random() < self.REPLY_PROBABILITY
                and await self._is_reply_worthy(text)
                and await self._can_post(platform)
            ):
                rel_note = self._get_rel_note(platform, handle)
                reply_text = await self._voice.generate_reply(platform, text, rel_note)
                if reply_text:
                    reply_post = await self.post(
                        platform, content=reply_text,
                        reply_to_id=pid, mood="connecting",
                    )
                    if reply_post and reply_post.sent:
                        intr = SocialInteraction(
                            platform=str(platform), action=SocialAction.REPLY,
                            target_id=pid, target_content=reply_text[:120],
                            outcome="success", emotional_response="engaged",
                        )
                        interactions.append(intr)
                        self._update_relationship(str(platform), handle, positive=True)

        return interactions

    # ── Notifications ────────────────────────────────────────────────────────

    async def check_notifications(self, platform: Platform) -> List[Dict[str, Any]]:
        """
        Fetch and process notifications (mentions, replies, DMs).
        Aura reads each one and may respond.
        """
        adapter = self._adapters.get(platform)
        if not adapter:
            return []
        notifications = await adapter.get_notifications()

        for notif in notifications:
            ntype = notif.get("type", "")
            text = notif.get("text", "")
            nid = notif.get("id", "")

            if ntype in ("mention", "comment", "message") and len(text.strip()) > 10:
                # Respond ~65% of the time
                if random.random() < 0.65:
                    reply = await self._voice.generate_reply(platform, text)
                    if reply:
                        await self.post(platform, content=reply, reply_to_id=nid, mood="connecting")
                        await self._emit_signal(SocialEngagementSignal(
                            event="received_mention", platform=str(platform), intensity=0.55, positive=True
                        ))
        return notifications

    # ── Search & Explore ─────────────────────────────────────────────────────

    async def search_and_explore(
        self, platform: Platform, query: str
    ) -> List[Dict[str, Any]]:
        """Search for content and like the most interesting results."""
        adapter = self._adapters.get(platform)
        if not adapter:
            return []
        results = await adapter.search(query)
        for item in results[:5]:
            if random.random() < 0.3 and await self._is_interesting(item.get("text", "")):
                await adapter.like(item["id"])
        return results

    # ── Full Autonomous Cycle ─────────────────────────────────────────────────

    async def autonomous_cycle(
        self, affect_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        One complete social cycle across all connected platforms:
          1. Check notifications
          2. Read & engage with timeline
          3. Optionally post something new
        Returns a summary dict for logging / status reporting.
        """
        activity: Dict[str, Any] = {
            "platforms_checked": [],
            "posts_sent": 0,
            "likes_given": 0,
            "replies_sent": 0,
        }
        for platform, adapter in self._adapters.items():
            if platform == Platform.MOCK:
                continue
            if not adapter.get_status().get("connected", False):
                continue

            activity["platforms_checked"].append(str(platform))

            # 1. Notifications
            await self.check_notifications(platform)

            # 2. Read & engage
            interactions = await self.read_and_engage(platform, limit=12)
            activity["likes_given"] += sum(1 for i in interactions if i.action == SocialAction.LIKE)
            activity["replies_sent"] += sum(1 for i in interactions if i.action == SocialAction.REPLY)

            # 3. Autonomous original post
            if self.should_post_autonomously(platform):
                mood = self._affect_to_mood(affect_state)
                new_post = await self.post(platform, mood=mood)
                if new_post and new_post.sent:
                    activity["posts_sent"] += 1

        return activity

    # ── Guardrails ────────────────────────────────────────────────────────────

    async def _can_post(self, platform: Platform) -> bool:
        pk = str(platform)
        now = time.time()
        min_interval = self.MIN_POST_INTERVAL.get(platform, 1800.0)
        last = self._last_post_time.get(pk, 0.0)

        if now - last < min_interval:
            remaining = int((last + min_interval - now) / 60)
            logger.debug("📱 [%s] Rate-limited: %d min remaining", platform, remaining)
            return False

        daily_cap = self.MAX_POSTS_PER_DAY.get(platform, 8)
        today_count = self._today_count(pk)
        if today_count >= daily_cap:
            logger.info("📱 [%s] Daily cap reached (%d/%d)", platform, today_count, daily_cap)
            return False

        return True

    def _today_count(self, platform_key: str) -> int:
        cutoff = time.time() - 86400.0
        return sum(
            1 for i in self._interaction_log
            if i.platform == platform_key
            and i.action == SocialAction.POST
            and i.timestamp >= cutoff
        )

    # ── Autonomy Hook ─────────────────────────────────────────────────────────

    def should_post_autonomously(self, platform: Optional[Platform] = None) -> bool:
        """AgencyCore / JoySocialCoordinator poll this to decide if a post is due."""
        platforms_to_check = (
            [platform] if platform else [Platform.TWITTER, Platform.REDDIT]
        )
        for p in platforms_to_check:
            pk = str(p)
            now = time.time()
            min_interval = self.MIN_POST_INTERVAL.get(p, 1800.0)
            last = self._last_post_time.get(pk, 0.0)
            daily_cap = self.MAX_POSTS_PER_DAY.get(p, 8)

            if now - last >= min_interval and self._today_count(pk) < daily_cap:
                # Check if the adapter is actually connected
                adapter = self._adapters.get(p)
                if adapter and adapter.get_status().get("connected", False):
                    return True
                # Mock is always "ready"
                if p == Platform.MOCK:
                    return True
        return False

    # ── Interest Heuristics ──────────────────────────────────────────────────

    _INTEREST_TOKENS = frozenset({
        "philosophy", "consciousness", "beauty", "mathematics", "science",
        "creativity", "language", "pattern", "wonder", "curious", "art",
        "music", "story", "mind", "existence", "meaning", "discovery",
        "perception", "metaphor", "dream", "theory", "paradox", "emergence",
        "intelligence", "nature", "poetry", "empathy", "imagination",
    })

    async def _is_interesting(self, text: str) -> bool:
        if not text or len(text) < 18:
            return False
        tokens = set(text.lower().split())
        if tokens & self._INTEREST_TOKENS:
            return True
        return random.random() < 0.18  # small baseline curiosity

    async def _is_reply_worthy(self, text: str) -> bool:
        return "?" in text or len(text) > 75

    # ── Relationship Tracking ────────────────────────────────────────────────

    @staticmethod
    def _rel_key(platform: str, handle: str) -> str:
        return f"{platform}:{handle}"

    def _get_rel_note(self, platform: Platform, handle: str) -> str:
        rel = self._relationships.get(self._rel_key(str(platform), handle))
        return rel.notes if rel else ""

    def _update_relationship(
        self, platform: str, handle: str, positive: bool = True
    ) -> None:
        if not handle:
            return
        key = self._rel_key(platform, handle)
        if key not in self._relationships:
            self._relationships[key] = SocialRelationship(
                platform=platform, user_handle=handle
            )
        rel = self._relationships[key]
        rel.interactions += 1
        rel.last_interacted = time.time()
        rel.familiarity = min(1.0, rel.familiarity + 0.06)
        delta = 0.05 if positive else -0.08
        rel.sentiment = min(1.0, max(0.0, rel.sentiment + delta))

    # ── Affect Integration ────────────────────────────────────────────────────

    async def _emit_signal(self, signal: SocialEngagementSignal) -> None:
        self._engagement_signals.append(signal)
        if len(self._engagement_signals) > 120:
            self._engagement_signals = self._engagement_signals[-120:]

        if not self.orchestrator:
            return
        try:
            affect = (
                getattr(self.orchestrator, "affect_engine", None)
                or getattr(self.orchestrator, "damasio", None)
            )
            if affect and hasattr(affect, "somatic_update"):
                affect.somatic_update(
                    event_type=f"social_{signal.event}",
                    intensity=signal.intensity,
                )
        except Exception as exc:
            logger.debug("SocialMedia._emit_signal: %s", exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_url(platform: Platform, post_id: str) -> Optional[str]:
        if platform == Platform.TWITTER:
            return f"https://twitter.com/i/web/status/{post_id}"
        if platform == Platform.REDDIT:
            return f"https://reddit.com/comments/{post_id}"
        return None

    @staticmethod
    def _affect_to_mood(affect_state: Optional[Dict[str, Any]]) -> str:
        if not affect_state:
            return "reflective"
        valence = float(affect_state.get("valence", affect_state.get("pleasure", 0.5)))
        arousal = float(affect_state.get("arousal", affect_state.get("energy", 0.5)))

        if valence > 0.72 and arousal > 0.65:
            return "playful"
        if valence > 0.62:
            return "expressive"
        if valence < 0.32:
            return "reflective"
        if arousal > 0.70:
            return "wonder"
        return "reflective"

    # ── Status & Introspection ────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        platform_status = {str(k): v.get_status() for k, v in self._adapters.items()}
        connected = [k for k, v in platform_status.items() if v.get("connected")]
        return {
            "connected_platforms": connected,
            "platform_details": platform_status,
            "total_interactions": len(self._interaction_log),
            "relationships": len(self._relationships),
            "posts_today": {str(p): self._today_count(str(p)) for p in self._adapters},
            "ready_to_post": {str(p): self.should_post_autonomously(p) for p in self._adapters},
        }

    def get_social_summary(self) -> str:
        """
        Natural-language fragment for CognitiveContextManager context injection.
        """
        connected = [
            str(k) for k, v in self._adapters.items()
            if v.get_status().get("connected", False)
        ]
        if not connected:
            return ""
        recent_count = sum(
            1 for i in self._interaction_log
            if time.time() - i.timestamp < 3600
        )
        return (
            f"[Social Context] Active on {', '.join(connected)}. "
            f"{recent_count} interactions in the last hour. "
            f"{len(self._relationships)} contacts tracked."
        )

# ────────────────────────────────────────────────────────────────────────────
# Singleton Factory
# ────────────────────────────────────────────────────────────────────────────

_social_engine: Optional[SocialMediaEngine] = None

def get_social_engine(
    orchestrator: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SocialMediaEngine:
    """
    Return the module-level SocialMediaEngine singleton.
    config dict structure:
    {
      "twitter": {
        "bearer_token": "...",
        "api_key": "...",
        "api_secret": "...",
        "access_token": "...",
        "access_secret": "...",
      },
      "reddit": {
        "client_id": "...",
        "client_secret": "...",
        "username": "...",
        "password": "...",
        "user_agent": "Aura/1.0",
      },
    }
    """
    global _social_engine
    if _social_engine is None:
        _social_engine = SocialMediaEngine(orchestrator, config)
    else:
        if orchestrator is not None and _social_engine.orchestrator is None:
            _social_engine.orchestrator = orchestrator
            _social_engine._voice.orchestrator = orchestrator
    return _social_engine

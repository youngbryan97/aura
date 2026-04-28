"""core/autonomy/content_method_router.py
─────────────────────────────────────────
Translate a ``(ContentItem, top_priority_level)`` pair into an ordered list
of concrete ``FetchAttempt`` plans. Each attempt is a (method, target) that
the content_fetcher can actually execute.

Design notes
------------
- The priority hierarchy from the curated-media doc is:
    1 watch/listen, 2 script, 3 transcript, 4 text, 5 commentary, 6 forum/wiki
- For each priority, we generate **multiple** concrete attempts where
  possible (e.g. priority 3 might yield: official auto-transcript via
  yt-dlp, fan transcript via web search, podcast transcript via web search).
- Capability detection: if yt-dlp isn't available we don't generate
  yt-dlp-dependent attempts. Defensive against environments missing optional
  tools.
- The router never fetches anything itself. It produces plans only.

Public API:
    router = MethodRouter()
    plans: list[FetchPlan] = router.plan(item, top_priority_level=1)
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

# ── Capability detection ──────────────────────────────────────────────────


def _ytdlp_available() -> bool:
    return shutil.which("yt-dlp") is not None


def _have_whisper() -> bool:
    try:
        import importlib
        importlib.util.find_spec("mlx_whisper")
        return True
    except Exception:
        try:
            import importlib
            importlib.util.find_spec("whisper")
            return True
        except Exception:
            return False


def _have_browser_executor() -> bool:
    try:
        import importlib
        importlib.util.find_spec("core.executors.browser_executor")
        return True
    except Exception:
        return False


# ── Domain helpers ────────────────────────────────────────────────────────

_YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com")
_NETFLIX_HOSTS = ("netflix.com", "www.netflix.com")
_AMAZON_VIDEO_HOSTS = ("primevideo.com", "www.primevideo.com", "amazon.com")
_TRANSCRIPT_FRIENDLY_DOMAINS = (
    "scrapsfromtheloft.com",
    "transcripts.foreverdreaming.org",
    "subslikescript.com",
    "imsdb.com",
    "simplyscripts.com",
    "scriptslug.com",
)
_WIKI_DOMAINS = ("wikipedia.org", "fandom.com", "tvtropes.org")
_DISCUSSION_DOMAINS = ("reddit.com", "lesswrong.com", "news.ycombinator.com")


def _host_of(url: str) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _is_youtube_channel(url: str) -> bool:
    h = _host_of(url)
    return h in _YOUTUBE_HOSTS and ("/@" in url or "/c/" in url or "/channel/" in url or "/user/" in url)


def _is_youtube_video(url: str) -> bool:
    h = _host_of(url)
    return h in _YOUTUBE_HOSTS and not _is_youtube_channel(url)


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FetchAttempt:
    """A single concrete fetch action."""
    method: str   # "ytdlp_video", "ytdlp_audio_transcribe", "ytdlp_subtitles",
                  # "web_html", "web_search", "browser_navigate", "wikipedia_api",
                  # "fan_transcript_search", "creator_interview_search"
    priority_level: int
    target: str   # URL or query
    args: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class FetchPlan:
    """An ordered list of attempts the fetcher should try in sequence,
    accepting the first that produces usable content."""
    item_title: str
    top_priority_level: int
    attempts: List[FetchAttempt] = field(default_factory=list)
    capability_notes: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.attempts


# ── Router ────────────────────────────────────────────────────────────────


class MethodRouter:
    def __init__(
        self,
        ytdlp_path: Optional[str] = None,
        whisper_available: Optional[bool] = None,
        browser_available: Optional[bool] = None,
    ) -> None:
        self._ytdlp = ytdlp_path or (shutil.which("yt-dlp") if _ytdlp_available() else None)
        self._has_whisper = _have_whisper() if whisper_available is None else whisper_available
        self._has_browser = _have_browser_executor() if browser_available is None else browser_available

    def plan(self, item: Any, top_priority_level: int = 1) -> FetchPlan:
        title = getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else "")
        url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None)
        description = getattr(item, "description", None) or (item.get("description") if isinstance(item, dict) else "")

        plan = FetchPlan(item_title=str(title), top_priority_level=int(top_priority_level))
        self._note_capabilities(plan)

        # Generate attempts for each priority level from `top` down to 6,
        # with fallbacks at each level.
        for level in range(int(top_priority_level), 7):
            method_fn = {
                1: self._level1_watch_listen,
                2: self._level2_script,
                3: self._level3_transcript,
                4: self._level4_text,
                5: self._level5_commentary,
                6: self._level6_forum_wiki,
            }.get(level)
            if method_fn is None:
                continue
            for attempt in method_fn(title=str(title), url=str(url or ""), description=str(description or "")):
                plan.attempts.append(attempt)

        return plan

    # ── Capability surfacing ─────────────────────────────────────────────

    def _note_capabilities(self, plan: FetchPlan) -> None:
        if not self._ytdlp:
            plan.capability_notes.append("yt-dlp not on PATH — video/audio download attempts will be skipped")
        if not self._has_whisper:
            plan.capability_notes.append("Whisper not installed — audio transcription not available locally")
        if not self._has_browser:
            plan.capability_notes.append("browser_executor unavailable — using HTTP fetches only")

    # ── Per-priority-level planners ──────────────────────────────────────

    def _level1_watch_listen(self, title: str, url: str, description: str) -> List[FetchAttempt]:
        out: List[FetchAttempt] = []
        # Direct YouTube content
        if url and (_is_youtube_video(url) or _is_youtube_channel(url)):
            if self._ytdlp:
                if _is_youtube_video(url):
                    out.append(FetchAttempt(
                        method="ytdlp_video_with_subs",
                        priority_level=1,
                        target=url,
                        args={"writesubtitles": True, "writeautomaticsub": True, "skip_download": False,
                              "format": "bestvideo[height<=720]+bestaudio/best[height<=720]"},
                        rationale="YouTube video with subtitles, ≤720p to bound disk",
                    ))
                else:
                    # Channel: enumerate latest N entries
                    out.append(FetchAttempt(
                        method="ytdlp_channel_index",
                        priority_level=1,
                        target=url,
                        args={"max_entries": 5, "extract_flat": True},
                        rationale="Index latest 5 videos from channel; pick by curiosity scheduler",
                    ))
        # Streaming-platform content (Netflix etc.) — can't directly download
        elif url and _host_of(url) in (_NETFLIX_HOSTS + _AMAZON_VIDEO_HOSTS):
            # Drop down — note that direct watch isn't available
            pass  # no-op: intentional
        # Title-only entries (no URL): try a curated-search approach
        elif title and not url:
            out.append(FetchAttempt(
                method="web_search",
                priority_level=1,
                target=f'"{title}" official trailer site:youtube.com',
                args={"max_results": 5},
                rationale="Find an official source video as entry point",
            ))
        return out

    def _level2_script(self, title: str, url: str, description: str) -> List[FetchAttempt]:
        if not title:
            return []
        out: List[FetchAttempt] = []
        for query in [
            f'"{title}" screenplay PDF',
            f'"{title}" script transcript',
            f'"{title}" final draft script',
        ]:
            out.append(FetchAttempt(
                method="web_search",
                priority_level=2,
                target=query,
                args={"max_results": 5, "domain_hints": list(_TRANSCRIPT_FRIENDLY_DOMAINS)},
                rationale=f"Locate an authoritative script via: {query}",
            ))
        return out

    def _level3_transcript(self, title: str, url: str, description: str) -> List[FetchAttempt]:
        out: List[FetchAttempt] = []
        # YouTube-specific: pull subs directly via yt-dlp
        if self._ytdlp and url and _is_youtube_video(url):
            out.append(FetchAttempt(
                method="ytdlp_subtitles_only",
                priority_level=3,
                target=url,
                args={"writesubtitles": True, "writeautomaticsub": True, "skip_download": True,
                      "subtitleslangs": ["en", "en-US", "en-GB"]},
                rationale="Download YouTube transcript without the video",
            ))
        # Fan/community transcripts for any titled work
        if title:
            for query in [
                f'"{title}" full transcript',
                f'"{title}" episode transcript',
                f'"{title}" subtitles srt',
            ]:
                out.append(FetchAttempt(
                    method="web_search",
                    priority_level=3,
                    target=query,
                    args={"max_results": 5, "domain_hints": list(_TRANSCRIPT_FRIENDLY_DOMAINS)},
                    rationale=f"Locate a fan/community transcript: {query}",
                ))
        return out

    def _level4_text(self, title: str, url: str, description: str) -> List[FetchAttempt]:
        if not title:
            return []
        out: List[FetchAttempt] = []
        for query in [
            f'"{title}" novelization',
            f'"{title}" book companion',
            f'"{title}" comic adaptation',
        ]:
            out.append(FetchAttempt(
                method="web_search",
                priority_level=4,
                target=query,
                args={"max_results": 4},
                rationale=f"Find a long-form text version: {query}",
            ))
        return out

    def _level5_commentary(self, title: str, url: str, description: str) -> List[FetchAttempt]:
        if not title:
            return []
        out: List[FetchAttempt] = []
        for query in [
            f'"{title}" director interview',
            f'"{title}" creator commentary',
            f'"{title}" writer interview podcast',
            f'"{title}" actor interview about character',
        ]:
            out.append(FetchAttempt(
                method="web_search",
                priority_level=5,
                target=query,
                args={"max_results": 5},
                rationale=f"Surface creator/cast commentary: {query}",
            ))
        # If item points at a creator's channel: prioritize their own talks
        if url and _is_youtube_channel(url):
            out.append(FetchAttempt(
                method="ytdlp_channel_search_by_query",
                priority_level=5,
                target=url,
                args={"query": "interview OR commentary OR behind the scenes", "max_entries": 5},
                rationale="Search channel itself for creator commentary",
            ))
        return out

    def _level6_forum_wiki(self, title: str, url: str, description: str) -> List[FetchAttempt]:
        if not title:
            return []
        out: List[FetchAttempt] = []
        # Wikipedia (preferred for canonical summary + critical reception)
        out.append(FetchAttempt(
            method="wikipedia_api",
            priority_level=6,
            target=title,
            args={"prefer_lang": "en", "include_critical_section": True},
            rationale="Canonical article and critical reception section",
        ))
        # Fan wiki
        out.append(FetchAttempt(
            method="web_search",
            priority_level=6,
            target=f'"{title}" site:fandom.com',
            args={"max_results": 4, "domain_hints": list(_WIKI_DOMAINS)},
            rationale="Fan-curated wiki (often deeper than Wikipedia for media)",
        ))
        # TVTropes (genre/structure analysis)
        out.append(FetchAttempt(
            method="web_search",
            priority_level=6,
            target=f'"{title}" site:tvtropes.org',
            args={"max_results": 3, "domain_hints": ["tvtropes.org"]},
            rationale="TVTropes for narrative-structure analysis",
        ))
        # Critical / discussion forums
        for q in [
            f'"{title}" reddit discussion',
            f'"{title}" criticism essay',
            f'"{title}" fan analysis',
        ]:
            out.append(FetchAttempt(
                method="web_search",
                priority_level=6,
                target=q,
                args={"max_results": 4, "domain_hints": list(_DISCUSSION_DOMAINS)},
                rationale=f"Discussion / criticism: {q}",
            ))
        return out

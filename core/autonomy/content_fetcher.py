"""core/autonomy/content_fetcher.py
────────────────────────────────────
Execute a ``FetchPlan``: try each ``FetchAttempt`` in order, return the
first one that produces usable content. Defensive against missing tools,
network failures, rate limits, and partial reads.

Methods supported (graceful degradation when capability missing):
- ``ytdlp_video_with_subs`` / ``ytdlp_subtitles_only`` / ``ytdlp_channel_index``
- ``web_search``  (delegates to browser_executor when available, else
  returns a clear failure note)
- ``web_html``  (HTTP fetch with sane timeouts and size limits)
- ``wikipedia_api``  (direct REST fetch)
- ``ytdlp_audio_transcribe``  (yt-dlp audio extract + Whisper transcription)

Caching:
- Content cache keyed by SHA-256 of (method, target, args). LRU-evicted
  when cache total exceeds size budget.

Bandwidth/disk safety:
- Per-attempt size limit (default 50MB). yt-dlp video attempts use
  ≤720p to bound disk use.
- Total cache size budget (default 5GB) enforced before each write.

Public API:
    fetcher = ContentFetcher()
    result: FetchedContent = await fetcher.execute(plan)
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.state_ownership import state_root
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.ContentFetcher")

DEFAULT_PER_ATTEMPT_BYTES = 50 * 1024 * 1024
DEFAULT_TOTAL_CACHE_BYTES = 5 * 1024 * 1024 * 1024
#: External content goes stale. 24h is long enough that a research session
#: re-reading the same source does not refetch it, short enough that a
#: changed or retracted page is noticed within a day.
DEFAULT_CACHE_TTL_SECONDS = 24 * 3600.0
HTTP_TIMEOUT_SECONDS = 30
YTDLP_TIMEOUT_SECONDS = 600

_CONTENT_FETCHER_RECOVERABLE_ERRORS = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
)


def _record_fetch_degradation(
    subsystem: str,
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    classification: FallbackClassification = FallbackClassification.SAFE_FALLBACK,
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        subsystem,
        error,
        severity=severity,
        action=action,
        classification=classification,
        receipt_required=True,
        extra=extra,
    )


@dataclass
class FetchedContent:
    """Result of fulfilling a fetch attempt."""

    method: str
    priority_level: int
    target: str
    success: bool
    text: str = ""
    transcript: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_path: str | None = None
    bytes_fetched: int = 0
    error: str | None = None
    truncated: bool = False
    sources: list[str] = field(default_factory=list)


@dataclass
class FetchExecution:
    """Aggregate result of executing a FetchPlan."""

    plan_title: str
    successful: list[FetchedContent] = field(default_factory=list)
    failed: list[FetchedContent] = field(default_factory=list)
    capability_notes: list[str] = field(default_factory=list)

    def best_text(self) -> str:
        for c in self.successful:
            if c.transcript:
                return c.transcript
            if c.text:
                return c.text
        return ""

    def all_sources(self) -> list[str]:
        out: list[str] = []
        for c in self.successful:
            out.extend(c.sources or [])
            if c.target and c.target not in out:
                out.append(c.target)
        return out

    def priority_levels_engaged(self) -> list[int]:
        return sorted({c.priority_level for c in self.successful})


class ContentFetcher:
    def __init__(
        self,
        cache_dir: Path | None = None,
        per_attempt_bytes: int = DEFAULT_PER_ATTEMPT_BYTES,
        total_cache_bytes: int = DEFAULT_TOTAL_CACHE_BYTES,
        browser_executor: Any | None = None,
        whisper_transcriber: Any | None = None,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._cache_dir = cache_dir if cache_dir is not None else state_root() / "content_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._per_attempt_bytes = per_attempt_bytes
        self._total_cache_bytes = total_cache_bytes
        # How long a cached fetch of EXTERNAL content stays valid. There was no
        # expiry at all: a page fetched once was served as a successful fetch
        # forever, so Aura could reason about a source that had since changed or
        # been retracted and never notice.
        self._cache_ttl_seconds = max(60.0, float(cache_ttl_seconds))
        self._browser = browser_executor
        self._whisper = whisper_transcriber
        self._cache_index = self._cache_dir / "index.json"
        self._index = self._load_index()

    # ── Main execution ────────────────────────────────────────────────────

    async def execute(self, plan: Any, stop_after_n_successes: int = 4) -> FetchExecution:
        title = getattr(plan, "item_title", "") or ""
        execution = FetchExecution(
            plan_title=title,
            capability_notes=list(getattr(plan, "capability_notes", []) or []),
        )

        attempts = list(getattr(plan, "attempts", []) or [])
        for attempt in attempts:
            content = await self._execute_attempt(attempt)
            if content.success:
                execution.successful.append(content)
                if len(execution.successful) >= stop_after_n_successes:
                    break
            else:
                execution.failed.append(content)

        return execution

    # ── Per-attempt dispatch ─────────────────────────────────────────────

    async def _execute_attempt(self, attempt: Any) -> FetchedContent:
        method = getattr(attempt, "method", "")
        target = getattr(attempt, "target", "")
        priority_level = int(getattr(attempt, "priority_level", 6))
        args = dict(getattr(attempt, "args", {}) or {})

        # Cache check
        cache_key = self._cache_key(method, target, args)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return FetchedContent(
                method=method,
                priority_level=priority_level,
                target=target,
                success=True,
                text=cached.get("text", ""),
                transcript=cached.get("transcript", ""),
                metadata=cached.get("metadata", {}),
                cache_path=cached.get("cache_path"),
                bytes_fetched=int(cached.get("bytes_fetched", 0)),
                sources=cached.get("sources", []),
            )

        # Method dispatch
        try:
            if method == "ytdlp_video_with_subs":
                content = await self._ytdlp_video(target, args, priority_level)
            elif method == "ytdlp_subtitles_only":
                content = await self._ytdlp_subtitles(target, args, priority_level)
            elif method == "ytdlp_channel_index":
                content = await self._ytdlp_channel_index(target, args, priority_level)
            elif method == "ytdlp_channel_search_by_query":
                content = await self._ytdlp_channel_search(target, args, priority_level)
            elif method == "ytdlp_audio_transcribe":
                content = await self._ytdlp_audio_transcribe(target, args, priority_level)
            elif method == "web_search":
                content = await self._web_search(target, args, priority_level)
            elif method == "web_html":
                content = await self._web_html(target, args, priority_level)
            elif method == "wikipedia_api":
                content = await self._wikipedia_api(target, args, priority_level)
            elif method == "browser_navigate":
                content = await self._browser_navigate(target, args, priority_level)
            else:
                content = FetchedContent(
                    method=method,
                    priority_level=priority_level,
                    target=target,
                    success=False,
                    error=f"unknown method '{method}'",
                )
        except asyncio.CancelledError:
            raise
        except _CONTENT_FETCHER_RECOVERABLE_ERRORS as e:
            _record_fetch_degradation(
                "content_fetcher_attempt",
                e,
                action="returned failed fetch attempt after method execution failed",
                extra={"method": method, "target": target[:240]},
            )
            logger.warning("attempt method=%s target=%s failed: %s", method, target, e)
            content = FetchedContent(
                method=method,
                priority_level=priority_level,
                target=target,
                success=False,
                error=str(e),
            )

        if content.success:
            self._cache_put(cache_key, content)
        return content

    # ── yt-dlp methods ────────────────────────────────────────────────────

    async def _ytdlp_video(self, url: str, args: dict[str, Any], priority: int) -> FetchedContent:
        if not shutil.which("yt-dlp"):
            return FetchedContent(
                method="ytdlp_video_with_subs",
                priority_level=priority,
                target=url,
                success=False,
                error="yt-dlp not available",
            )
        with tempfile.TemporaryDirectory(prefix="aura_ytdlp_") as workdir:
            outtmpl = os.path.join(workdir, "%(id)s.%(ext)s")
            cmd = [
                "yt-dlp",
                "--no-playlist",
                "--restrict-filenames",
                "-o",
                outtmpl,
                "--max-filesize",
                str(self._per_attempt_bytes),
                "--write-auto-subs",
                "--write-subs",
                "--sub-langs",
                "en.*",
                "--convert-subs",
                "srt",
                "-f",
                str(args.get("format") or "bestvideo[height<=720]+bestaudio/best[height<=720]"),
                url,
            ]
            ok, stdout, stderr = await _run_subprocess(cmd, timeout_seconds=YTDLP_TIMEOUT_SECONDS)
            if not ok:
                return FetchedContent(
                    method="ytdlp_video_with_subs",
                    priority_level=priority,
                    target=url,
                    success=False,
                    error=f"yt-dlp: {stderr[-200:]}",
                )
            transcript = self._read_first_srt(workdir)
            metadata_path = self._read_first_json_dump(workdir)
            text_excerpt = transcript[
                : 4 * 1024
            ]  # cache excerpt only; full file stays in workdir until copied
            cache_path = (
                self._copy_to_cache(workdir, url) if (transcript or metadata_path) else None
            )
            total = self._dir_size(workdir)
            return FetchedContent(
                method="ytdlp_video_with_subs",
                priority_level=priority,
                target=url,
                success=bool(transcript or cache_path),
                text=text_excerpt,
                transcript=transcript,
                metadata={"metadata_files": metadata_path},
                cache_path=cache_path,
                bytes_fetched=total,
                sources=[url],
            )

    async def _ytdlp_subtitles(
        self, url: str, args: dict[str, Any], priority: int
    ) -> FetchedContent:
        if not shutil.which("yt-dlp"):
            return FetchedContent(
                method="ytdlp_subtitles_only",
                priority_level=priority,
                target=url,
                success=False,
                error="yt-dlp not available",
            )
        with tempfile.TemporaryDirectory(prefix="aura_ytdlp_subs_") as workdir:
            outtmpl = os.path.join(workdir, "%(id)s.%(ext)s")
            cmd = [
                "yt-dlp",
                "--no-playlist",
                "--skip-download",
                "--write-auto-subs",
                "--write-subs",
                "--sub-langs",
                "en.*",
                "--convert-subs",
                "srt",
                "-o",
                outtmpl,
                url,
            ]
            ok, stdout, stderr = await _run_subprocess(cmd, timeout_seconds=120)
            if not ok:
                return FetchedContent(
                    method="ytdlp_subtitles_only",
                    priority_level=priority,
                    target=url,
                    success=False,
                    error=f"yt-dlp: {stderr[-200:]}",
                )
            transcript = self._read_first_srt(workdir)
            return FetchedContent(
                method="ytdlp_subtitles_only",
                priority_level=priority,
                target=url,
                success=bool(transcript),
                transcript=transcript,
                sources=[url],
                bytes_fetched=len(transcript.encode("utf-8")) if transcript else 0,
            )

    async def _ytdlp_channel_index(
        self, url: str, args: dict[str, Any], priority: int
    ) -> FetchedContent:
        if not shutil.which("yt-dlp"):
            return FetchedContent(
                method="ytdlp_channel_index",
                priority_level=priority,
                target=url,
                success=False,
                error="yt-dlp not available",
            )
        max_entries = int(args.get("max_entries", 5))
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--print",
            "%(id)s\t%(title)s\t%(upload_date)s",
            "--playlist-end",
            str(max_entries),
            url,
        ]
        ok, stdout, stderr = await _run_subprocess(cmd, timeout_seconds=60)
        if not ok:
            return FetchedContent(
                method="ytdlp_channel_index",
                priority_level=priority,
                target=url,
                success=False,
                error=f"yt-dlp: {stderr[-200:]}",
            )
        entries = []
        for line in stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                entries.append(
                    {
                        "video_id": parts[0],
                        "title": parts[1],
                        "upload_date": parts[2] if len(parts) > 2 else None,
                    }
                )
        return FetchedContent(
            method="ytdlp_channel_index",
            priority_level=priority,
            target=url,
            success=bool(entries),
            text=json.dumps(entries, indent=2),
            metadata={"entries": entries},
            sources=[url],
            bytes_fetched=len(stdout.encode("utf-8")),
        )

    async def _ytdlp_channel_search(
        self, url: str, args: dict[str, Any], priority: int
    ) -> FetchedContent:
        # We list the channel and post-filter by query
        idx = await self._ytdlp_channel_index(url, {"max_entries": 30}, priority)
        if not idx.success:
            return FetchedContent(
                method="ytdlp_channel_search_by_query",
                priority_level=priority,
                target=url,
                success=False,
                error=idx.error,
            )
        query = (args.get("query") or "").lower()
        entries = idx.metadata.get("entries", [])
        if query:
            entries = [
                e for e in entries if any(q in e.get("title", "").lower() for q in query.split())
            ]
        entries = entries[: int(args.get("max_entries", 5))]
        return FetchedContent(
            method="ytdlp_channel_search_by_query",
            priority_level=priority,
            target=url,
            success=bool(entries),
            metadata={"entries": entries},
            text=json.dumps(entries, indent=2),
            sources=[url],
        )

    async def _ytdlp_audio_transcribe(
        self, url: str, args: dict[str, Any], priority: int
    ) -> FetchedContent:
        if self._whisper is None:
            return FetchedContent(
                method="ytdlp_audio_transcribe",
                priority_level=priority,
                target=url,
                success=False,
                error="whisper transcriber not configured",
            )
        if not shutil.which("yt-dlp"):
            return FetchedContent(
                method="ytdlp_audio_transcribe",
                priority_level=priority,
                target=url,
                success=False,
                error="yt-dlp not available",
            )

        with tempfile.TemporaryDirectory(prefix="aura_ytdlp_audio_") as workdir:
            outtmpl = os.path.join(workdir, "%(id)s.%(ext)s")
            cmd = [
                "yt-dlp",
                "--no-playlist",
                "--restrict-filenames",
                "--max-filesize",
                str(self._per_attempt_bytes),
                "-x",
                "--audio-format",
                str(args.get("audio_format") or "mp3"),
                "-o",
                outtmpl,
                url,
            ]
            ok, _stdout, stderr = await _run_subprocess(cmd, timeout_seconds=YTDLP_TIMEOUT_SECONDS)
            if not ok:
                return FetchedContent(
                    method="ytdlp_audio_transcribe",
                    priority_level=priority,
                    target=url,
                    success=False,
                    error=f"yt-dlp: {stderr[-200:]}",
                )
            audio_path = self._first_existing_media(
                workdir, {".mp3", ".m4a", ".wav", ".opus", ".webm"}
            )
            if audio_path is None:
                return FetchedContent(
                    method="ytdlp_audio_transcribe",
                    priority_level=priority,
                    target=url,
                    success=False,
                    error="audio extraction produced no media",
                )
            transcript = await _maybe_await_transcription(self._whisper, audio_path)
            text = str(transcript or "").strip()
            cache_path = self._copy_to_cache(workdir, url) if text else None
            audio_bytes = await asyncio.to_thread(_file_size, audio_path)
            return FetchedContent(
                method="ytdlp_audio_transcribe",
                priority_level=priority,
                target=url,
                success=bool(text),
                transcript=text,
                text=text[: 4 * 1024],
                cache_path=cache_path,
                bytes_fetched=audio_bytes,
                sources=[url],
                metadata={"audio_path": str(audio_path)},
            )

    # ── Web methods ──────────────────────────────────────────────────────

    async def _web_search(self, query: str, args: dict[str, Any], priority: int) -> FetchedContent:
        if self._browser is None:
            try:
                from executors import browser_executor as be_mod  # type: ignore

                self._browser = be_mod
            except (ImportError, AttributeError):
                return FetchedContent(
                    method="web_search",
                    priority_level=priority,
                    target=query,
                    success=False,
                    error="browser_executor unavailable; cannot search",
                )
        # Best-effort: try common entry-point names the executor may export
        for fn_name in ("search_async", "search", "web_search"):
            fn = getattr(self._browser, fn_name, None)
            if fn is None:
                continue
            try:
                result = fn(query) if not inspect.iscoroutinefunction(fn) else await fn(query)
                results = self._normalize_search_results(result, args)
                return FetchedContent(
                    method="web_search",
                    priority_level=priority,
                    target=query,
                    success=bool(results),
                    text=json.dumps(results, indent=2),
                    metadata={"results": results},
                    sources=[r.get("url", "") for r in results if r.get("url")],
                )
            except (
                ConnectionError,
                OSError,
                RuntimeError,
                TimeoutError,
                TypeError,
                ValueError,
            ) as e:
                _record_fetch_degradation(
                    "content_fetcher_web_search",
                    e,
                    action="tried next browser search entry point after search function failed",
                    severity="warning",
                    extra={"entry_point": fn_name, "query": query[:240]},
                )
                logger.debug("browser %s failed: %s", fn_name, e)
                continue
        return FetchedContent(
            method="web_search",
            priority_level=priority,
            target=query,
            success=False,
            error="no working search entry point on browser_executor",
        )

    async def _web_html(self, url: str, args: dict[str, Any], priority: int) -> FetchedContent:
        try:
            from core.runtime.network_gateway import get_network_gateway

            response = await asyncio.to_thread(
                get_network_gateway().request,
                "GET",
                url,
                headers={"User-Agent": "Aura/1.0 (+research)"},
                timeout=HTTP_TIMEOUT_SECONDS,
                source="content_fetcher.web_html",
                read_only=True,
            )
            if not response.get("ok"):
                raise OSError(str(response.get("error") or "web_html fetch failed"))
            data = bytes(response.get("content") or b"")
            status = int(response.get("status_code") or 0)
            truncated = len(data) > self._per_attempt_bytes
            if truncated:
                data = data[: self._per_attempt_bytes]
            text = self._strip_html(data.decode("utf-8", errors="replace"))
            return FetchedContent(
                method="web_html",
                priority_level=priority,
                target=url,
                success=bool(text),
                text=text,
                sources=[url],
                bytes_fetched=len(data),
                truncated=truncated,
                metadata={"http_status": status},
            )
        except (ConnectionError, OSError, TimeoutError, ValueError) as e:
            _record_fetch_degradation(
                "content_fetcher_web_html",
                e,
                action="returned failed web_html fetch after bounded HTTP request failed",
                extra={"url": url[:240]},
            )
            return FetchedContent(
                method="web_html", priority_level=priority, target=url, success=False, error=str(e)
            )

    async def _wikipedia_api(
        self, title: str, args: dict[str, Any], priority: int
    ) -> FetchedContent:
        lang = args.get("prefer_lang", "en")
        api = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&prop=extracts&exintro=0&explaintext=1&format=json&redirects=1"
            f"&titles={_url_encode(title)}"
        )
        result = await self._web_html(api, {}, priority)
        if not result.success:
            result.method = "wikipedia_api"
            return result
        try:
            payload = json.loads(result.text)
            pages = payload.get("query", {}).get("pages", {})
            for _pid, page in pages.items():
                extract = page.get("extract", "")
                if extract:
                    return FetchedContent(
                        method="wikipedia_api",
                        priority_level=priority,
                        target=title,
                        success=True,
                        text=extract,
                        sources=[f"https://{lang}.wikipedia.org/wiki/{_url_encode(title)}"],
                        bytes_fetched=len(extract.encode("utf-8")),
                        metadata={"page_title": page.get("title")},
                    )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            _record_fetch_degradation(
                "content_fetcher_wikipedia_parse",
                e,
                action="returned failed wikipedia_api fetch after response parse failed",
                extra={"title": title[:240]},
            )
            return FetchedContent(
                method="wikipedia_api",
                priority_level=priority,
                target=title,
                success=False,
                error=f"parse: {e}",
            )
        return FetchedContent(
            method="wikipedia_api",
            priority_level=priority,
            target=title,
            success=False,
            error="no extract in response",
        )

    async def _browser_navigate(
        self, url: str, args: dict[str, Any], priority: int
    ) -> FetchedContent:
        # Direct delegate to browser_executor when a richer DOM read is needed
        return await self._web_html(url, args, priority)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _normalize_search_results(self, raw: Any, args: dict[str, Any]) -> list[dict[str, str]]:
        """Browser executor APIs vary; flatten into [{title, url, snippet}]."""
        max_n = int(args.get("max_results", 5))
        results: list[dict[str, str]] = []
        if isinstance(raw, list):
            iterable = raw
        elif isinstance(raw, dict):
            iterable = raw.get("results") or raw.get("items") or []
        else:
            return results
        for r in iterable[: max_n * 2]:
            if isinstance(r, dict):
                results.append(
                    {
                        "title": str(r.get("title") or r.get("name") or "")[:200],
                        "url": str(r.get("url") or r.get("link") or ""),
                        "snippet": str(r.get("snippet") or r.get("description") or "")[:400],
                    }
                )
        return [r for r in results if r["url"]][:max_n]

    def _strip_html(self, html: str) -> str:
        try:
            import re as _re

            text = _re.sub(
                r"<script[^>]*>.*?</script>", " ", html, flags=_re.DOTALL | _re.IGNORECASE
            )
            text = _re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=_re.DOTALL | _re.IGNORECASE)
            text = _re.sub(r"<[^>]+>", " ", text)
            text = _re.sub(r"\s+", " ", text)
            return text.strip()
        except (TypeError, re.error):
            return html

    def _read_first_srt(self, workdir: str) -> str:
        for name in os.listdir(workdir):
            if name.endswith(".srt"):
                try:
                    return self._strip_srt_timing(
                        Path(workdir, name).read_text(encoding="utf-8", errors="replace")
                    )
                except (OSError, UnicodeDecodeError):
                    continue
        return ""

    def _strip_srt_timing(self, srt: str) -> str:
        out: list[str] = []
        for line in srt.splitlines():
            line = line.strip()
            if not line or line.isdigit() or "-->" in line:
                continue
            out.append(line)
        return " ".join(out)

    def _read_first_json_dump(self, workdir: str) -> str | None:
        for name in os.listdir(workdir):
            if name.endswith(".info.json"):
                return os.path.join(workdir, name)
        return None

    def _first_existing_media(self, workdir: str, suffixes: set[str]) -> Path | None:
        for name in os.listdir(workdir):
            path = Path(workdir, name)
            if path.is_file() and path.suffix.lower() in suffixes:
                return path
        return None

    def _copy_to_cache(self, workdir: str, key: str) -> str | None:
        try:
            self._enforce_cache_budget()
            target_dir = self._cache_dir / hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
            target_dir.mkdir(parents=True, exist_ok=True)
            for name in os.listdir(workdir):
                src = os.path.join(workdir, name)
                if os.path.isfile(src):
                    shutil.copy2(src, target_dir / name)
            return str(target_dir)
        except (OSError, shutil.Error) as e:
            _record_fetch_degradation(
                "content_fetcher_cache_copy",
                e,
                action="continued fetch result without copying auxiliary files to cache",
                severity="warning",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"cache_key": hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]},
            )
            logger.debug("cache copy failed: %s", e)
            return None

    def _dir_size(self, path: str) -> int:
        total = 0
        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
        except OSError as e:
            _record_fetch_degradation(
                "content_fetcher_dir_size",
                e,
                action="treated inaccessible temporary directory as zero bytes during accounting",
                severity="warning",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"path": path},
            )
        return total

    def _enforce_cache_budget(self) -> None:
        try:
            entries: list[tuple[float, Path, int]] = []
            for child in self._cache_dir.iterdir():
                if child.is_dir():
                    size = sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
                    entries.append((child.stat().st_mtime, child, size))
            total = sum(s for _, _, s in entries)
            if total <= self._total_cache_bytes:
                return
            entries.sort(key=lambda e: e[0])
            evicted: list[str] = []
            for _mtime, path, size in entries:
                if total <= self._total_cache_bytes * 0.8:
                    break
                shutil.rmtree(path, ignore_errors=True)
                evicted.append(str(path))
                total -= size

            # Eviction removed the DIRECTORIES and left their index records
            # intact, so later reads returned success with a cache_path that no
            # longer existed. The index is pruned in the same operation.
            if evicted:
                stale = [
                    key for key, record in self._index.items()
                    if any(
                        str(record.get("cache_path") or "").startswith(prefix)
                        for prefix in evicted
                    )
                ]
                for key in stale:
                    self._index.pop(key, None)
                if stale:
                    self._save_index()
                    logger.info(
                        "Cache eviction removed %d directory(ies) and %d index "
                        "record(s).", len(evicted), len(stale),
                    )
        except OSError as e:
            _record_fetch_degradation(
                "content_fetcher_cache_budget",
                e,
                action="skipped cache budget enforcement after cache scan failed",
                severity="warning",
                classification=FallbackClassification.AUDIT_GAP,
            )
            logger.debug("cache budget enforcement failed: %s", e)

    def _cache_key(self, method: str, target: str, args: dict[str, Any]) -> str:
        h = hashlib.sha256()
        h.update(method.encode("utf-8"))
        h.update(b"::")
        h.update(target.encode("utf-8"))
        h.update(b"::")
        h.update(json.dumps(args, sort_keys=True, default=str).encode("utf-8"))
        return h.hexdigest()[:32]

    def _get_cached(self, key: str) -> dict[str, Any] | None:
        """Return a cache record only if it is fresh AND its files still exist.

        This used to be a bare index lookup, so a record was served forever with
        success=True: stored_at was written and never read, and there was no
        TTL, ETag, or invalidation of any kind. External content that had since
        changed — or been deleted from disk by budget eviction, which removes
        cache DIRECTORIES without touching their index records — was returned as
        a successful fetch, with a cache_path pointing at nothing and only the
        truncated inline copy behind it.
        """
        record = self._index.get(key)
        if record is None:
            return None

        stored_at = record.get("stored_at")
        try:
            age = time.time() - float(stored_at)
        except (TypeError, ValueError):
            age = None
        if age is None or age < 0 or age > self._cache_ttl_seconds:
            logger.debug("cache entry %s expired (age=%s); refetching.", key, age)
            self._drop_cache_entry(key)
            return None

        # Eviction deletes directories but left the index claiming success.
        cache_path = record.get("cache_path")
        if cache_path and not Path(cache_path).exists():
            logger.debug(
                "cache entry %s references a path that no longer exists; refetching.",
                key,
            )
            self._drop_cache_entry(key)
            return None
        return record

    def _drop_cache_entry(self, key: str) -> None:
        """Remove a stale or dangling index record."""
        if self._index.pop(key, None) is not None:
            self._save_index()

    def _cache_put(self, key: str, content: FetchedContent) -> None:
        self._index[key] = {
            "method": content.method,
            "target": content.target,
            "text": content.text[: 64 * 1024],
            "transcript": content.transcript[: 256 * 1024],
            "metadata": content.metadata,
            "cache_path": content.cache_path,
            "bytes_fetched": content.bytes_fetched,
            "sources": content.sources,
            "stored_at": time.time(),
        }
        self._save_index()

    def _load_index(self) -> dict[str, Any]:
        if not self._cache_index.exists():
            return {}
        try:
            return json.loads(self._cache_index.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError) as e:
            _record_fetch_degradation(
                "content_fetcher_cache_index",
                e,
                action="started with empty content cache index after index load failed",
                severity="warning",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"cache_index": str(self._cache_index)},
            )
            return {}

    def _save_index(self) -> None:
        try:
            atomic_write_text(self._cache_index, json.dumps(self._index), encoding="utf-8")
        except OSError as e:
            _record_fetch_degradation(
                "content_fetcher_cache_index",
                e,
                action="kept in-memory content cache index after index save failed",
                severity="warning",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"cache_index": str(self._cache_index)},
            )
            logger.debug("cache index save failed: %s", e)


# ── Subprocess helper ────────────────────────────────────────────────────


async def _terminate_subprocess(proc: Any, cmd: list[str], *, reason: str) -> None:
    """Kill and reap a child, and its process GROUP where the platform allows.

    yt-dlp and ffmpeg spawn their own children, so killing only the direct
    child can leave grandchildren running. Never raises: this runs on timeout
    and cancellation paths that must complete regardless.
    """
    binary = cmd[0] if cmd else ""
    try:
        try:
            # Terminate the whole group when the gateway gave us a session
            # leader; fall back to the direct child otherwise.
            pgid = os.getpgid(proc.pid) if hasattr(os, "getpgid") else None
        except (ProcessLookupError, OSError, AttributeError):
            pgid = None
        if pgid is not None and pgid != os.getpgid(os.getpid()):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError, AttributeError):
                proc.kill()
        else:
            proc.kill()
        await asyncio.wait_for(proc.wait(), timeout=5)
    except (TimeoutError, ProcessLookupError, OSError, AttributeError, RuntimeError) as e:
        _record_fetch_degradation(
            "content_fetcher_subprocess",
            e,
            action=f"subprocess {reason} cleanup incomplete; child may survive",
            severity="warning",
            extra={"binary": binary, "reason": reason},
        )


async def _run_subprocess(cmd: list[str], timeout_seconds: int) -> tuple[bool, str, str]:
    try:
        proc = await get_subprocess_gateway().spawn_async(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            source="external_action:content_fetcher.subprocess",
            accelerator_capability="auto",
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            await _terminate_subprocess(proc, cmd, reason="timeout")
            return False, "", "timeout"
        except asyncio.CancelledError:
            # Cancellation used to propagate straight out of communicate() with
            # the child still running — a yt-dlp download or ffmpeg transcode
            # kept going after the task that owned it was gone, holding
            # bandwidth, CPU and disk with nothing left to reap it. Cancellation
            # is the most common shutdown path, so this leaked routinely.
            await _terminate_subprocess(proc, cmd, reason="cancelled")
            raise
        ok = proc.returncode == 0
        return (
            ok,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except FileNotFoundError as e:
        return False, "", f"binary missing: {e}"
    except (OSError, RuntimeError, ValueError) as e:
        _record_fetch_degradation(
            "content_fetcher_subprocess",
            e,
            action="returned failed subprocess result after process launch failed",
            extra={"binary": cmd[0] if cmd else ""},
        )
        return False, "", str(e)


async def _maybe_await_transcription(transcriber: Any, audio_path: Path) -> Any:
    for attr_name in ("transcribe_file", "transcribe_path", "transcribe"):
        fn = getattr(transcriber, attr_name, None)
        if callable(fn):
            result = fn(str(audio_path))
            if inspect.isawaitable(result):
                return await result
            return result
    if callable(transcriber):
        result = transcriber(str(audio_path))
        if inspect.isawaitable(result):
            return await result
        return result
    raise RuntimeError("whisper transcriber exposes no supported transcribe method")


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _url_encode(s: str) -> str:
    try:
        from urllib.parse import quote

        return quote(s)
    except (ImportError, TypeError, ValueError):
        return s

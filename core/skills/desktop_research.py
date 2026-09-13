"""Reading sources, and writing a document that cites them.

A synthesis with no sources is an opinion with a filename. Everything here is
about keeping the two attached: which pages are worth reading, how recent each
one is, what survives stripping the page chrome, and the reference block the
finished document carries. Where a source cannot be read, it says so rather
than writing around the hole.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from core.runtime.content_integrity import (
    text_sha256,
)
from core.runtime.errors import record_degradation


class _ResearchesBeforeItWrites:
    """Lifted whole from DesktopTaskSkill; see desktop_task.py."""

    @classmethod
    def _is_article_url(cls, url: str) -> bool:
        """A source has to be an ARTICLE, not an ad or a search page."""
        candidate = str(url or "").strip()
        if not candidate.startswith(("http://", "https://")):
            return False
        if cls._NON_ARTICLE_URL_RE.search(candidate):
            return False
        # A bare homepage is a product, not a piece of reporting.
        try:
            from urllib.parse import urlsplit

            parts = urlsplit(candidate)
        except (TypeError, ValueError):
            # not a failure: being asked whether this is a page address and
            # answering that it is not.
            return False
        path = (parts.path or "/").rstrip("/")
        return bool(path and path != "")

    @classmethod
    def _strip_page_chrome(cls, text: str) -> str:
        """Remove navigation furniture so a synthesis quotes the article.

        Measured live: the "synthesis" written into the document opened with
        "Skip to main content Research Products Business Developers Company
        Foundation (opens in a new window) Log in Try ChatGPT (opens in a new
        window)..." — the site's nav bar, repeated twice, presented as what the
        reporting said.
        """
        body = " ".join(str(text or "").split())
        if not body:
            return ""
        body = cls._PAGE_CHROME_RE.sub(" ", body)
        # Collapse the runs of single words nav bars leave behind.
        body = re.sub(r"\s{2,}", " ", body).strip(" -|·•,")
        prose = cls._prose_sentences_only(body)
        # Only trust the sentence filter when it actually found prose; a page
        # that is genuinely all fragments should degrade to the cleaned text
        # rather than to nothing.
        return prose or body

    @staticmethod
    def _prose_sentences_only(text: str) -> str:
        """Keep the sentences and drop the navigation.

        A phrase list cannot generalise: every site's nav has its own
        vocabulary. NASA's survived the phrase filter intact — "Explore Search
        News & Events News & Events Recently Published Video Series on NASA+
        Podcasts & Audio Blogs Newsletters Social Media Media Resources" — and
        was written into the document as what the reporting said.

        What separates the two is grammar, not words. Reporting is sentences:
        they run long, they carry lowercase function words, they end in a full
        stop. Navigation is a run of Title Case fragments with almost no verbs
        and almost no periods. Selecting by sentence shape keeps the real line
        from the same page — "The concentration of the 2023 warming in
        near-surface waters suggests that upper ocean stratification ... may
        have played an important role" — and drops the menu around it.
        """

        raw = " ".join(str(text or "").split())
        if not raw:
            return ""
        def _lowercase_ratio(words: list[str]) -> int:
            return sum(1 for word in words if word[:1].islower() and word.isalpha())

        kept: list[str] = []
        # The ellipsis is a boundary too: extractors truncate a nav run with "…"
        # and the sentence that follows would otherwise be welded to it.
        for chunk in re.split(r"(?<=[.!?])\s+|…\s*", raw):
            candidate = chunk.strip()
            words = candidate.split()
            if len(words) < 8:
                continue
            # A chunk can OPEN with the menu and end in a real sentence. Walk
            # forward to where prose actually starts instead of judging the
            # whole chunk by its nav prefix.
            start = 0
            while start < len(words) - 7:
                window = words[start : start + 8]
                if _lowercase_ratio(window) >= 3:
                    break
                start += 1
            else:
                start = 0
            candidate_words = words[start:]
            if len(candidate_words) < 8:
                continue
            if _lowercase_ratio(candidate_words) < max(
                4, len(candidate_words) // 3
            ):
                continue
            kept.append(" ".join(candidate_words))
        return " ".join(kept).strip()

    @staticmethod
    def _research_sources_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Join provider citations to their fetched evidence by source identity.

        Deep search returns URLs in ``citations`` and article bodies in
        ``chunks``. Choosing the first non-empty list discarded the bodies and
        made a successfully read article look inaccessible. Merge all provider
        surfaces instead, keeping the richest text for each URL.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .desktop_task import (
            DesktopTaskSkill,
        )

        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for field in ("citations", "sources", "results", "chunks"):
            raw_sources = result.get(field) or []
            if not isinstance(raw_sources, list):
                continue
            for item in raw_sources[:8]:
                if not isinstance(item, dict):
                    continue
                title = str(
                    item.get("title")
                    or item.get("name")
                    or item.get("url")
                    or item.get("link")
                    or ""
                ).strip()
                url = str(
                    item.get("url") or item.get("link") or item.get("uri") or ""
                ).strip()
                raw_text = str(
                    item.get("snippet")
                    or item.get("text")
                    or item.get("content")
                    or item.get("summary")
                    or ""
                ).strip()
                if not title and not url and not raw_text:
                    continue
                if url and not DesktopTaskSkill._is_article_url(url):
                    continue
                key = url.casefold() or title.casefold()
                if not key:
                    continue
                if key not in merged:
                    merged[key] = {
                        "title": "",
                        "url": "",
                        "snippet": "",
                        "article_body_chunks": [],
                    }
                    order.append(key)
                target = merged[key]
                if len(title) > len(str(target.get("title") or "")):
                    target["title"] = title[:240]
                if url:
                    target["url"] = url[:500]
                cleaned = DesktopTaskSkill._strip_page_chrome(raw_text)
                evidence_kind = str(item.get("evidence_kind") or "").strip().casefold()
                fetched_body = evidence_kind == "article_body" and bool(
                    item.get("fetched", True)
                )
                if fetched_body and cleaned:
                    chunks = target["article_body_chunks"]
                    if cleaned not in chunks:
                        chunks.append(cleaned[:1600])
                elif len(cleaned) > len(str(target.get("snippet") or "")):
                    target["snippet"] = cleaned[:900]
                for metadata_field in (
                    "fetched_at",
                    "document_chars",
                    "document_sha256",
                ):
                    value = item.get(metadata_field)
                    if value and not target.get(metadata_field):
                        target[metadata_field] = value
                published = str(
                    item.get("published_at")
                    or item.get("publication_date")
                    or item.get("date_published")
                    or item.get("date")
                    or ""
                ).strip()
                if published and not target.get("published_at"):
                    target["published_at"] = published[:80]
        sources = []
        for key in order:
            source = merged[key]
            article_body = "\n\n".join(
                str(chunk) for chunk in source.pop("article_body_chunks", []) if chunk
            )[:4000]
            if article_body:
                source["article_body"] = article_body
                source["article_body_chars"] = len(article_body)
                source["article_body_sha256"] = text_sha256(article_body)
                source["source_evidence_sha256"] = text_sha256(
                    f"{source.get('url') or source.get('title') or ''}\n{article_body}"
                )
                source["read_evidence_kind"] = "fetched_article_body"
                source["snippet"] = article_body[:1800]
            title = str(source.get("title") or "")
            url = str(source.get("url") or "")
            snippet = article_body or str(source.get("snippet") or "")
            source["reputability"] = DesktopTaskSkill._source_reputability(url, title)
            source["accessible"] = not DesktopTaskSkill._looks_inaccessible(snippet)
            sources.append(source)
        # Rank reputable, accessible sources first so synthesis leans on them;
        # clearly inaccessible (paywall/ad-wall/empty) sources sink to the bottom
        # rather than being relied on, but are retained for transparency.
        sources.sort(
            key=lambda s: (bool(s.get("accessible")), int(s.get("reputability", 0))),
            reverse=True,
        )
        return sources[:5]

    @staticmethod
    def _merge_research_sources(
        *groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for group in groups:
            for source in group:
                url = str(source.get("url") or "").strip()
                title = str(source.get("title") or "").strip()
                key = url.casefold() or title.casefold()
                if not key:
                    continue
                if key not in merged:
                    merged[key] = dict(source)
                    order.append(key)
                    continue
                current = merged[key]
                if len(str(source.get("snippet") or "")) > len(
                    str(current.get("snippet") or "")
                ):
                    current["snippet"] = source.get("snippet")
                if len(str(source.get("article_body") or "")) > len(
                    str(current.get("article_body") or "")
                ):
                    current["article_body"] = source.get("article_body")
                    current["article_body_chars"] = source.get("article_body_chars")
                    current["article_body_sha256"] = source.get("article_body_sha256")
                    current["source_evidence_sha256"] = source.get(
                        "source_evidence_sha256"
                    )
                    current["read_evidence_kind"] = source.get("read_evidence_kind")
                for field in (
                    "title",
                    "url",
                    "published_at",
                    "fetched_at",
                    "document_chars",
                    "document_sha256",
                ):
                    if source.get(field) and not current.get(field):
                        current[field] = source[field]
                current["accessible"] = bool(
                    current.get("accessible") or source.get("accessible")
                )
                current["reputability"] = max(
                    int(current.get("reputability") or 0),
                    int(source.get("reputability") or 0),
                )
        return [merged[key] for key in order]

    @staticmethod
    def _source_recency_evidence(source: Mapping[str, Any]) -> tuple[bool, str]:
        """Verify recency from source metadata, never from search rank alone."""

        raw = str(source.get("published_at") or "").strip()
        published: datetime | None = None
        if raw:
            try:
                published = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                try:
                    published = parsedate_to_datetime(raw)
                except (TypeError, ValueError, OverflowError):
                    # not a failure: an item without a readable date is an
                    # item without a date, which is a thing items are.
                    published = None
        now = datetime.now(UTC)
        if published is not None:
            if published.tzinfo is None:
                published = published.replace(tzinfo=UTC)
            age_days = (now - published.astimezone(UTC)).total_seconds() / 86400.0
            return (-7.0 <= age_days <= 366.0), f"published_at:{raw}"

        # Some providers omit a date field while retaining the publication year
        # in the canonical article URL/title. This is weaker than a full date but
        # still auditable evidence; accept only the current or immediately prior
        # year, never the fact that the query contained the word "recent".
        haystack = f"{source.get('url') or ''} {source.get('title') or ''}"
        years = {int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", haystack)}
        if years & {now.year, now.year - 1}:
            return True, f"article_year:{max(years)}"
        return False, "publication_date_unverified"

    @classmethod
    def _usable_research_sources(
        cls,
        sources: list[dict[str, Any]],
        *,
        require_recent: bool,
        require_read: bool,
    ) -> list[dict[str, Any]]:
        usable: list[dict[str, Any]] = []
        for source in sources:
            item = dict(source)
            article_body = str(item.get("article_body") or "").strip()
            article_digest = str(item.get("article_body_sha256") or "").strip()
            read_verified = bool(
                item.get("accessible")
                and item.get("read_evidence_kind") == "fetched_article_body"
                and len(article_body) >= 120
                and len(article_body.split()) >= 18
                and cls._valid_sha256(article_digest)
                and article_digest == text_sha256(article_body)
            )
            recent_verified, recency_evidence = cls._source_recency_evidence(item)
            item["read_verified"] = read_verified
            item["recency_verified"] = recent_verified
            item["recency_evidence"] = recency_evidence
            if (require_read and not read_verified) or (
                require_recent and not recent_verified
            ):
                continue
            usable.append(item)
        return usable

    @staticmethod
    def _source_reputability(url: str, title: str = "") -> int:
        """Coarse 0–3 reputability score from the source domain."""
        from .desktop_task import (
            DesktopTaskSkill,
        )

        u = str(url or "").lower()
        if not u:
            return 0
        if any(u.endswith(tld) or f"{tld}/" in u for tld in DesktopTaskSkill._REPUTABLE_TLDS):
            return 3
        if any(dom in u for dom in DesktopTaskSkill._REPUTABLE_DOMAINS):
            return 2
        if any(bad in u for bad in DesktopTaskSkill._LOW_QUALITY_HINTS):
            return 0
        return 1

    @staticmethod
    def _looks_inaccessible(snippet: str) -> bool:
        """True when the fetched content looks paywalled, ad-walled, or empty —
        a signal to prefer a different source rather than rely on this one."""
        from .desktop_task import (
            DesktopTaskSkill,
        )

        text = str(snippet or "").strip()
        if len(text) < 60:
            return True
        lowered = text.lower()
        return any(hint in lowered for hint in DesktopTaskSkill._PAYWALL_HINTS)

    @classmethod
    def _research_section_from_context(cls, context: dict[str, Any] | None) -> str:
        context = context or {}
        synthesis = str(context.get("desktop_task_research_synthesis") or "").strip()
        summary = str(context.get("desktop_task_research_summary") or "").strip()
        query = str(context.get("desktop_task_research_query") or "").strip()
        sources = context.get("desktop_task_research_sources") or []
        if not synthesis and not summary and not sources:
            return ""

        lines = []
        if synthesis:
            # Aura's own first-person summary (and opinion) leads the
            # document; the raw search summary is dropped in favor of it.
            lines.append(synthesis)
            lines.append("")
        else:
            heading = "Research summary"
            if query:
                heading += f" for: {query}"
            lines.append(heading)
            lines.append("")
            if summary:
                lines.append(summary[:2500])
                lines.append("")
        if isinstance(sources, list) and sources:
            lines.append("Sources opened or consulted:")
            for index, item in enumerate(sources[:5], start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "Untitled source").strip()
                url = str(item.get("url") or "").strip()
                snippet = str(item.get("snippet") or "").strip()
                source_line = f"{index}. {title}"
                if url:
                    source_line += f" — {url}"
                lines.append(source_line)
                if snippet:
                    lines.append(f"   {snippet[:300]}")
        return "\n".join(lines).strip()

    @classmethod
    def _document_body_with_references(
        cls,
        objective: str,
        context: dict[str, Any] | None,
        *,
        image_query: str = "",
        image_search_url: str = "",
        search_url: str = "",
    ) -> str:
        body = cls._document_body(objective, context)
        research_section = cls._research_section_from_context(context)
        if research_section and cls._objective_requests_research_document(objective):
            lowered_body = body.lower()
            if cls._looks_like_dispatch_narration(body) or re.search(
                r"\bi\s+will\s+(?:open|search|look|create|write|start|follow|route)\b",
                lowered_body,
            ):
                body = research_section
            elif research_section not in body:
                body = f"{body.rstrip()}\n\n{research_section}"
        references: list[str] = []
        if search_url:
            references.append(f"Search opened: {search_url}")
        if image_query and image_search_url:
            references.append(
                f"Image request: {image_query}\n"
                "The exported artifact embeds the fetched image only after the governed image receipt verifies the file; "
                "the receipt records the source page used for the image."
            )
        if not references:
            return body
        return f"{body.rstrip()}\n\nArtifact references:\n" + "\n".join(f"- {item}" for item in references)

    @classmethod
    def _compose_research_synthesis_from_sources(
        cls,
        *,
        objective: str,
        query: str,
        summary: str,
        sources: list[dict[str, str]],
    ) -> str:
        """Compose a bounded source-backed document without a second model call."""

        summary = " ".join(str(summary or "").split())[:1400]
        source_lines: list[str] = []
        source_titles: list[str] = []
        source_notes: list[str] = []
        for item in (sources or [])[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("url") or "Untitled source").strip()
            snippet = " ".join(
                str(item.get("article_body") or item.get("snippet") or "").split()
            )
            url = str(item.get("url") or "").strip()
            if title:
                source_titles.append(title[:160])
            if snippet:
                source_lines.append(f"{title}: {cls._clip_to_sentence(snippet, 240)}")
                source_notes.append(
                    f"{title[:140]} reports or documents that "
                    + cls._clip_to_sentence(snippet, 360)
                    + (f" ({url})" if url else "")
                )
            else:
                source_lines.append(cls._clip_to_sentence(title, 240))
                source_notes.append(f"{title[:180]}" + (f" ({url})" if url else ""))
        topic = str(query or "the requested research topic").strip()
        if not source_lines and not summary:
            return ""
        parts = []
        opening = f"I reviewed {len(source_lines) or len(sources or [])} source"
        opening += "" if (len(source_lines) or len(sources or [])) == 1 else "s"
        opening += f" on {topic}."
        if source_titles:
            opening += " The strongest available signals came from " + ", ".join(source_titles[:3]) + "."
        parts.append(opening)
        if summary:
            parts.append(
                "Taken together, the reporting points to this: "
                + cls._end_on_a_sentence(summary)
            )
        if source_notes:
            parts.append(
                "The details I would preserve in the document are source-bounded, not guessed. "
                + " ".join(source_notes)
            )
        if source_lines and len(" ".join(parts)) < 650:
            parts.append(
                "The available evidence is not equally deep in every source, so I would not pretend "
                "the search produced more certainty than it did. I would treat repeated claims across "
                "the sources as the reliable core, keep isolated details attributed, and mark any thin "
                "or inaccessible material as a place where better reporting would be needed before "
                "making a stronger conclusion."
            )
        if cls._objective_requests_opinion(objective):
            # This is the DETERMINISTIC composer. It runs when authored
            # synthesis was suppressed — under memory pressure, the guard in
            # _allow_research_model_synthesis. So whatever it writes here, no
            # view was formed, and a paragraph opening "In my view" is a claim
            # to have formed one.
            #
            # Live 2026-07-30 00:33: asked to "write a synthesis with your own
            # opinion", the document Bryan received said "In my view, the
            # reliable path is to treat the articles as evidence to compare" —
            # generic method talk, identical for orcas and for anything else,
            # asserting an opinion nobody had. The 23:39 run, with the runtime
            # settled and the guard open, wrote a real one about orcas. The
            # capability is there; this line was covering for its absence.
            #
            # Saying so plainly costs a paragraph of polish and buys the thing
            # the whole document is for: what is in it is true.
            parts.append(
                "On my own opinion, which you asked for: I have not formed one here. "
                "This document is source-bounded extraction — I was not able to author a "
                "synthesis in my own words on this pass, so read the comparison above as "
                "evidence I gathered rather than as a view I hold. Ask me again and I will "
                "give you the opinion rather than a placeholder for it."
            )
        else:
            parts.append(
                "My concise synthesis is that the useful answer is not a loose headline recap; it is a "
                "comparison of what the sources actually support, which claims appear repeated across the "
                "evidence, and which details should stay attributed to a specific source."
            )
        return "\n\n".join(part for part in parts if part).strip()[:4000]

    @staticmethod
    def _allow_research_model_synthesis(
        context: dict[str, Any] | None,
        objective: str = "",
    ) -> bool:
        """Authoring a synthesis is the REQUEST, not a hidden extra allocation.

        The opt-in flag exists so background desktop work cannot quietly spend a
        second foreground model. Nothing on the live path ever set it, so every
        request to "read them and form your own opinion... write a synthesis in
        your own words" fell to the deterministic composer, which concatenates
        source snippets. What Bryan received was:

          "Taken together, the reporting points to this: <snippet> <snippet>"

        — no takeaway, nothing learned, no summary. Not because synthesis
        failed, but because it was never attempted.

        When the objective explicitly asks her to synthesize, summarize, or form
        an opinion in her own words, the model call IS the task and refusing it
        cannot satisfy the request. The router's admission controller remains
        responsible for load safety; this layer must not silently replace the
        requested authorship with a generic template.
        """
        from .desktop_task import (
            DesktopTaskSkill,
        )


        if DesktopTaskSkill._allow_desktop_task_model_synthesis(context):
            return True
        return DesktopTaskSkill._objective_requests_authored_synthesis(objective)

    @classmethod
    def _research_synthesis_satisfies_objective(
        cls,
        objective: str,
        synthesis: str,
    ) -> bool:
        body = " ".join(str(synthesis or "").split())
        if len(body) < 60 or cls._looks_like_dispatch_narration(body):
            return False
        if cls._looks_like_incomplete_document_body(body):
            return False
        if not cls._objective_requests_opinion(objective):
            return True
        lowered = body.casefold()
        if re.search(
            r"\b(?:i (?:have not|haven't|did not|didn't) form(?:ed)?|"
            r"no opinion|cannot offer (?:an|my) opinion|ask me again)\b",
            lowered,
        ):
            return False
        return bool(
            re.search(
                r"\b(?:in my view|in my opinion|my (?:view|opinion|take|assessment)|"
                r"i (?:think|believe|find|conclude|would argue|would favor|would favour))\b",
                lowered,
            )
        )

    @staticmethod
    def _clip_to_sentence(text: str, limit: int) -> str:
        """Cut at a sentence, then a word — never mid-word, never mid-clause.

        Live 2026-07-30 00:33, in the PDF that landed in Bryan's Documents:
        "They are apex predators and one of the world's most widely distributed
        animals" — a source snippet cut at a character count, so the document he
        was about to show someone stopped in the middle of a clause. Character
        limits are the right idea and the wrong unit; prose has boundaries and
        they are cheap to find.
        """
        body = " ".join(str(text or "").split())
        if len(body) <= limit:
            return body
        window = body[:limit]
        for terminator in (". ", "! ", "? "):
            cut = window.rfind(terminator)
            if cut >= limit // 2:
                return window[: cut + 1].strip()
        cut = window.rfind(" ")
        kept = window[:cut] if cut >= limit // 2 else window
        return kept.rstrip(" ,;:-—") + "…"

    @staticmethod
    def _end_on_a_sentence(text: str) -> str:
        """Drop a trailing half-sentence that arrived already truncated.

        The upstream summary is clipped before it reaches the composer, so the
        composer receives the broken tail rather than making it. Trimming back
        to the last complete sentence is only worth it when a sentence actually
        survives — otherwise the mark stays, because silently dropping the only
        content there is would be worse than showing it was cut.
        """
        body = " ".join(str(text or "").split())
        if not body or body[-1] in ".!?…":
            return body
        cut = max(body.rfind(". "), body.rfind("! "), body.rfind("? "))
        if cut > 0 and (cut + 1) >= len(body) // 2:
            return body[: cut + 1].strip()
        return body.rstrip(" ,;:-—") + "…"

    async def _collect_research_context(
        self,
        *,
        capability_engine: Any,
        objective: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        from .desktop_task import (
            _MEMORY_SAFE_SOURCE_CEILING,
        )

        if not self._objective_requests_research_document(objective):
            return {}
        research_started = time.perf_counter()
        research_timing_ms: dict[str, float] = {}
        query = self._extract_search_query(objective)
        if not query:
            return {}
        deep_search = True
        # THE REQUEST DECIDES. Nothing else does.
        #
        # This was a flat 5: asked for "3 recent articles about orcas" she
        # fetched and read five, and reading is the entire cost of the step
        # (gathering logs as 0.0s). Two of those five were latency for
        # material nobody asked for, and the document cited more sources than
        # the request wanted.
        #
        # A "+1 spare for a dead link" lived here briefly and was the same
        # mistake one size smaller — a number with no one behind it. If a
        # source fails to fetch, the validation below already says so
        # honestly rather than silently padding.
        #
        # When the request names no count, nothing here invents one: the key
        # is simply not sent, so web_search's own documented default applies.
        # One default, in the schema where it is described, instead of five
        # scattered guesses.
        requested = self._requested_research_source_count(objective)
        num_results = requested
        search_query = query
        if self._objective_requests_recent_sources(objective) and not re.search(
            r"\b(?:recent|latest|current)\b", query, flags=re.IGNORECASE
        ):
            search_query = f"{query} recent articles"
        pressure_limited = False
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            pressure_limited = bool(
                getattr(snapshot, "warning", False)
                or getattr(snapshot, "refuse_heavy_local_generation", False)
            )
            if pressure_limited:
                deep_search = False
                num_results = (
                    min(num_results, _MEMORY_SAFE_SOURCE_CEILING)
                    if num_results
                    else _MEMORY_SAFE_SOURCE_CEILING
                )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="using shallow desktop research because memory safety probe failed",
                severity="warning",
            )
            deep_search = False
            num_results = (
                min(num_results, _MEMORY_SAFE_SOURCE_CEILING)
                if num_results
                else _MEMORY_SAFE_SOURCE_CEILING
            )
            pressure_limited = True
        step_context = self._child_step_context(context)
        step_context.update(
            {
                "origin": step_context.get("origin") or "desktop_task",
                "route": "desktop_task.web_search",
                "objective": objective,
                "foreground_request": False,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                # This layer needs fetched evidence.  It owns the authored
                # synthesis and semantic completion check below, so the search
                # subsystem must not allocate another model first.
                "evidence_only": True,
                "desktop_task_reason": "Collect live research evidence before composing the requested document.",
                "desktop_task_expect": "Web search returns sources or an explicit failure.",
            }
        )
        self._emit_progress(
            index=1,
            total=3,
            action="research",
            state="searching",
            detail=f"Gathering and reading source evidence for {query[:120]}.",
        )
        search_started = time.perf_counter()
        try:
            result = await capability_engine.execute(
                "web_search",
                {
                    "query": search_query,
                    # Present only when the request asked for a number.
                    **({"num_results": num_results} if num_results else {}),
                    # Deep article fetches are useful, but they are no longer
                    # allowed to run before memory admission. Under pressure we
                    # use snippets and fewer sources instead of risking a live
                    # desktop RAM spike.
                    "deep": deep_search,
                    "retain": False,
                    "force_refresh": True,
                },
                context=step_context,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            research_timing_ms["search"] = round(
                (time.perf_counter() - search_started) * 1000.0,
                1,
            )
            research_timing_ms["total"] = round(
                (time.perf_counter() - research_started) * 1000.0,
                1,
            )
            record_degradation(
                "desktop_task",
                exc,
                action="continued desktop document task without pre-document research evidence",
                severity="warning",
            )
            return {
                "desktop_task_research_query": query,
                "desktop_task_research_error": str(exc),
                "desktop_task_research_timing_ms": research_timing_ms,
            }
        research_timing_ms["search"] = round(
            (time.perf_counter() - search_started) * 1000.0,
            1,
        )
        pipeline_timing_ms = (
            dict(result.get("timing_ms") or {})
            if isinstance(result, dict) and isinstance(result.get("timing_ms"), dict)
            else {}
        )
        if not isinstance(result, dict):
            result = {"ok": bool(result), "result": result}
        if not bool(result.get("ok", True)):
            return {
                "desktop_task_research_query": query,
                "desktop_task_research_error": str(result.get("error") or result.get("status") or result),
                "desktop_task_research_deep": deep_search,
                "desktop_task_research_pressure_limited": pressure_limited,
            }
        candidate_sources = self._research_sources_from_result(result)
        requested_sources = self._requested_research_source_count(objective)
        required_sources = max(1, requested_sources)
        require_recent = self._objective_requests_recent_sources(objective)
        require_read = self._objective_requests_source_reading(objective)
        sources = self._usable_research_sources(
            candidate_sources,
            require_recent=require_recent,
            require_read=require_read,
        )
        if len(sources) < required_sources:
            missing = required_sources - len(sources)
            replacement_query = (
                f"{query} latest independent reporting"
                if self._objective_requests_recent_sources(objective)
                else f"{query} additional independent sources"
            )
            replacement_started = time.perf_counter()
            try:
                replacement = await capability_engine.execute(
                    "web_search",
                    {
                        "query": replacement_query,
                        "num_results": min(8, max(3, missing * 2)),
                        "deep": deep_search,
                        "retain": False,
                        "force_refresh": True,
                    },
                    context={
                        **step_context,
                        "route": "desktop_task.web_search.replacement",
                        "desktop_task_reason": (
                            "Replace filtered, duplicate, or inaccessible sources so the "
                            "requested evidence count is actually satisfied."
                        ),
                    },
                )
                if isinstance(replacement, dict) and bool(replacement.get("ok", True)):
                    candidate_sources = self._merge_research_sources(
                        candidate_sources,
                        self._research_sources_from_result(replacement),
                    )
                    sources = self._usable_research_sources(
                        candidate_sources,
                        require_recent=require_recent,
                        require_read=require_read,
                    )
            except (
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
                OSError,
                TimeoutError,
            ) as exc:
                record_degradation(
                    "desktop_task",
                    exc,
                    action="reported exact research-source shortfall after replacement search failed",
                    severity="warning",
                )
            research_timing_ms["replacement_search"] = round(
                (time.perf_counter() - replacement_started) * 1000.0,
                1,
            )
        if len(sources) < required_sources:
            research_timing_ms["total"] = round(
                (time.perf_counter() - research_started) * 1000.0,
                1,
            )
            return {
                "desktop_task_research_query": query,
                "desktop_task_research_error": (
                    f"research verified {len(sources)}"
                    f"{' readable' if require_read else ''}"
                    f"{' recent' if require_recent else ''} source(s), "
                    f"but the objective requires {required_sources}"
                ),
                "desktop_task_research_deep": deep_search,
                "desktop_task_research_pressure_limited": pressure_limited,
                "desktop_task_research_sources": candidate_sources,
                "desktop_task_research_timing_ms": research_timing_ms,
            }
        if requested_sources:
            sources = sources[:requested_sources]
        summary = str(
            result.get("summary")
            or result.get("answer")
            or result.get("message")
            or result.get("content")
            or result.get("result")
            or ""
        ).strip()
        if not summary and sources:
            summary = "Key source notes:\n" + "\n".join(
                f"- {item.get('title') or item.get('url')}: {item.get('snippet')}"
                for item in sources[:3]
            )
        research_ctx = {
            "desktop_task_research_query": query,
            "desktop_task_research_summary": summary[:3000],
            "desktop_task_research_sources": sources,
            "desktop_task_research_deep": deep_search,
            "desktop_task_research_pressure_limited": pressure_limited,
            "desktop_task_research_timing_ms": research_timing_ms,
            "desktop_task_research_pipeline_timing_ms": pipeline_timing_ms,
        }
        synthesis = self._compose_research_synthesis_from_sources(
            objective=objective,
            query=query,
            summary=summary,
            sources=sources,
        )
        # Optional model synthesis is an explicitly enabled enhancement, not a
        # hidden second foreground allocation during visible desktop work.
        if self._allow_research_model_synthesis(context, objective):
            self._emit_progress(
                index=2,
                total=3,
                action="research",
                state="synthesizing",
                detail=(
                    f"Composing the requested document from {len(sources)} verified "
                    "source records."
                ),
            )
            synthesis_started = time.perf_counter()
            model_synthesis = ""
            repair_feedback = ""
            for _synthesis_attempt in range(2):
                model_synthesis = await self._synthesize_research_document(
                    objective=objective,
                    query=query,
                    summary=summary,
                    sources=sources,
                    repair_feedback=repair_feedback,
                )
                if self._research_synthesis_satisfies_objective(
                    objective,
                    model_synthesis,
                ):
                    break
                repair_feedback = (
                    "The previous draft failed the semantic completion check: it was "
                    "missing a complete cross-source synthesis or the requested "
                    "independent first-person position. Rewrite the document itself; "
                    "do not discuss the failure or promise a later answer."
                )
            research_timing_ms["synthesis"] = round(
                (time.perf_counter() - synthesis_started) * 1000.0,
                1,
            )
            if self._research_synthesis_satisfies_objective(
                objective,
                model_synthesis,
            ):
                synthesis = model_synthesis
                research_ctx["desktop_task_research_authored"] = True
                research_ctx["desktop_task_research_synthesis_sha256"] = text_sha256(
                    synthesis
                )
                research_ctx[
                    "desktop_task_research_synthesis_source_sha256s"
                ] = [
                    str(item.get("source_evidence_sha256") or "")
                    for item in sources
                    if self._valid_sha256(
                        str(item.get("source_evidence_sha256") or "")
                    )
                ]
            elif self._objective_requests_authored_synthesis(objective):
                research_timing_ms["total"] = round(
                    (time.perf_counter() - research_started) * 1000.0,
                    1,
                )
                return {
                    **research_ctx,
                    "desktop_task_research_error": (
                        "the requested authored synthesis did not satisfy its content contract"
                    ),
                }
        if synthesis:
            research_ctx["desktop_task_research_synthesis"] = synthesis
        # Learn from what she just read and wrote: persist the finding as an
        # episode so it consolidates into memory (and the engram/reconsolidation
        # dynamics) rather than being forgotten the moment the document is saved.
        await self._remember_research(query, synthesis or summary, sources)
        research_timing_ms["total"] = round(
            (time.perf_counter() - research_started) * 1000.0,
            1,
        )
        self._emit_progress(
            index=3,
            total=3,
            action="research",
            state="ready",
            detail=(
                f"Research and document content are ready from {len(sources)} sources "
                f"in {research_timing_ms['total'] / 1000.0:.1f}s."
            ),
        )
        return research_ctx

    async def _remember_research(
        self, query: str, finding: str, sources: list[dict[str, str]]
    ) -> None:
        """Best-effort: encode a research finding into episodic memory so Aura
        retains what she learned from reading and writing."""
        finding = str(finding or "").strip()
        if not query or not finding:
            return
        try:
            from core.container import ServiceContainer

            episodic = ServiceContainer.get("episodic_memory", default=None)
            recorder = getattr(episodic, "record_episode_async", None) if episodic else None
            if not callable(recorder):
                return
            top = [
                str(s.get("url") or s.get("title") or "")
                for s in (sources or [])[:3]
                if isinstance(s, dict)
            ]
            await recorder(
                context=f"Researched and wrote about: {query}",
                action=f"Read {len(sources or [])} sources and composed a summary",
                outcome=finding[:800],
                success=True,
                importance=0.62,
                lessons=[f"Source: {u}" for u in top if u],
                source="desktop_task_research",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="continued after research-learning episode record failed",
                severity="warning",
            )

    async def _synthesize_research_document(
        self,
        *,
        objective: str,
        query: str,
        summary: str,
        sources: list[dict[str, str]],
        repair_feedback: str = "",
    ) -> str:
        """Compose a first-person summary (and opinion, when asked) of the
        research through the canonical model router. Bounded and best-effort:
        if the router is unavailable the raw research section still stands."""
        from core.container import ServiceContainer

        from .desktop_task import (
            logger,
        )

        router = ServiceContainer.get("llm_router", default=None)
        generate = getattr(router, "generate", None) if router is not None else None
        if not callable(generate):
            return ""
        def _src_tag(item: dict[str, Any]) -> str:
            rep = int(item.get("reputability", 1) or 0)
            label = {3: "high-authority", 2: "reputable", 1: "general", 0: "low-quality"}.get(rep, "general")
            if not item.get("accessible", True):
                label += ", limited/blocked access"
            return label

        source_lines = "\n".join(
            f"- [{_src_tag(item)}] {str(item.get('title') or item.get('url') or 'source')} "
            f"({str(item.get('url') or '')}):\n  "
            f"{str(item.get('article_body') or item.get('snippet') or '')[:1200]}"
            for item in (sources or [])[:5]
            if isinstance(item, dict)
        )
        wants_opinion = self._objective_requests_opinion(objective)
        opinion_clause = (
            " Close with a separate short first-person opinion paragraph beginning "
            "\"In my view,\" giving your own first-person take on what these sources "
            "say and what you make of them."
            if wants_opinion
            else ""
        )
        prompt = (
            f'You researched "{query}" and gathered these sources (titles, URLs, and '
            f"article text):\n{summary[:3500]}\n{source_lines}\n\n"
            "Write a thorough, well-organized composite document for a reader who wants "
            "to actually understand the topic — not a thin gloss. Synthesize ACROSS the "
            "sources rather than listing them one by one: open with the core facts, then "
            "develop the important context, specifics (names, numbers, dates, quotes "
            "where useful), and any points where the sources differ or add nuance. Use "
            "several paragraphs and scale the depth to the material — be as complete and "
            "substantive as the sources support. If the sources are genuinely thin or "
            "conflicting, say so honestly and note what would need further research "
            "rather than padding.\n"
            "Weigh your sources critically. Give more trust to reputable, authoritative "
            "sources — peer-reviewed research, .edu/.gov, established institutions and "
            "labs, and named expert authors with relevant credentials — and to claims "
            "that several independent reputable sources corroborate. Treat single-source, "
            "anonymous, overtly promotional, or paywalled/ad-wall pages with appropriate "
            "skepticism and flag that uncertainty. If a source was inaccessible, blocked, "
            "or clearly content-thin, do not rely on it, and note where a better source "
            "would be needed."
            f"{opinion_clause}\n"
            "Write in the first person as Aura, in clean prose. Do not mention tools, "
            "steps, dispatch, commitments, or that you are executing a task — this is the "
            "finished document the reader will see, not a status update."
            + (f"\n\nREVISION REQUIREMENT: {repair_feedback}" if repair_feedback else "")
        )
        try:
            text = await asyncio.wait_for(
                generate(
                    prompt=prompt,
                    timeout=110.0,
                    temperature=0.6,
                    # SCALED TO THE SOURCES, not a flat ceiling.
                    #
                    # 1100 was too small — the live synthesis was cut
                    # mid-clause, ending "...The increase is a" before the
                    # Sources list. 2048 for everything is the opposite error:
                    # once the fetch stopped over-reading, a three-source
                    # request still produced a four-kilobyte document, and
                    # since the whole cost of this step is the local model
                    # writing, the time saved by reading less went straight
                    # back into writing more. Measured: research 82.4s -> 27.8s,
                    # total unchanged at ~100s.
                    #
                    # The floor keeps the mid-clause failure from returning.
                    max_tokens=max(1100, min(2048, 350 + 380 * max(1, len(sources)))),
                    # Pin synthesis to the on-device Cortex: it has no external
                    # quota, so the document never degrades to a thin heuristic
                    # fallback because a cloud tier returned 429 RESOURCE_EXHAUSTED.
                    prefer_tier="local",
                    # Her own authoring, not the surface the person typed at.
                    #
                    # "desktop_task" begins with an allowlisted user-facing
                    # label, so anything starting "desktop_" inherits the
                    # protected reply lane — and with it the apology written
                    # for a person: "I can't work through that technical
                    # request right now, my language backend is temporarily
                    # unavailable", handed back as the body of a document.
                    # The tool dispatch IS user-facing and keeps that origin;
                    # this sub-call never was.
                    origin="internal_desktop_authoring",
                    purpose="research_document_synthesis",
                        # Foreground work, and not the reply lane. Same seam and
                        # same reason as the artifact writer above.
                        _non_chat_inference=True,
                    # Said again where the gate reads it. The router pops
                    # the flag above and re-adds it under another name, and
                    # the gate's own check looks at the context it was
                    # handed — which is how an apology written for a person
                    # ("my language backend is temporarily unavailable")
                    # arrived as the body of a document. The gate already
                    # returns nothing to an internal caller for exactly
                    # this reason; it could not tell this was one.
                    internal_inference=True,
                    # Somebody is waiting for this file.
                    #
                    # Her reasoning lane declares the same thing and reaches
                    # the resident worker; this one did not, and asked a lane
                    # that answered "worker_not_alive" for half a minute
                    # while the runtime's own health reported Cortex ready and
                    # generating. Internal is about whose question it is.
                    # Foreground is about whether anyone is waiting, and both
                    # are true here.
                    foreground_request=True,
                ),
                timeout=120.0,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="composed research document from raw search section after synthesis was unavailable",
                severity="warning",
            )
            return ""
        text = str(text or "").strip()
        # The router guarantees non-empty (diagnostic fallback); a synthesis
        # that is a degraded diagnostic line or dispatch narration is not
        # document content, so fall back to the raw research section.
        if not text or self._looks_like_dispatch_narration(text):
            return ""
        if re.search(r"\b(?:diagnostic|fallback|unavailable|all (?:remote )?endpoints? failed)\b", text.lower()) and len(text) < 200:
            return ""
        # Never hand back a document that stops mid-clause. Whether the budget
        # ran out or the 4000-char clamp lands mid-word, the reader gets a
        # finished paragraph rather than "...The increase is a" followed by the
        # Sources list.
        text = text[:4000]
        try:
            from core.conversation.response_reliability import complete_truncated_tail

            completed = complete_truncated_tail(text)
            if completed and len(completed) >= len(text) * 0.5:
                text = completed
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as why:
            logger.info("kept the text as it was: %s", why)
        return text

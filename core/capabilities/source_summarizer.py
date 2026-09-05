"""core/capabilities/source_summarizer.py — Multi-Source Content Summarizer
============================================================================
Synthesizes multiple web articles, documents, or text sources into a
coherent summary with proper citation.

Used by the TaskDecomposer when missions involve research tasks.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.SourceSummarizer")


@dataclass
class SourceEntry:
    """A single source used in summarization."""
    url: str = ""
    title: str = ""
    body: str = ""
    author: str = ""
    domain: str = ""
    word_count: int = 0


@dataclass
class SummarizationResult:
    """Result of a multi-source summarization."""
    summary: str
    sources: List[Dict[str, str]]
    word_count: int = 0
    generated_at: float = field(default_factory=time.time)
    model_used: str = ""


class SourceSummarizer:
    """Synthesizes multiple sources into a coherent summary.

    Usage:
        summarizer = get_source_summarizer()
        result = await summarizer.summarize_sources(
            sources=[...],
            objective="Write a research summary about ocean conservation",
        )
    """

    SUMMARIZE_PROMPT = """You are summarizing multiple sources for a user.

OBJECTIVE: {objective}

SOURCES:
{sources_text}

Write a coherent, well-structured summary that:
1. Synthesizes information across all sources
2. Cites sources as [1], [2], etc.
3. Highlights key findings and insights
4. Notes any conflicting information
5. Is approximately {target_words} words long

Do NOT just list facts from each source separately. Weave them into a narrative."""

    def __init__(self) -> None:
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        register_runtime_service(
            "source_summarizer",
            self,
            required=False,
            owner="core/capabilities/source_summarizer.py",
            registered_by="SourceSummarizer.start",
        )
        self._started = True
        logger.info("SourceSummarizer ONLINE")

    async def summarize_sources(
        self,
        sources: List[SourceEntry],
        objective: str = "Summarize the following sources",
        target_words: int = 500,
    ) -> SummarizationResult:
        """Summarize multiple sources into a coherent document."""

        # Build sources text
        sources_text_parts = []
        source_refs = []
        for i, src in enumerate(sources, 1):
            body_preview = src.body[:2000] if src.body else "[No content]"
            sources_text_parts.append(
                f"[{i}] {src.title or 'Untitled'}\n"
                f"    Source: {src.url or src.domain or 'Unknown'}\n"
                f"    Content: {body_preview}\n"
            )
            source_refs.append({
                "index": i,
                "title": src.title or "Untitled",
                "url": src.url or "",
                "domain": src.domain or "",
            })

        sources_text = "\n".join(sources_text_parts)

        # Try LLM summarization
        summary = await self._llm_summarize(objective, sources_text, target_words)

        if not summary:
            # Fallback: extractive summary
            summary = self._extractive_summary(sources, objective, target_words)

        return SummarizationResult(
            summary=summary,
            sources=source_refs,
            word_count=len(summary.split()),
        )

    async def _llm_summarize(
        self, objective: str, sources_text: str, target_words: int
    ) -> str:
        """Use LLM for abstractive summarization."""
        try:
            router = get_runtime_service("llm_router", default=None)
            if not router:
                return ""

            prompt = self.SUMMARIZE_PROMPT.format(
                objective=objective,
                sources_text=sources_text[:8000],
                target_words=target_words,
            )

            response = await router.route(
                prompt=prompt,
                temperature=0.5,
                max_tokens=target_words * 2,
                route_hint="summarization",
            )

            if response and hasattr(response, "text"):
                return response.text
            return str(response) if response else ""

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("summarizer.llm", e)
            return ""

    def _extractive_summary(
        self, sources: List[SourceEntry], objective: str, target_words: int
    ) -> str:
        """Fallback extractive summarization when LLM unavailable."""
        parts = [f"# {objective}\n"]

        for i, src in enumerate(sources, 1):
            parts.append(f"\n## [{i}] {src.title or 'Source ' + str(i)}\n")
            if src.url:
                parts.append(f"*Source: {src.url}*\n")

            # Extract first few paragraphs
            if src.body:
                paragraphs = src.body.split("\n\n")
                words_used = 0
                for para in paragraphs:
                    para = para.strip()
                    if not para or len(para) < 20:
                        continue
                    para_words = len(para.split())
                    if words_used + para_words > target_words // len(sources):
                        break
                    parts.append(para + "\n")
                    words_used += para_words

        summary = "\n".join(parts)
        return summary

    async def summarize_urls(
        self,
        urls: List[str],
        objective: str = "Summarize",
        target_words: int = 500,
    ) -> SummarizationResult:
        """Fetch and summarize multiple URLs."""
        sources = []
        try:
            browser = get_runtime_service("browser_controller", default=None)
            if browser:
                for url in urls[:5]:
                    extract = await browser.extract_article_text(url)
                    sources.append(SourceEntry(
                        url=extract.url,
                        title=extract.title,
                        body=extract.body,
                        author=extract.author,
                        domain=extract.source_domain,
                        word_count=extract.word_count,
                    ))
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("summarizer.fetch", e)

        if not sources:
            return SummarizationResult(
                summary="Could not fetch any sources.",
                sources=[{"url": u} for u in urls],
            )

        return await self.summarize_sources(sources, objective, target_words)

    def get_status(self) -> Dict[str, Any]:
        return {"started": self._started}


_instance: Optional[SourceSummarizer] = None


def get_source_summarizer() -> SourceSummarizer:
    global _instance
    if _instance is None:
        _instance = SourceSummarizer()
    return _instance


__all__ = ["SourceSummarizer", "SourceEntry", "SummarizationResult", "get_source_summarizer"]

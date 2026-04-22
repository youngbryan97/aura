"""Enhanced web search and research skill for Aura."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from core.container import ServiceContainer
from core.search import ResearchSearchPipeline
from core.search.research_pipeline import query_requires_source_reading
from core.skills.base_skill import BaseSkill
from core.skills.deep_research import run_deep_research

logger = logging.getLogger("Skills.WebSearch")


class _DeepResearchBrainAdapter:
    """Compat adapter for deep_research's ``brain.generate() -> {'response': ...}`` contract."""

    def __init__(self, engine: Any):
        self.engine = engine

    async def generate(self, prompt: str, **kwargs) -> Dict[str, str]:
        raw = await self.engine.generate(
            prompt,
            origin="system",
            purpose="research",
            use_strategies=False,
            is_background=True,
        )
        if isinstance(raw, dict):
            text = raw.get("response") or raw.get("content") or raw.get("result") or ""
        else:
            text = str(raw or "")
        return {"response": str(text or "")}


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")
    deep: bool = Field(False, description="If True, fetch and synthesize multiple result pages.")
    num_results: int = Field(5, ge=1, le=20, description="Number of search hits to return.")
    retain: Optional[bool] = Field(
        None,
        description="Whether Aura should retain what she learned from this search.",
    )
    force_refresh: bool = Field(False, description="If True, bypass cache and force a new live search.")


class EnhancedWebSearchSkill(BaseSkill):
    """Hybrid live web search with retrieval, synthesis, and retention."""

    name = "web_search"
    description = (
        "Search the internet for current information, research a topic across multiple pages, "
        "synthesize an evidence-grounded answer, and retain what was learned when appropriate."
    )
    input_model = WebSearchInput
    timeout_seconds = 35.0
    metabolic_cost = 2

    def __init__(self):
        super().__init__()
        self.pipeline = ResearchSearchPipeline()
        self.browser = _StubBrowser()

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            query = params.get("query") or params.get("q", "")
            deep = bool(params.get("deep", False))
            num_results = int(params.get("num_results", 5))
            retain = params.get("retain")
            force_refresh = bool(params.get("force_refresh", False))
        elif isinstance(params, WebSearchInput):
            query = params.query
            deep = params.deep
            num_results = params.num_results
            retain = params.retain
            force_refresh = params.force_refresh
        else:
            query = str(params)
            deep = False
            num_results = 5
            retain = None
            force_refresh = False

        query = str(query or "").strip()
        if not query:
            return {"ok": False, "error": "No search query provided."}

        source_reading = query_requires_source_reading(query)
        effective_deep = bool(deep or source_reading)

        logger.info(
            "🔍 WebSearch: '%s' (deep=%s, effective_deep=%s, retain=%s, force_refresh=%s)",
            query[:80],
            deep,
            effective_deep,
            retain,
            force_refresh,
        )
        
        if deep and not source_reading:
            # v2.0: Deep Research LangGraph Pipeline implementation
            try:
                engine = (
                    ServiceContainer.get("cognitive_engine", default=None)
                    or ServiceContainer.get("brain", default=None)
                )
                if engine is None:
                    raise RuntimeError("No cognitive engine available for deep research")
                brain = _DeepResearchBrainAdapter(engine)
                
                # Adapting existing Search pipeline format to standard search_fn format
                async def _search_fn(q: str):
                    res = await self.pipeline.search(q, num_results=5, deep=False, force_refresh=force_refresh)
                    results = res.get("results", [])
                    # format sources
                    content = res.get("answer") or str([r.get("snippet", "") for r in results])
                    return {"ok": True, "content": content, "sources": results}
                
                res = await run_deep_research(query, brain, _search_fn)
                answer = str(res.get("answer") or "").strip()
                if answer:
                    res["summary"] = answer
                    res.setdefault("ok", True)
                    return res
                logger.warning("Deep Research returned an empty answer for '%s'; falling back to retrieval pipeline.", query)
            except Exception as e:
                logger.error("Deep Research failed, falling back to legacy: %s", e)

        # Legacy direct search
        result = await self.pipeline.search(
            query,
            num_results=num_results,
            deep=effective_deep,
            retain=retain,
            context=context or {},
            force_refresh=force_refresh,
        )
        if not result.get("ok") and force_refresh:
            logger.info(
                "WebSearch forced refresh failed for '%s'; retrying with retained-artifact fallback.",
                query[:80],
            )
            result = await self.pipeline.search(
                query,
                num_results=num_results,
                deep=effective_deep,
                retain=retain,
                context=context or {},
                force_refresh=False,
            )
        result.setdefault("summary", result.get("answer") or result.get("message") or "")
        return result

    async def on_stop_async(self):
        """Compatibility stub for legacy lifecycle hooks."""
        pass


class _StubBrowser:
    """Minimal browser stub kept for legacy tests that access skill.browser."""

    is_active = False

    async def ensure_ready(self):
        return None

    async def browse(self, url: str):
        return False

    async def click(self, text_match: str = "", selector: str = "") -> bool:
        return False

    async def close(self):
        return None

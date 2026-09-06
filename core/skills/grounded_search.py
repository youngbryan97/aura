"""Evidence-grounded web research synthesized inside Aura's local runtime."""

from __future__ import annotations

from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.skills.web_search import EnhancedWebSearchSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT


class _LocalOnlyRouter:
    """Force synthesis calls onto the host-local endpoint set."""

    def __init__(self, router: Any):
        self._router = router

    async def think(self, *args: Any, **kwargs: Any) -> Any:
        options = dict(kwargs)
        options["allow_cloud_fallback"] = False
        options["allow_auto_cloud_recovery"] = False
        options.pop("cloud_only", None)
        return await self._router.think(*args, **options)


class GroundedSearchSkill(EnhancedWebSearchSkill):
    """Live-source specialization of Aura's governed web-search pipeline.

    The base capability owns retrieval, source reading, citations, local-model
    synthesis, deterministic synthesis, retention, and offline corpus fallback.
    This skill keeps the historical ``grounded_search`` address while requiring
    a fresh, deep evidence pass by default.
    """
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT


    name = "grounded_search"
    description = (
        "Research the live web through Aura's governed retrieval pipeline, read source "
        "evidence, and synthesize a cited answer inside Aura's local runtime."
    )

    async def execute(
        self,
        goal: Any,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params = self._normalize_params(goal)
        query = str(params.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "No search query provided."}

        params["query"] = query
        deep = bool(params.get("deep", True))
        force_refresh = bool(params.get("force_refresh", True))
        retain = params.get("retain")
        try:
            num_results = max(1, min(int(params.get("num_results", 5)), 20))
        except (TypeError, ValueError):
            num_results = 5

        search_context = self._local_runtime_context(context)
        try:
            result = await self.pipeline.search(
                query,
                num_results=num_results,
                deep=deep,
                retain=retain,
                context=search_context,
                force_refresh=force_refresh,
            )
            if not result.get("ok") and force_refresh:
                result = await self.pipeline.search(
                    query,
                    num_results=num_results,
                    deep=deep,
                    retain=retain,
                    context=search_context,
                    force_refresh=False,
                )
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
            record_degradation(
                "grounded_search",
                exc,
                severity="warning",
                action="degraded governed web retrieval to the local corpus",
            )
            result = {"ok": False, "error": str(exc)[:240]}

        if not result.get("ok"):
            offline = self._local_corpus_fallback(query, num_results)
            if offline is not None:
                offline["web_error"] = str(
                    result.get("error") or result.get("message") or "web search failed"
                )[:240]
                result = offline
        if not isinstance(result, dict):
            return {
                "ok": False,
                "error": "Governed web search returned an invalid result payload.",
            }

        normalized = dict(result)
        normalized.setdefault(
            "summary",
            normalized.get("answer") or normalized.get("message") or "",
        )
        normalized.setdefault(
            "sources",
            normalized.get("citations")
            or normalized.get("chunks")
            or normalized.get("results")
            or [],
        )
        normalized["grounding"] = {
            "retrieval": "governed_web_search",
            "synthesis_boundary": "aura_local_runtime",
            "remote_model_used": False,
        }
        normalized.setdefault(
            "note",
            "Retrieved through Aura's governed search pipeline and synthesized locally.",
        )
        return normalized

    @staticmethod
    def _local_runtime_context(context: dict[str, Any] | None) -> dict[str, Any]:
        search_context = dict(context or {})
        search_context["allow_cloud_fallback"] = False
        search_context["allow_auto_cloud_recovery"] = False
        search_context["remote_model_allowed"] = False

        router = search_context.get("llm_router")
        if router is None:
            router = ServiceContainer.get("llm_router", default=None)
        if router is not None and hasattr(router, "think"):
            search_context["llm_router"] = _LocalOnlyRouter(router)
        return search_context

    @staticmethod
    def _normalize_params(goal: Any) -> dict[str, Any]:
        if not isinstance(goal, dict):
            return {"query": str(goal or "")}

        nested = goal.get("params")
        if isinstance(nested, dict):
            params = dict(nested)
            if not params.get("query"):
                params["query"] = goal.get("query") or goal.get("objective") or ""
            return params

        params = dict(goal)
        if not params.get("query"):
            params["query"] = params.get("objective") or ""
        return params

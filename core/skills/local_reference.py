"""Local reference search — the search tool that cannot lose the network.

First-class BaseSkill over the local knowledge corpus
(core/knowledge/local_corpus.py): 6.5M+ offline reference documents behind
FTS5/BM25. Exists so search intent NEVER dead-ends in a local-first
system — web_search can fail (offline, backend down, egress refused);
this lane answers in ~100ms from disk with explicit provenance.

Honesty contract: results carry title/source provenance and the corpus
snapshot is dated — the skill reports "not in the local corpus" rather
than inventing, and never claims freshness it does not have.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill


class LocalReferenceInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)


class LocalReferenceSearchSkill(BaseSkill):
    """Offline reference lookup over the local knowledge corpus."""
    #: This one answers with `success` rather than `ok`, and a list of hits.
    #: Written out rather than pointed at the shared contract, because
    #: declaring `ok` here would be declaring a field it does not return.
    result_schema = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "results": {"type": "array"},
            "message": {"type": "string"},
        },
        "required": ["success", "results"],
        "additionalProperties": True,
    }


    name = "local_reference_search"
    description = (
        "Search Aura's LOCAL offline knowledge corpus (a full Wikipedia "
        "snapshot, ~6.5M articles) for factual/background information. "
        "Instant, private, and works with no network. Use for general "
        "knowledge, definitions, science, history, and technical "
        "background. Results are from a dated snapshot — for breaking or "
        "current information use web_search instead."
    )
    input_model: type[BaseModel] | None = LocalReferenceInput
    timeout_seconds: float = 10.0
    metabolic_cost: int = 1  # Light: bounded disk reads, no model, no network
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = LocalReferenceInput(
                query=str(params.get("query") or params.get("q") or ""),
                limit=int(params.get("limit", 5) or 5),
            )
        from core.knowledge.local_corpus import get_local_corpus_store

        store = get_local_corpus_store()
        # The emptiness guard wants has_documents(), not a count of 7.19M rows.
        # document_count() is a full table scan and measured ~6s on this host —
        # paid on every lookup, before the search itself took 29-77ms.
        if not store.has_documents():
            return {
                "success": False,
                "results": [],
                "message": (
                    "Local knowledge corpus is empty — run "
                    "tools/knowledge_substrate/ingest_wikipedia.py to populate it."
                ),
                "provenance": "local_corpus",
            }

        hits = store.search(params.query, limit=params.limit)
        if not hits:
            return {
                "success": True,
                "results": [],
                "message": f"No local corpus match for: {params.query!r}",
                "provenance": "local_corpus",
            }
        return {
            "success": True,
            "results": [
                {
                    "title": hit.title,
                    "snippet": hit.snippet,
                    "source": hit.source,
                    "provenance": "local_corpus",
                }
                for hit in hits
            ],
            "count": len(hits),
            "provenance": "local_corpus",
        }

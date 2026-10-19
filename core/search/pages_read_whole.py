"""The pages a query finds, read whole and nothing synthesized from them.

The research pipeline (core/search/research_pipeline.py) reads pages to answer a question, and gives back the answer
and the passages that bear on it. A reader that wants the sources themselves, as a program's rebuild reads its own
manual and help pages (core/skills/program_dna_reconstruct.py), takes the pages here: the same search, the same
ranking, the same fetch, and every page's whole text.
"""
from __future__ import annotations

from typing import Any

__all__ = ["pages_about"]


async def pages_about(query: str, *, at_most: int = 4, deep: bool = False, pipeline: Any = None) -> list[Any]:
    """The pages ``query`` finds, best first, each read whole; [] where the query is empty or finds nothing."""
    from core.search.research_pipeline import ResearchSearchPipeline, _normalize_query

    pipeline = pipeline or ResearchSearchPipeline()
    cleaned = _normalize_query(query)
    if not cleaned:
        return []
    hits = pipeline._rerank_hits(cleaned, await pipeline._search_candidates([cleaned], num_results=max(at_most, 5)))
    return await pipeline._fetch_pages(hits[:at_most], deep=deep) if hits else []

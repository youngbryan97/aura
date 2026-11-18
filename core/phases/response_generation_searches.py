"""Searching each thing a request names, for UnitaryResponsePhase.

Its three methods, kept here so that class stays within its size (config/module_size_baseline.json counts methods per
class). What they use from the phase's own module (its logger, its recoverable errors, its degradation record) is read
from there when they run, so a test that replaces one there replaces it here too.
"""
from __future__ import annotations

import asyncio
import itertools
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["_SearchesEachThing"]


def _phase() -> Any:
    from core.phases import response_generation_unitary

    return response_generation_unitary


class _SearchesEachThing:
    """UnitaryResponsePhase's search of each thing a request names."""

    @classmethod
    def _grounded_search_queries(cls, objective: str, contract: Any | None = None) -> list[str]:
        """What to search for: the language substrate's reading of the request, one search for each sentence of it
        that names something (core/conversation/what_the_world_says.searches_for); the patterns above only where
        the substrate does not read a request to find something out.

        LIVE 17 November 2026 "You should be able to google them. All flights were grounded on Friday. Major banks
        were having outages yesterday. Apple outages today. NSA outage today" was searched as one query of all of
        it, three times; and the first pattern above reads "google it. Bam had an 83 point game" as a search for
        "it. Bam had an 83 point game", the defect of 3 October.
        """
        query = cls._extract_grounded_search_query(objective, contract)
        if query.startswith(("http://", "https://")):
            return [query]
        try:
            from core.conversation.what_the_world_says import searches_for
            from core.language.search_request import read_search_request

            asked = read_search_request(objective)
            if asked.about_the_world and asked.query:
                return searches_for(asked.query)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            _phase().logger.debug("the substrate's reading of the search was not available (%s); the patterns stand", exc)
        return [query] if query else []

    @staticmethod
    def _merged_searches(results: Sequence[dict[str, Any]], queries: Sequence[str]) -> dict[str, Any]:
        """Several searches' results as one payload: each list taken a turn at a time, so every search's best comes
        before any search's second; each text kept, one after another; anything else from the first."""
        merged: dict[str, Any] = dict(results[0])
        for key in {k for result in results for k in result}:
            values = [result.get(key) for result in results if result.get(key)]
            if values and all(isinstance(value, (list, tuple)) for value in values):
                merged[key] = [item for row in itertools.zip_longest(*values) for item in row if item is not None]
            elif values and all(isinstance(value, str) for value in values) and key != "query":
                merged[key] = "\n\n".join(dict.fromkeys(values))
        merged["queries"] = list(queries)
        merged["ok"] = True
        return merged

    @classmethod
    async def _search_each(cls, orchestrator: Any, queries: Sequence[str], args: Mapping[str, Any], *,
                           origin: str) -> dict[str, Any]:
        """web_search once for each query, together, each within the one search's 45 s."""
        unitary = _phase()

        async def one(query: str) -> dict[str, Any] | None:
            try:
                result = await asyncio.wait_for(
                    orchestrator.execute_tool("web_search", {**args, "query": query}, origin=origin), timeout=45.0)
            except TimeoutError:
                unitary.logger.warning("UnitaryResponse: web_search timed out after 45s for query: %s", query[:80])
                return None
            except unitary._RESPONSE_RECOVERABLE_ERRORS as exc:
                unitary._record_response_degradation(
                    exc, "UnitaryResponse: one of several searches failed: %s",
                    action="kept the other searches' results", severity="warning")
                return None
            return result if isinstance(result, dict) and result.get("ok") else None

        found = [result for result in await asyncio.gather(*(one(query) for query in queries)) if result]
        return cls._merged_searches(found, queries) if found else {"ok": False}

"""A turn about the world as it is now searches this week's pages (core/search/recent_pages.py).

LIVE 17 November 2026, 10:45, on 3a0ab997e: asked "What do you make of the current Israel/Palestine situation and
America's role in it? ... what do you think of America's role in the conflict with Iran?", she looked it up and read
an explainer and an opinion piece from earlier in the year. Bryan wanted her informed of the situation as it is now.
"""
from __future__ import annotations

import pytest

from core.search import research_pipeline
from core.search.recent_pages import ddg_within, ddgs_timelimit, recency_key

pytestmark = pytest.mark.unit


def test_the_engines_filter_covers_the_days_asked_for():
    assert ddg_within(0) == ""
    assert ddg_within(1) == "&df=d"
    assert ddg_within(7) == "&df=w"
    assert ddg_within(30) == "&df=m"
    assert ddgs_timelimit(7) == {"timelimit": "w"} and ddgs_timelimit(0) == {}


def test_a_recent_search_is_cached_apart_from_the_same_words_unfiltered():
    assert recency_key("israel gaza", 7) != recency_key("israel gaza", 0) == "israel gaza"


def test_the_html_lane_asks_duckduckgo_for_this_weeks_pages(monkeypatch):
    asked = []

    class _Gateway:
        def request(self, method, url, **_kwargs):
            asked.append(url)
            return {"ok": False, "error": "offline in a test"}

    monkeypatch.setattr(research_pipeline, "get_network_gateway", lambda: _Gateway())
    pipeline = research_pipeline.ResearchSearchPipeline()
    pipeline._legacy_html_search("israel gaza ceasefire", 5, 7)
    ddg = [url for url in asked if "duckduckgo" in url]
    assert ddg and all(url.endswith("&df=w") for url in ddg)
    asked.clear()
    pipeline._legacy_html_search("how photosynthesis works", 5)
    assert not any("&df=" in url for url in asked)


@pytest.mark.asyncio
async def test_the_search_hands_the_turns_recency_to_the_candidate_search(monkeypatch):
    seen = {}
    pipeline = research_pipeline.ResearchSearchPipeline()

    async def candidates(queries, *, num_results, recent_days=0):
        seen["recent_days"] = recent_days
        return []

    async def expand(query, ctx):
        return [query]

    monkeypatch.setattr(pipeline, "_search_candidates", candidates)
    monkeypatch.setattr(pipeline, "_expand_queries", expand)
    await pipeline.search("what is happening in gaza now", context={"recent_days": 7, "evidence_only": True},
                          force_refresh=True)
    assert seen["recent_days"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("asked, recent", [("Who won the election yesterday?", 7), ("How does photosynthesis work?", 0)])
async def test_the_turns_reading_of_the_world_asks_for_this_weeks_pages_when_it_is_about_now(monkeypatch, asked, recent):
    from interface.routes import chat_capability_inventory, chat_desktop_evidence
    from core.conversation import what_the_world_says

    sent = []

    async def skill(name, params, *, objective, extra_context=None, **_kwargs):
        sent.append(dict(extra_context or {}))
        return {"ok": True, "results": []}

    async def gather(request, *, search=None, **_kwargs):
        if search is not None:
            await search(request)
        return what_the_world_says.WorldEvidence()

    monkeypatch.setattr(chat_capability_inventory, "_execute_governed_live_skill", skill)
    monkeypatch.setattr(what_the_world_says, "gather_world_evidence", gather)
    monkeypatch.setattr(chat_desktop_evidence, "_about_the_world_now", lambda message, seeks: recent > 0)
    await chat_desktop_evidence._what_the_world_says_for_the_turn(asked, asked, session_id="")
    assert sent and sent[0]["recent_days"] == recent

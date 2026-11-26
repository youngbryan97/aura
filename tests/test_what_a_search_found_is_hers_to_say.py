"""What a search found is evidence for her model to answer from, never her reply.

LIVE 17 November 2026, 17:46: asked "You should be able to google them. All flights were grounded on Friday. ...", she
searched and read, and the runtime served the search skill's own report, cut at 420 characters mid-sentence, behind
"I searched it live." Her model never read what was found.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.phases.response_generation_unitary import UnitaryResponsePhase

ASKED = ("You should be able to google them. All flights were grounded on Friday. Major banks were having outages "
         "yesterday. Apple outages today. NSA outage today")
FOUND = {
    "ok": True,
    "answer": "# Major US Infrastructure Outages: November 13-17, 2026 A cluster of significant outages ...",
    "results": [{"title": "Why everything is breaking", "url": "https://news.example/outages",
                 "snippet": "A run of outages hit airlines, banks and Apple this week."}],
}


def test_a_search_answer_is_not_served_as_her_reply():
    assert UnitaryResponsePhase._format_grounded_search_reply(ASKED, FOUND) == ""


def test_the_one_datum_asked_for_is_still_given_as_it_is():
    asked = "Search the web for the official Python documentation homepage and tell me only the page title."
    assert UnitaryResponsePhase._format_grounded_search_reply(asked, FOUND) == "Why everything is breaking"


def test_a_search_line_in_her_working_memory_is_not_her_reply():
    line = {"role": "system", "content": "[SKILL RESULT: web_search] Outages hit airlines and banks.",
            "metadata": {"type": "skill_result", "skill": "web_search", "ok": True}}
    state = SimpleNamespace(cognition=SimpleNamespace(working_memory=[line]), response_modifiers={})
    contract = SimpleNamespace(requires_search=True)
    assert UnitaryResponsePhase._build_cached_deterministic_tool_reply(state, ASKED, contract) == ""


def test_what_to_search_for_is_the_substrates_reading_one_search_a_named_sentence():
    assert UnitaryResponsePhase._grounded_search_queries("Google it. Bam had an 83 point game") == [
        "Bam had an 83 point game"]
    assert UnitaryResponsePhase._grounded_search_queries("Google it. Apple had an outage today. So did the NSA.") == [
        "Apple had an outage today", "So did the NSA"]


def test_several_searches_are_one_payload_with_every_searchs_best_first():
    first = {"ok": True, "results": [{"title": "a1"}, {"title": "a2"}], "summary": "about Apple", "query": "Apple"}
    second = {"ok": True, "results": [{"title": "n1"}], "summary": "about the NSA", "query": "NSA"}
    merged = UnitaryResponsePhase._merged_searches([first, second], ["Apple", "NSA"])
    assert [r["title"] for r in merged["results"]] == ["a1", "n1", "a2"]
    assert merged["summary"] == "about Apple\n\nabout the NSA"
    assert merged["queries"] == ["Apple", "NSA"] and merged["ok"]


def test_each_query_is_searched_and_a_failed_one_leaves_the_others():
    import asyncio

    asked: list[str] = []

    class Orchestrator:
        async def execute_tool(self, name, args, origin=""):
            asked.append(args["query"])
            if args["query"] == "NSA":
                raise RuntimeError("the network went away")
            return {"ok": True, "results": [{"title": args["query"]}]}

    merged = asyncio.run(UnitaryResponsePhase._search_each(Orchestrator(), ["Apple", "NSA"], {"deep": False},
                                                           origin="user"))
    assert sorted(asked) == ["Apple", "NSA"]
    assert [r["title"] for r in merged["results"]] == ["Apple"]

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
    line = {"role": "system", "content": "[SKILL RESULT: web_search] ✅ Outages hit airlines and banks.",
            "metadata": {"type": "skill_result", "skill": "web_search", "ok": True}}
    state = SimpleNamespace(cognition=SimpleNamespace(working_memory=[line]), response_modifiers={})
    contract = SimpleNamespace(requires_search=True)
    assert UnitaryResponsePhase._build_cached_deterministic_tool_reply(state, ASKED, contract) == ""

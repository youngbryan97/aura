"""Her goal passes can look up what they need to know, in her library and on the web (26 November 2026).

Bryan: "Does Aura ever consult either the cortex or brainstem's llms ... Or the internet? Or internal Wikipedia? Not
just on whole problem sets but individual pieces of code and problems and issues." Her goal passes read her source and
her runtime's faults and nothing else. Now her way through a goal takes a step in her reference library and one on the
web, after reading her source and before changing it; each comes back with where it came from, and a step her model
writes no call for is let go.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.agency import her_goal_at_work as goals
from core.executive.her_own_passes import OWN_GOAL_TOOLS
from core.skills import her_draft
from core.skills.catalog_policy import HER_GOAL_TOOLS

pytestmark = pytest.mark.unit


def test_her_way_consults_before_it_changes_anything():
    way = list(goals.HER_WAY)
    assert way.index("read_source") < way.index("look_up") < way.index("search_the_web") < way.index("edit_draft")
    assert {"look_up", "search_the_web"} <= OWN_GOAL_TOOLS == HER_GOAL_TOOLS
    assert {"look_up", "search_the_web"} <= set(her_draft.tool_definitions())


def test_a_look_up_gives_each_passage_with_where_it_comes_from(monkeypatch):
    from core.knowledge import local_corpus

    hit = SimpleNamespace(title="Exec (system call)", source="wikipedia", snippet="An empty argument is passed as is.")
    monkeypatch.setattr(local_corpus, "get_local_corpus_store",
                        lambda: SimpleNamespace(search=lambda query, limit: [hit] if "argv" in query else []))
    found = asyncio.run(her_draft.LookUpSkill().execute({"query": "argv empty argument exec"}, {}))
    assert found["ok"] and found["found"] == 1 and found["text"] == "Exec (system call) (wikipedia): An empty argument is passed as is."
    none = asyncio.run(her_draft.LookUpSkill().execute({"query": "tide clocks"}, {}))
    assert none["ok"] and none["found"] == 0 and none["summary"] == "nothing in her library on this"


def test_a_web_search_gives_each_result_with_its_address_and_says_when_it_found_nothing(monkeypatch):
    from core.skills import web_search

    class Web:
        async def execute(self, params, context):
            assert context["origin"] == "her_goal_background" and params["retain"] is False
            if "nothing" in params["query"]:
                return {"ok": False, "error": "offline"}
            return {"ok": True, "results": [{"title": "subprocess", "url": "https://docs.python.org/3/library/subprocess.html",
                                             "snippet": "args should be a sequence of program arguments"}]}
    monkeypatch.setattr(web_search, "EnhancedWebSearchSkill", Web)
    found = asyncio.run(her_draft.SearchTheWebSkill().execute({"query": "python subprocess empty argument"}, {}))
    assert found["ok"] and "<https://docs.python.org/3/library/subprocess.html>" in found["text"]
    missed = asyncio.run(her_draft.SearchTheWebSkill().execute({"query": "nothing at all"}, {}))
    assert not missed["ok"] and "offline" in missed["error"]

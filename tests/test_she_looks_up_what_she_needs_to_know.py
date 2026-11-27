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


def test_a_fault_she_made_again_is_looked_up_once_before_she_mends_it(monkeypatch):
    """What her library and the web say of the error goes on her goal, with where each came from, before any change is
    searched for; asked under her own goal work's origin, and once."""
    from core.agency import mending_a_fault_of_hers as mending
    from core.runtime import service_access

    asked = []

    class Engine:
        async def execute(self, tool, params, context):
            asked.append((tool, params["query"], context["origin"]))
            if tool == "look_up":
                return {"ok": True, "text": "Exec (system call) (wikipedia): an empty argument is passed as is."}
            return {"ok": False, "error": "offline"}
    saved = []

    async def save(record):
        saved.append(record)
    monkeypatch.setattr(service_access, "optional_service", lambda name, default=None: Engine())
    monkeypatch.setattr(goals, "save", save)
    record = goals.GoalRecord(goal="Mend x", slug="x")
    fault = SimpleNamespace(call=SimpleNamespace(error_type="builtins.ValueError",
                                                 error_message="argv entries must not be empty"))
    asyncio.run(mending._what_her_sources_say(record, fault))
    asyncio.run(mending._what_her_sources_say(record, fault))
    assert [tool for tool, _query, _origin in asked] == ["look_up", "search_the_web"], "once"
    assert all(query == "ValueError argv entries must not be empty" and origin == "her_goal_background"
               for _tool, query, origin in asked)
    (entry,) = record.said
    assert entry["consulted"] and "Exec (system call) (wikipedia)" in entry["words"]
    assert "The web, of ValueError argv entries must not be empty: nothing (offline)" in entry["words"]
    assert saved == [record]


def test_what_she_looks_up_is_what_her_model_reads_as_worth_it_from_the_function(monkeypatch):
    """LIVE 26 November 2026, searched by the error's words alone, her library answered an empty argument to a program
    with a passage on LL parsers."""
    from core.agency import mending_a_fault_of_hers as mending
    from core.runtime import service_access
    from core.runtime.a_call_that_failed import FailedCall
    from core.self_modification import where_a_fault_lies as lies

    asked, read = [], []

    class Engine:
        async def execute(self, tool, params, context):
            asked.append((tool, params["query"]))
            return {"ok": True, "text": f"{tool} found it"}

    class HerModel:
        async def generate(self, prompt, *, is_background, deadline_s):
            read.append(prompt)
            return lies.ToLookUp(searches=["osascript display notification empty subtitle", "  exec argv empty  "])

    async def save(record):
        return None
    monkeypatch.setattr(service_access, "optional_service", lambda name, default=None: Engine())
    monkeypatch.setattr(goals, "save", save)
    monkeypatch.setattr(lies, "her_model", lambda schema=None: HerModel())
    source = "def coerce(argv):\n    if any(not part for part in argv):\n        raise ValueError('argv entries must not be empty')\n"
    call = FailedCall(module="core.notes", qualname="coerce", file="core/notes.py", line=3,
                      error_type="builtins.ValueError", error_message="argv entries must not be empty",
                      keywords=(("argv", "['osascript', 'Stretch', '']"),))
    record = goals.GoalRecord(goal="Mend coerce", slug="mend-coerce")
    asyncio.run(mending._what_her_sources_say(record, SimpleNamespace(call=call), source))
    assert "raise ValueError" in read[0] and "Stretch" in read[0], "her model read the function and the call"
    assert asked == [("look_up", "osascript display notification empty subtitle"),
                     ("search_the_web", "osascript display notification empty subtitle"),
                     ("look_up", "exec argv empty"), ("search_the_web", "exec argv empty")]
    assert record.said[0]["searched"] == ["osascript display notification empty subtitle", "exec argv empty"]

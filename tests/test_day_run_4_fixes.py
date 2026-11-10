"""What the first hour of day run 4 (9 November 2026) found, measured where it happens.

Her goal pass's calls were refused with "signed_standing_authority_lease_missing": the capability engine keeps a call's
origin only when it is user-facing or autonomous, and her goal pass's was neither, so its calls ran as
"capability_engine", which no grant names. The grant made for them (aura.own-goal-work) was tested against the
standing-authority manager and never through the engine's own origin resolution, which is where it failed.

CognitiveIntegrationPhase wrote affect.valence and affect.arousal outside its contract: the sentiment organ, live
since 5 November, blends the person's tone into her affect.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.agency import her_goal_at_work as work
from core.capability_engine import CapabilityEngine
from core.executive.execution_policy import classify_execution_risk, resolve_execution_effect_scope
from core.executive.standing_authority import OWN_GOAL_GRANT
from core.runtime import cognitive_provenance as provenance
from core.state.aura_state import AuraState
from tests.test_standing_authority import _manager


def test_the_engine_runs_her_goal_pass_s_calls_under_her_goal_pass_s_origin():
    engine = CapabilityEngine()
    context = {"origin": work.ORIGIN, "her_goal": "g", "her_goal_workspace": "/tmp/x", "requested_by": "aura"}
    assert engine._resolve_execution_source(context) == work.ORIGIN


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "args"), [
    ("read_source", {"path": "core/memory/cold_store.py", "from_line": 1}),
    ("edit_draft", {"path": "core/notes.py", "old_text": "", "new_text": "x = 1\n"}),
])
async def test_the_origin_the_engine_resolves_takes_her_goal_lease(tool, args):
    origin = CapabilityEngine()._resolve_execution_source({"origin": work.ORIGIN})
    scope = resolve_execution_effect_scope(tool, args)
    decision = await _manager().issue_child_lease(tool, args, origin=origin, effect_scope=scope,
                                                  risk_level=classify_execution_risk(tool, args, effect_scope=scope))
    assert decision.approved and decision.grant_id == OWN_GOAL_GRANT, decision.reason


def test_the_sentiment_organ_s_affect_blend_is_inside_its_phase_s_contract():
    import core.phases.cognitive_integration_phase as phase_module

    class Tracker:
        async def analyze(self, text, role="user"):
            return SimpleNamespace(valence=0.8, arousal=0.6, dominance=0.5, urgency=0.1, warmth=0.7, frustration=0.0)

        def get_mood_narrative(self):
            return ""

    phase = phase_module.CognitiveIntegrationPhase(SimpleNamespace(cycle_count=0))
    phase._sentiment_tracker = Tracker()
    state = AuraState.default()
    transformation = provenance.begin_transformation("CognitiveIntegrationPhase", state)
    asyncio.run(phase._run_sentiment(state, "That is wonderful news, thank you!"))
    receipt = transformation.complete(state, publish_violation=False)
    assert {"affect.valence", "affect.arousal"} <= set(receipt.observed_writes)
    assert receipt.undeclared_writes == ()


def test_a_skill_s_text_fields_keep_their_strings_and_other_fields_are_still_read():
    params = CapabilityEngine._normalize_execution_params(
        {"old_text": "", "new_text": "1", "read_only": "true", "max_files": "3"}, frozenset({"old_text", "new_text"}))
    assert params == {"old_text": "", "new_text": "1", "read_only": True, "max_files": 3}


def test_her_goal_tools_work_through_the_engine_from_her_goal_pass_and_from_nowhere_else(tmp_path, monkeypatch):
    """The whole path a call of hers takes: origin, lease, confirmation overlay, Will, the tool."""
    import subprocess

    workspace = tmp_path / "ws"
    (workspace / "core").mkdir(parents=True)
    (workspace / "core" / "notes.py").write_text("def note_count():\n    return 0\n")
    for args in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-q", "-m", "start"]):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=workspace, check=True)
    monkeypatch.setattr(work, "workspaces_dir", lambda: tmp_path)
    context = {"origin": work.ORIGIN, "her_goal": "probe", "her_goal_workspace": str(workspace), "requested_by": "aura"}

    async def calls():
        engine = CapabilityEngine()
        made = [await engine.execute(tool, args, dict(context)) for tool, args in [
            ("read_source", {"path": "core/notes.py"}),
            ("find_in_source", {"pattern": "note_count", "where": "core"}),
            ("edit_draft", {"path": "core/more.py", "old_text": "", "new_text": "X = 1\n"}),
            ("check_draft", {}),
        ]]
        elsewhere = await CapabilityEngine().execute(
            "edit_draft", {"path": "core/x.py", "old_text": "", "new_text": "Y = 1\n"},
            {"origin": "autonomous_initiative", "her_goal_workspace": str(workspace)})
        return made, elsewhere

    made, elsewhere = asyncio.run(calls())
    assert [result.get("ok") for result in made] == [True, True, True, True], made
    assert (workspace / "core" / "more.py").read_text() == "X = 1\n"
    assert elsewhere.get("ok") is False and not (workspace / "core" / "x.py").exists()


def test_a_simulator_hold_on_her_read_of_her_own_source_is_advisory_and_on_her_edit_still_holds(tmp_path, monkeypatch):
    """10 November 2026: the outcome simulator held her find_in_source at worst-case harm 0.80 and 1.00, scoring a
    search pattern of hers as harm. A read of her own source changes nothing; a hold on her edit still stands."""
    import subprocess
    from types import SimpleNamespace

    from core.container import ServiceContainer

    workspace = tmp_path / "ws"
    (workspace / "core").mkdir(parents=True)
    (workspace / "core" / "notes.py").write_text("def evict():\n    return 0\n")
    for args in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-q", "-m", "start"]):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=workspace, check=True)
    monkeypatch.setattr(work, "workspaces_dir", lambda: tmp_path)
    context = {"origin": work.ORIGIN, "her_goal": "probe", "her_goal_workspace": str(workspace), "requested_by": "aura"}

    class HoldsEverything:
        def assess_fast(self, description, context=None):
            return SimpleNamespace(recommendation="hold", worst_case_harm=0.8)

    original = ServiceContainer.get

    def get(name, default=None):
        return HoldsEverything() if name == "culture_mind" else original(name, default=default)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(get))

    async def calls():
        engine = CapabilityEngine()
        found = await engine.execute("find_in_source", {"pattern": "evict|forget", "where": "core"}, dict(context))
        edited = await engine.execute("edit_draft", {"path": "core/more.py", "old_text": "", "new_text": "X = 1\n"},
                                      dict(context))
        return found, edited

    found, edited = asyncio.run(calls())
    assert found.get("ok") is True, found
    assert edited.get("ok") is False and "Outcome simulator held" in str(edited.get("error")), edited

"""A goal she set herself is worked by her: her model's calls to her tools, kept across passes, a draft in a workspace
of her own, offered to Bryan, and in the source she runs on only once he takes it.

LIVE 8 November 2026: her goal "Implement persistent cold-store memory for user-specific context", held since 25
August and chosen hundreds of times, got an empty plan every time since dda8ab337 and nothing was built.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from core.agency import her_goal_at_work as work
from core.skills import her_draft

GOAL = "[AUTONOMOUS INITIATIVE] Keep a small note store for user-specific context"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def her_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "core").mkdir(parents=True)
    (source / "core" / "notes.py").write_text("def note_count():\n    return 0\n")
    (source / ".env").write_text("SECRET=1\n")
    _git("init", "-q", "-b", "main", cwd=source)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "add", "core/notes.py", cwd=source)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "start", cwd=source)
    monkeypatch.setattr(work, "source_root", lambda: source)
    monkeypatch.setattr(work, "workspaces_dir", lambda: tmp_path / "her_goals")
    monkeypatch.setattr(work, "records_dir", lambda: tmp_path / "records")
    return source


class _HerModelStandIn:
    """Makes the calls a pass of hers would, through her real tools, as the tool loop does."""

    def __init__(self, passes):
        self.passes = list(passes)
        self.seen_prior = []

    async def think_and_act(self, objective, *, system_prompt, tools, max_turns, context, origin, prior_turns,
                            work_in_background, native_tools_only):
        assert work_in_background and native_tools_only
        assert set(tools) == {skill.name for skill in her_draft.HER_TOOLS} and origin == work.ORIGIN
        self.seen_prior.append(list(prior_turns))
        if not self.passes:
            return {"content": "", "turns": 0, "tool_calls": []}
        made = []
        for number, (name, args) in enumerate(self.passes.pop(0)):
            skill = next(skill for skill in her_draft.HER_TOOLS if skill.name == name)()
            result = await skill.execute(args, context)
            made.append({"id": f"call_{len(self.seen_prior)}_{number}", "tool": name, "args": args, "result": str(result)})
        return {"content": "", "turns": len(made), "tool_calls": made}


NEW_TEST = "from core.notes import note_count\n\n\ndef test_counts_nothing_yet():\n    assert note_count() == 0\n"


def test_her_passes_build_a_draft_she_offers_and_bryan_takes(her_source):
    her = _HerModelStandIn([
        [("read_source", {"path": "core/notes.py"}), ("find_in_source", {"pattern": "note_count"})],
        [("edit_draft", {"path": "core/notes.py", "old_text": "return 0", "new_text": "return len(NOTES)"}),
         ("edit_draft", {"path": "core/notes.py", "old_text": "def note_count", "new_text": "NOTES = []\n\n\ndef note_count"}),
         ("edit_draft", {"path": "tests/test_notes.py", "old_text": "", "new_text": NEW_TEST}),
         ("check_draft", {}),
         ("offer_draft", {"summary": "Notes kept in a list, and counted from it."})],
    ])
    first = asyncio.run(work.work_a_while(GOAL, router=her))
    assert first.status == work.WORKING and [step["tool"] for step in first.steps] == ["read_source", "find_in_source"]
    assert "1: def note_count():" in first.steps[0]["result"]

    second = asyncio.run(work.work_a_while(GOAL, router=her))
    assert len(her.seen_prior[1]) == 4, "her next pass reads the calls she made before"
    assert second.status == work.OFFERED, second.steps[-2]["result"]
    assert "return 0" in (her_source / "core" / "notes.py").read_text(), "a draft changes nothing she runs on"

    assert work.approve(second.slug)["ok"]
    assert "return len(NOTES)" in (her_source / "core" / "notes.py").read_text()
    assert work.load(second.slug).status == work.MERGED
    assert "Drafted by Aura toward her goal" in _git("log", "-1", "--format=%B", cwd=her_source)


def test_a_draft_that_fails_its_check_cannot_be_offered(her_source):
    her = _HerModelStandIn([[
        ("edit_draft", {"path": "tests/test_notes.py", "old_text": "",
                        "new_text": "def test_wrong():\n    assert 1 == 2\n"}),
        ("check_draft", {}),
        ("offer_draft", {"summary": "A test that fails."}),
    ]])
    record = asyncio.run(work.work_a_while(GOAL, router=her))
    assert record.status == work.WORKING
    assert "has not passed a check" in record.steps[-1]["result"]


def test_bryan_s_no_goes_back_to_her_in_his_words(her_source):
    her = _HerModelStandIn([[
        ("edit_draft", {"path": "tests/test_notes.py", "old_text": "", "new_text": NEW_TEST}),
        ("check_draft", {}), ("offer_draft", {"summary": "A first test."}),
    ]])
    record = asyncio.run(work.work_a_while(GOAL, router=her))
    assert work.decline(record.slug, "Keep the notes on disk, not in a list.")["ok"]
    back = work.load(record.slug)
    assert back.status == work.WORKING
    assert work.what_she_works_toward(back).endswith("Bryan, on your draft: Keep the notes on disk, not in a list.")


@pytest.mark.parametrize("path", ["../outside.py", ".git/config", ".env", "keys/server.pem"])
def test_her_tools_stay_in_her_workspace_and_out_of_secrets(her_source, path):
    record = work.GoalRecord(goal=GOAL, slug=work.slug_of(GOAL))
    workspace = work.ensure_workspace(record)
    result = asyncio.run(her_draft.ReadSourceSkill().execute({"path": path}, {"her_goal_workspace": str(workspace)}))
    assert result["ok"] is False


def test_a_pass_her_model_cannot_take_now_changes_nothing(her_source):
    record = asyncio.run(work.work_a_while(GOAL, router=_HerModelStandIn([])))
    assert record.passes == 0 and work.load(record.slug).steps == []


def test_two_passes_without_a_call_are_a_goal_she_is_not_working(her_source):
    class SaysOnly:
        async def think_and_act(self, *args, **kwargs):
            return {"content": "I will get to it.", "turns": 1, "tool_calls": []}

    asyncio.run(work.work_a_while(GOAL, router=SaysOnly()))
    assert asyncio.run(work.work_a_while(GOAL, router=SaysOnly())).status == work.STALLED


def test_what_she_did_before_reaches_her_model_as_her_own_calls():
    from core.brain.llm.mlx_client import _prior_tool_turns

    steps = [{"id": "a", "tool": "read_source", "args": {"path": "x.py"}, "result": "1: x"}]
    turns = work.turns_from(steps, room=10_000)
    assert _prior_tool_turns(turns) == turns
    assert _prior_tool_turns(turns[:1]) == [], "a call without its answer is left out"


def test_the_task_engine_hands_her_own_unplanned_goal_to_her():
    import inspect

    from core.agency import autonomous_task_engine as engine

    assert engine._her_own_goal({"requested_by": "aura", "initiative": {"goal": GOAL}})
    assert not engine._her_own_goal({"requested_by": "user"})
    assert "_worked_by_her(goal, plan.plan_id, trace_id)" in inspect.getsource(engine.AutonomousTaskEngine.execute_goal)


def test_her_tools_say_their_limits_as_they_are():
    assert str(her_draft.READ_LINES) in her_draft.ReadSourceSkill.description
    assert str(her_draft.FOUND_LINES) in her_draft.FindInSourceSkill.description
    assert her_draft.CheckDraftSkill.timeout_seconds > her_draft.CHECK_SECONDS


def test_her_loop_on_her_own_goal_is_background_work_and_never_given_written_rules():
    import inspect

    from core.brain.llm import mlx_client

    assert mlx_client._in_the_background({}) == {}
    assert mlx_client._in_the_background({"work_in_background": True, "origin": work.ORIGIN}) == {
        "origin": work.ORIGIN, "is_background": True}
    source = inspect.getsource(mlx_client.MLXLocalClient.think_and_act)
    assert 'if native_tools and tools and kwargs.get("native_tools_only"):\n                    # Tools offered' in source

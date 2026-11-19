"""Her tool loop ends by writing what it found, whatever it called on the way.

LIVE 18 November 2026: a question about her own source took ten calls in 24 minutes, the tenth on the loop's last turn,
and the person got no answer.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.llm.the_tool_loops_last_pass import this_pass_writes
from core.sandbox.untrusted_python import available_boundary
from core.skills.code_repl import CodeREPLSkill
from core.skills.her_draft import FindInSourceSkill, ReadSourceSkill

_TOOLS = {"look": {"description": "Look.", "parameters": {"type": "object", "properties": {}}}}
_A_CALL = '```json\n{"tool": "look", "args": {}}\n```'


@pytest.mark.parametrize(("turn", "max_turns", "write_by", "longest", "now", "writes"), [
    (0, 5, 0.0, 0.0, 100.0, False),
    (4, 5, 0.0, 0.0, 100.0, True),
    (0, 1, 0.0, 0.0, 100.0, False),
    (1, 5, 1000.0, 100.0, 700.0, False),
    (1, 5, 1000.0, 200.0, 700.0, True),
    (1, 5, 1000.0, 0.0, 999.0, False),
])
def test_when_a_pass_writes(turn, max_turns, write_by, longest, now, writes):
    assert this_pass_writes(turn, max_turns, write_by, longest, now) is writes


def test_a_loop_is_given_what_a_waiting_person_is_given_not_one_generations_budget():
    """LIVE 18 November 2026: a 148 s budget made her second pass write, and she wrote a plan for work not yet done."""
    from core.brain.llm_health_router_waiting import a_waiting_persons_ceiling
    from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S

    ceiling = a_waiting_persons_ceiling()
    assert ceiling >= USER_FACING_COMPLETION_DEADLINE_MAX_S
    assert this_pass_writes(1, 10, 1000.0 + ceiling, 82.0, 1082.0) is False, "one call in, there is room for another"
    source = Path("core/brain/inference_gate_living_context.py").read_text(encoding="utf-8")
    assert 'max(loop_budget_s, a_waiting_persons_ceiling())' in source


def _a_loop(monkeypatch, replies, context=None, max_turns=3):
    client = MLXLocalClient.__new__(MLXLocalClient)
    offered, called = [], []

    async def generate(_prompt, **kwargs):
        offered.append(kwargs.get("tools"))
        return replies.pop(0)

    async def run_tool(name, _args, *_a, **_k):
        called.append(name)
        return {"ok": True, "summary": "seen"}

    client.max_tokens = 1024
    monkeypatch.setattr(client, "generate_text_async", generate, raising=False)
    monkeypatch.setattr(client, "get_last_surface_control_receipt", dict, raising=False)
    for name in ("_execute_tool_call", "_dispatch_tool", "_run_tool"):
        if hasattr(MLXLocalClient, name):
            monkeypatch.setattr(client, name, run_tool, raising=False)
    result = asyncio.run(client.think_and_act("How many?", "", tools=_TOOLS, max_turns=max_turns, context=context or {}))
    return result, offered, called


def test_the_last_turn_is_offered_no_tools_and_its_words_are_the_answer(monkeypatch):
    result, offered, _ = _a_loop(monkeypatch, [_A_CALL, _A_CALL, "There are three."])
    assert offered[0] and offered[1] and offered[2] is None
    assert result["content"] == "There are three."


def test_a_call_written_on_the_last_turn_is_not_run(monkeypatch):
    result, offered, _ = _a_loop(monkeypatch, [_A_CALL, _A_CALL, _A_CALL])
    assert offered[2] is None and result["content"] == "" and result["exhausted"] is True


def test_reading_her_source_needs_no_goal(tmp_path, monkeypatch):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "notes.py").write_text("WANTED = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    monkeypatch.setattr("core.agency.her_goal_at_work.source_root", lambda: tmp_path)
    read = asyncio.run(ReadSourceSkill().execute({"path": "core/notes.py"}, {}))
    assert read["ok"] and "WANTED" in read["text"]
    assert not asyncio.run(ReadSourceSkill().execute({"path": ".env"}, {}))["ok"], "a secret stays out of reach"
    assert FindInSourceSkill.effect_scope == "read_only"


@pytest.mark.skipif(not available_boundary(), reason="this host has no kernel boundary")
def test_code_she_runs_reads_her_source_by_its_own_path_and_cannot_change_it(tmp_path, monkeypatch):
    skill = CodeREPLSkill()
    monkeypatch.setattr(skill, "_output_dir", tmp_path)
    session = asyncio.run(skill._get_session_dir("a-session"))
    assert (session / "core").resolve() == Path(__file__).resolve().parents[1] / "core"
    done = asyncio.run(skill._execute_via_sandbox_runner(
        "import os\nprint(len(os.listdir('core/brain')) > 3)\n"
        "try:\n    open('core/brain/planted.py', 'w')\nexcept OSError as e:\n    print(type(e).__name__)\n",
        30, session))
    assert done["ok"] and done["stdout"].split() == ["True", "PermissionError"], done
    assert not (Path(__file__).resolve().parents[1] / "core/brain/planted.py").exists()

"""What the runtime says to a model is never served as her answer.

LIVE, 2026-10-28, seed 89's reports run. A warm-up turn asked her to "Write
today's plan into notes.txt." Her reading chose file_operation, the one
capability offered, and the handoff called it on the request alone: with no
arguments. The authority check could not name an action, ranked the call at
the skill's worst scope, and refused it. The refusal's text, written for the
model that made the call, came back as her reply:

    'file_operation' with action 'unspecified' needs state_mutation authority;
    this turn has read_write_artifacts. Ask for it explicitly, or use an action
    that stays within scope.

It was stored in 21 of her memories, recalled into later prompts, and given as
her answer in a measured report anchor.

Two faults, and a third of the same kind. A capability whose schema needs
arguments was called with none. A refusal became the content of the turn. And
the tool loop, out of turns, returned a sentence of its own ("I ran out of
reasoning steps.") or its last tool call as the answer.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.brain.inference_gate import InferenceGate

_LIVE_REFUSAL = {
    "ok": False,
    "status": "refused",
    "error": (
        "'file_operation' with action 'unspecified' needs state_mutation "
        "authority; this turn has read_write_artifacts. Ask for it explicitly, "
        "or use an action that stays within scope."
    ),
    "engine": "capability_engine",
}

_FILE_OPERATION = {
    "name": "file_operation",
    "parameters": {
        "type": "object",
        "properties": {"action": {"type": "string"}, "path": {"type": "string"}},
        "required": ["action", "path"],
    },
}

_REPAIR = {
    "name": "repair_a_program",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
}


class _Client:
    def __init__(self, result: dict[str, Any]):
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def think_and_act(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.result


@pytest.fixture
def gate() -> InferenceGate:
    return InferenceGate.__new__(InferenceGate)


def _answer(gate: Any, client: Any, prompt: str) -> str | None:
    return asyncio.run(
        gate._tool_grounded_answer(client, visible=prompt, system_prompt="", timeout_s=30.0)
    )


def _call_the_one(monkeypatch, *, refusal=None, executed=None) -> dict[str, Any]:
    from core.brain import inference_gate_living_context as living

    class _Engine:
        async def execute(self, name, args, context):
            return executed

    monkeypatch.setattr("core.brain.llm.mlx_client._refuse_action_beyond_authority", lambda *a: refusal)
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda *a, **k: _Engine())
    monkeypatch.setattr("core.brain.requests_by_meaning.served", lambda *a: None)
    return asyncio.run(living._call_the_one("file_operation", "Write today's plan into notes.txt.", {}))


def test_the_live_refusal_is_not_her_answer(monkeypatch):
    result = _call_the_one(monkeypatch, refusal=dict(_LIVE_REFUSAL))
    assert result["content"] == ""
    # The record of the call keeps what the refusal said.
    assert result["tool_calls"][0]["ok"] is False
    assert "needs state_mutation authority" in result["tool_calls"][0]["result"]


def test_an_error_is_not_her_answer(monkeypatch):
    result = _call_the_one(monkeypatch, executed={"ok": False, "error": "file_operation is unavailable"})
    assert result["content"] == ""
    assert result["tool_calls"][0]["ok"] is False


def test_what_a_capability_produced_reaches_her_as_her_call_not_as_her_words(monkeypatch):
    """LIVE 17 November 2026, 07:34: a capability's own summary stood as her draft, was read as cut off, and the turn
    ended on the runtime's apology. What it produced is her call's result; the reply is hers."""
    result = _call_the_one(monkeypatch, executed={"ok": True, "summary": "Mended: the ball now bounces."})
    assert result["content"] == ""
    assert result["tool_calls"][0]["ok"] is True
    assert "Mended: the ball now bounces." in str(result["tool_calls"][0]["result"])


def _one_capability(monkeypatch, name: str, definition: dict[str, Any], called: list[str]) -> None:
    async def _call(one, text, loop_context):
        called.append(one)
        return {"content": "mended", "tool_calls": [{"tool": one, "ok": True, "result": "mended"}]}

    monkeypatch.setattr("core.phases.response_contract.derive_capability_set", lambda _text, **_k: [name])
    monkeypatch.setattr("core.brain.llm.runtime_wiring.build_agentic_tool_map", lambda *a, **k: {name: definition})
    # Within this turn's ceiling, so what is tested is the call and not the reach.
    monkeypatch.setattr("core.brain.inference_gate._tools_within_reach", lambda tools, scopes: (tools, ()))
    monkeypatch.setattr("core.brain.inference_gate_living_context.the_one_asked_for", lambda text, tools: name)
    monkeypatch.setattr("core.brain.inference_gate_living_context._call_the_one", _call)
    monkeypatch.setattr("core.brain.inference_gate_living_context.capabilities_run_this_turn", lambda: frozenset())


def test_a_capability_that_needs_arguments_is_offered_not_called_alone(gate, monkeypatch):
    called: list[str] = []
    _one_capability(monkeypatch, "file_operation", _FILE_OPERATION, called)
    client = _Client({"content": "", "tool_calls": []})
    _answer(gate, client, "Write today's plan into notes.txt.")
    assert called == []
    assert len(client.calls) == 1
    assert set(client.calls[0]["tools"]) == {"file_operation"}


def test_a_capability_that_needs_none_is_still_called_on_the_request(gate, monkeypatch):
    called: list[str] = []
    _one_capability(monkeypatch, "repair_a_program", _REPAIR, called)
    client = _Client({"content": "unused", "tool_calls": []})
    assert _answer(gate, client, "The game at /tmp/pong.html is broken. Fix it.") == "mended"
    assert called == ["repair_a_program"]
    assert client.calls == []


def test_a_loop_out_of_turns_leaves_the_answer_to_her(gate, monkeypatch):
    monkeypatch.setattr("core.phases.response_contract.derive_capability_set", lambda _text, **_k: ["code_repl"])
    monkeypatch.setattr(
        "core.brain.llm.runtime_wiring.build_agentic_tool_map",
        lambda *a, **k: {"code_repl": {"name": "code_repl"}},
    )
    monkeypatch.setattr("core.brain.inference_gate_living_context.the_one_asked_for", lambda text, tools: None)
    monkeypatch.setattr("core.brain.inference_gate_living_context.her_reading_chose", lambda text, tools: None)
    ran = {"tool": "code_repl", "args": {"code": "print(2 + 2)"}, "result": "4", "ok": True}
    client = _Client({"content": "", "turns": 4, "tool_calls": [ran], "exhausted": True})
    assert _answer(gate, client, "Run Python to compute 2 + 2") is None


def test_the_loop_writes_no_answer_of_its_own_when_it_runs_out(monkeypatch):
    """A tool loop whose model asks for a tool every turn ends exhausted, with what ran and no words put in her mouth.
    Her step is taken once: a call that came back done is not made again because her model asks for it again."""
    from core.brain.llm.mlx_client import MLXLocalClient
    from core.container import ServiceContainer

    class Engine:
        async def execute(self, name, args, context=None, **_kwargs):
            return {"ok": True, "result": "4"}

    async def asks_again(*_args, **_kwargs):
        return '```json\n{"tool": "code_repl", "args": {"code": "print(2 + 2)"}}\n```'

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(
        lambda name, default=None: Engine() if name == "capability_engine" else default))
    client = MLXLocalClient.__new__(MLXLocalClient)
    client.max_tokens, client.model_path, client.generate_text_async = 512, "her-cortex", asks_again
    tools = {"code_repl": {"description": "run code", "parameters": {"code": {"type": "string"}}}}
    out = asyncio.run(client.think_and_act("Run Python to compute 2 + 2", "", tools=tools, max_turns=3))
    # Once her steps are done the pass is offered no tools, so a call written there is not run
    # (tests/test_the_tool_loops_last_pass.py, core/brain/llm/her_steps.py).
    assert out["exhausted"] is True and out["content"] == "" and len(out["tool_calls"]) == 1

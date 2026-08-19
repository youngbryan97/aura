"""The lane people type into can run a capability.

LIVE, 2026-08-19. Asked to run Python and report the number, with code_repl
READY, she wrote a snippet and stated an invented "Output:". Nothing ran.

The runtime has a tool loop — parse a call, bind it to the tool's advertised
schema, execute, feed the result back — reached through
``should_force_tool_handoff`` in ``llm_health_router``. The chat lane never
gets there. ``InferenceGate`` generates against the MLX client directly
(``local_client = self._mlx_client``), so the router's contract, its handoff,
and the loop behind it served every other caller and not the one people
actually type into. That is the chat-lane split: one capability, two paths,
wired to the path nobody uses.

These tests pin the properties that matter rather than the plumbing: a tool
that really ran produces an answer AND a receipt, and everything else leaves
ordinary generation alone.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.brain.inference_gate import InferenceGate


class _Client:
    """Stands in for the MLX client's ReAct loop."""

    def __init__(self, result: dict[str, Any]):
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def think_and_act(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.result


def _answer(gate: Any, client: Any, prompt: str) -> str | None:
    return asyncio.run(
        gate._tool_grounded_answer(
            client, visible=prompt, system_prompt="", timeout_s=30.0
        )
    )


@pytest.fixture
def gate() -> InferenceGate:
    return InferenceGate.__new__(InferenceGate)


def test_a_request_needing_no_capability_never_reaches_the_tool_loop(gate, monkeypatch):
    """Ordinary conversation must not pay for this, nor be changed by it."""
    monkeypatch.setattr(
        "core.phases.response_contract.derive_required_skill", lambda _text: None
    )
    client = _Client({"content": "hello", "tool_calls": [{"tool": "code_repl"}]})
    assert _answer(gate, client, "how are you feeling today") is None
    assert client.calls == []


def test_an_answer_with_no_tool_call_is_not_used(gate, monkeypatch):
    """The exact live failure: handed the tool, declined it, invented output."""
    monkeypatch.setattr(
        "core.phases.response_contract.derive_required_skill", lambda _text: "code_repl"
    )
    monkeypatch.setattr(
        "core.brain.llm.runtime_wiring.build_agentic_tool_map",
        lambda *a, **k: {"code_repl": {"name": "code_repl"}},
    )
    client = _Client({"content": "Output: 7", "tool_calls": []})
    assert _answer(gate, client, "run some python") is None


def test_a_tool_that_ran_produces_the_answer(gate, monkeypatch):
    monkeypatch.setattr(
        "core.phases.response_contract.derive_required_skill", lambda _text: "code_repl"
    )
    monkeypatch.setattr(
        "core.brain.llm.runtime_wiring.build_agentic_tool_map",
        lambda *a, **k: {"code_repl": {"name": "code_repl"}},
    )
    client = _Client(
        {"content": "It printed 7.", "tool_calls": [{"tool": "code_repl", "ok": True}]}
    )
    assert _answer(gate, client, "run some python") == "It printed 7."
    assert client.calls and client.calls[0]["tools"] == {"code_repl": {"name": "code_repl"}}


def test_a_failing_tool_loop_falls_back_to_ordinary_generation(gate, monkeypatch):
    """Never turn a working answer into no answer."""
    monkeypatch.setattr(
        "core.phases.response_contract.derive_required_skill", lambda _text: "code_repl"
    )
    monkeypatch.setattr(
        "core.brain.llm.runtime_wiring.build_agentic_tool_map",
        lambda *a, **k: {"code_repl": {"name": "code_repl"}},
    )

    class _Boom:
        async def think_and_act(self, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("worker died")

    assert _answer(gate, _Boom(), "run some python") is None


def test_no_tool_definition_means_no_loop(gate, monkeypatch):
    monkeypatch.setattr(
        "core.phases.response_contract.derive_required_skill", lambda _text: "code_repl"
    )
    monkeypatch.setattr(
        "core.brain.llm.runtime_wiring.build_agentic_tool_map", lambda *a, **k: None
    )
    client = _Client({"content": "x", "tool_calls": [{"tool": "code_repl"}]})
    assert _answer(gate, client, "run some python") is None
    assert client.calls == []


def test_an_empty_request_is_left_alone(gate):
    assert _answer(gate, _Client({}), "   ") is None


def test_the_persons_own_words_are_what_gets_read(gate, monkeypatch):
    """Not the assembled prompt.

    The scaffold runs to thousands of characters around a request of a
    hundred, and the first wiring passed the envelope. Live, that made every
    turn derive no capability at all — the helper returned before it even
    logged, which is why the fix looked like it had not run.
    """
    seen: list[str] = []

    def _derive(text: str) -> str | None:
        seen.append(text)
        return "code_repl"

    monkeypatch.setattr("core.phases.response_contract.derive_required_skill", _derive)
    monkeypatch.setattr(
        "core.brain.llm.runtime_wiring.build_agentic_tool_map",
        lambda *a, **k: {"code_repl": {"name": "code_repl"}},
    )
    client = _Client({"content": "7", "tool_calls": [{"tool": "code_repl"}]})
    _answer(gate, client, "run some python")
    assert seen == ["run some python"]

"""The pass of her tool loop that writes the answer cannot open a tool call, and any other pass can."""
from __future__ import annotations

import asyncio

import pytest

from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.llm.no_call_on_this_pass import install, no_call, tokens_that_open_a_call

mx = pytest.importorskip("mlx.core")


class _Tokenizer:
    """``<tool_call>`` is one token, as it is in her model, and ``<function=`` is three."""

    def encode(self, text, add_special_tokens=False):
        return {"<tool_call>": [7], "<function=": [3, 4, 5]}[text]


def _chosen(processor, tokens):
    return int(mx.argmax(processor(mx.array(tokens), mx.array([0.0, 1, 2, 3, 4, 5, 6, 9.0])), axis=-1).item())


def test_each_way_of_opening_a_call_is_found():
    assert tokens_that_open_a_call(_Tokenizer()) == {(): frozenset({7}), (3, 4): frozenset({5})}


def test_the_token_that_opens_a_call_cannot_be_chosen():
    assert _chosen(no_call(_Tokenizer()), [1, 2]) == 6, "7 had the highest score and is held"


def test_a_call_spelled_over_several_tokens_cannot_be_finished():
    processor = no_call(_Tokenizer())
    logits = mx.array([0.0, 0, 0, 0, 0, 9.0, 0, 0])
    assert int(mx.argmax(processor(mx.array([3, 4]), logits), axis=-1).item()) != 5
    assert int(mx.argmax(processor(mx.array([9, 4]), logits), axis=-1).item()) == 5, "the same token elsewhere is ordinary"


def test_only_a_pass_that_writes_is_held():
    held: list = []
    assert install(_Tokenizer(), {"writes_no_call": True}, held) is True and len(held) == 1
    free: list = []
    assert install(_Tokenizer(), {"writes_no_call": False}, free) is False and free == []
    assert install(_Tokenizer(), {}, free) is False and free == []


def test_her_loop_says_which_pass_writes(monkeypatch):
    client = MLXLocalClient.__new__(MLXLocalClient)
    client.max_tokens = 512
    told: list = []

    async def generate(_prompt, **kwargs):
        told.append(kwargs.get("writes_no_call"))
        return '```json\n{"tool": "look", "args": {}}\n```' if len(told) < 3 else "There are three."

    monkeypatch.setattr(client, "generate_text_async", generate, raising=False)
    monkeypatch.setattr(client, "get_last_surface_control_receipt", dict, raising=False)
    # It takes something, so each pass asks her model for the call: one that takes nothing her steps make themselves.
    tools = {"look": {"description": "Look.", "parameters": {"type": "object", "properties": {"at": {"type": "string"}}}}}
    asyncio.run(client.think_and_act("How many?", "", tools=tools, max_turns=3))
    assert told == [False, False, True]


@pytest.mark.parametrize(("job", "held"), [
    ({"clean_user_surface_contract": True, "offers_tools": False}, True),
    ({"clean_user_surface_contract": True, "offers_tools": True}, False),
    ({"clean_user_surface_contract": True}, False),
    ({"clean_user_surface_contract": False, "offers_tools": False}, False),
])
def test_the_pass_that_writes_her_reply_with_no_tool_offered_is_held(job, held):
    """LIVE 17 November 2026 she called code_repl from her reply pass, which no loop runs, and the worker rejected the
    call three times as prose. A job that does not say whether it offers tools is not taken to offer none."""
    processors: list = []
    assert install(_Tokenizer(), job, processors) is held and len(processors) == int(held)


def test_a_job_says_whether_its_pass_offers_tools_by_either_protocol():
    from core.brain.llm import mlx_client

    source = __import__("inspect").getsource(mlx_client._build_the_generation_request)
    assert '"offers_tools": bool(kwargs.get("tools") or kwargs.get("tool_budget_definitions"))' in source

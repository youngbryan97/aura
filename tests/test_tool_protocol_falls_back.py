"""A checkpoint that ends its turn has refused the protocol, not the work.

LIVE, 2026-08-19. Against a correct, small, well-formed ChatML tool prompt —
empty system message, a 150-character objective, five valid tool schemas, and
a properly open assistant turn — this model emitted <|im_end|> as its first
token. Every time. Nothing could be executed from chat, and from the outside
it looked like a model that would not use its tools.

Two protocols were already built in this function: the native one, where the
schema goes in the chat template, and a JSON contract that asks for the call
as an ordinary answer. Only the native one was reachable — the JSON block was
built solely when the template could not carry tools at all, which is a
different condition from the template carrying them and the model not
answering.
"""

from __future__ import annotations

import inspect

from core.brain.llm.mlx_client import MLXLocalClient

SOURCE = inspect.getsource(MLXLocalClient.think_and_act)


def test_both_protocols_are_built():
    """The JSON block used to be built only when native was impossible."""
    assert "tool_block" in SOURCE
    assert "if tools:" in SOURCE


def test_native_is_tried_first():
    assert "native_tools: list[dict[str, Any]] | None = template_tools or None" in SOURCE
    assert "tools=native_tools," in SOURCE


def test_an_empty_native_generation_retries_on_the_json_contract():
    assert "native_tools = None" in SOURCE
    assert "JSON tool contract" in SOURCE


def test_the_fallback_swaps_the_system_message_to_carry_the_block():
    """The JSON contract lives in the system message; native does not use it."""
    assert 'messages[0]["content"] = system_prompt + tool_block' in SOURCE


def test_the_fallback_cannot_loop_forever():
    """It clears native_tools before continuing, so the retry runs once."""
    fallback = SOURCE[SOURCE.index("Native tool protocol produced nothing") :]
    assert "native_tools = None" in fallback[: fallback.index("continue")]
    # And the loop is still bounded by max_turns.
    assert "for turn in range(max_turns):" in SOURCE

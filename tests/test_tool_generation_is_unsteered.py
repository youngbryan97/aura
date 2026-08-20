"""A tool call must reach the parser exactly as it was produced.

LIVE, 2026-08-19. Chasing "she will not use her tools", the tool generation
was put under the clean-user-surface contract to get an unsteered decode. That
contract does two things: it honours the decode controls AND it turns on
user-surface quality validation. So the surface quality gate was pointed at a
tool call — which is not a user-facing answer and never passes it — and the
worker generated 100 tokens and then cleared them:

    ⚠️ [WORKER] Generation produced 100 token(s) but no text survived to the
    caller — discarded downstream, not a decode failure

The generation was never the problem. Whether the eventual REPLY is a good
answer is judged later, on the reply.
"""

from __future__ import annotations

import inspect

from core.brain.llm.mlx_client import MLXLocalClient

SOURCE = inspect.getsource(MLXLocalClient.think_and_act)


def test_a_tool_call_is_not_judged_as_a_user_facing_answer():
    """The surface quality gate clears anything that is not one."""
    assert "clean_user_surface_contract=True" not in SOURCE


def test_the_empty_generation_is_recorded():
    """An empty generation and a declined call are different faults.

    Both end the loop with no tool calls, and only one of them is about tool
    calling. Without a record they are indistinguishable, which is how a
    downstream filter read for days as a model that would not call anything.
    """
    assert "came back empty" in SOURCE
    assert "none called" in SOURCE

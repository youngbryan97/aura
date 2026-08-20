"""A tool call is structured output, and steering destroys structured output.

LIVE, 2026-08-19. With five tools offered and the engine's affective alpha at
3.8, the tool-mode generation returned ONE token and no text survived:

    ⚠️ [WORKER] Generation produced 1 token(s) but no text survived to the
    caller — discarded downstream, not a decode failure

So every tool-using turn looked from the outside like a model that declined to
call anything, when nothing had been generated at all. Days of "she won't use
her tools" was a decoding fault the whole time.

The runtime already knows this. The clean user surface decodes at alpha 0.0,
and the code model is loaded unsteered outright — steering was corrupting
generated code. Tool calling is the third kind of structured generation the
runtime performs, and it was the one still being steered.
"""

from __future__ import annotations

import inspect

from core.brain.llm.mlx_client import MLXLocalClient


def test_the_tool_loop_asks_for_an_unsteered_decode():
    source = inspect.getsource(MLXLocalClient.think_and_act)
    assert "clean_user_surface_steering_alpha=0.0" in source
    # The worker honours the decode controls only under this contract. Without
    # it the alpha is carried and ignored, which is how the first attempt at
    # this fix changed nothing.
    assert "clean_user_surface_contract=True" in source
    # The worker's contract requires both controls together; alpha alone is
    # rejected as invalid runtime controls and silently leaves steering on.
    assert "clean_user_surface_recurrent_loops=1" in source


def test_the_controls_are_on_the_generation_that_carries_the_tools():
    """Not on some other call in the same function."""
    source = inspect.getsource(MLXLocalClient.think_and_act)
    start = source.index("tools=template_tools")
    end = source.index(")", source.index("clean_user_surface_recurrent_loops=1"))
    window = source[start:end]
    assert "clean_user_surface_steering_alpha=0.0" in window


def test_the_empty_generation_is_recorded():
    """An empty generation and a declined call are different faults.

    Both end the loop with no tool calls, and only one of them is about tool
    calling. Without a record they are indistinguishable, which is why the
    decoding fault was read as a behaviour problem.
    """
    source = inspect.getsource(MLXLocalClient.think_and_act)
    assert "came back empty" in source
    assert "none called" in source

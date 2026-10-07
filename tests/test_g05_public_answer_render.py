"""G05 renders her answer as her runtime does: channel open, or closed once an upstream phase settled it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("thinking, effort, raw, public", [
    (True, "medium", "work it out</think>The answer is 42.", "The answer is 42."),
    (False, None, "The answer is 42.", "The answer is 42."),
])
def test_the_channel_setting_reaches_the_template_and_the_split(monkeypatch, thinking, effort, raw, public) -> None:
    import mlx_lm

    import core.brain.llm.chat_format as chat_format
    from tools.run_g05_public_answers import decode_public

    seen = {}

    def render(tokenizer, conversation, **kwargs):
        seen.update(kwargs)
        return "prompt"

    def stream(model, tokenizer, tokens, **kwargs):
        yield SimpleNamespace(text=raw, finish_reason=None)
        yield SimpleNamespace(text="", finish_reason="stop")

    monkeypatch.setattr(chat_format, "render_chat_template", render)
    monkeypatch.setattr(mlx_lm, "stream_generate", stream)
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: [1, 2, 3])
    decoded = decode_public(object(), tokenizer, [{"role": "user", "content": "q"}], max_tokens=8,
                            thinking=thinking)
    assert seen["enable_thinking"] is thinking and seen["reasoning_effort"] == effort
    assert decoded["public_text"] == public and decoded["termination"] == "stop"

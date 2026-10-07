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


def test_the_open_ordinary_arm_shows_no_evidence() -> None:
    from tools.run_g05_public_answers import OPEN_ORDINARY, messages

    assert messages(OPEN_ORDINARY, "what is 2 and 3?", "Semantic program reader: ...") == [
        {"role": "user", "content": "what is 2 and 3?"}]
    assisted = messages("assisted", "what is 2 and 3?", "Semantic program reader: r1 = add(2, 3) = 5")
    assert len(assisted) == 2 and "r1 = add" in assisted[1]["content"]



def test_a_budget_stop_is_not_exact_even_when_an_answer_was_written_on_the_way() -> None:
    """G05 2026-10-07: a reply stopped at 4,096 tokens after "Answer: 871." was graded exact."""
    from tools.run_g05_public_answers import answer_exact

    assert answer_exact({"termination": "stop"}, 871, 871)
    assert not answer_exact({"termination": "length"}, 871, 871)
    assert not answer_exact({"termination": "native_thinking_incomplete"}, 871, 871)
    assert not answer_exact({"termination": "stop"}, 870, 871)

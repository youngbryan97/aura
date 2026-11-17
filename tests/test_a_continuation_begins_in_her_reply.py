"""A continuation's decode begins where her cut reply ends, in the reply, not in a private channel.

LIVE 16 November 2026, 22:17 local: "Could you put up a desktop notification that says time to stretch?" Her reply was
cut at "Help you adjust your quiet-hours settings if 10:15 PM shouldn't be a", and its continuation was decoded with the
channel taken as open, because the job allowed thinking. The 12 tokens that finished her sentence were read as private
reasoning and dropped ("Continuing from private channel at token 12 (end of turn)"), the loop asked her for the answer it
thought she had not begun, and she wrote the whole reply again after the cut: "...shouldn't be a I tried to push that
notification, but it got suppressed".
"""
from __future__ import annotations

import inspect

from core.brain.llm.chat_format import render_where_the_decode_begins

CUT = "If you'd like, I can also:\n- Help you adjust your quiet-hours settings if 10:15 PM shouldn't be a"


class _ChatML:
    """Her template's shape: a fresh turn opens the private channel; a continued one ends in its prefix."""

    chat_template = "{% if enable_thinking %}thinking{% endif %}"
    think_start, think_end = "<think>", "</think>"

    def apply_chat_template(self, messages, *, add_generation_prompt=False, continue_final_message=False,
                            enable_thinking=None, tokenize=False, tools=None, **_kwargs):
        turns = [f"<|im_start|>{m['role']}\n{m['content']}" for m in messages]
        if continue_final_message:
            return "<|im_end|>\n".join(turns)
        rendered = "".join(f"{turn}<|im_end|>\n" for turn in turns)
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
        return rendered


ASKED = [{"role": "user", "content": "Could you put up a desktop notification that says time to stretch?"}]


def test_a_continuation_begins_in_her_reply_though_the_turn_may_think():
    prompt, in_the_channel = render_where_the_decode_begins(
        _ChatML(), [*ASKED, {"role": "assistant", "content": CUT}], enable_thinking=True, continuation=True)

    assert prompt.endswith(CUT)
    assert in_the_channel is False


def test_a_prefix_that_left_the_channel_open_is_continued_in_it():
    open_channel = "<think>\nShe asked for a notification and quiet hours"
    _prompt, in_the_channel = render_where_the_decode_begins(
        _ChatML(), [*ASKED, {"role": "assistant", "content": open_channel}], enable_thinking=True, continuation=True)

    assert in_the_channel is True


def test_a_fresh_turn_keeps_what_the_turn_was_given():
    prompt, in_the_channel = render_where_the_decode_begins(_ChatML(), ASKED, enable_thinking=True)

    assert prompt.endswith("<|im_start|>assistant\n<think>\n") and in_the_channel is True
    assert render_where_the_decode_begins(_ChatML(), ASKED, enable_thinking=None)[1] is None


def test_her_worker_takes_the_channel_from_the_render():
    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker._mlx_worker_loop)
    assert "prompt, native_thinking = render_where_the_decode_begins(" in source
    assert "render_chat_continuation_template(" not in source, "one render decides both the prompt and the channel"

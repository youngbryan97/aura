"""The template's own reasoning control, driven rather than defaulted.

The Qwen3.8 template sets ``reasoning_effort`` to ``xhigh`` when nothing passes
it, and writes "Reasoning effort is set to xhigh. Please think carefully
through the task..." into the system block of every thinking render. Nothing
in the runtime had ever passed it, so every private channel — including one a
fast mode opened only because the answer is made in that call — was told to be
as careful as the model knows how. ``low`` is the template's own way of saying
what the fast mode meant.
"""

from __future__ import annotations

import pytest

from core.brain.llm.chat_format import (
    REASONING_EFFORTS,
    reasoning_effort_for_generation,
    render_chat_template,
    template_supports_reasoning_effort,
)


class _Template:
    """A tokenizer whose template honours the control, and one that ignores it."""

    def __init__(self, *, honours: bool) -> None:
        self.honours = honours
        self.chat_template = "{{ reasoning_effort }}" if honours else "{{ messages }}"

    def apply_chat_template(self, messages, **kwargs):
        effort = kwargs.get("reasoning_effort", "xhigh") if self.honours else ""
        thinking = kwargs.get("enable_thinking", True)
        head = f"[effort={effort}]" if (self.honours and thinking) else ""
        return head + "".join(str(m.get("content", "")) for m in messages)


# The per-turn choice measured its own cost before it measured its value.
# LIVE 2026-09-15: a turn with the channel closed retained 10,730 tokens; the
# next, a fast mode with the channel open and ``low`` rendered at the head of
# the system message, matched 0 of them and re-read 10,620 — 271 seconds on
# that host — because the first sentence differed. The head of the prompt is
# the cache's prefix. ``medium`` renders no sentence, so every turn on a lane
# shares its head whether its channel is open or closed.


@pytest.mark.parametrize("mode", ["fast", "deep", "reactive", "deliberate", None])
def test_an_open_channel_renders_the_same_head_whatever_the_mode(mode) -> None:
    assert reasoning_effort_for_generation(cognitive_mode=mode, thinking=True) == "medium"


def test_a_closed_channel_is_not_told_how_to_think() -> None:
    assert reasoning_effort_for_generation(cognitive_mode="deep", thinking=False) is None


def test_open_and_closed_channels_share_a_prompt_head() -> None:
    tokenizer = _Template(honours=True)
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]
    closed = render_chat_template(tokenizer, messages, enable_thinking=False)
    opened = render_chat_template(
        tokenizer,
        messages,
        enable_thinking=True,
        reasoning_effort=reasoning_effort_for_generation(cognitive_mode="fast", thinking=True),
    )
    # The stand-in renders "[effort=medium]" for any honoured value; the real
    # template renders nothing for medium. What this pins is that the head is
    # decided by the effort alone, and the effort no longer varies by mode.
    assert opened.startswith("[effort=medium]")
    assert closed == "Sq"


def test_the_effort_reaches_a_template_that_honours_it() -> None:
    tokenizer = _Template(honours=True)
    assert template_supports_reasoning_effort(tokenizer)
    rendered = render_chat_template(
        tokenizer, [{"role": "user", "content": "q"}], reasoning_effort="low"
    )
    assert rendered.startswith("[effort=low]")


def test_a_template_that_ignores_the_control_is_left_alone() -> None:
    tokenizer = _Template(honours=False)
    assert not template_supports_reasoning_effort(tokenizer)
    rendered = render_chat_template(
        tokenizer, [{"role": "user", "content": "q"}], reasoning_effort="low"
    )
    assert rendered == "q"


def test_an_effort_the_template_never_named_is_refused_here() -> None:
    with pytest.raises(ValueError):
        render_chat_template(
            _Template(honours=True), [{"role": "user", "content": "q"}], reasoning_effort="max"
        )
    assert "max" not in REASONING_EFFORTS

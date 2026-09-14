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


def test_a_fast_mode_that_opened_the_channel_asks_for_brief_thinking() -> None:
    assert reasoning_effort_for_generation(cognitive_mode="fast", thinking=True) == "low"


def test_a_thinking_mode_asks_for_careful_thinking() -> None:
    assert reasoning_effort_for_generation(cognitive_mode="deep", thinking=True) == "xhigh"


def test_a_closed_channel_is_not_told_how_to_think() -> None:
    assert reasoning_effort_for_generation(cognitive_mode="deep", thinking=False) is None


def test_no_mode_leaves_the_model_its_default() -> None:
    assert reasoning_effort_for_generation(cognitive_mode=None, thinking=True) is None


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

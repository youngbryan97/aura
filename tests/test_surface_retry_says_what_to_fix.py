"""A draft her worker could not serve is asked again as her turn was asked, with a reading of what became of it.

The retry used to carry instructions ("Regenerate the assistant reply ... avoid generic assistant identity, do not
invent unsupported prior topics ...", and advice per reason): instruction on the path to her words (F06; Bryan's rule,
no prompt engineering, ever). LIVE 13 November 2026 it ran on drafts a detector had wrongly thrown away. Only text that
is not her speech reaches a retry now (core/conversation/surface_disposition.UNSPEAKABLE_REASONS).
"""
from __future__ import annotations

from core.brain.llm.mlx_worker import (
    _build_user_surface_quality_retry_prompt,
    _messages_with_user_surface_retry,
)

_MESSAGES = [
    {"role": "user", "content": "Earlier question."},
    {"role": "assistant", "content": "Earlier answer."},
    {"role": "user", "content": "Keep my locker code 4919 in mind. Is forgetting a loss?"},
]


def test_the_retry_is_her_turn_as_asked_with_a_reading_of_what_became_of_the_draft():
    retry = _messages_with_user_surface_retry(_MESSAGES, ["prompt_artifact", "corrupted_language"])
    assert [m["role"] for m in retry] == ["user", "assistant", "user"], "no system message is added"
    assert retry[:2] == _MESSAGES[:2]
    last = retry[-1]["content"]
    assert last.startswith("[WHAT THE RECORD SAYS]\n- Your draft of this reply was not served")
    assert "prompt artifact, corrupted language" in last
    assert last.endswith(_MESSAGES[-1]["content"])
    for instruction in ("Regenerate", "avoid", "do not", "Answer only", "must"):
        assert instruction not in last, f"a reading, never an instruction: {instruction!r}"


def test_without_a_template_the_draft_is_asked_again_unchanged():
    prompt = _build_user_surface_quality_retry_prompt(
        tokenizer=object(), messages=None, tools=None, fallback_prompt="the draft's prompt", reasons=["prompt_artifact"])
    assert prompt == "the draft's prompt"


def test_the_retry_renders_with_the_drafts_channel_and_effort():
    """Rendered with the template's defaults, the retry opened the channel at
    xhigh — an effort sentence at the head of the prompt, which the cache
    matched 3 tokens of (2026-09-16)."""
    from core.brain.llm.mlx_worker_surface_repair import (
        _build_user_surface_quality_retry_prompt,
    )

    seen: dict = {}

    class _Tok:
        # A template that demonstrably honours both controls, so the render
        # passes them through rather than leaving them out as unsupported.
        chat_template = "{{ enable_thinking }}{{ reasoning_effort }}"

        def apply_chat_template(self, messages, **kwargs):
            seen.update(kwargs)
            head = ""
            if kwargs.get("enable_thinking", True):
                head = f"[effort={kwargs.get('reasoning_effort', 'xhigh')}]"
            return head + "".join(str(m.get("content", "")) for m in messages)

    prompt = _build_user_surface_quality_retry_prompt(
        tokenizer=_Tok(),
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        fallback_prompt="fallback",
        reasons=["internal_task_prompt_leak"],
        job={},
        enable_thinking=True,
        reasoning_effort="medium",
    )
    assert prompt.startswith("[effort=medium]")
    assert seen.get("enable_thinking") is True
    assert seen.get("reasoning_effort") == "medium"

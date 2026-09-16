"""A retry that does not say what to fix is a wasted decode.

Live: a draft was rejected three times with the IDENTICAL reason and the
identical validation hash — `generic_memory_pin_acknowledgement` — because the
retry prompt handed the model the raw reason name and no instruction. The turn's
whole budget went into regenerating the same mistake until "Request deadline
reached at token 23" ended it and the person got a refusal.
"""
from __future__ import annotations

from core.brain.llm.mlx_worker import (
    _SURFACE_RETRY_INSTRUCTIONS,
    _build_user_surface_quality_retry_prompt,
    _messages_with_user_surface_retry,
    _surface_retry_repair_instructions,
)

_MESSAGES = [
    {"role": "system", "content": "You are Aura."},
    {"role": "user", "content": "Keep my locker code 4919 in mind. Is forgetting a loss?"},
]


def test_the_live_reason_carries_an_actionable_instruction():
    text = _surface_retry_repair_instructions(["generic_memory_pin_acknowledgement"])
    assert "exact value" in text
    assert "answer the rest of the turn" in text


def test_instructions_reach_the_templated_retry_messages():
    retry = _messages_with_user_surface_retry(
        _MESSAGES, ["generic_memory_pin_acknowledgement"], {}
    )
    assert retry is not None
    system = next(m["content"] for m in retry if m["role"] == "system")
    assert "bare acknowledgement is not a reply" in system, (
        "the templated retry path must carry the repair instruction"
    )


def test_instructions_reach_the_suffix_fallback_path():
    prompt = _build_user_surface_quality_retry_prompt(
        tokenizer=object(),  # no apply_chat_template -> suffix path
        messages=_MESSAGES,
        tools=None,
        fallback_prompt="PROMPT",
        reasons=["truncated_tail"],
        job={},
    )
    assert "complete sentence" in prompt, (
        "the suffix fallback path must carry the repair instruction too"
    )


def test_unknown_reasons_are_tolerated():
    assert _surface_retry_repair_instructions(["some_future_reason"]) == ""
    assert _surface_retry_repair_instructions([]) == ""
    assert _surface_retry_repair_instructions(None) == ""


def test_duplicate_reasons_do_not_repeat_the_instruction():
    text = _surface_retry_repair_instructions(
        ["truncated_tail", "truncated_tail"]
    )
    assert text.count("complete sentence") == 1


def test_every_instruction_is_advice_not_jargon():
    for reason, instruction in _SURFACE_RETRY_INSTRUCTIONS.items():
        assert reason not in instruction, (
            f"{reason} echoes its own internal name back at the model"
        )
        assert len(instruction.split()) >= 8, f"{reason} instruction is too vague"


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

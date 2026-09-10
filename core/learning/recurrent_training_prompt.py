"""Canonical prompt rendering shared by recurrent training and launch custody."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

RECURRENT_TRAINING_COT_PREAMBLE = (
    "Work through this step by step, then end with your answer on its own line."
)


def answer_contract_instruction(task: Any) -> str:
    """Describe the machine-graded answer channel without revealing an answer."""

    try:
        expected = task.expected
    except (AttributeError, TypeError, ValueError):
        expected = None
    keys = sorted(str(key) for key in expected) if isinstance(expected, Mapping) else []
    key_text = f" Use exactly these JSON keys: {', '.join(keys)}." if keys else ""
    return (
        "Solve the task, then end with exactly one final line in this form: "
        "FINAL_ANSWER: {JSON object}. Do not write anything after that line."
        f"{key_text}"
    )


def render_recurrent_training_prompt(
    tokenizer: Any,
    task: Any,
    *,
    include_chain_of_thought: bool,
) -> tuple[str, tuple[int, ...]]:
    """Render and tokenize one exact prompt under the frozen training contract."""

    rendered = render_recurrent_training_prompt_text(
        tokenizer,
        task,
        include_chain_of_thought=include_chain_of_thought,
    )
    tokens = tuple(tokenizer.encode(rendered))
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise RuntimeError("rendered recurrent task produced invalid prompt tokens")
    return rendered, tokens


def render_recurrent_training_prompt_text(
    tokenizer: Any,
    task: Any,
    *,
    include_chain_of_thought: bool,
) -> str:
    """Render one exact prompt without requiring tokenizer encode support."""

    content = answer_contract_instruction(task) + "\n\n" + task.prompt
    if include_chain_of_thought:
        content = RECURRENT_TRAINING_COT_PREAMBLE + "\n\n" + content
    from core.brain.llm.chat_format import for_this_template

    rendered = tokenizer.apply_chat_template(
        for_this_template(tokenizer, [{"role": "user", "content": content}]),
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("recurrent training prompt renderer returned invalid text")
    return rendered


__all__ = [
    "RECURRENT_TRAINING_COT_PREAMBLE",
    "answer_contract_instruction",
    "render_recurrent_training_prompt",
    "render_recurrent_training_prompt_text",
]

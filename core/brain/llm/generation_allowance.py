"""Measured resident prompt and answer cost shared by response paths."""

from __future__ import annotations


def resident_generation_seconds(messages: list[dict], decode_max_tokens: int) -> float:
    """Return measured read/decode time, or zero without a loaded model rate."""
    from core.brain.llm.chat_format import thinking_enabled_for_generation
    from core.brain.llm.mlx_client import get_mlx_client
    from core.brain.llm.thinking_reserve import (
        reserve_tokens,
        seconds_to_decode,
        seconds_to_read,
    )

    client = get_mlx_client()
    identity = client.get_worker_identity_snapshot()
    model = str(identity.get("worker_model_path") or "")
    if not model:
        return 0.0
    private_tokens = (
        max(0, int(reserve_tokens(model)))
        if thinking_enabled_for_generation(model, answer_is_derived_here=True)
        else 0
    )
    capacity = max(decode_max_tokens, min(8192, decode_max_tokens + private_tokens))
    decode_seconds = seconds_to_decode(capacity, model)
    if decode_seconds <= 0.0:
        return 0.0
    prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
    read_seconds = max(
        seconds_to_read(prompt_chars), client.least_time_to_read(prompt_chars),
    )
    # Bridge and service each reserve eight seconds for delivery.
    return read_seconds + decode_seconds + 16.0

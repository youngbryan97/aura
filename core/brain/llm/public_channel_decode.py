"""Measure cached model decoding on its public channel, not private text."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.brain.llm.chat_format import split_native_thinking_generation

PUBLIC_CHANNEL_DECODE_POLICY = "native_public_channel_v1"


@dataclass(frozen=True, slots=True)
class PublicChannelDecode:
    text: str
    generated_tokens: int
    prefill_tokens: int
    boundary_tokens: int
    latency_ms: int
    stop_reason: str
    native_thinking: bool
    boundary_closed: bool
    reasoning_chars: int
    reasoning_sha256: str
    token_ids_sha256: str

    @property
    def stopped(self) -> bool:
        return self.stop_reason in {"eos", "public_contract"}

    def receipt(self) -> dict[str, Any]:
        return {
            "policy": PUBLIC_CHANNEL_DECODE_POLICY,
            "generated_tokens": self.generated_tokens,
            "prefill_tokens": self.prefill_tokens,
            "boundary_tokens": self.boundary_tokens,
            "latency_ms": self.latency_ms,
            "stop_reason": self.stop_reason,
            "native_thinking": self.native_thinking,
            "boundary_closed": self.boundary_closed,
            "reasoning_chars": self.reasoning_chars,
            "reasoning_sha256": self.reasoning_sha256,
            "token_ids_sha256": self.token_ids_sha256,
            "public_text_sha256": hashlib.sha256(self.text.encode()).hexdigest(),
        }


def decode_public_greedy(
    model: Any,
    tokenizer: Any,
    prompt_tokens: Sequence[int],
    *,
    max_tokens: int,
    public_prefill: Sequence[int] = (),
    completion_check: Callable[[str], bool] | None = None,
    progress: Callable[[int], None] | None = None,
) -> PublicChannelDecode:
    """Apply the same channel-aware stop rule to every experimental arm.

    A public wire prefix explicitly closes a template-opened private channel.
    That protocol prefix is counted separately from the wire and sampled
    tokens. Ordinary generation may finish thinking naturally. No private
    prose is edited into a public answer or retained in the returned receipt.
    """

    from core.brain.llm.unified_recurrent_transfer_decode import decode_base_greedy_tokens

    prompt = tuple(prompt_tokens)
    prefix = tuple(public_prefill)
    rendered = tokenizer.decode(list(prompt), skip_special_tokens=False)
    native = str(rendered).rstrip().endswith("<think>")
    boundary = ()
    if native and prefix:
        boundary = tuple(int(t) for t in tokenizer.encode("</think>\n\n", add_special_tokens=False))
        if not boundary or "</think>" not in tokenizer.decode(list(boundary), skip_special_tokens=False):
            raise ValueError("native public prefill has no valid closing boundary")
    effective_prefix = (*boundary, *prefix)
    eos = getattr(tokenizer, "eos_token_id", None)
    completed = False
    last_tokens = None
    last_channels = None

    def channels(values):
        nonlocal last_tokens, last_channels
        if values == last_tokens:
            return last_channels
        visible = values[:-1] if values and eos is not None and values[-1] == eos else values
        raw = tokenizer.decode(list(visible), skip_special_tokens=False)
        last_tokens = values
        last_channels = split_native_thinking_generation(
            raw, native_thinking=native or str(raw).lstrip().startswith("<think>"),
        )
        return last_channels

    def complete(values):
        nonlocal completed
        current = channels(values)
        completed = bool(
            current.boundary_closed and completion_check is not None
            and completion_check(current.surface)
        )
        return completed

    values, stopped, latency = decode_base_greedy_tokens(
        model, prompt, eos_token_id=eos, max_tokens=max_tokens,
        prefill_tokens=effective_prefix, completion_check=complete, progress=progress,
    )
    result = channels(values)
    generated = len(values) - len(effective_prefix)
    reason = (
        "public_contract" if completed else
        "eos" if stopped and values and values[-1] == eos else
        "token_limit" if generated >= max_tokens else "generator_exhausted"
    )
    return PublicChannelDecode(
        result.surface, generated, len(prefix), len(boundary), latency, reason,
        native, result.boundary_closed, len(result.reasoning),
        hashlib.sha256(result.reasoning.encode()).hexdigest(),
        hashlib.sha256(",".join(str(token) for token in values).encode("ascii")).hexdigest(),
    )

"""Measure cached model decoding on its public channel, not private text."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.brain.llm.chat_format import split_native_thinking_generation

PUBLIC_CHANNEL_DECODE_POLICY = "native_public_channel_v1"
PUBLIC_CHANNEL_SAMPLE_POLICY = "native_public_sample_v1"


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
    policy: str = PUBLIC_CHANNEL_DECODE_POLICY

    @property
    def stopped(self) -> bool:
        return self.stop_reason in {"eos", "public_contract"}

    def receipt(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
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


def decode_public_sample(
    model: Any, tokenizer: Any, prompt: str | Sequence[int], *,
    max_tokens: int, sampler: Any, progress: Callable[[int], None] | None = None,
    logits_processors: Sequence[Callable] | None = None,
) -> PublicChannelDecode:
    """Sample through MLX's streaming API and retain only the public channel.

    Sampling, token allocation, and template bytes are unchanged. A private
    continuation that exhausts its allocation remains an unanswered sample.
    """
    from mlx_lm.generate import stream_generate

    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("sample token allocation must be positive")
    rendered = prompt if isinstance(prompt, str) else tokenizer.decode(
        list(prompt), skip_special_tokens=False,
    )
    native = str(rendered).rstrip().endswith("<think>")
    segments, tokens = [], []
    last = None
    started = time.monotonic()
    options = {} if logits_processors is None else {"logits_processors": list(logits_processors)}
    for response in stream_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=max_tokens, sampler=sampler, **options):
        segments.append(response.text)
        tokens.append(int(response.token))
        last = response
        if progress is not None:
            progress(int(response.generation_tokens))
    raw = "".join(segments)
    native = native or raw.lstrip().startswith("<think>") or "</think>" in raw
    channels = split_native_thinking_generation(raw, native_thinking=native)
    finish = getattr(last, "finish_reason", None)
    reason = "eos" if finish == "stop" else "token_limit" if finish == "length" else "generator_exhausted"
    return PublicChannelDecode(
        channels.surface, int(last.generation_tokens) if last is not None else 0,
        0, 0, int((time.monotonic() - started) * 1000), reason,
        native, channels.boundary_closed, len(channels.reasoning),
        hashlib.sha256(channels.reasoning.encode()).hexdigest(),
        hashlib.sha256(",".join(str(token) for token in tokens).encode("ascii")).hexdigest(),
        policy=PUBLIC_CHANNEL_SAMPLE_POLICY,
    )


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

    def channels(values: Any) -> Any:
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

    def complete(values: Any) -> bool:
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

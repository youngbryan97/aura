"""Several greedy decodes in one pass, each as ``run_g05_public_answers.decode_public`` makes one.

G12's benchmarks run hundreds of long reasoning decodes on a model whose
decoding is bound by memory bandwidth, not arithmetic, so a batch of requests
costs little more per step than one. Each conversation is rendered exactly as
the single path renders it (her chat template, native thinking at her serving
effort), decoded greedily by mlx_lm's BatchGenerator, and split into private
and public channels the same way. Greedy decoding in a batch can differ from
a single decode in the last digits of the arithmetic; a harness using this says
so, and checks its first rows against single decodes.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from typing import Any


def render_tokens(tokenizer: Any, conversation: list, *, thinking: bool = True) -> list[int]:
    from core.brain.llm.chat_format import reasoning_effort_for_generation, render_chat_template

    prompt = render_chat_template(tokenizer, conversation, add_generation_prompt=True,
                                  enable_thinking=thinking,
                                  reasoning_effort=reasoning_effort_for_generation(thinking=thinking))
    return [int(t) for t in tokenizer.encode(prompt, add_special_tokens=False)]


def decode_stream(
    model: Any,
    tokenizer: Any,
    conversations: Sequence[list],
    *,
    max_tokens: int,
    width: int,
    on_record: Any,
    thinking: bool = True,
) -> None:
    """Decode every conversation, ``width`` at a time, and hand each record over as it finishes.

    The batch refills: the moment one sequence ends the next conversation is
    admitted, so a long reply holds one slot rather than a whole batch.
    ``on_record(index, record)`` receives the conversation's index and the
    fields decode_public returns; ``seconds`` runs from that sequence's first
    generated token, so time spent waiting for a slot is not counted.
    """
    from mlx_lm.generate import BatchGenerator

    from core.brain.llm.chat_format import split_native_thinking_generation

    prompts = [render_tokens(tokenizer, conversation, thinking=thinking) for conversation in conversations]
    if not prompts:
        return
    generator = BatchGenerator(model, stop_tokens=[[token] for token in tokenizer.eos_token_ids],
                               completion_batch_size=width, prefill_batch_size=min(width, 8))
    uids = generator.insert(prompts, [max_tokens] * len(prompts))
    index_of = {uid: index for index, uid in enumerate(uids)}
    tokens: dict[int, list[int]] = {uid: [] for uid in uids}
    started: dict[int, float] = {}
    try:
        while responses := generator.next_generated():
            for response in responses:
                uid = response.uid
                started.setdefault(uid, time.monotonic())
                if response.finish_reason != "stop":
                    tokens[uid].append(int(response.token))
                if response.finish_reason is None:
                    continue
                raw = tokenizer.decode(tokens[uid])
                channels = split_native_thinking_generation(raw, native_thinking=thinking)
                finish = "stop" if response.finish_reason == "stop" else "token_limit"
                on_record(index_of[uid], {
                    "public_text": channels.surface,
                    "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "prompt_tokens": len(prompts[index_of[uid]]),
                    "generated_tokens": len(tokens[uid]),
                    "termination": finish if channels.boundary_closed else "native_thinking_incomplete",
                    "seconds": round(time.monotonic() - started[uid], 3),
                    "batch_size": width,
                    "batching": "continuous",
                })
                tokens[uid] = []
    finally:
        generator.close()


def decode_batch(
    model: Any, tokenizer: Any, conversations: Sequence[list], *, max_tokens: int, thinking: bool = True
) -> list[dict[str, Any]]:
    """One record per conversation, in order, with the fields decode_public returns."""
    from mlx_lm.generate import BatchGenerator

    from core.brain.llm.chat_format import split_native_thinking_generation

    prompts = [render_tokens(tokenizer, conversation, thinking=thinking) for conversation in conversations]
    if not prompts:
        return []
    generator = BatchGenerator(model, stop_tokens=[[token] for token in tokenizer.eos_token_ids],
                               completion_batch_size=len(prompts), prefill_batch_size=min(len(prompts), 8))
    began = time.monotonic()
    uids = generator.insert(prompts, [max_tokens] * len(prompts))
    tokens: dict[int, list[int]] = {uid: [] for uid in uids}
    finish: dict[int, str] = {uid: "token_limit" for uid in uids}
    ended: dict[int, float] = {}
    try:
        while responses := generator.next_generated():
            for response in responses:
                if response.finish_reason != "stop":
                    tokens[response.uid].append(int(response.token))
                if response.finish_reason is not None:
                    finish[response.uid] = "stop" if response.finish_reason == "stop" else "token_limit"
                    ended[response.uid] = time.monotonic()
    finally:
        generator.close()
    records = []
    for uid, prompt in zip(uids, prompts, strict=True):
        raw = tokenizer.decode(tokens[uid])
        channels = split_native_thinking_generation(raw, native_thinking=thinking)
        records.append({
            "public_text": channels.surface,
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "prompt_tokens": len(prompt),
            "generated_tokens": len(tokens[uid]),
            "termination": finish[uid] if channels.boundary_closed else "native_thinking_incomplete",
            "seconds": round(ended.get(uid, time.monotonic()) - began, 3),
            "batch_size": len(prompts),
        })
    return records

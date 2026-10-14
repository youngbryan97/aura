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
from collections.abc import Iterator, Sequence
from typing import Any

#: Bound lazy metadata chains without synchronizing every token's forward pass.
CACHE_METADATA_EVERY = 64


def _evaluate_cache_metadata(generator: Any) -> None:
    """Materialize cache bookkeeping on the stream that advances it.

    MLX-LM 0.31.3's ArraysCache.state omits length and left-padding arrays.
    Unread per-step decrements retain live Metal buffers even when token
    outputs are evaluated. Clearing the allocator cache cannot release them.
    See https://github.com/ml-explore/mlx-lm/issues/1641.

    BatchGenerator's two cache collections and stream are internal APIs here;
    an incompatible dependency must fail rather than silently skip the repair.
    """
    import mlx.core as mx

    try:
        stream = generator.stream
        batches = (generator._prompt_batch, generator._generation_batch)
        members = [member for batch in batches for member in batch.prompt_cache]
    except AttributeError as why:
        raise RuntimeError("BatchGenerator no longer exposes its cache metadata and stream") from why
    with mx.stream(stream):
        arrays = []
        seen_members: set[int] = set()
        seen_arrays: set[int] = set()
        while members:
            member = members.pop()
            if id(member) in seen_members:
                continue
            seen_members.add(id(member))
            members.extend(getattr(member, "caches", ()))
            for name in ("left_padding", "lengths", "offset"):
                value = getattr(member, name, None)
                if isinstance(value, mx.array) and id(value) not in seen_arrays:
                    seen_arrays.add(id(value))
                    arrays.append(value)
        if arrays:
            mx.eval(arrays)


def _generation_steps(generator: Any) -> Iterator[list[Any]]:
    """Keep lazy cache metadata bounded while preserving each generated response."""
    _evaluate_cache_metadata(generator)
    steps = 0
    while True:
        responses = generator.next_generated()
        steps += 1
        if (steps % CACHE_METADATA_EVERY == 0 or not responses
                or any(response.finish_reason is not None for response in responses)):
            _evaluate_cache_metadata(generator)
        if not responses:
            return
        yield responses


def render_tokens(tokenizer: Any, conversation: list, *, thinking: bool = True) -> list[int]:
    from core.brain.llm.chat_format import reasoning_effort_for_generation, render_chat_template

    prompt = render_chat_template(tokenizer, conversation, add_generation_prompt=True,
                                  enable_thinking=thinking,
                                  reasoning_effort=reasoning_effort_for_generation(thinking=thinking))
    return [int(t) for t in tokenizer.encode(prompt, add_special_tokens=False)]


def _decode_token_groups(model: Any, tokenizer: Any, prompts: Sequence[list[int]], *, max_tokens: int | Sequence[int],
                         width: int, on_finish: Any, sampler: Any = None, seed: int | None = None,
                         carry: bool = False) -> None:
    """Decodes of token prompts in fixed groups of ``width``: greedy, or by ``sampler`` from ``seed``.

    Each group gets its own generator, prefilled once before any of it
    generates, and nothing new is admitted while it generates; the next group
    starts when the last of this one ends. On 8 October a run that admitted
    new prompts into a generating batch kernel-panicked the Mac in the GPU
    driver (IOGPUGroupMemory::remove_memory_object, the panicked task this
    process); fixed groups have no panic on record. A group's buffers are
    released after the GPU has finished with them. ``on_finish(index, tokens,
    finish_reason, seconds, group_size)`` is called as each sequence ends;
    ``tokens`` excludes the stop token and ``seconds`` runs from its first
    generated token. A sampled run seeds MLX's generator once per group from
    ``seed`` and the group's first index, so a resumed run draws the same.
    ``max_tokens`` is one bound for all, or one per prompt.

    With ``carry``, a full group that falls to half its width or fewer while
    prompts are still waiting is ended there, and its unfinished sequences
    go first into the next group, their prompt followed by what they have
    generated so far, with what is left of their bound. A group otherwise
    spends its last stretch with one or two sequences on a device built for
    eight: measured occupancy of the G09 induction rounds ran from 0.57 to
    0.94. Nothing is ever added to a generating batch, so the panic-free
    shape holds; the cost is reading each carried sequence again as a prompt.
    """
    import gc

    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator

    bounds_all = [int(max_tokens)] * len(prompts) if isinstance(max_tokens, int) else [int(b) for b in max_tokens]
    # (index, tokens to read, generated so far, bound left, first generated at)
    pending = [(i, list(prompts[i]), [], bounds_all[i], None) for i in range(len(prompts))]
    width = max(1, width)
    first_of_group = 0
    while pending:
        group, pending = pending[:width], pending[width:]
        if sampler is not None and seed is not None:
            mx.random.seed(int(seed) * 1_000_003 + first_of_group)
        first_of_group += len(group)
        # A greedy run calls the generator exactly as before samplers existed here.
        sampling = {"sampler": sampler} if sampler is not None else {}
        generator = BatchGenerator(model, stop_tokens=[[token] for token in tokenizer.eos_token_ids],
                                   completion_batch_size=len(group), prefill_batch_size=len(group), **sampling)
        uids = generator.insert([item[1] for item in group], [item[3] for item in group])
        item_of = dict(zip(uids, group, strict=True))
        tokens: dict[int, list[int]] = {uid: list(item_of[uid][2]) for uid in uids}
        started: dict[int, float] = {uid: item_of[uid][4] for uid in uids if item_of[uid][4] is not None}
        active = set(uids)
        carried: list[tuple] = []
        try:
            for responses in _generation_steps(generator):
                for response in responses:
                    uid = response.uid
                    started.setdefault(uid, time.monotonic())
                    if response.finish_reason != "stop":
                        tokens[uid].append(int(response.token))
                    if response.finish_reason is None:
                        continue
                    active.discard(uid)
                    on_finish(item_of[uid][0], tokens[uid], response.finish_reason,
                              round(time.monotonic() - started[uid], 3), len(group))
                    tokens[uid] = []
                if carry and pending and len(group) == width and active and len(active) <= width // 2:
                    for uid in sorted(active, key=lambda u: item_of[u][0]):
                        index, read, before, bound, _ = item_of[uid]
                        new = tokens[uid][len(before):]
                        carried.append((index, read + new, tokens[uid], bound - len(new), started.get(uid)))
                    break
        finally:
            generator.close()
            del generator
            mx.synchronize()
            gc.collect()
            mx.clear_cache()
        pending = carried + pending


def decode_stream(
    model: Any,
    tokenizer: Any,
    conversations: Sequence[list],
    *,
    max_tokens: int | Sequence[int],
    width: int,
    on_record: Any,
    thinking: bool = True,
    sampler: Any = None,
    seed: int | None = None,
    carry: bool = False,
) -> None:
    """Decode every conversation in fixed groups of ``width``, handing each record over as it finishes.

    ``on_record(index, record)`` receives the conversation's index and the
    fields decode_public returns (see _decode_token_groups for the grouping).
    """
    from core.brain.llm.chat_format import split_native_thinking_generation

    prompts = [render_tokens(tokenizer, conversation, thinking=thinking) for conversation in conversations]

    def finished(index: int, generated: list[int], reason: str, seconds: float, group_size: int) -> None:
        raw = tokenizer.decode(generated)
        channels = split_native_thinking_generation(raw, native_thinking=thinking)
        finish = "stop" if reason == "stop" else "token_limit"
        on_record(index, {
            "public_text": channels.surface,
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "prompt_tokens": len(prompts[index]),
            "generated_tokens": len(generated),
            "termination": finish if channels.boundary_closed else "native_thinking_incomplete",
            "seconds": seconds,
            "batch_size": group_size,
            "batching": "fixed_groups_carrying" if carry else "fixed_groups",
        })

    _decode_token_groups(model, tokenizer, prompts, max_tokens=max_tokens, width=width, on_finish=finished,
                         sampler=sampler, seed=seed, carry=carry)


#: What her chat template writes to close the private channel (chat_template.jinja).
CLOSE_THINKING = "\n</think>\n\n"


def decode_with_thinking_budget(
    model: Any,
    tokenizer: Any,
    conversations: Sequence[list],
    *,
    max_tokens: int,
    answer_tokens: int,
    width: int,
    on_record: Any,
) -> None:
    """Decode with the private channel open, closing it for any sequence that is still thinking
    when only ``answer_tokens`` of its ``max_tokens`` remain (budget forcing, Muennighoff et al.
    2025, "s1: Simple test-time scaling").

    Every sequence first decodes up to ``max_tokens - answer_tokens``. One that ends there is
    done. One cut there continues for up to ``answer_tokens`` more from its own text, re-read as
    its prompt; if its private channel was still open, the template's closing marker is written
    first, so what follows is its reply. The record says whether the channel was closed for it.
    """
    from core.brain.llm.chat_format import split_native_thinking_generation

    if not 0 < answer_tokens < max_tokens:
        raise ValueError("answer_tokens must leave some of max_tokens for thinking")
    prompts = [render_tokens(tokenizer, conversation, thinking=True) for conversation in conversations]
    close = [int(t) for t in tokenizer.encode(CLOSE_THINKING, add_special_tokens=False)]

    def emit(index: int, generated: list[int], reason: str, seconds: float, group_size: int, forced: bool) -> None:
        raw = tokenizer.decode(generated)
        channels = split_native_thinking_generation(raw, native_thinking=True)
        finish = "stop" if reason == "stop" else "token_limit"
        on_record(index, {
            "public_text": channels.surface,
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "prompt_tokens": len(prompts[index]),
            "generated_tokens": len(generated),
            "termination": finish if channels.boundary_closed else "native_thinking_incomplete",
            "thinking_closed_by_budget": forced,
            "seconds": seconds,
            "batch_size": group_size,
            "batching": "fixed_groups",
        })

    cut: dict[int, tuple[list[int], float, bool]] = {}

    def first_finished(index: int, generated: list[int], reason: str, seconds: float, group_size: int) -> None:
        if reason == "stop":  # done within the first budget: handed over now
            emit(index, list(generated), reason, seconds, group_size, False)
            return
        forced = "</think>" not in tokenizer.decode(generated)
        cut[index] = (list(generated) + (close if forced else []), seconds, forced)

    _decode_token_groups(model, tokenizer, prompts, max_tokens=max_tokens - answer_tokens, width=width,
                         on_finish=first_finished)
    order = sorted(cut)

    def second_finished(position: int, generated: list[int], reason: str, seconds: float, group_size: int) -> None:
        index = order[position]
        joined, first_seconds, forced = cut[index]
        emit(index, joined + list(generated), reason, first_seconds + seconds, group_size, forced)

    if order:
        _decode_token_groups(model, tokenizer, [prompts[i] + cut[i][0] for i in order], max_tokens=answer_tokens,
                             width=width, on_finish=second_finished)


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
        for responses in _generation_steps(generator):
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

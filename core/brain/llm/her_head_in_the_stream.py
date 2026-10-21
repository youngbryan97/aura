"""Her own next-next-token head, drafting inside her worker's stream: two tokens a pass with everything she has on.

Bryan, 20 October 2026: one person, one Aura; as fast as is physically possible while keeping everything. Decoding
reads all 14 GB of her 4-bit weights for each pass, and a pass that reads two tokens costs about the same as one, so
the way past her model's 17 tokens a second on this machine is fewer passes a token. Her architecture carries a
multi-token prediction head (core/brain/llm/two_tokens_a_step.py ``MTPHead``): from her hidden state at a position
and the next token's embedding it predicts the token after. MEASURED that day on her persona 27B, its drafts were
kept 94 to 98 per cent of the time (``tools/measure_two_tokens_a_step.py``).

mlx_lm's speculative step cannot run her: it needs caches it can trim, and her gated delta net layers keep a running
state that cannot be trimmed. core/brain/llm/several_tokens_a_pass.py reads several tokens keeping each linear
layer's state after every one, and puts back the state at the last kept token. This is that pass in the shape her
worker's stream takes (mlx_lm ``stream_generate`` and its ``GenerationResponse``), with what her generations carry:

* her logits processors, applied to each checked position in order, with the token history mlx_lm gives them (the
  prompt's last token and every token since), so her private channel's bound counts as it does one token at a time;
* her hidden-state tap: before a position's processors run, the tap holds that position's hidden state, so her
  one-shot memory recalls from the token it is choosing after;
* the worker's prompt cache, read on from where it stands;
* her sampler: a draft is kept when the token sampled at its position is the draft, which keeps her distribution
  exactly; greedy is the case of an argmax sampler.

Two reads back from the GPU a pass (the tokens chosen, then the next draft), not one a token.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger("LLM.HerHead")

__all__ = ["HER_HEAD", "head_generate_step", "her_head_may_check", "her_own_head", "stream_generate_with_her_head", "timed"]

#: Where her head's weights are: the original release's multi-token prediction layer.
HER_HEAD = Path.home() / ".aura" / "models" / "qwen3.8-27b-mtp-source" / "mtp_head.safetensors"


def _switched_on() -> bool:
    """AURA_HER_HEAD_DRAFTS, on unless set to 0. MEASURED 20 October 2026 on her 27B with every channel on
    (tools/her_decode_with_each_channel.py --with-her-head): 15.2 tokens a second without it, 21.5 with it, and her
    tokens the same up to the 59th, where her first and second choices stood 0.125 nats apart."""
    return os.environ.get("AURA_HER_HEAD_DRAFTS", "1").strip().lower() not in {"0", "false", "no", "off"}


def her_own_head(model: Any) -> Any:
    """Her multi-token prediction head joined to her model, where she has one and it is switched on; else None."""
    if model is None or not _switched_on():
        return None
    text = getattr(model, "language_model", model)
    if getattr(getattr(text, "args", None), "model_type", "") != "qwen3_5_text" or not HER_HEAD.is_file():
        return None
    try:
        from core.brain.llm.two_tokens_a_step import load_mtp_head

        head = load_mtp_head(model, HER_HEAD)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation("her_head_in_the_stream", exc, severity="warning", action="decoded without her head's drafts")
        return None
    logger.info("Speculative decoding: her own head drafts (%s).", HER_HEAD.name)
    return head


def her_head_may_check(draft_model: Any, generation_kwargs: dict) -> bool | None:
    """Whether her head may draft for these generation arguments; None when the draft is not her head. Her pass keeps
    her logits processors, her tap and the prompt cache, which another draft model's could not."""
    from core.brain.llm.two_tokens_a_step import MTPHead

    if not isinstance(draft_model, MTPHead):
        return None
    return generation_kwargs.get("kv_bits") is None


def head_generate_step(prompt: mx.array, model: Any, head: Any, *, max_tokens: int = 256,
                       sampler: Callable[[mx.array], mx.array] | None = None,
                       logits_processors: Sequence[Callable[[Any, Any], Any]] | None = None,
                       prompt_cache: list | None = None, prefill_step_size: int = 512,
                       prompt_progress_callback: Callable[[int, int], None] | None = None,
                       tap: Any = None, **_ignored: Any) -> Generator[tuple[int, mx.array, bool], None, None]:
    """Her tokens for ``prompt``, a pass reading the last kept token and her head's draft after it.

    Yields ``(token, logprobs, from_draft)`` as mlx_lm's speculative step does. A position's processors run only
    once the positions before it are kept: they keep state (her private channel's count, her memory's last entry)."""
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    from core.brain.llm.several_tokens_a_pass import _logits, _read, _text, _unread
    from core.consciousness.spoken_positions import speaking_several

    text = _text(model)
    # Her worker's hidden-state tap, where it is in: its proxy around her backbone holds it (nonparametric_worker).
    tap = tap if tap is not None else getattr(text.model, "_tap", None)
    embed = text.model.embed_tokens
    cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    head_cache = KVCache()
    processors = list(logits_processors or [])
    sampler = sampler or (lambda logprobs: mx.argmax(logprobs, axis=-1))
    y = prompt.astype(mx.uint32)
    total, done = int(y.size), 0
    if prompt_progress_callback is not None:
        prompt_progress_callback(0, total)
    # The prompt but its last token, in steps, as mlx_lm prefills; her head reads each step beside her model, the
    # hidden state at each position with the token after it, so its own attention has the prompt.
    while total - done > 1:
        n = min(int(prefill_step_size or 512), total - done - 1)
        read = text.model(y[done:done + n][None], cache)
        head(read, embed(y[done + 1:done + n + 1][None]), head_cache)
        mx.eval([c.state for c in cache] + [c.state for c in [head_cache]])
        done += n
        if prompt_progress_callback is not None:
            prompt_progress_callback(done, total)
    history = y[done:]  # what mlx_lm hands its processors first: the prompt's tail
    hiddens: Any = None

    def choose(row: mx.array, at: int) -> tuple[int, mx.array]:
        """Her token at a position: her processors on its logits (shape [1, vocab]), then her sampler."""
        if tap is not None and hiddens is not None:
            tap.last_hidden = hiddens[at]
        for processor in processors:
            row = processor(history, row)
        logprobs = row - mx.logsumexp(row, axis=-1, keepdims=True)
        chosen = sampler(logprobs)
        mx.eval(chosen)
        return int(chosen.reshape(-1)[0].item()), logprobs.squeeze(0)

    def kept(token: int) -> None:
        nonlocal history
        history = mx.concatenate([history, mx.array([token], dtype=history.dtype)])

    def drafted_after(hidden: mx.array, following: list[int]) -> int:
        out = head(hidden, embed(mx.array([following])), head_cache)
        return int(mx.argmax(_logits(text, out[:, -1, :]), axis=-1).item())

    # First pass: the prompt's last token gives her first token, and her head its first draft.
    hidden = text.model(history[None], cache)
    hiddens = hidden[0] if tap is not None else None
    token, logprobs = choose(_logits(text, hidden[:, -1, :]), -1)
    draft = drafted_after(hidden[:, -1:], [token])
    from_draft, made = False, 0
    while True:
        yield token, logprobs, from_draft
        made += 1
        if 0 <= max_tokens <= made:
            return
        kept(token)
        with speaking_several():  # both are tokens she speaks: steered, and sampled for Φ, as one at a time
            read, saved = _read(text, mx.array([[token, draft]]), cache)
        hiddens = read[0] if tap is not None else None
        logits = _logits(text, read)
        checked, checked_logprobs = choose(logits[:, 0, :], 0)
        if checked != draft:
            _unread(cache, saved, 2, 1)
            token, logprobs, from_draft = checked, checked_logprobs, False
            draft = drafted_after(read[:, :1], [token])
            continue
        # Kept: her token after the draft comes from the same pass.
        yield draft, checked_logprobs, True
        made += 1
        if 0 <= max_tokens <= made:
            return
        kept(draft)
        token, logprobs = choose(logits[:, 1, :], 1)
        from_draft = False
        draft = drafted_after(read, [draft, token])


def stream_generate_with_her_head(model: Any, tokenizer: Any, prompt: Any, max_tokens: int = 256,
                                  draft_model: Any = None, **kwargs: Any) -> Generator[Any, None, None]:
    """mlx_lm's ``stream_generate`` where ``draft_model`` is her head; mlx_lm's own for anything else."""
    from mlx_lm import stream_generate
    from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from core.brain.llm.two_tokens_a_step import MTPHead

    if not isinstance(draft_model, MTPHead):
        yield from stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, draft_model=draft_model, **kwargs)
        return
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
            prompt = tokenizer.encode(prompt, add_special_tokens=special)
        prompt = mx.array(prompt)
    for unused in ("num_draft_tokens", "max_kv_size"):
        kwargs.pop(unused, None)
    detokenizer = tokenizer.detokenizer
    tic = time.perf_counter()
    prompt_tps, n, token, logprobs, from_draft = 0.0, -1, 0, None, False
    # Her weights held in memory for the generation, as mlx_lm's own stream holds them: unwired, they could be paged
    # out between passes while her app's other processes ran (MEASURED 20 October 2026: 9.9 tokens a second in her
    # worker, 21.5 by itself; prompts read at 145 a second against 390).
    with wired_limit(model, [generation_stream]):
        for n, (token, logprobs, from_draft) in enumerate(head_generate_step(prompt, model, draft_model,
                                                                              max_tokens=max_tokens, **kwargs)):
            if n == 0:
                prompt_tps = prompt.size / max(1e-9, time.perf_counter() - tic)
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break
            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break
            yield GenerationResponse(text=detokenizer.last_segment, token=token, logprobs=logprobs, from_draft=from_draft,
                                     prompt_tokens=prompt.size, prompt_tps=prompt_tps, generation_tokens=n + 1,
                                     generation_tps=(n + 1) / max(1e-9, time.perf_counter() - tic),
                                     peak_memory=mx.get_peak_memory() / 1e9, finish_reason=None)
    detokenizer.finalize()
    yield GenerationResponse(text=detokenizer.last_segment, token=token, logprobs=logprobs, from_draft=from_draft,
                             prompt_tokens=prompt.size, prompt_tps=prompt_tps, generation_tokens=n + 1,
                             generation_tps=(n + 1) / max(1e-9, time.perf_counter() - tic),
                             peak_memory=mx.get_peak_memory() / 1e9,
                             finish_reason="stop" if token in tokenizer.eos_token_ids else "length")


def timed(responses: Any, each: Callable[[], None]) -> Generator[Any, None, None]:
    """``responses`` passed on with ``each`` called as each comes, and at the end how long the generation waited on the
    model and how long the token loop around it took. LIVE 20 October 2026 her worker decoded at 8 tokens a second
    where her model, every channel and her head's drafts decode at 21.5 by themselves."""
    waited = looped = 0.0
    count = 0
    asked = time.perf_counter()
    try:
        for response in responses:
            came = time.perf_counter()
            waited, count = waited + came - asked, count + 1
            each()
            yield response
            asked = time.perf_counter()
            looped += asked - came
    finally:
        if count > 1:
            logger.info("⏱️ [WORKER] %d tokens: %.2fs waiting on the model, %.2fs in the token loop around it",
                        count, waited, looped)

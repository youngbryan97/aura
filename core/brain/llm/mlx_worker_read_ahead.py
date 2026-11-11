"""Read text into her model's cache without answering: the read-ahead of her voice mode's live context.

A voice session keeps one conversation for her model and only ever adds to it (core/voice/duplex/live_context.py).
While a person is still speaking, the turn's evidence is read here, so when they stop only their words are new. Proved
on her cortex on 11 November 2026 (tools/voice/prove_a_live_context.py): first token 0.32 s after the words against
5.4 to 6.7 s with the prompt rebuilt.

The text is tokenized as a generation tokenizes its prompt, and kept under the scope the job names, which the
generation that follows names too; that is what lets it find this entry as the exact prefix of its own prompt.
"""
from __future__ import annotations

import time
from typing import Any


def read_ahead(job: dict[str, Any], *, model: Any, tokenizer: Any, model_path: Any, prompt_cache_lru: Any,
               ipc_writer: Any, metal_semaphore: Any, mx: Any) -> None:
    """Extend the nearest cached prefix of ``job["prompt"]`` to all of it, keep it, and say how much was read."""
    from core.brain.llm.mlx_worker import (
        _prompt_cache_model_key,
        _prompt_cache_scope_for_job,
        _runtime_prefill_step_size,
    )

    reply: dict[str, Any] = {"id": job.get("id"), "action": "read_ahead"}
    started = time.perf_counter()
    try:
        prompt = str(job.get("prompt") or "")
        if not prompt or prompt_cache_lru is None:
            raise ValueError("nothing to read" if not prompt else "this worker keeps no prompt cache")
        tokens = list(tokenizer.encode(prompt))
        model_key = (_prompt_cache_model_key(model_path), _prompt_cache_scope_for_job(job))
        if prompt_cache_lru._holds(model_key, tokens):
            # Read already. Feeding the same tokens again would put them in the cache twice.
            reply.update(status="ok", held=len(tokens), read=0, seconds=0.0)
            ipc_writer.put(reply)
            return
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

        cache, remaining = prompt_cache_lru.fetch_nearest_cache(
            model_key, tokens, can_trim_prompt_cache=can_trim_prompt_cache, trim_prompt_cache=trim_prompt_cache,
            describe=getattr(tokenizer, "decode", None))
        if cache is None:
            cache, remaining = make_prompt_cache(model), tokens
        step = max(1, int(_runtime_prefill_step_size(model_path)))
        with metal_semaphore:
            for start in range(0, len(remaining), step):
                model(mx.array(remaining[start:start + step])[None], cache=cache)
                mx.eval([entry.state for entry in cache])
        prompt_cache_lru.insert_cache(model_key, tokens, cache)
        reply.update(status="ok", held=len(tokens), read=len(remaining),
                     seconds=round(time.perf_counter() - started, 3))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        reply.update(status="error", error=f"{type(exc).__name__}: {exc}")
    ipc_writer.put(reply)

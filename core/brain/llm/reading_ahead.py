"""Reading the conversation before the next turn needs it read.

Her model reads a prompt at some 340 tokens a second, and what it has read of a conversation is kept to go on from in two ways:
the very state her last reply was written in, held four minutes and only where the bytes delivered are the bytes she wrote; and
a prefix kept on the way past it. LIVE 25 November 2026, 02:47: a reply had a sentence taken out of it, so the state was let go,
and the prefix kept was one turn old, so the next turn read 4,115 tokens in 12.4 s before her first word. The conversation it
read had stood there since the turn before, with her worker idle.

Between turns the worker reads the conversation as the next turn will render it, up to where that turn's own part starts, and
keeps it. A turn that comes meanwhile stops it, and what was read by then is kept.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from core.runtime.errors import record_degradation

__all__ = ["as_a_job_of_the_worker", "read_through", "the_talk_so_far"]

logger = logging.getLogger("MLXWorker")

#: The turns that could come next, told apart from one another wherever they start: with readings, and without.
_COULD_COME = (
    [{"role": "user", "content": "A"}],
    [{"role": "user", "content": "Z"}],
    [{"role": "runtime_evidence", "content": "A"}, {"role": "user", "content": "A"}],
    [{"role": "runtime_evidence", "content": "Z"}, {"role": "user", "content": "Z"}],
)
_A_MESSAGE_STARTS = ("<|im_start|>", "<|start_header_id|>", "<|start_of_turn|>", "<|turn>")


def the_talk_so_far(messages: Sequence[dict[str, Any]], render: Callable[[list[dict[str, Any]]], str]) -> str:
    """The conversation as any next turn renders it, up to where that turn's own part starts; nothing where that cannot be told."""
    rendered = [render([*messages, *following]) for following in _COULD_COME]
    shared = os.path.commonprefix(rendered)
    cut = max(shared.rfind(mark) for mark in _A_MESSAGE_STARTS)
    return shared[:cut] if cut > 0 else ""


def read_through(
    tokens: list[int], *, lru: Any, model_key: Any, make_cache: Callable[[], Any], forward: Callable[[list[int], Any], None],
    step: int, stopped: Callable[[], bool], can_trim: Callable[[Any], bool], trim: Callable[[Any, int], Any],
) -> tuple[int, int]:
    """Read ``tokens`` into the cache from where it already reaches, and keep it: how many were read, and how many were held."""
    if lru._holds(model_key, tokens):
        return 0, len(tokens)
    cache, left = lru.fetch_nearest_cache(model_key, tokens, can_trim_prompt_cache=can_trim, trim_prompt_cache=trim)
    if cache is None:
        cache, left = make_cache(), tokens
    held = reached = len(tokens) - len(left)
    for at in range(0, len(left), max(1, step)):
        if stopped():
            break
        part = list(left[at:at + step])
        forward(part, cache)
        reached += len(part)
    if reached > held:
        lru.insert_cache(model_key, list(tokens[:reached]), cache)
    return reached - held, held


@contextlib.contextmanager
def _steered_as_her_replies_are(engine: Any, model: Any, put_back: dict[str, bool]) -> Iterator[None]:
    """What is read for her replies is read as they are written, with the steering they have; whether it was put back after."""
    from core.brain.llm.mlx_worker_surface_quality import (
        _apply_surface_generation_controls,
        _restore_surface_generation_controls,
    )

    state = _apply_surface_generation_controls(engine, model, {"clean_user_surface_contract": True})
    try:
        yield
    finally:
        put_back["steering"] = bool(_restore_surface_generation_controls(state))


def as_a_job_of_the_worker(
    job: Any, model: Any, tokenizer: Any, engine: Any, metal: Any, *, lru: Any, model_key: Any, step: int,
    stopped: Callable[[], bool], alive: Callable[[], None],
) -> tuple[dict[str, Any], bool]:
    """The reading as a job of the worker that owns the model. The answer is JSON in the place a generation's text goes: how many
    tokens were read, and how many had been already. And whether the worker is as it was before."""
    response: dict[str, Any] = {
        "id": job.get("id"), "action": "generate", "status": "ok", "text": "[0, 0]", "tokens_used": 0,
        "prompt_tokenization": {"chars": 0, "tokens": 0, "generated_tokens": 0}, "generation_performance": {},
    }
    put_back = {"steering": True}
    messages = [m for m in job.get("messages") or () if isinstance(m, dict)]
    if lru is None or not messages:
        return response, True
    try:
        import mlx.core as mx
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

        from core.brain.llm.chat_format import render_chat_template

        text = the_talk_so_far(messages, lambda all_of_it: render_chat_template(tokenizer, all_of_it, enable_thinking=False))
        tokens = list(tokenizer.encode(text)) if text else []
        if not tokens:
            return response, True

        def forward(part: list[int], cache: Any) -> None:
            model(mx.array([part]), cache=cache)
            mx.eval([layer.state for layer in cache])
            mx.clear_cache()
            alive()

        with metal, _steered_as_her_replies_are(engine, model, put_back):
            read, held = read_through(
                tokens, lru=lru, model_key=model_key, make_cache=lambda: make_prompt_cache(model), forward=forward,
                step=step, stopped=stopped, can_trim=can_trim_prompt_cache, trim=trim_prompt_cache,
            )
        logger.info("📖 [WORKER] Read %d tokens of the talk so far for the turn to come; %d were read already.", read, held)
        response["text"] = json.dumps([read, held])
    except (ImportError, AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        response.update({"status": "error", "message": f"{type(exc).__name__}: {exc}"})
    return response, put_back["steering"]


class _ReadsAhead:
    """The client's side of the job: a part of MLXLocalClient, kept here with the reading it asks for."""

    async def read_ahead(self, messages: Sequence[dict[str, Any]]) -> tuple[int, int]:
        """Have the worker read the conversation ahead of its next turn: how much it read, and how much it held."""
        if not messages:
            return 0, 0
        try:
            raw = await self.generate_text_async(  # type: ignore[attr-defined]
                str(messages[-1].get("content") or "")[-400:], read_ahead=True, messages=[dict(message) for message in messages], max_tokens=1, temperature=0.0,
                origin="reading_ahead", internal_inference=True, foreground_request=True,
            )
            read = json.loads(str(raw or "[0, 0]"))
        except (RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            record_degradation("reading_ahead", exc, severity="warning",
                               action="the next turn reads the conversation for itself")
            return 0, 0
        return (int(read[0]), int(read[1])) if isinstance(read, list) and len(read) == 2 else (0, 0)

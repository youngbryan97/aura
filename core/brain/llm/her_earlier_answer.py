"""While she answers a question she has answered before in this conversation, her decoder does not read that answer back.

LIVE 13 November 2026, 15:12: asked again what she had been up to, her draft began with the whole first sentence of
her answer from 14:50, which sat in her transcript, and her two later drafts were that answer again; her worker
refused both, and the first draft was served. Refusing after the fact costs a generation each time and the next draft
copies again, because the copy is the likeliest continuation. So the decoder holds it: no run of ``RUN`` tokens from
her earlier answer to the same question (core/conversation/asked_again.py) is continued. Shorter phrases stay hers,
so she can still say what she said in other words, or say it was the same. A request to say it again sends no answer.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from core.brain.llm.mlx_worker_surface_quality import A_RUN_OF_WORDS

logger = logging.getLogger(__name__)

#: The length of run a repeat is compared by elsewhere (mlx_worker_surface_quality), here in tokens.
RUN = A_RUN_OF_WORDS


def _encoded(tokenizer: Any, text: str) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return [int(i) for i in ids]


def runs_to_leave(tokenizer: Any, answers: Iterable[str]) -> dict[tuple[int, ...], frozenset[int]]:
    """For each ``RUN - 1`` tokens of her earlier answers, the tokens that would continue them into that answer."""
    following: dict[tuple[int, ...], set[int]] = {}
    for answer in answers:
        ids = _encoded(tokenizer, str(answer or ""))
        for at in range(len(ids) - RUN + 1):
            following.setdefault(tuple(ids[at:at + RUN - 1]), set()).add(ids[at + RUN - 1])
    return {start: frozenset(nexts) for start, nexts in following.items()}


def not_read_back(tokenizer: Any, answers: Iterable[str]) -> Callable[[Any, Any], Any] | None:
    """A logits processor under which no run of her earlier answers is continued; None with nothing to hold."""
    try:
        import mlx.core as mx
    # not a failure: no MLX here, so there is no decoder to hold.
    except ImportError:
        return None
    following = runs_to_leave(tokenizer, answers)
    if not following:
        return None
    held = {"tokens": 0}

    def leave_it(tokens: Any, logits: Any) -> Any:
        tail = tokens[-(RUN - 1):]
        tail = tuple(int(t) for t in (tail.tolist() if hasattr(tail, "tolist") else tail))
        nexts = following.get(tail)
        width = int(logits.shape[-1])
        banned = sorted(i for i in nexts or () if 0 <= i < width)
        if not banned:
            return logits
        held["tokens"] += 1
        mask = mx.zeros((width,), dtype=logits.dtype)
        mask[mx.array(banned)] = -mx.inf
        return logits + mask

    leave_it.held = held  # type: ignore[attr-defined]
    return leave_it


def install(tokenizer: Any, job: dict[str, Any], processors: list[Any]) -> int:
    """Append the processor for this job's earlier answers; how many answers it holds."""
    answers = [str(a) for a in job.get("her_answers_when_asked_this") or () if str(a or "").strip()]
    processor = not_read_back(tokenizer, answers) if answers else None
    if processor is None:
        return 0
    processors.append(processor)
    logger.info("🔁 [WORKER] Asked again: her %d earlier answer(s) to this are not read back.", len(answers))
    return len(answers)

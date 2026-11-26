"""On the pass of her tool loop that writes the answer, her decoder does not open a tool call.

LIVE 18 November 2026: her loop offered her no tools on its last pass, and she wrote one more call anyway, because the
calls she had already made sat just above in her conversation. The worker rejected it and the person got no answer. The
tools she may call are a fact about the pass, so the decoder holds it: the tokens that open a call cannot be chosen.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: How her native tool protocol opens a call (core/brain/llm/mlx_client.py reads both forms).
_A_CALL_OPENS_WITH = ("<tool_call>", "<function=")


def _encoded(tokenizer: Any, text: str) -> tuple[int, ...]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return tuple(int(i) for i in ids)


def tokens_that_open_a_call(tokenizer: Any) -> dict[tuple[int, ...], frozenset[int]]:
    """For each run of tokens that leads into an opener, the token that would finish it. The run is empty for an opener
    that is one token."""
    finishing: dict[tuple[int, ...], set[int]] = {}
    for opener in _A_CALL_OPENS_WITH:
        ids = _encoded(tokenizer, opener)
        if ids:
            finishing.setdefault(ids[:-1], set()).add(ids[-1])
    return {run: frozenset(last) for run, last in finishing.items()}


def no_call(tokenizer: Any) -> Callable[[Any, Any], Any] | None:
    """A logits processor under which no call is opened; None where there is no decoder to hold."""
    try:
        import mlx.core as mx
    # not a failure: no MLX here, so there is no decoder to hold.
    except ImportError:
        return None
    finishing = tokens_that_open_a_call(tokenizer)
    if not finishing:
        return None
    longest = max(len(run) for run in finishing)

    def hold(tokens: Any, logits: Any) -> Any:
        tail = tokens[-longest:] if longest else []
        tail = [int(t) for t in (tail.tolist() if hasattr(tail, "tolist") else tail)]
        width = int(logits.shape[-1])
        banned = sorted({token for run, last in finishing.items() if not run or tail[-len(run):] == list(run)
                         for token in last if 0 <= token < width})
        if not banned:
            return logits
        mask = mx.zeros((width,), dtype=logits.dtype)
        mask[mx.array(banned)] = -mx.inf
        return logits + mask

    return hold


def writes_her_reply(job: dict[str, Any]) -> bool:
    """Whether this pass writes words and can run no call: the tool loop's last pass, or the pass that writes her reply
    to the person with no tool offered. LIVE 17 November 2026 she called code_repl from her reply pass, which no loop
    runs, and the worker rejected the call three times as prose before anything reached the person."""
    if job.get("writes_no_call") is True:
        return True
    # Only where the job says it offers nothing: a job built without saying is not taken to offer nothing.
    return bool(job.get("clean_user_surface_contract")) and job.get("offers_tools") is False


def install(tokenizer: Any, job: dict[str, Any], processors: list[Any]) -> bool:
    """Append the processor when this job is a pass that writes; whether it did."""
    processor = no_call(tokenizer) if writes_her_reply(job) else None
    if processor is None:
        return False
    processors.append(processor)
    logger.info("[WORKER] This pass writes her answer: no tool call can be opened.")
    return True

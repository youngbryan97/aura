"""One of the choices on offer out of the decoder, because the decoder can produce nothing else.

A decision between the moves on a screen was asked of her model as "name the
move you choose and say why in one sentence", with room for ninety-six tokens.
The move comes first, so the choice was settled by the first few tokens, and
the rest explained a choice already made. On the live 27B that explanation is
most of the cost: LIVE 2026-10-08 a prompt of 2,522 tokens read in 10.1 s and
99 tokens written in 6.9 s. A game wanted its move in eight seconds, so the
question was not asked at all and the move was made without her.

This holds her answer to the choices themselves. At each step the decoder may
take only a token that carries on one of the choices as the tokenizer writes
it, and once a choice is whole the only token left is the end of the turn. The
weighing is still her model's, under the same prompt and the same sampling:
only the tokens that begin no choice are taken away. The answer is as long as
the name of the move, a handful of tokens.

A generation with a private channel is left alone until the channel closes, as
the JSON shape is (core/brain/llm/a_shape_the_decoder_enforces.py).
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger("Aura.Brain.LLM.Choice")

__all__ = ["enforce_one_of", "the_ways_through"]

#: The most choices one answer is held to: a screen's moves, not a vocabulary.
MOST_CHOICES = 64


def _encoded(tokenizer: Any, text: str) -> tuple[int, ...]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return tuple(int(i) for i in ids)


def _ends(tokenizer: Any) -> frozenset[int]:
    declared = getattr(tokenizer, "eos_token_ids", None)
    ends = {declared} if type(declared) is int else set(declared or ())
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ends.add(eos)
    return frozenset(int(i) for i in ends if isinstance(i, int))


def the_ways_through(tokenizer: Any, choices: Sequence[str]) -> tuple[dict[tuple[int, ...], frozenset[int]], frozenset[tuple[int, ...]]]:
    """For each start of a choice as tokens, the tokens that may follow it; and the choices whole."""
    following: dict[tuple[int, ...], set[int]] = {}
    whole: set[tuple[int, ...]] = set()
    for choice in list(dict.fromkeys(" ".join(str(c or "").split()) for c in choices))[:MOST_CHOICES]:
        ids = _encoded(tokenizer, choice) if choice else ()
        if not ids:
            continue
        whole.add(ids)
        for at in range(len(ids)):
            following.setdefault(ids[:at], set()).add(ids[at])
    return {start: frozenset(nexts) for start, nexts in following.items()}, frozenset(whole)


def enforce_one_of(tokenizer: Any, choices: Sequence[str], *, after_token: int | None = None) -> Callable[[Any, Any], Any] | None:
    """A logits processor under which the decoder can only write one of ``choices``, whole, and then end.

    Returns None where MLX is not importable, where no choice has any tokens, or
    where the tokenizer names no end of turn, so a caller knows the choice is
    not held rather than assuming it is.
    """
    try:
        import mlx.core as mx
    # not a failure: no MLX here, so there is no decoder to hold.
    except ImportError:
        return None
    following, whole = the_ways_through(tokenizer, choices)
    ends = _ends(tokenizer)
    if not whole or not ends:
        return None
    state: dict[str, Any] = {"prompt": None, "from": None if after_token is not None else -1, "lost": False}

    def hold_the_choice(tokens: Any, logits: Any) -> Any:
        token_ids = [int(t) for t in (tokens.tolist() if hasattr(tokens, "tolist") else list(tokens))]
        # The first call carries the prompt (or what of it was not cached); the answer is what comes after.
        if state["prompt"] is None:
            state["prompt"] = len(token_ids)
        if state["lost"]:
            return logits
        if state["from"] is None:
            answered = token_ids[state["prompt"]:]
            if after_token not in answered:
                return logits
            state["from"] = state["prompt"] + answered.index(after_token) + 1
        begun = state["prompt"] if state["from"] == -1 else state["from"]
        so_far = tuple(token_ids[begun:])
        allowed = set(following.get(so_far, ()))
        if so_far in whole:
            allowed |= ends
        width = int(logits.shape[-1])
        allowed = {i for i in allowed if 0 <= i < width}
        if not allowed:
            # A token this did not allow got through (a sampler bypass, a rewind): the choice is no longer claimed.
            state["lost"] = True
            logger.warning("🧠 [WORKER] Choice lost after %d token(s); no longer held.", len(so_far))
            return logits
        mask = mx.full((width,), -mx.inf, dtype=logits.dtype)
        mask[mx.array(sorted(allowed))] = 0.0
        return logits + mask

    hold_the_choice.state = state  # type: ignore[attr-defined]
    return hold_the_choice

"""A reply does not copy her own earlier replies, held by the decoder.

Live, 13 October, three different invitations to watch something were each
answered "I am ready. I have my eyes closed, but my attention is fully open to
your narration. Please begin. Is the water still or moving? And where is the
heron positioned?" Her model, handed a transcript in which it had answered a
message like this one, wrote its earlier answer out again nearly token for
token. A penalty on repeated tokens cannot see that: every token of the copy
is ordinary, and only the run of them is a repeat.

DRY (Weidmann 2024) penalises the run. When the text being written repeats a
stretch of earlier text, the token that would carry the repeat on is pushed
down by an amount that grows with the stretch,

    multiplier * base ** (length - allowed_length),

so a shared word or two costs nothing and a copied sentence soon costs more
than any continuation of it is worth. The three numbers are DRY's own defaults
as llama.cpp and the other runtimes that adopted it ship them; they were not
tuned here. So are its sequence breakers: a stretch is counted only since the
last new line, colon, quotation mark or asterisk, and those are never pushed
down. Without them a list or a worked derivation given again, item after
item, ran up a penalty with no ceiling; live on 16 October, asked to sort the
same six words a third time, she doubted "ember" out loud before writing it.

Only her own earlier turns count. Quoting what somebody said to her, a page,
or the screen is not repeating herself, and her earlier turns are read from
the job's messages rather than from the token history, because a cached
prefix never reaches a logits processor at all.

A copy is a run of words, so only a token that starts a word is pushed down.
A token that carries on a word already begun is left alone: her tokenizer
writes a number one digit at a time, and live on 16 October, asked a greatest
common divisor of 4862 and 7854 a second time, she wrote "486" and "785",
because the digits after the first two were the run of her earlier answer.
Cutting a word short does not avoid a copy; it writes a different word.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger("Aura.Brain.LLM.NotAgain")

__all__ = ["ALLOWED_LENGTH", "BASE", "MULTIPLIER", "SEQUENCE_BREAKERS", "her_earlier_turns", "not_her_own_words_again"]

#: DRY's defaults (Weidmann 2024; llama.cpp `--dry-multiplier 0.8 --dry-base
#: 1.75 --dry-allowed-length 2`).
MULTIPLIER = 0.8
BASE = 1.75
ALLOWED_LENGTH = 2
#: DRY's default sequence breakers (llama.cpp `--dry-sequence-breaker`, Weidmann 2024).
SEQUENCE_BREAKERS = ("\n", ":", '"', "*")

#: Past this the penalty is far beyond any logit, and the exponent stays finite.
_LONGEST = 64


def her_earlier_turns(messages: Any) -> list[str]:
    """What she said in the transcript, before the turn being written now."""
    said: list[str] = []
    for message in messages if isinstance(messages, list) else []:
        if isinstance(message, dict) and str(message.get("role") or "").lower() == "assistant":
            text = str(message.get("content") or "").strip()
            if text:
                said.append(text)
    return said


def _encoded(tokenizer: Any, text: str) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return [int(i) for i in ids]


def _within_a_word(tokenizer: Any, ids: set[int]) -> tuple[set[int], set[int], set[int]]:
    """Of ``ids``: those that end mid-word, those that carry a word on, and the sequence breakers.

    All empty when the tokenizer cannot decode a token.
    """
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return set(), set(), set()
    ends_in_a_word: set[int] = set()
    carries_a_word_on: set[int] = set()
    breakers: set[int] = set()
    for token in ids:
        try:
            text = str(decode([token]))
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if text[-1:].isalnum():
            ends_in_a_word.add(token)
        if text[:1].isalnum():
            carries_a_word_on.add(token)
        if any(breaker in text for breaker in SEQUENCE_BREAKERS):
            breakers.add(token)
    return ends_in_a_word, carries_a_word_on, breakers


def not_her_own_words_again(tokenizer: Any, earlier: Sequence[str]) -> Callable[[Any, Any], Any] | None:
    """A logits processor that penalises carrying on a copy of her earlier turns; None with nothing to copy."""
    try:
        import mlx.core as mx
    # not a failure: no MLX here, so there is no decoder to hold.
    except ImportError:
        return None
    turns = [ids for ids in (_encoded(tokenizer, text) for text in earlier) if len(ids) > ALLOWED_LENGTH]
    if not turns:
        return None
    ends_in_a_word, carries_a_word_on, breakers = _within_a_word(tokenizer, {token for ids in turns for token in ids})
    # Where each run of `ALLOWED_LENGTH` tokens occurs, and what came next.
    after: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for turn, ids in enumerate(turns):
        for end in range(ALLOWED_LENGTH, len(ids)):
            after.setdefault(tuple(ids[end - ALLOWED_LENGTH:end]), []).append((turn, end))
    state: dict[str, Any] = {"prompt": None, "penalised": 0}

    def not_again(tokens: Any, logits: Any) -> Any:
        token_ids = [int(t) for t in (tokens.tolist() if hasattr(tokens, "tolist") else list(tokens))]
        # The first call carries the prompt, or what of it was not cached.
        if state["prompt"] is None:
            state["prompt"] = len(token_ids)
        written = token_ids[state["prompt"]:]
        # How much of what she has written counts: what came after the last sequence breaker.
        since = len(written)
        for back in range(len(written) - 1, -1, -1):
            if written[back] in breakers:
                since = len(written) - back - 1
                break
        if since < ALLOWED_LENGTH:
            return logits
        penalty: dict[int, float] = {}
        for turn, end in after.get(tuple(written[-ALLOWED_LENGTH:]), ()):
            ids = turns[turn]
            # How far back the copy runs, written against what she said before.
            length = ALLOWED_LENGTH
            while (
                length < min(_LONGEST, since, end)
                and written[-length - 1] == ids[end - length - 1]
            ):
                length += 1
            following = ids[end]
            if following in breakers:
                continue  # a line's end, a colon or a quotation mark is the shape of the text, not a copy
            if written[-1] in ends_in_a_word and following in carries_a_word_on:
                continue  # the word already begun is finished as written
            penalty[following] = max(penalty.get(following, 0.0), MULTIPLIER * BASE ** (length - ALLOWED_LENGTH))
        width = int(logits.shape[-1])
        penalty = {token: amount for token, amount in penalty.items() if 0 <= token < width}
        if not penalty:
            return logits
        state["penalised"] += 1
        down = mx.zeros((width,), dtype=logits.dtype)
        down[mx.array(sorted(penalty))] = mx.array([penalty[token] for token in sorted(penalty)], dtype=logits.dtype)
        return logits - down

    not_again.state = state  # type: ignore[attr-defined]
    return not_again

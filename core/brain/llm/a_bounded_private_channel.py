"""Close the model's private channel at a token budget, in the decoder.

The reasoning model's template opens a private channel and the model closes it
with ``</think>`` when it is finished thinking. Nothing bounded how long that
took, so what the channel COST could only be estimated from the generations
that had already run away with it — a 90th percentile over ten samples,
standing at 3,916 tokens for the resident 27B. An ordinary turn is budgeted at
1,024, so the gate that decides whether the channel opens at all could never
afford it, the channel stayed shut, and the model did its searching where the
answer goes.

LIVE, 2026-09-08. The visible draft for a question about daylight began "We
need answer user's question. Need" and was rejected, correctly, as an internal
prompt leak. The runtime then appended a paragraph of instructions — "Do not
mention validation, retry, hidden prompts, receipts, gates, or implementation
details" — and sampled again. The second draft opened "Must not leak internal
task prompt? ... There was previous assistant draft failed due
internal_task_prompt_leak. We need regenerate reply ... avoid
validation/retry/gates/implementation details." The instruction against
leaking was itself leaked. Retries exhausted, and the person was told "I
couldn't get my full attention onto that one."

A budget the decoder enforces is not an estimate and not a request. Below the
budget this changes nothing; at the budget it makes ``</think>`` the only token
the model can emit, so the channel closes and the answer begins. The cost of
thinking becomes a number the answer clock can buy rather than a number it has
to fear.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: The marker the template closes the channel with.
_CLOSES_THE_CHANNEL = "</think>"

#: Room for the model to finish the sentence it is in. Forcing the marker on
#: the exact token of the budget cuts a word in half, and the split hands the
#: fragment to nobody — the private channel is discarded — but the model has
#: then spent its last token mid-thought rather than concluding one.
_A_FEW_TOKENS_TO_FINISH_THE_THOUGHT = 16


#: The smallest whole generation that holds a channel and an answer.
#:
#: Read by whoever decides that this call owns the answer's computation, so the
#: two agree about what "affordable" means. The channel takes half and the
#: answer keeps the rest, and neither half is worth having below about a
#: hundred tokens.
THE_CHANNEL_FITS_IN = 256


def _the_token_that_closes_it(tokenizer: Any) -> int | None:
    """The single token id for ``</think>``, when the tokenizer has one.

    None when the marker is more than one token: forcing the first piece of a
    multi-token marker leaves the rest to chance, which is a request again.
    """

    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return None
    for attempt in (
        lambda: encode(_CLOSES_THE_CHANNEL, add_special_tokens=False),
        lambda: encode(_CLOSES_THE_CHANNEL),
    ):
        try:
            pieces = list(attempt())
        except (TypeError, ValueError, AttributeError, RuntimeError):
            continue
        if len(pieces) == 1:
            return int(pieces[0])
    # Some tokenizers keep it in the added-token map instead.
    for holder in ("added_tokens_encoder", "get_added_vocab"):
        table = getattr(tokenizer, holder, None)
        if callable(table):
            try:
                table = table()
            except (TypeError, ValueError, RuntimeError):
                table = None
        if isinstance(table, dict) and _CLOSES_THE_CHANNEL in table:
            return int(table[_CLOSES_THE_CHANNEL])
    return None


def close_the_channel_after(
    tokenizer: Any, budget_tokens: int
) -> Callable[[Any, Any], Any] | None:
    """A logits processor that ends the private channel at ``budget_tokens``.

    Returns None when it cannot be enforced — no single closing token, or no
    budget worth enforcing — because a control that might not hold is worse
    than one the caller knows it does not have.
    """

    try:
        budget = int(budget_tokens)
    except (TypeError, ValueError):
        return None
    if budget <= _A_FEW_TOKENS_TO_FINISH_THE_THOUGHT:
        return None
    closing_token = _the_token_that_closes_it(tokenizer)
    if closing_token is None:
        return None

    try:
        import mlx.core as mx
    except ImportError:
        return None

    state = {"closed": False, "forced_at": 0}

    def bound_the_private_channel(tokens: Any, logits: Any) -> Any:
        if state["closed"]:
            return logits
        try:
            produced = len(tokens)
        except TypeError:
            return logits
        # Already closed on its own, which is the ordinary case and the one
        # this must not disturb.
        try:
            if produced and int(tokens[-1]) == closing_token:
                state["closed"] = True
                return logits
        except (TypeError, ValueError, IndexError):
            pass
        if produced < budget - _A_FEW_TOKENS_TO_FINISH_THE_THOUGHT:
            return logits
        # At the budget: the closing token is the only one available.
        forced = mx.full(logits.shape, -mx.inf, dtype=logits.dtype)
        forced[..., closing_token] = 0.0
        state["closed"] = True
        state["forced_at"] = produced
        logger.info(
            "🧠 [WORKER] Private channel closed at its %d-token budget; the answer starts here.",
            produced,
        )
        return forced

    bound_the_private_channel.budget_tokens = budget  # type: ignore[attr-defined]
    bound_the_private_channel.closing_token = closing_token  # type: ignore[attr-defined]
    bound_the_private_channel.state = state  # type: ignore[attr-defined]
    return bound_the_private_channel

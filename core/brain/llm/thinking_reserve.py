"""What the private reasoning channel costs, measured rather than assumed.

A thinking model spends its output budget twice: once on reasoning nobody
reads, then on the answer. Callers size ``max_tokens`` against the answer,
because that is the part they can picture. The worker enforces it against
everything the model emits.

LIVE, 2026-08-27: a question about a number sequence was planned at
``max_tokens=1536`` and served 1,469 characters — a shade under one character
per token, which no prose reaches. The reasoning channel had taken the rest,
and the answer stopped mid-paragraph before it reached the part the person had
actually asked for. The failure lands on hard questions only, because those are
the ones that reason.

The worker already records ``native_thinking_private_chars`` on every
generation and nothing read it. This is the reader. It holds no assumed
constants: the characters-per-token ratio comes from the generation it is
measuring, and the reserve is a percentile of what reasoning has cost so far.
With too few observations to express that percentile it reserves nothing,
because a guess dressed as a measurement is worse than an honest zero — and
the runtime generates often enough that the window fills within minutes.
"""

from __future__ import annotations

import threading
from collections import deque

#: The smallest window in which a 90th percentile is a real observation rather
#: than a restatement of the largest sample.
_ENOUGH_TO_EXPRESS_A_PERCENTILE = 10

#: How far back reserves are learned from. Long enough to cover a mix of
#: question kinds, short enough that a model swap is forgotten quickly.
_WINDOW = 128

_PERCENTILE = 0.9

_observed: deque[int] = deque(maxlen=_WINDOW)

#: The largest budget a generation has been given and still not finished
#: thinking inside. Unlike the window this is a proof rather than a sample: the
#: channel demonstrably needs more than this, so one observation is enough and
#: no percentile applies.
_proved_insufficient = 0

_lock = threading.Lock()


def record_reasoning_cost(
    *,
    reasoning_chars: int,
    surface_chars: int,
    generated_tokens: int,
) -> None:
    """Log what one generation spent on its private channel.

    The characters-per-token ratio is taken from this generation rather than
    assumed, so a change of tokenizer needs no edit here.
    """

    try:
        reasoning = max(0, int(reasoning_chars))
        surface = max(0, int(surface_chars))
        tokens = max(0, int(generated_tokens))
    except (TypeError, ValueError):
        return
    total_chars = reasoning + surface
    if total_chars <= 0 or tokens <= 0:
        return
    spent = int(round(tokens * (reasoning / total_chars)))
    with _lock:
        _observed.append(max(0, spent))


def record_budget_that_ran_out_thinking(*, budget_tokens: int) -> None:
    """A generation ended still inside the private channel.

    That is not a sample of what reasoning costs, it is a proof that reasoning
    cost more than this. It needs no percentile and no second opinion, which
    matters because the only generations that open the channel are the ones
    this reserve exists to rescue — so waiting for a window of them to
    accumulate means every one of them fails first.
    """

    global _proved_insufficient
    try:
        spent = max(0, int(budget_tokens))
    except (TypeError, ValueError):
        return
    with _lock:
        _proved_insufficient = max(_proved_insufficient, spent)


def reserve_tokens() -> int:
    """Tokens to add to an answer budget so reasoning does not eat it.

    The larger of what the window has measured and what a generation has
    already proved to be too little. The window needs enough observations to
    carry a percentile; the proof needs one.
    """

    with _lock:
        seen = sorted(_observed)
        proved = _proved_insufficient
    measured = 0
    if len(seen) >= _ENOUGH_TO_EXPRESS_A_PERCENTILE:
        index = min(len(seen) - 1, int(_PERCENTILE * len(seen)))
        measured = max(0, seen[index])
    return max(measured, proved)


def observations() -> int:
    """How many generations the reserve is learned from."""

    with _lock:
        return len(_observed)


def forget() -> None:
    """Drop what has been learned. For tests and for a model swap."""

    global _proved_insufficient
    with _lock:
        _observed.clear()
        _proved_insufficient = 0

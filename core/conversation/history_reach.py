"""How much of the conversation this turn can afford to read.

LIVE 2026-09-17, asked "Aura, what is it like to be you":

    📏 Prefill size: 10414 tokens from 44192 rendered chars (83 messages)
    ⏱️ prefill=10414 tokens/83.44s, decode=457 tokens/71.27s

Eighty-three of the hundred and fifty-five seconds went into reading a
conversation, for a question thirty-one characters long. Forty exchanges
were admitted because forty existed: recency deciding relevance, and the
one input to the prompt with no budget at all. The system prompt got one
on 2026-08-28, for the same defect measured the same way — a 96,430-character
system message serving an answer of fifty tokens — and history was left
out of it, counted as ``room_taken`` and treated as fixed.

So the rule here is the rule there, and it is not a number chosen by
hand: a turn may read for about as long as its own answer takes to write,
at rates the thinking reserve measured on this machine and kept across
restarts. The system prompt is what she cannot answer without and is
served first; history takes what is left, oldest exchanges dropped first,
so what is retained is always a contiguous suffix and no reference
resolves across a hole.

**What this deliberately does not do, and why.** The obvious mechanism is
to find the topic the turn is part of and keep that — admit the run of
exchanges that hang together and drop the rest. It was built and it was
measured, and on the live 33-exchange conversation of 2026-09-17 it found
nothing: scoring each junction by how much vocabulary the exchange before
it shares with everything after, against a null that places each term
across the conversation in proportion to how many exchanges carry it, the
z-score sat between -2.2 and +2.6 with no structure — 0.02, -0.09, 0.12,
0.13, 0.30 through the middle of it. Shared vocabulary between exchanges
in that conversation is what chance predicts, to two decimal places,
because almost all of it is function words. Lexical overlap does not find
a topic boundary in real dialogue, and three different nulls
(more-than-average, less-than-the-least, and drawn bags of words) each
failed differently before the honest one said there was nothing there.
A budget does not need to know what the turn is about.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "HistoryReach",
    "measure_reach",
]


def _exchange_chars(entry: Any) -> int:
    if isinstance(entry, dict):
        return len(str(entry.get("user") or "")) + len(str(entry.get("aura") or ""))
    return len(str(entry or ""))


@dataclass
class HistoryReach:
    """What this turn can afford, and what it leaves behind."""

    #: Index of the oldest retained exchange. Everything from here on is kept.
    boundary: int
    #: Number of exchanges retained.
    retained: int
    #: Exchanges older than the boundary, not shown to the model.
    dropped: int = 0
    #: Characters retained, and the budget they were fitted to.
    kept_chars: int = 0
    budget_chars: int = 0
    #: Why the boundary landed where it did — for the turn record.
    reason: str = ""

    @property
    def cuts_anything(self) -> bool:
        return self.dropped > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "retained": self.retained,
            "dropped": self.dropped,
            "kept_chars": self.kept_chars,
            "budget_chars": self.budget_chars,
            "reason": self.reason,
        }


def measure_reach(
    exchanges: Sequence[Any], *, budget_chars: int = 0
) -> HistoryReach:
    """Fit ``exchanges`` to ``budget_chars``, dropping the oldest first.

    A budget of zero means nothing was timed, so there is nothing for the
    reading to be proportionate to. The conversation goes through whole
    rather than cut by a guess, which is the same answer
    :func:`core.brain.llm.context_budget.budget_for_answer` gives.
    """

    count = len(exchanges)
    sizes = [_exchange_chars(entry) for entry in exchanges]
    total = sum(sizes)
    if count == 0:
        return HistoryReach(boundary=0, retained=0, reason="no delivered exchanges")
    if budget_chars <= 0:
        return HistoryReach(
            boundary=0,
            retained=count,
            kept_chars=total,
            reason="nothing timed yet, so no budget to be proportionate to",
        )
    if total <= budget_chars:
        return HistoryReach(
            boundary=0,
            retained=count,
            kept_chars=total,
            budget_chars=budget_chars,
            reason=(
                f"the whole conversation fits: {total} of {budget_chars} chars"
            ),
        )

    # The last completed exchange is the adjacency pair this turn answers
    # into. Cutting what was just said is never the right saving, so it is
    # retained even when it alone is over budget.
    boundary = count - 1
    kept = sizes[-1]
    for index in range(count - 2, -1, -1):
        if kept + sizes[index] > budget_chars:
            break
        kept += sizes[index]
        boundary = index

    dropped = boundary
    return HistoryReach(
        boundary=boundary,
        retained=count - boundary,
        dropped=dropped,
        kept_chars=kept,
        budget_chars=budget_chars,
        reason=(
            f"an answer this long affords {budget_chars} chars of reading; "
            f"{count - boundary} of {count} exchanges fit in {kept}"
        ),
    )

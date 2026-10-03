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

**What meaning adds, measured 2026-10-02.** A budget still reads everything
it can afford, related or not, and on the desktop path nothing was timed, so
there was no budget at all. LIVE 20:26, "Bam 83 point game or Kobe 81 point
game?" reached all 40 exchanges about her code and a personality test; the
cortex read 9,935 tokens of them in 32.6s and the fallback read 18,137 in
84.8s, for an answer of two words. The judge in
``core/cognition/evidence_relevance.py`` is not the lexical overlap that
failed above: it compares meaning with the local encoder, at a boundary
calibrated on matched and mismatched query-passage pairs. Over that
conversation's 130 requests it placed the Bam question at 0.341 against every
earlier exchange, "how do you feel about getting tested like this" at 0.56
against the test runs, and 0.53 is the closest a request about her code came
to a test run. So a request also reaches back only as far as the oldest
earlier exchange the judge says it bears on. That can only shorten what a
budget allows, never lengthen it, and the suffix stays contiguous.
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


def _exchange_text(entry: Any) -> str:
    if isinstance(entry, dict):
        return f"{entry.get('user') or ''}\n{entry.get('aura') or ''}"
    return str(entry or "")


def _oldest_exchange_it_bears_on(exchanges: Sequence[Any], request: str) -> tuple[int, str]:
    """Index of the oldest exchange before the last that ``request`` bears on.

    The last exchange is the adjacency pair the turn answers into and is never
    judged; with nothing earlier bearing on the request, the reach is the last
    exchange alone.
    """
    from core.cognition.evidence_relevance import assess_evidence_alignments

    earlier = exchanges[:-1]
    verdicts = assess_evidence_alignments(request, [_exchange_text(entry) for entry in earlier])
    bearing = [index for index, verdict in enumerate(verdicts) if verdict.relevant]
    how = "by meaning" if verdicts and all(verdict.measured for verdict in verdicts) else "by shared words"
    if not bearing:
        return len(exchanges) - 1, f"no earlier exchange bears on this request ({how})"
    oldest = min(bearing)
    return oldest, f"{len(bearing)} earlier exchange(s) bear on this request ({how}), the oldest {len(earlier) - oldest} back"


def measure_reach(
    exchanges: Sequence[Any], *, budget_chars: int = 0, request: str = ""
) -> HistoryReach:
    """Fit ``exchanges`` to ``budget_chars`` and to what ``request`` bears on.

    A budget of zero means nothing was timed, so there is nothing for the
    reading to be proportionate to; only meaning then shortens the reach.
    Without a request either, the conversation goes through whole rather
    than cut by a guess, which is the same answer
    :func:`core.brain.llm.context_budget.budget_for_answer` gives.
    """

    count = len(exchanges)
    sizes = [_exchange_chars(entry) for entry in exchanges]
    total = sum(sizes)
    if count == 0:
        return HistoryReach(boundary=0, retained=0, reason="no delivered exchanges")

    boundary = 0
    reasons = []
    if budget_chars <= 0:
        reasons.append("nothing timed yet, so no budget to be proportionate to")
    elif total <= budget_chars:
        reasons.append(f"the whole conversation fits: {total} of {budget_chars} chars")
    else:
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
        reasons.append(
            f"an answer this long affords {budget_chars} chars of reading; "
            f"{count - boundary} of {count} exchanges fit in {kept}"
        )
    if str(request or "").strip() and count > 1:
        bears_on, said = _oldest_exchange_it_bears_on(exchanges, str(request))
        reasons.append(said)
        boundary = max(boundary, bears_on)

    return HistoryReach(
        boundary=boundary,
        retained=count - boundary,
        dropped=boundary,
        kept_chars=sum(sizes[boundary:]),
        budget_chars=max(0, budget_chars),
        reason="; ".join(reasons),
    )

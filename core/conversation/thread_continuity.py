"""Does the reply answer the person, or something else?

Every substance check in ``response_reliability`` asks whether a reply has
enough of the right *kind* of content for the *kind* of turn it answers. None
of them asks whether it engages what was actually said. So this passed the
gate, live:

    Bryan:  "Just the part about getting them to see that…"
    Aura:   "Getting them to see that the octopus's camouflage isn't just
             brain-controlled — it might be partly managed by their skin."

Fluent, substantial, grammatical, and about nothing the conversation was
about. The quality log recorded ``coherence=0.814 assessment=ok``, because
``coherence`` there is ``state.cognition.coherence_score`` — an internal
state score that never looks at the thread.

The bar for firing is deliberately high. The dominant defect class in this
runtime is a gate discarding a good answer and then reporting an
infrastructure failure over it, so this reports *abandonment* only when the
reply shares essentially nothing with either the user's turn or the live
thread, and exempts every shape that legitimately has low overlap:
clarifying questions, acknowledgements, refusals, and answers to open
prompts that name no subject of their own.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9][a-z0-9'’-]*", re.IGNORECASE)

# Function words carry no topical signal; counting them makes any two English
# sentences look related.
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he her here hers herself him himself his how i i'd i'll i'm i've if in
    into is isn't it it's its itself just let's me more most mustn't my myself no nor not
    of off on once only or other ought our ours ourselves out over own same shan't she
    should shouldn't so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was wasn't we were weren't
    what when where which while who whom why with won't would wouldn't you your yours
    yourself yourselves thing things really quite kind sort lot bit way get got make made
    going go come came know think feel see say said tell told want need like yeah yes ok
    okay sure right well now still even also maybe perhaps actually
    """.split()
)

#: Replies that legitimately share little vocabulary with the prompt.
_CLARIFYING = re.compile(
    r"\b(what do you mean|can you (tell|say|clarify|explain)|"
    r"i (might have )?misunderstood|not sure what you|which (one|part)|"
    r"say more|could you rephrase|what are you looking for)\b",
    re.IGNORECASE,
)
_ACKNOWLEDGEMENT = re.compile(
    r"\b(you'?re welcome|thanks|thank you|no problem|glad|sounds good|"
    r"i'?m here|i'?m back|sorry about that|got it|understood)\b",
    re.IGNORECASE,
)
_REFUSAL = re.compile(
    r"\b(i (can'?t|won'?t|am not able|couldn'?t)|i'?d rather not|"
    r"i don'?t have|i'?m not going to)\b",
    re.IGNORECASE,
)


def content_terms(text: str) -> set[str]:
    """Topical vocabulary of a passage — the part that can be *about* something."""
    return {
        token
        for token in (m.group(0).lower() for m in _WORD.finditer(str(text or "")))
        if len(token) >= 3 and token not in _STOPWORDS
    }


@dataclass(frozen=True)
class ThreadContinuityVerdict:
    """Whether a reply stayed with the conversation, and the numbers behind it."""

    abandoned: bool
    reason: str
    overlap_with_turn: float
    overlap_with_thread: float
    shared: tuple[str, ...]

    def as_metrics(self) -> dict[str, object]:
        return {
            "thread_abandoned": self.abandoned,
            "thread_reason": self.reason,
            "overlap_turn": round(self.overlap_with_turn, 3),
            "overlap_thread": round(self.overlap_with_thread, 3),
            "shared_terms": list(self.shared)[:8],
        }


def _overlap(reply_terms: set[str], other: set[str]) -> tuple[float, set[str]]:
    if not other or not reply_terms:
        return 0.0, set()
    shared = reply_terms & other
    return len(shared) / float(len(other)), shared


def assess_thread_continuity(
    user_message: str,
    reply_text: str,
    *,
    recent_thread: Sequence[str] | Iterable[str] | None = None,
    min_turn_overlap: float = 0.08,
    min_thread_overlap: float = 0.05,
) -> ThreadContinuityVerdict:
    """Report whether ``reply_text`` engages ``user_message`` or the thread.

    Conservative by construction: a reply is only called abandoned when it is
    substantial, the user's turn is substantial, and it shares essentially
    nothing with either. Short replies, clarifications, acknowledgements and
    refusals are never flagged — they are the shapes with honest low overlap.
    """
    reply = " ".join(str(reply_text or "").split())
    turn = " ".join(str(user_message or "").split())

    reply_terms = content_terms(reply)
    turn_terms = content_terms(turn)

    if not reply or not turn:
        return ThreadContinuityVerdict(False, "insufficient_text", 1.0, 1.0, ())

    # A reply cannot abandon a turn that named no subject ("go on", "and?").
    if len(turn_terms) < 3:
        return ThreadContinuityVerdict(False, "open_prompt", 1.0, 1.0, ())

    # Too short to be a topical excursion — these are backchannels.
    if len(reply_terms) < 6:
        return ThreadContinuityVerdict(False, "brief_reply", 1.0, 1.0, ())

    if _CLARIFYING.search(reply):
        return ThreadContinuityVerdict(False, "clarifying_question", 1.0, 1.0, ())
    if _ACKNOWLEDGEMENT.search(reply) and len(reply_terms) < 25:
        return ThreadContinuityVerdict(False, "acknowledgement", 1.0, 1.0, ())
    if _REFUSAL.search(reply):
        return ThreadContinuityVerdict(False, "refusal", 1.0, 1.0, ())

    turn_overlap, shared_turn = _overlap(reply_terms, turn_terms)

    thread_terms: set[str] = set()
    for message in list(recent_thread or [])[-6:]:
        thread_terms |= content_terms(message)
    thread_overlap, shared_thread = _overlap(reply_terms, thread_terms)

    shared = tuple(sorted(shared_turn | shared_thread))

    if turn_overlap >= min_turn_overlap or thread_overlap >= min_thread_overlap:
        return ThreadContinuityVerdict(
            False, "engaged", turn_overlap, thread_overlap, shared
        )

    return ThreadContinuityVerdict(
        True, "reply_abandons_thread", turn_overlap, thread_overlap, shared
    )

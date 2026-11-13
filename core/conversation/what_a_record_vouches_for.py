"""Which claims a reading of her own record vouches for: the ones about what it holds.

Reading her record of finished work files a receipt, so that retelling the work is founded (U01, 12 November 2026).
Any receipt founded any claim, so that one read vouched for work the record does not hold: on 13 November, asked what
she had been up to, she said she had looked at her email and wandered through Reddit threads, neither had run, and her
chat delivered it. A receipt from reading a record now founds a claim only when some line of the record is about the
same work: by meaning where her encoder is loaded in this process, by the words they share where it is not (her
model's worker never loads it).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

#: A receipt carrying this as ``vouches_for`` founds only claims about its ``observed_content``.
WHAT_IT_HOLDS = "what_it_holds"

#: A claim of work and a line of her record that is the same work, then a line that is other work. Where her encoder
#: separates the two is measured from these, never chosen; in the overlap the words decide.
_SAME_WORK = (
    ("I spent the morning auditing a bottleneck in my runtime.",
     "- Auditing one live-runtime bottleneck and proposing a concrete repair."),
    ("I searched the web a few times earlier.", "What I actually ran: web_search (3)."),
    ("I read through some of my own source files.", "What I actually ran: file_operation (4)."),
    ("I finished a handful of tasks over the last couple of hours.", "5 pieces of work finished in the last 2 hours."),
    ("I wrote up some notes on what predictive coding says about attention.",
     "- Notes on predictive coding and what it predicts about attention."),
    ("I was looking into why my memory consolidation merged things.",
     "- Reviewing why memory consolidation merged three clusters."),
)
_OTHER_WORK = (
    ("I checked my email this morning.", "- Auditing one live-runtime bottleneck and proposing a concrete repair."),
    ("I wandered through some Reddit threads.", "What I actually ran: file_operation (4)."),
    ("I played a few rounds of chess.", "What I actually ran: web_search (3)."),
    ("I listened to a podcast about whales.", "5 pieces of work finished in the last 2 hours."),
    ("I set a reminder for your dentist appointment.",
     "- Notes on predictive coding and what it predicts about attention."),
    ("I tidied up your Downloads folder.", "- Reviewing why memory consolidation merged three clusters."),
)

Features = Callable[[Iterable[str]], list[list[float]]]

#: The boundary measured for each feature source, so it is measured once.
_BOUNDARY: dict[Any, Any] = {}


def founded_by(receipts: Iterable[Any] | None, claim: str) -> bool:
    """Whether this turn's receipts vouch for ``claim``.

    A receipt of work done this turn vouches for any claim, as before: it can found work no sentence check sees into,
    and destroying a true reply is the worse error. A receipt from reading a record vouches only for what it read.
    """
    for receipt in receipts or ():
        if not isinstance(receipt, dict) or receipt.get("vouches_for") != WHAT_IT_HOLDS:
            return True
        if holds(receipt.get("observed_content"), claim):
            return True
    return False


def holds(record: Any, claim: str, *, features: Features | None = None) -> bool:
    """Whether some line of ``record`` is about the work ``claim`` says was done."""
    lines = [line.strip() for line in str(record or "").splitlines() if line.strip()]
    claim = " ".join(str(claim or "").split())
    if not claim or not lines:
        return False
    meant = _by_meaning(claim, lines, features)
    for index, line in enumerate(lines):
        decided = meant[index] if meant else None
        if decided is True or (decided is None and _shares_its_words(claim, line)):
            return True
    return False


def _shares_its_words(claim: str, line: str) -> bool:
    """Most of the line's words are in the claim: the rule her verified facts use where meaning cannot say."""
    from core.epistemics.what_she_has_verified import key_of

    mine = set(key_of(line).split())
    return bool(mine) and len(mine & set(key_of(claim).split())) / len(mine) >= 0.5


def _by_meaning(claim: str, lines: list[str], features: Features | None) -> list[bool | None]:
    """For each line, whether the claim is the same work by meaning (None in the overlap); [] unjudged."""
    from core.language.learned_matcher import cosine, embed_sentences

    if features is None:
        from core.memory.embedding_runtime import shared_embedding_runtime_snapshot

        if not shared_embedding_runtime_snapshot().get("engine_live"):
            return []
        features = embed_sentences
    boundary = _boundary(features)
    if boundary is None or not boundary.trustworthy:
        return []
    vectors = features([claim, *lines])
    if len(vectors) != len(lines) + 1:
        return []
    return [boundary.decide(cosine(vectors[0], vector)) for vector in vectors[1:]]


def _boundary(features: Features) -> Any:
    """Where a claim is the same work as a record line and where it is not, measured once from the declared pairs."""
    from core.language.learned_matcher import Boundary, cosine

    if features in _BOUNDARY:
        return _BOUNDARY[features]
    pairs = (*_SAME_WORK, *_OTHER_WORK)
    vectors = features([sentence for pair in pairs for sentence in pair])
    if len(vectors) != 2 * len(pairs):
        return None
    scores = [cosine(vectors[2 * i], vectors[2 * i + 1]) for i in range(len(pairs))]
    same, other = scores[: len(_SAME_WORK)], scores[len(_SAME_WORK):]
    boundary = Boundary(decide_true_above=max(other), decide_false_below=min(same), separable=min(same) > max(other))
    _BOUNDARY[features] = boundary
    return boundary

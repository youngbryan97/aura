"""Which claims a reading of her own record vouches for: the ones about what it holds.

Reading her record of finished work files a receipt, so that retelling the work is founded (U01, 12 November 2026).
As any receipt it founded any claim, including work the record does not hold. It now founds a claim when some line of
the record is the same work, judged by meaning with her encoder against a boundary measured from declared pairs.

Where her encoder is not loaded (her model's worker never loads it) the read vouches as before, and her main process
judges. Shared words were tried first and refused a true retelling: on 13 November her record held "Read connected
Reddit state for bounded autonomous social awareness" and she said she had been "wandering through some recent
threads on Reddit to keep my social awareness current", which shares four of its nine words. By meaning it is the same
work (0.547 against a boundary of 0.435), and "I played a few rounds of chess online" matches no line.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

#: A receipt carrying this as ``vouches_for`` founds only claims about its ``observed_content``.
WHAT_IT_HOLDS = "what_it_holds"

#: A claim of work and a line of her record that is the same work, then a line that is other work. Where her encoder
#: separates the two is measured from these, never chosen; in the overlap the read vouches.
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
    """Whether some line of ``record`` is the work ``claim`` says was done; True where that cannot be judged here."""
    lines = [line.strip() for line in str(record or "").splitlines() if line.strip()]
    claim = " ".join(str(claim or "").split())
    if not claim or not lines:
        return False
    meant = _by_meaning(claim, lines, features)
    return not meant or any(decided is not False for decided in meant)


def _by_meaning(claim: str, lines: list[str], features: Features | None) -> list[bool | None]:
    """For each line, whether the claim is the same work by meaning (None in the overlap); [] unjudged."""
    from core.cognition.evidence_relevance import vectors_if_ready
    from core.language.learned_matcher import cosine

    if features is None:
        # The encoder her evidence judge uses, only once it is loaded: the shared runtime's handle held no model in
        # her app, so asking it left every live record read unjudged (13 November 2026).
        features = vectors_if_ready
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

"""Deciding what a sentence is, from examples rather than from a word list.

Every matcher in this runtime is a regex with a list of words in it, and
every one of them has been wrong in the same way: a phrasing nobody thought
of. "I saved it as sitting_timer.html" missed an action-claim rule by four
characters. "Chen and Dara sit next to each other" stated a relation after
both names instead of between them. "Let's break down the problem and use
code" put six words between the two the pattern wanted adjacent. Each was
repaired by widening the pattern, which is the same move that will be needed
again for the next phrasing.

The labels for doing better already exist and nothing reads them. Every
Observable declares ``examples`` and ``counter_examples``, and the registry
test fails a matcher whose examples it gets wrong — so each matcher ships
with a small, curated, adversarial dataset. This turns those declarations
into a decision.

Three properties make it safe to put in front of a live runtime:

* **The boundary is measured, not chosen.** Leave-one-out over the declared
  examples gives a score for each; the boundary sits midway between the worst
  positive and the best negative. A declaration whose examples do not separate
  produces no boundary at all.
* **It abstains.** Between those two scores it returns None, and the caller
  keeps whatever it did before. A learned matcher that guesses in the middle
  is worse than a pattern that is merely narrow.
* **It never needs the model to answer.** Embedding is not generation, so
  there is nothing to steer and no prompt to write.

Learning continues from use: :func:`observe` records a sentence whose truth
was settled by something other than this matcher — a tool receipt, an accepted
answer — and those become examples with the same standing as the declared
ones. The receipts are the labels; nobody has to write them down.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from core.runtime.errors import record_degradation

__all__ = [
    "Boundary",
    "FeatureSource",
    "LearnedMatcher",
    "cosine",
    "embed_sentences",
]

_RECOVERABLE = (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError)

#: Nothing shorter carries enough signal to place, and the callers all guard
#: for empties anyway.
_MIN_CHARS = 3


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, safe on zero vectors."""
    total = norm_left = norm_right = 0.0
    for a, b in zip(left, right, strict=False):
        total += float(a) * float(b)
        norm_left += float(a) * float(a)
        norm_right += float(b) * float(b)
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return total / math.sqrt(norm_left * norm_right)


def embed_sentences(sentences: Iterable[str]) -> list[list[float]]:
    """Vectors for these sentences, or [] when no embedder is available.

    Uses the runtime's own vector memory, so there is one embedding model in
    the process rather than a second one loaded beside it.
    """
    texts = [str(text or "").strip() for text in sentences]
    if not any(texts):
        return []
    try:
        from core.memory.embedding_runtime import acquire_shared_embedding_engine

        with acquire_shared_embedding_engine("language.learned_matcher") as engine:
            return [[float(value) for value in engine.embed(text)] for text in texts]
    except _RECOVERABLE as exc:
        record_degradation(
            "language.learned_matcher",
            exc,
            severity="debug",
            action="left the decision to the declared pattern",
            enforce_failure_policy=False,
        )
        return []


def _spread(values: Sequence[float]) -> float:
    """How much the scores vary, as a standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


@dataclass(frozen=True, slots=True)
class Boundary:
    """Where the declared examples stop agreeing with each other."""

    lower: float
    upper: float
    separable: bool
    spread: float = 0.0

    @property
    def gap(self) -> float:
        return self.upper - self.lower

    @property
    def trustworthy(self) -> bool:
        """Whether the separation is bigger than the noise inside a class.

        Measured across the 25 declarations in this runtime, eight separated
        and most of those by a hair — gaps of 0.003 to 0.117 while the
        positives themselves varied by more than that. A boundary narrower
        than the spread of the examples it was drawn from is describing
        sampling noise, and acting on it is worse than the pattern it was
        meant to improve on.
        """
        return self.separable and self.gap > self.spread

    def decide(self, score: float) -> bool | None:
        """True, False, or None when the score falls in the gap."""
        if not self.trustworthy:
            return None
        if score >= self.upper:
            return True
        if score <= self.lower:
            return False
        return None


#: Turning sentences into vectors. Swappable on purpose.
#:
#: MEASURED, 2026-08-20. Against all twenty-five declarations in this runtime,
#: a topical sentence embedder separated eight and none by more than the
#: spread inside its own classes — zero trustworthy boundaries. That is not a
#: bug in the arithmetic; it is what the feature space is for. An embedder is
#: trained to make "I saved it as report.csv" and "you could save it as
#: report.csv" NEAR each other, and the axis these decisions turn on — who
#: acts, asserted or offered, done or hypothetical — is the one it discards.
#:
#: The axis is present in the resident model's own residual stream, which is
#: what lets it answer at all, and this runtime already taps that stream
#: during generation. So the feature source is a parameter: the decision
#: surface, its measured boundary and its abstention are the same whichever
#: one is plugged in.
FeatureSource = Callable[[Iterable[str]], list[list[float]]]


@dataclass
class LearnedMatcher:
    """One decision, learned from what it was declared to get right.

    ``name`` is only for reporting. ``positives`` and ``negatives`` are the
    declaration — normally the same tuples an Observable already carries.
    ``features`` is how sentences become vectors.
    """

    name: str
    positives: tuple[str, ...] = ()
    negatives: tuple[str, ...] = ()
    features: FeatureSource = field(default=None, repr=False)  # type: ignore[assignment]
    _positive_vectors: list[list[float]] = field(default_factory=list, repr=False)
    _negative_vectors: list[list[float]] = field(default_factory=list, repr=False)
    _boundary: Boundary | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _ready: bool = field(default=False, repr=False)

    def observe(self, sentence: str, *, holds: bool) -> None:
        """Record a sentence whose truth something else settled.

        A tool receipt says an action happened; an accepted answer says a
        reading was the right one. Those are labels nobody had to write, and
        they carry the same standing as the declared examples.
        """
        text = str(sentence or "").strip()
        if len(text) < _MIN_CHARS:
            return
        with self._lock:
            bucket = self.positives if holds else self.negatives
            if text in bucket:
                return
            if holds:
                self.positives = (*self.positives, text)
            else:
                self.negatives = (*self.negatives, text)
            self._ready = False

    def _score(self, vector: Sequence[float], *, skip_positive: int = -1, skip_negative: int = -1) -> float:
        """How much more this looks like a positive than a negative.

        Nearest neighbour on both sides rather than a centroid: these
        declarations are deliberately adversarial, so their negatives sit close
        to their positives and a mean would put the two clouds on top of each
        other.
        """
        best_positive = max(
            (
                cosine(vector, candidate)
                for index, candidate in enumerate(self._positive_vectors)
                if index != skip_positive
            ),
            default=-1.0,
        )
        best_negative = max(
            (
                cosine(vector, candidate)
                for index, candidate in enumerate(self._negative_vectors)
                if index != skip_negative
            ),
            default=-1.0,
        )
        return best_positive - best_negative

    def _prepare(self) -> bool:
        """Embed the declaration and measure its boundary. Once."""
        with self._lock:
            if self._ready:
                return self._boundary is not None
            self._ready = True
            self._boundary = None
            if len(self.positives) < 2 or len(self.negatives) < 2:
                return False
            source = self.features or embed_sentences
            positives = source(self.positives)
            negatives = source(self.negatives)
            if len(positives) != len(self.positives) or len(negatives) != len(self.negatives):
                return False
            self._positive_vectors = positives
            self._negative_vectors = negatives

            # Leave-one-out: score every declared example against the others.
            # The boundary is where the two sets stop overlapping, which is a
            # measurement of this declaration rather than a number chosen for
            # it.
            positive_scores = [
                self._score(vector, skip_positive=index)
                for index, vector in enumerate(positives)
            ]
            negative_scores = [
                self._score(vector, skip_negative=index)
                for index, vector in enumerate(negatives)
            ]
            worst_positive = min(positive_scores)
            best_negative = max(negative_scores)
            spread = _spread(positive_scores + negative_scores)
            self._boundary = Boundary(
                lower=best_negative,
                upper=worst_positive,
                separable=worst_positive > best_negative,
                spread=spread,
            )
            return True

    def decide(self, sentence: str) -> bool | None:
        """Whether this sentence is one of the things declared, or None.

        None means the declaration does not settle it — too few examples, no
        embedder, examples that do not separate, or a sentence that lands
        between them. Every caller keeps its own answer for None.
        """
        text = str(sentence or "").strip()
        if len(text) < _MIN_CHARS:
            return None
        if not self._prepare():
            return None
        vectors = (self.features or embed_sentences)([text])
        if not vectors:
            return None
        boundary = self._boundary
        return boundary.decide(self._score(vectors[0])) if boundary else None

    def report(self) -> dict[str, object]:
        """What this matcher knows, for a health page or a test."""
        self._prepare()
        boundary = self._boundary
        return {
            "name": self.name,
            "positives": len(self.positives),
            "negatives": len(self.negatives),
            "separable": bool(boundary.separable) if boundary else False,
            "trustworthy": bool(boundary.trustworthy) if boundary else False,
            "gap": round(boundary.gap, 4) if boundary else None,
            "spread": round(boundary.spread, 4) if boundary else None,
        }

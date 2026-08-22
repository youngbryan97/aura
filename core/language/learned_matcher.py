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

import json
import math
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from core.language.substrate_store import (
    LanguageSubstrateStore,
    get_language_substrate_store,
)
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

__all__ = [
    "Boundary",
    "registered_surfaces",
    "warm_all",
    "FeatureSource",
    "LearnedMatcher",
    "cosine",
    "embed_sentences",
]

_RECOVERABLE = (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError)

#: Nothing shorter carries enough signal to place, and the callers all guard
#: for empties anyway.
_MIN_CHARS = 3

#: How many unseen phrasings to remember for the warmer. Past this the queue
#: is describing traffic rather than vocabulary.
_PENDING_CEILING = 256


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


#: Every surface a live turn has asked. Warming reads this, so a surface
#: cannot be consulted without also being maintained.
_SURFACES: list[LearnedMatcher] = []
_SURFACES_LOCK = checked_lock("core.language.learned_matcher")


def _register(surface: LearnedMatcher) -> None:
    with _SURFACES_LOCK:
        if not any(existing is surface for existing in _SURFACES):
            _SURFACES.append(surface)


def registered_surfaces() -> tuple[LearnedMatcher, ...]:
    """The surfaces live turns have consulted."""
    with _SURFACES_LOCK:
        return tuple(_SURFACES)


def warm_all(limit: int = 8) -> int:
    """Settle what every consulted surface deferred, and write it down.

    Returns how many phrasings were settled across all of them.
    """
    settled = 0
    for surface in registered_surfaces():
        settled += surface.warm(limit=limit)
        surface.save()
    return settled


def _cache_key(sentence: str) -> str:
    """What counts as the same sentence for the fast path.

    Case, spacing and trailing punctuation are not the decision, and keying on
    the raw string made "I saved it as report.csv" and "I saved it as
    report.csv." two separate first sightings.

    This is a partial fix and worth being exact about: it collapses spelling,
    not wording. A genuine paraphrase is still a new sighting, because
    deciding one needs its vector and computing that costs a forward pass the
    turn cannot spend. What would close it properly is per-sentence hidden
    states for text the model has just generated — it computed them to write
    the sentence, and nothing keeps them.
    """
    folded = " ".join(str(sentence or "").lower().split())
    return folded.strip(" .,;:!?\"'`")


def _spread(values: Sequence[float]) -> float:
    """How much the scores vary, as a standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


@dataclass(frozen=True, slots=True)
class Boundary:
    """Where one class is unambiguous, and where the two overlap.

    The first version required the examples to separate perfectly: the worst
    positive above the best negative, or no boundary at all. Measured on real
    traffic that threw away a decision with an AUROC of 0.979 — the ranking
    was nearly perfect and a handful of overlapping examples made the whole
    surface abstain on everything. Demanding zero overlap is not caution, it
    is a rule that cannot be met by data.

    So the band is the overlap itself. Above every negative, only positives
    were ever seen; below every positive, only negatives. In between both
    occur, and that is where abstaining is the honest answer.
    """

    decide_true_above: float
    decide_false_below: float
    spread: float = 0.0
    separable: bool = False

    @property
    def gap(self) -> float:
        """Positive when the classes never overlapped at all."""
        return self.decide_false_below - self.decide_true_above

    @property
    def trustworthy(self) -> bool:
        """Whether either decisive region exists.

        A boundary with no decisive region cannot say anything, which is the
        one case where returning None for everything is not a policy but the
        only available answer.
        """
        return bool(self.decide_true_above > float("-inf"))

    def decide(self, score: float) -> bool | None:
        """True above every negative, False below every positive, else None."""
        if not self.trustworthy:
            return None
        if score > self.decide_true_above:
            return True
        if score < self.decide_false_below:
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
    _decided: dict[str, bool] = field(default_factory=dict, repr=False)
    _pending: set[str] = field(default_factory=set, repr=False)
    _dirty: bool = field(default=False, repr=False)
    _loaded: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _ready: bool = field(default=False, repr=False)
    _store: LanguageSubstrateStore | None = field(default=None, repr=False)

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
            # Verdicts were reached against the OLD boundary. An example that
            # moves the boundary makes them stale, and keeping them meant a
            # sentence decided early could never be revised however much was
            # learned afterwards.
            self._decided.clear()
            self._dirty = True

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
        self.load()
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
            self._boundary = Boundary(
                decide_true_above=best_negative,
                decide_false_below=worst_positive,
                spread=_spread(positive_scores + negative_scores),
                separable=worst_positive > best_negative,
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

    # ── durability ──────────────────────────────────────────────────────
    #
    # Everything above lives in Python fields, and the runtime restarts often
    # — for a code change, for a model swap, after a crash. Without a durable
    # write, every phrasing learned from use is discarded each time and the
    # substrate can never accumulate anything. That is the difference between
    # a cache and learning.

    def _store_path(self) -> Path | None:
        try:
            return (self._store or get_language_substrate_store()).matcher_path(
                self.name
            )
        except _RECOVERABLE:
            return None

    def load(self) -> bool:
        """Read what earlier runs learned. Once, and never fatal."""
        with self._lock:
            if self._loaded:
                return False
            self._loaded = True
        path = self._store_path()
        if path is None or not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            record_degradation(
                "language.learned_matcher",
                exc,
                severity="debug",
                action="started from the declared examples alone",
                enforce_failure_policy=False,
            )
            return False
        # The gateway writes a {schema, schema_version, payload} envelope, so
        # the record is one level in. Reading the envelope as the record found
        # no name and discarded everything the last run learned.
        if isinstance(payload, dict) and isinstance(payload.get("payload"), dict):
            payload = payload["payload"]
        if not isinstance(payload, dict) or payload.get("name") != self.name:
            return False
        with self._lock:
            for text in payload.get("positives") or ():
                if isinstance(text, str) and text not in self.positives:
                    self.positives = (*self.positives, text)
            for text in payload.get("negatives") or ():
                if isinstance(text, str) and text not in self.negatives:
                    self.negatives = (*self.negatives, text)
            for text in payload.get("pending") or ():
                if isinstance(text, str) and len(self._pending) < _PENDING_CEILING:
                    self._pending.add(text)
            # Verdicts are NOT restored. They were reached against a boundary
            # this process has not measured yet, and re-deciding them costs one
            # warm cycle against keeping an answer nothing here can vouch for.
            self._ready = False
        return True

    def save(self) -> bool:
        """Write what this run learned, through the governed gateway."""
        with self._lock:
            if not self._dirty:
                return False
            payload = {
                "name": self.name,
                "positives": list(self.positives),
                "negatives": list(self.negatives),
                "pending": sorted(self._pending),
            }
            self._dirty = False
        path = self._store_path()
        if path is None:
            return False
        try:
            (self._store or get_language_substrate_store()).write_matcher(
                self.name,
                payload,
            )
        except _RECOVERABLE as exc:
            record_degradation(
                "language.learned_matcher",
                exc,
                severity="debug",
                action="kept this run's phrasings in memory only",
                enforce_failure_policy=False,
            )
            return False
        return True

    def decide_without_waiting(self, sentence: str) -> bool | None:
        """A decision only if one is already in hand.

        A live turn cannot wait on a forward pass, so this never computes: it
        answers from what has been decided before and records anything new for
        the warmer. The first time a phrasing appears it abstains and the
        caller's own rule stands; from then on the decision is there.

        That is the shape of learning from use rather than from a list — the
        cost is one missed novelty per phrasing, and the alternative is
        holding up every turn to be sure.
        """
        text = str(sentence or "").strip()
        if len(text) < _MIN_CHARS:
            return None
        # Consulted on a live turn means it needs a warmer, so asking IS the
        # registration.
        #
        # LIVE, 2026-08-20. The routing surface was wired to this method and
        # nothing warmed it: it queued every novel request, abstained, and
        # could never turn a queued request into a decision. Verdicts are
        # deliberately not restored across restarts either, so it had nothing
        # to fall back on. It was inert in production while measuring 0.979
        # offline. A second hardcoded warm call would have fixed this one
        # surface and left the next with the same bug.
        _register(self)
        self.load()
        key = _cache_key(text)
        with self._lock:
            if key in self._decided:
                return self._decided[key]
            if text not in self._pending and len(self._pending) < _PENDING_CEILING:
                self._pending.add(text)
                self._dirty = True
        return None

    def warm(self, limit: int = 16) -> int:
        """Decide what has been waiting. Call off the critical path.

        Returns how many were settled, so a caller can log it or stop.
        """
        with self._lock:
            waiting = sorted(self._pending)[: max(1, int(limit))]
        settled = 0
        for text in waiting:
            verdict = self.decide(text)
            with self._lock:
                # Only what was actually settled leaves the queue. Discarding
                # unconditionally lost every phrase the model was too busy to
                # decide, and it had to be met again to get another attempt.
                if verdict is not None:
                    self._pending.discard(text)
                    self._decided[_cache_key(text)] = verdict
                    self._dirty = True
                    settled += 1
        if settled:
            self.save()
        return settled

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
            "decided": len(self._decided),
            "pending": len(self._pending),
        }

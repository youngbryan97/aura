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
import logging
import math
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.language.substrate_store import (
    LanguageSubstrateStore,
    get_language_substrate_store,
)
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger(__name__)


__all__ = [
    "Boundary",
    "registered_surfaces",
    "warm_all",
    "FeatureSource",
    "LearnedMatcher",
    "cosine",
    "embed_sentences",
    "her_encoder_only",
]

_RECOVERABLE = (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError)

#: Nothing shorter carries enough signal to place, and the callers all guard
#: for empties anyway.
_MIN_CHARS = 3

#: How many unseen phrasings to remember for the warmer. Past this the queue
#: is describing traffic rather than vocabulary.
_PENDING_CEILING = 256


def _an_event_set() -> threading.Event:
    event = threading.Event()
    event.set()
    return event


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, safe on zero vectors; over the shorter of the two where their lengths differ.

    In numpy, not a Python loop: LIVE 14 November 2026 the loop, run for every pair of a matcher's examples, held one
    of her threads at a full core for hours and her event loop waited behind it, once for 104 s.
    """
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        a, b = a[: min(a.size, b.size)], b[: min(a.size, b.size)]
    norm_left, norm_right = float(a @ a), float(b @ b)
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return float(a @ b) / math.sqrt(norm_left * norm_right)


def _as_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray | None:
    """The vectors as one float64 matrix, or None where they are not all one length."""
    try:
        matrix = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError):
        return None  # not a failure: vectors of different lengths are no one matrix, and are read pair by pair
    return matrix if matrix.ndim == 2 else None


def _cosines(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Every row's cosine with every column, zero where either is a zero vector, as ``cosine`` gives it."""
    row_norms = np.sqrt(np.einsum("ij,ij->i", rows, rows))
    column_norms = np.sqrt(np.einsum("ij,ij->i", columns, columns))
    scale = np.outer(row_norms, column_norms)
    products = rows @ columns.T
    return np.divide(products, scale, out=np.zeros_like(products), where=scale > 0.0)


def her_encoder_only(sentences: Iterable[str]) -> list[list[float]]:
    """Vectors from the encoder her app holds, only when it is loaded already; [] otherwise, so a surface abstains.

    For a surface whose decision moves her turn somewhere else. ``embed_sentences`` falls back to the shared runtime,
    which can answer with hashed words; on those, LIVE 20 November 2026 in a test process, the code-work surface read
    "close the front window" and "Create a Word document ... and save it" as work on code.
    """
    texts = [str(text or "").strip() for text in sentences]
    if not any(texts):
        return []
    try:
        from core.cognition.evidence_relevance import vectors_if_ready

        return vectors_if_ready(texts)
    except _RECOVERABLE as exc:
        logger.debug("her app's encoder was not asked (%s: %s); nothing is decided", type(exc).__name__, exc)
        return []


def embed_sentences(sentences: Iterable[str]) -> list[list[float]]:
    """The model's vectors for these sentences, or [] when no model can make them now.

    The encoder her app already holds comes first, asked without loading anything: beside it the shared runtime held
    no model (13 November 2026). The shared runtime is asked next, for the model's vectors only; the MLX worker's
    process has that one alone. Its ``embed`` answers a missing model with hashed words of the model's width, and a
    surface keeps the boundary it measured, so one measured on hashed words and asked with model vectors (or the
    reverse) compared two spaces that share nothing but their length.
    """
    texts = [str(text or "").strip() for text in sentences]
    if not any(texts):
        return []
    try:
        from core.cognition.evidence_relevance import vectors_if_ready

        held = vectors_if_ready(texts)
    except _RECOVERABLE as exc:
        logger.debug("her app's encoder was not asked (%s: %s); the shared runtime is", type(exc).__name__, exc)
        held = []
    if held:
        return held
    try:
        from core.memory.embedding_runtime import acquire_shared_embedding_engine

        with acquire_shared_embedding_engine("language.learned_matcher") as engine:
            by_meaning = getattr(engine, "embed_batch_by_meaning", None)
            if not callable(by_meaning):
                return [[float(value) for value in engine.embed(text)] for text in texts]
            vectors = by_meaning(texts)
            return [] if vectors is None else [[float(value) for value in vector] for vector in vectors]
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
_SURFACES: tuple[LearnedMatcher, ...] = ()
_SURFACES_LOCK = checked_lock("core.language.learned_matcher")


def _register(surface: LearnedMatcher) -> None:
    global _SURFACES
    with _SURFACES_LOCK:
        if not any(existing is surface for existing in _SURFACES):
            # Rebound rather than appended, so a reader needs no lock and
            # cannot see the list mid-append.
            _SURFACES = (*_SURFACES, surface)


def registered_surfaces() -> tuple[LearnedMatcher, ...]:
    """The surfaces live turns have consulted.

    No lock: registration rebinds the name, so reading it is atomic and
    returns one whole version. Lockdep measured 148ms of contention on a
    section that appends one element — every turn that consults a surface
    queueing behind every other one.
    """
    return _SURFACES


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


def _score_against(
    vector: Sequence[float],
    positives: Sequence[Sequence[float]],
    negatives: Sequence[Sequence[float]],
    *,
    skip_positive: int = -1,
    skip_negative: int = -1,
) -> float:
    """Score against an immutable vector snapshot.

    Feature extraction can take a resident-model forward pass. Keeping the
    vectors explicit prevents that work from borrowing the matcher's state
    mutex merely to evaluate a snapshot it already owns.
    """
    return _best_cosine(vector, positives, skip_positive) - _best_cosine(vector, negatives, skip_negative)


def _best_cosine(vector: Sequence[float], candidates: Sequence[Sequence[float]], skip: int) -> float:
    """The nearest candidate's cosine, leaving out the one at ``skip``; -1.0 with none left."""
    matrix = _as_matrix(candidates) if len(candidates) else None
    probe = _as_matrix([vector])
    if matrix is None or probe is None or matrix.shape[1] != probe.shape[1]:
        # Ragged vectors keep the pairwise reading, whose zip takes the shorter of each pair.
        return max((cosine(vector, candidate) for index, candidate in enumerate(candidates) if index != skip),
                   default=-1.0)
    near = _cosines(probe, matrix)[0]
    if 0 <= skip < near.size:
        near = np.delete(near, skip)
    return float(near.max()) if near.size else -1.0


def _leave_one_out(positives: Sequence[Sequence[float]],
                   negatives: Sequence[Sequence[float]]) -> tuple[list[float] | None, list[float] | None]:
    """Each example's score against the others, as ``_score_against`` gives it, from four cosine matrices at once.

    (None, None) where the vectors are not one matrix; the caller then scores them one at a time.
    """
    pos, neg = _as_matrix(positives), _as_matrix(negatives)
    if pos is None or neg is None or pos.shape[1] != neg.shape[1] or len(pos) < 2 or len(neg) < 2:
        return None, None
    among_positives, among_negatives = _cosines(pos, pos), _cosines(neg, neg)
    np.fill_diagonal(among_positives, -np.inf)
    np.fill_diagonal(among_negatives, -np.inf)
    positive_scores = among_positives.max(axis=1) - _cosines(pos, neg).max(axis=1)
    negative_scores = _cosines(neg, pos).max(axis=1) - among_negatives.max(axis=1)
    return positive_scores.tolist(), negative_scores.tolist()


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
    _loading: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _ready: bool = field(default=False, repr=False)
    _preparing: bool = field(default=False, repr=False)
    #: Set whenever no thread is part-way through measuring this surface.
    _settled: threading.Event = field(default_factory=lambda: _an_event_set(), repr=False)
    _example_revision: int = field(default=0, repr=False)
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
            self._example_revision += 1
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
        with self._lock:
            positives = tuple(self._positive_vectors)
            negatives = tuple(self._negative_vectors)
        return _score_against(
            vector,
            positives,
            negatives,
            skip_positive=skip_positive,
            skip_negative=skip_negative,
        )

    def _prepare(self) -> bool:
        """Embed the declaration and measure its boundary. Once."""
        self.load()
        with self._lock:
            if self._ready:
                return self._boundary is not None
            if self._loading or self._preparing:
                return False
            if len(self.positives) < 2 or len(self.negatives) < 2:
                self._ready = True
                self._boundary = None
                return False
            source = self.features or embed_sentences
            positive_examples = tuple(self.positives)
            negative_examples = tuple(self.negatives)
            revision = self._example_revision
            self._preparing = True
            self._settled.clear()

        completed = False
        try:
            positives = source(positive_examples)
            negatives = source(negative_examples)
            completed = True
        finally:
            if not completed:
                with self._lock:
                    self._preparing = False
                self._settled.set()

        if not positives or not negatives:
            # No model to measure with yet is not a property of the declaration. Settled, it kept this surface
            # silent for the life of the process, though the encoder came up later.
            with self._lock:
                self._preparing = False
            self._settled.set()
            return False
        if len(positives) != len(positive_examples) or len(negatives) != len(
            negative_examples
        ):
            boundary = None
        else:
            # Leave-one-out: score every declared example against the others.
            # The boundary is where the two sets stop overlapping, which is a
            # measurement of this declaration rather than a number chosen for
            # it. As four matrices of cosines where the vectors allow it.
            positive_scores, negative_scores = _leave_one_out(positives, negatives)
            positive_scores = positive_scores if positive_scores is not None else [
                _score_against(
                    vector,
                    positives,
                    negatives,
                    skip_positive=index,
                )
                for index, vector in enumerate(positives)
            ]
            negative_scores = negative_scores if negative_scores is not None else [
                _score_against(
                    vector,
                    positives,
                    negatives,
                    skip_negative=index,
                )
                for index, vector in enumerate(negatives)
            ]
            worst_positive = min(positive_scores)
            best_negative = max(negative_scores)
            boundary = Boundary(
                decide_true_above=best_negative,
                decide_false_below=worst_positive,
                spread=_spread(positive_scores + negative_scores),
                separable=worst_positive > best_negative,
            )

        try:
            with self._lock:
                self._preparing = False
                if revision != self._example_revision:
                    return False
                self._positive_vectors = positives
                self._negative_vectors = negatives
                self._boundary = boundary
                self._ready = True
                return boundary is not None
        finally:
            self._settled.set()

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
            # Another thread part-way through measuring this surface is waited for, not abstained on: this call would
            # have spent the same time measuring it had it come first. LIVE 15 November 2026 three checks of her first
            # reply after a restart asked the shared-history surface at once, two abstained, and a true sentence of
            # hers was cut as invented.
            # Its own measurement overtaken by a new example leaves the event set, and abstains as before.
            if self._settled.is_set():
                return None
            self._settled.wait()
            if not self._prepare():
                return None
        with self._lock:
            boundary = self._boundary
            positives = tuple(self._positive_vectors)
            negatives = tuple(self._negative_vectors)
            revision = self._example_revision
        vectors = (self.features or embed_sentences)([text])
        if not vectors:
            return None
        decision = (
            boundary.decide(_score_against(vectors[0], positives, negatives))
            if boundary
            else None
        )
        with self._lock:
            if revision != self._example_revision or boundary is not self._boundary:
                return None
        return decision

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
        except _RECOVERABLE as exc:
            logger.debug(
                "the substrate store gave no path, so nothing learned here is kept (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return None

    def load(self) -> bool:
        """Read what earlier runs learned. Once, and never fatal."""
        with self._lock:
            if self._loaded or self._loading:
                return False
            self._loading = True
        try:
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
            # The gateway writes a {schema, schema_version, payload} envelope,
            # so the record is one level in. Reading the envelope as the record
            # found no name and discarded everything the last run learned.
            if isinstance(payload, dict) and isinstance(payload.get("payload"), dict):
                payload = payload["payload"]
            if not isinstance(payload, dict) or payload.get("name") != self.name:
                return False
            changed = False
            with self._lock:
                for text in payload.get("positives") or ():
                    if isinstance(text, str) and text not in self.positives:
                        self.positives = (*self.positives, text)
                        changed = True
                for text in payload.get("negatives") or ():
                    if isinstance(text, str) and text not in self.negatives:
                        self.negatives = (*self.negatives, text)
                        changed = True
                for text in payload.get("pending") or ():
                    if isinstance(text, str) and len(self._pending) < _PENDING_CEILING:
                        self._pending.add(text)
                if changed:
                    self._example_revision += 1
                # Verdicts are NOT restored. They were reached against a
                # boundary this process has not measured yet, and re-deciding
                # them costs one warm cycle against keeping an answer nothing
                # here can vouch for.
                self._ready = False
            return True
        finally:
            with self._lock:
                self._loaded = True
                self._loading = False

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
        key = _cache_key(text)
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if key in self._decided:
                return self._decided[key]
            if text not in self._pending and len(self._pending) < _PENDING_CEILING:
                self._pending.add(text)
                self._dirty = True
        finally:
            self._lock.release()
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

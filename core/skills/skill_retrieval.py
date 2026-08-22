"""Finding the skill that fits, when nobody wrote a regex for the way it was asked.

What Aura had
-------------
``CapabilityEngine._rank_tool_candidates`` finds candidates with
``detect_intent``, which matches trigger patterns — regular expressions authored
per skill. That works exactly as far as somebody anticipated the phrasing. Ask
for the same capability in words nobody listed and the skill is not proposed;
it is not ranked low, it never enters the list.

The second half is worse. ``SkillLibrary`` learns macros, persists them, and
exposes them through ``get_available_skills_prompt()`` — which has no callers
anywhere in the tree. Skills were being learned into a store that nothing reads,
and the store's own retrieval method dumps every skill it holds rather than
choosing among them, so wiring it as written would have traded invisibility for
a context flood.

Voyager's skill manager is the reference for the other half of this. It embeds
each skill's description, stores it in a vector index, and retrieves the top *k*
for whatever the agent is about to attempt. Retrieval by meaning is what lets a
library grow past the point where a human can maintain a trigger list for it.

Two backends, and the default is the one that always works
-----------------------------------------------------------
:class:`LexicalIndex` — TF-IDF cosine over whole words and character trigrams —
is what runs unless an encoder is installed. It is deliberate engineering rather
than a placeholder: embedding the catalog needs a leased model lane, which does
not exist during the boot window when the catalog is first consulted, and a
retriever that returns nothing until a model is resident would be dead exactly
when it is first needed. N-gram TF-IDF handles the cases that break substring
matching — "summarise" against "summarize", "csv" inside "read_csv_file", word
order, inflection. It is worse than embeddings at synonymy and better than a
regex at everything.

:meth:`SkillRetriever.install_encoder` swaps in the semantic backend. The
encoder is injected rather than reached for, so this module never decides to
take a model lane, and the caller that owns the lane owns the lifetime. An
encoder that raises is dropped once and the retriever continues on lexical —
retrieval degrading to a worse ranking is survivable, retrieval raising inside
tool selection is not.

Which backend answered is on every result, because "nothing in the catalog is
relevant" and "the n-gram index found nothing" are different statements, and a
caller that cannot tell them apart will read the first as a fact.

Additive by construction
------------------------
The engine's existing lexical and heuristic candidates are kept and ranked
first; retrieval appends what they missed. It can therefore only widen the
field, never remove a candidate the current code would have chosen — so turning
it on cannot regress a working path, and what it contributes is visible as the
tail of the list.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.SkillRetrieval")

__all__ = [
    "SkillDocument",
    "RetrievedSkill",
    "LexicalIndex",
    "SkillRetriever",
    "get_skill_retriever",
]

#: Character n-gram width for the lexical backend.
#:
#: Three, which is the shortest width that still carries morphology: it matches
#: "summari" across "summarise"/"summarize", and it keeps "csv" as a unit inside
#: "read_csv_file". Two would match so much that every skill scores against
#: every query; four stops bridging the s/z and -ing/-ed endings that are the
#: whole reason for using n-grams instead of words.
_NGRAM = 3

#: Floor on a lexical score before a skill is offered at all.
#:
#: Not a tuned relevance bar. TF-IDF cosine over n-grams gives a small non-zero
#: score to almost any pair of strings that share a letter trigram, so without a
#: floor "retrieve the top 3" always returns three skills, including for a query
#: about nothing in the catalog. The floor is what makes an empty result
#: possible, and an empty result is the honest answer to "no skill does this".
_LEXICAL_FLOOR = 0.05

_TOKEN = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _TOKEN.findall(str(text or "").lower())


def _features(text: str) -> Counter[str]:
    """Whole words plus character trigrams over the whole normalised string.

    Both, because they fail differently. Words carry meaning and miss
    "read_csv_file" against "csv"; trigrams bridge morphology and would rank a
    long skill name above an exact word match if used alone.
    """
    words = _words(text)
    features: Counter[str] = Counter(f"w:{w}" for w in words)
    joined = " ".join(words)
    for i in range(len(joined) - _NGRAM + 1):
        gram = joined[i : i + _NGRAM]
        if gram.strip():
            features[f"g:{gram}"] += 1
    return features


@dataclass(frozen=True)
class SkillDocument:
    """One retrievable capability."""

    name: str
    description: str
    #: Where it came from — "catalog", "macro", "forged". Carried through to the
    #: result so a caller can treat a learned macro differently from a shipped
    #: skill without having to look it up again.
    source: str = "catalog"

    @property
    def text(self) -> str:
        return f"{self.name} {self.description}".strip()


@dataclass(frozen=True)
class RetrievedSkill:
    """A hit, its score, and which backend produced it."""

    name: str
    score: float
    source: str
    backend: str

    def __str__(self) -> str:
        return f"{self.name} ({self.score:.3f} via {self.backend})"


class LexicalIndex:
    """TF-IDF cosine over words and character trigrams.

    Rebuilt whenever the corpus changes, which is cheap: the catalog is a few
    hundred short strings, and document frequencies have to be recomputed anyway
    when a document is added or removed.
    """

    def __init__(self, documents: Sequence[SkillDocument] = ()) -> None:
        self._documents: tuple[SkillDocument, ...] = ()
        self._vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self.rebuild(documents)

    @property
    def documents(self) -> tuple[SkillDocument, ...]:
        return self._documents

    def rebuild(self, documents: Sequence[SkillDocument]) -> None:
        self._documents = tuple(documents)
        if not self._documents:
            self._vectors = []
            self._idf = {}
            return

        counts = [_features(doc.text) for doc in self._documents]
        n = len(counts)
        frequency: Counter[str] = Counter()
        for feature_counts in counts:
            frequency.update(feature_counts.keys())
        # Smoothed IDF. The +1 inside and outside keeps a feature present in
        # every document at a small positive weight instead of exactly zero,
        # which matters when the catalog is small and a common word is still the
        # only thing distinguishing two skills.
        self._idf = {
            feature: math.log((n + 1) / (document_count + 1)) + 1.0
            for feature, document_count in frequency.items()
        }
        self._vectors = [self._vector(feature_counts) for feature_counts in counts]

    def _vector(self, feature_counts: Mapping[str, int]) -> dict[str, float]:
        weighted = {
            feature: (1.0 + math.log(count)) * self._idf.get(feature, 1.0)
            for feature, count in feature_counts.items()
        }
        norm = math.sqrt(sum(v * v for v in weighted.values()))
        if norm <= 0.0:
            return {}
        return {feature: value / norm for feature, value in weighted.items()}

    def search(self, query: str, k: int) -> list[tuple[SkillDocument, float]]:
        if not self._documents or k <= 0:
            return []
        query_vector = self._vector(_features(query))
        if not query_vector:
            return []
        scored: list[tuple[SkillDocument, float]] = []
        for document, vector in zip(self._documents, self._vectors):
            if not vector:
                continue
            # Iterate the shorter side; a query is far smaller than a document
            # vector and the intersection is what the dot product needs.
            small, large = (
                (query_vector, vector)
                if len(query_vector) <= len(vector)
                else (vector, query_vector)
            )
            score = sum(weight * large.get(feature, 0.0) for feature, weight in small.items())
            if score >= _LEXICAL_FLOOR:
                scored.append((document, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].name))
        return scored[:k]


@dataclass
class SkillRetriever:
    """Top-k skills for an objective, over whatever corpus is registered.

    The corpus is supplied by *providers* rather than pushed in, so a caller
    registers "where skills come from" once and the retriever re-reads them when
    the catalog generation changes. Pushing would mean every site that adds a
    skill has to remember to update the index, which is the coupling that makes
    an index go stale.
    """

    _providers: dict[str, callable[[], Iterable[SkillDocument]]] = field(default_factory=dict)
    _index: LexicalIndex = field(default_factory=LexicalIndex)
    _signature: tuple[str, ...] = ()
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _encoder: callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None
    _embeddings: list[list[float]] = field(default_factory=list)

    def install_encoder(
        self, encoder: callable[[Sequence[str]], Sequence[Sequence[float]]] | None
    ) -> None:
        """Install (or clear with ``None``) the semantic backend.

        ``encoder`` takes a list of strings and returns one vector per string.
        That signature is the whole contract: this module never loads a model,
        never leases a lane, and never decides when the encoder should exist.
        """
        with self._lock:
            self._encoder = encoder
            self._embeddings = []
            self._signature = ()

    def register_provider(
        self, name: str, provider: callable[[], Iterable[SkillDocument]]
    ) -> None:
        with self._lock:
            self._providers[str(name)] = provider

    def unregister_provider(self, name: str) -> None:
        with self._lock:
            self._providers.pop(str(name), None)

    def _collect_locked(self) -> list[SkillDocument]:
        documents: dict[str, SkillDocument] = {}
        for name, provider in self._providers.items():
            try:
                produced = list(provider() or ())
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("skill provider %r failed: %s", name, exc)
                continue
            for document in produced:
                if not isinstance(document, SkillDocument) or not document.name:
                    continue
                # First provider wins on a name clash. Providers are registered
                # in priority order, and a learned macro must not shadow the
                # shipped skill it was named after.
                documents.setdefault(document.name, document)
        return sorted(documents.values(), key=lambda d: d.name)

    def _refresh_locked(self) -> None:
        documents = self._collect_locked()
        signature = tuple(f"{d.name}\x00{d.description}" for d in documents)
        if signature == self._signature:
            return
        self._index.rebuild(documents)
        self._signature = signature
        self._embeddings = []
        if self._encoder is not None and documents:
            self._embeddings = self._encode_locked([d.text for d in documents])

    def _encode_locked(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed and L2-normalise, or drop the encoder and carry on lexically."""
        try:
            raw = self._encoder(list(texts))  # type: ignore[misc]
            vectors = [[float(v) for v in row] for row in raw]
        except (ArithmeticError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "skill retrieval encoder failed; continuing on the lexical index: %s", exc
            )
            self._encoder = None
            return []
        if len(vectors) != len(texts) or any(not v for v in vectors):
            logger.warning(
                "skill retrieval encoder returned %d vectors for %d texts; "
                "continuing on the lexical index",
                len(vectors),
                len(texts),
            )
            self._encoder = None
            return []
        normalised: list[list[float]] = []
        for vector in vectors:
            norm = math.sqrt(sum(v * v for v in vector))
            normalised.append([v / norm for v in vector] if norm > 0.0 else vector)
        return normalised

    def retrieve(self, objective: str, *, k: int = 5) -> list[RetrievedSkill]:
        """The k most relevant skills, or an empty list when none clear the floor."""
        query = str(objective or "").strip()
        if not query or k <= 0:
            return []
        with self._lock:
            self._refresh_locked()
            if self._encoder is not None and self._embeddings:
                hits = self._semantic_search_locked(query, k)
                if hits is not None:
                    return hits
            documents_and_scores = self._index.search(query, k)
        return [
            RetrievedSkill(
                name=document.name, score=score, source=document.source, backend="lexical"
            )
            for document, score in documents_and_scores
        ]

    def _semantic_search_locked(self, query: str, k: int) -> list[RetrievedSkill] | None:
        """Cosine over embeddings, or None to fall through to lexical."""
        encoded = self._encode_locked([query])
        if not encoded:
            return None
        query_vector = encoded[0]
        documents = self._index.documents
        if len(documents) != len(self._embeddings):
            return None
        scored = [
            (document, sum(a * b for a, b in zip(query_vector, embedding)))
            for document, embedding in zip(documents, self._embeddings)
        ]
        # Cosine over normalised vectors is in [-1, 1]; anything at or below
        # zero is unrelated or opposed, and offering it would put the retriever
        # back to always returning k results whatever was asked.
        scored = [pair for pair in scored if pair[1] > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0].name))
        return [
            RetrievedSkill(
                name=document.name, score=score, source=document.source, backend="semantic"
            )
            for document, score in scored[:k]
        ]

    def corpus_size(self) -> int:
        with self._lock:
            self._refresh_locked()
            return len(self._index.documents)

    def report(self) -> dict[str, object]:
        with self._lock:
            self._refresh_locked()
            return {
                "providers": sorted(self._providers),
                "corpus_size": len(self._index.documents),
                "backend": "semantic" if self._encoder is not None else "lexical",
                "embedded": len(self._embeddings),
                "sources": sorted({d.source for d in self._index.documents}),
            }


_retriever: SkillRetriever | None = None
_retriever_lock = checked_lock("core.skills.skill_retrieval")


def get_skill_retriever() -> SkillRetriever:
    """The process-wide retriever. Created on first use, never reset implicitly."""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = SkillRetriever()
    return _retriever

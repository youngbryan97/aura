"""Deciding what evidence a turn needs, by meaning rather than by phrasing.

Bryan, live 2026-08-04: "a lot of these requests are tied into specific
phrases and that shouldn't be the case. part of her reasoning has to include
general associations and a general understanding of what is being asked."

He was right, and he had the evidence. "Which file in your repository does
that function live in?" reached her source and was answered correctly.
"What python module is that from" — the same question — did not, because it
missed a regex. A keyword gate that misses does not merely mis-route: it
leaves her blind to something she can actually see, and then the phrasing
IS the behaviour.

So relevance is measured, not matched. Each kind of evidence is described by
a few sentences that say what the KIND OF QUESTION is about, the request is
embedded in the same space (MiniLM, local, already resident), and the
decision is a cosine distance. Paraphrases she has never seen land near the
concept; a question about arithmetic does not.

The anchors are concept descriptions, not triggers — nothing here fires
because a particular word appeared. The distinction that matters for Bryan's
objection: adding a new way to ask "show me your code" requires no change
here, because the sentence does not have to resemble any of these, only to
MEAN something similar.

Lexical patterns remain as a floor, and only as a floor: when the embedding
model is unavailable or has fallen back to hashing — where cosine distance
carries no meaning — a missed gate would put her back to answering from
weights. Degrading to the old behaviour is acceptable; degrading to blindness
is not.
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Cognition.EvidenceRelevance")

__all__ = [
    "semantic_routing_ready",
    "warm_semantic_routing",
    "PHYSICAL_PERCEPTION",
    "SCREEN_PERCEPTION",
    "OWN_SOURCE",
    "SOURCE_PROVENANCE",
    "OTHER_SOURCE",
    "EXTERNAL_WORLD",
    "EvidenceAlignment",
    "assess_evidence_alignment",
    "relevance",
    "prewarm_evidence_relevance",
    "wants_evidence",
    "semantic_routing_available",
]

#: Questions whose answer depends on sensing the person's present physical
#: surroundings. This is deliberately separate from screen perception: pixels
#: from a desktop cannot establish who is in the room, what somebody is
#: holding, or what is in front of the camera.
PHYSICAL_PERCEPTION = "physical_perception"

#: Questions about what is on the person's screen, in several unrelated
#: phrasings. These describe the CONCEPT; the match is by meaning.
SCREEN_PERCEPTION = "screen_perception"

#: Questions about Aura's own implementation — her files, her functions,
#: where a piece of her code lives.
OWN_SOURCE = "own_source"

#: Asking where something she ALREADY put on the table came from, or whether
#: it was real. Its subject is the previous turn, not this one, so it rarely
#: repeats any of the words that made the first request — which is exactly
#: why a pattern kept missing it. Live 2026-08-04, "Where did you get that
#: from?" and "Where in the codebase can I find that" both matched nothing,
#: so the citation she had on record was never spoken and the turn was left
#: to the model, which then failed a reliability gate and served nothing.
SOURCE_PROVENANCE = "source_provenance"

#: Somebody ELSE's code. Nothing requests this kind — it exists to be a
#: better match than :data:`OWN_SOURCE` when the question is about a library,
#: so the dominance test drops her source rather than answering "show me the
#: actual code for numpy" by checking numpy against HER tree, correctly
#: finding it absent, and substituting a piece of herself for it.
#:
#: A contrast sentence in the baseline set was the wrong instrument: phrased
#: near "show me code" it suppressed every genuine request too, including
#: "share a snippet of code and let me know where it's from". Whose code it
#: is was never a question of degree — it is a rival reading, and this
#: module already knows how to let one reading beat another.
OTHER_SOURCE = "third_party_source"

#: Facts whose source of truth is outside Aura and the local machine: named
#: organizations and people, current events, releases, prices, schedules, and
#: other claims that memory alone cannot establish.  This is distinct from a
#: timeless explanation *about* an external-world concept.
EXTERNAL_WORLD = "external_world"

_ANCHORS: dict[str, tuple[str, ...]] = {
    PHYSICAL_PERCEPTION: (
        "look at the physical surroundings right now and report what is there",
        "use a present sensor reading to determine whether another person is nearby",
        "tell who or what is currently in the room with the person",
        "inspect what the person is holding or showing in front of the camera",
        "determine from current physical perception what is happening around us",
        "which of your senses can actually establish what is present here now",
        "take a fresh look rather than answer from memory or general knowledge",
    ),
    SCREEN_PERCEPTION: (
        "what is currently displayed on the computer screen",
        "describe the windows and applications that are open right now",
        "what can you see in front of you on the monitor",
        "is a particular thing visible somewhere on the display",
        "what was on the screen a moment ago",
        "what is hidden behind or underneath the front window",
        "what are you looking at at this moment",
    ),
    OWN_SOURCE: (
        "show me a piece of your own program code",
        "which file of your implementation does that function live in",
        "what module or path in your repository contains that code",
        "where in your codebase is that written",
        "let me see the actual source you are built from",
        # Asking what a part of her IS, without ever saying "code". Live
        # 2026-08-04 "what does your memory system actually look like"
        # scored 0.08 here and 0.26 against the screen — "look like" read
        # as a request to LOOK — so a question about her implementation
        # pulled a window listing.
        "what does that part of you look like on the inside",
        "how is your memory system actually implemented",
        "walk me through how you really do that internally",
        # Asking her to OPEN something of hers, rather than to show code.
        # The act is the same; none of the words are.
        "open one of your own files and read me part of it",
        "pull up something you are built out of",
    ),
    OTHER_SOURCE: (
        "show me the source code of the numpy library",
        "how is that implemented inside the pandas package",
        "what does the standard library do in that function",
        "show me how that open source project wrote it",
    ),
    SOURCE_PROVENANCE: (
        "where did you get that from",
        "which file did that come out of",
        "where can I find that in the repository",
        "is that real code or did you make it up",
        "can I look that up on github myself",
        "what is the path to the thing you just showed me",
        "did you read that somewhere or write it just now",
    ),
    EXTERNAL_WORLD: (
        "find current facts about a specifically named company and cite reliable sources",
        "who founded this named organization and when did it happen",
        "what did this company announce or release recently",
        "check the present price schedule weather score or office holder",
        "look up a named person product institution or event outside this machine",
        "verify a factual claim about a specific external entity against public evidence",
    ),
}

#: Sentences that are emphatically NOT about the above, used as a contrast
#: set. A question is routed to evidence only when it is closer to the
#: concept than to ordinary conversation — an absolute threshold alone drifts
#: with how verbose the person happens to be.
_BASELINE_ANCHORS: tuple[str, ...] = (
    "what is seventeen multiplied by four",
    "how are you feeling today",
    "tell me a joke about penguins",
    "write a python function that sorts a list",
    "what did we decide about the schedule",
    "explain how photosynthesis works",
    # Asking her to WRITE ABOUT a subject is a different act from asking her
    # to LOOK at something or to open a file. Without this contrast, "give me
    # two concise sentences about reliable desktop tool use" scored as a
    # question about the desktop and pulled a screen reading into a request
    # for prose.
    "write a couple of concise sentences about a general principle",
    "summarise a topic briefly in your own words",
    # Asking what she CAN DO is not asking to see what she is made of.
    # Live: "what external tools could you use from the live desktop
    # path" scored as a question about her source and had a capability
    # answer replaced with a file excerpt.
    "what tools and capabilities do you have available to use",
    "describe what you are able to do for me",
)

# Near-neighbour contrasts belong to one concept, not the global baseline. A
# physical-perception contrast such as "how cameras work" must not make a
# source-code paraphrase harder to route. Conversely, generic visual language
# must not activate a present sensor merely because it is close to "look".
_CONTRAST_ANCHORS: dict[str, tuple[str, ...]] = {
    PHYSICAL_PERCEPTION: (
        "answer a general question about the visible world from knowledge",
        "discuss camera or microphone hardware without using it now",
        "describe what kinds of sensors a robot could use in general",
        "explain a visual metaphor rather than inspect the surroundings",
    ),
    EXTERNAL_WORLD: (
        "explain a timeless engineering concept from established knowledge",
        "describe a general tradeoff in an AI system architecture",
        "reason about a hypothetical design without looking up current facts",
        "answer a mathematics logic or programming question",
        "discuss a broad scientific idea rather than a named current entity",
        "explain how a CPU architecture works in general",
    ),
}

#: How much closer to the concept than to ordinary talk a request must be.
#: Calibrated against the live 2026-08-04 transcript: every phrasing Bryan
#: actually used, plus the unrelated turns from the same conversation that
#: must NOT pull evidence in.
_MARGIN = 0.12

# Calibrated per evidence family. Short source/provenance follow-ups naturally
# contain little semantic material ("where can it be found?") and need a lower
# absolute margin than a privacy-sensitive decision to activate a camera.
_KIND_MARGINS: dict[str, float] = {
    # Leave-one-out centroid calibration on the current local encoder puts the
    # balanced physical-perception boundary at 0.0368 (balanced accuracy
    # 0.929).  The former 0.12 belonged to the retired embedding geometry and
    # rejected ordinary requests for a current reading.
    PHYSICAL_PERCEPTION: 0.0368,
    # The weakest retained multi-intent screen reading is +0.0398 after the
    # embedding batch-invariance repair; the strongest unrelated prose case
    # is -0.053.  Keep a measured gap on both sides instead of rounding the
    # positive case out of its own class.
    SCREEN_PERCEPTION: 0.035,
    OWN_SOURCE: 0.02,
    SOURCE_PROVENANCE: 0.04,
    OTHER_SOURCE: 0.04,
    # Re-measured after MPS batch invariance was restored. Entity/current-fact
    # positives are +0.0416..+0.0882; the architecture contrasts are
    # -0.2073..-0.2521. Explicit current/search language remains an independent
    # structural floor for short local-event questions below this semantic
    # boundary.
    EXTERNAL_WORLD: 0.04,
}

#: How far behind the best-matching concept a kind may fall and still be
#: considered part of what was asked.
_DOMINANCE = 0.20

_LOCK = checked_lock("evidence_relevance.cache")
_ANCHOR_CACHE: dict[str, Any] = {}
_REQUEST_CACHE: dict[str, Any] = {}
_ALIGNMENT_QUERY_CACHE: dict[str, Any] = {}
_SHARED_ENGINE_LEASE: Any | None = None
_CACHE_ENGINE_TOKEN: tuple[str, str, int, int] | None = None
#: Bounded: this is a per-turn lookup, not a store.
_REQUEST_CACHE_MAX = 256

_ALIGNMENT_POSITIVES: tuple[tuple[str, str], ...] = (
    (
        "Who founded Hugging Face?",
        "Hugging Face was founded by Clement Delangue, Julien Chaumond, and Thomas Wolf.",
    ),
    (
        "What is the latest Mistral model release?",
        "Mistral announced a new model release this week.",
    ),
    (
        "What events are happening in San Jose this weekend?",
        "San Jose hosts a downtown arts festival this Saturday.",
    ),
    (
        "What is one tradeoff in hybrid recurrent transformer architectures?",
        "Hybrid recurrent designs reduce growing KV-cache memory but make state recovery harder.",
    ),
)
_ALIGNMENT_NEGATIVES: tuple[tuple[str, str], ...] = (
    (
        "Who founded Hugging Face?",
        "A tomato sauce improves when onions are browned before the liquid is added.",
    ),
    (
        "What is the latest Mistral model release?",
        "Ocean Network Express provides international container shipping services.",
    ),
    (
        "What events are happening in San Jose this weekend?",
        "The Antarctic ice sheet contains most of Earth's fresh water.",
    ),
    (
        "What is one tradeoff in hybrid recurrent transformer architectures?",
        "Ocean Network Express provides cargo tracking and shipping schedules.",
    ),
)

# Measured on the local evidence encoder shipped with Aura, 2026-08-24:
# matched query-document pairs scored 0.6892..0.8783 and mismatched pairs
# 0.1291..0.3173.  The midpoint leaves at least 0.18 cosine margin on either
# side.  The calibration cohort stays beside the value so a model migration
# can remeasure it rather than inheriting it silently.
_EVIDENCE_ALIGNMENT_BOUNDARY = 0.50


@dataclass(frozen=True, slots=True)
class EvidenceAlignment:
    """Whether retrieved material addresses the request that authorized it."""

    relevant: bool
    measured: bool
    score: float | None
    boundary: float | None
    lexical_overlap: tuple[str, ...]
    reason: str


def _engine_token(engine: Any) -> tuple[str, str, int, int]:
    """Identity of one vector space and the object currently producing it."""

    kind = f"{type(engine).__module__}.{type(engine).__qualname__}"
    model = str(
        getattr(engine, "PREFERRED_MODEL", "")
        or getattr(engine, "model_name", "")
        or getattr(engine, "identity", "")
    )
    try:
        width = int(getattr(engine, "VECTOR_DIM", 0) or 0)
    except (TypeError, ValueError):
        width = 0
    return kind, model, width, id(engine)


def _bind_cache_to_engine(engine: Any) -> None:
    """Prevent cosines between vectors produced by different engines.

    The container may install or remove its vector-memory engine after this
    module has warmed.  Anchor and request caches are process globals, so an
    engine switch previously left old anchors beside new request vectors.
    Their widths still matched and the resulting cosine looked valid.  Bind
    both caches to the producing object and invalidate them as one operation.
    """

    global _CACHE_ENGINE_TOKEN
    token = _engine_token(engine)
    with _LOCK:
        if _CACHE_ENGINE_TOKEN == token:
            return
        _ANCHOR_CACHE.clear()
        _REQUEST_CACHE.clear()
        _ALIGNMENT_QUERY_CACHE.clear()
        _CACHE_ENGINE_TOKEN = token


def _embedder() -> Any | None:
    """The live embedding engine, or None when there is not one."""

    global _SHARED_ENGINE_LEASE
    engine: Any | None = None
    try:
        from core.container import get_container

        memory = get_container().get("vector_memory_engine", default=None)
        engine = getattr(memory, "embedder", None)
    except (ImportError, AttributeError, RuntimeError, LookupError):
        pass
    if engine is None:
        with _LOCK:
            engine = _SHARED_ENGINE_LEASE
        if engine is None:
            try:
                from core.memory.embedding_runtime import acquire_shared_embedding_engine

                candidate = acquire_shared_embedding_engine("evidence-relevance")
            except (ImportError, AttributeError, RuntimeError):
                return None
            with _LOCK:
                if _SHARED_ENGINE_LEASE is None:
                    _SHARED_ENGINE_LEASE = candidate
                else:
                    candidate.close()
                engine = _SHARED_ENGINE_LEASE
    if engine is None:
        return None
    _bind_cache_to_engine(engine)
    return engine


#: Set once a warm has been asked for, so a turn asks at most once.
_WARMING = {"asked": False}


def semantic_routing_ready() -> bool:
    """Whether the embedding model is loaded ALREADY, loading nothing.

    The routing question at the end of `sight_intent.classify` runs on every
    turn the cheap lexical rules do not settle, and answering it called
    `semantic_routing_available`, which checks the model out — that is, loads
    it. The first turn after a restart paid for that load in the foreground,
    before anything had been said back.

    A readiness question must not do the thing it is asking about.
    """
    embedder = _embedder()
    if embedder is None:
        return False
    return getattr(embedder, "_model", None) is not None


def warm_semantic_routing() -> bool:
    """Load the embedding model. For a background task or boot, never a turn."""
    try:
        return semantic_routing_available()
    except (AttributeError, RuntimeError, OSError):
        return False


def _ask_for_a_warm() -> None:
    """Get the model loaded off the critical path, once."""
    if _WARMING["asked"]:
        return
    _WARMING["asked"] = True
    try:
        import asyncio

        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.run_in_executor(None, warm_semantic_routing)


def semantic_routing_available() -> bool:
    """Whether cosine distance means anything here right now.

    The engine falls back to a character-hash embedding when
    sentence-transformers is missing. Hash vectors are stable and
    meaningless: two paraphrases are as far apart as two unrelated
    sentences. Routing on them would be worse than the lexical floor,
    because it would look like it was working.
    """
    embedder = _embedder()
    if embedder is None:
        return False
    try:
        embedder._checkout_model()  # noqa: SLF001 - availability probe
    except (AttributeError, RuntimeError, OSError):
        return False
    try:
        model = getattr(embedder, "_model", None)
    finally:
        try:
            embedder._return_model()  # noqa: SLF001
        except (AttributeError, RuntimeError):
            pass
    return model is not None


def _embed(text: str, *, as_query: bool = False) -> Any | None:
    """Embed one side of an evidence comparison.

    The claim is the QUERY side and carries the "evidence" instruction; the
    candidate evidence is the DOCUMENT side and carries none. That instruction
    matters here more than anywhere else in the tree: the encoder's shipped
    default asks for "passages that answer the query", which quietly demotes
    evidence that CONTRADICTS the claim — the one thing an audit most needs to
    surface. See TASK_INSTRUCTIONS in core/memory/embedding_model.py.
    """
    embedder = _embedder()
    if embedder is None:
        return None
    try:
        if as_query:
            embed_query = getattr(embedder, "embed_query", None)
            if callable(embed_query):
                return embed_query(str(text or ""), task="evidence")
        return embedder.embed(str(text or ""))
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        logger.debug("Evidence relevance embedding failed: %s", exc)
        return None


def _embed_documents(texts: Sequence[str]) -> list[Any | None]:
    """Encode document-side evidence anchors in one model invocation.

    The routing catalogue is a fixed document cohort. Encoding each sentence
    separately paid the transformer setup cost for every anchor and made chat
    readiness wait tens of seconds. Keep result cardinality stable so callers
    can bind each vector to the exact sentence that produced it.
    """

    wanted = [str(text or "") for text in texts]
    if not wanted:
        return []
    embedder = _embedder()
    if embedder is not None:
        embed_batch = getattr(embedder, "embed_batch", None)
        if callable(embed_batch):
            try:
                vectors = list(embed_batch(wanted))
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                logger.debug("Evidence relevance batch embedding failed: %s", exc)
            else:
                if len(vectors) == len(wanted):
                    return vectors
                logger.warning(
                    "Evidence relevance batch cardinality mismatch: wanted=%d got=%d",
                    len(wanted),
                    len(vectors),
                )
    return [_embed(text) for text in wanted]


def _cosine(left: Any, right: Any) -> float:
    try:
        dot = float(
            sum(float(a) * float(b) for a, b in zip(left, right, strict=False))
        )
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    except (TypeError, ValueError):
        return 0.0
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _centroid(vectors: Sequence[Any]) -> tuple[float, ...]:
    """Mean direction for one declared semantic class."""

    if not vectors:
        return ()
    rows = [tuple(float(value) for value in vector) for vector in vectors]
    width = min((len(row) for row in rows), default=0)
    if width <= 0:
        return ()
    scale = 1.0 / float(len(rows))
    return tuple(sum(row[index] for row in rows) * scale for index in range(width))


_ALIGNMENT_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'’-]*", re.IGNORECASE)
_ALIGNMENT_STOPWORDS = frozenset(
    """
    a about an and are as at be by can did do does for from had has have how i
    in into is it its me more of on one or our should that the their them there
    they this to was were what when where which who why will with would you your
    """.split()
)


def _alignment_terms(text: str) -> set[str]:
    return {
        token
        for token in (
            match.group(0).casefold()
            for match in _ALIGNMENT_TOKEN_RE.finditer(str(text or ""))
        )
        if len(token) >= 3 and token not in _ALIGNMENT_STOPWORDS
    }


def assess_evidence_alignment(request: Any, evidence: Any) -> EvidenceAlignment:
    """Measure whether one retrieved passage can answer ``request``.

    The boundary is derived from declared matched and mismatched query-document
    pairs encoded by the same local evidence model used at runtime.  If that
    model is unavailable, one shared content term is the conservative floor:
    absence cannot prove relevance, so unmatched material is withheld.
    """

    query = " ".join(str(request or "").split())
    passage = " ".join(str(evidence or "").split())
    query_terms = _alignment_terms(query)
    passage_terms = _alignment_terms(passage)
    overlap = tuple(sorted(query_terms & passage_terms))
    if not query or not passage:
        return EvidenceAlignment(False, False, None, None, overlap, "empty")

    if semantic_routing_ready():
        boundary = _EVIDENCE_ALIGNMENT_BOUNDARY
        # A search result set has one query and several documents. The old
        # loop paid one identical query forward per result. Keep the query in
        # its asymmetric retrieval geometry, but cache it independently from
        # the symmetric intent vectors.
        with _LOCK:
            query_vector = _ALIGNMENT_QUERY_CACHE.get(query)
            engine_before = _CACHE_ENGINE_TOKEN
        if query_vector is None:
            query_vector = _embed(query, as_query=True)
            if query_vector is not None:
                with _LOCK:
                    if engine_before == _CACHE_ENGINE_TOKEN:
                        if len(_ALIGNMENT_QUERY_CACHE) >= _REQUEST_CACHE_MAX:
                            _ALIGNMENT_QUERY_CACHE.clear()
                        _ALIGNMENT_QUERY_CACHE[query] = query_vector
        document_vector = _embed_documents([passage])[0]
        with _LOCK:
            same_engine = engine_before == _CACHE_ENGINE_TOKEN
        if (
            boundary is not None
            and same_engine
            and query_vector is not None
            and document_vector is not None
        ):
            score = _cosine(query_vector, document_vector)
            return EvidenceAlignment(
                score >= boundary,
                True,
                score,
                boundary,
                overlap,
                "semantic_match" if score >= boundary else "semantic_mismatch",
            )

    return EvidenceAlignment(
        bool(overlap),
        False,
        None,
        None,
        overlap,
        "lexical_overlap" if overlap else "no_measured_alignment",
    )


def _anchor_vectors(key: str, sentences: Sequence[str]) -> list[Any]:
    with _LOCK:
        cached = _ANCHOR_CACHE.get(key)
    if cached is not None:
        return cached
    vectors = [vector for vector in _embed_documents(sentences) if vector is not None]
    with _LOCK:
        _ANCHOR_CACHE[key] = vectors
    return vectors


def _prewarm_anchor_vectors() -> int:
    """Materialize every missing anchor cache from one deduplicated batch."""

    cohorts: dict[str, tuple[str, ...]] = {}
    for kind, anchors in _ANCHORS.items():
        cohorts[kind] = tuple(anchors)
        cohorts[f"__baseline__:{kind}"] = tuple(
            _BASELINE_ANCHORS + _CONTRAST_ANCHORS.get(kind, ())
        )
    with _LOCK:
        missing = {
            key: sentences
            for key, sentences in cohorts.items()
            if key not in _ANCHOR_CACHE
        }
    if not missing:
        return 0

    unique_sentences = tuple(
        dict.fromkeys(sentence for sentences in missing.values() for sentence in sentences)
    )
    vectors = _embed_documents(unique_sentences)
    if len(vectors) != len(unique_sentences) or any(vector is None for vector in vectors):
        raise RuntimeError("semantic evidence routing batch is incomplete")
    by_sentence = dict(zip(unique_sentences, vectors, strict=True))
    with _LOCK:
        for key, sentences in missing.items():
            _ANCHOR_CACHE.setdefault(key, [by_sentence[sentence] for sentence in sentences])
    return len(unique_sentences)


def _request_vector(request: str) -> Any | None:
    """Embed an utterance in the same geometry as the intent catalogue.

    Evidence retrieval is asymmetric: a query asks which document answers it.
    Intent classification is symmetric: an utterance and an intent description
    are two sentences whose meanings are compared.  Sending the utterance
    through the retrieval query adapter while the catalogue used document
    vectors made the adapter's task instruction part of the classification.
    On the current encoder, that turned a request to *write about* desktop
    tools into a request to inspect the screen.
    """

    with _LOCK:
        cached = _REQUEST_CACHE.get(request)
    if cached is not None:
        return cached
    vector = _embed(request)
    if vector is None:
        return None
    with _LOCK:
        if len(_REQUEST_CACHE) >= _REQUEST_CACHE_MAX:
            _REQUEST_CACHE.clear()
        _REQUEST_CACHE[request] = vector
    return vector


def _relevance_once(text: str, kind: str) -> tuple[float, bool]:
    """One score and whether every vector came from one engine generation."""

    _embedder()
    with _LOCK:
        engine_before = _CACHE_ENGINE_TOKEN
    vector = _request_vector(text)
    if vector is None:
        return 0.0, True
    concept = _anchor_vectors(kind, _ANCHORS[kind])
    baseline_sentences = _BASELINE_ANCHORS + _CONTRAST_ANCHORS.get(kind, ())
    baseline = _anchor_vectors(f"__baseline__:{kind}", baseline_sentences)
    with _LOCK:
        same_engine = engine_before == _CACHE_ENGINE_TOKEN
    if not same_engine:
        return 0.0, False
    if not concept or not baseline:
        return 0.0, True
    concept_center = _centroid(concept)
    baseline_center = _centroid(baseline)
    if not concept_center or not baseline_center:
        return 0.0, True
    return (
        _cosine(vector, concept_center) - _cosine(vector, baseline_center),
        True,
    )


def relevance(request: Any, kind: str) -> float:
    """How much closer this request is to ``kind`` than to ordinary talk.

    Positive means the concept fits better than small talk does. Returns
    0.0 when nothing can be measured, so a caller falls through to its
    floor rather than acting on a number that means nothing.
    """
    text = " ".join(str(request or "").split())
    if not text or kind not in _ANCHORS:
        return 0.0
    # A class is the shared direction across its examples, not whichever one
    # sentence happens to share a noun with the request.  Nearest-example
    # scoring made "CPU architecture" look like Aura's own source because one
    # implementation anchor dominated the rest of that concept.
    score, stable = _relevance_once(text, kind)
    if stable:
        return score
    # The container can install its vector-memory engine while boot warming is
    # still using the retained shared lease. Retry once after the cache was
    # rebound; never compare local vectors across that transition.
    score, stable = _relevance_once(text, kind)
    return score if stable else 0.0


def prewarm_evidence_relevance() -> dict[str, Any]:
    """Materialize every semantic routing surface before chat is advertised.

    A resident embedding model is not enough: every evidence family has an
    encoded concept and contrast set. Constructing those on the first user
    turn previously consumed tens of seconds before generation even began.
    """

    started = time.perf_counter()
    if not semantic_routing_available():
        raise RuntimeError("semantic evidence routing is unavailable")

    encoded_documents = _prewarm_anchor_vectors()
    dimensions: dict[str, dict[str, int]] = {}
    for kind, anchors in _ANCHORS.items():
        concept = _anchor_vectors(kind, anchors)
        baseline_sentences = _BASELINE_ANCHORS + _CONTRAST_ANCHORS.get(kind, ())
        baseline = _anchor_vectors(f"__baseline__:{kind}", baseline_sentences)
        if not concept or not baseline:
            raise RuntimeError(f"semantic evidence routing cache is empty for {kind}")
        dimensions[kind] = {
            "concept_vectors": len(concept),
            "baseline_vectors": len(baseline),
        }

    # Exercise the same symmetric sentence encoder the live intent path uses
    # before declaring readiness. Query-side warmup belongs to evidence
    # retrieval and is performed when query-document alignment is measured.
    probe = _request_vector("How are you feeling today?")
    if probe is None:
        raise RuntimeError("semantic evidence query encoder is unavailable")
    return {
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "families": dimensions,
        "query_dimensions": len(probe),
        "encoded_documents": encoded_documents,
        "evidence_alignment_boundary": _EVIDENCE_ALIGNMENT_BOUNDARY,
    }


def wants_evidence(
    request: Any,
    kind: str,
    *,
    lexical_floor: Callable[[str], bool] | None = None,
    margin: float = _MARGIN,
) -> bool:
    """Whether this turn should be given ``kind`` evidence.

    Meaning decides. ``lexical_floor`` is consulted as a floor — it can add
    a turn the embedding missed, and it is the whole decision when semantic
    routing is unavailable — but it can never veto one the meaning found.
    """
    text = " ".join(str(request or "").split())
    if not text:
        return False

    floor = False
    if lexical_floor is not None:
        try:
            floor = bool(lexical_floor(text))
        except (RuntimeError, TypeError, ValueError):
            floor = False
    if floor:
        return True
    # Never load a model to answer this. A turn that arrives before the
    # embedding is warm gets the lexical floor and asks for a warm in the
    # background; the turn after it gets meaning.
    if not semantic_routing_ready():
        _ask_for_a_warm()
        return False
    score = relevance(text, kind)
    required_margin = (
        _KIND_MARGINS.get(kind, margin)
        if margin == _MARGIN
        else margin
    )
    if score < required_margin:
        return False
    # Competitive, not independent. "What's on my screen?" scores +0.59
    # against perception and +0.09 against her source — both above an
    # absolute floor, and only one of them is the question. A kind that is
    # plainly beaten by another kind is not what was asked about, and
    # attaching both would put a file listing in front of a question about
    # a window. A turn that genuinely concerns two things scores near-level
    # on both and keeps them.
    best = max(relevance(text, other) for other in _ANCHORS)
    return score >= best - _DOMINANCE

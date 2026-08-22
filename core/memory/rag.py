from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any

from core.memory import embedding_model, retrieval_calibration
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.RAG")

# ── Semantic layer (July capability raise) ────────────────────────────────
# Real dense embeddings (sentence-transformers MiniLM via the existing
# EmbeddingEngine) blended with TF-IDF. The fallback chain is honest:
# semantic+lexical hybrid → TF-IDF cosine → never substring again.
# Bounded: per-text vectors are LRU-cached; a query never embeds more than
# _SEMANTIC_MAX_UNCACHED cold texts synchronously (beyond that, this query
# uses TF-IDF while a background thread warms the cache for the next one).

_SEMANTIC_CACHE: OrderedDict[str, Any] = OrderedDict()
_SEMANTIC_CACHE_MAX = 4096
_SEMANTIC_MAX_UNCACHED = 128
_SEMANTIC_WARM_COHORT = 4
_SEMANTIC_LOCK = threading.Lock()
_EMBED_ENGINE: Any = None
_EMBED_ENGINE_FAILED = False
_WARM_INFLIGHT = False
#: Set while the dense engine is being built off-loop, so concurrent callers
#: start exactly one warm thread instead of one each.
_ENGINE_WARM_INFLIGHT = False


_SEMANTIC_RAG_FLAG = None


def _semantic_enabled() -> bool:
    global _SEMANTIC_RAG_FLAG
    if _SEMANTIC_RAG_FLAG is None:
        from core.runtime.flags import FlagKind, declare

        _SEMANTIC_RAG_FLAG = declare(
            "AURA_SEMANTIC_RAG",
            kind=FlagKind.BOOL,
            default=True,
            description="Hybrid dense+lexical retrieval in the RAG bridge; "
            "0 reverts to pure lexical TF-IDF",
            owner="core/memory/rag.py",
        )
    return bool(_SEMANTIC_RAG_FLAG.value())


def _on_event_loop() -> bool:
    """Whether this call is running on an asyncio event loop thread."""
    try:
        import asyncio

        asyncio.get_running_loop()
        return True
    except (RuntimeError, ImportError):
        return False


def _build_embed_engine() -> Any:
    """Construct and probe the dense embedder. Blocking; never raises."""
    global _EMBED_ENGINE, _EMBED_ENGINE_FAILED, _ENGINE_WARM_INFLIGHT
    with _SEMANTIC_LOCK:
        if _EMBED_ENGINE is not None or _EMBED_ENGINE_FAILED:
            _ENGINE_WARM_INFLIGHT = False
            return _EMBED_ENGINE
    try:
        from core.memory.embedding_runtime import acquire_shared_embedding_engine

        engine = acquire_shared_embedding_engine("rag")
        # This is the expensive call: it loads SentenceTransformer (~5s) and
        # is why the construction must never happen on the event loop.
        probe = engine.embed("semantic backend probe")
        ok = getattr(engine, "_model", None) is not None and probe is not None
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
        if "engine" in locals():
            engine.close()
        with _SEMANTIC_LOCK:
            _EMBED_ENGINE_FAILED = True
            _ENGINE_WARM_INFLIGHT = False
        logger.info("Semantic RAG backend failed to initialize (%s); TF-IDF only.", exc)
        return None
    with _SEMANTIC_LOCK:
        _ENGINE_WARM_INFLIGHT = False
        if not ok:
            _EMBED_ENGINE_FAILED = True
            logger.info("Semantic RAG backend unavailable; TF-IDF only.")
            engine.close()
            return None
        if _EMBED_ENGINE is None:
            _EMBED_ENGINE = engine
            return engine
        winner = _EMBED_ENGINE
    # Two non-loop callers may race through the expensive probe. They still
    # share one underlying model, but only the cached RAG lease remains owned.
    engine.close()
    return winner


def _get_embed_engine() -> Any:
    """The real dense embedder, or None. Never raises; one failure latches.

    Building the engine loads a SentenceTransformer model, which takes about
    five seconds. That used to happen inline on whatever thread asked first —
    including the event loop, where it froze the runtime for the whole load.
    Lockdep saw it once the vector engine's lifecycle lock became visible:

        loop_blocking_hold: 'vector_memory_engine.lifecycle_lock' held 5368ms
        on the event loop thread (limit 50ms)

    and mind_tick was declared stalled 33s later, which is what a user
    experiences as the app hanging on launch.

    The lock was never the defect — a five-second synchronous model load on
    the loop is. So on the loop we decline to build, start the build in a
    background thread, and answer with TF-IDF for this query. The next query
    gets the dense engine. Crucially this does NOT latch _EMBED_ENGINE_FAILED:
    declining is not failing, and treating it as failure would disable
    semantic retrieval permanently for the lifetime of the process.
    """
    global _ENGINE_WARM_INFLIGHT
    if _EMBED_ENGINE_FAILED or not _semantic_enabled():
        return None
    if _EMBED_ENGINE is not None:
        return _EMBED_ENGINE

    if not _on_event_loop():
        return _build_embed_engine()

    with _SEMANTIC_LOCK:
        if _EMBED_ENGINE is not None or _EMBED_ENGINE_FAILED:
            return _EMBED_ENGINE
        already_warming = _ENGINE_WARM_INFLIGHT
        _ENGINE_WARM_INFLIGHT = True
    if not already_warming:
        logger.info("Semantic RAG engine warming off-loop; this query uses TF-IDF.")
        threading.Thread(
            target=_build_embed_engine,
            name="rag-embed-engine-warm",
            daemon=True,
        ).start()
    return None


def _text_key(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).hexdigest()


def _cached_vector(text: str) -> Any:
    key = _text_key(text)
    with _SEMANTIC_LOCK:
        vec = _SEMANTIC_CACHE.get(key)
        if vec is not None:
            _SEMANTIC_CACHE.move_to_end(key)
        return vec


def _store_vectors(texts: list[str], vectors: Any) -> None:
    with _SEMANTIC_LOCK:
        for text, vec in zip(texts, vectors, strict=False):
            _SEMANTIC_CACHE[_text_key(text)] = vec
            _SEMANTIC_CACHE.move_to_end(_text_key(text))
        while len(_SEMANTIC_CACHE) > _SEMANTIC_CACHE_MAX:
            _SEMANTIC_CACHE.popitem(last=False)


def _background_embedding_should_defer() -> bool:
    try:
        from core.runtime.backpressure import primary_inference_active

        return bool(primary_inference_active())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.info(
            "Semantic cache warm yielded because foreground state could not be read: %s",
            exc,
        )
        return True


def _embed_document_views(
    engine: Any,
    texts: list[str],
    *,
    background: bool,
) -> list[Any]:
    method = getattr(engine, "embed_document_views", None)
    if callable(method):
        return list(method(texts, background=background))
    vectors = engine.embed_batch(texts)
    try:
        import numpy as np

        return [np.asarray([vector], dtype=np.float32) for vector in vectors]
    except (ImportError, TypeError, ValueError):
        return [[vector] for vector in vectors]


def _warm_cache_in_background(texts: list[str]) -> None:
    """One background warm at a time; the NEXT query gets the semantic path."""
    global _WARM_INFLIGHT
    if _background_embedding_should_defer():
        logger.info("Semantic cache warm deferred before launch for foreground work.")
        return
    with _SEMANTIC_LOCK:
        if _WARM_INFLIGHT:
            return
        _WARM_INFLIGHT = True

    def _warm() -> None:
        global _WARM_INFLIGHT
        try:
            engine = _get_embed_engine()
            if engine is not None and texts:
                unique = list(dict.fromkeys(texts))
                for start in range(0, len(unique), _SEMANTIC_WARM_COHORT):
                    if _background_embedding_should_defer():
                        logger.info(
                            "Semantic cache warm yielded to foreground after %d/%d texts.",
                            start,
                            len(unique),
                        )
                        break
                    cohort = unique[start : start + _SEMANTIC_WARM_COHORT]
                    views = _embed_document_views(
                        engine,
                        cohort,
                        background=True,
                    )
                    _store_vectors(cohort, views)
        except (RuntimeError, AttributeError, ValueError, TypeError, OSError) as exc:
            if "foreground_inference_active" in str(exc):
                logger.info("Semantic cache warm yielded during a bounded encode.")
            else:
                logger.debug("Semantic cache warm failed: %s", exc)
        finally:
            with _SEMANTIC_LOCK:
                _WARM_INFLIGHT = False

    threading.Thread(target=_warm, name="SemanticRagWarm", daemon=True).start()


def _semantic_scores(query: str, texts: list[str], task: str | None = None) -> list[float] | None:
    """Dense cosine per text, or None when the semantic path must sit out.

    ``task`` selects the query-side instruction (see embedding_model). The
    document vectors are task-independent, which is why they stay cacheable
    across retrieval jobs while the query is re-encoded per call.
    """
    engine = _get_embed_engine()
    if engine is None or not texts:
        return None
    try:
        import numpy as np

        uncached = [t for t in dict.fromkeys(texts) if _cached_vector(t) is None]
        if len(uncached) > _SEMANTIC_MAX_UNCACHED:
            _warm_cache_in_background(uncached)
            return None
        if uncached:
            views = _embed_document_views(engine, uncached, background=False)
            _store_vectors(uncached, views)
        # The query goes through the asymmetric path; the documents above do
        # not. Same width, different prompt — see EmbeddingEngine.embed_query.
        _embed_query = getattr(engine, "embed_query", None)
        qvec = np.asarray(
            _embed_query(query, task=task) if callable(_embed_query) else engine.embed(query),
            dtype=np.float32,
        )
        qn = float(np.linalg.norm(qvec))
        if qn <= 1e-8:
            return None
        scores: list[float] = []
        for text in texts:
            document = _cached_vector(text)
            if document is None:
                scores.append(0.0)
                continue
            scores.append(_document_dense_score(qvec, document))
        return scores
    except (RuntimeError, AttributeError, ValueError, TypeError, OSError) as exc:
        logger.debug("Semantic scoring failed; TF-IDF only for this query: %s", exc)
        return None


def _document_dense_score(query: Any, document: Any) -> float:
    """Best semantic match across every bounded view of one source."""

    import numpy as np

    qvec = np.asarray(query, dtype=np.float32).reshape(-1)
    qnorm = float(np.linalg.norm(qvec))
    if qnorm <= 1e-8:
        return 0.0
    views = np.asarray(document, dtype=np.float32)
    if views.ndim == 1:
        views = views.reshape(1, -1)
    if views.ndim != 2 or views.shape[1] != qvec.shape[0]:
        return 0.0
    norms = np.linalg.norm(views, axis=1)
    valid = norms > 1e-8
    if not bool(np.any(valid)):
        return 0.0
    scores = np.dot(views[valid], qvec) / (norms[valid] * qnorm)
    return float(np.max(scores))


def reset_semantic_state_for_test() -> None:
    global _EMBED_ENGINE, _EMBED_ENGINE_FAILED, _WARM_INFLIGHT, _ENGINE_WARM_INFLIGHT
    with _SEMANTIC_LOCK:
        _SEMANTIC_CACHE.clear()
        engine, _EMBED_ENGINE = _EMBED_ENGINE, None
        _EMBED_ENGINE_FAILED = False
        _WARM_INFLIGHT = False
        _ENGINE_WARM_INFLIGHT = False
    if engine is not None:
        close = getattr(engine, "close", None)
        if callable(close):
            close()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in (text or "").split() if t.strip()]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Basic text chunking for RAG ingestion.

    ``chunk_size`` is bounded by what the encoder can actually read. Until
    12 Aug 2026 it was not: chunks of 500 and 800 words were handed to an
    encoder with a 256-token window, which silently dropped 64-77% of each
    one. The dense half of the hybrid score was ranking prefixes while the
    lexical half read whole documents. See core/memory/embedding_model.py.
    """
    if not text:
        return []
    ceiling = embedding_model.max_chunk_words()
    if chunk_size > ceiling:
        # Clamp rather than raise: a too-large chunk is a tuning mistake, not
        # a reason to lose the ingest. But it is never silent.
        record_degradation(
            "rag.chunk_text",
            ValueError(
                f"chunk_size={chunk_size} words exceeds the encoder window "
                f"({embedding_model.REPO_ID}, {embedding_model.MAX_INPUT_TOKENS} tokens "
                f"= ~{ceiling} words); clamping so the tail is not silently dropped"
            ),
            action=f"clamped chunk_size to {ceiling}",
        )
        chunk_size = ceiling
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 10)
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def compute_term_freq(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    total = float(len(tokens))
    return {k: v / total for k, v in counts.items()}


def compute_cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = float(sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys))
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def _idf_weights(doc_token_sets: list[set[str]]) -> dict[str, float]:
    """Smoothed inverse document frequency over the candidate memories."""
    import math

    n_docs = len(doc_token_sets)
    df: dict[str, int] = {}
    for tokens in doc_token_sets:
        for tok in tokens:
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log(1.0 + n_docs / (1.0 + count)) for tok, count in df.items()}


def _memory_text(memory: dict[str, Any]) -> str:
    """The memory's text, under whichever key the caller's store uses.

    `text` and `content` are both canonical in this codebase — the vault
    stores one and returns the other — so reading only one silently produced
    empty documents for half the callers.
    """
    for key in ("text", "content"):
        value = memory.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def retrieve_memories(
    query: str,
    memories: list[dict[str, Any]],
    top_k: int = 5,
    threshold: float = 0.01,
    task: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Hybrid semantic + lexical retrieval over memory texts.

    History (July 2026 external review): this was a bare substring match
    while ``compute_term_freq`` and ``compute_cosine_similarity`` sat unused
    directly above it. First fix made TF-IDF real; this revision adds the
    semantic layer the vault's vocabulary always promised: dense-embedding
    cosine (MiniLM via EmbeddingEngine) blended 60/40 with the lexical
    TF-IDF score, plus a bounded exact-phrase bonus (verbatim evidence
    outranks thematic overlap at equal similarity). Fallback chain is
    honest and bounded: hybrid → TF-IDF (backend down, or too many cold
    texts for one query — a background warm fixes the next one) — never
    substring.
    """
    import math

    query_tokens = tokenize(query)
    if not query_tokens or not memories:
        return []

    # Memories arrive under both keys. The vault stores `text` and hands
    # callers `content` (black_hole_vault maps one to the other on the way
    # out), and most stores in core/memory use `content` throughout. Reading
    # only `text` meant a caller passing the other shape got every document
    # tokenized to nothing — and NOT an empty result, because the dense layer
    # still embedded the empty strings and returned small positive cosines
    # that cleared the 0.01 threshold. Confident retrieval over documents that
    # were never read. Measured 2026-08-06: 36 of 40 queries returned "hits"
    # whose text the scorer had never seen.
    texts = [_memory_text(m) for m in memories]
    if not any(texts):
        # Nothing readable. Returning [] is the honest answer; scoring blanks
        # is how an empty corpus produces a ranked list.
        return []

    doc_tokens = [tokenize(text) for text in texts]
    idf = _idf_weights([set(toks) for toks in doc_tokens])
    # unseen query terms get max-rarity weight instead of vanishing
    default_idf = math.log(1.0 + len(doc_tokens))

    def weigh(tf: dict[str, float]) -> dict[str, float]:
        return {tok: freq * idf.get(tok, default_idf) for tok, freq in tf.items()}

    query_vec = weigh(compute_term_freq(query_tokens))
    query_lower = (query or "").lower().strip()

    # Semantic layer: dense-embedding cosine per memory when the real
    # backend is up and the cache admits this query's texts (bounded work).
    dense = _semantic_scores(query, texts, task=task)

    scored: list[dict[str, Any]] = []
    for position, (memory, tokens) in enumerate(zip(memories, doc_tokens, strict=False)):
        if not tokens:
            # An unreadable memory is not a weak match, it is not a match. The
            # dense layer will happily return a small positive cosine for the
            # empty string, and 0.6 * that clears the default threshold.
            continue
        doc_vec = weigh(compute_term_freq(tokens))
        lexical = compute_cosine_similarity(query_vec, doc_vec)
        if dense is not None:
            # Hybrid: meaning first, exact terms still matter. Dense cosine
            # is clamped at 0 (negative similarity is just 'unrelated').
            score = 0.6 * max(0.0, dense[position]) + 0.4 * lexical
        else:
            score = lexical
        if query_lower and query_lower in texts[position].lower():
            score = min(1.0, score + 0.25)
        item = dict(memory)
        item["score"] = round(float(score), 6)
        item["retrieval"] = "hybrid_semantic" if dense is not None else "tfidf"
        scored.append((float(score), item))

    # Cut where the scores stop looking like chance, not at a constant.
    # `threshold` survives only as an optional absolute backstop — it was
    # measured admitting 5/6 unrelated pairs under MiniLM and 6/6 under
    # Qwen3-Embedding, i.e. it never filtered anything. See
    # core/memory/retrieval_calibration.py for the null measurement.
    floor = float(threshold) if threshold and threshold > 0 else None
    return list(retrieval_calibration.select_above_chance(scored, top_k=top_k, floor=floor))


def retrieve_memories_v2(
    query: str,
    memories: list[dict[str, Any]],
    top_k: int = 5,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return retrieve_memories(query, memories, top_k=top_k, **kwargs)

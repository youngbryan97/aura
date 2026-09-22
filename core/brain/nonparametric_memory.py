"""Non-parametric memory — growable capacity for a fixed model (kNN-LM, made Aura's).

The honest way to increase a frozen model's capacity without enlarging its weights:
keep a datastore of (hidden-state key -> next token) pairs and, at generation time,
interpolate the nearest neighbors' tokens into the model's own next-token distribution.
In the kNN-LM literature a small model plus a large datastore can close much of the
knowledge gap to a bigger model, because the datastore is *non-parametric* capacity that
grows without retraining. That is the result this design is BASED on; it is not a
measurement of this implementation. Nothing here benchmarks Aura against a larger model,
so this module claims a mechanism, not a capability.

This is NOT prompt-level RAG — it operates on the next-token distribution itself, so the
information lives in a reachable store instead of in parameters. It is fail-open logit
reweighting (same risk class as the steering hooks, NOT weight mutation).

Aura-specific novelty:
  * the datastore is fed only TRUSTED knowledge (verifier-clean answers, beliefs) — sound,
    not a raw corpus that can teach errors;
  * the interpolation weight λ is Φ-ADAPTIVE — high free-energy (model surprised) + a
    confident neighbor ⇒ trust memory more; confident model ⇒ trust the weights;
  * entries carry affect/recency weight (the vault's "gravity") applied at the token level.

Bounded by construction (max entries, weight×recency eviction). Pure numpy.

Retrieval is an EXACT SCAN, not an approximate index. "FAISS optional" was
in this line and nowhere in the code: `query` computes two matrix-vector
products over every visible entry while holding the store lock, so
decode-time cost is O(entries × hidden width) and it blocks concurrent
adds and persistence (CP126 ``3ac36b7a``). The principal filter bounds the
candidate set, and the recall receipt reports the store size and the
neighbours for every call, so that cost is measurable rather than assumed.
The datastore + kNN + adaptive-λ interpolation are built and tested here. Populating keys
from real model hidden states and applying per-token interpolation inside MLX generation is
the documented seam (`apply_to_logits`), flag-gated (AURA_NONPARAMETRIC_MEMORY).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.brain.nonparametric_identity import (
    EntryProvenance,
    StoreIdentity,
    TrustLevel,
    identity_from_mapping,
)
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.NonParametricMemory")

# Φ thresholds mirror phi_consciousness so "is cognition integrated enough to trust
# non-parametric recall" matches the rest of the system.
PHI_DORMANT = 0.05
PHI_DELIBERATE = 0.55

#: A principal every caller may read. Facts that belong to the system
#: rather than to a person — Aura's own verified reasoning results, not
#: anyone's conversation.
SHARED_MEMORY_PRINCIPAL = "shared"
_SHARED_PRINCIPAL = SHARED_MEMORY_PRINCIPAL


def _flag_on(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "on", "yes", "enabled"}


def _clamp(v: float, lo: float, hi: float) -> float:
    """Finite value inside [lo, hi]. NaN goes to the FLOOR, not the ceiling.

    ``max(lo, min(hi, nan))`` returns ``hi``: both comparisons are False, so
    a NaN lambda clamped to the MAXIMUM interpolation weight and handed the
    whole distribution to recall (CP126 ``cb95d7d9``). A value nobody can
    compute is the least trust, not the most.
    """
    try:
        value = float(v)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _finite_probabilities(probs: Any) -> dict[int, float]:
    """A usable probability map, or empty. Nothing propagates a NaN.

    ``lm_probs`` arrived from a caller and went straight into the blend, so
    a NaN or a value above one produced negative model mass or spread the
    NaN across every token.
    """
    if not isinstance(probs, dict):
        return {}
    clean: dict[int, float] = {}
    for token, value in probs.items():
        try:
            token_id = int(token)
            p = float(value)
        except (TypeError, ValueError):
            continue
        if token_id < 0 or not math.isfinite(p) or p < 0.0:
            continue
        clean[token_id] = min(1.0, p)
    return clean


# Below this, a centered vector has no usable direction: the cosine would be
# noise divided by ~zero. Queries and keys that degenerate this far are
# reported as unknown rather than ranked.
_DEGENERATE_NORM = 1e-9


def _provenance_from_mapping(value: Any) -> EntryProvenance:
    if not isinstance(value, dict):
        return EntryProvenance(source_id="unattributed")
    return EntryProvenance(
        source_id=str(value.get("source_id", "unattributed")),
        trust=str(value.get("trust", TrustLevel.UNVERIFIED)),
        verifier=str(value.get("verifier", "")),
        evidence_id=str(value.get("evidence_id", "")),
        principal=str(value.get("principal", "anonymous")),
        content_sha256=str(value.get("content_sha256", "")),
    )


def _streamed_keys_digest(keys: Any, rows_per_block: int = 512) -> str:
    """Digest of a persisted key matrix without materialising it."""
    hasher = hashlib.sha256()
    total = int(keys.shape[0])
    for start in range(0, total, rows_per_block):
        block = np.ascontiguousarray(keys[start : start + rows_per_block], dtype=np.float32)
        hasher.update(memoryview(block).cast("B"))
    return hasher.hexdigest()


def _keys_digest(keys: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(keys, dtype=np.float32)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    # not a failure: a directory that will not open has nothing to fsync.
    except OSError:
        return
    try:
        os.fsync(fd)
    # not a failure: an fsync that refuses on this filesystem leaves the write where
    # the OS put it, which is the same as before this call existed.
    except OSError:
        pass
    finally:
        os.close(fd)


@dataclass
class Neighbor:
    token_id: int
    token: str
    distance: float
    weight: float
    # Anisotropy-corrected similarity in [-1, 1]: cosine between the query
    # and this key AFTER subtracting the running mean of the hidden space.
    # Raw last-token hidden states share a dominant common direction —
    # measured on the live 1.5B: UNRELATED prompts score raw cos 0.81-0.93
    # while identical prompts score 1.0000, so raw-cos gates cannot
    # separate them; mean-centered cosines put unrelated prompts at ≤0.36.
    # Until the mean estimate is ready, this falls back to raw cosine and
    # the gate compensates with a much higher threshold.
    similarity: float = -1.0
    # Position of this entry in the store — lets generation loops apply the
    # anti-stutter guard (the same entry may not fire on consecutive steps).
    index: int = -1
    #: The threshold this similarity was computed against, captured under the
    #: same lock hold. Reading the gate afterwards let a concurrent mode
    #: change pair a raw-mode similarity with the centered threshold.
    gate_threshold: float = 1.0
    centered: bool = False
    #: Where the entry came from, so a recall receipt can name it and a
    #: discredited source can be revoked.
    source_id: str = ""
    trust: str = "unverified"


class NonParametricMemory:
    """Datastore of (key, next-token) pairs + kNN + Φ-adaptive distribution interpolation."""

    _ERRORS = (OSError, ValueError, TypeError, json.JSONDecodeError)

    def __init__(
        self,
        dim: int,
        path: str | Path | None = None,
        *,
        max_entries: int = 4_096,
        base_lambda: float = 0.25,
        max_lambda: float = 0.7,
        dist_scale: float = 1.0,
        identity: StoreIdentity | None = None,
    ) -> None:
        self._dim = int(dim)
        if self._dim <= 0:
            raise ValueError("non-parametric memory dimension must be positive")
        # The store's identity is not its width. Two models of the same
        # width write vectors that mean different things and token ids from
        # different vocabularies; sharing a store between them combined both
        # and reported successful reuse (CP126 ``aba3eb39``).
        self._identity = identity or StoreIdentity(dim=self._dim)
        self._path = Path(
            path or state_root() / "data" / "runtime" / f"nonparametric_memory_{self._identity.slug()}"
        )
        self._max = max(64, int(max_entries))
        self._base_lambda = float(base_lambda)
        self._max_lambda = float(max_lambda)
        self._dist_scale = max(1e-6, float(dist_scale))
        self._lock = threading.RLock()
        self._capacity = min(64, self._max)
        self._size = 0
        self._keys = np.empty((self._capacity, self._dim), dtype=np.float32)
        self._key_norms = np.empty(self._capacity, dtype=np.float32)
        self._token_ids: list[int] = []
        self._tokens: list[str] = []
        self._weights: list[float] = []
        self._ts: list[float] = []
        self._stats = {"added": 0, "queried": 0, "evicted": 0, "interpolated": 0, "fallthrough": 0}
        self._content_generation = 0
        self._identity_cache: tuple[int, dict[str, Any]] | None = None
        # Running mean of QUERY keys — the estimator of the hidden space's
        # anisotropic common direction. Every query is a sample; persisted
        # with the store so the correction survives restarts.
        self._query_mu = np.zeros(self._dim, dtype=np.float32)
        self._query_mu_n = 0
        #: Per-entry provenance, parallel to the key rows. Nothing could say
        #: why an entry was admitted or drop every entry from a discredited
        #: source (CP126 ``ff3a4505``), and no entry named a principal, so
        #: one person's query searched another's memory (``62309dad``).
        self._provenance: list[EntryProvenance] = []
        self._last_receipt: dict[str, Any] = {"reason": "never_called"}
        #: Content address per entry, so a replay writes the same fact once
        #: instead of stacking duplicate votes (CP126 ``4354909``).
        self._content_keys: dict[str, int] = {}
        self._load()

    # How many query samples the mean needs before centered similarity is
    # trusted; below this the raw-cosine fallback gate applies.
    MU_READY_N = 16
    # Gate thresholds per mode (measured, see Neighbor.similarity).
    MIN_SIM_CENTERED = 0.60
    MIN_SIM_RAW = 0.98

    def similarity_ready(self) -> bool:
        return self._query_mu_n >= self.MU_READY_N

    def min_similarity(self) -> float:
        """The confident-recall gate matched to the active similarity mode."""
        return self.MIN_SIM_CENTERED if self.similarity_ready() else self.MIN_SIM_RAW

    # ── population ────────────────────────────────────────────────────────────
    #: A finite float32 whose square overflows is still finite. ||k||^2 feeds
    #: every distance in this store, so the magnitude is bounded at admission
    #: rather than discovered as an inf halfway through a matrix product
    #: (CP126 ``2979489a``).
    MAX_KEY_NORM = 1.0e12

    def add(
        self,
        key: np.ndarray,
        token_id: int,
        token: str = "",
        *,
        weight: float = 1.0,
        provenance: EntryProvenance,
    ) -> bool:
        """Admit one (key -> next token) pair, with its provenance.

        ``provenance`` names the source, the trust level, the verifier and
        the principal, and it is required. It was once optional with an
        anonymous default, which meant the live ingest path wrote every
        entry as ``unattributed``/``UNVERIFIED``/``anonymous`` — the three
        values that make revocation by source and erasure by principal
        impossible. An optional isolation argument is not isolation, so the
        default is gone (CP126 ``ff3a4505``, ``62309dad``).
        """
        k = np.asarray(key, dtype=np.float32).reshape(-1)
        if k.shape[0] != self._dim or not np.all(np.isfinite(k)):
            return False
        norm_sq = float(np.dot(k.astype(np.float64), k.astype(np.float64)))
        if not math.isfinite(norm_sq) or norm_sq > self.MAX_KEY_NORM:
            self._stats["rejected_magnitude"] = int(self._stats.get("rejected_magnitude", 0)) + 1
            return False
        token_id = int(token_id)
        if not self._token_id_in_vocabulary(token_id):
            # An id outside the tokenizer's range cannot name a token in this
            # model. It used to be stored, exported by interpolate(), and then
            # silently skipped by apply_to_logits AFTER the lambda mass was
            # reserved (CP126 ``957c060f``).
            self._stats["rejected_token_id"] = int(self._stats.get("rejected_token_id", 0)) + 1
            return False

        record = provenance or EntryProvenance(source_id="unattributed")
        content_key = self._content_key(k, token_id, record.principal)
        with self._lock:
            existing = self._content_keys.get(content_key)
            if existing is not None and existing < self._size:
                # Same fact, same principal, already here. Appending again
                # gave one fact several independent nearest neighbours whose
                # kNN votes summed, so a replay amplified a token and ate the
                # bounded capacity (CP126 ``4354909``).
                self._weights[existing] = self._clamped_weight(weight)
                self._ts[existing] = time.time()
                self._provenance[existing] = record
                self._stats["deduplicated"] = int(self._stats.get("deduplicated", 0)) + 1
                self._content_generation += 1
                self._identity_cache = None
                return True
            if self._size >= self._max:
                self._evict_one_for_incoming()
            self._ensure_capacity(self._size + 1)
            self._keys[self._size] = k
            self._key_norms[self._size] = np.float32(norm_sq)
            self._token_ids.append(token_id)
            self._tokens.append(str(token))
            self._weights.append(self._clamped_weight(weight))
            self._ts.append(time.time())
            self._provenance.append(record)
            self._content_keys[content_key] = self._size
            self._size += 1
            self._stats["added"] += 1
            self._content_generation += 1
            self._identity_cache = None
        return True

    @staticmethod
    def _clamped_weight(weight: Any) -> float:
        """A finite, non-negative weight. NaN poisons the mass it joins."""
        try:
            w = float(weight)
        except (TypeError, ValueError):
            return 1.0
        return w if math.isfinite(w) and w >= 0.0 else 0.0

    def _token_id_in_vocabulary(self, token_id: int) -> bool:
        """Whether this id can name a token under the store's tokenizer."""
        if token_id < 0:
            return False
        vocab = int(self._identity.tokenizer_vocab_size or 0)
        # An unknown vocabulary cannot refuse an id; it can only refuse a
        # negative one. That is recorded on the identity so a reader knows
        # the check is weaker than it looks.
        return vocab <= 0 or token_id < vocab

    def _content_key(self, key: np.ndarray, token_id: int, principal: str) -> str:
        """Content address for deduplication, quantised so near-identical
        replays of the same hidden state collapse to one entry."""
        quantised = np.round(key.astype(np.float64), 4).astype(np.float32)
        digest = hashlib.sha256(memoryview(quantised).cast("B")).hexdigest()[:32]
        return f"{principal}:{token_id}:{digest}"

    def revoke_source(self, source_id: str) -> int:
        """Drop every entry admitted from one source. Returns how many.

        The revocation handle CP126 ``ff3a4505`` says was missing: when a
        source turns out to be wrong, its entries were indistinguishable
        from every other and stayed forever.
        """
        return self._drop_where(lambda record: record.source_id == str(source_id))

    def forget_principal(self, principal: str) -> int:
        """Drop every entry belonging to one principal. Returns how many."""
        return self._drop_where(lambda record: record.principal == str(principal))

    def _drop_where(self, predicate: Any) -> int:
        with self._lock:
            keep = [i for i in range(self._size) if not predicate(self._provenance[i])]
            dropped = self._size - len(keep)
            if dropped <= 0:
                return 0
            if keep:
                self._keys[: len(keep)] = self._keys[keep]
                self._key_norms[: len(keep)] = self._key_norms[keep]
            self._token_ids = [self._token_ids[i] for i in keep]
            self._tokens = [self._tokens[i] for i in keep]
            self._weights = [self._weights[i] for i in keep]
            self._ts = [self._ts[i] for i in keep]
            self._provenance = [self._provenance[i] for i in keep]
            self._size = len(keep)
            self._reindex_content_keys()
            self._stats["revoked"] = int(self._stats.get("revoked", 0)) + dropped
            self._content_generation += 1
            self._identity_cache = None
        return dropped

    def _reindex_content_keys(self) -> None:
        """Caller holds the lock. Rebuild the dedup index after a compaction."""
        self._content_keys = {
            self._content_key(self._keys[i], self._token_ids[i], self._provenance[i].principal): i
            for i in range(self._size)
        }

    def __len__(self) -> int:
        with self._lock:
            return self._size

    # ── retrieval ─────────────────────────────────────────────────────────────
    def query(
        self,
        key: np.ndarray,
        k: int = 8,
        *,
        principal: str = "",
        update_mean: bool = True,
    ) -> list[Neighbor]:
        """Nearest entries VISIBLE TO THIS PRINCIPAL.

        Every query searched the whole process-wide store, so one person
        could retrieve another's memory (CP126 ``62309dad``). An empty
        ``principal`` searches everything, which is what the internal
        maintenance callers need and what a request lane must not use.

        ``update_mean`` exists because every query — unauthorized,
        unrelated, and eventual fallthroughs alike — mutated the one global
        running mean that the anisotropy correction is built on, and that
        mean is persisted. A query mix therefore changed future similarity
        decisions for everyone (CP126 ``7a4bfedb``).
        """
        q = np.asarray(key, dtype=np.float32).reshape(-1)
        # Reject a non-finite query outright: searching with NaN/inf produces
        # NaN distances and garbage neighbor selection, and must never poison
        # the persisted running mean below.
        if q.shape[0] != self._dim or not np.all(np.isfinite(q)):
            return []
        with self._lock:
            n = self._size
            if n == 0:
                return []
            self._stats["queried"] += 1
            # Every query is a sample of the hidden space: feed the running
            # mean that powers the anisotropy correction (cumulative mean up
            # to 256 samples, then a slow EMA that tracks drift).
            if update_mean:
                if self._query_mu_n < 256:
                    self._query_mu_n += 1
                    self._query_mu += (q - self._query_mu) / float(self._query_mu_n)
                else:
                    self._query_mu += 0.005 * (q - self._query_mu)
            # CP126 d5cc1faf. Candidates used to be selected by raw
            # Euclidean distance while the confidence gate judged them by
            # mean-centered cosine — two metrics that disagree precisely
            # where it matters. Raw hidden states share a dominant common
            # direction (measured on the live 1.5B: UNRELATED prompts at raw
            # cos 0.81-0.93), so ||x-q|| is dominated by vector norm and that
            # shared direction. The semantically strongest centered-cosine
            # neighbour could therefore be ranked outside the top k and never
            # reach the gate at all — and no gate threshold can repair a
            # candidate set that already excluded the right answer.
            #
            # Selection now uses the SAME metric as the gate, computed for
            # every entry without materialising an n-by-dim difference
            # matrix:
            #
            #   <q-mu, x-mu> = x.q - x.mu - q.mu + mu.mu
            #   ||x-mu||^2   = ||x||^2 - 2 x.mu + ||mu||^2
            #
            # so two matrix-vector products (keys@q, keys@mu) give exact
            # centered cosines for all n — the same asymptotic cost as the
            # distance scan it replaces. Before the mean estimate is ready mu
            # is zero and this reduces exactly to raw cosine.
            # The principal filter runs BEFORE the scan, so an entry the
            # caller may not see never enters the ranking.
            if principal:
                visible = [
                    i for i in range(n)
                    if self._provenance[i].principal in (principal, _SHARED_PRINCIPAL)
                ]
                if not visible:
                    return []
                rows = np.asarray(visible, dtype=np.int64)
            else:
                rows = np.arange(n, dtype=np.int64)
            keys = self._keys[rows]
            key_norms = self._key_norms[rows]
            # The mode is captured HERE, under the same lock hold that
            # computes the similarities. Reading it again after the lock let
            # a concurrent crossing of MU_READY_N pair raw-mode similarities
            # with the lower centered threshold (CP126 ``7a4bfedb``).
            centered = self._query_mu_n >= self.MU_READY_N
            gate_threshold = self.MIN_SIM_CENTERED if centered else self.MIN_SIM_RAW
            mu = self._query_mu if centered else np.zeros_like(q)
            kq = keys @ q
            km = keys @ mu
            qm = float(np.dot(q, mu))
            mm = float(np.dot(mu, mu))
            numerator = kq - km - qm + mm
            key_centered_sq = np.maximum(key_norms - 2.0 * km + mm, 0.0)
            q_centered_norm = math.sqrt(max(float(np.dot(q, q)) - 2.0 * qm + mm, 0.0))
            if q_centered_norm <= _DEGENERATE_NORM:
                # The query sits on the common direction itself: after
                # centering it has no direction left to compare, so every
                # cosine here would be numerical noise amplified by a
                # near-zero denominator. Report unknown (-1) and let the
                # confidence gate refuse rather than inventing a ranking.
                self._stats["degenerate_query"] = (
                    int(self._stats.get("degenerate_query", 0)) + 1
                )
                return []
            denominator = np.sqrt(key_centered_sq) * q_centered_norm
            sims_all = np.where(
                denominator > _DEGENERATE_NORM,
                numerator / np.maximum(denominator, _DEGENERATE_NORM),
                -1.0,
            )
            kk = min(max(1, int(k)), int(rows.size))
            # Highest similarity first — argpartition on the negated scores.
            idx = np.argpartition(-sims_all, kk - 1)[:kk]
            idx = idx[np.argsort(-sims_all[idx])]
            # Euclidean distance is still REPORTED (callers and telemetry use
            # it), computed only for the k that were selected.
            selected = keys[idx]
            diff_sq = key_norms[idx] + float(np.dot(q, q)) - 2.0 * (selected @ q)
            dists = np.sqrt(np.maximum(diff_sq, 0.0))
            similarities = sims_all[idx]
            entries = rows[idx]
            return [
                Neighbor(
                    self._token_ids[int(i)],
                    self._tokens[int(i)],
                    float(dist),
                    float(self._weights[int(i)]),
                    similarity=float(sim),
                    index=int(i),
                    gate_threshold=float(gate_threshold),
                    centered=bool(centered),
                    source_id=self._provenance[int(i)].source_id,
                    trust=self._provenance[int(i)].trust,
                )
                for i, dist, sim in zip(entries, dists, similarities, strict=True)
            ]

    def _similarities_for(self, q: np.ndarray, idx: Any) -> list[float]:
        """Gate-ready similarity per neighbor (caller holds the lock).

        Centered cosine once the mean is ready; raw cosine before that.
        O(k·dim) — computed only for the k returned neighbors.
        """
        centered = self._query_mu_n >= self.MU_READY_N
        mu = self._query_mu if centered else None
        sims: list[float] = []
        for i in idx:
            key_vec = self._keys[int(i)]
            if mu is not None:
                a = q - mu
                b = key_vec - mu
            else:
                a = q
                b = key_vec
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na <= 1e-8 or nb <= 1e-8:
                sims.append(-1.0)
                continue
            sims.append(float(np.dot(a, b)) / (na * nb))
        return sims

    def knn_probs(self, neighbors: list[Neighbor], *, temperature: float = 1.0) -> dict[int, float]:
        """kNN-LM distribution: softmax over -distance, weighted by entry gravity, per token."""
        if not neighbors:
            return {}
        t = max(1e-6, float(temperature)) * self._dist_scale
        logits = np.array([-(nb.distance / t) for nb in neighbors], dtype=np.float64)
        if not np.all(np.isfinite(logits)):
            return {}
        logits -= logits.max()
        w = np.array([max(0.0, nb.weight) for nb in neighbors], dtype=np.float64)
        ex = np.exp(logits) * (w if w.sum() > 0 else 1.0)
        total = float(ex.sum())
        # `if total <= 0` is False for NaN, so a NaN total used to sail past
        # the emptiness check and divide every probability into NaN.
        if not math.isfinite(total) or total <= 0.0:
            return {}
        probs: dict[int, float] = {}
        for nb, p in zip(neighbors, ex / total, strict=True):
            probs[nb.token_id] = probs.get(nb.token_id, 0.0) + float(p)
        return probs

    # ── Φ-adaptive interpolation weight ─────────────────────────────────────────
    def adaptive_lambda(
        self,
        neighbors: list[Neighbor],
        *,
        phi: float | None = None,
        free_energy: float | None = None,
    ) -> float:
        """How much to trust non-parametric memory vs. the model's own weights → [0, max_lambda].

        Closer neighbors (confident recall) and higher free-energy (surprised model) raise λ;
        low Φ (fragmented cognition) caps it. No neighbors ⇒ 0 (pure model).
        """
        if not neighbors:
            return 0.0
        # Anisotropy-corrected gate: below the confident-recall threshold the
        # blend contributes NOTHING (measured: raw hidden-state cosine cannot
        # separate unrelated prompts — the July proof caught the old
        # distance-only confidence corrupting even unrelated generations).
        best_sim = max(nb.similarity for nb in neighbors)
        min_sim = self.min_similarity()
        if best_sim < min_sim:
            return 0.0
        confidence = _clamp((best_sim - min_sim) / max(1e-6, 1.0 - min_sim), 0.0, 1.0)
        fe = _clamp(float(free_energy), 0.0, 1.0) if free_energy is not None else 0.5
        lam = self._base_lambda * (0.5 + 0.5 * confidence) * (0.6 + 0.8 * fe)
        if phi is not None:
            if phi < PHI_DORMANT:
                return 0.0                                          # too fragmented to trust recall
            cap = _clamp(0.3 + 0.7 * (phi / PHI_DELIBERATE), 0.0, 1.0)
            lam *= cap
        return _clamp(lam, 0.0, self._max_lambda)

    def interpolate(
        self,
        lm_probs: dict[int, float],
        query_key: np.ndarray,
        *,
        k: int = 8,
        temperature: float = 1.0,
        phi: float | None = None,
        free_energy: float | None = None,
        lam_override: float | None = None,
        principal: str = "",
    ) -> dict[int, float]:
        """Blend the model's next-token probs with kNN recall: p = λ·p_kNN + (1-λ)·p_LM.

        The guarantee is narrower than "never degrades", and the difference matters.

        What IS guaranteed, and tested: when there are no neighbours above the
        similarity gate, when λ resolves to ≈0, or when anything in the
        recall/blend path raises, this returns ``lm_probs`` UNCHANGED — the
        same object contents, not merely something similar. Those three paths
        cannot make generation worse because they do not touch the
        distribution at all.

        What is NOT guaranteed: that a blend which DOES fire improves the
        output. Once λ>0 and neighbours pass the gate, probability mass moves,
        and moving it can be wrong — a confidently-recalled neighbour from a
        different fact is exactly the failure the similarity gate exists to
        catch, and the July measurement recorded below shows it happening.
        Saying "generation never degrades below the raw model" claimed the
        second thing while only implementing the first (CP126: "comments
        overstate demonstrated capacity and safety").
        """
        lm_probs = _finite_probabilities(lm_probs)
        try:
            neighbors = self.query(query_key, k=k, principal=principal)
            # Per-neighbor confidence filter: entries below the gate must not
            # leak probability mass into the kNN distribution. Measured failure
            # (July proof): with the soft filter, digits from OTHER facts at raw
            # cos ~0.93 outvoted the exact-match entry and corrupted recall.
            neighbors = [nb for nb in neighbors if nb.similarity >= nb.gate_threshold]
            if not neighbors:
                with self._lock:
                    self._stats["fallthrough"] += 1
                return dict(lm_probs)
            # An override was used verbatim, so a caller could pass 4.0 and
            # create negative model mass, or NaN and propagate it.
            lam = (
                _clamp(lam_override, 0.0, self._max_lambda)
                if lam_override is not None
                else self.adaptive_lambda(neighbors, phi=phi, free_energy=free_energy)
            )
            if not math.isfinite(lam) or lam <= 1e-6:
                with self._lock:
                    self._stats["fallthrough"] += 1
                return dict(lm_probs)
            lam = min(1.0, max(0.0, float(lam)))
            knn = self.knn_probs(neighbors, temperature=temperature)
            blended: dict[int, float] = {}
            for tok in set(lm_probs) | set(knn):
                blended[tok] = (1.0 - lam) * float(lm_probs.get(tok, 0.0)) + lam * float(knn.get(tok, 0.0))
            s = sum(blended.values())
            if s > 0:
                blended = {t: p / s for t, p in blended.items()}
            with self._lock:
                self._stats["interpolated"] += 1
            return blended
        except (
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            ZeroDivisionError,
            FloatingPointError,
            # The docstring promised "an error ANYWHERE in the recall/blend
            # path fails open", and the list stopped short of the ones a
            # backing index actually raises. A numpy-backed query
            # can raise RuntimeError, OSError, IndexError or OverflowError,
            # and each of those escaped into the token loop — killing the
            # generation the fail-open contract exists to protect (CP126:
            # "comments overstate demonstrated capacity and safety").
            RuntimeError,
            OSError,
            IndexError,
            OverflowError,
            MemoryError,
        ) as exc:
            # Honor the fail-open contract: any recall/blend error returns the
            # raw model distribution rather than breaking generation.
            with self._lock:
                self._stats["fallthrough"] = self._stats.get("fallthrough", 0) + 1
            logger.debug("Nonparametric interpolate failed open: %s", exc)
            return dict(lm_probs)

    def apply_to_logits(
        self, logits: np.ndarray, query_key: np.ndarray, **kw: Any
    ) -> np.ndarray:
        """Seam for the live MLX generation loop: reweight a full logit vector in-place-safe.

        Converts logits→probs over the top candidates, interpolates with recall, writes back
        log-probs. Flag-gated; fail-open to the original logits. (Live per-token wiring inside
        the MLX worker is the remaining, latency-validated step.)
        """
        if not _flag_on("AURA_NONPARAMETRIC_MEMORY"):
            self._note_recall("flag_off")
            return logits
        try:
            original = np.asarray(logits)
            arr = np.asarray(logits, dtype=np.float64).reshape(-1)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                self._note_recall("invalid_logits")
                return logits
            neighbors = self.query(
                query_key,
                k=int(kw.pop("k", 8)),
                principal=str(kw.pop("principal", "")),
            )
            if not neighbors:
                self._note_recall("no_neighbors")
                return logits
            # SAME CONFIDENCE GATE AS interpolate(). This path fed the raw
            # query result straight into knn_probs, so every neighbor BELOW
            # the active similarity threshold was still mixed back into the
            # recall distribution — the two recall paths enforced different
            # standards, and the one wired to logits was the permissive one.
            # A single weak neighbour was enough to obtain a nonzero kNN mass
            # and shift the token distribution.
            gated = [nb for nb in neighbors if nb.similarity >= nb.gate_threshold]
            if not gated:
                with self._lock:
                    self._stats["fallthrough"] += 1
                self._note_recall("below_confidence_gate", neighbors=neighbors)
                return logits
            neighbors = gated
            lam = kw.pop("lam_override", None)
            if lam is None:
                lam = self.adaptive_lambda(
                    neighbors,
                    phi=kw.pop("phi", None),
                    free_energy=kw.pop("free_energy", None),
                )
            lam = _clamp(lam, 0.0, self._max_lambda)
            if lam <= 1e-6:
                self._note_recall("lambda_zero", lam=lam, neighbors=neighbors)
                return logits
            knn = self.knn_probs(
                neighbors,
                temperature=float(kw.pop("temperature", 1.0)),
            )
            if not knn:
                self._note_recall("empty_knn", lam=lam, neighbors=neighbors)
                return logits
            max_logit = float(np.max(arr))
            log_z = max_logit + math.log(float(np.exp(arr - max_logit).sum()))
            out = (arr - log_z) + math.log1p(-lam)
            log_lam = math.log(lam)
            changed: list[int] = []
            for token_id, probability in knn.items():
                if 0 <= int(token_id) < out.size and probability > 0.0:
                    out[int(token_id)] = np.logaddexp(
                        out[int(token_id)],
                        log_lam + math.log(float(probability)),
                    )
                    changed.append(int(token_id))
            if not changed:
                # Every neighbour's id fell outside the logit vector. The
                # lambda mass was reserved and nothing received it, and this
                # still counted as an interpolation (CP126 ``957c060f``).
                self._note_recall("no_token_in_vocabulary", lam=lam, neighbors=neighbors)
                return logits
            with self._lock:
                self._stats["interpolated"] += 1
            self._note_recall(
                "applied", lam=lam, neighbors=neighbors, changed_tokens=changed
            )
            return out.reshape(original.shape).astype(
                original.dtype if hasattr(original, "dtype") else np.float32
            )
        except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
            record_degradation("nonparametric_memory_logits", exc)
            self._note_recall(f"error:{type(exc).__name__}")
            return logits

    def last_recall_receipt(self) -> dict[str, Any]:
        """Why the last apply_to_logits did or did not change anything.

        It returned the same untyped logits for a flag that was off, invalid
        logits, no neighbours, low confidence, an empty kNN map and a caught
        failure — six different outcomes with one indistinguishable return,
        no store identity, no neighbour ids, no lambda and no reason
        (CP126 ``da1019b0``).
        """
        with self._lock:
            return dict(self._last_receipt)

    def _note_recall(
        self,
        reason: str,
        *,
        lam: float | None = None,
        neighbors: list[Neighbor] | None = None,
        changed_tokens: list[int] | None = None,
    ) -> None:
        receipt: dict[str, Any] = {
            "schema": "aura.nonparametric_memory.recall_receipt.v1",
            "reason": reason,
            "at": time.time(),
            "store": self._identity.slug(),
            "dim": self._dim,
            "entries": self._size,
            "lambda": None if lam is None else round(float(lam), 6),
            "neighbor_ids": [nb.index for nb in (neighbors or [])][:8],
            "neighbor_sources": sorted({nb.source_id for nb in (neighbors or []) if nb.source_id})[:8],
            "gate_mode": "centered" if self.similarity_ready() else "raw",
            "changed_tokens": list(changed_tokens or [])[:16],
        }
        with self._lock:
            self._last_receipt = receipt
            key = f"recall_{reason.split(':')[0]}"
            self._stats[key] = int(self._stats.get(key, 0)) + 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "entries": self._size,
                "dim": self._dim,
                "max_entries": self._max,
                # Every array this store holds, not just the two matrices.
                # The old figure undercounted by the whole metadata side.
                "allocated_bytes": int(
                    self._keys.nbytes
                    + self._key_norms.nbytes
                    + self._query_mu.nbytes
                    + sum(len(t) for t in self._tokens)
                    + 8 * (len(self._token_ids) + len(self._weights) + len(self._ts))
                ),
                "identity": self._identity.to_dict(),
                "store": self._identity.slug(),
                "last_recall_reason": self._last_receipt.get("reason", "never_called"),
                "persisted": bool(self._path.with_suffix(".meta.json").exists()),
            }

    def identity_receipt_with_work(self) -> tuple[dict[str, Any], int]:
        """Content-address the active store and report bytes hashed this call."""

        with self._lock:
            cached = self._identity_cache
            if cached is not None and cached[0] == self._content_generation:
                return dict(cached[1]), 0
            hasher = hashlib.sha256()
            keys = self._keys[: self._size]
            dtype_bytes = str(keys.dtype).encode("ascii")
            shape_bytes = str(keys.shape).encode("ascii")
            hasher.update(dtype_bytes)
            hasher.update(shape_bytes)
            hasher.update(memoryview(keys).cast("B"))
            metadata = json.dumps(
                {
                    "token_ids": self._token_ids,
                    "tokens": self._tokens,
                    "weights": self._weights,
                    "timestamps": self._ts,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            hasher.update(metadata)
            payload = {
                "schema": "aura.nonparametric_memory.identity.v1",
                "dimension": self._dim,
                "entries": self._size,
                "source_bytes": int(
                    len(dtype_bytes) + len(shape_bytes) + keys.nbytes + len(metadata)
                ),
                "content_sha256": hasher.hexdigest(),
            }
            receipt = {
                **payload,
                "receipt_sha256": hashlib.sha256(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            self._identity_cache = (self._content_generation, receipt)
            return dict(receipt), int(receipt["source_bytes"])

    def identity_receipt(self) -> dict[str, Any]:
        """Content-address the active datastore without exposing learned keys."""

        receipt, _bytes_hashed = self.identity_receipt_with_work()
        return receipt

    # ── internals ───────────────────────────────────────────────────────────────
    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_capacity = min(self._max, max(needed, self._capacity * 2))
        keys = np.empty((new_capacity, self._dim), dtype=np.float32)
        norms = np.empty(new_capacity, dtype=np.float32)
        keys[: self._size] = self._keys[: self._size]
        norms[: self._size] = self._key_norms[: self._size]
        self._keys = keys
        self._key_norms = norms
        self._capacity = new_capacity

    def _evict_one_for_incoming(self) -> None:
        n = self._size
        if n == 0:
            return
        now = time.time()
        gravity = np.array(
            [self._weights[i] * math.exp(-(now - self._ts[i]) / (14 * 24 * 3600.0)) for i in range(n)]
        )
        drop = int(np.argmin(gravity))
        if drop < n - 1:
            self._keys[drop : n - 1] = self._keys[drop + 1 : n]
            self._key_norms[drop : n - 1] = self._key_norms[drop + 1 : n]
        for values in (self._token_ids, self._tokens, self._weights, self._ts):
            values.pop(drop)
        self._size -= 1
        self._stats["evicted"] += 1

    def _load(self) -> None:
        """Read a persisted generation, or start fresh. Never a hybrid.

        Two findings meet here.

        ``b8b9656b`` — keys and metadata were published with two independent
        ``os.replace`` calls, so a crash between them, or two writers
        interleaving, paired vectors from one generation with token metadata
        from another. There was no generation id, no manifest checksum, no
        keys fsync and no directory fsync. Both files now carry the same
        generation id and the metadata carries the keys' digest; a pair that
        does not agree is refused rather than mixed.

        ``848cf532`` — the loader trusted the files after checking rank,
        width and list lengths. Everything else is validated now: schema,
        declared dim, store identity, finite and bounded keys, weights,
        timestamps, token-id range.

        ``1d1d0bfd`` — ``np.load`` materialised the entire persisted matrix
        before the entry count was capped, so a stale or replaced file could
        allocate far past ``max_entries``. It is memory-mapped, sliced, then
        copied.
        """
        keys_p, meta_p = self._path.with_suffix(".keys.npy"), self._path.with_suffix(".meta.json")
        if not (keys_p.exists() and meta_p.exists()):
            return
        keys = None
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("non-parametric memory metadata is not an object")
            if int(meta.get("schema_version", 0) or 0) < 3:
                # A generation written before the manifest existed cannot be
                # paired safely with its keys file. Starting fresh loses a
                # cache; loading it risks pairing across generations.
                record_degradation(
                    "nonparametric_memory_load",
                    ValueError("persisted store predates the generation manifest"),
                    severity="info",
                )
                return
            if int(meta.get("dim", -1)) != self._dim:
                raise ValueError("persisted dim does not match this store")

            saved_identity = identity_from_mapping(meta.get("identity"), dim=self._dim)
            if saved_identity is None:
                raise ValueError("persisted store carries no readable identity")
            compatible, why = self._identity.compatible_with(saved_identity)
            if not compatible:
                # Same width, different model. Reading it would combine one
                # model's vectors with another's token ids (CP126 aba3eb39).
                record_degradation(
                    "nonparametric_memory_load",
                    ValueError(f"refusing incompatible store: {why}"),
                    severity="warning",
                )
                return

            # Memory-mapped, so the cap applies before the allocation.
            keys = np.load(keys_p, mmap_mode="r")
            if keys.ndim != 2 or keys.shape[1] != self._dim:
                return
            digest = str(meta.get("keys_sha256") or "")
            token_ids = list(meta.get("token_ids", []))
            tokens = list(meta.get("tokens", []))
            weights = list(meta.get("weights", []))
            timestamps = list(meta.get("ts", []))
            provenance_rows = list(meta.get("provenance", []))
            lengths = {
                len(keys), len(token_ids), len(tokens), len(weights), len(timestamps)
            }
            if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
                raise ValueError("non-parametric memory persistence metadata is inconsistent")

            total = next(iter(lengths))
            if digest and digest != _streamed_keys_digest(keys):
                # The manifest names a keys file this is not — a torn publish
                # or a replaced file. Refusing is the whole point of writing
                # the digest. Streamed from the memory map so verifying it
                # does not undo the bounded read.
                raise ValueError("persisted keys do not match the manifest digest")
            order = self._retention_order(weights, timestamps, total)
            count = min(total, self._max)
            keep = order[:count]

            capacity = min(max(64, count), self._max)
            new_keys = np.empty((capacity, self._dim), dtype=np.float32)
            new_keys[:count] = np.asarray(keys[keep], dtype=np.float32)
            if not np.all(np.isfinite(new_keys[:count])):
                raise ValueError("non-parametric memory persisted keys are non-finite")
            new_norms = np.empty(capacity, dtype=np.float32)
            norms = np.einsum("ij,ij->i", new_keys[:count], new_keys[:count])
            if not np.all(np.isfinite(norms)) or float(np.max(norms, initial=0.0)) > self.MAX_KEY_NORM:
                raise ValueError("non-parametric memory persisted keys overflow their norms")
            new_norms[:count] = norms

            new_token_ids = [int(token_ids[i]) for i in keep]
            if not all(self._token_id_in_vocabulary(t) for t in new_token_ids):
                raise ValueError("non-parametric memory persisted token ids are out of vocabulary")
            new_tokens = [str(tokens[i]) for i in keep]
            new_weights = [self._clamped_weight(weights[i]) for i in keep]
            new_ts = [float(timestamps[i]) for i in keep]
            if not all(math.isfinite(t) and t > 0.0 for t in new_ts):
                raise ValueError("non-parametric memory persisted timestamps are invalid")
            new_provenance = [
                _provenance_from_mapping(
                    provenance_rows[i] if i < len(provenance_rows) else None
                )
                for i in keep
            ]


            saved_mu = meta.get("query_mu")
            new_mu = None
            new_mu_n = 0
            if isinstance(saved_mu, list) and len(saved_mu) == self._dim:
                candidate_mu = np.asarray(saved_mu, dtype=np.float32)
                if np.all(np.isfinite(candidate_mu)):
                    new_mu = candidate_mu
                    new_mu_n = max(0, min(10_000, int(meta.get("query_mu_n", 0) or 0)))

            with self._lock:
                self._capacity = capacity
                self._keys = new_keys
                self._key_norms = new_norms
                self._token_ids = new_token_ids
                self._tokens = new_tokens
                self._weights = new_weights
                self._ts = new_ts
                self._provenance = new_provenance
                self._size = count
                if new_mu is not None:
                    self._query_mu = new_mu
                    self._query_mu_n = new_mu_n
                self._reindex_content_keys()
                self._content_generation += 1
                self._identity_cache = None
        except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            record_degradation("nonparametric_memory_load", exc)
        finally:
            if keys is not None and hasattr(keys, "_mmap"):
                try:
                    keys._mmap.close()
                # not a failure: a mapping already closed is the state this is reaching.
                except (AttributeError, OSError, ValueError):
                    pass

    def _retention_order(self, weights: list[Any], timestamps: list[Any], total: int) -> list[int]:
        """Rows to keep, best first, by the SAME gravity eviction uses.

        Loading an oversized store kept the last N records while live
        eviction dropped the lowest weight-times-recency, so retention after
        a restart differed from retention during a run (CP126 ``4d9d6616``).
        """
        now = time.time()
        scored: list[tuple[float, int]] = []
        for i in range(total):
            weight = self._clamped_weight(weights[i] if i < len(weights) else 0.0)
            try:
                age = now - float(timestamps[i])
            except (TypeError, ValueError, IndexError):
                age = float("inf")
            if not math.isfinite(age) or age < 0.0:
                # A future or unreadable timestamp cannot be trusted to rank
                # anything; it goes last rather than winning by accident.
                age = float("inf")
            gravity = weight * math.exp(-min(age, 1e9) / (14 * 24 * 3600.0))
            scored.append((gravity, i))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [index for _gravity, index in scored]



    def persist(self) -> bool:
        """Publish keys and metadata as ONE generation, or neither.

        The two files were replaced independently, so a crash in between
        left a keys file from one generation beside metadata from another —
        vectors paired with the wrong tokens, with nothing to detect it. The
        metadata now carries the generation id and the keys' digest, both
        files are fsynced, the directory is fsynced, and the keys land first
        so a torn publish leaves the OLD metadata pointing at a digest that
        no longer matches, which the loader refuses.
        """
        temporary_keys: Path | None = None
        temporary_meta: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                keys = self._keys[: self._size].copy()
                generation = int(self._content_generation)
                meta = {
                    "schema_version": 3,
                    "generation": generation,
                    "dim": self._dim,
                    "identity": self._identity.to_dict(),
                    "saved_at": time.time(),
                    "keys_sha256": _keys_digest(keys),
                    "token_ids": list(self._token_ids),
                    "tokens": list(self._tokens),
                    "weights": list(self._weights),
                    "ts": list(self._ts),
                    "provenance": [record.to_dict() for record in self._provenance],
                    "query_mu": [float(v) for v in self._query_mu],
                    "query_mu_n": int(self._query_mu_n),
                }
            keys_path = self._path.with_suffix(".keys.npy")
            meta_path = self._path.with_suffix(".meta.json")
            with tempfile.NamedTemporaryFile(
                dir=self._path.parent, suffix=".npy", delete=False
            ) as handle:
                np.save(handle, keys)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_keys = Path(handle.name)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._path.parent,
                suffix=".json", delete=False
            ) as handle:
                json.dump(meta, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
                temporary_meta = Path(handle.name)
            os.replace(temporary_keys, keys_path)
            temporary_keys = None
            os.replace(temporary_meta, meta_path)
            temporary_meta = None
            _fsync_directory(self._path.parent)
            return True
        except self._ERRORS as exc:
            record_degradation("nonparametric_memory_persist", exc)
            return False
        finally:
            # A failed write left its temporary file behind every time.
            for leftover in (temporary_keys, temporary_meta):
                if leftover is None:
                    continue
                try:
                    # Through the gateway: this is a file this process
                    # created and must remove, and the gateway is where
                    # deletions are governed.
                    from core.runtime.file_write_gateway import get_file_write_gateway

                    get_file_write_gateway().delete_file(
                        leftover, source="brain.nonparametric_memory.persist_cleanup"
                    )
                except (ImportError, OSError, RuntimeError, ValueError) as cleanup_exc:
                    record_degradation(
                        "nonparametric_memory_persist",
                        cleanup_exc,
                        severity="info",
                        action="left a temporary persistence file behind",
                    )


#: ONE STORE PER EMBEDDING SPACE, not one store per process.
#
# This was a single global bound to whichever dimension asked first. If that
# was a 32-wide probe, every later 64-wide request from the actual model was
# refused for the life of the process:
#
#   Non-parametric memory dimension mismatch (64 requested, 32 active);
#   refusing cross-model reuse.        x517 in one session
#
# Refusing to MIX the spaces is right — vectors from different models are not
# comparable and averaging them is nonsense. Refusing to HOLD both is the
# defect. They are two datastores, and the first caller through the door
# should not decide which one exists.
_stores: dict[str, NonParametricMemory] = {}
_active_key: str = ""
_lock = threading.Lock()


def get_nonparametric_memory(
    dim: int = 0, *, identity: StoreIdentity | None = None
) -> NonParametricMemory | None:
    """The datastore for one MODEL IDENTITY, created on demand.

    Keyed by the identity fingerprint, not the hidden width. Two models of
    the same width write vectors that mean different things and token ids
    from different vocabularies, and the store combined them and reported
    successful reuse (CP126 ``aba3eb39``).

    ``dim=0`` means "whatever the active model is using" — the space most
    recently asked for by name.

    Every read of the registry, including the active-key lookup, is inside
    the lifecycle lock. The width check and the singleton read used to sit
    outside it, so two callers could each build a store for the same path
    and persist over one another (CP126 ``13a3ce91``).
    """
    global _active_key
    with _lock:
        width = int(dim or 0)
        if identity is None and width <= 0:
            return _stores.get(_active_key) if _active_key else None
        resolved = identity or StoreIdentity(dim=width)
        if resolved.dim <= 0:
            return None
        key = resolved.slug()
        store = _stores.get(key)
        if store is None:
            store = NonParametricMemory(resolved.dim, identity=resolved)
            _stores[key] = store
            logger.info(
                "Non-parametric memory: opened %s (%d store(s) held).",
                key,
                len(_stores),
            )
        _active_key = key
        return store


def reset_nonparametric_memory_for_test() -> None:
    """Drop every held store.

    Clearing the pointer left every existing caller holding the old object
    while the next caller built a new one at the same path, so two
    non-transactional writers published to one file. The stores are dropped
    under the same lock that hands them out.
    """
    global _active_key
    with _lock:
        _stores.clear()
        _active_key = ""


def validate_nonparametric_memory_identity(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "dimension",
        "entries",
        "source_bytes",
        "content_sha256",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("non-parametric memory identity fields differ")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    digest = value.get("content_sha256")
    if (
        value.get("schema") != "aura.nonparametric_memory.identity.v1"
        or type(value.get("dimension")) is not int
        or value["dimension"] <= 0
        or type(value.get("entries")) is not int
        or value["entries"] <= 0
        or type(value.get("source_bytes")) is not int
        or value["source_bytes"] <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or value.get("receipt_sha256") != expected
    ):
        raise ValueError("non-parametric memory identity is invalid")
    return dict(value)


def reset_nonparametric_memory() -> None:
    """Drop every held embedding space."""
    reset_nonparametric_memory_for_test()

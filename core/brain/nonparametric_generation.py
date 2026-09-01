"""Generation with non-parametric memory — the real, KV-cached causal loop.

* ``MLXEncoder``               — text/ids → normalized hidden key + first continuation token.
* ``generate_with_memory``     — KV-CACHED generation: prefill once, then one incremental step per
                                 token. At each step the model's hidden state is the query key;
                                 confident recall is interpolated into the next-token distribution.
                                 This is the production form: it avoids re-running the whole prefix
                                 each step. Per-token cost is NOT O(1) — attention still reads the
                                 whole KV cache, so a step is O(sequence length) and total decode is
                                 O(n²) in attention reads; what caching removes is the O(n²) *recompute*
                                 of the prefix itself. Stated precisely because capacity planning
                                 depends on it.
* ``make_nonparametric_logits_processor`` — the same gating as an mlx_lm logits-processor.

Gating uses the datastore's **anisotropy-corrected similarity** (``Neighbor.similarity`` +
``memory.min_similarity()``). This matters: raw last-token hidden states share a dominant common
direction — measured, UNRELATED prompts score raw cosine 0.81–0.93 — so a naive raw-cosine gate
cannot separate related from unrelated. Mean-centred similarity does (unrelated ≤0.36).

Anti-stutter: the entry that fired last step is excluded from the next step's neighbors, so a
single stored entry can't lock generation into a repeat loop.

Fail-open everywhere: any memory error defers to the bare model.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from core.brain.llm.decoder_topology import (
    DecoderTopologyError,
    decoder_backbone,
    decoder_backbone_owner,
    decoder_hidden_size,
)
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.NonParametricGeneration")

_GEN_ERRORS = (RuntimeError, ValueError, TypeError, AttributeError, IndexError, KeyError)
# Φ floor mirrors phi_consciousness: fragmented cognition must not trust recall.
PHI_DORMANT = 0.05


def _as_float32_numpy(value: Any) -> np.ndarray:
    """Cross the MLX/NumPy boundary without exposing unsupported dtypes.

    NumPy cannot consume an MLX ``bfloat16`` buffer through PEP 3118.  Cast
    numerically on the MLX device first; viewing the storage as ``uint16``
    would preserve bytes but corrupt the represented hidden values.
    """

    if type(value).__module__.startswith("mlx"):
        import mlx.core as mx

        value = value.astype(mx.float32)
        mx.eval(value)
    return np.asarray(value, dtype=np.float32)


def normalize(vec: np.ndarray) -> np.ndarray:
    """Unit-normalize a key. L2 distance on unit vectors encodes cosine: cos = 1 - d²/2."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def cosine_from_l2(distance: float) -> float:
    """Cosine similarity from the L2 distance between two UNIT vectors."""
    return 1.0 - (float(distance) ** 2) / 2.0


class MLXEncoder:
    """Real encoder over a loaded mlx_lm model: text -> (normalized hidden key, first token)."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tok = tokenizer
        self.dim = decoder_hidden_size(model)
        self._hidden_model = decoder_backbone(model)
        self._specials = set(getattr(tokenizer, "all_special_ids", []) or [])

    def encode_hidden(self, text: str) -> np.ndarray:
        return normalize(self._hidden_from_ids(self.tok.encode(text)))

    def encode_hidden_ids(self, ids: list[int]) -> np.ndarray:
        return normalize(self._hidden_from_ids(list(ids)))

    def encode_hidden_sequence_ids(self, ids: list[int]) -> np.ndarray:
        """Encode every prefix position with one causal model forward."""

        hidden = self._hidden_sequence_from_ids(list(ids))
        return self._normalize_hidden_rows(hidden)

    def encode_lexical_contextual_sequence_ids(self, ids: list[int]) -> np.ndarray:
        """Combine stable token identity with final causal context.

        The embedding lookup is not a second transformer forward.  Each
        channel is normalized independently, then assigned equal energy before
        the combined row is normalized.  A linear reader can therefore retain
        lexical identity without losing the contextual state needed for
        reference resolution.
        """

        import mlx.core as mx

        token_ids = list(ids)
        tensor = mx.array([token_ids])
        lexical = _as_float32_numpy(self._hidden_model.embed_tokens(tensor)[0])
        contextual = self._hidden_sequence_from_tensor(tensor)
        combined = np.concatenate(
            (
                self._normalize_hidden_rows(lexical),
                self._normalize_hidden_rows(contextual),
            ),
            axis=-1,
        )
        return self._normalize_hidden_rows(combined)

    @staticmethod
    def _normalize_hidden_rows(hidden: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(hidden, axis=-1, keepdims=True)
        return np.divide(
            hidden,
            norms,
            out=np.array(hidden, copy=True),
            where=norms > 1e-8,
        )

    def _hidden_from_ids(self, ids: list[int]) -> np.ndarray:
        return self._hidden_sequence_from_ids(ids)[-1]

    def _hidden_sequence_from_ids(self, ids: list[int]) -> np.ndarray:
        import mlx.core as mx

        return self._hidden_sequence_from_tensor(mx.array([ids]))

    def _hidden_sequence_from_tensor(self, token_ids: Any) -> np.ndarray:
        h = self._hidden_model(token_ids)
        return _as_float32_numpy(h[0])

    def encode_tokens(self, text: str) -> list[int]:
        return list(self.tok.encode(text))

    #: Returned when a continuation yields no usable token. NOT 0 — zero is a
    #: real token id in most vocabularies, so the old sentinel turned "no
    #: continuation data" into a fabricated recall target.
    NO_TOKEN = -1

    def first_token(self, continuation: str) -> int:
        ids = [i for i in self.tok.encode(continuation) if i not in self._specials]
        return int(ids[0]) if ids else self.NO_TOKEN


def _lm_head(model: Any, hidden: Any) -> Any:
    """Project hidden states to logits (handles tied-embedding models)."""
    language = decoder_backbone_owner(model)
    head = getattr(language, "lm_head", None)
    if callable(head):
        return head(hidden)
    return decoder_backbone(model).embed_tokens.as_linear(hidden)


def _full_probs(logits: np.ndarray) -> np.ndarray:
    """Softmax over the WHOLE vocabulary — the model's actual distribution."""
    shifted = logits - float(np.max(logits))
    ex = np.exp(shifted)
    total = float(ex.sum())
    return ex / total if total > 0.0 else np.full_like(ex, 1.0 / max(1, ex.size))


def _topk_probs(logits: np.ndarray, k: int = 64) -> dict[int, float]:
    """Top-k of the model's TRUE distribution — mass outside the k is dropped,
    not redistributed.

    The previous version softmaxed over only the k selected logits, which
    assigned that subset a total probability of one. Every truncated token was
    inflated, so the interpolation below did not mix the LM distribution with
    memory at lambda — it mixed a fictitious distribution. Normalising over the
    full vocabulary first keeps the retained probabilities true.
    """
    k = min(k, logits.shape[0])
    idx = np.argpartition(logits, -k)[-k:]
    probs = _full_probs(logits)
    return {int(t): float(probs[t]) for t in idx}


def _valid_vocab_token(token_id: Any, vocab_size: int) -> bool:
    """Memory-supplied ids reach the sampler and then the embedding table.

    A corrupt, stale-tokenizer, or cross-model datastore entry could otherwise
    select an id outside the active vocabulary and break decoding outright.
    """
    try:
        value = int(token_id)
    except (TypeError, ValueError, OverflowError):
        return False
    return 0 <= value < int(vocab_size)


def _gated_lambda(similarity: float, min_sim: float, free_energy: float | None, base_lam: float) -> float:
    """λ scaled by how far the neighbor clears the confident-recall gate. 0 below the gate."""
    if similarity < min_sim:
        return 0.0
    fe = 0.5 if free_energy is None else float(free_energy)
    span = max(1e-6, 1.0 - min_sim)
    lam = base_lam * ((similarity - min_sim) / span) * (0.6 + 0.8 * fe)
    return max(0.0, min(lam, 0.9))


def _select_with_memory(
    memory: Any,
    key: np.ndarray,
    logits: np.ndarray,
    *,
    k: int,
    temperature: float,
    phi: float | None,
    free_energy: float | None,
    base_lam: float,
    exclude_index: int,
) -> tuple[int, int]:
    """Return (next_token_id, fired_entry_index). fired_index=-1 when memory didn't fire."""
    bare = int(np.argmax(logits))
    if phi is not None and float(phi) < PHI_DORMANT:
        return bare, -1
    neighbors = [nb for nb in memory.query(key, k=k) if int(getattr(nb, "index", -1)) != exclude_index]
    if not neighbors:
        return bare, -1
    top = neighbors[0]
    min_sim = memory.min_similarity() if hasattr(memory, "min_similarity") else 0.98
    lam = _gated_lambda(float(getattr(top, "similarity", -1.0)), float(min_sim), free_energy, base_lam)
    if lam <= 1e-6:
        return bare, -1
    knn = memory.knn_probs(neighbors, temperature=temperature)
    # Reject datastore ids that are not addressable in this model's vocabulary
    # before they can reach the sampler or the embedding table.
    vocab_size = int(logits.shape[0])
    knn = {int(t): float(p) for t, p in knn.items() if _valid_vocab_token(t, vocab_size)}
    if not knn:
        return bare, -1
    lm_probs = _topk_probs(logits)
    blended = {
        t: (1.0 - lam) * lm_probs.get(t, 0.0) + lam * knn.get(t, 0.0)
        for t in set(lm_probs) | set(knn)
    }
    return int(max(blended, key=blended.get)), int(getattr(top, "index", -1))


def _blended_distribution(
    memory: Any,
    key: np.ndarray,
    logits: np.ndarray,
    *,
    k: int,
    temperature: float,
    phi: float | None,
    free_energy: float | None,
    base_lam: float,
    exclude_index: int,
) -> tuple[np.ndarray | None, int]:
    """The advertised mixture, as a full distribution: (1-λ)·p_lm + λ·p_knn.

    Returns ``(None, -1)`` when memory does not fire, so the caller leaves the
    model's own logits untouched.
    """
    if phi is not None and float(phi) < PHI_DORMANT:
        return None, -1
    neighbors = [nb for nb in memory.query(key, k=k) if int(getattr(nb, "index", -1)) != exclude_index]
    if not neighbors:
        return None, -1
    top = neighbors[0]
    min_sim = memory.min_similarity() if hasattr(memory, "min_similarity") else 0.98
    lam = _gated_lambda(float(getattr(top, "similarity", -1.0)), float(min_sim), free_energy, base_lam)
    if lam <= 1e-6:
        return None, -1
    knn = memory.knn_probs(neighbors, temperature=temperature)
    vocab_size = int(logits.shape[0])
    knn = {int(t): float(p) for t, p in knn.items() if _valid_vocab_token(t, vocab_size)}
    if not knn:
        return None, -1
    mixture = (1.0 - lam) * _full_probs(logits)
    for token_id, probability in knn.items():
        mixture[token_id] += lam * probability
    return mixture, int(getattr(top, "index", -1))


def generate_with_memory(
    model: Any,
    tokenizer: Any,
    prompt: str,
    memory: Any,
    *,
    max_tokens: int = 40,
    k: int = 4,
    temperature: float = 2.0,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    use_memory: bool = True,
    base_lam: float = 0.75,
) -> str:
    """KV-cached greedy generation with confident non-parametric recall interpolated per token.

    Prefill once, then one incremental step per token — no full-prefix recompute.
    (A step is still O(sequence length) in attention reads, not O(1).)
    ``use_memory=False`` gives the bare-model baseline for A/B.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    # A zero or negative cap means NO generation. max(1, ...) silently turned a
    # hard no-generation budget into one token of model work plus output.
    try:
        token_budget = int(max_tokens)
    except (TypeError, ValueError, OverflowError):
        record_degradation(
            "nonparametric_generation",
            ValueError(f"invalid max_tokens: {max_tokens!r}"),
            severity="debug",
            action="refused generation rather than guess a budget",
        )
        return ""
    if token_budget <= 0:
        return ""

    ids = list(tokenizer.encode(prompt))
    if not ids:
        # An empty id sequence indexes h[0, -1] on an empty matrix; refuse
        # rather than fail into an empty response from deep inside the backend.
        return ""
    eos = getattr(tokenizer, "eos_token_id", None)
    out: list[int] = []
    last_index = -1
    try:
        cache = make_prompt_cache(model)
        hidden_model = decoder_backbone(model)
        h = hidden_model(mx.array([ids]), cache=cache)
    except _GEN_ERRORS as exc:
        record_degradation("nonparametric_generation_prefill", exc)
        return ""
    for _ in range(token_budget):
        try:
            logits = _as_float32_numpy(_lm_head(model, h[:, -1:, :])[0, -1])
        except _GEN_ERRORS as exc:
            record_degradation("nonparametric_generation_head", exc)
            break
        next_id = int(np.argmax(logits))
        if use_memory:
            try:
                key = normalize(_as_float32_numpy(h[0, -1]))
                next_id, last_index = _select_with_memory(
                    memory, key, logits, k=k, temperature=temperature, phi=phi,
                    free_energy=free_energy, base_lam=base_lam, exclude_index=last_index,
                )
            except _GEN_ERRORS as exc:
                record_degradation("nonparametric_generation_select", exc)
        out.append(next_id)
        if eos is not None and next_id == eos:
            break
        try:
            h = hidden_model(mx.array([[next_id]]), cache=cache)  # incremental: no prefix recompute
        except _GEN_ERRORS as exc:
            record_degradation("nonparametric_generation_step", exc)
            break
    return tokenizer.decode(out).strip()


def make_nonparametric_logits_processor(
    model: Any,
    memory: Any,
    *,
    k: int = 4,
    temperature: float = 2.0,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    base_lam: float = 0.75,
) -> Any:
    """mlx_lm ``(tokens, logits) -> logits`` processor applying the same gated recall.

    Note: a logits-processor has no access to the hidden state, so this form recomputes it
    (an uncached forward over the whole prefix per token — O(n²) overall). Prefer
    ``generate_with_memory`` (KV-cached) for production; this exists for drop-in use inside
    an existing stream_generate call. The returned logits are log-probabilities of the
    interpolated mixture, so downstream temperature/top-p sampling stays meaningful.
    Fail-open: any error returns the logits unchanged.
    """
    import mlx.core as mx

    try:
        hidden_model = decoder_backbone(model)
    except DecoderTopologyError:
        hidden_model = None

    # Anti-stutter state is PER SEQUENCE. It used to be one closure variable
    # shared by every use of the processor, so reusing it across batches,
    # concurrent generations, or successive requests excluded an entry because
    # of a different sequence, and races updated it from two places at once.
    # Keying by sequence length plus the last token detects a continuation and
    # resets whenever the processor is handed an unrelated sequence.
    state: dict[str, Any] = {"last_index": -1, "seq_key": None}
    lock = threading.Lock()

    def _proc(tokens: Any, logits: Any) -> Any:
        try:
            seq = tokens.reshape(1, -1) if hasattr(tokens, "reshape") else mx.array([tokens])
            token_list = np.array(seq).reshape(-1).tolist()
            seq_key = (len(token_list), int(token_list[-1]) if token_list else -1)
            with lock:
                previous = state.get("seq_key")
                continues = (
                    previous is not None
                    and seq_key[0] == previous[0] + 1
                )
                exclude_index = state["last_index"] if continues else -1

            if hidden_model is None:
                return logits
            h = hidden_model(seq)
            key = normalize(_as_float32_numpy(h[0, -1]))
            lg = _as_float32_numpy(logits).reshape(-1)
            mixture, fired = _blended_distribution(
                memory, key, lg, k=k, temperature=temperature, phi=phi,
                free_energy=free_energy, base_lam=base_lam, exclude_index=exclude_index,
            )
            with lock:
                state["last_index"] = fired
                state["seq_key"] = seq_key
            if mixture is None:
                return logits
            # Apply the ACTUAL interpolated distribution. The previous version
            # called the selector only to learn the winning token, then forced
            # that logit to max+1 and left every other logit untouched —
            # discarding lambda and the blended probabilities entirely, which
            # made this a hard argmax override rather than the gated recall
            # interpolation it advertises. Returning log-probabilities makes
            # softmax(out) equal the mixture exactly, so temperature, top-p and
            # every other sampler control keep working.
            out = np.log(np.maximum(mixture, 1e-38)).astype(np.float32)
            return mx.array(out).reshape(logits.shape)
        except _GEN_ERRORS as exc:
            record_degradation("nonparametric_logits_processor", exc)
            return logits

    return _proc

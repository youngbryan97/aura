"""KV-cached foreground wiring for non-parametric memory — the latency-correct form.

The validation-grade processor in ``nonparametric_generation`` recomputes a full forward
over the running tokens every step just to recover the hidden-state query key. That is
O(n²) and would make the 32B foreground take minutes — which is exactly why it was kept out
of the live response path.

This module closes that gap. The model's *normal* generation forward (the one
``stream_generate`` already runs with a KV cache, O(1) per token) computes the hidden state
we need. We capture it as a side effect of that forward instead of recomputing:

  * ``HiddenStateTap`` resolves the loaded checkpoint's text backbone (plain
    ``model.model`` or a wrapped ``model.language_model.model``) and records the
    last-token hidden from each call. Calling the model during generation
    therefore fills ``tap.last_key`` for free.
  * ``make_tapped_nonparametric_processor`` is a standard ``(tokens, logits) -> logits``
    mlx_lm logits-processor that reads ``tap.last_key`` — no extra forward — and interpolates
    the non-parametric recall into the logits. If the tap has nothing (structure mismatch,
    first call), it returns the logits unchanged: fail-open, and crucially **never** falls
    back to the O(n²) recompute on the foreground path.

``cached_generate_with_memory`` is the standalone O(n) reference loop (real KV cache, one
forward per token) used to validate the mechanism end-to-end without the worker.

Everything is fail-open: a tap that can't install, a model whose head can't be found, or any
per-token error leaves generation exactly as it would have been without memory.
"""
from __future__ import annotations

import logging
import pathlib
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from core.brain.llm.decoder_topology import (
    decoder_backbone,
    decoder_backbone_owner,
    decoder_hidden_size,
)
from core.brain.nonparametric_binding import MemoryBinding, binding_for_job
from core.brain.nonparametric_generation import normalize
from core.brain.nonparametric_memory import get_nonparametric_memory
from core.runtime.errors import record_degradation

logger = logging.getLogger("Brain.NonParametricWorker")


def foreground_enabled() -> bool:
    """Whether the foreground non-parametric memory path is switched on.

    Default ON since the July end-to-end proof (tools/nonparametric_proof.py):
    one-shot recall of session-random facts verified on the real model with
    the anisotropy-corrected gate, and an unrelated control generation
    byte-identical with the datastore loaded. Every layer stays fail-open
    (empty store = no processor; below-gate similarity = untouched logits).
    Kill switch: AURA_NONPARAMETRIC_FOREGROUND=0.
    """
    from core.runtime.flags import FlagKind, declare

    return bool(
        declare(
            "AURA_NONPARAMETRIC_FOREGROUND",
            kind=FlagKind.BOOL,
            default=True,
            description="Foreground non-parametric recall blend (proven by tools/nonparametric_proof.py)",
            owner="core.brain.nonparametric_worker",
        ).value()
    )


_STRUCTURAL_OUTPUT_CONTRACT_KINDS = frozenset(
    {
        "exact_reply",
        "list_count",
        "paragraph_count",
        "sentence_count",
        "word_count",
    }
)

_CONTROL_OR_MEASUREMENT_JOB_FLAGS = frozenset(
    {
        "health_probe",
        "warmup_precompile",
        "proof_evaluation_contract",
        "operator_evidence_contract",
        "strict_answer_contract",
        "strict_value_contract",
    }
)


def foreground_memory_admitted_for_job(job: Any) -> bool:
    """Keep associative recall out of control and structural decoding.

    A readiness probe measures whether the resident model can answer. Letting a
    mutable recall store steer that probe instead measures the store/model pair
    and can deterministically replace the requested probe answer with an
    unrelated memorized continuation. The same isolation applies to proof,
    operator-evidence, warmup, and strict-value jobs: supplemental memory must
    not alter the control plane or the object being measured.

    A structural contract is likewise a decoder constraint, not a retrieval request.
    The non-parametric store may still participate when the caller explicitly
    declares that memory grounding is required; otherwise Aura's ordinary
    model and response-contract machinery own the turn.
    """

    if not isinstance(job, dict):
        return True
    if any(bool(job.get(flag)) for flag in _CONTROL_OR_MEASUREMENT_JOB_FLAGS):
        return False
    contract = job.get("requested_output_contract")
    kind = (
        str(contract.get("kind") or "").strip().lower()
        if isinstance(contract, dict)
        else ""
    )
    if kind not in _STRUCTURAL_OUTPUT_CONTRACT_KINDS:
        return True
    return bool(
        job.get("requires_memory_grounding")
        or job.get("memory_state_contract")
        or job.get("grounded_recall_contract")
    )


# Last foreground-recall outcome, so a request can report whether memory was
# installed, deliberately skipped, empty, or failed to build. Every path
# returned None before, which made those indistinguishable (CP126 29374cf0).
_RECALL_OUTCOME: dict[str, Any] = {"status": "not_attempted", "detail": ""}


def _set_recall_outcome(status: str, detail: str = "") -> None:
    _RECALL_OUTCOME["status"] = status
    _RECALL_OUTCOME["detail"] = detail


def last_recall_outcome() -> dict[str, Any]:
    """What the most recent foreground-recall build actually did.

    status is one of: not_attempted, disabled, not_admitted, unavailable,
    empty, installed, failed.
    """
    return dict(_RECALL_OUTCOME)


# A datastore has to be able to answer before it is allowed to speak.
#
# LIVE DEFECT, 2026-07-26. The resident 32B served fluent, grammatical,
# meaning-free replies to ordinary questions — and kept serving them with
# substrate steering clamped to 0.01 and recurrent depth off, which is what
# ruled both of those out as the cause:
#
#   "Define S as extracting for Draw [w] from colored ([I:E]): (card frequency
#    in the bag - matching cards already know to counted)"
#
# The live datastore (~/.aura/data/runtime/nonparametric_memory_5120, built
# 2026-07-13) holds 1,689 hidden-state keys of which 1,677 — 99.3% — carry no
# decoded token text at all. Their token_ids are the most ordinary tokens in
# the vocabulary: space, digits, "the", "is", "to". The remaining 12 store an
# entire ANSWER as a single "token", which is not what a per-token kNN store
# holds either.
#
# Blending THAT into the model's top-64 logits, at a weight that reaches 0.87,
# is a recipe for text that is grammatically shaped and says nothing — which
# is precisely the failure. Recall is an enhancement; a store that cannot
# support recall must decline, and the module's stated contract is already to
# fail open to normal generation.
_MIN_USABLE_ENTRY_FRACTION = 0.5
_MIN_USABLE_ENTRIES = 32

#: DECODABLE IS NOT THE SAME AS APPLICABLE.
#:
#: Measured 2026-07-29, and I caused it. The 2026-07-13 store was refused for
#: carrying no token text; decoding its ids from the resident tokenizer took it
#: from 12 usable entries to 1,488, the guard above passed, and the store began
#: steering live generation. Two demo turns immediately degraded:
#:
#:   "Neurotransmitter profile actually includes dopamine, serotonin and
#:    norepagephrine like apes"
#:   "the inner core could prefer crystallishing iron alloys ... beneath a
#:    potential 'lagging' crust"
#:
#: Garbled words and a fabricated premise — exactly the "grammatically shaped
#: and says nothing" failure the original guard was written for. The reason is
#: not the decode: those 1,689 keys were ingested from a CODING corpus ("Here
#: is the fix:", "def add(a, b)"), and blending a narrow domain into open
#: conversation at a weight reaching 0.87 corrupts it.
#:
#: A kNN store is only safe to blend when it is large enough that its
#: neighbours are actually near the query. At 1,689 keys over a 5120-wide
#: space the nearest neighbour of "octopus cognition" is whatever coding token
#: happens to be least far away, which is noise wearing the shape of grammar.
#: This floor is what that costs; it is a density requirement, not a taste.
_MIN_ENTRIES_TO_STEER_GENERATION = 50_000


#: Refusals already reported, so a permanently-unusable store is named once
#: per process rather than once per turn. The 2026-07-13 store produced 591
#: identical warnings in a single session — the guard was working and the log
#: was the only thing that suffered.
_REPORTED_UNUSABLE: set[str] = set()


def _quarantine_unusable_datastore(memory: Any, reason: str) -> str:
    """Move a provably-unusable store aside so a good one can be built.

    Renamed, never deleted: the files are evidence of how the store went wrong
    and they are the user's data. What matters is that they stop being loaded,
    because the guard below is permanent — nothing else retires the store, so
    without this the faculty is dark for every future session too.
    """
    raw_path = str(getattr(memory, "_path", "") or "")
    if not raw_path:
        return ""
    base = pathlib.Path(raw_path).expanduser()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moved: list[str] = []
    for suffix in (".keys.npy", ".meta.json"):
        source = base.with_name(base.name + suffix)
        if not source.exists():
            continue
        target = source.with_name(f"{source.name}.unusable-{stamp}")
        try:
            source.rename(target)
            moved.append(target.name)
        except OSError as exc:
            logger.warning("Could not quarantine %s: %s", source, exc)
            return ""
    if not moved:
        return ""
    logger.warning(
        "Non-parametric memory: quarantined an unusable datastore (%s). "
        "Renamed to %s; a fresh store will build from this session onward.",
        reason,
        ", ".join(moved),
    )
    return ", ".join(moved)


def _unusable_datastore_reason(memory: Any) -> str:
    """Why this datastore may not steer live generation, or "" if it may."""
    try:
        tokens = list(getattr(memory, "_tokens", None) or [])
    except (AttributeError, TypeError, ValueError):
        return ""
    total = len(tokens)
    if total == 0:
        return ""
    usable = sum(1 for token in tokens if str(token or "").strip())
    reason = ""
    if total < _MIN_ENTRIES_TO_STEER_GENERATION:
        # Sparse: report and decline, but do NOT quarantine. A small store is
        # a store that has not grown yet, not a broken one, and deleting it
        # would throw away the beginning of a good one.
        return (
            f"{total} entries is too sparse to steer a {getattr(memory, '_dim', '?')}"
            f"-wide space (need {_MIN_ENTRIES_TO_STEER_GENERATION:,}); "
            "recall stays off until the store is dense enough for its "
            "neighbours to be near"
        )
    if usable < _MIN_USABLE_ENTRIES:
        reason = (
            f"only {usable} of {total} entries carry a recallable token "
            f"(need at least {_MIN_USABLE_ENTRIES})"
        )
    elif usable < total * _MIN_USABLE_ENTRY_FRACTION:
        reason = (
            f"{total - usable} of {total} entries carry no recallable token "
            f"({100.0 * usable / total:.1f}% usable)"
        )
    if not reason:
        return ""
    # SAY IT ONCE, AND THEN DO SOMETHING ABOUT IT.
    #
    # This condition cannot improve on its own — the store on disk is what it
    # is — so repeating the verdict every turn is noise and leaving the store
    # in place keeps the faculty dark forever. Quarantine it once and let a
    # fresh one accumulate.
    key = f"{str(getattr(memory, '_path', '') or '?')}:{total}:{usable}"
    if key not in _REPORTED_UNUSABLE:
        _REPORTED_UNUSABLE.add(key)
        _quarantine_unusable_datastore(memory, reason)
    return reason


def maybe_build_foreground(
    model: Any,
    *,
    job: Any = None,
) -> tuple[HiddenStateTap, Callable[[Any, Any], Any]] | None:
    """Build (tap, processor) for the live worker iff foreground memory is on and non-empty.

    Returns None when disabled, when there is no datastore, or when the datastore is empty —
    so the live path pays nothing (no tap, no processor) unless there is genuinely something
    to recall. Fully fail-open: any error returns None and the worker generates normally.
    """
    if not foreground_enabled():
        _set_recall_outcome("disabled", "foreground recall is switched off")
        return None
    if not foreground_memory_admitted_for_job(job):
        _set_recall_outcome("not_admitted", "this job did not admit foreground recall")
        return None
    try:
        dim = decoder_hidden_size(model)
        memory = get_nonparametric_memory(dim)
        if memory is None:
            _set_recall_outcome("unavailable", "no datastore")
            return None
        if len(memory) == 0:
            _set_recall_outcome("empty", "datastore holds no entries")
            return None
        unusable = _unusable_datastore_reason(memory)
        if unusable:
            _set_recall_outcome("not_admitted", unusable)
            logger.warning(
                "🧠 [WORKER] Foreground non-parametric memory REFUSED: %s. "
                "Generating from the model alone.",
                unusable,
            )
            return None
        binding = binding_for_job(job, source_id="foreground_recall")
        if binding is None:
            # A job that does not say whose turn this is gets no recall.
            # Reading every principal's entries because the request forgot
            # to identify itself is precisely the leak the store's
            # principal argument exists to prevent.
            _set_recall_outcome(
                "not_admitted",
                "this job names no principal, and unscoped recall reads every "
                "principal's entries",
            )
            return None
        tap = HiddenStateTap(model)
        proc = make_tapped_nonparametric_processor(tap, memory, binding=binding)
        logger.info("🧠 [WORKER] Foreground non-parametric memory ACTIVE (%d entries, dim=%d).",
                    len(memory), dim)
        _set_recall_outcome("installed", f"{len(memory)} entries at dim {dim}")
        return tap, proc
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        # CP126 29374cf0. Failing open is RIGHT here — recall is an
        # enhancement, and generating normally without it is the correct
        # degradation. What was missing is that the caller could not tell
        # installed from skipped from broken: every path returned None, and
        # a genuine build error was filed at debug alongside routine
        # "nothing to recall".
        #
        # The outcome is now recorded so a request can say which happened,
        # and a real failure is a warning rather than debug noise.
        _set_recall_outcome("failed", f"{type(exc).__name__}: {exc}")
        record_degradation(
            "nonparametric_foreground_build",
            exc,
            severity="warning",
            action="foreground non-parametric memory not installed; normal generation",
        )
        return None


# ── head detection: hidden states → logits, model-agnostic ──────────────────

def _logits_from_hidden(model: Any, hidden: Any) -> Any:
    """Project hidden states to vocab logits using the model's own (possibly tied) head."""
    language = decoder_backbone_owner(model)
    args = getattr(language, "args", None)
    if getattr(args, "tie_word_embeddings", False):
        inner = decoder_backbone(model)
        embed = getattr(inner, "embed_tokens", None)
        if embed is not None and hasattr(embed, "as_linear"):
            return embed.as_linear(hidden)
    lm_head = getattr(language, "lm_head", None)
    if lm_head is not None:
        return lm_head(hidden)
    # Last resort: a tied embedding without the flag set.
    inner = decoder_backbone(model)
    embed = getattr(inner, "embed_tokens", None)
    if embed is not None and hasattr(embed, "as_linear"):
        return embed.as_linear(hidden)
    raise AttributeError("could not locate the model's output head")


# ── the tap: capture the hidden the generation forward already computes ──────

class _TappedInner:
    """Transparent proxy around the decoder backbone that records last-token hidden."""

    def __init__(self, inner: Any, tap: HiddenStateTap) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_tap", tap)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self._inner(*args, **kwargs)
        try:
            # out is hidden states [B, T, H]; record the last position of the last call.
            self._tap.last_key = normalize(np.array(out[0, -1], dtype=np.float32))
        except (IndexError, ValueError, TypeError):
            self._tap.last_key = None
        return out

    # Delegate everything else so the proxy is indistinguishable from the real module.
    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)


class HiddenStateTap:
    """Installs a recording proxy around the resolved decoder for one generation.

    Use as a context manager. If the swap can't be done (unexpected structure, mlx setattr
    quirk), ``active`` stays False and the tap simply records nothing — the caller's
    processor then no-ops, so generation is unaffected.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._owner: Any = None
        self._real_inner: Any = None
        self.active = False
        self.last_key: np.ndarray | None = None

    def __enter__(self) -> HiddenStateTap:
        try:
            self._owner = decoder_backbone_owner(self._model)
            inner = decoder_backbone(self._model)
            self._real_inner = inner
            self._owner.model = _TappedInner(inner, self)
            self.active = True
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("nonparametric_worker_tap", exc, severity="debug",
                               action="hidden-state tap disabled; foreground memory inert")
            self.active = False
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._real_inner is not None:
            try:
                self._owner.model = self._real_inner
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                record_degradation("nonparametric_worker_tap", e, severity="warning",
                                   action="failed to restore model.model after tap")
        self._real_inner = None
        self._owner = None
        self.active = False


# ── the foreground processor: reads the tap, no recompute ───────────────────

def make_tapped_nonparametric_processor(
    tap: HiddenStateTap,
    memory: Any,
    *,
    binding: MemoryBinding,
    k: int = 4,
    temperature: float = 0.1,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    min_cos: float = 0.55,
    # A retrieved neighbour may INFORM the next token; it may not choose it.
    # base_lam was 0.75 and the free-energy term below multiplies by up to
    # 1.16, so a single neighbour could take 87% of the next-token
    # distribution away from the model — against ~0.25 in the kNN-LM
    # literature, and tuned there on held-out data rather than asserted.
    # At 0.87 the datastore is not augmenting generation, it is performing it.
    base_lam: float = 0.25,
    max_lam: float = 0.35,
) -> Callable[[Any, Any], Any]:
    """O(1)-per-token non-parametric logits-processor driven by the hidden-state tap.

    ``binding`` is required. This called ``memory.query(key, k=k)`` with no
    principal, and the store's own docstring says an empty principal
    searches EVERYTHING — so the isolation the storage layer implements was
    not enforced at the one seam where a person's turn meets the datastore.
    """
    import mlx.core as mx

    state = {"last_fired_index": -1}

    def _proc(tokens: Any, logits: Any) -> Any:
        key = tap.last_key
        if key is None:
            return logits  # tap inactive / no hidden yet → leave logits untouched (fail-open)
        try:
            neighbors = memory.query(key, k=k, principal=binding.principal)
            if not neighbors:
                return logits
            # Anisotropy-corrected gate (see Neighbor.similarity): raw cosine
            # cannot separate unrelated prompts on real hidden states.
            sim = float(getattr(neighbors[0], "similarity", -1.0))
            gate = float(getattr(memory, "min_similarity", lambda: min_cos)())
            nearest_index = int(getattr(neighbors[0], "index", -1))
            if sim < gate:
                return logits
            # Anti-stutter: the same nearest entry twice in a row means the
            # recalled chain ended and its tail is re-firing.
            if nearest_index == state["last_fired_index"]:
                return logits
            state["last_fired_index"] = nearest_index
            fe = 0.5 if free_energy is None else float(free_energy)
            lam = base_lam * ((sim - gate) / max(1e-6, 1.0 - gate)) * (0.6 + 0.8 * fe)
            lam = min(float(max_lam), max(0.0, float(lam)))
            lg = np.array(logits, dtype=np.float32).reshape(-1)
            ktop = min(64, lg.shape[0])
            idx = np.argpartition(lg, -ktop)[-ktop:]
            sub = lg[idx] - lg[idx].max()
            ex = np.exp(sub)
            ex /= ex.sum()
            lm_probs = {int(t): float(p) for t, p in zip(idx, ex, strict=True)}
            blended = memory.interpolate(
                lm_probs, key, k=k, temperature=temperature, phi=phi,
                free_energy=free_energy, lam_override=min(lam, 0.9),
                principal=binding.principal,
            )
            out = lg.copy()
            import math as _m

            for t, p in blended.items():
                out[int(t)] = _m.log(max(p, 1e-12))
            return mx.array(out).reshape(logits.shape)
        except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
            record_degradation("nonparametric_tapped_processor", exc)
            return logits

    return _proc


# ── standalone O(n) reference loop (validation without the worker) ──────────

def cached_generate_with_memory(
    model: Any,
    tokenizer: Any,
    prompt: str,
    memory: Any,
    *,
    max_tokens: int = 40,
    k: int = 4,
    # kNN softmax temperature over UNIT-key L2 distances: exact match d=0
    # must dominate an unrelated entry at d≈0.35, so the scale is ~0.1 —
    # the old 2.0 made the kNN distribution nearly uniform across entries
    # (measured: cross-fact digit leakage corrupted recall).
    temperature: float = 0.1,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    use_memory: bool = True,
    principal: str = "",
    min_cos: float = 0.55,
    base_lam: float = 0.75,
) -> str:
    """Greedy generation with a real KV cache: one incremental forward per token.

    The hidden key for step t is read from the *same* cached forward that produces the
    step-t logits, so adding non-parametric recall costs a datastore query, not a forward.
    This is the latency-correct shape the foreground tap mirrors. Fail-open per token.
    """
    import mlx.core as mx

    try:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("nonparametric_cached_generate", exc,
                           action="KV cache unavailable; aborting cached generation")
        return ""

    ids = list(tokenizer.encode(prompt))
    eos = getattr(tokenizer, "eos_token_id", None)
    out_ids: list[int] = []
    last_fired_index = -1  # anti-stutter: an entry may not fire twice in a row

    # Prefill the cache with the full prompt, then decode one token at a time.
    cursor = mx.array([ids])
    try:
        hidden_model = decoder_backbone(model)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("nonparametric_cached_generate", exc)
        return ""
    for _step in range(max(1, int(max_tokens))):
        try:
            hidden = hidden_model(cursor, cache=cache)          # cached forward (incremental)
            logits = _logits_from_hidden(model, hidden)
            key = normalize(np.array(hidden[0, -1], dtype=np.float32))
            lg = np.array(logits[0, -1], dtype=np.float32)
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            record_degradation("nonparametric_cached_generate", exc)
            break

        next_id = int(np.argmax(lg))
        if use_memory:
            try:
                # Same scoping as the foreground processor. An empty
                # principal searches every entry in the store, so a proof
                # harness that omits one is measuring a different store
                # from the one a request would see.
                neighbors = memory.query(key, k=k, principal=principal)
                if neighbors:
                    # Anisotropy-corrected gate (see Neighbor.similarity).
                    sim = float(getattr(neighbors[0], "similarity", -1.0))
                    gate = float(getattr(memory, "min_similarity", lambda: min_cos)())
                    nearest_index = int(getattr(neighbors[0], "index", -1))
                    # Anti-stutter: a chain walks DIFFERENT entries each
                    # step; the same nearest entry twice in a row means the
                    # chain ended and the stale tail is re-firing.
                    if sim >= gate and nearest_index != last_fired_index:
                        fe = 0.5 if free_energy is None else float(free_energy)
                        lam = base_lam * max(0.0, (sim - gate) / max(1e-6, 1.0 - gate)) * (0.6 + 0.8 * fe)
                        from core.brain.nonparametric_generation import _topk_probs
                        blended = memory.interpolate(
                            _topk_probs(lg), key, k=k, temperature=temperature, phi=phi,
                            free_energy=free_energy, lam_override=min(lam, 0.9),
                            principal=principal,
                        )
                        memory_choice = int(max(blended, key=blended.get))
                        if memory_choice != next_id:
                            last_fired_index = nearest_index
                        next_id = memory_choice
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                record_degradation("nonparametric_cached_generate_interp", exc)

        out_ids.append(next_id)
        if eos is not None and next_id == eos:
            break
        cursor = mx.array([[next_id]])   # only the new token next step — O(1) forward

    return tokenizer.decode(out_ids).strip()

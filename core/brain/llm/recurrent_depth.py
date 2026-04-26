"""core/brain/llm/recurrent_depth.py — Inference-time layer looping.

Implements the Mythos-inspired recurrent-depth concept:
loop a subset of transformer layers multiple times before the final layers,
giving the model more "think time" in latent space before committing to output.

This is NOT optional fluff. This changes how the LLM fundamentally processes
information. Instead of: input → layers → output (one pass), it becomes:

    input → prelude → [recurrent block × N] → coda → output

The recurrent block iterates, refining its hidden representation each pass.
The model "thinks" in latent space before the coda commits to words.

Architecture (Prelude-Recurrent-Coda):
  - PRELUDE layers [0..split_start): Run once. Build initial representation.
  - RECURRENT layers [split_start..split_end): Run N times. The "thinking."
  - CODA layers [split_end..end): Run once. Produce final output.

KV Cache Handling (CRITICAL):
  Each transformer layer's attention appends K/V to its cache on every call.
  If we naively loop the recurrent layers, the cache gets N copies of K/V
  entries for the same token position, corrupting generation.

  Solution: On extra loops (all but the last), we snapshot and restore the
  cache state for the recurrent layers. Only the FINAL loop's K/V is kept.
  This means the model gets N passes of computation through the layers,
  but the cache only sees one pass — preserving correct attention patterns
  for future tokens.

Residual Injection (LTI-Stable):
  Each loop mixes a fraction (alpha) of the original embedding back into
  the hidden state. This prevents hidden state explosion — the representation
  stays grounded to the input even after multiple loops.

Per-Lane Configuration:
  Different model sizes need different loop counts:
  - Cortex (32B):   2 loops — meaningful improvement without excessive latency
  - Solver (72B):   1 loop — 72B is already deep; extra looping is too slow
                     for interactive solver turns in the live handoff path
  - Brainstem (7B): 1 loop — small model, looping doesn't help much
  - Reflex (1.5B):  1 loop — too small, looping would slow without benefit
"""
from __future__ import annotations

import copy
import logging
import os
from typing import Optional

logger = logging.getLogger("Aura.RecurrentDepth")


# ── Per-lane default configurations ──────────────────────────────────────
# Maps model size heuristics to default loop counts.
# These are applied automatically unless overridden by env vars.
#
# NOTE: Layer counts corrected — Qwen2.5-32B is 64 layers (NOT 48-55) and
# Qwen2.5-72B is 80 layers. The old ranges had 32B and 72B colliding into
# the same bucket via (56, 999), which was correct by accident but the
# in-file labels were misleading.
LANE_DEFAULTS = {
    # (min_layers, max_layers): (n_loops, prelude_frac, coda_frac, alpha)
    (72, 999):  (1, 0.15, 0.15, 0.1),   # 72B (80 layers) — interactive solver
    (56, 71):   (2, 0.20, 0.20, 0.1),   # 32B (64 layers) — good balance
    (24, 55):   (1, 0.20, 0.20, 0.1),   # 14B (40 layers) — marginal benefit
    (0,  23):   (1, 0.20, 0.20, 0.1),   # 7B and below — too small
}


def _get_lane_defaults(num_layers: int) -> tuple:
    """Get default recurrent depth config based on model size."""
    for (min_l, max_l), config in LANE_DEFAULTS.items():
        if min_l <= num_layers <= max_l:
            return config
    return (1, 0.20, 0.20, 0.1)  # fallback: standard


def _lane_specific_loop_env(num_layers: int) -> Optional[str]:
    if num_layers >= 72:
        return os.environ.get("AURA_RECURRENT_LOOPS_72B")
    if num_layers >= 56:
        return os.environ.get("AURA_RECURRENT_LOOPS_32B")
    if num_layers >= 24:
        return os.environ.get("AURA_RECURRENT_LOOPS_14B")
    return os.environ.get("AURA_RECURRENT_LOOPS_SMALL")


class CacheSnapshotError(RuntimeError):
    """Raised when the KV cache does not support snapshot/restore.

    Silent failure here is worse than useless: the recurrent block would
    accumulate N copies of K/V into the cache, corrupting attention for
    future tokens. Fail loud so the operator sees the bug.
    """


def _snapshot_recurrent_caches(cache, start: int, end: int) -> list:
    """Save cache state for recurrent layers before an extra loop.

    Each cache entry has an offset and stored K/V arrays. We save enough
    state to restore after the loop, so only the final pass persists.

    Raises:
        CacheSnapshotError — if the cache type does not expose a restorable
        snapshot. Never silently returns partial state.
    """
    snapshots = []
    for i in range(start, end):
        c = cache[i]
        if c is None:
            snapshots.append(None)
            continue
        # Prefer the canonical mlx_lm KVCache contract: state / meta_state.
        if hasattr(c, "state") and hasattr(c, "meta_state"):
            snapshots.append(("state", c.state, c.meta_state))
            continue
        # Fallback: direct attribute snapshot for simple cache types.
        if all(hasattr(c, attr) for attr in ("keys", "values", "offset")):
            snapshots.append(("attrs", c.keys, c.values, c.offset))
            continue
        raise CacheSnapshotError(
            f"KV cache at layer {i} ({type(c).__name__}) supports neither "
            "state/meta_state nor keys/values/offset — recurrent depth cannot "
            "run safely on this cache. Set AURA_RECURRENT_LOOPS=0 or upgrade "
            "mlx_lm."
        )
    return snapshots


def _restore_recurrent_caches(cache, start: int, end: int, snapshots: list):
    """Restore cache state for recurrent layers after a non-final loop."""
    for idx, i in enumerate(range(start, end)):
        c = cache[i]
        snap = snapshots[idx]
        if c is None or snap is None:
            continue
        kind = snap[0]
        if kind == "state":
            c.state = snap[1]
            c.meta_state = snap[2]
        elif kind == "attrs":
            c.keys = snap[1]
            c.values = snap[2]
            c.offset = snap[3]
        else:
            raise CacheSnapshotError(f"Unknown cache snapshot kind: {kind!r}")


def _self_test_cache_snapshot() -> None:
    """Boot-time sanity check: prove the KV cache genuinely rewinds.

    This is the load-bearing assumption of the whole module. If the installed
    mlx_lm changes the KVCache contract and `state` becomes read-only, the
    old fallback path would silently pass — that was the bug. We now fail
    loud on the first call to :func:`apply_recurrent_depth`.
    """
    try:
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
    except Exception as exc:
        raise CacheSnapshotError(f"mlx_lm.models.cache.KVCache unavailable: {exc}") from exc

    c = KVCache()
    k0 = mx.ones((1, 1, 4, 8))
    v0 = mx.ones((1, 1, 4, 8))
    c.update_and_fetch(k0, v0)
    pre_offset = c.offset

    snaps = _snapshot_recurrent_caches([c], 0, 1)

    # Simulate a recurrent-loop mutation.
    c.update_and_fetch(mx.ones((1, 1, 1, 8)) * 2, mx.ones((1, 1, 1, 8)) * 2)
    if c.offset <= pre_offset:
        raise CacheSnapshotError(
            f"Sanity-check mutation did not advance cache offset "
            f"({pre_offset} → {c.offset}); cache semantics unexpected."
        )

    _restore_recurrent_caches([c], 0, 1, snaps)
    if c.offset != pre_offset:
        raise CacheSnapshotError(
            f"Cache snapshot/restore FAILED: offset {pre_offset} → "
            f"{c.offset} after restore. Recurrent depth would corrupt K/V. "
            "Refusing to patch. Set AURA_RECURRENT_LOOPS=0 or upgrade mlx_lm."
        )


def apply_recurrent_depth(
    model,
    n_loops: int = 2,
    prelude_frac: float = 0.20,
    coda_frac: float = 0.20,
    residual_alpha: float = 0.1,
) -> bool:
    """Patch a Qwen2-style model to loop its middle layers.

    This changes the fundamental forward pass of the model. After patching,
    every inference call runs the recurrent block n_loops times.

    Args:
        model: The MLX model object (must have model.model.layers)
        n_loops: Number of times to run the recurrent block (1 = standard)
        prelude_frac: Fraction of layers for prelude (default 20%)
        coda_frac: Fraction of layers for coda (default 20%)
        residual_alpha: Embedding residual injection strength per loop

    Returns:
        True if patch was applied, False if model structure not recognized.
    """
    try:
        import mlx.core as mx
    except ImportError:
        logger.warning("mlx not available — recurrent depth not applied")
        return False

    # ── Locate the inner model ───────────────────────────────────────
    inner = getattr(model, "model", None)
    if inner is None:
        logger.warning("Model has no .model attribute — cannot apply recurrent depth")
        return False

    layers = getattr(inner, "layers", None)
    if layers is None or not isinstance(layers, list):
        logger.warning("Model has no .layers list — cannot apply recurrent depth")
        return False

    num_layers = len(layers)
    if num_layers < 4:
        logger.warning("Model has only %d layers — too few for recurrent depth", num_layers)
        return False

    if n_loops <= 1:
        logger.info("Recurrent depth: n_loops=%d for %d-layer model — standard pass", n_loops, num_layers)
        return True

    # Gate: prove the cache supports snapshot/restore BEFORE we patch. This
    # guards against a silently-broken setup that would corrupt K/V.
    try:
        _self_test_cache_snapshot()
    except CacheSnapshotError as exc:
        logger.error("🚫 Recurrent depth DISABLED: %s", exc)
        return False

    # ── Compute split points ─────────────────────────────────────────
    prelude_end = max(1, int(num_layers * prelude_frac))
    coda_start = min(num_layers - 1, num_layers - max(1, int(num_layers * coda_frac)))

    if coda_start <= prelude_end:
        logger.warning("Recurrent block empty after split (prelude=%d, coda=%d) — not patching",
                       prelude_end, coda_start)
        return False

    recurrent_count = coda_start - prelude_end

    logger.info(
        "🧠 Recurrent Depth: %d layers → Prelude[0:%d] Recurrent[%d:%d]×%d Coda[%d:%d]",
        num_layers, prelude_end,
        prelude_end, coda_start, n_loops,
        coda_start, num_layers,
    )

    # ── Remove existing patch if present ─────────────────────────────
    if hasattr(inner, "_recurrent_depth_original_call"):
        inner.__class__.__call__ = inner._recurrent_depth_original_call
        del inner._recurrent_depth_original_call
        if hasattr(inner, "_recurrent_depth_config"):
            del inner._recurrent_depth_config

    # ── Save original ────────────────────────────────────────────────
    original_call = inner.__class__.__call__
    inner._recurrent_depth_original_call = original_call

    # ── Build the patched forward pass ───────────────────────────────
    # Closure captures: prelude_end, coda_start, num_layers, n_loops,
    #                   residual_alpha
    def recurrent_forward(self, inputs, cache=None, input_embeddings=None):
        """Mythos-inspired recurrent-depth forward pass.

        Prelude → [Recurrent × N with cache save/restore] → Coda
        """
        # ── Embedding ────────────────────────────────────────────
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        # Save embedding for residual injection (LTI-stable grounding)
        h_embed = h

        if cache is None:
            cache = [None] * len(self.layers)

        # Build attention mask using the model-family-agnostic helper so this
        # module doesn't break when swapping to Llama / Mistral / etc.
        try:
            from mlx_lm.models.base import create_attention_mask
        except ImportError:
            from mlx_lm.models.qwen2 import create_attention_mask  # type: ignore
        mask = create_attention_mask(h, cache[0])

        # ── PRELUDE: layers [0..prelude_end) — run once ──────────
        for i in range(prelude_end):
            h = self.layers[i](h, mask, cache[i])

        # ── RECURRENT: layers [prelude_end..coda_start) — run N times ─
        for loop_idx in range(n_loops):
            is_final_loop = (loop_idx == n_loops - 1)

            # Before non-final loops: snapshot cache state for recurrent layers
            # so we can restore after — only the final loop's K/V persists
            if not is_final_loop:
                cache_snapshot = _snapshot_recurrent_caches(
                    cache, prelude_end, coda_start
                )

            # Run the recurrent block
            for i in range(prelude_end, coda_start):
                h = self.layers[i](h, mask, cache[i])

            # After non-final loops: restore cache (undo K/V append)
            # and inject residual from embedding to stabilize hidden state
            if not is_final_loop:
                _restore_recurrent_caches(
                    cache, prelude_end, coda_start, cache_snapshot
                )
                # LTI-stable residual injection: ground the representation
                h = h + residual_alpha * h_embed

        # ── CODA: layers [coda_start..end) — run once ────────────
        for i in range(coda_start, num_layers):
            h = self.layers[i](h, mask, cache[i])

        return self.norm(h)

    # ── Apply the patch ──────────────────────────────────────────────
    inner.__class__.__call__ = recurrent_forward

    # Store config for inspection and status APIs
    inner._recurrent_depth_config = {
        "n_loops": n_loops,
        "prelude_end": prelude_end,
        "coda_start": coda_start,
        "recurrent_layers": recurrent_count,
        "num_layers": num_layers,
        "residual_alpha": residual_alpha,
        "prelude_frac": prelude_frac,
        "coda_frac": coda_frac,
    }

    logger.info(
        "✅ RECURRENT DEPTH ACTIVE: %d loops, α=%.2f, "
        "recurrent block = %d layers [%d→%d], "
        "model now THINKS before answering",
        n_loops, residual_alpha,
        recurrent_count, prelude_end, coda_start,
    )
    return True


def remove_recurrent_depth(model) -> bool:
    """Remove the recurrent depth patch, restoring standard forward pass."""
    inner = getattr(model, "model", None)
    if inner is None:
        return False

    if not hasattr(inner, "_recurrent_depth_original_call"):
        logger.debug("No recurrent depth patch found — nothing to remove")
        return False

    inner.__class__.__call__ = inner._recurrent_depth_original_call
    del inner._recurrent_depth_original_call
    if hasattr(inner, "_recurrent_depth_config"):
        del inner._recurrent_depth_config

    logger.info("Recurrent depth removed — standard forward pass restored")
    return True


def get_recurrent_config(model) -> Optional[dict]:
    """Get the current recurrent depth configuration, or None if not patched."""
    inner = getattr(model, "model", None)
    if inner is None:
        return None
    return getattr(inner, "_recurrent_depth_config", None)


def resolve_loops_for_model(model) -> int:
    """Determine the correct number of loops for a model based on its size.

    Checks env vars first, then falls back to per-lane defaults based on
    the number of transformer layers.
    """
    # Explicit env override takes priority
    env_loops = os.environ.get("AURA_RECURRENT_LOOPS")
    if env_loops is not None:
        n = int(env_loops)
        if n == 0:
            # Explicitly disabled
            return 0
        return n

    # Auto-detect based on model size
    inner = getattr(model, "model", None)
    if inner is None:
        return 1

    layers = getattr(inner, "layers", None)
    if layers is None:
        return 1

    num_layers = len(layers)
    lane_override = _lane_specific_loop_env(num_layers)
    if lane_override is not None:
        n = int(lane_override)
        if n == 0:
            return 0
        return n
    defaults = _get_lane_defaults(num_layers)
    return defaults[0]  # n_loops


def apply_for_model(model) -> bool:
    """Apply recurrent depth with auto-detected or env-configured settings.

    This is the primary entry point. Call after model load.
    It automatically determines the right loop count for the model size,
    and can be overridden with AURA_RECURRENT_LOOPS env var.

    Set AURA_RECURRENT_LOOPS=0 to explicitly disable.
    """
    n_loops = resolve_loops_for_model(model)

    if n_loops <= 1:
        inner = getattr(model, "model", None)
        num_layers = len(getattr(inner, "layers", [])) if inner else 0
        logger.info(
            "Recurrent depth: standard pass for %d-layer model (n_loops=%d)",
            num_layers, n_loops,
        )
        return False

    # Get other params from env or defaults
    inner = getattr(model, "model", None)
    num_layers = len(getattr(inner, "layers", [])) if inner else 64
    defaults = _get_lane_defaults(num_layers)

    prelude_frac = float(os.environ.get("AURA_RECURRENT_PRELUDE", defaults[1]))
    coda_frac = float(os.environ.get("AURA_RECURRENT_CODA", defaults[2]))
    alpha = float(os.environ.get("AURA_RECURRENT_ALPHA", defaults[3]))

    return apply_recurrent_depth(
        model,
        n_loops=n_loops,
        prelude_frac=prelude_frac,
        coda_frac=coda_frac,
        residual_alpha=alpha,
    )

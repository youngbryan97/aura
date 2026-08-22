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

Depth Policy:
  Loop counts are resolved from model shape and runtime budget, not from
  named lanes. Operators can still override by model-size class for
  diagnostics, but the default policy is a general compute/latency tradeoff.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.model_layers import resolve_model_layers

logger = logging.getLogger("Aura.RecurrentDepth")

_DEFAULT_MAX_RECURRENT_LOOPS = 4
_ABSOLUTE_MAX_RECURRENT_LOOPS = 8


# ── Model-profile default configurations ─────────────────────────────────
# Maps model depth heuristics to default loop counts.
# These are applied automatically unless overridden by env vars.
#
# NOTE: Layer counts corrected — Qwen2.5-32B is 64 layers (NOT 48-55) and
# Qwen2.5-72B is 80 layers. The old ranges had 32B and 72B colliding into
# the same bucket via (56, 999), which was correct by accident but the
# in-file labels were misleading.
MODEL_PROFILE_DEFAULTS = {
    # (min_layers, max_layers): (n_loops, prelude_frac, coda_frac, alpha)
    (72, 999):  (1, 0.15, 0.15, 0.1),   # 72B (80 layers) — interactive solver
    (56, 71):   (2, 0.20, 0.20, 0.1),   # 32B (64 layers) — recurrent thinking
    (24, 55):   (1, 0.20, 0.20, 0.1),   # 14B (40 layers) — marginal benefit
    (0,  23):   (1, 0.20, 0.20, 0.1),   # 7B and below — too small
}

# 2026-07-26: this default was briefly set to 1 while chasing the cause of
# fluent-nonsense replies on the live desktop. The A/B disproved it — the
# nonsense persisted with depth off AND substrate steering clamped to 0.01
# (worker log: "Surface decode: steering α=0.01, depth_present=False"), and
# the actual cause was the foreground kNN datastore, whose 1,689 entries were
# 99.3% empty and were being blended into the logits at up to λ=0.87. See
# core/brain/nonparametric_worker.py.
#
# Depth is restored to the recurrent arc's intended configuration. If it is
# ever suspected again, AURA_RECURRENT_LOOPS_32B=1 turns it off for the
# interactive lane WITHOUT touching training (which sets AURA_RECURRENT_LOOPS
# explicitly) or the RLC (which uses its own RecurrenceConfig) —
# tests/test_recurrent_depth_lane_separation.py pins that separation.


def _get_model_profile_defaults(num_layers: int) -> tuple:
    """Get default recurrent depth config based on model size."""
    for (min_l, max_l), config in MODEL_PROFILE_DEFAULTS.items():
        if min_l <= num_layers <= max_l:
            return config
    return (1, 0.20, 0.20, 0.1)  # fallback: standard


def _get_lane_defaults(num_layers: int) -> tuple:
    """Backward-compatible alias for external recurrent-depth validators."""
    return _get_model_profile_defaults(num_layers)


def _model_size_loop_env(num_layers: int) -> tuple[str, str] | None:
    if num_layers >= 72:
        name = "AURA_RECURRENT_LOOPS_72B"
        value = os.environ.get(name)
        return (name, value) if value is not None else None
    if num_layers >= 56:
        name = "AURA_RECURRENT_LOOPS_32B"
        value = os.environ.get(name)
        return (name, value) if value is not None else None
    if num_layers >= 24:
        name = "AURA_RECURRENT_LOOPS_14B"
        value = os.environ.get(name)
        return (name, value) if value is not None else None
    name = "AURA_RECURRENT_LOOPS_SMALL"
    value = os.environ.get(name)
    return (name, value) if value is not None else None


def _parse_loop_count(raw: str, *, env_name: str) -> int:
    try:
        loops = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be an integer, got {raw!r}") from exc
    if loops < 0:
        raise RuntimeError(f"{env_name} must be >= 0, got {loops}")
    return loops


def _max_recurrent_loops() -> int:
    raw = os.environ.get("AURA_RECURRENT_MAX_LOOPS")
    if raw is None:
        return _DEFAULT_MAX_RECURRENT_LOOPS
    loops = _parse_loop_count(raw, env_name="AURA_RECURRENT_MAX_LOOPS")
    if loops < 1:
        raise RuntimeError("AURA_RECURRENT_MAX_LOOPS must be >= 1")
    if loops > _ABSOLUTE_MAX_RECURRENT_LOOPS:
        raise RuntimeError(
            "AURA_RECURRENT_MAX_LOOPS must be <= "
            f"{_ABSOLUTE_MAX_RECURRENT_LOOPS}, got {loops}"
        )
    return loops


def _validate_configured_loop_count(loops: int, *, source: str) -> int:
    maximum = _max_recurrent_loops()
    if loops > maximum:
        raise RuntimeError(
            f"{source} requests {loops} recurrent loops, above safe maximum "
            f"{maximum}. Increase AURA_RECURRENT_MAX_LOOPS only for supervised "
            "diagnostics."
        )
    return loops


def _coerce_runtime_loop_count(value: object, *, default: int, maximum: int) -> int:
    try:
        loops = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        loops = default
    return max(1, min(loops, max(1, maximum)))


def _parse_fraction(raw: str, *, env_name: str, minimum: float, maximum: float) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be numeric, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{env_name} must be between {minimum:g} and {maximum:g}, got {value:g}")
    return value


class CacheSnapshotError(RuntimeError):
    """Raised when the KV cache does not support snapshot/restore.

    Silent failure here is worse than useless: the recurrent block would
    accumulate N copies of K/V into the cache, corrupting attention for
    future tokens. Fail loud so the operator sees the bug.
    """


_CACHE_COORDINATE_ATTRS = (
    "offset",
    "_idx",
    "left_padding",
    "_right_padding",
    "start_position",
)


def _owned_cache_coordinate(value):
    """Copy small mutable cache coordinates without cloning K/V tensors."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, tuple):
        return tuple(_owned_cache_coordinate(item) for item in value)
    if isinstance(value, list):
        return [_owned_cache_coordinate(item) for item in value]
    if isinstance(value, dict):
        return {
            _owned_cache_coordinate(key): _owned_cache_coordinate(item)
            for key, item in value.items()
        }
    try:
        import mlx.core as mx

        if isinstance(value, mx.array):
            return mx.array(value)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    copier = getattr(value, "copy", None)
    if callable(copier):
        try:
            return copier()
        except (RuntimeError, TypeError, ValueError):
            pass
    raise CacheSnapshotError(
        f"Cache coordinate {type(value).__name__} has no owned-copy contract"
    )


def _snapshot_cache_coordinates(cache_entry) -> dict[str, object]:
    return {
        attr: _owned_cache_coordinate(getattr(cache_entry, attr))
        for attr in _CACHE_COORDINATE_ATTRS
        if hasattr(cache_entry, attr)
    }


def _restore_cache_coordinates(cache_entry, coordinates: dict[str, object]) -> None:
    for attr, value in coordinates.items():
        setattr(cache_entry, attr, _owned_cache_coordinate(value))


def _cache_coordinate_value(value):
    """Return a value-stable description of small cache coordinates.

    K/V tensors are intentionally compared by storage identity. Cursor and
    padding metadata are different: MLX exposes some of them as tiny arrays,
    and snapshotting those arrays creates new objects with the same logical
    value. Normalizing coordinates recursively keeps commitments stable
    without reading or copying the potentially multi-gigabyte K/V buffers.
    """

    if hasattr(value, "tolist"):
        return _cache_coordinate_value(value.tolist())
    if isinstance(value, tuple):
        return tuple(_cache_coordinate_value(item) for item in value)
    if isinstance(value, list):
        return [_cache_coordinate_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _cache_coordinate_value(item)
            for key, item in value.items()
        }
    return value


def _snapshot_value_matches(current, expected) -> bool:
    """Compare immutable cache storage by identity and metadata by value."""

    if current is expected:
        return True
    if isinstance(expected, tuple):
        return (
            isinstance(current, tuple)
            and len(current) == len(expected)
            and all(
                _snapshot_value_matches(left, right)
                for left, right in zip(current, expected, strict=True)
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(current, list)
            and len(current) == len(expected)
            and all(
                _snapshot_value_matches(left, right)
                for left, right in zip(current, expected, strict=True)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(current, dict)
            and set(current) == set(expected)
            and all(
                _snapshot_value_matches(current[key], expected[key])
                for key in expected
            )
        )
    return isinstance(expected, (str, int, float, bool, type(None))) and (
        type(current) is type(expected) and current == expected
    )


def _cache_snapshot_commitment_parts(snapshot) -> tuple[str, object, object]:
    """Decode every supported cache snapshot into commitment components.

    This is the shared producer/consumer contract for recurrent depth and the
    latent KV lineage. Keeping it here prevents a new snapshot representation
    from silently disabling the cortex in another consumer.
    """

    if not isinstance(snapshot, tuple) or not snapshot:
        raise CacheSnapshotError("Cache snapshot entry is malformed")
    kind = snapshot[0]
    if kind == "buffers" and len(snapshot) == 5:
        return (
            kind,
            (snapshot[1], snapshot[2]),
            {
                "meta_state": snapshot[3],
                "coordinates": _cache_coordinate_value(snapshot[4]),
            },
        )
    if kind == "state" and len(snapshot) in {3, 4}:
        coordinates = snapshot[3] if len(snapshot) == 4 else {}
        return (
            kind,
            snapshot[1],
            {
                "meta_state": snapshot[2],
                "coordinates": _cache_coordinate_value(coordinates),
            },
        )
    if kind == "attrs" and len(snapshot) == 4:
        coordinates = (
            snapshot[3]
            if isinstance(snapshot[3], dict)
            else {"offset": snapshot[3]}
        )
        return (
            kind,
            (snapshot[1], snapshot[2]),
            {"coordinates": _cache_coordinate_value(coordinates)},
        )
    raise CacheSnapshotError(f"Unknown cache snapshot kind: {kind!r}")


def _cache_entry_matches_snapshot(cache_entry, snapshot) -> bool:
    """Prove that one live cache entry is at an exact retained boundary."""

    if cache_entry is None or snapshot is None:
        return cache_entry is None and snapshot is None
    try:
        kind, _state_value, metadata = _cache_snapshot_commitment_parts(snapshot)
    except CacheSnapshotError:
        return False
    coordinates = metadata.get("coordinates", {})
    if kind == "buffers":
        if cache_entry.keys is not snapshot[1] or cache_entry.values is not snapshot[2]:
            return False
        if snapshot[3] is not None and hasattr(cache_entry, "meta_state"):
            if not _snapshot_value_matches(cache_entry.meta_state, snapshot[3]):
                return False
    elif kind == "state":
        if not _snapshot_value_matches(cache_entry.state, snapshot[1]):
            return False
        if not _snapshot_value_matches(cache_entry.meta_state, snapshot[2]):
            return False
    elif kind == "attrs":
        if cache_entry.keys is not snapshot[1] or cache_entry.values is not snapshot[2]:
            return False
    else:  # pragma: no cover - decoded by the exhaustive contract above.
        return False
    return all(
        _cache_coordinate_value(getattr(cache_entry, attr, None)) == expected
        for attr, expected in coordinates.items()
    )


def _verify_cache_coordinates(cache_entry, coordinates: dict[str, object]) -> None:
    mismatches = []
    for attr, expected in coordinates.items():
        observed = getattr(cache_entry, attr, None)
        if _cache_coordinate_value(observed) != _cache_coordinate_value(expected):
            mismatches.append(attr)
    if mismatches:
        raise CacheSnapshotError(
            "Cache restore postcondition failed for coordinates: "
            + ", ".join(sorted(mismatches))
        )


def _materialize_recurrent_prefill_boundary(
    hidden_state,
    *,
    input_tokens,
    input_embeddings=None,
) -> bool:
    """Retire a non-final recurrent prefill graph before the next pass.

    MLX is lazy. Without an evaluation boundary, a two-loop 32B prefill builds
    one graph containing the prelude, both recurrent passes, and the coda. The
    resident 32B live lane measured roughly 13 GB of transient growth before
    its first token and crossed the host emergency fuse. Materializing the
    refined hidden state after the discarded-cache pass preserves the exact
    numerical state consumed by the final pass while allowing MLX to release
    the first pass's temporary graph.

    Decode stays asynchronous: a one-token call would pay this synchronization
    on every generated token, while its graph is already bounded to one token.
    """

    source = input_embeddings if input_embeddings is not None else input_tokens
    shape = getattr(source, "shape", ())
    try:
        sequence_tokens = int(shape[-2] if input_embeddings is not None else shape[-1])
    except (IndexError, TypeError, ValueError):
        sequence_tokens = 1
    if sequence_tokens <= 1:
        return False

    import mlx.core as mx

    mx.eval(hidden_state)
    mx.clear_cache()
    return True


def _clear_recurrent_depth_attrs(inner) -> None:
    for attr in (
        "_recurrent_depth_original_call",
        "_recurrent_depth_original_class",
        "_recurrent_depth_config",
        "_recurrent_depth_patch_scope",
        "_recurrent_depth_runtime_loops",
    ):
        if hasattr(inner, attr):
            delattr(inner, attr)


def _restore_existing_patch(inner) -> bool:
    original_class = getattr(inner, "_recurrent_depth_original_class", None)
    if original_class is not None:
        inner.__class__ = original_class
        _clear_recurrent_depth_attrs(inner)
        return True

    original_call = getattr(inner, "_recurrent_depth_original_call", None)
    if original_call is not None:
        inner.__class__.__call__ = original_call
        _clear_recurrent_depth_attrs(inner)
        return True

    return False


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
        # Live MLX caches expose capacity-bearing K/V buffers. Rewinding those
        # buffers in place preserves their spare allocation; assigning the
        # persistence ``state`` crops to the logical cursor and forces a fresh
        # 256-token allocation on every recurrent decode step.
        if all(hasattr(c, attr) for attr in ("keys", "values", "offset")):
            snapshots.append(
                (
                    "buffers",
                    c.keys,
                    c.values,
                    c.meta_state if hasattr(c, "meta_state") else None,
                    _snapshot_cache_coordinates(c),
                )
            )
            continue
        # Composite caches use the canonical persistence contract because
        # they own no direct K/V buffers.
        if hasattr(c, "state") and hasattr(c, "meta_state"):
            snapshots.append(
                ("state", c.state, c.meta_state, _snapshot_cache_coordinates(c))
            )
            continue
        raise CacheSnapshotError(
            f"KV cache at layer {i} ({type(c).__name__}) supports neither "
            "state/meta_state nor keys/values/offset — recurrent depth cannot "
            "run safely on this cache. Set AURA_RECURRENT_LOOPS=0 or upgrade "
            "mlx_lm."
        )
    return snapshots


def isolate_cache_buffers(cache: Any, start: int, end: int) -> None:
    """Give this cache its own K/V storage, so writing it writes nothing else.

    ``_restore_recurrent_caches`` assigns the snapshot's arrays straight onto
    the cache — ``c.keys = snap[1]`` — which is what preserves the spare
    allocation and is exactly right for the rewind it was written for, where
    one cache goes back to its own earlier state.

    Restoring ONE snapshot into TWO caches is a different thing, and it
    aliases them: both lanes end up holding the same buffer objects, so the
    "cache-isolated" lanes in the heterogeneous dual-lane decode wrote into
    each other. Their persisted logits differed by 34.9 and their bridged
    logits were byte-identical, which reported a Jensen-Shannon divergence of
    exactly zero — a measurement that could not vary, presented as a
    measurement.
    """
    import mlx.core as mx

    for index in range(start, end):
        entry = cache[index]
        if entry is None:
            continue
        for attribute in ("keys", "values"):
            value = getattr(entry, attribute, None)
            if value is None:
                continue
            setattr(entry, attribute, mx.array(value))
        mx.eval(
            *[
                getattr(entry, attribute)
                for attribute in ("keys", "values")
                if getattr(entry, attribute, None) is not None
            ]
        )


def _restore_recurrent_caches(cache, start: int, end: int, snapshots: list):
    """Restore cache state for recurrent layers after a non-final loop."""
    for idx, i in enumerate(range(start, end)):
        c = cache[i]
        snap = snapshots[idx]
        if c is None or snap is None:
            continue
        kind = snap[0]
        if kind == "buffers":
            c.keys = snap[1]
            c.values = snap[2]
            if snap[3] is not None and hasattr(c, "meta_state"):
                c.meta_state = snap[3]
            coordinates = snap[4]
            _restore_cache_coordinates(c, coordinates)
            _verify_cache_coordinates(c, coordinates)
        elif kind == "state":
            c.state = snap[1]
            c.meta_state = snap[2]
            coordinates = snap[3] if len(snap) > 3 else {}
            _restore_cache_coordinates(c, coordinates)
            _verify_cache_coordinates(c, coordinates)
        elif kind == "attrs":
            c.keys = snap[1]
            c.values = snap[2]
            if isinstance(snap[3], dict):
                coordinates = snap[3]
            else:
                # Read snapshots from the original three-attribute format.
                coordinates = {"offset": snap[3]}
            _restore_cache_coordinates(c, coordinates)
            _verify_cache_coordinates(c, coordinates)
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
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('recurrent_depth', exc)
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
        model: A supported MLX model or wrapper exposing transformer layers.
        n_loops: Number of times to run the recurrent block (1 = standard)
        prelude_frac: Fraction of layers for prelude (default 20%)
        coda_frac: Fraction of layers for coda (default 20%)
        residual_alpha: Embedding residual injection strength per loop

    Returns:
        True if patch was applied, False if model structure not recognized.
    """
    n_loops = _validate_configured_loop_count(int(n_loops), source="n_loops")

    try:
        import mlx.core  # noqa: F401
    except ImportError:
        logger.warning("mlx not available — recurrent depth not applied")
        return False

    # ── Locate the transformer owner through the shared topology contract ──
    layer_view = resolve_model_layers(model)
    if layer_view is None:
        logger.warning("Unsupported model layer topology — cannot apply recurrent depth")
        return False
    inner = layer_view.owner
    layers = layer_view.layers

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
    _restore_existing_patch(inner)

    # ── Save original ────────────────────────────────────────────────
    original_class = inner.__class__

    # ── Build the patched forward pass ───────────────────────────────
    # Closure captures: prelude_end, coda_start, num_layers, n_loops,
    #                   residual_alpha
    def recurrent_forward(self, inputs, cache=None, input_embeddings=None):
        """Mythos-inspired recurrent-depth forward pass.

        Prelude → [Recurrent × N with cache save/restore] → Coda
        """
        runtime_loops = getattr(self, "_recurrent_depth_runtime_loops", n_loops)
        effective_loops = _coerce_runtime_loop_count(
            runtime_loops,
            default=n_loops,
            maximum=n_loops,
        )
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
        for loop_idx in range(effective_loops):
            is_final_loop = loop_idx == effective_loops - 1

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
                _materialize_recurrent_prefill_boundary(
                    h,
                    input_tokens=inputs,
                    input_embeddings=input_embeddings,
                )

        # ── CODA: layers [coda_start..end) — run once ────────────
        for i in range(coda_start, num_layers):
            h = self.layers[i](h, mask, cache[i])

        return self.norm(h)

    # ── Apply the patch ──────────────────────────────────────────────
    # Special methods such as __call__ are resolved on the class, not on the
    # instance. Mutating inner.__class__.__call__ globally leaks recurrent-depth
    # into every model instance of that class. Use a one-off dynamic subclass so
    # this model lane is patched without contaminating the other MLX lanes.
    patched_class = type(
        f"{original_class.__name__}RecurrentDepth_{id(inner):x}",
        (original_class,),
        {
            "__call__": recurrent_forward,
            "__module__": original_class.__module__,
            "__doc__": original_class.__doc__,
        },
    )
    try:
        inner.__class__ = patched_class
    except TypeError as exc:
        logger.error(
            "🚫 Recurrent depth DISABLED: cannot install instance-scoped patch "
            "on %s: %s",
            original_class.__name__,
            exc,
        )
        return False
    inner._recurrent_depth_original_class = original_class
    inner._recurrent_depth_patch_scope = "instance_subclass"

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
        "patch_scope": "instance_subclass",
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
    layer_view = resolve_model_layers(model)
    if layer_view is None:
        return False
    inner = layer_view.owner

    if not (
        hasattr(inner, "_recurrent_depth_original_class")
        or hasattr(inner, "_recurrent_depth_original_call")
    ):
        logger.debug("No recurrent depth patch found — nothing to remove")
        return False

    _restore_existing_patch(inner)

    logger.info("Recurrent depth removed — standard forward pass restored")
    return True


def get_recurrent_config(model) -> dict | None:
    """Get the current recurrent depth configuration, or None if not patched."""
    layer_view = resolve_model_layers(model)
    if layer_view is None:
        return None
    inner = layer_view.owner
    return getattr(inner, "_recurrent_depth_config", None)


def resolve_loops_for_model(model) -> int:
    """Determine the correct number of loops for a model based on its size.

    Checks env vars first, then falls back to model-profile defaults based on
    the number of transformer layers.
    """
    # Explicit env override takes priority
    env_loops = os.environ.get("AURA_RECURRENT_LOOPS")
    if env_loops is not None:
        n = _parse_loop_count(env_loops, env_name="AURA_RECURRENT_LOOPS")
        if n == 0:
            # Explicitly disabled
            return 0
        return _validate_configured_loop_count(n, source="AURA_RECURRENT_LOOPS")

    # Auto-detect based on model size
    layer_view = resolve_model_layers(model)
    if layer_view is None:
        return 1
    num_layers = len(layer_view.layers)
    size_override = _model_size_loop_env(num_layers)
    if size_override is not None:
        env_name, raw_loops = size_override
        n = _parse_loop_count(raw_loops, env_name=env_name)
        if n == 0:
            return 0
        return _validate_configured_loop_count(n, source=env_name)
    defaults = _get_model_profile_defaults(num_layers)
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
        layer_view = resolve_model_layers(model)
        num_layers = len(layer_view.layers) if layer_view is not None else 0
        logger.info(
            "Recurrent depth: standard pass for %d-layer model (n_loops=%d)",
            num_layers, n_loops,
        )
        return False

    # Get other params from env or defaults
    layer_view = resolve_model_layers(model)
    num_layers = len(layer_view.layers) if layer_view is not None else 64
    defaults = _get_model_profile_defaults(num_layers)

    prelude_frac = _parse_fraction(
        os.environ.get("AURA_RECURRENT_PRELUDE", str(defaults[1])),
        env_name="AURA_RECURRENT_PRELUDE",
        minimum=0.05,
        maximum=0.45,
    )
    coda_frac = _parse_fraction(
        os.environ.get("AURA_RECURRENT_CODA", str(defaults[2])),
        env_name="AURA_RECURRENT_CODA",
        minimum=0.05,
        maximum=0.45,
    )
    alpha = _parse_fraction(
        os.environ.get("AURA_RECURRENT_ALPHA", str(defaults[3])),
        env_name="AURA_RECURRENT_ALPHA",
        minimum=0.0,
        maximum=0.5,
    )

    return apply_recurrent_depth(
        model,
        n_loops=n_loops,
        prelude_frac=prelude_frac,
        coda_frac=coda_frac,
        residual_alpha=alpha,
    )

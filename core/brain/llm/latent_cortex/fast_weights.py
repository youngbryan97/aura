"""Episode-scoped fast weights: the checkpoint writes temporary synapses.

During one reasoning episode the effective weights may become

    W_t = W₀ + s·U Vᵀ        (rank-r, per selected window-layer linear)

with three hard guarantees the spec demands and this module PROVES:

1. **Identity at attach.** V is zero-initialized, so the instant a wrapper
   attaches, the model's function is bit-for-bit unchanged (same guarantee
   the expert-adapter seam relies on: a LoRA with B=0 is behaviorally inert).
2. **Erase is proven, not assumed.** ``detach`` restores the original module
   objects, and ``prove_erase`` re-runs a caller-supplied probe and asserts
   exact output equality with the pre-attach baseline. The receipt carries
   the verdict.
3. **No persistent learning without governance.** A ΔW that earns its keep
   is EXPORTED to the governed consolidation queue for the existing LoRA
   compounding loop (with its regression gates); it never mutates W₀ here.

U is seeded deterministically from the episode's workspace statistics — the
latent state literally parameterizes the temporary synapses — and both U and
V are then optimized against the episode's proxy/verifier loss with all base
weights frozen (grads flow only to U, V).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.plasticity_sites import (
    PLASTICITY_TARGET_PATHS,
    plasticity_target_module,
    select_compatible_plasticity_layers,
)
from core.brain.llm.latent_cortex.types import ComputeBudget, FastWeightsConfig
from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger("Aura.LatentCortex.FastWeights")
FAST_WEIGHT_OPTIMIZER = "rms_normalized_sgd_backtracking_v1"
FAST_WEIGHT_ACTIVATION_POLICIES = frozenset(
    {"none", "recurrence_only", "decode_only", "both"}
)
FAST_WEIGHT_EXECUTION_PHASES = ("recurrence", "decode", "unscoped")

_MODEL_LEASE_LOCK = checked_lock(
    "latent_cortex.fast_weights.model_lease",
    rank=LockRank.REGISTRY,
)
_MODEL_LEASES: dict[int, tuple[Any, str]] = {}

# Targets whose adaptation provably cannot change any cached K/V value.
#
# It is empty, and that is the finding rather than an omission. The prompt
# cache is filled under base weights and then reused by adapted probes and the
# final decode. Both supported targets write into the residual stream, which
# every LATER layer's k_proj and v_proj consume, so cached entries for those
# layers describe a function that is no longer the one running. A target could
# only join this set with a proof, and neither has one.
CACHE_INDEPENDENT_TARGETS: frozenset[str] = frozenset()


def target_cache_attestation(target: str) -> dict[str, Any]:
    """What is true about reusing a base-weight cache under this target.

    The alternative remedy is a full re-prefill after every attach, rescale
    and erase. That is a different cost profile and needs its own paired
    evidence; until it exists, the mixture is stated rather than implied.
    """

    independent = target in CACHE_INDEPENDENT_TARGETS
    return {
        "schema": "aura.fast_weights.cache_attestation.v1",
        "target": str(target),
        "cache_independent": independent,
        "prefix_kv_under_base_weights": not independent,
        "reason": (
            "target proven not to affect any cached key/value"
            if independent
            else "target writes into the residual stream consumed by later "
            "layers' k/v projections, so cached prefix entries were computed "
            "under different weights"
        ),
    }


def _linear_dimension_source(module):
    """Find the projection that owns shape metadata without bypassing wrappers.

    MLX LoRA modules intentionally expose their frozen base projection as
    ``.linear`` and do not duplicate ``.weight``.  Fast weights must inspect
    that nested projection for dimensions while continuing to invoke the
    outer module, otherwise attaching an episodic delta would either crash or
    silently bypass the durable adapter.
    """
    current = module
    seen: set[int] = set()
    # Wrapper chains are shallow (LoRA-over-quantized is depth 2); 32 is a
    # hard bound against pathological nesting, backed by cycle detection.
    for _depth in range(32):
        identity = id(current)
        if identity in seen:
            raise TypeError("linear wrapper cycle while resolving dimensions")
        seen.add(identity)
        if hasattr(current, "weight"):
            return current
        nested = getattr(current, "linear", None)
        if nested is None:
            raise TypeError(
                f"{type(module).__name__} has no weight-bearing linear projection"
            )
        current = nested
    raise TypeError("linear wrapper nesting exceeds supported depth")


def _linear_dims(module) -> tuple[int, int]:
    """Return ``(out_features, in_features)`` for wrapped or bare linears."""
    source = _linear_dimension_source(module)
    weight = source.weight
    if getattr(weight, "ndim", None) != 2:
        raise TypeError("linear projection weight must be two-dimensional")
    if hasattr(source, "scales"):  # QuantizedLinear packs weights
        out_features = weight.shape[0]
        bits = int(getattr(source, "bits", 4))
        if bits <= 0 or 32 % bits:
            raise TypeError(f"unsupported quantized linear bit width: {bits}")
        in_features = weight.shape[1] * (32 // bits)
        return int(out_features), int(in_features)
    return int(weight.shape[0]), int(weight.shape[1])


class EpisodicDeltaLinear:
    """y = base(x) + s·((x Vᵀ) Uᵀ) — a temporary synapse over a frozen linear.

    Not an ``nn.Module`` on purpose: keeping U/V as plain attributes outside
    the model's parameter tree means nothing about the model's trainable
    state, freeze bookkeeping, or serialization changes while a wrapper is
    attached. Gradients reach U/V functionally (see ``optimize``).
    """

    def __init__(
        self,
        base,
        rank: int,
        scale: float,
        seed_stat: float,
        tag: str,
        seed_vectors=None,
        seed_source: str = "retrieval",
    ) -> None:
        import mlx.core as mx

        self.base = base
        self.scale = float(scale)
        self.tag = tag
        # Quantized projections can change dtype/evaluation semantics merely
        # by adding an algebraic zero. Attachment therefore remains a literal
        # base-function pass-through until the engine has measured identity
        # and explicitly activates the adaptation path.
        self.identity_bypass = True
        # ``both`` preserves the historical execution graph.  A frozen delta
        # can later be lesioned by causal phase without changing U, V, scale,
        # the base projection, or any model/cache state.
        self.activation_policy = "both"
        self.phase_calls = {phase: 0 for phase in FAST_WEIGHT_EXECUTION_PHASES}
        self.phase_delta_calls = {
            phase: 0 for phase in FAST_WEIGHT_EXECUTION_PHASES
        }
        self.capture_input = False
        self.last_input_summary = None
        self.input_summary_history = []
        self.last_input_features = None
        self.query_gate_keys = None
        self.query_gate_active = False
        self.query_gate_threshold = 0.0
        self.query_gate_temperature = 1.0
        self.capture_input_positions = False
        self.input_position_limit = 0
        self.input_position_phase: str | None = None
        self.input_position_history = []
        self.capture_output = False
        self.capture_output_start = 0
        self.last_input_trajectory_features = None
        self.last_output_features = None
        out_features, in_features = _linear_dims(base)
        seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")
        key = mx.random.key(seed)
        # U uses dimension-normalized LoRA-style scale, modulated by the
        # workspace statistic; V remains exactly zero, so attachment is still
        # bit-identical while the first V gradient stays above fp16/bf16 noise.
        seed_scale = min(1.0, max(0.1, abs(float(seed_stat))))
        init_std = seed_scale / math.sqrt(max(1, out_features))
        self.U = mx.random.normal((out_features, rank), key=key) * init_std
        self.V = mx.zeros((rank, in_features))
        # Retrieval-to-fast-weight compilation: refined retrieval slot
        # states become the leading columns of U, so the adaptation
        # SUBSPACE is spanned by retrieved knowledge — the temporary
        # synapses can write those directions into the residual stream once
        # V learns when to fire. V stays zero, so attach remains exactly
        # identity and the erase proof is untouched.
        self.retrieval_seeded_columns = 0
        self.semantic_seeded_columns = 0
        if (
            seed_vectors is not None
            and getattr(seed_vectors, "ndim", 0) == 2
            and int(seed_vectors.shape[1]) == out_features
        ):
            self.reseed_output_subspace(seed_vectors, seed_source=seed_source)
        mx.eval(self.U, self.V)

    def reseed_output_subspace(self, seed_vectors, *, seed_source: str) -> int:
        """Replace leading U columns while V=0 preserves exact identity."""

        import mlx.core as mx

        if seed_source not in {"retrieval", "verified_semantic_contrast"}:
            raise ValueError("fast-weight seed source is unsupported")
        if (
            getattr(seed_vectors, "ndim", 0) != 2
            or int(seed_vectors.shape[1]) != int(self.U.shape[0])
        ):
            raise ValueError("fast-weight seed vectors differ from output width")
        mx.eval(self.V)
        if bool(mx.any(self.V != 0)):
            raise RuntimeError("fast-weight output subspace can only change while V is zero")
        k = min(int(self.U.shape[1]), int(seed_vectors.shape[0]))
        if k <= 0:
            raise ValueError("fast-weight seed vectors are empty")
        target_norm = math.sqrt(
            float(mx.mean(mx.square(self.U[:, :k]))) * int(self.U.shape[0])
        )
        target_norm = max(target_norm, 1e-6)
        columns = []
        for index in range(k):
            vector = seed_vectors[index].astype(self.U.dtype)
            norm = mx.maximum(mx.linalg.norm(vector), 1e-6)
            columns.append(vector / norm * target_norm)
        seeded = mx.stack(columns, axis=1)
        self.U = mx.concatenate([seeded, self.U[:, k:]], axis=1)
        self.retrieval_seeded_columns = k if seed_source == "retrieval" else 0
        self.semantic_seeded_columns = (
            k if seed_source == "verified_semantic_contrast" else 0
        )
        mx.eval(self.U)
        return k

    def __call__(self, x):
        execution_phase: str | None = None
        if self.capture_input_positions or not self.identity_bypass:
            from core.brain.llm.latent_cortex.recurrence_adapter import (
                current_coda_adapter_scope,
                current_recurrence_adapter_scope,
            )

            if current_recurrence_adapter_scope() is not None:
                execution_phase = "recurrence"
            elif current_coda_adapter_scope() is not None:
                execution_phase = "decode"
            else:
                execution_phase = "unscoped"
        if self.capture_input:
            import mlx.core as mx

            axes = tuple(range(max(0, int(x.ndim) - 1)))
            summary = mx.mean(x, axis=axes) if axes else x
            self.last_input_summary = mx.stop_gradient(summary.astype(mx.float32))
            if len(self.input_summary_history) < int(self.U.shape[1]):
                self.input_summary_history.append(self.last_input_summary)
        if self.capture_input_positions and (
            self.input_position_phase is None
            or self.input_position_phase == execution_phase
        ):
            import mlx.core as mx

            from core.brain.llm.latent_cortex.recurrence_adapter import (
                current_recurrence_adapter_scope,
            )

            scoped = x
            activation = current_recurrence_adapter_scope()
            if activation is not None and activation.start is not None:
                start = int(activation.start)
                stop = int(activation.stop)
                if not 0 <= start < stop <= int(x.shape[-2]):
                    raise ValueError("position capture recurrence scope is invalid")
                scoped = x[..., start:stop, :]
            positions = scoped.reshape((-1, int(scoped.shape[-1])))
            remaining = max(
                0,
                int(self.input_position_limit) - len(self.input_position_history),
            )
            for index in range(min(remaining, int(positions.shape[0]))):
                self.input_position_history.append(
                    mx.stop_gradient(positions[index].astype(mx.float32))
                )
        if self.capture_output:
            import mlx.core as mx

            sequence = x.reshape((-1, int(x.shape[-1])))
            start = min(max(0, int(self.capture_output_start)), int(sequence.shape[0]) - 1)
            suffix = sequence[start:]
            candidates = [mx.mean(suffix, axis=0), suffix[-1]]
            if int(self.U.shape[1]) > 2:
                count = int(suffix.shape[0])
                for index in range(int(self.U.shape[1]) - 2):
                    position = min(
                        count - 1,
                        ((index + 1) * count) // (int(self.U.shape[1]) - 1),
                    )
                    candidates.append(suffix[position])
            self.last_input_trajectory_features = mx.stop_gradient(
                mx.stack(candidates[: int(self.U.shape[1])], axis=0).astype(mx.float32)
            )
        base = self.base(x)
        if self.capture_output:
            import mlx.core as mx

            sequence = base.reshape((-1, int(base.shape[-1])))
            start = min(max(0, int(self.capture_output_start)), int(sequence.shape[0]) - 1)
            suffix = sequence[start:]
            candidates = [mx.mean(suffix, axis=0), suffix[-1]]
            if int(self.U.shape[1]) > 2:
                count = int(suffix.shape[0])
                for index in range(int(self.U.shape[1]) - 2):
                    position = min(
                        count - 1,
                        ((index + 1) * count) // (int(self.U.shape[1]) - 1),
                    )
                    candidates.append(suffix[position])
            self.last_output_features = mx.stop_gradient(
                mx.stack(candidates[: int(self.U.shape[1])], axis=0).astype(mx.float32)
            )
        if self.identity_bypass:
            return base
        phase = execution_phase or "unscoped"
        self.phase_calls[phase] += 1
        policy = self.activation_policy
        active = (
            policy == "both"
            or (policy == "recurrence_only" and phase == "recurrence")
            or (policy == "decode_only" and phase == "decode")
        )
        if not active:
            return base
        self.phase_delta_calls[phase] += 1
        delta = (x @ self.V.T) @ self.U.T
        if self.query_gate_keys is not None and self.query_gate_active:
            import mlx.core as mx

            width = int(x.shape[-1])
            flat = x.reshape((-1, width)).astype(mx.float32)
            norms = mx.maximum(mx.linalg.norm(flat, axis=1, keepdims=True), 1e-6)
            similarities = (flat / norms) @ self.query_gate_keys.T
            maximum = mx.max(similarities, axis=1, keepdims=True)
            gate = mx.sigmoid(
                (maximum - float(self.query_gate_threshold))
                / float(self.query_gate_temperature)
            )
            gate = gate.reshape((*x.shape[:-1], 1)).astype(delta.dtype)
            delta = delta * gate
        return base + self.scale * delta

    def install_query_gate(self, keys, *, threshold: float, temperature: float) -> None:
        """Bind the episodic delta to normalized private query activations."""

        import mlx.core as mx

        if (
            getattr(keys, "ndim", 0) != 2
            or int(keys.shape[0]) <= 0
            or int(keys.shape[1]) != int(self.V.shape[1])
        ):
            raise ValueError("fast-weight query-gate key shape differs")
        if not bool(mx.all(mx.isfinite(keys))):
            raise ValueError("fast-weight query-gate keys are non-finite")
        norms = mx.linalg.norm(keys.astype(mx.float32), axis=1, keepdims=True)
        if bool(mx.any(norms <= 1e-8)):
            raise ValueError("fast-weight query-gate keys contain a zero vector")
        if not math.isfinite(float(threshold)) or not -1.0 < float(threshold) < 1.0:
            raise ValueError("fast-weight query-gate threshold must be inside (-1, 1)")
        if (
            not math.isfinite(float(temperature))
            or not 0.0 < float(temperature) <= 1.0
        ):
            raise ValueError("fast-weight query-gate temperature must be inside (0, 1]")
        self.query_gate_keys = mx.stop_gradient(keys.astype(mx.float32) / norms)
        self.query_gate_active = True
        self.query_gate_threshold = float(threshold)
        self.query_gate_temperature = float(temperature)
        mx.eval(self.query_gate_keys)


@dataclass
class FastWeightHandle:
    layer_index: int
    parent: Any
    attr: str
    original: Any
    wrapper: EpisodicDeltaLinear


@dataclass
class FastWeightsLifecycle:
    """Auditable state machine: ATTACHED → (OPTIMIZED) → ERASED, with proof."""

    attached_at: float = 0.0
    layers: list[int] = field(default_factory=list)
    target: str = ""
    rank: int = 0
    optimizer: str = FAST_WEIGHT_OPTIMIZER
    optimization_attempts: int = 0
    optimized_steps: int = 0
    rejected_steps: int = 0
    gradient_evaluations: int = 0
    line_search_evaluations: int = 0
    line_search_backtracks: int = 0
    budget_exhausted: bool = False
    detach_conflicts: int = 0
    canary_rescales: int = 0
    canary_erased: bool = False
    retrieval_seeded_columns: int = 0
    semantic_seeded_columns: int = 0
    loss_trail: list[float] = field(default_factory=list)
    gradient_global_norm_trail: list[float] = field(default_factory=list)
    accepted_step_sizes: list[float] = field(default_factory=list)
    erased: bool = False
    erase_proven: bool | None = None
    exported: bool = False
    lease_owner_sha256: str = ""
    lease_model_sha256: str = ""
    lease_acquired: bool = False
    lease_released: bool = False
    lease_conflicts: int = 0
    erase_probe_before_sha256: str = ""
    erase_probe_after_sha256: str = ""
    # Module identities, live wrappers and parameter digests over every layer
    # this episode touched. The probe can only say one input did not notice a
    # residual; this says whether one is there.
    structural_erase: dict[str, Any] = field(default_factory=dict)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "attached_at": self.attached_at,
            "structural_erase": dict(self.structural_erase),
            "layers": list(self.layers),
            "target": self.target,
            "rank": self.rank,
            "optimizer": self.optimizer,
            "optimization_attempts": self.optimization_attempts,
            "optimized_steps": self.optimized_steps,
            "rejected_steps": self.rejected_steps,
            "gradient_evaluations": self.gradient_evaluations,
            "line_search_evaluations": self.line_search_evaluations,
            "line_search_backtracks": self.line_search_backtracks,
            "budget_exhausted": self.budget_exhausted,
            "detach_conflicts": self.detach_conflicts,
            "canary_rescales": self.canary_rescales,
            "canary_erased": self.canary_erased,
            "retrieval_seeded_columns": self.retrieval_seeded_columns,
            "semantic_seeded_columns": self.semantic_seeded_columns,
            "loss_trail": [round(v, 6) for v in self.loss_trail],
            "gradient_global_norm_trail": [
                round(v, 6) for v in self.gradient_global_norm_trail
            ],
            "accepted_step_sizes": [
                round(v, 12) for v in self.accepted_step_sizes
            ],
            "erased": self.erased,
            "erase_proven": self.erase_proven,
            "exported": self.exported,
            "lease_owner_sha256": self.lease_owner_sha256,
            "lease_model_sha256": self.lease_model_sha256,
            "lease_acquired": self.lease_acquired,
            "lease_released": self.lease_released,
            "lease_conflicts": self.lease_conflicts,
            "erase_probe_before_sha256": self.erase_probe_before_sha256,
            "erase_probe_after_sha256": self.erase_probe_after_sha256,
        }


class EpisodicFastWeights:
    """Owns the full lifecycle of one episode's temporary synapses."""

    def __init__(self, config: FastWeightsConfig) -> None:
        self.config = config
        self.handles: list[FastWeightHandle] = []
        self.lifecycle = FastWeightsLifecycle()
        self.last_export_receipt: dict[str, Any] | None = None
        self.last_export_error = ""
        self._lease_model: Any | None = None
        self._lease_owner = ""
        # Anything caching model OUTPUTS (probe memoization) registers here:
        # attach/rescale/detach change the model function, so every such
        # transition must flush downstream caches or they become lies.
        self.on_function_change: Callable[[str], None] | None = None

    def _notify_function_change(self, reason: str) -> None:
        callback = self.on_function_change
        if callback is None:
            return
        try:
            callback(reason)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            logger.warning("Fast-weight function-change listener failed (%s)", reason)

    # ── Attach / detach ─────────────────────────────────────────────────
    def _acquire_model_lease(self, inner_model: Any, episode_id: str) -> None:
        """Acquire the process-local exclusive mutation lease for one model.

        The MLX request lane normally serializes resident generations, but
        fast-weight safety cannot depend on a caller remembering that outer
        convention. This registry owns the actual mutable model object and
        refuses wrapper composition across independent episode managers.
        """

        owner = f"{episode_id}:{uuid.uuid4().hex}"
        model_key = id(inner_model)
        with _MODEL_LEASE_LOCK:
            current = _MODEL_LEASES.get(model_key)
            if current is not None:
                current_model, _current_owner = current
                if current_model is inner_model:
                    self.lifecycle.lease_conflicts += 1
                    raise RuntimeError("fast-weight model lease already held by another episode")
                # Defensive id-reuse cleanup. A live strong reference makes
                # this branch practically unreachable, but the identity check
                # keeps the registry correct by construction.
                _MODEL_LEASES.pop(model_key, None)
            _MODEL_LEASES[model_key] = (inner_model, owner)
        self._lease_model = inner_model
        self._lease_owner = owner
        self.lifecycle.lease_owner_sha256 = hashlib.sha256(owner.encode("utf-8")).hexdigest()
        model_identity = (
            f"{type(inner_model).__module__}.{type(inner_model).__qualname__}:"
            f"{model_key}"
        )
        self.lifecycle.lease_model_sha256 = hashlib.sha256(
            model_identity.encode("utf-8")
        ).hexdigest()
        self.lifecycle.lease_acquired = True
        self.lifecycle.lease_released = False

    def _release_model_lease(self) -> None:
        model = self._lease_model
        owner = self._lease_owner
        if model is None or not owner:
            return
        released = False
        with _MODEL_LEASE_LOCK:
            current = _MODEL_LEASES.get(id(model))
            if current is not None and current[0] is model and current[1] == owner:
                _MODEL_LEASES.pop(id(model), None)
                released = True
            else:
                self.lifecycle.lease_conflicts += 1
        self.lifecycle.lease_released = released
        self._lease_model = None
        self._lease_owner = ""

    def lease_receipt(self) -> dict[str, Any]:
        return {
            "schema": "aura.rlc.fast_weight_model_lease.v1",
            "owner_sha256": self.lifecycle.lease_owner_sha256,
            "model_sha256": self.lifecycle.lease_model_sha256,
            "acquired": self.lifecycle.lease_acquired,
            "released": self.lifecycle.lease_released,
            "conflicts": self.lifecycle.lease_conflicts,
        }

    def attach(
        self,
        inner_model,
        layer_range: tuple[int, int],
        *,
        seed_stat: float,
        episode_id: str,
        seed_vectors=None,
        seed_source: str = "retrieval",
    ) -> int:
        """Wrap the target linear in up to ``max_wrapped_layers`` window layers."""
        if self.handles:
            raise RuntimeError("fast weights already attached — one episode at a time")
        # EPISODE BOUNDARY: clear the previous episode's proof state.
        #
        # detach() empties `handles`, which is the only thing this guard
        # checks, so the same object could be re-attached for a NEW episode
        # while still carrying the last one's erase_proven, exported flag and
        # exported handle snapshots. A stale erase_proven=True would then
        # vouch for weights it never saw — the proof would survive the thing
        # it was proving. Nothing about the prior episode may outlive it.
        self.lifecycle = FastWeightsLifecycle()
        self._exported_handles = []
        target_attrs = PLASTICITY_TARGET_PATHS.get(self.config.target)
        if target_attrs is None:
            raise ValueError("unsupported fast-weight projection target")
        parent_attr, leaf_attr = target_attrs
        start, end = layer_range
        candidates = select_compatible_plasticity_layers(
            inner_model.layers,
            start,
            end,
            max(1, self.config.max_wrapped_layers),
            target=self.config.target,
            placement=self.config.layer_placement,
        )
        self._acquire_model_lease(inner_model, episode_id)
        attached = False
        try:
            for i in candidates:
                layer = inner_model.layers[i]
                parent = getattr(layer, parent_attr)
                original = plasticity_target_module(layer, self.config.target)
                if original is None:
                    raise RuntimeError(
                        "qualified fast-weight layer lost its target projection"
                    )
                wrapper = EpisodicDeltaLinear(
                    original,
                    rank=self.config.rank,
                    scale=self.config.scale,
                    seed_stat=seed_stat,
                    tag=f"{episode_id}:{i}:{self.config.target}",
                    seed_vectors=seed_vectors,
                    seed_source=seed_source,
                )
                setattr(parent, leaf_attr, wrapper)
                self.handles.append(
                    FastWeightHandle(
                        layer_index=i,
                        parent=parent,
                        attr=leaf_attr,
                        original=original,
                        wrapper=wrapper,
                    )
                )
            attached = True
        finally:
            if not attached:
                # Attachment is a transaction. A malformed layer in the
                # middle must not leave earlier layers wrapped, including when
                # control exits through a non-Exception base error.
                self.detach()
        if not self.handles:
            self.detach()
            raise RuntimeError("fast-weight attachment selected no model layers")
        self._attached_sites = [
            (h.parent, h.attr, h.original, h.layer_index) for h in self.handles
        ]
        self._attached_parameter_sha256 = self._touched_parameter_sha256()
        self.lifecycle.attached_at = time.time()
        self.lifecycle.layers = [h.layer_index for h in self.handles]
        self.lifecycle.target = self.config.target
        self.lifecycle.rank = self.config.rank
        self.lifecycle.retrieval_seeded_columns = max(
            (h.wrapper.retrieval_seeded_columns for h in self.handles),
            default=0,
        )
        self.lifecycle.semantic_seeded_columns = max(
            (h.wrapper.semantic_seeded_columns for h in self.handles),
            default=0,
        )
        # V=0 makes attach bit-identical, but retrieval-seeded columns can
        # start non-zero — notify conservatively either way.
        self._notify_function_change("fast_weights_attached")
        return len(self.handles)

    def detach(self) -> int:
        """Restore every original module object. Idempotent and conflict-aware."""
        restored = 0
        conflicts = 0
        remaining: list[FastWeightHandle] = []
        for handle in reversed(self.handles):
            current = getattr(handle.parent, handle.attr)
            if current is handle.original:
                continue
            if current is not handle.wrapper:
                # Another writer touched a module owned by this episode. We
                # still restore W0, but the conflict invalidates the proof.
                conflicts += 1
            try:
                setattr(handle.parent, handle.attr, handle.original)
                restored += 1
            except (AttributeError, RuntimeError, TypeError):
                remaining.append(handle)
        self.handles = list(reversed(remaining))
        self.lifecycle.detach_conflicts += conflicts
        self.lifecycle.erased = not self.handles
        if not self.handles:
            self._release_model_lease()
        if restored:
            self._notify_function_change("fast_weights_detached")
        return restored

    def activate_adaptation_path(self) -> None:
        """Release the exact-identity bypass after attachment is proven."""

        if not self.handles:
            raise RuntimeError(
                "fast-weight adaptation activation requires attached wrappers"
            )
        changed = False
        for handle in self.handles:
            if getattr(handle.wrapper, "identity_bypass", False):
                handle.wrapper.identity_bypass = False
                changed = True
        if changed:
            self._notify_function_change("fast_weights_adaptation_activated")

    def set_activation_policy(self, policy: str) -> None:
        """Select where one frozen episodic delta may affect computation."""

        if policy not in FAST_WEIGHT_ACTIVATION_POLICIES:
            raise ValueError("fast-weight activation policy is unsupported")
        if not self.handles:
            raise RuntimeError("fast-weight activation policy requires wrappers")
        changed = False
        for handle in self.handles:
            if handle.wrapper.activation_policy != policy:
                handle.wrapper.activation_policy = policy
                changed = True
        if changed:
            self._notify_function_change(
                f"fast_weights_activation_policy:{policy}"
            )

    def set_query_gate_active(self, active: bool) -> None:
        """Suspend query scoping only while fitting a private teacher loss."""

        if type(active) is not bool:
            raise TypeError("query gate active state must be boolean")
        if not self.handles or any(
            handle.wrapper.query_gate_keys is None for handle in self.handles
        ):
            raise RuntimeError("fast-weight query gate is not installed")
        changed = False
        for handle in self.handles:
            if handle.wrapper.query_gate_active is not active:
                handle.wrapper.query_gate_active = active
                changed = True
        if changed:
            self._notify_function_change(
                f"fast_weights_query_gate_active:{active}"
            )

    def activation_locality_receipt(self) -> dict[str, Any]:
        """Publish phase applications without exposing temporary tensors."""

        if not self.handles:
            raise RuntimeError("fast-weight locality receipt requires wrappers")
        policies = {handle.wrapper.activation_policy for handle in self.handles}
        if len(policies) != 1:
            raise RuntimeError("fast-weight activation policies disagree")
        observed = {phase: 0 for phase in FAST_WEIGHT_EXECUTION_PHASES}
        applied = {phase: 0 for phase in FAST_WEIGHT_EXECUTION_PHASES}
        for handle in self.handles:
            for phase in FAST_WEIGHT_EXECUTION_PHASES:
                observed[phase] += int(handle.wrapper.phase_calls[phase])
                applied[phase] += int(handle.wrapper.phase_delta_calls[phase])
        return {
            "schema": "aura.rlc.fast_weight_activation_locality.v1",
            "policy": next(iter(policies)),
            "wrapped_layers": len(self.handles),
            "observed_calls": observed,
            "delta_calls": applied,
        }

    def delta_commitment(self) -> str:
        """Commit the exact frozen U,V inventory used by locality arms."""

        from core.brain.llm.latent_cortex.verified_best import tensor_sha256

        if not self.handles:
            raise RuntimeError("fast-weight delta commitment requires wrappers")
        rows = [
            {
                "layer": int(handle.layer_index),
                "scale": float(handle.wrapper.scale),
                "u_sha256": tensor_sha256(handle.wrapper.U),
                "v_sha256": tensor_sha256(handle.wrapper.V),
            }
            for handle in self.handles
        ]
        return hashlib.sha256(
            json.dumps(
                rows,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()

    def capture_input_summaries(self, forward_fn: Callable[[], Any]) -> Any:
        """Capture one query activation per wrapped projection under identity."""

        import mlx.core as mx

        if not self.handles:
            raise RuntimeError("fast-weight activation capture requires wrappers")
        if any(bool(mx.any(handle.wrapper.V != 0)) for handle in self.handles):
            raise RuntimeError("fast-weight activation capture requires zero V")
        for handle in self.handles:
            wrapper = handle.wrapper
            wrapper.last_input_summary = None
            wrapper.input_summary_history = []
            wrapper.last_input_features = None
            wrapper.capture_input = True
        try:
            output = forward_fn()
            summaries = [handle.wrapper.last_input_summary for handle in self.handles]
            if any(summary is None for summary in summaries):
                raise RuntimeError("fast-weight activation capture missed a projection")
            for handle in self.handles:
                wrapper = handle.wrapper
                history = list(wrapper.input_summary_history)
                while len(history) < int(wrapper.U.shape[1]):
                    history.append(history[-1])
                wrapper.last_input_features = mx.stop_gradient(
                    mx.stack(history[: int(wrapper.U.shape[1])], axis=0).astype(
                        mx.float32
                    )
                )
            features = [handle.wrapper.last_input_features for handle in self.handles]
            mx.eval(output, *summaries, *features)
            return output
        finally:
            for handle in self.handles:
                handle.wrapper.capture_input = False

    def input_feature_commitments(self) -> dict[int, str]:
        """Commit captured query keys without exposing private activations."""

        from core.brain.llm.latent_cortex.verified_best import tensor_sha256

        commitments = {}
        for handle in self.handles:
            features = handle.wrapper.last_input_features
            if features is None:
                raise RuntimeError("fast-weight query-key capture is absent")
            commitments[int(handle.layer_index)] = tensor_sha256(features)
        return commitments

    def captured_input_features(self) -> dict[int, Any]:
        """Return private, detached copies of the captured live-query keys."""

        import mlx.core as mx

        captured = {}
        for handle in self.handles:
            features = handle.wrapper.last_input_features
            if features is None:
                raise RuntimeError("fast-weight query-key capture is absent")
            copied = mx.stop_gradient(mx.array(features).astype(mx.float32))
            mx.eval(copied)
            captured[int(handle.layer_index)] = copied
        if not captured:
            raise RuntimeError("fast-weight query-key capture requires wrappers")
        return captured

    def install_captured_query_gates(
        self,
        *,
        threshold: float,
        temperature: float,
    ) -> dict[str, Any]:
        """Make captured query activations the private scope of every delta."""

        from core.brain.llm.latent_cortex.verified_best import tensor_sha256

        if not self.handles:
            raise RuntimeError("fast-weight query gate requires wrappers")
        rows = []
        for handle in self.handles:
            keys = handle.wrapper.last_input_features
            if keys is None:
                raise RuntimeError("fast-weight query gate lacks captured activations")
            handle.wrapper.install_query_gate(
                keys,
                threshold=threshold,
                temperature=temperature,
            )
            rows.append(
                {
                    "layer": int(handle.layer_index),
                    "key_count": int(keys.shape[0]),
                    "keys_sha256": tensor_sha256(keys),
                }
            )
        self._notify_function_change("fast_weights_query_gate_installed")
        return {
            "schema": "aura.fast_weight_query_gate.v1",
            "threshold": float(threshold),
            "temperature": float(temperature),
            "layers": rows,
        }

    def capture_input_position_features(
        self,
        forward_fn: Callable[[], Any],
        *,
        max_features: int,
        phase: str | None = None,
    ) -> tuple[Any, dict[int, Any]]:
        """Capture distinct recurrent positions without averaging slot roles.

        The original summary capture remains the query-scoped minimum-norm
        key used by CP144. Persistent transfer needs a richer training object:
        averaging objective, constraint, hypothesis, and counterexample slots
        destroys the distinctions a shared operator must learn to act on.
        """

        import mlx.core as mx

        if type(max_features) is not int or not 1 <= max_features <= 256:
            raise ValueError("position feature limit must be inside [1, 256]")
        if phase not in {None, *FAST_WEIGHT_EXECUTION_PHASES}:
            raise ValueError("position feature phase is unsupported")
        if not self.handles:
            raise RuntimeError("position feature capture requires wrappers")
        if any(bool(mx.any(handle.wrapper.V != 0)) for handle in self.handles):
            raise RuntimeError("position feature capture requires zero V")
        for handle in self.handles:
            wrapper = handle.wrapper
            wrapper.input_position_history = []
            wrapper.input_position_limit = max_features
            wrapper.input_position_phase = phase
            wrapper.capture_input_positions = True
        try:
            output = forward_fn()
            features: dict[int, Any] = {}
            for handle in self.handles:
                history = handle.wrapper.input_position_history
                if len(history) != max_features:
                    raise RuntimeError(
                        "position feature capture did not reach its declared inventory"
                    )
                features[int(handle.layer_index)] = mx.stack(history, axis=0)
            mx.eval(output, *features.values())
            return output, features
        finally:
            for handle in self.handles:
                wrapper = handle.wrapper
                wrapper.capture_input_positions = False
                wrapper.input_position_limit = 0
                wrapper.input_position_phase = None

    def capture_output_features(
        self,
        forward_fn: Callable[[], Any],
        *,
        token_start: int,
    ) -> dict[int, Any]:
        """Capture rank-bounded projection features from an answer suffix.

        The wrappers remain exact identity (`V=0`) while the frozen checkpoint
        processes a teacher-forced sequence. Returned tensors are private,
        detached layer-local representations; no teacher text enters decode.
        """

        import mlx.core as mx

        if type(token_start) is not int or token_start < 0:
            raise ValueError("fast-weight output capture token start is invalid")
        if not self.handles:
            raise RuntimeError("fast-weight output capture requires wrappers")
        if any(bool(mx.any(handle.wrapper.V != 0)) for handle in self.handles):
            raise RuntimeError("fast-weight output capture requires zero V")
        for handle in self.handles:
            wrapper = handle.wrapper
            wrapper.last_output_features = None
            wrapper.capture_output_start = token_start
            wrapper.capture_output = True
        try:
            output = forward_fn()
            features = {
                int(handle.layer_index): handle.wrapper.last_output_features
                for handle in self.handles
            }
            if any(value is None for value in features.values()):
                raise RuntimeError("fast-weight output capture missed a projection")
            mx.eval(output, *features.values())
            return features
        finally:
            for handle in self.handles:
                handle.wrapper.capture_output = False

    def capture_io_features(
        self,
        forward_fn: Callable[[], Any],
        *,
        token_start: int,
    ) -> tuple[dict[int, Any], dict[int, Any]]:
        """Capture aligned input and output features from an answer suffix."""

        import mlx.core as mx

        if type(token_start) is not int or token_start < 0:
            raise ValueError("fast-weight I/O capture token start is invalid")
        if not self.handles:
            raise RuntimeError("fast-weight I/O capture requires wrappers")
        if any(bool(mx.any(handle.wrapper.V != 0)) for handle in self.handles):
            raise RuntimeError("fast-weight I/O capture requires zero V")
        for handle in self.handles:
            wrapper = handle.wrapper
            wrapper.last_input_trajectory_features = None
            wrapper.last_output_features = None
            wrapper.capture_output_start = token_start
            wrapper.capture_output = True
        try:
            output = forward_fn()
            inputs = {
                int(handle.layer_index): handle.wrapper.last_input_trajectory_features
                for handle in self.handles
            }
            outputs = {
                int(handle.layer_index): handle.wrapper.last_output_features
                for handle in self.handles
            }
            if any(value is None for value in (*inputs.values(), *outputs.values())):
                raise RuntimeError("fast-weight I/O capture missed a projection")
            for layer in inputs:
                if int(inputs[layer].shape[0]) != int(outputs[layer].shape[0]):
                    raise RuntimeError("fast-weight I/O feature counts differ")
            mx.eval(output, *inputs.values(), *outputs.values())
            return inputs, outputs
        finally:
            for handle in self.handles:
                handle.wrapper.capture_output = False

    def reseed_output_subspace_by_layer(
        self,
        seed_vectors_by_layer: Mapping[int, Any],
        *,
        seed_source: str,
    ) -> int:
        """Install a distinct checkpoint-native correction basis per layer."""

        if not self.handles:
            raise RuntimeError("fast-weight layerwise reseed requires wrappers")
        expected = {int(handle.layer_index) for handle in self.handles}
        observed = {int(index) for index in seed_vectors_by_layer}
        if observed != expected:
            raise ValueError("fast-weight layerwise seed inventory differs")
        counts = [
            handle.wrapper.reseed_output_subspace(
                seed_vectors_by_layer[int(handle.layer_index)],
                seed_source=seed_source,
            )
            for handle in self.handles
        ]
        if len(set(counts)) != 1:
            raise RuntimeError("fast-weight layerwise seed rank differs")
        self.lifecycle.retrieval_seeded_columns = max(
            (handle.wrapper.retrieval_seeded_columns for handle in self.handles),
            default=0,
        )
        self.lifecycle.semantic_seeded_columns = max(
            (handle.wrapper.semantic_seeded_columns for handle in self.handles),
            default=0,
        )
        self._notify_function_change("fast_weights_layerwise_output_subspace_reseeded")
        return counts[0]

    def install_minimum_norm_keys(
        self,
        *,
        gain: float,
        regularization: float,
    ) -> dict[str, Any]:
        """Compile captured query activations into regularized rank-r keys.

        For each seeded semantic column ``u_j`` and captured input ``x``, the
        key ``v_j = g*x/(||x||^2 + lambda)`` is the minimum-norm linear write
        that makes the temporary projection emit ``g*u_j`` on that query.
        """

        import mlx.core as mx

        if (
            isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or not math.isfinite(float(gain))
            or not 0.0 < float(gain) <= 4.0
        ):
            raise ValueError("minimum-norm key gain must be inside (0, 4]")
        if (
            isinstance(regularization, bool)
            or not isinstance(regularization, (int, float))
            or not math.isfinite(float(regularization))
            or not 0.0 < float(regularization) <= 1.0
        ):
            raise ValueError("minimum-norm key regularization must be inside (0, 1]")
        if not self.handles:
            raise RuntimeError("minimum-norm key write requires wrappers")
        rows = []
        for handle in self.handles:
            wrapper = handle.wrapper
            x = wrapper.last_input_summary
            features = wrapper.last_input_features
            if features is None and x is not None:
                features = mx.stack([x for _ in range(int(wrapper.U.shape[1]))])
            if (
                x is None
                or getattr(x, "ndim", 0) != 1
                or getattr(features, "ndim", 0) != 2
            ):
                raise RuntimeError("minimum-norm key write lacks captured activation")
            if int(features.shape[1]) != int(wrapper.V.shape[1]):
                raise RuntimeError("minimum-norm key activation width differs")
            seeded = max(
                int(wrapper.semantic_seeded_columns),
                int(wrapper.retrieval_seeded_columns),
            )
            if seeded <= 0:
                raise RuntimeError("minimum-norm key write lacks seeded output directions")
            per_direction_gain = float(gain) / math.sqrt(seeded)
            active_rows = []
            denominators = []
            input_norms = []
            for index in range(seeded):
                feature = features[index]
                denominator = mx.sum(mx.square(feature)) + float(regularization)
                active_rows.append(
                    (per_direction_gain * feature / denominator).astype(wrapper.V.dtype)
                )
                denominators.append(denominator)
                input_norms.append(mx.linalg.norm(feature))
            active = mx.stack(active_rows, axis=0)
            wrapper.V = mx.concatenate([active, mx.zeros_like(wrapper.V[seeded:])], axis=0)
            mx.eval(wrapper.V, *denominators, *input_norms)
            rows.append(
                {
                    "layer": int(handle.layer_index),
                    "seeded_columns": seeded,
                    "input_norms": [round(float(value), 8) for value in input_norms],
                    "denominators": [round(float(value), 8) for value in denominators],
                }
            )
        self._notify_function_change("fast_weights_minimum_norm_keys_installed")
        return {
            "schema": "aura.fast_weight_minimum_norm_write.v1",
            "gain": float(gain),
            "regularization": float(regularization),
            "layers": rows,
        }

    def install_supervised_trajectory_map(
        self,
        input_features_by_layer: Mapping[int, Any],
        output_corrections_by_layer: Mapping[int, Any],
        *,
        gain: float,
        regularization: float,
        normalize_corrections: bool = True,
        key_source: str = "live_query",
    ) -> dict[str, Any]:
        """Install the minimum-norm map from live-query states to corrections.

        For rows ``X`` and verified corrections ``Y``, the rank-bounded dual
        ridge map is ``X.T @ solve(X @ X.T + lambda I, Y)``.  The wrapper's
        orientation is ``x @ V.T @ U.T``; setting ``V=X`` and ``U=dual.T``
        therefore realizes the same map without constructing a width-square
        matrix.  Each row remains a distinct autoregressive position instead
        of being averaged into one query key.
        """

        import mlx.core as mx

        from core.brain.llm.latent_cortex.verified_best import tensor_sha256

        if (
            isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or not math.isfinite(float(gain))
            or not 0.0 < float(gain) <= 16.0
        ):
            raise ValueError("supervised trajectory gain must be inside (0, 16]")
        if (
            isinstance(regularization, bool)
            or not isinstance(regularization, (int, float))
            or not math.isfinite(float(regularization))
            or not 0.0 < float(regularization) <= 1.0
        ):
            raise ValueError(
                "supervised trajectory regularization must be inside (0, 1]"
            )
        if type(normalize_corrections) is not bool:
            raise TypeError("normalize_corrections must be boolean")
        if key_source not in {"live_query", "incumbent_trajectory"}:
            raise ValueError("supervised trajectory key source is unsupported")
        expected = {int(handle.layer_index) for handle in self.handles}
        if (
            not expected
            or {int(index) for index in input_features_by_layer} != expected
            or {int(index) for index in output_corrections_by_layer} != expected
        ):
            raise ValueError("supervised trajectory layer inventory differs")

        rows = []
        for handle in self.handles:
            layer = int(handle.layer_index)
            wrapper = handle.wrapper
            inputs = input_features_by_layer[layer].astype(mx.float32)
            corrections = output_corrections_by_layer[layer].astype(mx.float32)
            captured_query = wrapper.last_input_features
            if key_source == "live_query" and (
                captured_query is None
                or tensor_sha256(inputs) != tensor_sha256(captured_query)
            ):
                raise ValueError(
                    "supervised trajectory keys differ from captured query activations"
                )
            if (
                getattr(inputs, "ndim", 0) != 2
                or getattr(corrections, "ndim", 0) != 2
                or int(inputs.shape[0]) != int(corrections.shape[0])
                or int(inputs.shape[1]) != int(wrapper.V.shape[1])
                or int(corrections.shape[1]) != int(wrapper.U.shape[0])
                or not 1 <= int(inputs.shape[0]) <= int(wrapper.U.shape[1])
            ):
                raise ValueError("supervised trajectory feature shape differs")
            if not bool(mx.all(mx.isfinite(inputs))) or not bool(
                mx.all(mx.isfinite(corrections))
            ):
                raise ValueError("supervised trajectory features are non-finite")

            input_norms = mx.linalg.norm(inputs, axis=1, keepdims=True)
            if bool(mx.any(input_norms <= 1e-8)):
                raise ValueError("supervised trajectory contains a zero input")
            keys = inputs
            correction_norms = mx.linalg.norm(corrections, axis=1, keepdims=True)
            if bool(mx.any(correction_norms <= 1e-8)):
                raise ValueError("supervised trajectory contains a zero correction")
            targets = (
                corrections / correction_norms
                if normalize_corrections
                else corrections
            )
            gram = keys @ keys.T
            ridge_scale = mx.maximum(
                mx.mean(mx.diag(gram)),
                1e-8,
            )
            identity = mx.eye(int(keys.shape[0]), dtype=mx.float32)
            # MLX does not implement linalg.solve on the GPU.  This system is
            # at most rank-by-rank (normally 2-16), so the CPU stream is both
            # the supported path and negligible beside the model forwards.
            dual = mx.linalg.solve(
                gram + float(regularization) * ridge_scale * identity,
                targets,
                stream=mx.cpu,
            )
            active = int(keys.shape[0])
            scale = max(abs(float(wrapper.scale)), 1e-12)
            new_u = (float(gain) / scale * dual.T).astype(wrapper.U.dtype)
            new_v = keys.astype(wrapper.V.dtype)
            wrapper.U = mx.concatenate(
                [new_u, mx.zeros_like(wrapper.U[:, active:])],
                axis=1,
            )
            wrapper.V = mx.concatenate(
                [new_v, mx.zeros_like(wrapper.V[active:])],
                axis=0,
            )
            predicted = float(wrapper.scale) * (keys @ wrapper.V.T) @ wrapper.U.T
            target = float(gain) * targets
            relative_error = mx.linalg.norm(predicted - target) / mx.maximum(
                mx.linalg.norm(target), 1e-8
            )
            mx.eval(
                wrapper.U,
                wrapper.V,
                relative_error,
                ridge_scale,
                input_norms,
                correction_norms,
            )
            rows.append(
                {
                    "layer": layer,
                    "teaching_pairs": active,
                    "input_width": int(inputs.shape[1]),
                    "output_width": int(corrections.shape[1]),
                    "inputs_sha256": tensor_sha256(inputs),
                    "corrections_sha256": tensor_sha256(corrections),
                    "u_sha256": tensor_sha256(wrapper.U),
                    "v_sha256": tensor_sha256(wrapper.V),
                    "training_relative_error": round(float(relative_error), 8),
                    "ridge_scale": round(float(ridge_scale), 8),
                    "input_norm_min": round(float(mx.min(input_norms)), 8),
                    "input_norm_max": round(float(mx.max(input_norms)), 8),
                    "correction_norm_min": round(float(mx.min(correction_norms)), 8),
                    "correction_norm_max": round(float(mx.max(correction_norms)), 8),
                }
            )
        self._notify_function_change("fast_weights_supervised_trajectory_map_installed")
        return {
            "schema": "aura.fast_weight_supervised_trajectory_map.v2",
            "key_source": (
                "captured_query_activation"
                if key_source == "live_query"
                else "verified_incumbent_trajectory"
            ),
            "gain": float(gain),
            "regularization": float(regularization),
            "corrections_normalized": normalize_corrections,
            "layers": rows,
        }

    def reseed_output_subspace(self, seed_vectors, *, seed_source: str) -> int:
        """Install an equal-rank matched-arm subspace before V learns a key."""

        if not self.handles:
            raise RuntimeError("fast-weight reseed requires attached wrappers")
        counts = [
            handle.wrapper.reseed_output_subspace(
                seed_vectors,
                seed_source=seed_source,
            )
            for handle in self.handles
        ]
        if len(set(counts)) != 1:
            raise RuntimeError("fast-weight reseed inventory differs across layers")
        self.lifecycle.retrieval_seeded_columns = max(
            (handle.wrapper.retrieval_seeded_columns for handle in self.handles),
            default=0,
        )
        self.lifecycle.semantic_seeded_columns = max(
            (handle.wrapper.semantic_seeded_columns for handle in self.handles),
            default=0,
        )
        self._notify_function_change("fast_weights_output_subspace_reseeded")
        return counts[0]

    def rescale(self, factor: float) -> float:
        """Multiply every wrapper's scale — the canary ladder's step-down.

        Scale lives outside U/V, so a rescale needs no re-optimization and no
        new forward pass; the next canary measurement decides whether the
        weaker ΔW is now behaviorally safe.
        """
        if (
            isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or not math.isfinite(float(factor))
            or not 0.0 < float(factor) < 1.0
        ):
            raise ValueError("fast-weight rescale factor must be inside (0, 1)")
        if not self.handles:
            raise RuntimeError("fast-weight rescale requires attached wrappers")
        for handle in self.handles:
            handle.wrapper.scale *= float(factor)
        self.lifecycle.canary_rescales += 1
        self._notify_function_change("fast_weights_rescaled")
        return float(self.handles[0].wrapper.scale)

    def effective_delta_metrics(self) -> dict[str, Any]:
        """Measure the exact effective-delta RMS without materializing U@V.T.

        For D = s*U@V.T, ||D||_F^2 is
        s^2*trace((U.T@U)*(V@V.T)). Both Gram matrices are rank-by-rank, so
        this structural safety check stays cheap even on the resident 32B.
        """

        import mlx.core as mx

        rows: list[dict[str, Any]] = []
        all_finite = bool(self.handles)
        max_rms = 0.0
        for handle in self.handles:
            wrapper = handle.wrapper
            gram_u = wrapper.U.T @ wrapper.U
            gram_v = wrapper.V @ wrapper.V.T
            mx.eval(gram_u, gram_v)
            squared_frobenius = float(
                (float(wrapper.scale) ** 2) * mx.sum(gram_u * gram_v.T)
            )
            finite = math.isfinite(squared_frobenius) and squared_frobenius >= 0.0
            all_finite = all_finite and finite
            if finite:
                out_features = int(wrapper.U.shape[0])
                in_features = int(wrapper.V.shape[1])
                frobenius = math.sqrt(squared_frobenius)
                rms = frobenius / math.sqrt(max(1, out_features * in_features))
                max_rms = max(max_rms, rms)
            else:
                frobenius = math.inf
                rms = math.inf
            rows.append(
                {
                    "layer": int(handle.layer_index),
                    "scale": round(float(wrapper.scale), 12),
                    "effective_delta_frobenius": (
                        round(frobenius, 12) if math.isfinite(frobenius) else None
                    ),
                    "effective_delta_rms": (
                        round(rms, 12) if math.isfinite(rms) else None
                    ),
                    "finite": finite,
                    "query_conditioned": wrapper.query_gate_keys is not None,
                }
            )
        return {
            "schema": "aura.fast_weight_delta_magnitude.v1",
            "finite": all_finite,
            "layer_count": len(rows),
            "max_effective_delta_rms": (
                round(max_rms, 12) if all_finite else None
            ),
            "layers": rows,
        }

    def has_effective_delta(self, *, epsilon: float = 1e-12) -> bool:
        """Return whether the attached function is observably non-identity.

        Direct supervised writes can materialize a useful delta before the
        gradient optimizer accepts a step. Lifecycle gates must inspect the
        function that exists, not infer identity from the optimizer counter.
        """

        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or float(epsilon) < 0.0
        ):
            raise ValueError("fast-weight effective-delta epsilon is invalid")
        metrics = self.effective_delta_metrics()
        maximum = metrics.get("max_effective_delta_rms")
        return bool(
            metrics.get("finite") is True
            and not isinstance(maximum, bool)
            and isinstance(maximum, (int, float))
            and math.isfinite(float(maximum))
            and float(maximum) > float(epsilon)
        )

    def canary_erase(self) -> None:
        """Erase ΔW because the protected battery regressed under it.

        The episode continues on base weights with its refined latent state
        intact; the lifecycle records that the canaries — not cleanup —
        removed the adaptation, and consolidation export is off the table
        because the post-detach snapshot is deliberately never taken.
        """
        self.detach()
        self.lifecycle.canary_erased = True

    def snapshot_delta(self) -> tuple[dict[str, Any], ...]:
        """Copy the complete temporary parameter state for a matched arm."""

        import mlx.core as mx

        if not self.handles:
            raise RuntimeError(
                "fast-weight snapshot requires attached wrappers"
            )
        snapshots: list[dict[str, Any]] = []
        for handle in self.handles:
            wrapper = handle.wrapper
            u = mx.array(wrapper.U)
            v = mx.array(wrapper.V)
            mx.eval(u, v)
            snapshots.append(
                {
                    "layer": int(handle.layer_index),
                    "scale": float(wrapper.scale),
                    "U": u,
                    "V": v,
                }
            )
        return tuple(snapshots)

    def restore_delta(
        self,
        snapshots: Sequence[Mapping[str, Any]],
        *,
        reason: str,
    ) -> None:
        """Restore one arm's temporary parameters without touching W0."""

        import mlx.core as mx

        rows = list(snapshots)
        if not self.handles or len(rows) != len(self.handles):
            raise RuntimeError("fast-weight restore inventory differs")
        rebound = []
        for handle, row in zip(self.handles, rows, strict=True):
            if (
                row.get("layer") != handle.layer_index
                or not isinstance(row.get("scale"), (int, float))
                or isinstance(row.get("scale"), bool)
            ):
                raise ValueError("fast-weight restore identity differs")
            u = row.get("U")
            v = row.get("V")
            if (
                getattr(u, "shape", None) != handle.wrapper.U.shape
                or getattr(v, "shape", None) != handle.wrapper.V.shape
            ):
                raise ValueError("fast-weight restore tensor shape differs")
            # Durable candidates cross the NPZ boundary as NumPy arrays.
            # Wrappers execute in MLX, so assigning those arrays directly
            # leaves a mixed-backend projection that fails only on its first
            # real matmul. Normalize at the restore boundary; native MLX
            # snapshots preserve their dtype and serialized NumPy tensors are
            # reconstructed into the same device representation.
            u = mx.array(u)
            v = mx.array(v)
            rebound.extend((u, v))
            handle.wrapper.U = u
            handle.wrapper.V = v
            handle.wrapper.scale = float(row["scale"])
        mx.eval(*rebound)
        self._notify_function_change(reason)

    def interpolate_delta(
        self,
        initial: Sequence[Mapping[str, Any]],
        candidate: Sequence[Mapping[str, Any]],
        *,
        gain: float,
        reason: str,
    ) -> None:
        """Install a signed point on one episodic update trajectory."""

        import mlx.core as mx

        if (
            isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or not math.isfinite(float(gain))
            or not -4.0 <= float(gain) <= 4.0
        ):
            raise ValueError("fast-weight interpolation gain is outside [-4, 4]")
        left = list(initial)
        right = list(candidate)
        if len(left) != len(right) or len(left) != len(self.handles):
            raise RuntimeError("fast-weight interpolation inventory differs")
        interpolated = []
        for before, after in zip(left, right, strict=True):
            if before["layer"] != after["layer"]:
                raise ValueError("fast-weight interpolation layer identity differs")
            interpolated.append(
                {
                    "layer": before["layer"],
                    "scale": float(before["scale"]),
                    "U": before["U"] + float(gain) * (after["U"] - before["U"]),
                    "V": before["V"] + float(gain) * (after["V"] - before["V"]),
                }
            )
        mx.eval(*[row[name] for row in interpolated for name in ("U", "V")])
        self.restore_delta(interpolated, reason=reason)

    def optimization_trace(self) -> dict[str, Any]:
        lifecycle = self.lifecycle
        return {
            "optimizer": lifecycle.optimizer,
            "attempts": lifecycle.optimization_attempts,
            "accepted_steps": lifecycle.optimized_steps,
            "rejected_steps": lifecycle.rejected_steps,
            "gradient_evaluations": lifecycle.gradient_evaluations,
            "line_search_evaluations": lifecycle.line_search_evaluations,
            "loss_trail": list(lifecycle.loss_trail),
            "gradient_norm_trail": list(
                lifecycle.gradient_global_norm_trail
            ),
            "accepted_step_sizes": list(
                lifecycle.accepted_step_sizes
            ),
            "line_search_backtracks": lifecycle.line_search_backtracks,
            "budget_exhausted": lifecycle.budget_exhausted,
        }

    def reset_optimization_trace(self) -> None:
        lifecycle = self.lifecycle
        lifecycle.optimization_attempts = 0
        lifecycle.optimized_steps = 0
        lifecycle.rejected_steps = 0
        lifecycle.gradient_evaluations = 0
        lifecycle.line_search_evaluations = 0
        lifecycle.line_search_backtracks = 0
        lifecycle.budget_exhausted = False
        lifecycle.loss_trail.clear()
        lifecycle.gradient_global_norm_trail.clear()
        lifecycle.accepted_step_sizes.clear()

    def restore_optimization_trace(
        self,
        trace: Mapping[str, Any],
    ) -> None:
        required = {
            "optimizer",
            "attempts",
            "accepted_steps",
            "rejected_steps",
            "gradient_evaluations",
            "line_search_evaluations",
            "loss_trail",
            "gradient_norm_trail",
            "accepted_step_sizes",
            "line_search_backtracks",
            "budget_exhausted",
        }
        if not isinstance(trace, Mapping) or set(trace) != required:
            raise ValueError(
                "fast-weight optimization trace fields differ"
            )
        lifecycle = self.lifecycle
        lifecycle.optimizer = str(trace["optimizer"])
        lifecycle.optimization_attempts = int(trace["attempts"])
        lifecycle.optimized_steps = int(trace["accepted_steps"])
        lifecycle.rejected_steps = int(trace["rejected_steps"])
        lifecycle.gradient_evaluations = int(
            trace["gradient_evaluations"]
        )
        lifecycle.line_search_evaluations = int(
            trace["line_search_evaluations"]
        )
        lifecycle.loss_trail = list(trace["loss_trail"])
        lifecycle.gradient_global_norm_trail = list(
            trace["gradient_norm_trail"]
        )
        lifecycle.accepted_step_sizes = list(
            trace["accepted_step_sizes"]
        )
        lifecycle.line_search_backtracks = int(
            trace["line_search_backtracks"]
        )
        lifecycle.budget_exhausted = bool(trace["budget_exhausted"])

    def _touched_parameter_sha256(self) -> str:
        """One digest over the parameters of every module this episode wrapped.

        Taken at attach over the ORIGINAL modules and again after detach. A
        residual mutation that an eight-token probe happens not to excite is
        invisible behaviourally and obvious here.
        """
        import hashlib

        import numpy as np

        digest = hashlib.sha256(b"aura.fast_weights.touched_parameters.v1")
        for parent, attr, original, layer_index in getattr(self, "_attached_sites", []):
            digest.update(str(layer_index).encode())
            digest.update(str(attr).encode())
            del parent
            for name in ("weight", "bias", "scales", "biases"):
                value = getattr(original, name, None)
                if value is None:
                    continue
                digest.update(name.encode())
                digest.update(np.ascontiguousarray(np.array(value)).tobytes())
        return digest.hexdigest()

    def structural_erase_report(self) -> dict[str, Any]:
        """What is measurably true about the model's structure after detach.

        Byte equality on one fixed probe cannot say that every wrapper, handle
        and parameter reference went away — only that this input did not
        notice. These are the facts that can be checked directly.
        """

        sites = getattr(self, "_attached_sites", [])
        restored = [
            int(layer_index)
            for parent, attr, original, layer_index in sites
            if getattr(parent, attr, None) is original
        ]
        wrappers_live = [
            int(layer_index)
            for parent, attr, original, layer_index in sites
            if isinstance(getattr(parent, attr, None), EpisodicDeltaLinear)
        ]
        parameters_after = self._touched_parameter_sha256() if sites else ""
        return {
            "schema": "aura.fast_weights.structural_erase.v1",
            "touched_layers": [int(row[3]) for row in sites],
            "restored_layers": restored,
            "wrapped_layers_remaining": wrappers_live,
            "handles_remaining": len(self.handles),
            "parameters_before_sha256": getattr(
                self, "_attached_parameter_sha256", ""
            ),
            "parameters_after_sha256": parameters_after,
            "structurally_restored": bool(
                sites
                and not self.handles
                and not wrappers_live
                and len(restored) == len(sites)
                and parameters_after == getattr(self, "_attached_parameter_sha256", "")
            ),
        }

    def prove_erase(self, probe_fn: Callable[[], Any], baseline) -> bool:
        """Prove restoration structurally AND behaviourally.

        The behavioural half — byte equality on a fixed eight-token probe —
        stayed the whole proof for a long time. It cannot see a residual delta
        that this particular input does not excite, a wrapper still installed
        on a layer the probe does not reach, or a handle nobody released. The
        structural half checks those directly and the probe remains as
        corroboration.
        """
        from core.brain.llm.latent_cortex.verified_best import tensor_sha256

        if self.handles or not self.lifecycle.erased:
            raise RuntimeError("prove_erase called while fast weights still attached")
        after = probe_fn()
        structural = self.structural_erase_report()
        self.lifecycle.structural_erase = structural
        self.lifecycle.erase_probe_before_sha256 = tensor_sha256(baseline)
        self.lifecycle.erase_probe_after_sha256 = tensor_sha256(after)
        proven = (
            self.lifecycle.detach_conflicts == 0
            and structural["structurally_restored"] is True
            and self.lifecycle.erase_probe_before_sha256
            == self.lifecycle.erase_probe_after_sha256
        )
        self.lifecycle.erase_proven = proven
        if not proven:
            from core.runtime.errors import record_degradation

            record_degradation(
                "latent_cortex",
                RuntimeError("fast-weight erase failed probe equality"),
                action="flagged episode receipt and refused consolidation export",
            )
        return proven

    # ── Optimization (grads to U/V only; base frozen by construction) ──
    def optimize(
        self,
        loss_fn: Callable[[], Any],
        *,
        steps: int | None = None,
        budget: ComputeBudget | None = None,
        layer_apps_per_forward: int = 0,
        tokens_per_forward: int = 0,
        layers_per_forward: int = 0,
        reserve_layer_apps: int = 0,
        fixed_line_search_evaluations: int | None = None,
        operation_prefix: str = "fast_weight",
    ) -> None:
        """Functional gradient steps on every wrapper's (U, V).

        ``loss_fn`` closes over the model (with wrappers attached) and
        returns a scalar. We lift the wrapper params into an explicit list,
        rebind them inside the traced function, and step along a per-tensor
        RMS-preconditioned descent direction with bounded backtracking. This
        keeps a resident-scale update numerically visible without allowing the
        number of adapter elements to inflate its RMS magnitude. A candidate is
        retained only when it improves the proxy beyond floating-point noise;
        base weights never appear as grad targets.
        """
        import mlx.core as mx

        if not self.handles:
            return
        n_steps = steps if steps is not None else self.config.opt_steps
        if type(n_steps) is not int or n_steps < 0:
            raise ValueError("fast-weight optimization steps must be a non-negative integer")
        if (
            isinstance(layer_apps_per_forward, bool)
            or not isinstance(layer_apps_per_forward, int)
            or layer_apps_per_forward < 0
        ):
            raise ValueError("layer_apps_per_forward must be a non-negative integer")
        if (
            isinstance(reserve_layer_apps, bool)
            or not isinstance(reserve_layer_apps, int)
            or reserve_layer_apps < 0
        ):
            raise ValueError("reserve_layer_apps must be a non-negative integer")
        for name, value in (
            ("tokens_per_forward", tokens_per_forward),
            ("layers_per_forward", layers_per_forward),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        exact_forward_shape = (
            tokens_per_forward > 0
            and layers_per_forward > 0
            and tokens_per_forward * layers_per_forward
            == layer_apps_per_forward
        )
        if budget is not None and not exact_forward_shape:
            budget.resource_ledger.mark_unknown("fast_weight_training_shape")
        if budget is not None and layer_apps_per_forward <= 0:
            raise ValueError(
                "budgeted fast-weight optimization requires a positive forward cost"
            )
        if (
            fixed_line_search_evaluations is not None
            and (
                type(fixed_line_search_evaluations) is not int
                or not 1 <= fixed_line_search_evaluations <= 12
            )
        ):
            raise ValueError(
                "fixed line-search evaluations must be an integer inside [1, 12]"
            )
        if (
            not isinstance(operation_prefix, str)
            or not operation_prefix
            or not operation_prefix.replace("_", "").isalnum()
        ):
            raise ValueError("fast-weight operation prefix is invalid")

        # Direct users of EpisodicFastWeights receive the same boundary as
        # the engine: attach is exact identity; the first real optimization
        # explicitly activates the delta path. The engine also calls this
        # before potentially monkeypatched optimizers, so both interfaces are
        # safe and the operation is idempotent.
        if n_steps > 0:
            self.activate_adaptation_path()

        def bind_params(params) -> None:
            parameter_pairs = zip(params[0::2], params[1::2], strict=True)
            for h, (u, v) in zip(self.handles, parameter_pairs, strict=True):
                h.wrapper.U = u
                h.wrapper.V = v

        def with_params(params):
            bind_params(params)
            return loss_fn()

        params = []
        for h in self.handles:
            params.extend([h.wrapper.U, h.wrapper.V])

        grad_fn = mx.value_and_grad(with_params)
        for _ in range(n_steps):
            gradient_cost = layer_apps_per_forward * 3
            if budget is not None and (
                budget.exhausted
                or gradient_cost + reserve_layer_apps > budget.remaining_layer_apps
            ):
                self.lifecycle.budget_exhausted = True
                break
            if budget is not None:
                if exact_forward_shape:
                    budget.charge_training_work(
                        f"{operation_prefix}_gradient",
                        tokens=tokens_per_forward,
                        layers=layers_per_forward,
                        attention_pairs_per_forward=(
                            tokens_per_forward
                            * tokens_per_forward
                            * layers_per_forward
                        ),
                        forward_evaluations=1,
                        backward_evaluations=1,
                    )
                else:
                    budget.charge_layer_apps(
                        gradient_cost,
                        operation=f"{operation_prefix}_gradient",
                    )
            self.lifecycle.optimization_attempts += 1
            self.lifecycle.gradient_evaluations += 1
            value, grads = grad_fn(params)
            current_loss = float(value)
            if not self.lifecycle.loss_trail:
                self.lifecycle.loss_trail.append(current_loss)
            flat = mx.concatenate([mx.reshape(g, (-1,)) for g in grads])
            gnorm = mx.maximum(mx.linalg.norm(flat), 1e-12)
            gnorm_value = float(gnorm)
            if not math.isfinite(current_loss) or not math.isfinite(gnorm_value):
                self.lifecycle.rejected_steps += 1
                break
            self.lifecycle.gradient_global_norm_trail.append(gnorm_value)
            directions = []
            for grad in grads:
                grad_rms = mx.maximum(mx.sqrt(mx.mean(mx.square(grad))), 1e-12)
                directions.append(mx.clip(grad / grad_rms, -8.0, 8.0))
            step_size = float(self.config.lr)
            accepted = False
            best_candidate = None
            best_candidate_loss = math.inf
            best_step_size = 0.0
            best_backtrack = 0
            line_search_limit = fixed_line_search_evaluations or 12
            for backtrack in range(line_search_limit):
                candidate_cost = layer_apps_per_forward
                if budget is not None and (
                    budget.exhausted
                    or candidate_cost + reserve_layer_apps > budget.remaining_layer_apps
                ):
                    self.lifecycle.budget_exhausted = True
                    break
                if budget is not None:
                    if exact_forward_shape:
                        budget.charge_training_work(
                            f"{operation_prefix}_line_search",
                            tokens=tokens_per_forward,
                            layers=layers_per_forward,
                            attention_pairs_per_forward=(
                                tokens_per_forward
                                * tokens_per_forward
                                * layers_per_forward
                            ),
                            forward_evaluations=1,
                            backward_evaluations=0,
                        )
                    else:
                        budget.charge_layer_apps(
                            candidate_cost,
                            operation=f"{operation_prefix}_line_search",
                        )
                candidate = [
                    parameter - step_size * direction
                    for parameter, direction in zip(params, directions, strict=True)
                ]
                self.lifecycle.line_search_evaluations += 1
                try:
                    candidate_value = with_params(candidate)
                    mx.eval(candidate_value, *candidate)
                except BaseException:  # noqa: BLE001 - always restore bound params on interruption
                    bind_params(params)
                    raise
                candidate_loss = float(candidate_value)
                # Some model objectives exhibit small repeat-evaluation drift
                # even with identical parameters.  Acceptance must improve
                # both the value observed for this step and the last state we
                # committed to the signed trace; otherwise a locally lower
                # candidate can still make the authoritative trail regress.
                committed_loss = (
                    self.lifecycle.loss_trail[-1]
                    if self.lifecycle.loss_trail
                    else current_loss
                )
                reference_loss = min(current_loss, committed_loss)
                minimum_improvement = max(1e-6, abs(reference_loss) * 1e-7)
                if (
                    math.isfinite(candidate_loss)
                    and reference_loss - candidate_loss >= minimum_improvement
                ):
                    if fixed_line_search_evaluations is None:
                        best_candidate = candidate
                        best_candidate_loss = candidate_loss
                        best_step_size = step_size
                        best_backtrack = backtrack
                        accepted = True
                        break
                    if candidate_loss < best_candidate_loss:
                        best_candidate = candidate
                        best_candidate_loss = candidate_loss
                        best_step_size = step_size
                        best_backtrack = backtrack
                step_size *= 0.5
            if best_candidate is not None:
                params = best_candidate
                self.lifecycle.loss_trail.append(best_candidate_loss)
                self.lifecycle.optimized_steps += 1
                self.lifecycle.line_search_backtracks += best_backtrack
                self.lifecycle.accepted_step_sizes.append(best_step_size)
                accepted = True
            if not accepted:
                self.lifecycle.rejected_steps += 1
                bind_params(params)
                if fixed_line_search_evaluations is None:
                    break
        bind_params(params)  # leave the best params installed without another forward pass
        if self.lifecycle.optimized_steps:
            self._notify_function_change("fast_weights_optimized")

    # ── Consolidation handoff ───────────────────────────────────────────
    def export_candidate(
        self,
        queue_dir: Path | str,
        *,
        episode_id: str,
        evidence: dict[str, Any],
    ) -> Path | None:
        """Serialize ΔW + evidence into the governed consolidation queue.

        Refused unless erase was PROVEN — a candidate from an episode whose
        cleanup could not be verified is not trustworthy evidence. The
        permanent-learning decision belongs to the LoRA compounding loop's
        regression gates, never to this module.
        """
        import numpy as np

        if self.lifecycle.erase_proven is not True:
            self.last_export_error = "erase_not_proven"
            logger.info("Consolidation export refused: erase not proven for %s", episode_id)
            return None
        if not getattr(self, "_exported_handles", None):
            self.last_export_error = "snapshot_unavailable"
            logger.info(
                "Consolidation export refused: no snapshot taken before detach for %s",
                episode_id,
            )
            return None
        if any(bool(handle.get("query_conditioned")) for handle in self._exported_handles):
            self.last_export_error = "query_conditioned_candidate_not_generalized"
            logger.info(
                "Consolidation export refused: episode-scoped query gate cannot become "
                "an unconditional durable adapter for %s",
                episode_id,
            )
            return None
        self.last_export_receipt = None
        self.last_export_error = ""
        try:
            from core.brain.llm.latent_cortex.persistence import (
                get_latent_cortex_persistence,
            )

            target_dir = (Path(queue_dir).expanduser() / episode_id).resolve()
            arrays: dict[str, Any] = {}
            for handle in self._exported_handles:
                arrays[f"layer{handle['layer_index']}_U"] = handle["U"]
                arrays[f"layer{handle['layer_index']}_V"] = handle["V"]
            buffer = io.BytesIO()
            np.savez(buffer, **arrays)
            delta_payload = buffer.getvalue()
            delta_sha256 = hashlib.sha256(delta_payload).hexdigest()
            lifecycle_receipt = self.lifecycle.to_receipt()
            lifecycle_receipt["exported"] = True
            payload = {
                "schema": "aura.latent_cortex.fast_weight_candidate.v1",
                "episode_id": episode_id,
                "created_at": time.time(),
                "target": self.lifecycle.target,
                "rank": self.lifecycle.rank,
                "layers": self.lifecycle.layers,
                "evidence": evidence,
                "lifecycle": lifecycle_receipt,
                "artifacts": {
                    "delta_weights.npz": {
                        "sha256": delta_sha256,
                        "size_bytes": len(delta_payload),
                    }
                },
            }
            evidence_payload = json.dumps(payload, indent=1, sort_keys=True).encode("utf-8")
            evidence_sha256 = hashlib.sha256(evidence_payload).hexdigest()
            receipt = get_latent_cortex_persistence().publish_fast_weight_candidate(
                target_dir,
                delta_payload=delta_payload,
                evidence_payload=evidence_payload,
            )
            committed_hashes = dict(receipt.sha256)
            expected_hashes = {
                str(target_dir / "delta_weights.npz"): delta_sha256,
                str(target_dir / "evidence.json"): evidence_sha256,
            }
            if set(receipt.paths) != set(expected_hashes) or committed_hashes != expected_hashes:
                raise RuntimeError("fast-weight batch receipt does not match payloads")
            self.lifecycle.exported = True
            self.last_export_receipt = {
                "transaction_id": receipt.transaction_id,
                "paths": list(receipt.paths),
                "sha256": committed_hashes,
            }
            return target_dir
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            detail = " ".join(str(exc).split())[:240]
            self.last_export_error = f"{type(exc).__name__}:{detail}"
            from core.runtime.errors import record_degradation

            record_degradation(
                "latent_cortex",
                exc,
                action="dropped consolidation candidate after queue export failed",
            )
            return None

    def snapshot_for_export(self) -> None:
        """Capture U/V as numpy BEFORE detach (arrays outlive the wrappers)."""
        import numpy as np

        # A vanilla-incumbent episode may have already staged an accepted
        # candidate and detached it before public decode. Final cleanup calls
        # this method again; an empty second snapshot must not erase the
        # candidate captured while the wrappers were still attached.
        if not self.handles and getattr(self, "_exported_handles", None):
            return
        self._exported_handles = [
            {
                "layer_index": h.layer_index,
                "U": np.array(h.wrapper.U),
                "V": np.array(h.wrapper.V),
                "query_conditioned": h.wrapper.query_gate_keys is not None,
            }
            for h in self.handles
        ]

    def stage_for_deferred_export(self) -> None:
        """Preserve an accepted delta while returning public decode to base.

        This is not canary erasure: the adaptation passed its causal gate but
        the incumbent output policy does not grant it serving authority. The
        candidate remains private evidence for later consolidation while the
        current answer is decoded with the frozen checkpoint.
        """

        if not self.handles:
            raise RuntimeError("deferred export requires attached fast weights")
        self.snapshot_for_export()
        self.detach()


__all__ = [
    "EpisodicDeltaLinear",
    "EpisodicFastWeights",
    "FAST_WEIGHT_OPTIMIZER",
    "FastWeightHandle",
    "FastWeightsLifecycle",
]

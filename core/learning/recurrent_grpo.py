"""Group-relative policy optimization through Aura's recurrent live path.

This module is the missing objective bridge between verifier rewards and the
Recursive Latent Cortex. It does not generate or grade tasks. It consumes exact
completion tokens, the branch that produced each completion, frozen old-policy
log-probabilities captured at sampling time, and programmatic rewards. Current
policy probabilities are recomputed by the differentiable resident KV-cached
path so their gradient passes through latent slots, recurrent window layers,
branch exchange, and the persisted answer path.

The adapter-disabled reference still executes the same RLC graph. This keeps KL
anchoring from comparing a recurrent policy to an architecturally different
standard decoder.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.grpo import group_advantages
from core.learning.recurrence_native_objective_v2 import (
    EXACT_ADJOINT_TRAJECTORY_RECEIPT_SCHEMA,
    INTERVENTION_MEASUREMENT_TRUST_BOUNDARY,
    ExactAdjointInterventionConfig,
    ExactAdjointTrajectoryConfig,
    LivePathForward,
    PreparedFinalRecurrentTransition,
    cached_live_path_token_logprobs,
    exact_adjoint_composite_live_path_value_and_grad,
    prepare_final_recurrent_transition,
    validate_exact_adjoint_live_path_receipt,
    validate_final_recurrent_transition_receipt,
)

RECURRENT_GRPO_SCHEMA = "aura.recurrent_grpo.v1"
VERIFIED_TRAJECTORY_GROUP_SCHEMA = "aura.recurrent_grpo.verified_trajectory_composite.v1"
VERIFIED_TRAJECTORY_SOURCE_SCHEMA = "aura.recurrent_grpo.verified_trajectory_source.v1"
VERIFIED_TRAJECTORY_GROUP_SCHEMA_V2 = "aura.recurrent_grpo.verified_trajectory_composite.v2"
VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V2 = "aura.recurrent_grpo.verified_trajectory_source.v2"
VERIFIED_TRAJECTORY_GROUP_SCHEMA_V3 = "aura.recurrent_grpo.verified_trajectory_composite.v3"
VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3 = "aura.recurrent_grpo.verified_trajectory_source.v3"
RECURRENT_SAMPLING_SCHEMA = "aura.recurrent_sampling_behavior.v4"
CAUSAL_RECURRENT_DECODE_SCHEMA = "aura.causal_recurrent_decode.v1"
CAUSAL_RECURRENT_PAIR_SCHEMA = "aura.causal_recurrent_transition_pair.v2"


@dataclass(frozen=True, slots=True)
class RecurrentSamplingConfig:
    """Proof-grade sampling controls for the fixed recurrent policy graph."""

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    # KV-cached and full teacher-forced quantized kernels are behaviorally
    # equivalent but not numerically identical, especially at the first token.
    # PPO's exact behavior ratio and clipped-token bound are the primary safety
    # contract; these wider secondary bounds catch wholesale graph divergence.
    max_abs_logprob_drift: float = 4.0
    max_mean_abs_logprob_drift: float = 0.5
    clip_epsilon: float = 0.2
    max_clipped_token_fraction: float = 0.25
    max_old_policy_approx_kl: float = 0.1

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 8192:
            raise ValueError("max_tokens must be inside [1, 8192]")
        for name, value in (
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("max_abs_logprob_drift", self.max_abs_logprob_drift),
            ("max_mean_abs_logprob_drift", self.max_mean_abs_logprob_drift),
            ("clip_epsilon", self.clip_epsilon),
            ("max_clipped_token_fraction", self.max_clipped_token_fraction),
            ("max_old_policy_approx_kl", self.max_old_policy_approx_kl),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        # Temperature scaling, nucleus truncation, repetition penalties, and
        # minimum-token masks define a different behavior policy from the raw
        # recurrent model scored by the exact adjoint. Proof campaigns use the
        # unmodified categorical distribution; product decoding stays free to
        # use its richer controls outside this training surface.
        if float(self.temperature) != 1.0:
            raise ValueError("proof-grade recurrent sampling requires temperature=1")
        if float(self.top_p) != 1.0:
            raise ValueError("proof-grade recurrent sampling requires top_p=1")
        if not 0.0 <= float(self.max_abs_logprob_drift) <= 100.0:
            raise ValueError("max_abs_logprob_drift must be inside [0, 100]")
        if not 0.0 <= float(self.max_mean_abs_logprob_drift) <= 100.0:
            raise ValueError("max_mean_abs_logprob_drift must be inside [0, 100]")
        if not 0.0 < float(self.clip_epsilon) <= 1.0:
            raise ValueError("clip_epsilon must be inside (0, 1]")
        if not 0.0 <= float(self.max_clipped_token_fraction) <= 1.0:
            raise ValueError("max_clipped_token_fraction must be inside [0, 1]")
        if not 0.0 <= float(self.max_old_policy_approx_kl) <= 100.0:
            raise ValueError("max_old_policy_approx_kl must be inside [0, 100]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "max_abs_logprob_drift": float(self.max_abs_logprob_drift),
            "max_mean_abs_logprob_drift": float(self.max_mean_abs_logprob_drift),
            "clip_epsilon": float(self.clip_epsilon),
            "max_clipped_token_fraction": float(self.max_clipped_token_fraction),
            "max_old_policy_approx_kl": float(self.max_old_policy_approx_kl),
        }


@dataclass(frozen=True, slots=True)
class RecurrentPolicySample:
    tokens: tuple[int, ...]
    branch_index: int
    behavior_logprobs: tuple[float, ...]
    differentiable_logprobs: tuple[float, ...]
    max_abs_logprob_drift: float
    mean_abs_logprob_drift: float
    max_abs_logprob_drift_token_index: int
    clipped_token_fraction: float
    old_policy_approx_kl: float
    behavior_admitted: bool
    execution_spec_sha256: str
    policy_sha256: str
    prompt_tokens_sha256: str
    seed: int
    sampling_config: RecurrentSamplingConfig
    episode_receipt: dict[str, Any]
    episode_id: str
    rng_root_sha256: str
    sample_kind: str = "engine_episode"
    causal_transition_pair: dict[str, Any] | None = None

    def receipt(self) -> dict[str, Any]:
        encoded = json.dumps(list(self.tokens), separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
        behavior_encoded = json.dumps(
            list(self.behavior_logprobs),
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        differentiable_encoded = json.dumps(
            list(self.differentiable_logprobs),
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        episode_receipt = dict(self.episode_receipt)
        receipt = {
            "schema": RECURRENT_SAMPLING_SCHEMA,
            "sample_kind": self.sample_kind,
            "episode_id": self.episode_id,
            "rng_root_sha256": self.rng_root_sha256,
            "execution_spec_sha256": self.execution_spec_sha256,
            "policy_sha256": self.policy_sha256,
            "prompt_tokens_sha256": self.prompt_tokens_sha256,
            "seed": self.seed,
            "sampling_config": self.sampling_config.to_dict(),
            "token_count": len(self.tokens),
            "tokens": list(self.tokens),
            "tokens_sha256": hashlib.sha256(encoded).hexdigest(),
            "behavior_logprobs": list(self.behavior_logprobs),
            "behavior_logprobs_sha256": hashlib.sha256(behavior_encoded).hexdigest(),
            "differentiable_logprobs": list(self.differentiable_logprobs),
            "differentiable_logprobs_sha256": hashlib.sha256(differentiable_encoded).hexdigest(),
            "branch_index": self.branch_index,
            "max_abs_logprob_drift": self.max_abs_logprob_drift,
            "mean_abs_logprob_drift": self.mean_abs_logprob_drift,
            "max_abs_logprob_drift_token_index": (self.max_abs_logprob_drift_token_index),
            "clipped_token_fraction": self.clipped_token_fraction,
            "old_policy_approx_kl": self.old_policy_approx_kl,
            "behavior_admitted": self.behavior_admitted,
            "episode_receipt": episode_receipt,
            "episode_receipt_sha256": _seal_receipt(episode_receipt),
            "causal_transition_pair": (
                dict(self.causal_transition_pair)
                if self.causal_transition_pair is not None
                else None
            ),
            "cached_decode_termination": self.episode_receipt.get("decode_termination", ""),
            "cached_params_unchanged": self.episode_receipt.get("params_unchanged"),
            "cached_runtime_integrity": dict(self.episode_receipt.get("runtime_integrity", {})),
            "cached_nonparametric_memory_status": str(
                self.episode_receipt.get("nonparametric_memory", {}).get(
                    "status",
                    "",
                )
            ),
            "cached_recurrence_adapter": dict(self.episode_receipt.get("recurrence_adapter", {})),
        }
        return json.loads(
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )


@dataclass(frozen=True, slots=True)
class FrozenRecurrentStateDecode:
    tokens: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    branch_index: int
    depth: int
    state_sha256: str
    ensemble_sha256: str
    seed: int
    token_budget: int
    receipt_sha256: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": CAUSAL_RECURRENT_DECODE_SCHEMA,
            "branch_index": self.branch_index,
            "depth": self.depth,
            "state_sha256": self.state_sha256,
            "ensemble_sha256": self.ensemble_sha256,
            "seed": self.seed,
            "token_budget": self.token_budget,
            "token_count": len(self.tokens),
            "tokens": list(self.tokens),
            "tokens_sha256": _tokens_sha256(self.tokens),
            "behavior_logprobs": list(self.behavior_logprobs),
            "behavior_logprobs_sha256": _float_sequence_sha256(self.behavior_logprobs),
            "termination": "fixed_token_budget",
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class CausalRecurrentTransitionPair:
    transition: PreparedFinalRecurrentTransition
    parent: FrozenRecurrentStateDecode
    child: FrozenRecurrentStateDecode
    child_differentiable_logprobs: tuple[float, ...]
    max_abs_child_logprob_drift: float
    mean_abs_child_logprob_drift: float
    child_behavior_admitted: bool
    policy_sha256: str
    episode_id: str
    rng_root_sha256: str
    recurrence_adapter: dict[str, Any]
    runtime_integrity: dict[str, Any]
    params_unchanged: bool
    sampling_config: RecurrentSamplingConfig
    receipt_sha256: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": CAUSAL_RECURRENT_PAIR_SCHEMA,
            "episode_id": self.episode_id,
            "rng_root_sha256": self.rng_root_sha256,
            "policy_sha256": self.policy_sha256,
            "transition": self.transition.receipt(),
            "parent": self.parent.receipt(),
            "child": self.child.receipt(),
            "common_random_seed": self.parent.seed,
            "fixed_token_budget": self.parent.token_budget,
            "sampling_config": self.sampling_config.to_dict(),
            "child_differentiable_logprobs": list(self.child_differentiable_logprobs),
            "child_differentiable_logprobs_sha256": _float_sequence_sha256(
                self.child_differentiable_logprobs
            ),
            "max_abs_child_logprob_drift": self.max_abs_child_logprob_drift,
            "mean_abs_child_logprob_drift": self.mean_abs_child_logprob_drift,
            "child_behavior_admitted": self.child_behavior_admitted,
            "recurrence_adapter": dict(self.recurrence_adapter),
            "runtime_integrity": dict(self.runtime_integrity),
            "params_unchanged": self.params_unchanged,
            "receipt_sha256": self.receipt_sha256,
        }


class RecurrentSamplingAdmissionError(RuntimeError):
    """Raised when cached behavior is outside the bounded PPO contract."""

    def __init__(self, sample: RecurrentPolicySample) -> None:
        self.sample = sample
        super().__init__(
            "cached recurrent behavior failed PPO admission: "
            f"max={sample.max_abs_logprob_drift:.6f} "
            f"mean={sample.mean_abs_logprob_drift:.6f} "
            f"clipped={sample.clipped_token_fraction:.6f}"
        )


# Compatibility for CP256 callers. New receipts and failures use the accurate
# behavior-policy name because the cached and adjoint kernels are not identical.
RecurrentSamplingParityError = RecurrentSamplingAdmissionError


class RecurrentGroupClipAdmissionError(RuntimeError):
    """Raised before adjoint replay when the cached policy is too far away."""

    def __init__(
        self,
        *,
        clip_fraction: float,
        max_clip_fraction: float,
        old_policy_approx_kl: float,
    ) -> None:
        self.clip_fraction = float(clip_fraction)
        self.max_clip_fraction = float(max_clip_fraction)
        self.old_policy_approx_kl = float(old_policy_approx_kl)
        super().__init__(
            "cached behavior policy exceeds recurrent PPO clip admission: "
            f"observed={self.clip_fraction:.6f} "
            f"limit={self.max_clip_fraction:.6f}"
        )


def _tokens_sha256(tokens: Sequence[int]) -> str:
    encoded = json.dumps(list(tokens), separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _float_sequence_sha256(values: Sequence[float]) -> str:
    encoded = json.dumps(
        [float(value) for value in values],
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _seal_receipt(body: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(body),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def recurrent_sampling_rng_root_sha256(
    *,
    episode_id: str,
    prompt_tokens_sha256: str,
    policy_sha256: str,
    execution_spec_sha256: str,
    branch_index: int,
    seed: int,
    sampling_config: RecurrentSamplingConfig,
) -> str:
    """Commit the complete deterministic categorical stream before sampling."""

    return _seal_receipt(
        {
            "schema": "aura.recurrent_sampling_rng_root.v1",
            "episode_id": episode_id,
            "prompt_tokens_sha256": prompt_tokens_sha256,
            "policy_sha256": policy_sha256,
            "execution_spec_sha256": execution_spec_sha256,
            "branch_index": branch_index,
            "seed": seed,
            "sampling_config": sampling_config.to_dict(),
            "algorithm": "mlx_categorical_seed_plus_golden_ratio_draw_v1",
        }
    )


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def recurrent_policy_tensor_map_sha256(
    tensors: Mapping[str, Any],
    execution_spec_sha256: str,
) -> str:
    """Hash one flat adapter tensor map plus its recurrent graph identity."""

    if (
        not isinstance(tensors, Mapping)
        or not tensors
        or not _valid_sha256(execution_spec_sha256)
        or any(not isinstance(name, str) or not name for name in tensors)
    ):
        raise ValueError("recurrent policy tensor map is invalid")
    import mlx.core as mx
    import numpy as np

    ordered = sorted(tensors.items())
    mx.eval([value for _name, value in ordered])
    digest = hashlib.sha256()
    digest.update(b"aura.recurrent_policy.v1\0")
    digest.update(execution_spec_sha256.encode("ascii"))
    for name, value in ordered:
        name_bytes = name.encode("utf-8")
        dtype = str(value.dtype).encode("ascii")
        shape = json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        try:
            array = np.asarray(value)
        except RuntimeError:
            # NumPy lacks a portable bfloat16 buffer code. Conversion to
            # float32 is exact for every bfloat16 value; the original dtype is
            # already bound separately above.
            array = np.asarray(value.astype(mx.float32))
        payload = array.tobytes(order="C")
        for part in (name_bytes, dtype, shape, payload):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def attach_recurrent_policy_adapters(
    model: Any,
    spec: RLCExecutionSpec,
    *,
    lora_rank: int,
    lora_layers: int,
    lora_targets: Sequence[str],
    initialization_seed: int,
    lora_dropout: float = 0.0,
    lora_scale: float = 20.0,
    lora_layer_placement: str = "late",
    depth_conditioned_steps: int | None = None,
    role_conditioned_branches: int | None = None,
    coda_lora_layers: int = 0,
    coda_lora_targets: Sequence[str] = (),
) -> tuple[str, ...]:
    """Attach the exact proof-campaign adapter topology to a frozen model.

    Training and independent state replay call this same source-bound
    constructor. The complete topology is preflighted before the first module
    mutation so a missing, ambiguous, or already-wrapped projection cannot
    leave behind a partial policy.
    """

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        ScopedCodaLoRALinear,
        ScopedLoRALinear,
    )

    if (
        type(lora_rank) is not int
        or lora_rank < 1
        or type(lora_layers) is not int
        or lora_layers < 1
        or type(initialization_seed) is not int
        or not 0 <= initialization_seed <= 0xFFFFFFFF
        or isinstance(lora_dropout, bool)
        or not isinstance(lora_dropout, (int, float))
        or not math.isfinite(float(lora_dropout))
        or not 0.0 <= float(lora_dropout) <= 0.5
        or isinstance(lora_scale, bool)
        or not isinstance(lora_scale, (int, float))
        or not math.isfinite(float(lora_scale))
        or not 0.0 < float(lora_scale) <= 1024.0
        or lora_layer_placement not in {"early", "distributed", "late"}
        or not isinstance(lora_targets, Sequence)
        or isinstance(lora_targets, (str, bytes, bytearray))
        or type(coda_lora_layers) is not int
        or coda_lora_layers < 0
        or not isinstance(coda_lora_targets, Sequence)
        or isinstance(coda_lora_targets, (str, bytes, bytearray))
        or (
            depth_conditioned_steps is not None
            and (
                type(depth_conditioned_steps) is not int
                or not 1 <= depth_conditioned_steps <= 64
            )
        )
        or (
            role_conditioned_branches is not None
            and (
                type(role_conditioned_branches) is not int
                or not 2 <= role_conditioned_branches <= 32
                or role_conditioned_branches != len(spec.branch_roles)
            )
        )
    ):
        raise ValueError("recurrent policy adapter configuration is invalid")
    targets = tuple(lora_targets)
    coda_targets = tuple(coda_lora_targets)
    if (
        not targets
        or len(targets) != len(set(targets))
        or any(
            not isinstance(target, str)
            or not target
            or target != target.strip()
            for target in targets
        )
    ):
        raise ValueError("recurrent policy adapter targets are invalid")
    if (
        bool(coda_lora_layers) != bool(coda_targets)
        or len(coda_targets) != len(set(coda_targets))
        or any(
            not isinstance(target, str)
            or not target
            or target != target.strip()
            for target in coda_targets
        )
    ):
        raise ValueError("coda policy adapter targets are invalid")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("model exposes no decoder layer inventory")
    prelude_end = max(1, int(len(layers) * spec.prelude_frac))
    coda_start = min(
        len(layers) - 1,
        len(layers) - max(1, int(len(layers) * spec.coda_frac)),
    )
    available_layers = coda_start - prelude_end
    if available_layers < 1 or lora_layers > available_layers:
        raise ValueError(
            "requested recurrent policy adapter layers exceed the recurrent window"
        )
    available_coda_layers = len(layers) - coda_start
    if coda_lora_layers > available_coda_layers:
        raise ValueError("requested coda adapter layers exceed the coda window")
    from core.brain.llm.latent_cortex.plasticity_sites import (
        select_plasticity_layers,
    )

    adapted_indices = select_plasticity_layers(
        prelude_end,
        coda_start,
        lora_layers,
        placement=lora_layer_placement,
    )
    planned: list[tuple[Any, str, Any, int, str, type[Any]]] = []

    def plan_layer(
        layer_index: int,
        layer_targets: Sequence[str],
        adapter_type: type[Any],
    ) -> None:
        layer = layers[layer_index]
        for target in layer_targets:
            matches: list[tuple[str, Any, Any]] = []
            for parent_name in ("self_attn", "mlp"):
                parent = getattr(layer, parent_name, None)
                if parent is None:
                    continue
                projection = getattr(parent, target, None)
                if projection is not None:
                    matches.append((parent_name, parent, projection))
            if len(matches) != 1:
                raise ValueError(
                    "recurrent policy adapter target must resolve exactly once: "
                    f"layer={layer_index} target={target} matches={len(matches)}"
                )
            parent_name, parent, projection = matches[0]
            if isinstance(projection, (ScopedLoRALinear, ScopedCodaLoRALinear)):
                raise ValueError(
                    "cognitive policy adapter target is already wrapped: "
                    f"layer={layer_index} target={target}"
                )
            site = f"model.layers.{layer_index}.{parent_name}.{target}"
            planned.append((parent, target, projection, layer_index, site, adapter_type))

    for layer_index in adapted_indices:
        plan_layer(layer_index, targets, ScopedLoRALinear)
    for layer_index in range(coda_start, coda_start + coda_lora_layers):
        plan_layer(layer_index, coda_targets, ScopedCodaLoRALinear)

    import mlx.core as mx

    model.freeze()
    mx.random.seed(initialization_seed)
    for parent, target, projection, layer_index, site, adapter_type in planned:
        setattr(
            parent,
            target,
            adapter_type.from_base(
                projection,
                r=lora_rank,
                dropout=float(lora_dropout),
                scale=float(lora_scale),
                block_index=layer_index,
                site=site,
            ),
        )
    if depth_conditioned_steps is not None:
        from core.learning.depth_conditioned_lora import wrap_depth_conditioned

        banks = wrap_depth_conditioned(model, depths=depth_conditioned_steps)
        recurrent_sites = {
            site for *_identity, site, adapter_type in planned
            if adapter_type is ScopedLoRALinear
        }
        if set(banks) != recurrent_sites:
            raise RuntimeError("depth-conditioned adapter topology drift")
    if role_conditioned_branches is not None:
        from core.learning.role_conditioned_lora import wrap_role_conditioned

        role_banks = wrap_role_conditioned(
            model,
            branches=role_conditioned_branches,
        )
        recurrent_sites = {
            site for *_identity, site, adapter_type in planned
            if adapter_type is ScopedLoRALinear
        }
        if set(role_banks) != recurrent_sites:
            raise RuntimeError("role-conditioned adapter topology drift")
    return tuple(site for _parent, _target, _projection, _index, site, _type in planned)


def attach_coda_policy_adapters_at_sites(
    model: Any,
    sites: Sequence[str],
    *,
    lora_rank: int,
    initialization_seed: int,
    lora_dropout: float = 0.0,
    lora_scale: float = 20.0,
) -> tuple[str, ...]:
    """Attach decode-scoped tissue at an exact, non-contiguous site inventory.

    Empirical localization may select middle-layer projections rather than a
    contiguous anatomical coda window.  The site list is therefore the
    topology contract: every path is preflighted before mutation, attachment
    is transactional, and returned order is canonical.
    """

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        ScopedCodaLoRALinear,
        ScopedLoRALinear,
    )

    if (
        not isinstance(sites, Sequence)
        or isinstance(sites, (str, bytes, bytearray))
        or not sites
        or type(lora_rank) is not int
        or lora_rank < 1
        or type(initialization_seed) is not int
        or not 0 <= initialization_seed <= 0xFFFFFFFF
        or isinstance(lora_dropout, bool)
        or not isinstance(lora_dropout, (int, float))
        or not math.isfinite(float(lora_dropout))
        or not 0.0 <= float(lora_dropout) <= 0.5
        or isinstance(lora_scale, bool)
        or not isinstance(lora_scale, (int, float))
        or not math.isfinite(float(lora_scale))
        or not 0.0 < float(lora_scale) <= 1024.0
    ):
        raise ValueError("exact coda adapter configuration is invalid")
    canonical_sites = tuple(sorted(sites))
    if (
        len(canonical_sites) != len(set(canonical_sites))
        or any(
            not isinstance(site, str)
            or not site
            or site != site.strip()
            for site in canonical_sites
        )
    ):
        raise ValueError("exact coda adapter site inventory is invalid")

    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("model exposes no decoder layer inventory")
    planned: list[tuple[Any, str, Any, int, str]] = []
    for site in canonical_sites:
        parts = site.split(".")
        if len(parts) != 5 or parts[:2] != ["model", "layers"]:
            raise ValueError(f"exact coda adapter site path is invalid: {site}")
        try:
            layer_index = int(parts[2])
        except ValueError as exc:
            raise ValueError(f"exact coda adapter layer is invalid: {site}") from exc
        if not 0 <= layer_index < len(layers) or parts[3] not in {"self_attn", "mlp"}:
            raise ValueError(f"exact coda adapter site path is invalid: {site}")
        parent = getattr(layers[layer_index], parts[3], None)
        projection = getattr(parent, parts[4], None)
        if projection is None:
            raise ValueError(f"exact coda adapter site is absent: {site}")
        if isinstance(projection, (ScopedLoRALinear, ScopedCodaLoRALinear)):
            raise ValueError(f"exact coda adapter site is already wrapped: {site}")
        planned.append((parent, parts[4], projection, layer_index, site))

    import mlx.core as mx

    model.freeze()
    mx.random.seed(initialization_seed)
    attached: list[tuple[Any, str, Any]] = []
    try:
        for parent, target, projection, layer_index, site in planned:
            wrapped = ScopedCodaLoRALinear.from_base(
                projection,
                r=lora_rank,
                dropout=float(lora_dropout),
                scale=float(lora_scale),
                block_index=layer_index,
                site=site,
            )
            setattr(parent, target, wrapped)
            attached.append((parent, target, projection))
    except BaseException:
        for parent, target, projection in reversed(attached):
            setattr(parent, target, projection)
        raise
    return canonical_sites


def recurrent_policy_optimizer_config(
    learning_rate: float,
) -> dict[str, Any]:
    """Return the explicit optimizer contract shared by trainer and replayer."""

    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or not 0.0 < float(learning_rate) <= 1.0
    ):
        raise ValueError("recurrent policy learning rate is invalid")
    return {
        "class_name": "mlx.optimizers.Adam",
        "learning_rate_hex": float(learning_rate).hex(),
        "betas_hex": [(0.9).hex(), (0.999).hex()],
        "eps_hex": (1e-8).hex(),
        "bias_correction": False,
    }


def build_recurrent_policy_optimizer(learning_rate: float) -> Any:
    """Build the exact optimizer committed by the proof campaign."""

    import mlx.optimizers as optim

    config = recurrent_policy_optimizer_config(learning_rate)
    return optim.Adam(
        learning_rate=float.fromhex(config["learning_rate_hex"]),
        betas=[
            float.fromhex(value) for value in config["betas_hex"]
        ],
        eps=float.fromhex(config["eps_hex"]),
        bias_correction=config["bias_correction"],
    )


def recurrent_policy_sha256(model: Any, spec: RLCExecutionSpec) -> str:
    """Hash the exact trainable tensor tree plus its recurrent graph."""

    from mlx.utils import tree_flatten

    return recurrent_policy_tensor_map_sha256(
        dict(tree_flatten(model.trainable_parameters())),
        spec.sha256,
    )


def _decode_frozen_recurrent_state(
    model: Any,
    prompt_tokens: Sequence[int],
    transition: PreparedFinalRecurrentTransition,
    *,
    state: Any,
    state_sha256: str,
    ensemble_sha256: str,
    branch_index: int,
    depth: int,
    seed: int,
    token_budget: int,
) -> FrozenRecurrentStateDecode:
    """Decode one frozen state with a live KV cache and fixed random stream."""

    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.base import create_attention_mask

    from core.brain.llm.latent_cortex.recurrence import WindowRunner
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        coda_adapter_scope,
        recurrence_adapter_scope,
    )
    from core.brain.llm.latent_cortex.types import ComputeBudget
    from core.learning.role_conditioned_lora import recurrent_branch_index

    if type(branch_index) is not int or not 0 <= branch_index < len(transition.seeds):
        raise ValueError("branch_index is outside the frozen transition")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be inside [0, 2^32-1]")
    if type(token_budget) is not int or not 1 <= token_budget <= 8192:
        raise ValueError("token_budget must be inside [1, 8192]")
    if type(depth) is not int or depth < 0:
        raise ValueError("decode depth must be non-negative")

    from core.brain.llm.decoder_topology import (
        decoder_layers,
        resolve_language_model,
    )
    from core.learning.intrinsic_recurrence import model_layer_caches

    inner = resolve_language_model(model).model
    layers = tuple(decoder_layers(model))
    cache = model_layer_caches(model)
    prompt = tuple(prompt_tokens)
    hidden = inner.embed_tokens(mx.array([list(prompt)]))
    mask = create_attention_mask(hidden, cache)
    for index, layer in enumerate(layers):
        hidden = layer(hidden, mask, cache[index])

    budget = ComputeBudget(
        max_layer_apps=max(
            1,
            (len(prompt) + 3 * int(transition.seeds[branch_index].shape[1]) + token_budget)
            * len(layers),
        ),
        wall_clock_s=600.0,
    )
    runner = WindowRunner(inner, budget)
    seed_slots = transition.seeds[branch_index]
    runner.run(
        seed_slots,
        cache,
        0,
        transition.prelude_end,
        persist=True,
    )
    with recurrent_branch_index(branch_index), recurrence_adapter_scope():
        persisted = runner.run(
            state,
            cache,
            transition.prelude_end,
            transition.coda_start,
            persist=True,
        )
    with coda_adapter_scope():
        output = runner.run(
            persisted,
            cache,
            transition.coda_start,
            len(layers),
            persist=True,
        )

    def logits_for(value: Any) -> Any:
        normalized = inner.norm(value)
        head = getattr(model, "lm_head", None)
        if head is not None and not isinstance(head, type(inner.embed_tokens)):
            return head(normalized)
        return inner.embed_tokens.as_linear(normalized)

    logits = logits_for(output)[0, -1]
    tokens: list[int] = []
    logprobs: list[float] = []
    for draw in range(token_budget):
        key = mx.random.key((seed + draw * 0x9E3779B1) & 0x7FFFFFFF)
        token = int(mx.random.categorical(logits, key=key))
        logprob = logits[token].astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32))
        mx.eval(logprob)
        tokens.append(token)
        logprobs.append(float(logprob))
        if draw + 1 == token_budget:
            break
        hidden = inner.embed_tokens(mx.array([[token]]))
        mask = create_attention_mask(hidden, cache)
        with coda_adapter_scope():
            for index, layer in enumerate(layers):
                hidden = layer(hidden, mask, cache[index])
        logits = logits_for(hidden)[0, -1]

    # The durable state digest is recomputed at the decode boundary.  This
    # catches caller substitution between transition preparation and decode.
    try:
        array = np.asarray(state)
    except RuntimeError:
        array = np.asarray(state.astype(mx.float32))
    digest = hashlib.sha256()
    for part in (
        str(state.dtype).encode("ascii"),
        json.dumps(list(state.shape), separators=(",", ":")).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    if digest.hexdigest() != state_sha256:
        raise RuntimeError("frozen recurrent state changed before decode")
    body = {
        "schema": CAUSAL_RECURRENT_DECODE_SCHEMA,
        "branch_index": branch_index,
        "depth": depth,
        "state_sha256": state_sha256,
        "ensemble_sha256": ensemble_sha256,
        "seed": seed,
        "token_budget": token_budget,
        "token_count": len(tokens),
        "tokens": tokens,
        "tokens_sha256": _tokens_sha256(tokens),
        "behavior_logprobs": logprobs,
        "behavior_logprobs_sha256": _float_sequence_sha256(logprobs),
        "termination": "fixed_token_budget",
    }
    return FrozenRecurrentStateDecode(
        tokens=tuple(tokens),
        behavior_logprobs=tuple(logprobs),
        branch_index=branch_index,
        depth=depth,
        state_sha256=state_sha256,
        ensemble_sha256=ensemble_sha256,
        seed=seed,
        token_budget=token_budget,
        receipt_sha256=_seal_receipt(body),
    )


def sample_final_recurrent_transition_pair(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    seed: int,
    sampling: RecurrentSamplingConfig | None = None,
    require_admission: bool = True,
    episode_id: str | None = None,
    tokenizer: Any | None = None,
    model_path: str | None = None,
) -> CausalRecurrentTransitionPair:
    """Decode the two ends of one real recurrent edge under matched noise."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.governance import CheckpointInvariant
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_activation_collector,
        scoped_recurrence_adapter_sites,
    )
    from core.brain.llm.latent_cortex.types import EpisodeReceipt

    resolved = sampling or RecurrentSamplingConfig()
    prompt = tuple(prompt_tokens)
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("prompt_tokens must contain non-negative integers")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be inside [0, 2^32-1]")
    resolved_episode_id = episode_id or (
        f"causal-{_tokens_sha256(prompt)[:16]}-{branch_index}-{seed}"
    )
    if (
        not isinstance(resolved_episode_id, str)
        or not resolved_episode_id
        or len(resolved_episode_id) > 192
        or not resolved_episode_id[0].isalnum()
        or any(
            not (character.isalnum() or character in "._:/;=+-")
            for character in resolved_episode_id
        )
    ):
        raise ValueError("episode_id is not a valid campaign identifier")

    layer_count = len(model.model.layers)
    prelude_end = max(1, int(layer_count * spec.prelude_frac))
    coda_start = min(
        layer_count - 1,
        layer_count - max(1, int(layer_count * spec.coda_frac)),
    )
    expected_adapter_sites = scoped_recurrence_adapter_sites(
        model,
        layer_indices=range(prelude_end, coda_start),
    )
    invariant = CheckpointInvariant(
        model,
        model_path,
        tokenizer=tokenizer,
        adapted_layer_indices=tuple(range(prelude_end, coda_start)),
        adapted_target="o_proj",
    )
    integrity_receipt = EpisodeReceipt(
        episode_id=resolved_episode_id,
        input_tokens_sha256=_tokens_sha256(prompt),
        input_token_count=len(prompt),
        n_layers=layer_count,
        prelude_end=prelude_end,
        coda_start=coda_start,
        n_slots=spec.n_slots,
        n_branches=len(spec.branch_roles),
        decode_requested_tokens=resolved.max_tokens,
        decode_generated_tokens=resolved.max_tokens,
        decode_termination="fixed_token_budget",
        nonparametric_memory={"status": "disabled_by_policy"},
    )
    invariant.pre_episode()
    integrity_receipt.checkpoint_fingerprint = invariant.file_receipt.get("fingerprint", "")
    integrity_receipt.checkpoint_fingerprint_method = invariant.file_receipt.get("method", "")
    integrity_receipt.checkpoint_file_count = int(invariant.file_receipt.get("files", 0) or 0)
    policy_sha256 = recurrent_policy_sha256(model, spec)
    rng_root_sha256 = recurrent_sampling_rng_root_sha256(
        episode_id=resolved_episode_id,
        prompt_tokens_sha256=_tokens_sha256(prompt),
        policy_sha256=policy_sha256,
        execution_spec_sha256=spec.sha256,
        branch_index=branch_index,
        seed=seed,
        sampling_config=resolved,
    )
    with recurrence_adapter_activation_collector() as activation:
        transition = prepare_final_recurrent_transition(
            model,
            prompt,
            spec=spec,
        )
        parent = _decode_frozen_recurrent_state(
            model,
            prompt,
            transition,
            state=transition.parent_states[branch_index],
            state_sha256=transition.parent_branch_sha256s[branch_index],
            ensemble_sha256=transition.parent_ensemble_sha256,
            branch_index=branch_index,
            depth=transition.transition_index,
            seed=seed,
            token_budget=resolved.max_tokens,
        )
        child = _decode_frozen_recurrent_state(
            model,
            prompt,
            transition,
            state=transition.child_states[branch_index],
            state_sha256=transition.child_branch_sha256s[branch_index],
            ensemble_sha256=transition.child_ensemble_sha256,
            branch_index=branch_index,
            depth=transition.transition_index + 1,
            seed=seed,
            token_budget=resolved.max_tokens,
        )
        differentiable = recurrent_completion_token_logprobs(
            model,
            prompt,
            child.tokens,
            spec=spec,
            branch_index=branch_index,
        )
        mx.eval(differentiable)
        differentiable_values = tuple(float(value) for value in differentiable)
    integrity_receipt.recurrence_adapter = {
        "schema": "aura.recurrence_adapter_activation.v1",
        "scope": "latent_slots_only",
        "calls": activation.calls,
        "adapted_positions": activation.adapted_positions,
        "observed_positions": activation.observed_positions,
        "active": activation.calls > 0,
        "applied_blocks": dict(sorted(activation.applied_blocks.items())),
        "applied_sites": dict(sorted(activation.applied_sites.items())),
        "expected_sites": list(expected_adapter_sites),
        "unfired_sites": activation.unfired_sites(expected_adapter_sites),
        "complete": not activation.unfired_sites(expected_adapter_sites),
    }
    integrity_receipt.params_unchanged = invariant.post_episode(integrity_receipt)
    differences = tuple(
        abs(behavior - target)
        for behavior, target in zip(
            child.behavior_logprobs,
            differentiable_values,
            strict=True,
        )
    )
    maximum = max(differences)
    mean = sum(differences) / len(differences)
    admitted = (
        maximum <= float(resolved.max_abs_logprob_drift)
        and mean <= float(resolved.max_mean_abs_logprob_drift)
        and integrity_receipt.params_unchanged is True
        and activation.calls > 0
        and activation.adapted_positions > 0
        and not activation.unfired_sites(expected_adapter_sites)
    )
    if recurrent_policy_sha256(model, spec) != policy_sha256:
        raise RuntimeError("recurrent policy changed during causal pair sampling")
    body = {
        "schema": CAUSAL_RECURRENT_PAIR_SCHEMA,
        "episode_id": resolved_episode_id,
        "rng_root_sha256": rng_root_sha256,
        "policy_sha256": policy_sha256,
        "transition": transition.receipt(),
        "parent": parent.receipt(),
        "child": child.receipt(),
        "common_random_seed": seed,
        "fixed_token_budget": resolved.max_tokens,
        "sampling_config": resolved.to_dict(),
        "child_differentiable_logprobs": list(differentiable_values),
        "child_differentiable_logprobs_sha256": _float_sequence_sha256(differentiable_values),
        "max_abs_child_logprob_drift": maximum,
        "mean_abs_child_logprob_drift": mean,
        "child_behavior_admitted": admitted,
        "recurrence_adapter": dict(integrity_receipt.recurrence_adapter),
        "runtime_integrity": dict(integrity_receipt.runtime_integrity),
        "params_unchanged": integrity_receipt.params_unchanged,
    }
    result = CausalRecurrentTransitionPair(
        transition=transition,
        parent=parent,
        child=child,
        child_differentiable_logprobs=differentiable_values,
        max_abs_child_logprob_drift=maximum,
        mean_abs_child_logprob_drift=mean,
        child_behavior_admitted=admitted,
        policy_sha256=policy_sha256,
        episode_id=resolved_episode_id,
        rng_root_sha256=rng_root_sha256,
        recurrence_adapter=dict(integrity_receipt.recurrence_adapter),
        runtime_integrity=dict(integrity_receipt.runtime_integrity),
        params_unchanged=bool(integrity_receipt.params_unchanged),
        sampling_config=resolved,
        receipt_sha256=_seal_receipt(body),
    )
    if require_admission and not admitted:
        raise RuntimeError(
            "causal recurrent child decode failed differentiable-policy admission: "
            f"max={maximum:.6f} mean={mean:.6f}"
        )
    return result


def validate_causal_recurrent_transition_pair_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema",
        "episode_id",
        "rng_root_sha256",
        "policy_sha256",
        "transition",
        "parent",
        "child",
        "common_random_seed",
        "fixed_token_budget",
        "sampling_config",
        "child_differentiable_logprobs",
        "child_differentiable_logprobs_sha256",
        "max_abs_child_logprob_drift",
        "mean_abs_child_logprob_drift",
        "child_behavior_admitted",
        "recurrence_adapter",
        "runtime_integrity",
        "params_unchanged",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise ValueError("causal_recurrent_pair_schema_invalid")
    normalized = dict(receipt)
    if normalized.get("schema") != CAUSAL_RECURRENT_PAIR_SCHEMA:
        raise ValueError("causal_recurrent_pair_version_invalid")
    transition = validate_final_recurrent_transition_receipt(normalized.get("transition"))
    parent = normalized.get("parent")
    child = normalized.get("child")
    if not isinstance(parent, Mapping) or not isinstance(child, Mapping):
        raise ValueError("causal_recurrent_pair_decode_missing")
    for decode, side, expected_depth, expected_state, expected_ensemble in (
        (
            parent,
            "parent",
            transition["parent_depth"],
            transition["parent_branch_sha256s"],
            transition["parent_ensemble_sha256"],
        ),
        (
            child,
            "child",
            transition["child_depth"],
            transition["child_branch_sha256s"],
            transition["child_ensemble_sha256"],
        ),
    ):
        branch_index = decode.get("branch_index")
        unsigned_decode = dict(decode)
        observed_decode = unsigned_decode.pop("receipt_sha256", None)
        if (
            decode.get("schema") != CAUSAL_RECURRENT_DECODE_SCHEMA
            or type(branch_index) is not int
            or not 0 <= branch_index < transition["branch_count"]
            or decode.get("depth") != expected_depth
            or decode.get("state_sha256") != expected_state[branch_index]
            or decode.get("ensemble_sha256") != expected_ensemble
            or decode.get("seed") != normalized.get("common_random_seed")
            or decode.get("token_budget") != normalized.get("fixed_token_budget")
            or decode.get("token_count") != len(decode.get("tokens", []))
            or decode.get("token_count") != decode.get("token_budget")
            or decode.get("tokens_sha256") != _tokens_sha256(decode.get("tokens", []))
            or decode.get("behavior_logprobs_sha256")
            != _float_sequence_sha256(decode.get("behavior_logprobs", []))
            or decode.get("termination") != "fixed_token_budget"
            or _seal_receipt(unsigned_decode) != observed_decode
        ):
            raise ValueError(f"causal_recurrent_pair_{side}_invalid")
    values = normalized.get("child_differentiable_logprobs")
    activation = normalized.get("recurrence_adapter")
    try:
        sampling = RecurrentSamplingConfig(**dict(normalized["sampling_config"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("causal_recurrent_pair_sampling_config_invalid") from exc
    unsigned = dict(normalized)
    observed = unsigned.pop("receipt_sha256", None)
    if (
        not isinstance(normalized.get("episode_id"), str)
        or not normalized["episode_id"]
        or len(normalized["episode_id"]) > 192
        or not _valid_sha256(normalized.get("rng_root_sha256"))
        or not _valid_sha256(normalized.get("policy_sha256"))
        or not isinstance(values, list)
        or len(values) != child.get("token_count")
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        )
        or normalized.get("child_differentiable_logprobs_sha256") != _float_sequence_sha256(values)
        or type(normalized.get("child_behavior_admitted")) is not bool
        or type(normalized.get("params_unchanged")) is not bool
        or not isinstance(activation, Mapping)
        or activation.get("schema") != "aura.recurrence_adapter_activation.v1"
        or activation.get("scope") != "latent_slots_only"
        or type(activation.get("active")) is not bool
        or type(activation.get("calls")) is not int
        or activation["calls"] < 0
        or type(activation.get("adapted_positions")) is not int
        or activation["adapted_positions"] < 0
        or type(activation.get("observed_positions")) is not int
        or activation["observed_positions"] < activation["adapted_positions"]
        or activation["active"] is not (activation["calls"] > 0)
        or not isinstance(activation.get("expected_sites"), list)
        or not activation["expected_sites"]
        or len(set(activation["expected_sites"])) != len(activation["expected_sites"])
        or not isinstance(activation.get("applied_sites"), Mapping)
        or not isinstance(activation.get("unfired_sites"), list)
        or activation["unfired_sites"]
        != sorted(set(activation["expected_sites"]) - set(activation["applied_sites"]))
        or type(activation.get("complete")) is not bool
        or activation["complete"] is not (not activation["unfired_sites"])
        or normalized.get("fixed_token_budget") != sampling.max_tokens
        or _seal_receipt(unsigned) != observed
    ):
        raise ValueError("causal_recurrent_pair_identity_invalid")
    from core.brain.llm.latent_cortex.runtime_integrity import (
        validate_runtime_integrity_receipt,
    )

    try:
        runtime = validate_runtime_integrity_receipt(
            normalized.get("runtime_integrity"),
            require_worker=False,
            expected_episode_id=normalized["episode_id"],
            expected_input_tokens_sha256=transition["prompt_tokens_sha256"],
            expected_fast_weights_applied=False,
            expected_fast_weights_attach_attempted=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("causal_recurrent_pair_runtime_integrity_invalid") from exc
    differences = [
        abs(float(behavior) - float(target))
        for behavior, target in zip(child["behavior_logprobs"], values, strict=True)
    ]
    maximum = max(differences)
    mean = sum(differences) / len(differences)
    if not math.isclose(
        float(normalized.get("max_abs_child_logprob_drift")),
        maximum,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(normalized.get("mean_abs_child_logprob_drift")),
        mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("causal_recurrent_pair_drift_metrics_invalid")
    runtime_safe = runtime["verdict"]["engine_measurements_complete"] is True
    expected_admitted = bool(
        maximum <= float(sampling.max_abs_logprob_drift)
        and mean <= float(sampling.max_mean_abs_logprob_drift)
        and runtime_safe
        and activation["active"] is True
        and activation["adapted_positions"] > 0
        and activation["complete"] is True
    )
    if (
        normalized["params_unchanged"] is not runtime_safe
        or normalized["child_behavior_admitted"] is not expected_admitted
        or normalized["rng_root_sha256"]
        != recurrent_sampling_rng_root_sha256(
            episode_id=normalized["episode_id"],
            prompt_tokens_sha256=transition["prompt_tokens_sha256"],
            policy_sha256=normalized["policy_sha256"],
            execution_spec_sha256=transition["execution_spec_sha256"],
            branch_index=child["branch_index"],
            seed=normalized["common_random_seed"],
            sampling_config=sampling,
        )
    ):
        raise ValueError("causal_recurrent_pair_admission_verdict_invalid")
    return normalized


def recurrent_policy_sample_from_causal_pair(
    pair: CausalRecurrentTransitionPair,
    *,
    sampling: RecurrentSamplingConfig | None = None,
    require_admission: bool = True,
) -> RecurrentPolicySample:
    """Promote one measured final-edge child into the optimizer sample type."""

    resolved = sampling or pair.sampling_config
    if resolved.to_dict() != pair.sampling_config.to_dict():
        raise ValueError("sampling config differs from causal pair")
    pair_receipt = validate_causal_recurrent_transition_pair_receipt(pair.receipt())
    behavior = tuple(pair.child.behavior_logprobs)
    target = tuple(pair.child_differentiable_logprobs)
    differences = tuple(abs(left - right) for left, right in zip(behavior, target, strict=True))
    maximum = max(differences)
    maximum_index = differences.index(maximum)
    mean = sum(differences) / len(differences)
    ratios = tuple(
        math.exp(target_value - behavior_value)
        for target_value, behavior_value in zip(target, behavior, strict=True)
    )
    clipped_fraction = sum(
        abs(ratio - 1.0) > float(resolved.clip_epsilon) for ratio in ratios
    ) / len(ratios)
    old_policy_approx_kl = sum((ratio - 1.0) - math.log(ratio) for ratio in ratios) / len(ratios)
    admitted = bool(
        pair.child_behavior_admitted
        and maximum <= float(resolved.max_abs_logprob_drift)
        and mean <= float(resolved.max_mean_abs_logprob_drift)
        and clipped_fraction <= float(resolved.max_clipped_token_fraction)
        and old_policy_approx_kl <= float(resolved.max_old_policy_approx_kl)
    )
    episode_receipt = {
        "episode_id": pair.episode_id,
        "input_tokens_sha256": pair.transition.prompt_tokens_sha256,
        "decode_termination": "fixed_token_budget",
        "params_unchanged": pair.params_unchanged,
        "runtime_integrity": dict(pair.runtime_integrity),
        "nonparametric_memory": {"status": "disabled_by_policy"},
        "recurrence_adapter": dict(pair.recurrence_adapter),
        "causal_transition_pair_sha256": pair.receipt_sha256,
    }
    sample = RecurrentPolicySample(
        tokens=tuple(pair.child.tokens),
        branch_index=pair.child.branch_index,
        behavior_logprobs=behavior,
        differentiable_logprobs=target,
        max_abs_logprob_drift=maximum,
        mean_abs_logprob_drift=mean,
        max_abs_logprob_drift_token_index=maximum_index,
        clipped_token_fraction=clipped_fraction,
        old_policy_approx_kl=old_policy_approx_kl,
        behavior_admitted=admitted,
        execution_spec_sha256=pair.transition.execution_spec_sha256,
        policy_sha256=pair.policy_sha256,
        prompt_tokens_sha256=pair.transition.prompt_tokens_sha256,
        seed=pair.child.seed,
        sampling_config=resolved,
        episode_receipt=episode_receipt,
        episode_id=pair.episode_id,
        rng_root_sha256=pair.rng_root_sha256,
        sample_kind="causal_final_transition",
        causal_transition_pair=pair_receipt,
    )
    if require_admission and not admitted:
        raise RecurrentSamplingAdmissionError(sample)
    return sample


def sample_final_recurrent_transition_completion(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    seed: int,
    episode_id: str,
    sampling: RecurrentSamplingConfig | None = None,
    tokenizer: Any | None = None,
    model_path: str | None = None,
    require_admission: bool = True,
) -> RecurrentPolicySample:
    """Sample one preidentified causal edge as a replay-complete GRPO member."""

    resolved = sampling or RecurrentSamplingConfig()
    pair = sample_final_recurrent_transition_pair(
        model,
        prompt_tokens,
        spec=spec,
        branch_index=branch_index,
        seed=seed,
        sampling=resolved,
        require_admission=False,
        episode_id=episode_id,
        tokenizer=tokenizer,
        model_path=model_path,
    )
    return recurrent_policy_sample_from_causal_pair(
        pair,
        sampling=resolved,
        require_admission=require_admission,
    )


def validate_recurrent_policy_sample_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct a v4 behavior sample, including its full replay payload."""

    required = {
        "schema",
        "sample_kind",
        "episode_id",
        "rng_root_sha256",
        "execution_spec_sha256",
        "policy_sha256",
        "prompt_tokens_sha256",
        "seed",
        "sampling_config",
        "token_count",
        "tokens",
        "tokens_sha256",
        "behavior_logprobs",
        "behavior_logprobs_sha256",
        "differentiable_logprobs",
        "differentiable_logprobs_sha256",
        "branch_index",
        "max_abs_logprob_drift",
        "mean_abs_logprob_drift",
        "max_abs_logprob_drift_token_index",
        "clipped_token_fraction",
        "old_policy_approx_kl",
        "behavior_admitted",
        "episode_receipt",
        "episode_receipt_sha256",
        "causal_transition_pair",
        "cached_decode_termination",
        "cached_params_unchanged",
        "cached_runtime_integrity",
        "cached_nonparametric_memory_status",
        "cached_recurrence_adapter",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise ValueError("recurrent_policy_sample_schema_invalid")
    normalized = dict(receipt)
    episode_id = normalized.get("episode_id")
    sample_kind = normalized.get("sample_kind")
    if (
        normalized.get("schema") != RECURRENT_SAMPLING_SCHEMA
        or sample_kind not in {"engine_episode", "causal_final_transition"}
        or not isinstance(episode_id, str)
        or not episode_id
        or len(episode_id) > 192
        or not _valid_sha256(normalized.get("rng_root_sha256"))
        or type(normalized.get("seed")) is not int
        or not 0 <= normalized["seed"] <= 0xFFFFFFFF
        or type(normalized.get("branch_index")) is not int
        or normalized["branch_index"] < 0
        or not all(
            _valid_sha256(normalized.get(field))
            for field in (
                "execution_spec_sha256",
                "policy_sha256",
                "prompt_tokens_sha256",
            )
        )
    ):
        raise ValueError("recurrent_policy_sample_identity_invalid")
    try:
        sampling = RecurrentSamplingConfig(**dict(normalized["sampling_config"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("recurrent_policy_sample_sampling_config_invalid") from exc
    tokens = normalized.get("tokens")
    behavior = normalized.get("behavior_logprobs")
    differentiable = normalized.get("differentiable_logprobs")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or token < 0 for token in tokens)
        or normalized.get("token_count") != len(tokens)
        or len(tokens) != sampling.max_tokens
        or normalized.get("tokens_sha256") != _tokens_sha256(tokens)
        or not isinstance(behavior, list)
        or not isinstance(differentiable, list)
        or len(behavior) != len(tokens)
        or len(differentiable) != len(tokens)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (*behavior, *differentiable)
        )
        or normalized.get("behavior_logprobs_sha256") != _float_sequence_sha256(behavior)
        or normalized.get("differentiable_logprobs_sha256")
        != _float_sequence_sha256(differentiable)
    ):
        raise ValueError("recurrent_policy_sample_trace_invalid")
    differences = [
        abs(float(left) - float(right))
        for left, right in zip(behavior, differentiable, strict=True)
    ]
    maximum = max(differences)
    mean = sum(differences) / len(differences)
    ratios = [
        math.exp(float(target) - float(old))
        for old, target in zip(behavior, differentiable, strict=True)
    ]
    clipped = sum(abs(ratio - 1.0) > float(sampling.clip_epsilon) for ratio in ratios) / len(ratios)
    old_kl = sum((ratio - 1.0) - math.log(ratio) for ratio in ratios) / len(ratios)
    if (
        not math.isclose(
            float(normalized.get("max_abs_logprob_drift")),
            maximum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(normalized.get("mean_abs_logprob_drift")),
            mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or normalized.get("max_abs_logprob_drift_token_index") != differences.index(maximum)
        or not math.isclose(
            float(normalized.get("clipped_token_fraction")),
            clipped,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(normalized.get("old_policy_approx_kl")),
            old_kl,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or type(normalized.get("behavior_admitted")) is not bool
    ):
        raise ValueError("recurrent_policy_sample_admission_metrics_invalid")
    episode = normalized.get("episode_receipt")
    if (
        not isinstance(episode, Mapping)
        or episode.get("episode_id") != episode_id
        or episode.get("input_tokens_sha256") != normalized["prompt_tokens_sha256"]
        or normalized.get("episode_receipt_sha256") != _seal_receipt(episode)
        or normalized.get("cached_decode_termination") != episode.get("decode_termination")
        or normalized.get("cached_params_unchanged") != episode.get("params_unchanged")
        or normalized.get("cached_runtime_integrity") != episode.get("runtime_integrity")
        or normalized.get("cached_nonparametric_memory_status")
        != episode.get("nonparametric_memory", {}).get("status")
        or normalized.get("cached_recurrence_adapter") != episode.get("recurrence_adapter")
    ):
        raise ValueError("recurrent_policy_sample_episode_binding_invalid")
    from core.brain.llm.latent_cortex.runtime_integrity import (
        runtime_integrity_safe,
    )

    runtime_safe = runtime_integrity_safe(
        normalized["cached_runtime_integrity"],
        require_worker=False,
        expected_episode_id=episode_id,
        expected_input_tokens_sha256=normalized["prompt_tokens_sha256"],
    )
    activation = normalized["cached_recurrence_adapter"]
    activation_safe = bool(
        isinstance(activation, Mapping)
        and activation.get("schema") == "aura.recurrence_adapter_activation.v1"
        and activation.get("scope") == "latent_slots_only"
        and activation.get("active") is True
        and type(activation.get("calls")) is int
        and activation["calls"] > 0
        and type(activation.get("adapted_positions")) is int
        and activation["adapted_positions"] > 0
    )
    expected_admitted = bool(
        maximum <= float(sampling.max_abs_logprob_drift)
        and mean <= float(sampling.max_mean_abs_logprob_drift)
        and clipped <= float(sampling.max_clipped_token_fraction)
        and old_kl <= float(sampling.max_old_policy_approx_kl)
        and runtime_safe
        and activation_safe
        and normalized["cached_nonparametric_memory_status"] == "disabled_by_policy"
        and not any(str(flag).startswith("fallback_") for flag in episode.get("honest_flags", []))
    )
    if normalized["behavior_admitted"] is not expected_admitted:
        raise ValueError("recurrent_policy_sample_admission_verdict_invalid")
    if normalized["rng_root_sha256"] != recurrent_sampling_rng_root_sha256(
        episode_id=episode_id,
        prompt_tokens_sha256=normalized["prompt_tokens_sha256"],
        policy_sha256=normalized["policy_sha256"],
        execution_spec_sha256=normalized["execution_spec_sha256"],
        branch_index=normalized["branch_index"],
        seed=normalized["seed"],
        sampling_config=sampling,
    ):
        raise ValueError("recurrent_policy_sample_rng_root_invalid")
    causal = normalized.get("causal_transition_pair")
    if sample_kind == "causal_final_transition":
        pair = validate_causal_recurrent_transition_pair_receipt(causal)
        child = pair["child"]
        if (
            pair["episode_id"] != episode_id
            or pair["policy_sha256"] != normalized["policy_sha256"]
            or pair["transition"]["execution_spec_sha256"] != normalized["execution_spec_sha256"]
            or pair["transition"]["prompt_tokens_sha256"] != normalized["prompt_tokens_sha256"]
            or child["branch_index"] != normalized["branch_index"]
            or child["seed"] != normalized["seed"]
            or child["tokens"] != tokens
            or child["behavior_logprobs"] != behavior
            or pair["child_differentiable_logprobs"] != differentiable
        ):
            raise ValueError("recurrent_policy_sample_causal_pair_mismatch")
    elif causal is not None:
        raise ValueError("recurrent_policy_sample_unexpected_causal_pair")
    return normalized


def recurrent_policy_sample_from_receipt(
    receipt: Mapping[str, Any],
) -> RecurrentPolicySample:
    """Reconstruct a numeric policy sample with an exact receipt round trip."""

    normalized = validate_recurrent_policy_sample_receipt(receipt)
    sample = RecurrentPolicySample(
        tokens=tuple(normalized["tokens"]),
        branch_index=normalized["branch_index"],
        behavior_logprobs=tuple(normalized["behavior_logprobs"]),
        differentiable_logprobs=tuple(normalized["differentiable_logprobs"]),
        max_abs_logprob_drift=normalized["max_abs_logprob_drift"],
        mean_abs_logprob_drift=normalized["mean_abs_logprob_drift"],
        max_abs_logprob_drift_token_index=normalized["max_abs_logprob_drift_token_index"],
        clipped_token_fraction=normalized["clipped_token_fraction"],
        old_policy_approx_kl=normalized["old_policy_approx_kl"],
        behavior_admitted=normalized["behavior_admitted"],
        execution_spec_sha256=normalized["execution_spec_sha256"],
        policy_sha256=normalized["policy_sha256"],
        prompt_tokens_sha256=normalized["prompt_tokens_sha256"],
        seed=normalized["seed"],
        sampling_config=RecurrentSamplingConfig(**dict(normalized["sampling_config"])),
        episode_receipt=dict(normalized["episode_receipt"]),
        episode_id=normalized["episode_id"],
        rng_root_sha256=normalized["rng_root_sha256"],
        sample_kind=normalized["sample_kind"],
        causal_transition_pair=(
            dict(normalized["causal_transition_pair"])
            if normalized["causal_transition_pair"] is not None
            else None
        ),
    )
    if sample.receipt() != normalized:
        raise ValueError("recurrent_policy_sample_round_trip_mismatch")
    return sample


def cortex_config_from_execution_spec(
    spec: RLCExecutionSpec,
    *,
    sampling: RecurrentSamplingConfig | None = None,
) -> Any:
    """Construct the fixed live-engine graph named by an execution spec."""

    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    problems = spec.validate()
    if problems:
        raise ValueError(f"invalid execution spec: {problems}")
    if spec.decode_bridge_policy != "none":
        raise ValueError("recurrent sampling v1 supports decode_bridge_policy=none only")
    resolved = sampling or RecurrentSamplingConfig()
    config = CortexConfig(
        workspace=WorkspaceConfig(
            n_slots=spec.n_slots,
            seed=spec.slot_seed,
            roles=spec.slot_roles,
            anchor_scale=spec.anchor_scale,
        ),
        recurrence=RecurrenceConfig(
            max_steps=spec.recurrent_steps,
            min_steps=spec.recurrent_steps,
            alpha=spec.alpha,
            alpha_schedule=spec.alpha_schedule,
            rms_clip_ratio=spec.rms_clip_ratio,
            convergence_eps=1e-9,
            divergence_ratio=1000.0,
            fixed_depth=True,
        ),
        branches=BranchConfig(
            n_branches=len(spec.branch_roles),
            exchange_interval=spec.exchange_interval,
            exchange_gamma=spec.exchange_gamma,
            comm_slot=spec.comm_slot,
            collapse_cos_threshold=spec.collapse_cos_threshold,
            jitter_scale=spec.jitter_scale,
            roles=spec.branch_roles,
        ),
        prelude_frac=spec.prelude_frac,
        coda_frac=spec.coda_frac,
        decode_max_tokens=resolved.max_tokens,
        decode_temperature=resolved.temperature,
        decode_top_p=resolved.top_p,
        decode_bridge_policy="none",
        # Scientific execution must measure the recurrent policy itself. The
        # resident worker's safer default preserves vanilla output instead.
        decode_incumbent_policy="latent",
        decode_contract="none",
        decode_min_tokens=0,
        decode_repetition_penalty=1.0,
        # Research sampling must retain refuted policy outputs as negative
        # examples. Live serving keeps confidence-bound replacement enabled.
        answer_replacement_enabled=False,
        allow_vanilla_fallback=False,
        escape={"enabled": False},
        telemetry_enabled=False,
        probe_cache_enabled=False,
    )
    config_problems = config.validate()
    if config_problems:
        raise ValueError(f"execution spec produced invalid CortexConfig: {config_problems}")
    return config


def sample_recurrent_completion(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    seed: int,
    sampling: RecurrentSamplingConfig | None = None,
    tokenizer: Any | None = None,
    model_path: str | None = None,
    require_admission: bool = True,
    episode_id: str | None = None,
) -> RecurrentPolicySample:
    """Sample through cached RLC, then admit it as a bounded behavior policy."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    resolved = sampling or RecurrentSamplingConfig()
    prompt = tuple(prompt_tokens)
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("prompt_tokens must contain non-negative integers")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    policy_sha256 = recurrent_policy_sha256(model, spec)
    engine = LatentCortexEngine(
        model,
        tokenizer=tokenizer,
        config=cortex_config_from_execution_spec(spec, sampling=resolved),
        model_path=model_path,
        schedule_library=None,
    )
    result = engine.reason(
        token_ids=list(prompt),
        decode_max_tokens=resolved.max_tokens,
        capture_decode_logprobs=True,
        decode_sentence_grace_tokens=0,
        nonparametric_memory_enabled=False,
        sample_seed=seed,
        episode_id=episode_id,
    )
    scored_policy_termination = bool(
        not result.ok and result.reason.startswith("decode_incomplete:")
    )
    if not result.ok and not scored_policy_termination:
        raise RuntimeError(f"cached recurrent sampling failed: {result.reason}")
    if not result.tokens:
        raise RuntimeError("cached recurrent sampling produced no completion tokens")
    if len(result.tokens) != len(result.decode_token_logprobs):
        raise RuntimeError("cached recurrent sampling logprobs do not align with tokens")
    branch_index = int(result.receipt.selected_branch)
    differentiable = recurrent_completion_token_logprobs(
        model,
        prompt,
        result.tokens,
        spec=spec,
        branch_index=branch_index,
    )
    mx.eval(differentiable)
    if recurrent_policy_sha256(model, spec) != policy_sha256:
        raise RuntimeError("recurrent policy changed during cached sampling")
    target = tuple(float(value) for value in differentiable)
    behavior = tuple(float(value) for value in result.decode_token_logprobs)
    if any(not math.isfinite(value) for value in (*target, *behavior)):
        raise FloatingPointError("recurrent sampling produced non-finite logprobs")
    differences = tuple(abs(left - right) for left, right in zip(behavior, target, strict=True))
    maximum = max(differences)
    maximum_index = differences.index(maximum)
    mean = sum(differences) / len(differences)
    ratios = tuple(
        math.exp(target_value - behavior_value)
        for target_value, behavior_value in zip(target, behavior, strict=True)
    )
    clipped_fraction = sum(
        abs(ratio - 1.0) > float(resolved.clip_epsilon) for ratio in ratios
    ) / len(ratios)
    old_policy_approx_kl = sum((ratio - 1.0) - math.log(ratio) for ratio in ratios) / len(ratios)
    from core.brain.llm.latent_cortex.runtime_integrity import (
        runtime_integrity_safe,
    )

    measured_runtime_safe = runtime_integrity_safe(
        result.receipt.runtime_integrity,
        require_worker=False,
        expected_episode_id=result.receipt.episode_id,
        expected_input_tokens_sha256=result.receipt.input_tokens_sha256,
        expected_fast_weights_applied=result.receipt.fast_weights_applied,
        expected_fast_weights_attach_attempted=(
            result.receipt.fast_weights_attach_attempted
        ),
        expected_checkpoint_fingerprint=(result.receipt.checkpoint_fingerprint),
        expected_checkpoint_method=(result.receipt.checkpoint_fingerprint_method),
        expected_checkpoint_file_count=(result.receipt.checkpoint_file_count),
    )
    activation = result.receipt.recurrence_adapter
    activation_safe = bool(
        isinstance(activation, Mapping)
        and activation.get("schema") == "aura.recurrence_adapter_activation.v1"
        and activation.get("scope") == "latent_slots_only"
        and activation.get("active") is True
        and type(activation.get("calls")) is int
        and activation["calls"] > 0
        and type(activation.get("adapted_positions")) is int
        and activation["adapted_positions"] > 0
    )
    admitted = (
        maximum <= float(resolved.max_abs_logprob_drift)
        and mean <= float(resolved.max_mean_abs_logprob_drift)
        and clipped_fraction <= float(resolved.max_clipped_token_fraction)
        and old_policy_approx_kl <= float(resolved.max_old_policy_approx_kl)
        and measured_runtime_safe
        and activation_safe
        and result.receipt.nonparametric_memory.get("status") == "disabled_by_policy"
        and not any(flag.startswith("fallback_") for flag in result.receipt.honest_flags)
    )
    sample = RecurrentPolicySample(
        tokens=tuple(result.tokens),
        branch_index=branch_index,
        behavior_logprobs=behavior,
        differentiable_logprobs=target,
        max_abs_logprob_drift=maximum,
        mean_abs_logprob_drift=mean,
        max_abs_logprob_drift_token_index=maximum_index,
        clipped_token_fraction=clipped_fraction,
        old_policy_approx_kl=old_policy_approx_kl,
        behavior_admitted=admitted,
        execution_spec_sha256=spec.sha256,
        policy_sha256=policy_sha256,
        prompt_tokens_sha256=_tokens_sha256(prompt),
        seed=seed,
        sampling_config=resolved,
        episode_receipt=result.receipt.to_dict(),
        episode_id=result.receipt.episode_id,
        rng_root_sha256=recurrent_sampling_rng_root_sha256(
            episode_id=result.receipt.episode_id,
            prompt_tokens_sha256=result.receipt.input_tokens_sha256,
            policy_sha256=policy_sha256,
            execution_spec_sha256=spec.sha256,
            branch_index=branch_index,
            seed=seed,
            sampling_config=resolved,
        ),
        sample_kind="engine_episode",
        causal_transition_pair=None,
    )
    if require_admission and not admitted:
        raise RecurrentSamplingAdmissionError(sample)
    return sample


@dataclass(frozen=True, slots=True)
class RecurrentGRPOConfig:
    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.04
    advantage_clip: float = 4.0
    max_initial_clip_fraction: float = 0.25
    max_initial_old_policy_approx_kl: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (
            ("clip_epsilon", self.clip_epsilon),
            ("kl_coefficient", self.kl_coefficient),
            ("advantage_clip", self.advantage_clip),
            ("max_initial_clip_fraction", self.max_initial_clip_fraction),
            (
                "max_initial_old_policy_approx_kl",
                self.max_initial_old_policy_approx_kl,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < float(self.clip_epsilon) <= 1.0:
            raise ValueError("clip_epsilon must be inside (0, 1]")
        if not 0.0 <= float(self.kl_coefficient) <= 100.0:
            raise ValueError("kl_coefficient must be inside [0, 100]")
        if not 0.0 < float(self.advantage_clip) <= 100.0:
            raise ValueError("advantage_clip must be inside (0, 100]")
        if not 0.0 <= float(self.max_initial_clip_fraction) <= 1.0:
            raise ValueError("max_initial_clip_fraction must be inside [0, 1]")
        if not 0.0 <= float(self.max_initial_old_policy_approx_kl) <= 100.0:
            raise ValueError("max_initial_old_policy_approx_kl must be inside [0, 100]")


@dataclass(frozen=True, slots=True)
class VerifiedTrajectoryGroupConfig:
    """Policy for composing verified outcome and recurrent-trajectory gradients.

    Improvement terms are allocated only to completions with positive
    verifier-derived advantage. Displacement, oscillation, and diversity are
    prompt-level structural terms and therefore execute once over the complete
    exchange-coupled branch ensemble instead of once per sampled completion.
    """

    trajectory_config: ExactAdjointTrajectoryConfig | None = None
    intervention_config: ExactAdjointInterventionConfig | None = None
    diversity_weight: float = 0.0
    diversity_target_cos: float = 0.98

    def __post_init__(self) -> None:
        if self.trajectory_config is not None and not isinstance(
            self.trajectory_config, ExactAdjointTrajectoryConfig
        ):
            raise TypeError("trajectory_config must be an ExactAdjointTrajectoryConfig")
        if self.intervention_config is not None and not isinstance(
            self.intervention_config, ExactAdjointInterventionConfig
        ):
            raise TypeError("intervention_config must be an ExactAdjointInterventionConfig")
        for name, value, high in (
            ("diversity_weight", self.diversity_weight, 10.0),
            ("diversity_target_cos", self.diversity_target_cos, 1.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= high
            ):
                raise ValueError(f"{name} must be finite inside [0, {high:g}]")
        if (
            self.trajectory_config is None
            and self.intervention_config is None
            and float(self.diversity_weight) == 0.0
        ):
            raise ValueError(
                "verified trajectory group must enable a trajectory, intervention, "
                "or diversity term"
            )

    def to_dict(self) -> dict[str, Any]:
        if self.intervention_config is not None:
            return {
                "schema": VERIFIED_TRAJECTORY_GROUP_SCHEMA_V2,
                "trajectory_config": (
                    self.trajectory_config.to_dict() if self.trajectory_config is not None else None
                ),
                "intervention_config": self.intervention_config.to_dict(),
                "diversity_weight": float(self.diversity_weight),
                "diversity_target_cos": float(self.diversity_target_cos),
                "quality_credit": "positive_advantage_l1_normalized",
                "structural_scope": "all_exchange_coupled_branches_once_per_prompt",
                "anchor_selection": "maximum_verified_reward_then_lowest_index",
                "causal_lesion_policy": (
                    "skip_exchange_coupled_transition_detached_lesion_baseline"
                ),
                "stopping_policy": "detached_cost_aware_soft_optimal_teacher",
                "measurement_trust_boundary": (INTERVENTION_MEASUREMENT_TRUST_BOUNDARY),
            }
        return {
            "schema": VERIFIED_TRAJECTORY_GROUP_SCHEMA,
            "trajectory_config": (
                self.trajectory_config.to_dict() if self.trajectory_config is not None else None
            ),
            "diversity_weight": float(self.diversity_weight),
            "diversity_target_cos": float(self.diversity_target_cos),
            "improvement_credit": "positive_advantage_l1_normalized",
            "structural_scope": "all_exchange_coupled_branches_once_per_prompt",
            "anchor_selection": "maximum_verified_reward_then_lowest_index",
        }

    @classmethod
    def from_dict(cls, value: Any) -> VerifiedTrajectoryGroupConfig:
        v1_required = {
            "schema",
            "trajectory_config",
            "diversity_weight",
            "diversity_target_cos",
            "improvement_credit",
            "structural_scope",
            "anchor_selection",
        }
        v2_required = {
            "schema",
            "trajectory_config",
            "intervention_config",
            "diversity_weight",
            "diversity_target_cos",
            "quality_credit",
            "structural_scope",
            "anchor_selection",
            "causal_lesion_policy",
            "stopping_policy",
            "measurement_trust_boundary",
        }
        if not isinstance(value, Mapping) or frozenset(value) not in {
            frozenset(v1_required),
            frozenset(v2_required),
        }:
            raise ValueError("verified trajectory group config fields do not match")
        is_v2 = value.get("schema") == VERIFIED_TRAJECTORY_GROUP_SCHEMA_V2
        if is_v2:
            valid_policy = bool(
                set(value) == v2_required
                and isinstance(value.get("intervention_config"), Mapping)
                and value.get("quality_credit") == "positive_advantage_l1_normalized"
                and value.get("causal_lesion_policy")
                == "skip_exchange_coupled_transition_detached_lesion_baseline"
                and value.get("stopping_policy") == "detached_cost_aware_soft_optimal_teacher"
                and value.get("measurement_trust_boundary")
                == INTERVENTION_MEASUREMENT_TRUST_BOUNDARY
            )
        else:
            valid_policy = bool(
                value.get("schema") == VERIFIED_TRAJECTORY_GROUP_SCHEMA
                and set(value) == v1_required
                and value.get("improvement_credit") == "positive_advantage_l1_normalized"
            )
        if (
            not valid_policy
            or value.get("structural_scope") != "all_exchange_coupled_branches_once_per_prompt"
            or value.get("anchor_selection") != "maximum_verified_reward_then_lowest_index"
        ):
            raise ValueError("verified trajectory group policy is unsupported")
        trajectory = value.get("trajectory_config")
        intervention = value.get("intervention_config") if is_v2 else None
        return cls(
            trajectory_config=(
                ExactAdjointTrajectoryConfig.from_dict(trajectory)
                if trajectory is not None
                else None
            ),
            intervention_config=(
                ExactAdjointInterventionConfig.from_dict(intervention)
                if intervention is not None
                else None
            ),
            diversity_weight=value["diversity_weight"],
            diversity_target_cos=value["diversity_target_cos"],
        )


@dataclass(frozen=True, slots=True)
class RecurrentGRPOObjective:
    loss: Any
    policy_loss: Any
    reference_kl: Any
    old_policy_approx_kl: Any
    clip_fraction: Any
    completion_count: int
    token_count: int

    def receipt(self) -> dict[str, Any]:
        import mlx.core as mx

        mx.eval(
            self.loss,
            self.policy_loss,
            self.reference_kl,
            self.old_policy_approx_kl,
            self.clip_fraction,
        )
        return {
            "schema": RECURRENT_GRPO_SCHEMA,
            "loss": float(self.loss),
            "policy_loss": float(self.policy_loss),
            "reference_kl": float(self.reference_kl),
            "old_policy_approx_kl": float(self.old_policy_approx_kl),
            "clip_fraction": float(self.clip_fraction),
            "completion_count": self.completion_count,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class ExactAdjointRecurrentGRPOResult:
    gradients: Any | None
    advantage_report: dict[str, Any]
    reference_kl: float
    old_policy_approx_kl: float
    clip_fraction: float
    policy_loss: float
    objective_at_sampling: float
    gradient_surrogate_value: float
    completion_count: int
    token_count: int
    branch_indices: tuple[int, ...]
    trajectory_objective_value: float = 0.0
    trajectory_receipt: dict[str, Any] | None = None

    def receipt(self) -> dict[str, Any]:
        receipt = {
            "schema": RECURRENT_GRPO_SCHEMA,
            "mode": "exact_adjoint_single_update",
            "advantage_report": self.advantage_report,
            "reference_kl": self.reference_kl,
            "old_policy_approx_kl": self.old_policy_approx_kl,
            "clip_fraction": self.clip_fraction,
            "policy_loss": self.policy_loss,
            "objective_at_sampling": self.objective_at_sampling,
            "gradient_surrogate_value": self.gradient_surrogate_value,
            "completion_count": self.completion_count,
            "token_count": self.token_count,
            "branch_indices": list(self.branch_indices),
            "has_gradient": self.gradients is not None,
        }
        if self.trajectory_receipt is None:
            if float(self.trajectory_objective_value) != 0.0:
                raise ValueError("trajectory objective value requires a trajectory receipt")
            return receipt
        trajectory_value = float(self.trajectory_objective_value)
        if not math.isfinite(trajectory_value):
            raise ValueError("trajectory objective value must be finite")
        return {
            **receipt,
            "mode": "exact_adjoint_trajectory_composite_single_update",
            "trajectory_objective_value": trajectory_value,
            "composite_objective_at_sampling": (
                float(self.objective_at_sampling) + trajectory_value
            ),
            "composite_gradient_surrogate_value": (
                float(self.gradient_surrogate_value) + trajectory_value
            ),
            "trajectory_receipt": dict(self.trajectory_receipt),
        }


def build_verified_trajectory_group_source_binding(
    group_admission_receipt: Mapping[str, Any],
    reward_receipt: Mapping[str, Any],
    samples: Sequence[RecurrentPolicySample],
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    trajectory_group_config: VerifiedTrajectoryGroupConfig,
    advantage_clip: float,
    optimization_token_counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Derive the trajectory source binding from independently admitted inputs."""

    from core.learning.verified_transition_reward import rewards_for_recurrent_samples

    if not isinstance(group_admission_receipt, Mapping) or not isinstance(reward_receipt, Mapping):
        raise ValueError("verified trajectory source receipts are invalid")
    if not isinstance(trajectory_group_config, VerifiedTrajectoryGroupConfig):
        raise TypeError("trajectory_group_config must be a VerifiedTrajectoryGroupConfig")
    clip = float(advantage_clip)
    if not math.isfinite(clip) or not 0.0 < clip <= 100.0:
        raise ValueError("verified trajectory advantage clip is invalid")
    branch_count = len(spec.branch_roles)
    expected_branches = tuple(range(branch_count))
    if (
        branch_count < 2
        or len(samples) != branch_count
        or any(not isinstance(sample, RecurrentPolicySample) for sample in samples)
        or tuple(sample.branch_index for sample in samples) != expected_branches
    ):
        raise ValueError("verified trajectory requires one canonical sample per execution branch")
    prompt_sha256 = _tokens_sha256(prompt_tokens)
    sample_receipts = [_seal_receipt(sample.receipt()) for sample in samples]
    full_token_counts = tuple(len(sample.tokens) for sample in samples)
    bounded_token_counts = (
        full_token_counts
        if optimization_token_counts is None
        else tuple(optimization_token_counts)
    )
    if (
        len(bounded_token_counts) != branch_count
        or any(
            type(count) is not int or not 1 <= count <= full_count
            for count, full_count in zip(
                bounded_token_counts,
                full_token_counts,
                strict=True,
            )
        )
    ):
        raise ValueError("verified trajectory optimization boundaries are invalid")
    prefix_bounded = bounded_token_counts != full_token_counts
    completion_receipts = [
        _tokens_sha256(sample.tokens[:count])
        for sample, count in zip(samples, bounded_token_counts, strict=True)
    ]
    policies = {sample.policy_sha256 for sample in samples}
    if len(policies) != 1:
        raise ValueError("verified trajectory source mixes policy identities")
    policy_sha256 = next(iter(policies))
    rewards = tuple(
        float(value)
        for value in rewards_for_recurrent_samples(
            reward_receipt,
            samples,
            prompt_tokens,
        )
    )
    admission_sha256 = group_admission_receipt.get("receipt_sha256")
    reward_sha256 = reward_receipt.get("receipt_sha256")
    admission_bindings = group_admission_receipt.get("sample_bindings")
    if (
        not _valid_sha256(admission_sha256)
        or not _valid_sha256(reward_sha256)
        or group_admission_receipt.get("reward_receipt_sha256") != reward_sha256
        or group_admission_receipt.get("policy_sha256") != policy_sha256
        or group_admission_receipt.get("recurrent_execution_spec_sha256") != spec.sha256
        or group_admission_receipt.get("prompt_tokens_sha256") != prompt_sha256
        or group_admission_receipt.get("group_size") != branch_count
        or not isinstance(admission_bindings, list)
        or [binding.get("sample_sha256") for binding in admission_bindings] != sample_receipts
        or any(sample.execution_spec_sha256 != spec.sha256 for sample in samples)
        or any(sample.prompt_tokens_sha256 != prompt_sha256 for sample in samples)
    ):
        raise ValueError("verified trajectory source differs from group admission")
    if prefix_bounded:
        source_schema = VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3
    elif trajectory_group_config.intervention_config is not None:
        source_schema = VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V2
    else:
        source_schema = VERIFIED_TRAJECTORY_SOURCE_SCHEMA
    payload = {
        "schema": source_schema,
        "group_admission_sha256": admission_sha256,
        "reward_receipt_sha256": reward_sha256,
        "policy_sha256": policy_sha256,
        "execution_spec_sha256": spec.sha256,
        "prompt_tokens_sha256": prompt_sha256,
        "sample_receipt_sha256s": sample_receipts,
        "completion_tokens_sha256s": completion_receipts,
        "sample_branch_indices": list(expected_branches),
        "execution_branch_count": branch_count,
        "verified_rewards": list(rewards),
        "advantage_clip": clip,
        "config": trajectory_group_config.to_dict(),
    }
    if prefix_bounded:
        payload["optimization_token_counts"] = list(bounded_token_counts)
    return {**payload, "source_sha256": _seal_receipt(payload)}


def validate_verified_trajectory_group_source_binding(value: Any) -> dict[str, Any]:
    """Validate model-independent inputs sealed beside a trajectory objective."""

    if not isinstance(value, Mapping):
        raise ValueError("verified trajectory source binding fields do not match")
    schema = value.get("schema")
    prefix_bounded = schema == VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3
    required = {
        "schema",
        "group_admission_sha256",
        "reward_receipt_sha256",
        "policy_sha256",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "sample_receipt_sha256s",
        "completion_tokens_sha256s",
        "sample_branch_indices",
        "execution_branch_count",
        "verified_rewards",
        "advantage_clip",
        "config",
        "source_sha256",
    }
    if prefix_bounded:
        required.add("optimization_token_counts")
    if set(value) != required:
        raise ValueError("verified trajectory source binding fields do not match")
    binding = dict(value)
    observed_source_sha256 = binding.pop("source_sha256")
    if not _valid_sha256(observed_source_sha256) or observed_source_sha256 != _seal_receipt(
        binding
    ):
        raise ValueError("verified trajectory source binding commitment mismatch")
    if binding["schema"] not in {
        VERIFIED_TRAJECTORY_SOURCE_SCHEMA,
        VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V2,
        VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3,
    }:
        raise ValueError("verified trajectory source binding schema is unsupported")
    for role in (
        "group_admission_sha256",
        "reward_receipt_sha256",
        "policy_sha256",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
    ):
        if not _valid_sha256(binding[role]):
            raise ValueError(f"verified trajectory source {role} is invalid")
    sample_receipts = binding["sample_receipt_sha256s"]
    completion_receipts = binding["completion_tokens_sha256s"]
    branches = binding["sample_branch_indices"]
    branch_count = binding["execution_branch_count"]
    rewards = binding["verified_rewards"]
    clip = binding["advantage_clip"]
    optimization_token_counts = binding.get("optimization_token_counts")
    if (
        type(branch_count) is not int
        or branch_count < 2
        or not isinstance(sample_receipts, list)
        or len(sample_receipts) != branch_count
        or any(not _valid_sha256(item) for item in sample_receipts)
        or not isinstance(completion_receipts, list)
        or len(completion_receipts) != branch_count
        or any(not _valid_sha256(item) for item in completion_receipts)
        or not isinstance(branches, list)
        or branches != list(range(branch_count))
        or not isinstance(rewards, list)
        or len(rewards) != branch_count
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in rewards
        )
        or isinstance(clip, bool)
        or not isinstance(clip, (int, float))
        or not math.isfinite(float(clip))
        or not 0.0 < float(clip) <= 100.0
        or (
            prefix_bounded
            and (
                not isinstance(optimization_token_counts, list)
                or len(optimization_token_counts) != branch_count
                or any(
                    type(count) is not int or count < 1
                    for count in optimization_token_counts
                )
            )
        )
    ):
        raise ValueError("verified trajectory source binding values are invalid")
    config = VerifiedTrajectoryGroupConfig.from_dict(binding["config"])
    if prefix_bounded:
        expected_schema = VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3
    elif config.intervention_config is not None:
        expected_schema = VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V2
    else:
        expected_schema = VERIFIED_TRAJECTORY_SOURCE_SCHEMA
    if binding["schema"] != expected_schema:
        raise ValueError("verified trajectory source schema/config mismatch")
    return dict(value)


def validate_verified_trajectory_group_receipt(
    value: Any,
    *,
    advantage_report: Mapping[str, Any] | None = None,
    expected_source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay the policy and arithmetic of one verified trajectory composite."""

    schema = value.get("schema") if isinstance(value, Mapping) else None
    prefix_bounded = schema == VERIFIED_TRAJECTORY_GROUP_SCHEMA_V3
    raw_config = value.get("config") if isinstance(value, Mapping) else None
    intervention_group = bool(
        isinstance(raw_config, Mapping)
        and raw_config.get("intervention_config") is not None
    )
    required = {
        "schema",
        "group_admission_sha256",
        "reward_receipt_sha256",
        "policy_sha256",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "sample_receipt_sha256s",
        "completion_tokens_sha256s",
        "sample_branch_indices",
        "execution_branch_count",
        "verified_rewards",
        "advantage_clip",
        "advantages",
        "config",
        "positive_completion_indices",
        "positive_advantage_weights",
        "anchor_completion_index",
        "anchor_branch_index",
        ("quality_receipts" if intervention_group else "improvement_receipts"),
        "structural_receipt",
        "trajectory_objective_value",
        "receipt_sha256",
    }
    if prefix_bounded:
        required.add("optimization_token_counts")
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("verified trajectory group receipt fields do not match")
    receipt = dict(value)
    observed_seal = receipt.pop("receipt_sha256")
    if not _valid_sha256(observed_seal) or observed_seal != _seal_receipt(receipt):
        raise ValueError("verified trajectory group receipt commitment mismatch")
    if receipt["schema"] not in {
        VERIFIED_TRAJECTORY_GROUP_SCHEMA,
        VERIFIED_TRAJECTORY_GROUP_SCHEMA_V2,
        VERIFIED_TRAJECTORY_GROUP_SCHEMA_V3,
    }:
        raise ValueError("verified trajectory group receipt schema is unsupported")
    for role in (
        "group_admission_sha256",
        "reward_receipt_sha256",
        "policy_sha256",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
    ):
        if not _valid_sha256(receipt[role]):
            raise ValueError(f"verified trajectory {role} is invalid")

    def finite_number(item: Any, *, role: str) -> float:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"verified trajectory {role} is not finite")
        return float(item)

    sample_receipts = receipt["sample_receipt_sha256s"]
    token_receipts = receipt["completion_tokens_sha256s"]
    branches = receipt["sample_branch_indices"]
    rewards = receipt["verified_rewards"]
    advantages = receipt["advantages"]
    branch_count = receipt["execution_branch_count"]
    optimization_token_counts = receipt.get("optimization_token_counts")
    if (
        not isinstance(sample_receipts, list)
        or len(sample_receipts) < 2
        or any(not _valid_sha256(digest) for digest in sample_receipts)
        or not isinstance(token_receipts, list)
        or len(token_receipts) != len(sample_receipts)
        or any(not _valid_sha256(digest) for digest in token_receipts)
        or not isinstance(branches, list)
        or len(branches) != len(sample_receipts)
        or type(branch_count) is not int
        or branch_count < 2
        or branches != list(range(branch_count))
        or len(sample_receipts) != branch_count
        or not isinstance(rewards, list)
        or len(rewards) != len(sample_receipts)
        or (
            prefix_bounded
            and (
                not isinstance(optimization_token_counts, list)
                or len(optimization_token_counts) != len(sample_receipts)
                or any(
                    type(count) is not int or count < 1
                    for count in optimization_token_counts
                )
            )
        )
        or not isinstance(advantages, list)
        or len(advantages) != len(sample_receipts)
    ):
        raise ValueError("verified trajectory sample bindings are invalid")
    normalized_rewards = [finite_number(number, role="reward") for number in rewards]
    normalized_advantages = [finite_number(number, role="advantage") for number in advantages]
    advantage_clip = finite_number(receipt["advantage_clip"], role="advantage clip")
    if not 0.0 < advantage_clip <= 100.0:
        raise ValueError("verified trajectory advantage clip is invalid")
    replayed_advantages = group_advantages(
        normalized_rewards,
        clip=advantage_clip,
    )
    if any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
        for observed, expected in zip(
            normalized_advantages,
            replayed_advantages["advantages"],
            strict=True,
        )
    ):
        raise ValueError("verified trajectory advantages do not replay from rewards")
    if advantage_report is not None and dict(advantage_report) != replayed_advantages:
        raise ValueError("verified trajectory advantages differ from the policy objective")

    if expected_source_binding is not None:
        source = validate_verified_trajectory_group_source_binding(expected_source_binding)
        source_fields = [
            "group_admission_sha256",
            "reward_receipt_sha256",
            "policy_sha256",
            "execution_spec_sha256",
            "prompt_tokens_sha256",
            "sample_receipt_sha256s",
            "completion_tokens_sha256s",
            "sample_branch_indices",
            "execution_branch_count",
            "verified_rewards",
            "advantage_clip",
            "config",
        ]
        if source["schema"] == VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3:
            source_fields.append("optimization_token_counts")
        for field in source_fields:
            if receipt[field] != source[field]:
                raise ValueError(f"verified trajectory {field} differs from admitted source")

    config = VerifiedTrajectoryGroupConfig.from_dict(receipt["config"])
    if prefix_bounded:
        expected_group_schema = VERIFIED_TRAJECTORY_GROUP_SCHEMA_V3
    elif config.intervention_config is not None:
        expected_group_schema = VERIFIED_TRAJECTORY_GROUP_SCHEMA_V2
    else:
        expected_group_schema = VERIFIED_TRAJECTORY_GROUP_SCHEMA
    if (
        intervention_group is not (config.intervention_config is not None)
        or receipt["schema"] != expected_group_schema
    ):
        raise ValueError("verified trajectory group schema/config mismatch")
    trajectory = config.trajectory_config
    intervention = config.intervention_config
    improvement_enabled = bool(
        trajectory is not None and float(trajectory.improvement_weight) > 0.0
    )
    intervention_enabled = bool(
        intervention is not None
        and (
            float(intervention.causality_weight) > 0.0 or float(intervention.stopping_weight) > 0.0
        )
    )
    quality_enabled = improvement_enabled or intervention_enabled
    expected_positive = (
        [index for index, advantage in enumerate(normalized_advantages) if advantage > 0.0]
        if quality_enabled
        else []
    )
    positive = receipt["positive_completion_indices"]
    weights = receipt["positive_advantage_weights"]
    if (
        not isinstance(positive, list)
        or positive != expected_positive
        or not isinstance(weights, list)
        or len(weights) != len(positive)
    ):
        raise ValueError("verified trajectory positive credit assignment is invalid")
    normalized_weights = [
        finite_number(weight, role="positive advantage weight") for weight in weights
    ]
    if quality_enabled:
        denominator = sum(normalized_advantages[index] for index in expected_positive)
        if denominator <= 0.0:
            raise ValueError("verified trajectory has no positive advantage mass")
        expected_weights = [
            normalized_advantages[index] / denominator for index in expected_positive
        ]
        if any(
            weight <= 0.0
            or not math.isclose(
                weight,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for weight, expected in zip(
                normalized_weights,
                expected_weights,
                strict=True,
            )
        ) or not math.isclose(
            sum(normalized_weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("verified trajectory advantage weights do not replay")
    elif normalized_weights:
        raise ValueError("verified trajectory weights exist without a quality term")

    anchor = receipt["anchor_completion_index"]
    anchor_branch = receipt["anchor_branch_index"]
    expected_anchor = max(
        range(len(normalized_rewards)),
        key=lambda index: (normalized_rewards[index], -index),
    )
    if (
        type(anchor) is not int
        or anchor != expected_anchor
        or type(anchor_branch) is not int
        or anchor_branch != branches[anchor]
    ):
        raise ValueError("verified trajectory structural anchor is invalid")

    quality_receipts = receipt["quality_receipts" if intervention_group else "improvement_receipts"]
    if not isinstance(quality_receipts, list) or len(quality_receipts) != len(positive):
        raise ValueError("verified trajectory quality receipt count is invalid")
    replayed_total = 0.0
    for completion_index, weight, entry in zip(
        positive,
        normalized_weights,
        quality_receipts,
        strict=True,
    ):
        if not isinstance(entry, Mapping) or set(entry) != {
            "completion_index",
            "completion_tokens_sha256",
            "advantage_weight",
            "objective_receipt",
        }:
            raise ValueError("verified trajectory quality receipt is invalid")
        if (
            entry["completion_index"] != completion_index
            or entry["completion_tokens_sha256"] != token_receipts[completion_index]
            or not math.isclose(
                finite_number(
                    entry["advantage_weight"],
                    role="quality advantage weight",
                ),
                weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("verified trajectory quality binding differs")
        objective = validate_exact_adjoint_live_path_receipt(entry["objective_receipt"])
        expected_config = (
            ExactAdjointTrajectoryConfig(
                probe_steps=trajectory.probe_steps,
                improvement_weight=(float(trajectory.improvement_weight) * weight),
                improvement_margin=trajectory.improvement_margin,
                displacement_weight=0.0,
                displacement_floor=trajectory.displacement_floor,
                oscillation_weight=0.0,
            )
            if improvement_enabled
            else None
        )
        expected_intervention = (
            ExactAdjointInterventionConfig(
                lesion_steps=intervention.lesion_steps,
                causality_weight=(float(intervention.causality_weight) * weight),
                causality_margin=intervention.causality_margin,
                stopping_steps=intervention.stopping_steps,
                stopping_weight=(float(intervention.stopping_weight) * weight),
                stopping_ponder_cost=intervention.stopping_ponder_cost,
                stopping_temperature=intervention.stopping_temperature,
            )
            if intervention_enabled
            else None
        )
        if (
            objective["policy_sha256"] != receipt["policy_sha256"]
            or objective["prompt_tokens_sha256"] != receipt["prompt_tokens_sha256"]
            or objective["answer_tokens_sha256"] != token_receipts[completion_index]
            or objective["bridge_tokens_sha256"] != _tokens_sha256(())
            or objective["bridge_token_count"] != 0
            or any(float(weight) != 0.0 for weight in objective["token_loss_weights"])
            or objective["execution_spec_sha256"] != receipt["execution_spec_sha256"]
            or objective["execution_branch_count"] != branch_count
            or objective["branch_indices"] != [branches[completion_index]]
            or objective["trajectory_config"]
            != (expected_config.to_dict() if expected_config is not None else None)
            or (
                objective.get("intervention_config")
                != (expected_intervention.to_dict() if expected_intervention is not None else None)
            )
            or float(objective["diversity_weight"]) != 0.0
            or not math.isclose(
                float(objective["terminal_value"]),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(objective["diversity_value"]),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or any(
                not math.isclose(
                    float(objective["trajectory_values"][name]),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for name in ("displacement", "oscillation")
            )
        ):
            objective_label = "quality" if intervention_group else "improvement"
            raise ValueError(f"verified trajectory {objective_label} objective differs")
        replayed_total += float(objective["value"])

    structural_enabled = bool(
        float(config.diversity_weight) > 0.0
        or (
            trajectory is not None
            and (
                float(trajectory.displacement_weight) > 0.0
                or float(trajectory.oscillation_weight) > 0.0
            )
        )
    )
    structural_entry = receipt["structural_receipt"]
    if structural_enabled:
        if not isinstance(structural_entry, Mapping) or set(structural_entry) != {
            "anchor_completion_index",
            "anchor_completion_tokens_sha256",
            "objective_receipt",
        }:
            raise ValueError("verified trajectory structural receipt is invalid")
        if (
            structural_entry["anchor_completion_index"] != anchor
            or structural_entry["anchor_completion_tokens_sha256"] != token_receipts[anchor]
        ):
            raise ValueError("verified trajectory structural receipt is misbound")
        objective = validate_exact_adjoint_live_path_receipt(structural_entry["objective_receipt"])
        structural_config = (
            ExactAdjointTrajectoryConfig(
                probe_steps=trajectory.probe_steps,
                improvement_weight=0.0,
                improvement_margin=trajectory.improvement_margin,
                displacement_weight=trajectory.displacement_weight,
                displacement_floor=trajectory.displacement_floor,
                oscillation_weight=trajectory.oscillation_weight,
            )
            if trajectory is not None
            and (
                float(trajectory.displacement_weight) > 0.0
                or float(trajectory.oscillation_weight) > 0.0
            )
            else None
        )
        if (
            objective["schema"] != EXACT_ADJOINT_TRAJECTORY_RECEIPT_SCHEMA
            or objective["policy_sha256"] != receipt["policy_sha256"]
            or objective["prompt_tokens_sha256"] != receipt["prompt_tokens_sha256"]
            or objective["answer_tokens_sha256"] != token_receipts[anchor]
            or objective["bridge_tokens_sha256"] != _tokens_sha256(())
            or objective["bridge_token_count"] != 0
            or any(float(weight) != 0.0 for weight in objective["token_loss_weights"])
            or objective["execution_spec_sha256"] != receipt["execution_spec_sha256"]
            or objective["execution_branch_count"] != branch_count
            or objective["branch_indices"] != list(range(branch_count))
            or objective["trajectory_config"]
            != (structural_config.to_dict() if structural_config is not None else None)
            or not math.isclose(
                float(objective["diversity_weight"]),
                float(config.diversity_weight),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(objective["diversity_target_cos"]),
                float(config.diversity_target_cos),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(objective["terminal_value"]),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(objective["trajectory_values"]["improvement"]),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("verified trajectory structural objective differs")
        replayed_total += float(objective["value"])
    elif structural_entry is not None:
        raise ValueError("verified trajectory has an unexpected structural receipt")

    total = finite_number(
        receipt["trajectory_objective_value"],
        role="objective value",
    )
    if not math.isclose(total, replayed_total, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("verified trajectory objective total does not replay")
    return dict(value)


def branch_token_logprobs(
    forward: LivePathForward,
    answer_tokens: Sequence[int],
    *,
    branch_index: int,
) -> Any:
    """Teacher-forced per-token log-probabilities for one producing branch."""
    import mlx.core as mx
    import mlx.nn as nn

    tokens = tuple(answer_tokens)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("answer_tokens must contain non-negative integers")
    if type(branch_index) is not int or not 0 <= branch_index < len(forward.branch_logits):
        raise ValueError("branch_index is outside the live-path branch set")
    logits = forward.branch_logits[branch_index]
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("branch logits must have shape [1, tokens, vocabulary]")
    if int(logits.shape[1]) != len(tokens):
        raise ValueError("branch logits and answer tokens do not align")
    targets = mx.array(tokens)[None, :]
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    return -mx.squeeze(losses, axis=0)


def recurrent_completion_token_logprobs(
    model: Any,
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    bridge_tokens: Sequence[int] = (),
    adapters_on: bool = True,
) -> Any:
    """Score exact completion tokens through the resident cached policy."""

    return cached_live_path_token_logprobs(
        model,
        prompt_tokens,
        completion_tokens,
        spec=spec,
        branch_index=branch_index,
        bridge_tokens=bridge_tokens,
        adapters_on=adapters_on,
    )


def _cached_weighted_completion_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    bridge_tokens: Sequence[int],
    token_loss_weights: Sequence[float],
) -> tuple[Any, Any]:
    """Differentiate one weighted completion through its cached live path."""

    import mlx.core as mx
    import mlx.nn as nn

    weights = tuple(float(value) for value in token_loss_weights)
    if len(weights) != len(completion_tokens) or any(
        not math.isfinite(value) for value in weights
    ):
        raise ValueError("token loss weights must be finite and token-aligned")
    weight_tensor = mx.array(weights, dtype=mx.float32)

    def objective(current_model: Any) -> Any:
        logprobs = recurrent_completion_token_logprobs(
            current_model,
            prompt_tokens,
            completion_tokens,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge_tokens,
            adapters_on=True,
        )
        return -mx.mean(logprobs * weight_tensor)

    return nn.value_and_grad(model, objective)(model)


def clipped_recurrent_grpo_objective(
    policy_logprobs: Sequence[Any],
    old_policy_logprobs: Sequence[Any],
    advantages: Sequence[float],
    *,
    reference_logprobs: Sequence[Any] | None = None,
    config: RecurrentGRPOConfig | None = None,
) -> RecurrentGRPOObjective:
    """Token-normalized clipped GRPO objective with same-RLC reference KL."""
    import mlx.core as mx

    resolved = config or RecurrentGRPOConfig()
    count = len(policy_logprobs)
    if count < 2:
        raise ValueError("recurrent GRPO needs at least two completions")
    if len(old_policy_logprobs) != count or len(advantages) != count:
        raise ValueError("policy, old policy, and advantages must align")
    if reference_logprobs is not None and len(reference_logprobs) != count:
        raise ValueError("reference policy must align with completions")
    normalized_advantages = [float(value) for value in advantages]
    if any(not math.isfinite(value) for value in normalized_advantages):
        raise ValueError("advantages must be finite")

    policy_terms: list[Any] = []
    old_kl_terms: list[Any] = []
    reference_kl_terms: list[Any] = []
    clipped_tokens: list[Any] = []
    token_count = 0
    references: Sequence[Any | None] = (
        reference_logprobs if reference_logprobs is not None else (None,) * count
    )
    for policy, old_policy, advantage, reference in zip(
        policy_logprobs,
        old_policy_logprobs,
        normalized_advantages,
        references,
        strict=True,
    ):
        if policy.ndim != 1 or old_policy.ndim != 1 or policy.shape != old_policy.shape:
            raise ValueError("current and old token logprobs must be aligned vectors")
        if int(policy.shape[0]) < 1:
            raise ValueError("completion token logprobs cannot be empty")
        if reference is not None and (reference.ndim != 1 or reference.shape != policy.shape):
            raise ValueError("reference token logprobs must align with policy")
        token_count += int(policy.shape[0])
        old = mx.stop_gradient(old_policy)
        log_ratio = policy - old
        ratio = mx.exp(log_ratio)
        clipped_ratio = mx.clip(
            ratio,
            1.0 - float(resolved.clip_epsilon),
            1.0 + float(resolved.clip_epsilon),
        )
        surrogate = mx.minimum(ratio * advantage, clipped_ratio * advantage)
        policy_terms.append(-mx.mean(surrogate))
        old_kl_terms.append(mx.mean((ratio - 1.0) - log_ratio))
        clipped_tokens.append(
            mx.mean((mx.abs(ratio - 1.0) > resolved.clip_epsilon).astype(mx.float32))
        )
        if reference is not None:
            delta = mx.stop_gradient(reference) - policy
            reference_kl_terms.append(mx.mean(mx.exp(delta) - delta - 1.0))

    policy_loss = mx.mean(mx.stack(policy_terms))
    old_policy_approx_kl = mx.mean(mx.stack(old_kl_terms))
    clip_fraction = mx.mean(mx.stack(clipped_tokens))
    reference_kl = mx.mean(mx.stack(reference_kl_terms)) if reference_kl_terms else mx.zeros(())
    loss = policy_loss + float(resolved.kl_coefficient) * reference_kl
    return RecurrentGRPOObjective(
        loss=loss,
        policy_loss=policy_loss,
        reference_kl=reference_kl,
        old_policy_approx_kl=old_policy_approx_kl,
        clip_fraction=clip_fraction,
        completion_count=count,
        token_count=token_count,
    )


def verifier_group_objective(
    policy_logprobs: Sequence[Any],
    old_policy_logprobs: Sequence[Any],
    rewards: Sequence[float],
    *,
    reference_logprobs: Sequence[Any] | None = None,
    config: RecurrentGRPOConfig | None = None,
) -> tuple[RecurrentGRPOObjective, dict[str, Any]]:
    """Derive group advantages only from verifier rewards, then optimize."""
    resolved = config or RecurrentGRPOConfig()
    advantage_report = group_advantages(rewards, clip=resolved.advantage_clip)
    objective = clipped_recurrent_grpo_objective(
        policy_logprobs,
        old_policy_logprobs,
        advantage_report["advantages"],
        reference_logprobs=reference_logprobs,
        config=resolved,
    )
    return objective, advantage_report


def exact_adjoint_verifier_group_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[Sequence[int]],
    branch_indices: Sequence[int],
    rewards: Sequence[float],
    *,
    spec: RLCExecutionSpec,
    behavior_logprobs: Sequence[Sequence[float]] | None = None,
    bridge_tokens: Sequence[int] = (),
    config: RecurrentGRPOConfig | None = None,
) -> ExactAdjointRecurrentGRPOResult:
    """Exact first-update GRPO gradient with one recurrent graph resident.

    Completions must come from the bounded cached behavior policy and this
    result must feed exactly one optimizer update. The
    cached behavior probabilities are supplied, so the clipped ratio is
    measured rather than assumed across numerically different MLX kernels. A
    verifier advantage ``A`` weights answer CE by the exact unclipped derivative
    ``A * p/q``; clipped tokens receive zero policy-gradient coefficient. The
    same-RLC k3 KL derivative is represented exactly by the fixed
    token coefficient ``exp(log p_ref - log p_policy) - 1``. The existing
    discrete adjoint then backpropagates each weighted completion through the
    recurrent graph before its activations are released.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    resolved = config or RecurrentGRPOConfig()
    count = len(completion_tokens)
    if count < 2:
        raise ValueError("recurrent GRPO needs at least two completions")
    if len(branch_indices) != count or len(rewards) != count:
        raise ValueError("completions, branches, and rewards must align")
    if behavior_logprobs is not None and len(behavior_logprobs) != count:
        raise ValueError("behavior logprobs must align with completions")
    normalized_branches = tuple(branch_indices)
    if any(type(index) is not int or index < 0 for index in normalized_branches):
        raise ValueError("branch indices must be non-negative integers")
    advantage_report = group_advantages(rewards, clip=resolved.advantage_clip)
    token_count = sum(len(tokens) for tokens in completion_tokens)
    if any(not tokens for tokens in completion_tokens):
        raise ValueError("completion token sequences cannot be empty")
    if advantage_report["degenerate"]:
        return ExactAdjointRecurrentGRPOResult(
            gradients=None,
            advantage_report=advantage_report,
            reference_kl=0.0,
            old_policy_approx_kl=0.0,
            clip_fraction=0.0,
            policy_loss=0.0,
            objective_at_sampling=0.0,
            gradient_surrogate_value=0.0,
            completion_count=count,
            token_count=token_count,
            branch_indices=normalized_branches,
        )

    accumulated: Any | None = None
    reference_kl = 0.0
    old_policy_approx_kl = 0.0
    clip_fraction = 0.0
    policy_loss = 0.0
    surrogate_value = 0.0
    group_scale = 1.0 / count
    behaviors: Sequence[Sequence[float] | None] = (
        behavior_logprobs if behavior_logprobs is not None else (None,) * count
    )
    for tokens, branch_index, advantage, behavior_values in zip(
        completion_tokens,
        normalized_branches,
        advantage_report["advantages"],
        behaviors,
        strict=True,
    ):
        policy = recurrent_completion_token_logprobs(
            model,
            prompt_tokens,
            tokens,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge_tokens,
            adapters_on=True,
        )
        reference = recurrent_completion_token_logprobs(
            model,
            prompt_tokens,
            tokens,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge_tokens,
            adapters_on=False,
        )
        mx.eval(policy, reference)
        if behavior_values is None:
            behavior = mx.stop_gradient(policy)
        else:
            normalized_behavior = tuple(float(value) for value in behavior_values)
            if len(normalized_behavior) != len(tokens) or any(
                not math.isfinite(value) for value in normalized_behavior
            ):
                raise ValueError("behavior logprobs must be finite and token-aligned")
            behavior = mx.array(normalized_behavior, dtype=mx.float32)
        log_ratio = policy - behavior
        ratio = mx.exp(log_ratio)
        clipped_ratio = mx.clip(
            ratio,
            1.0 - float(resolved.clip_epsilon),
            1.0 + float(resolved.clip_epsilon),
        )
        surrogate = mx.minimum(
            ratio * float(advantage),
            clipped_ratio * float(advantage),
        )
        active = (
            ratio <= 1.0 + float(resolved.clip_epsilon)
            if float(advantage) >= 0.0
            else ratio >= 1.0 - float(resolved.clip_epsilon)
        )
        policy_coefficients = float(advantage) * ratio * active.astype(mx.float32)
        delta = reference - policy
        k3_tokens = mx.exp(delta) - delta - 1.0
        mx.eval(k3_tokens)
        reference_kl += group_scale * float(mx.mean(k3_tokens))
        old_policy_approx_kl += group_scale * float(mx.mean((ratio - 1.0) - log_ratio))
        clip_fraction += group_scale * float(
            mx.mean((mx.abs(ratio - 1.0) > resolved.clip_epsilon).astype(mx.float32))
        )
        policy_loss += group_scale * float(-mx.mean(surrogate))
        coefficients = policy_coefficients + float(resolved.kl_coefficient) * (mx.exp(delta) - 1.0)
        mx.eval(coefficients)
        weights = [group_scale * float(value) for value in coefficients]
        value, gradients = _cached_weighted_completion_value_and_grad(
            model,
            prompt_tokens,
            tokens,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge_tokens,
            token_loss_weights=weights,
        )
        surrogate_value += float(value)
        accumulated = (
            gradients
            if accumulated is None
            else tree_map(
                lambda previous, current: previous + current,
                accumulated,
                gradients,
            )
        )
        mx.eval(accumulated)
        del (
            policy,
            reference,
            behavior,
            log_ratio,
            ratio,
            clipped_ratio,
            surrogate,
            active,
            policy_coefficients,
            delta,
            k3_tokens,
            coefficients,
            gradients,
        )
        mx.clear_cache()

    if accumulated is None:
        raise RuntimeError("recurrent GRPO exact-adjoint gradient is empty")
    finite = [mx.all(mx.isfinite(gradient)) for _name, gradient in tree_flatten(accumulated)]
    mx.eval(finite)
    if not finite or not all(bool(flag) for flag in finite):
        raise FloatingPointError("recurrent GRPO gradient is non-finite")
    if clip_fraction > float(resolved.max_initial_clip_fraction):
        raise RuntimeError(
            "cached behavior policy exceeds recurrent PPO clip admission: "
            f"observed={clip_fraction:.6f} "
            f"limit={resolved.max_initial_clip_fraction:.6f}"
        )
    if old_policy_approx_kl > float(resolved.max_initial_old_policy_approx_kl):
        raise RuntimeError(
            "cached behavior policy exceeds recurrent PPO KL admission: "
            f"observed={old_policy_approx_kl:.6f} "
            f"limit={resolved.max_initial_old_policy_approx_kl:.6f}"
        )
    objective_at_sampling = policy_loss + float(resolved.kl_coefficient) * reference_kl
    return ExactAdjointRecurrentGRPOResult(
        gradients=accumulated,
        advantage_report=advantage_report,
        reference_kl=reference_kl,
        old_policy_approx_kl=old_policy_approx_kl,
        clip_fraction=clip_fraction,
        policy_loss=policy_loss,
        objective_at_sampling=objective_at_sampling,
        gradient_surrogate_value=surrogate_value,
        completion_count=count,
        token_count=token_count,
        branch_indices=normalized_branches,
    )


def exact_adjoint_sampled_group_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    samples: Sequence[RecurrentPolicySample],
    rewards: Sequence[float],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
    config: RecurrentGRPOConfig | None = None,
    optimization_token_counts: Sequence[int] | None = None,
) -> ExactAdjointRecurrentGRPOResult:
    """Validate immutable sample provenance before one recurrent update."""

    resolved = config or RecurrentGRPOConfig()
    if len(samples) < 2 or len(samples) != len(rewards):
        raise ValueError("samples and rewards must align with at least two entries")
    token_counts = (
        tuple(len(sample.tokens) for sample in samples)
        if optimization_token_counts is None
        else tuple(optimization_token_counts)
    )
    if (
        len(token_counts) != len(samples)
        or any(
            type(count) is not int or not 1 <= count <= len(sample.tokens)
            for sample, count in zip(samples, token_counts, strict=True)
        )
    ):
        raise ValueError("optimization token counts must align with sampled traces")
    prompt_sha256 = _tokens_sha256(prompt_tokens)
    expected_policy = recurrent_policy_sha256(model, spec)
    old_policy_approx_kl = 0.0
    clip_fraction = 0.0
    group_scale = 1.0 / len(samples)
    for index, (sample, optimization_token_count) in enumerate(
        zip(samples, token_counts, strict=True)
    ):
        if not isinstance(sample, RecurrentPolicySample):
            raise TypeError(f"sample {index} is not a RecurrentPolicySample")
        validate_recurrent_policy_sample_receipt(sample.receipt())
        if not sample.behavior_admitted:
            raise RecurrentSamplingAdmissionError(sample)
        if sample.execution_spec_sha256 != spec.sha256:
            raise ValueError(f"sample {index} execution spec differs")
        if sample.prompt_tokens_sha256 != prompt_sha256:
            raise ValueError(f"sample {index} prompt differs")
        if sample.policy_sha256 != expected_policy:
            raise ValueError(f"sample {index} recurrent policy differs")
        if not math.isclose(
            float(sample.sampling_config.clip_epsilon),
            float(resolved.clip_epsilon),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"sample {index} PPO clip epsilon differs")
        if not (
            len(sample.tokens)
            == len(sample.behavior_logprobs)
            == len(sample.differentiable_logprobs)
        ):
            raise ValueError(f"sample {index} token probabilities differ")
        ratios = []
        differences = []
        for policy, behavior in zip(
            sample.differentiable_logprobs,
            sample.behavior_logprobs,
            strict=True,
        ):
            if not math.isfinite(policy) or not math.isfinite(behavior):
                raise ValueError(f"sample {index} token probabilities are non-finite")
            differences.append(abs(policy - behavior))
            ratios.append(math.exp(policy - behavior))
        maximum = max(differences)
        mean = sum(differences) / len(differences)
        maximum_index = differences.index(maximum)
        if not math.isclose(
            maximum,
            float(sample.max_abs_logprob_drift),
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(
            mean,
            float(sample.mean_abs_logprob_drift),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"sample {index} drift receipt differs")
        if maximum_index != sample.max_abs_logprob_drift_token_index:
            raise ValueError(f"sample {index} maximum-drift index differs")
        full_sample_clip_fraction = sum(
            abs(ratio - 1.0) > float(resolved.clip_epsilon) for ratio in ratios
        ) / len(ratios)
        if not math.isclose(
            full_sample_clip_fraction,
            float(sample.clipped_token_fraction),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"sample {index} clipped-token receipt differs")
        if (
            maximum > float(sample.sampling_config.max_abs_logprob_drift)
            or mean > float(sample.sampling_config.max_mean_abs_logprob_drift)
            or full_sample_clip_fraction
            > float(sample.sampling_config.max_clipped_token_fraction)
            or float(sample.old_policy_approx_kl)
            > float(sample.sampling_config.max_old_policy_approx_kl)
        ):
            raise ValueError(f"sample {index} admission receipt is inconsistent")
        observable_ratios = ratios[:optimization_token_count]
        observable_old_kl = sum(
            (ratio - 1.0) - math.log(ratio) for ratio in observable_ratios
        ) / len(observable_ratios)
        old_policy_approx_kl += group_scale * observable_old_kl
        observable_clip_fraction = sum(
            abs(ratio - 1.0) > float(resolved.clip_epsilon)
            for ratio in observable_ratios
        ) / len(observable_ratios)
        clip_fraction += group_scale * observable_clip_fraction
    if clip_fraction > float(resolved.max_initial_clip_fraction):
        raise RecurrentGroupClipAdmissionError(
            clip_fraction=clip_fraction,
            max_clip_fraction=float(resolved.max_initial_clip_fraction),
            old_policy_approx_kl=old_policy_approx_kl,
        )
    if old_policy_approx_kl > float(resolved.max_initial_old_policy_approx_kl):
        raise RuntimeError(
            "cached behavior policy exceeds recurrent PPO KL admission: "
            f"observed={old_policy_approx_kl:.6f} "
            f"limit={resolved.max_initial_old_policy_approx_kl:.6f}"
        )
    return exact_adjoint_verifier_group_value_and_grad(
        model,
        prompt_tokens,
        [
            sample.tokens[:token_count]
            for sample, token_count in zip(samples, token_counts, strict=True)
        ],
        [sample.branch_index for sample in samples],
        rewards,
        spec=spec,
        behavior_logprobs=[
            sample.behavior_logprobs[:token_count]
            for sample, token_count in zip(samples, token_counts, strict=True)
        ],
        bridge_tokens=bridge_tokens,
        config=resolved,
    )


def _with_verified_trajectory_group_objective(
    model: Any,
    prompt_tokens: Sequence[int],
    samples: Sequence[RecurrentPolicySample],
    rewards: Sequence[float],
    base: ExactAdjointRecurrentGRPOResult,
    *,
    group_admission_receipt: Mapping[str, Any],
    reward_receipt: Mapping[str, Any],
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int],
    trajectory_group_config: VerifiedTrajectoryGroupConfig,
    advantage_clip: float,
    optimization_token_counts: Sequence[int] | None = None,
) -> ExactAdjointRecurrentGRPOResult:
    """Add quality-weighted trajectory gradients after verified admission."""

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    if base.gradients is None or base.advantage_report.get("degenerate") is True:
        raise ValueError("verified trajectory objective requires a non-degenerate policy gradient")
    if bridge_tokens:
        raise ValueError(
            "verified trajectory proof requires empty bridge_tokens until bridge "
            "provenance is admitted"
        )
    if len(samples) != len(rewards) or len(samples) != base.completion_count:
        raise ValueError("verified trajectory samples and rewards must align")
    token_counts = (
        tuple(len(sample.tokens) for sample in samples)
        if optimization_token_counts is None
        else tuple(optimization_token_counts)
    )
    if (
        len(token_counts) != len(samples)
        or any(
            type(count) is not int or not 1 <= count <= len(sample.tokens)
            for sample, count in zip(samples, token_counts, strict=True)
        )
    ):
        raise ValueError("verified trajectory optimization boundaries are invalid")
    if any(not isinstance(sample, RecurrentPolicySample) for sample in samples):
        raise TypeError("verified trajectory samples must be RecurrentPolicySample")
    trajectory = trajectory_group_config.trajectory_config
    intervention = trajectory_group_config.intervention_config
    if trajectory is not None:
        trajectory.validate_depth(spec.recurrent_steps)
    if intervention is not None:
        intervention.validate_depth(spec.recurrent_steps)
    normalized_rewards = tuple(float(reward) for reward in rewards)
    if any(not math.isfinite(reward) for reward in normalized_rewards):
        raise ValueError("verified trajectory rewards must be finite")
    advantages = tuple(
        float(advantage) for advantage in base.advantage_report.get("advantages", ())
    )
    if len(advantages) != len(samples) or any(
        not math.isfinite(advantage) for advantage in advantages
    ):
        raise ValueError("verified trajectory advantages are invalid")
    branch_count = len(spec.branch_roles)
    if (
        branch_count < 2
        or len(samples) != branch_count
        or tuple(sample.branch_index for sample in samples) != tuple(range(branch_count))
    ):
        raise ValueError("verified trajectory requires one canonical sample per execution branch")
    policies = {sample.policy_sha256 for sample in samples}
    if len(policies) != 1:
        raise ValueError("verified trajectory sample policy identity differs")
    policy_sha256 = next(iter(policies))
    if policy_sha256 != group_admission_receipt.get("policy_sha256"):
        raise ValueError("verified trajectory sample policy identity differs")
    admission_sha256 = group_admission_receipt.get("receipt_sha256")
    reward_sha256 = reward_receipt.get("receipt_sha256")
    if not _valid_sha256(admission_sha256) or not _valid_sha256(reward_sha256):
        raise ValueError("verified trajectory source receipt digest is invalid")

    accumulated = base.gradients
    trajectory_value = 0.0

    def add_gradients(gradients: Any) -> None:
        nonlocal accumulated
        accumulated = tree_map(
            lambda previous, current: previous + current,
            accumulated,
            gradients,
        )
        mx.eval(accumulated)

    improvement_enabled = bool(
        trajectory is not None and float(trajectory.improvement_weight) > 0.0
    )
    intervention_enabled = bool(
        intervention is not None
        and (
            float(intervention.causality_weight) > 0.0 or float(intervention.stopping_weight) > 0.0
        )
    )
    quality_enabled = improvement_enabled or intervention_enabled
    positive_indices = (
        [index for index, advantage in enumerate(advantages) if advantage > 0.0]
        if quality_enabled
        else []
    )
    positive_mass = sum(advantages[index] for index in positive_indices)
    if quality_enabled and positive_mass <= 0.0:
        raise ValueError("verified trajectory objective has no positive credit")
    positive_weights = [advantages[index] / positive_mass for index in positive_indices]
    quality_receipts: list[dict[str, Any]] = []
    for completion_index, weight in zip(
        positive_indices,
        positive_weights,
        strict=True,
    ):
        sample = samples[completion_index]
        scaled_trajectory = (
            ExactAdjointTrajectoryConfig(
                probe_steps=trajectory.probe_steps,
                improvement_weight=(float(trajectory.improvement_weight) * weight),
                improvement_margin=trajectory.improvement_margin,
                displacement_weight=0.0,
                displacement_floor=trajectory.displacement_floor,
                oscillation_weight=0.0,
            )
            if improvement_enabled
            else None
        )
        scaled_intervention = (
            ExactAdjointInterventionConfig(
                lesion_steps=intervention.lesion_steps,
                causality_weight=(float(intervention.causality_weight) * weight),
                causality_margin=intervention.causality_margin,
                stopping_steps=intervention.stopping_steps,
                stopping_weight=(float(intervention.stopping_weight) * weight),
                stopping_ponder_cost=intervention.stopping_ponder_cost,
                stopping_temperature=intervention.stopping_temperature,
            )
            if intervention_enabled
            else None
        )
        completion_tokens = sample.tokens[
            : token_counts[completion_index]
        ]
        exact = exact_adjoint_composite_live_path_value_and_grad(
            model,
            prompt_tokens,
            completion_tokens,
            spec=spec,
            trajectory_config=scaled_trajectory,
            intervention_config=scaled_intervention,
            policy_sha256=policy_sha256,
            bridge_tokens=bridge_tokens,
            branch_index=sample.branch_index,
            token_loss_weights=(0.0,) * len(completion_tokens),
        )
        add_gradients(exact.gradients)
        trajectory_value += float(exact.value)
        quality_receipts.append(
            {
                "completion_index": completion_index,
                "completion_tokens_sha256": _tokens_sha256(completion_tokens),
                "advantage_weight": weight,
                "objective_receipt": exact.receipt(),
            }
        )
        del exact
        mx.clear_cache()

    structural_enabled = bool(
        float(trajectory_group_config.diversity_weight) > 0.0
        or (
            trajectory is not None
            and (
                float(trajectory.displacement_weight) > 0.0
                or float(trajectory.oscillation_weight) > 0.0
            )
        )
    )
    anchor_index = max(
        range(len(normalized_rewards)),
        key=lambda index: (normalized_rewards[index], -index),
    )
    anchor = samples[anchor_index]
    anchor_tokens = anchor.tokens[: token_counts[anchor_index]]
    structural_receipt: dict[str, Any] | None = None
    if structural_enabled:
        structural_config = (
            ExactAdjointTrajectoryConfig(
                probe_steps=trajectory.probe_steps,
                improvement_weight=0.0,
                improvement_margin=trajectory.improvement_margin,
                displacement_weight=trajectory.displacement_weight,
                displacement_floor=trajectory.displacement_floor,
                oscillation_weight=trajectory.oscillation_weight,
            )
            if trajectory is not None
            and (
                float(trajectory.displacement_weight) > 0.0
                or float(trajectory.oscillation_weight) > 0.0
            )
            else None
        )
        exact = exact_adjoint_composite_live_path_value_and_grad(
            model,
            prompt_tokens,
            anchor_tokens,
            spec=spec,
            trajectory_config=structural_config,
            policy_sha256=policy_sha256,
            bridge_tokens=bridge_tokens,
            branch_index=None,
            diversity_weight=trajectory_group_config.diversity_weight,
            diversity_target_cos=trajectory_group_config.diversity_target_cos,
            token_loss_weights=(0.0,) * len(anchor_tokens),
        )
        add_gradients(exact.gradients)
        trajectory_value += float(exact.value)
        structural_receipt = {
            "anchor_completion_index": anchor_index,
            "anchor_completion_tokens_sha256": _tokens_sha256(anchor_tokens),
            "objective_receipt": exact.receipt(),
        }
        del exact
        mx.clear_cache()

    finite = [mx.all(mx.isfinite(gradient)) for _name, gradient in tree_flatten(accumulated)]
    mx.eval(finite)
    if not finite or not all(bool(flag) for flag in finite):
        raise FloatingPointError("verified trajectory composite gradient is non-finite")
    source_binding = build_verified_trajectory_group_source_binding(
        group_admission_receipt,
        reward_receipt,
        samples,
        prompt_tokens,
        spec=spec,
        trajectory_group_config=trajectory_group_config,
        advantage_clip=advantage_clip,
        optimization_token_counts=token_counts,
    )
    prefix_bounded = (
        source_binding["schema"] == VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3
    )
    if prefix_bounded:
        group_schema = VERIFIED_TRAJECTORY_GROUP_SCHEMA_V3
    elif intervention_enabled:
        group_schema = VERIFIED_TRAJECTORY_GROUP_SCHEMA_V2
    else:
        group_schema = VERIFIED_TRAJECTORY_GROUP_SCHEMA
    payload = {
        "schema": group_schema,
        **{
            field: source_binding[field]
            for field in (
                "group_admission_sha256",
                "reward_receipt_sha256",
                "policy_sha256",
                "execution_spec_sha256",
                "prompt_tokens_sha256",
                "sample_receipt_sha256s",
                "completion_tokens_sha256s",
                "sample_branch_indices",
                "execution_branch_count",
                "verified_rewards",
                "advantage_clip",
            )
        },
        "advantages": list(advantages),
        "config": trajectory_group_config.to_dict(),
        "positive_completion_indices": positive_indices,
        "positive_advantage_weights": positive_weights,
        "anchor_completion_index": anchor_index,
        "anchor_branch_index": anchor.branch_index,
        "structural_receipt": structural_receipt,
        "trajectory_objective_value": trajectory_value,
    }
    if prefix_bounded:
        payload["optimization_token_counts"] = list(token_counts)
    payload["quality_receipts" if intervention_enabled else "improvement_receipts"] = (
        quality_receipts
    )
    trajectory_receipt = {
        **payload,
        "receipt_sha256": _seal_receipt(payload),
    }
    validate_verified_trajectory_group_receipt(
        trajectory_receipt,
        advantage_report=base.advantage_report,
        expected_source_binding=source_binding,
    )
    return replace(
        base,
        gradients=accumulated,
        trajectory_objective_value=trajectory_value,
        trajectory_receipt=trajectory_receipt,
    )


def exact_adjoint_verified_transition_group_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    samples: Sequence[RecurrentPolicySample],
    group_admission_receipt: Mapping[str, Any],
    reward_receipt: Mapping[str, Any],
    transition_evidence: Sequence[Any],
    *,
    transition_store: Any,
    group_manifest: Mapping[str, Any],
    group_manifest_attestation: Mapping[str, Any],
    independent_scorer: Any,
    token_encoder: Any,
    token_decoder: Any,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
    config: RecurrentGRPOConfig | None = None,
    trajectory_group_config: VerifiedTrajectoryGroupConfig | None = None,
) -> ExactAdjointRecurrentGRPOResult:
    """Admit only independently replayed CP419 transitions to the adjoint.

    Validation, optimizer admission, and exact sample/trace binding all happen
    before recurrent policy hashing or gradient construction. Claim-level
    regression control remains an independently receipted downstream gate; a
    caller-supplied scalar reward has no path into this proof-grade objective.
    """

    from core.learning.verified_transition_group_admission import (
        validate_verified_transition_group_admission,
    )
    from core.learning.verified_transition_reward import rewards_for_recurrent_samples

    validate_verified_transition_group_admission(
        transition_store,
        group_admission_receipt,
        reward_receipt,
        transition_evidence,
        samples,
        prompt_tokens,
        group_manifest=group_manifest,
        group_manifest_attestation=group_manifest_attestation,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
    )
    rewards = rewards_for_recurrent_samples(
        reward_receipt,
        samples,
        prompt_tokens,
    )
    optimization_token_counts = verified_optimization_token_counts(
        samples,
        transition_evidence,
    )
    if len(optimization_token_counts) != len(samples):
        raise ValueError("verified transition optimization boundaries differ")
    if trajectory_group_config is not None and not isinstance(
        trajectory_group_config,
        VerifiedTrajectoryGroupConfig,
    ):
        raise TypeError("trajectory_group_config must be a VerifiedTrajectoryGroupConfig")
    resolved_config = config or RecurrentGRPOConfig()
    if trajectory_group_config is not None:
        if bridge_tokens:
            raise ValueError(
                "verified trajectory proof requires empty bridge_tokens until bridge "
                "provenance is admitted"
            )
        expected_branches = tuple(range(len(spec.branch_roles)))
        if (
            len(samples) != len(expected_branches)
            or tuple(sample.branch_index for sample in samples) != expected_branches
        ):
            raise ValueError(
                "verified trajectory requires one canonical sample per execution branch"
            )
    result = exact_adjoint_sampled_group_value_and_grad(
        model,
        prompt_tokens,
        samples,
        rewards,
        spec=spec,
        bridge_tokens=bridge_tokens,
        config=resolved_config,
        optimization_token_counts=optimization_token_counts,
    )
    if trajectory_group_config is None:
        return result
    return _with_verified_trajectory_group_objective(
        model,
        prompt_tokens,
        samples,
        rewards,
        result,
        group_admission_receipt=group_admission_receipt,
        reward_receipt=reward_receipt,
        spec=spec,
        bridge_tokens=bridge_tokens,
        trajectory_group_config=trajectory_group_config,
        advantage_clip=resolved_config.advantage_clip,
        optimization_token_counts=optimization_token_counts,
    )


def verified_optimization_token_counts(
    samples: Sequence[Any],
    transition_evidence: Sequence[Any],
) -> tuple[int, ...]:
    """Resolve evidence-bounded prefixes without changing legacy sample replay."""

    if all(
        isinstance(sample, RecurrentPolicySample)
        and sample.sample_kind == "causal_final_transition"
        for sample in samples
    ):
        from core.learning.verified_recurrent_transition_evidence import (
            optimization_token_counts_from_evidence,
        )

        return optimization_token_counts_from_evidence(transition_evidence)
    return tuple(len(sample.tokens) for sample in samples)


__all__ = [
    "CAUSAL_RECURRENT_DECODE_SCHEMA",
    "CAUSAL_RECURRENT_PAIR_SCHEMA",
    "CausalRecurrentTransitionPair",
    "FrozenRecurrentStateDecode",
    "RECURRENT_GRPO_SCHEMA",
    "RECURRENT_SAMPLING_SCHEMA",
    "VERIFIED_TRAJECTORY_GROUP_SCHEMA",
    "VERIFIED_TRAJECTORY_GROUP_SCHEMA_V2",
    "VERIFIED_TRAJECTORY_GROUP_SCHEMA_V3",
    "VERIFIED_TRAJECTORY_SOURCE_SCHEMA",
    "VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V2",
    "VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3",
    "ExactAdjointRecurrentGRPOResult",
    "RecurrentGRPOConfig",
    "RecurrentGRPOObjective",
    "RecurrentGroupClipAdmissionError",
    "RecurrentPolicySample",
    "RecurrentSamplingAdmissionError",
    "RecurrentSamplingConfig",
    "RecurrentSamplingParityError",
    "VerifiedTrajectoryGroupConfig",
    "attach_recurrent_policy_adapters",
    "attach_coda_policy_adapters_at_sites",
    "branch_token_logprobs",
    "build_recurrent_policy_optimizer",
    "build_verified_trajectory_group_source_binding",
    "clipped_recurrent_grpo_objective",
    "cortex_config_from_execution_spec",
    "exact_adjoint_sampled_group_value_and_grad",
    "exact_adjoint_verified_transition_group_value_and_grad",
    "exact_adjoint_verifier_group_value_and_grad",
    "recurrent_policy_sha256",
    "recurrent_policy_tensor_map_sha256",
    "recurrent_policy_sample_from_causal_pair",
    "recurrent_policy_sample_from_receipt",
    "recurrent_policy_optimizer_config",
    "recurrent_sampling_rng_root_sha256",
    "recurrent_completion_token_logprobs",
    "sample_final_recurrent_transition_completion",
    "sample_recurrent_completion",
    "sample_final_recurrent_transition_pair",
    "validate_causal_recurrent_transition_pair_receipt",
    "validate_recurrent_policy_sample_receipt",
    "validate_verified_trajectory_group_receipt",
    "validate_verified_trajectory_group_source_binding",
    "verified_optimization_token_counts",
    "verifier_group_objective",
]

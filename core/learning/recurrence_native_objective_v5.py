"""Generated-prefix, branch-selective training on the resident RLC backend.

The v1 resident bootstrap optimized an equal mean of teacher-forced terminal
cross-entropies. Live inference does neither: it conditions on its own emitted
prefix and selects one branch. This module closes both mismatches while keeping
the exact KV-cached, recurrent execution path from objective v2.

Generated roll-ins are detached behavior samples. Gold answer tokens remain the
labels, so the objective learns recovery from its own prefixes without treating
mistakes as truth. The legacy v1 config combined branch gradients with detached
soft-min weights; that let one branch solve while every other branch merely
stayed geometrically different. Config v2 defaults to an equal branch mean so
every branch must carry lexical responsibility. Legacy receipts retain their
exact replay semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrence_adapter import recurrence_adapter_scope
from core.learning.recurrence_native_objective_v2 import (
    _advance_recurrent_states,
    _cached_causal_layers,
    _logits,
    _prepare_recurrent_prefix,
    cached_live_path_token_logprobs,
    cached_supervised_live_path_value_and_grad,
    generate_cached_live_path_rollin,
    transformer_layer_group_checkpointing,
)
from core.learning.role_conditioned_lora import recurrent_branch_index

RECURRENCE_NATIVE_OBJECTIVE_V5_SCHEMA = "aura.recurrence_native_objective.v5"
GENERATED_ROLLIN_CONFIG_SCHEMA_V1 = "aura.generated_rollin_selection_config.v1"
GENERATED_ROLLIN_CONFIG_SCHEMA = "aura.generated_rollin_selection_config.v2"
GENERATED_ROLLIN_RECEIPT_SCHEMA = "aura.generated_rollin_selection_receipt.v1"
GENERATED_ROLLIN_TRUST_BOUNDARY = "producer_sealed_tokens_external_policy_replay_required"
_ROLLIN_SEED_DOMAIN = b"aura.generated_rollin.branch_seed.v1\0"
_MIX_MASK_DOMAIN = b"aura.generated_rollin.mix_mask.v1\0"
_EXAMPLE_SEED_DOMAIN = b"aura.generated_rollin.example_seed.v1\0"


def _sha256_tokens(tokens: Sequence[int], *, allow_empty: bool = False) -> str:
    normalized = list(tokens)
    if (not allow_empty and not normalized) or any(
        type(token) is not int or token < 0 for token in normalized
    ):
        raise ValueError("tokens must contain non-negative integers")
    return hashlib.sha256(
        json.dumps(
            normalized,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class GeneratedRollinSelectionConfig:
    """Bound controls for generated-prefix and branch-selection training."""

    student_forcing_probability: float = 0.5
    sampling_temperature: float = 0.8
    branch_softmin_temperature: float = 0.5
    branch_aggregation: str = "equal_mean"
    schema: str = GENERATED_ROLLIN_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GENERATED_ROLLIN_CONFIG_SCHEMA:
            if self.schema != GENERATED_ROLLIN_CONFIG_SCHEMA_V1:
                raise ValueError("generated roll-in config schema is unsupported")
        if self.schema == GENERATED_ROLLIN_CONFIG_SCHEMA_V1:
            if self.branch_aggregation != "detached_softmin":
                raise ValueError("v1 generated roll-in config requires detached_softmin")
        elif self.branch_aggregation != "equal_mean":
            raise ValueError("v2 generated roll-in config requires equal_mean")
        for name, value, lower, upper in (
            (
                "student_forcing_probability",
                self.student_forcing_probability,
                0.0,
                1.0,
            ),
            ("sampling_temperature", self.sampling_temperature, 0.0, 10.0),
            (
                "branch_softmin_temperature",
                self.branch_softmin_temperature,
                1e-3,
                100.0,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"{name} must be inside [{lower}, {upper}]")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "student_forcing_probability": float(self.student_forcing_probability),
            "sampling_temperature": float(self.sampling_temperature),
            "branch_softmin_temperature": float(self.branch_softmin_temperature),
        }
        if self.schema != GENERATED_ROLLIN_CONFIG_SCHEMA_V1:
            payload["branch_aggregation"] = self.branch_aggregation
        return payload

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> GeneratedRollinSelectionConfig:
        common = {
            "schema",
            "student_forcing_probability",
            "sampling_temperature",
            "branch_softmin_temperature",
        }
        if not isinstance(value, Mapping):
            raise ValueError("generated roll-in config fields do not match")
        schema = value.get("schema")
        required = (
            common
            if schema == GENERATED_ROLLIN_CONFIG_SCHEMA_V1
            else common | {"branch_aggregation"}
        )
        if set(value) != required:
            raise ValueError("generated roll-in config fields do not match")
        return cls(
            schema=schema,
            student_forcing_probability=value["student_forcing_probability"],
            sampling_temperature=value["sampling_temperature"],
            branch_softmin_temperature=value["branch_softmin_temperature"],
            branch_aggregation=value.get("branch_aggregation", "detached_softmin"),
        )


@dataclass(frozen=True, slots=True)
class GeneratedRollinBranchEvidence:
    branch_index: int
    branch_seed: int
    loss: float
    selection_weight: float
    generated_tokens_sha256: str
    effective_rollin_sha256: str
    student_forced_positions: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_index": self.branch_index,
            "branch_seed": self.branch_seed,
            "loss": self.loss,
            "selection_weight": self.selection_weight,
            "generated_tokens_sha256": self.generated_tokens_sha256,
            "effective_rollin_sha256": self.effective_rollin_sha256,
            "student_forced_positions": list(self.student_forced_positions),
        }


@dataclass(frozen=True, slots=True)
class GeneratedRollinLivePathEvaluation:
    value: float
    branches: tuple[GeneratedRollinBranchEvidence, ...]
    answer_token_count: int
    execution_spec_sha256: str
    prompt_tokens_sha256: str
    answer_tokens_sha256: str
    bridge_tokens_sha256: str
    config: GeneratedRollinSelectionConfig
    base_seed: int

    @property
    def config_sha256(self) -> str:
        return self.config.sha256

    @property
    def branch_values(self) -> tuple[float, ...]:
        return tuple(branch.loss for branch in self.branches)

    @property
    def branch_weights(self) -> tuple[float, ...]:
        return tuple(branch.selection_weight for branch in self.branches)

    @property
    def branch_indices(self) -> tuple[int, ...]:
        return tuple(branch.branch_index for branch in self.branches)

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": GENERATED_ROLLIN_RECEIPT_SCHEMA,
            "objective_schema": RECURRENCE_NATIVE_OBJECTIVE_V5_SCHEMA,
            "trust_boundary": GENERATED_ROLLIN_TRUST_BOUNDARY,
            "value": self.value,
            "branches": [branch.to_dict() for branch in self.branches],
            "answer_token_count": self.answer_token_count,
            "execution_spec_sha256": self.execution_spec_sha256,
            "prompt_tokens_sha256": self.prompt_tokens_sha256,
            "answer_tokens_sha256": self.answer_tokens_sha256,
            "bridge_tokens_sha256": self.bridge_tokens_sha256,
            "config": self.config.to_dict(),
            "config_sha256": self.config_sha256,
            "base_seed": self.base_seed,
        }
        return {**body, "receipt_sha256": _sha256_json(body)}


@dataclass(frozen=True, slots=True)
class GeneratedRollinLivePathResult:
    evaluation: GeneratedRollinLivePathEvaluation
    gradients: Any

    @property
    def value(self) -> float:
        return self.evaluation.value

    @property
    def branch_values(self) -> tuple[float, ...]:
        return self.evaluation.branch_values

    @property
    def branch_weights(self) -> tuple[float, ...]:
        return self.evaluation.branch_weights

    @property
    def branch_indices(self) -> tuple[int, ...]:
        return self.evaluation.branch_indices

    @property
    def answer_token_count(self) -> int:
        return self.evaluation.answer_token_count

    @property
    def execution_spec_sha256(self) -> str:
        return self.evaluation.execution_spec_sha256

    @property
    def prompt_tokens_sha256(self) -> str:
        return self.evaluation.prompt_tokens_sha256

    @property
    def answer_tokens_sha256(self) -> str:
        return self.evaluation.answer_tokens_sha256

    @property
    def bridge_tokens_sha256(self) -> str:
        return self.evaluation.bridge_tokens_sha256


def detached_softmin_weights(
    losses: Sequence[float],
    *,
    temperature: float,
) -> tuple[float, ...]:
    """Return numerically stable detached best-branch weights."""

    normalized = tuple(float(loss) for loss in losses)
    if (
        not normalized
        or any(not math.isfinite(loss) or loss < 0.0 for loss in normalized)
        or isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 1e-3 <= float(temperature) <= 100.0
    ):
        raise ValueError("softmin losses or temperature are invalid")
    reference = min(normalized)
    exponents = tuple(-(loss - reference) / float(temperature) for loss in normalized)
    if min(exponents) < -700.0:
        raise FloatingPointError("branch loss spread exceeds selection envelope")
    raw = tuple(math.exp(value) for value in exponents)
    total = sum(raw)
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError("branch selection weights are non-finite")
    return tuple(value / total for value in raw)


def derive_rollin_seed(
    *,
    campaign_seed: int,
    phase: str,
    example_id: str,
    sample_ordinal: int,
    execution_spec_sha256: str,
) -> int:
    """Derive a resume-stable 32-bit behavior seed from bound sample identity."""

    if type(campaign_seed) is not int or not 0 <= campaign_seed <= 2**63 - 1:
        raise ValueError("campaign_seed must be inside [0, 2^63-1]")
    if phase not in {"train", "validation"}:
        raise ValueError("roll-in phase must be train or validation")
    if (
        not isinstance(example_id, str)
        or not example_id
        or len(example_id) > 256
        or any(ord(character) < 0x20 for character in example_id)
    ):
        raise ValueError("example_id is invalid")
    if type(sample_ordinal) is not int or not 0 <= sample_ordinal <= 10_000_000:
        raise ValueError("sample_ordinal must be inside [0, 10000000]")
    if (
        not isinstance(execution_spec_sha256, str)
        or len(execution_spec_sha256) != 64
        or any(character not in "0123456789abcdef" for character in execution_spec_sha256)
    ):
        raise ValueError("execution_spec_sha256 is invalid")
    payload = {
        "campaign_seed": campaign_seed,
        "phase": phase,
        "example_id": example_id,
        "sample_ordinal": sample_ordinal,
        "execution_spec_sha256": execution_spec_sha256,
    }
    digest = hashlib.sha256()
    digest.update(_EXAMPLE_SEED_DOMAIN)
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    return int.from_bytes(digest.digest()[:4], "big")


def _softmin_value(losses: Sequence[float], *, temperature: float) -> float:
    reference = min(losses)
    exponents = tuple(-(loss - reference) / temperature for loss in losses)
    if min(exponents) < -700.0:
        raise FloatingPointError("branch loss spread exceeds selection envelope")
    return reference - temperature * math.log(
        sum(math.exp(value) for value in exponents) / len(exponents)
    )


def _branch_objective_weights(
    losses: Sequence[float],
    *,
    config: GeneratedRollinSelectionConfig,
) -> tuple[float, ...]:
    if config.branch_aggregation == "equal_mean":
        if not losses:
            raise ValueError("branch objective requires at least one loss")
        weight = 1.0 / len(losses)
        return tuple(weight for _loss in losses)
    return detached_softmin_weights(
        losses,
        temperature=config.branch_softmin_temperature,
    )


def _branch_objective_value(
    losses: Sequence[float],
    *,
    config: GeneratedRollinSelectionConfig,
) -> float:
    if config.branch_aggregation == "equal_mean":
        if not losses:
            raise ValueError("branch objective requires at least one loss")
        return sum(float(loss) for loss in losses) / len(losses)
    return _softmin_value(
        losses,
        temperature=config.branch_softmin_temperature,
    )


def _branch_seed(
    *,
    base_seed: int,
    branch_index: int,
    execution_spec_sha256: str,
    prompt_tokens_sha256: str,
) -> int:
    payload = {
        "base_seed": base_seed,
        "branch_index": branch_index,
        "execution_spec_sha256": execution_spec_sha256,
        "prompt_tokens_sha256": prompt_tokens_sha256,
    }
    digest = hashlib.sha256()
    digest.update(_ROLLIN_SEED_DOMAIN)
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    return int.from_bytes(digest.digest()[:4], "big")


def deterministic_mixed_rollin(
    answer_tokens: Sequence[int],
    generated_tokens: Sequence[int],
    *,
    probability: float,
    base_seed: int,
    branch_index: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Mix generated decoder inputs without changing supervised labels."""

    answer = tuple(answer_tokens)
    generated = tuple(generated_tokens)
    if (
        not answer
        or len(answer) != len(generated)
        or any(type(token) is not int or token < 0 for token in (*answer, *generated))
    ):
        raise ValueError("generated roll-in must be answer-aligned")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ValueError("student forcing probability must be inside [0, 1]")
    if type(base_seed) is not int or not 0 <= base_seed <= 0xFFFFFFFF:
        raise ValueError("base_seed must be inside [0, 2^32-1]")
    if type(branch_index) is not int or branch_index < 0:
        raise ValueError("branch_index must be non-negative")

    mixed: list[int] = []
    selected: list[int] = []
    threshold = int(float(probability) * (1 << 64))
    for position, (target, sampled) in enumerate(zip(answer, generated, strict=True)):
        # The final decoder input has no successor label and is never consumed.
        use_generated = False
        if position + 1 < len(answer):
            digest = hashlib.sha256()
            digest.update(_MIX_MASK_DOMAIN)
            digest.update(base_seed.to_bytes(4, "big"))
            digest.update(branch_index.to_bytes(4, "big"))
            digest.update(position.to_bytes(8, "big"))
            use_generated = int.from_bytes(digest.digest()[:8], "big") < threshold
        mixed.append(sampled if use_generated else target)
        if use_generated:
            selected.append(position)
    return tuple(mixed), tuple(selected)


def _inputs(
    answer_tokens: Sequence[int],
    bridge_tokens: Sequence[int],
    branch_indices: Sequence[int] | None,
    *,
    spec: RLCExecutionSpec,
    base_seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    answer = tuple(answer_tokens)
    bridge = tuple(bridge_tokens)
    if not answer or any(type(token) is not int or token < 0 for token in answer):
        raise ValueError("answer_tokens must contain non-negative integers")
    if any(type(token) is not int or token < 0 for token in bridge):
        raise ValueError("bridge_tokens must contain non-negative integers")
    if type(base_seed) is not int or not 0 <= base_seed <= 0xFFFFFFFF:
        raise ValueError("base_seed must be inside [0, 2^32-1]")
    indices = (
        tuple(range(len(spec.branch_roles))) if branch_indices is None else tuple(branch_indices)
    )
    if (
        not indices
        or len(indices) != len(set(indices))
        or any(
            type(index) is not int or not 0 <= index < len(spec.branch_roles) for index in indices
        )
    ):
        raise ValueError("branch indices must be unique members of the live branch set")
    return answer, bridge, indices


def _branch_rollin(
    model: Any,
    prompt_tokens: Sequence[int],
    answer: tuple[int, ...],
    bridge: tuple[int, ...],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    base_seed: int,
    config: GeneratedRollinSelectionConfig,
) -> tuple[tuple[int, ...], tuple[int, ...], int, str]:
    prompt_sha256 = _sha256_tokens(prompt_tokens)
    seed = _branch_seed(
        base_seed=base_seed,
        branch_index=branch_index,
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=prompt_sha256,
    )
    generated = generate_cached_live_path_rollin(
        model,
        prompt_tokens,
        spec=spec,
        branch_index=branch_index,
        token_count=len(answer),
        seed=seed,
        temperature=config.sampling_temperature,
        bridge_tokens=bridge,
    )
    mixed, positions = deterministic_mixed_rollin(
        answer,
        generated.tokens,
        probability=config.student_forcing_probability,
        base_seed=seed,
        branch_index=branch_index,
    )
    return mixed, positions, seed, generated.tokens_sha256


def _rematerialized_branch_loss(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    bridge_tokens: Sequence[int],
    rollin_tokens: Sequence[int],
    weight_tensor: Any,
    weight_total: float,
) -> Any:
    """Return the canonical branch loss with nested activation rematerialization."""

    import mlx.core as mx

    with transformer_layer_group_checkpointing(
        model,
        model.trainable_parameters(),
        group_size=1,
    ):
        logprobs = cached_live_path_token_logprobs(
            model,
            prompt_tokens,
            answer_tokens,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge_tokens,
            adapters_on=True,
            rollin_tokens=rollin_tokens,
        )
    return -mx.sum(logprobs * weight_tensor) / weight_total


def _frozen_recurrent_prefix(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], int, int]:
    import mlx.core as mx

    values = _prepare_recurrent_prefix(model, prompt_tokens, spec=spec)
    prompt_embeddings, seeds, prompts, states, anchors, prelude_end, coda_start = values

    def detached(items: Sequence[Any]) -> tuple[Any, ...]:
        result = tuple(mx.stop_gradient(value) for value in items)
        mx.eval(result)
        return result

    frozen_prompt = mx.stop_gradient(prompt_embeddings)
    mx.eval(frozen_prompt)
    return (
        frozen_prompt,
        detached(seeds),
        detached(prompts),
        detached(states),
        detached(anchors),
        prelude_end,
        coda_start,
    )


def _recomputed_states(
    model: Any,
    prompts: tuple[Any, ...],
    anchors: tuple[Any, ...],
    initial_states: tuple[Any, ...],
    *,
    spec: RLCExecutionSpec,
    recurrent_steps: int,
    prelude_end: int,
    coda_start: int,
) -> tuple[Any, ...]:
    import mlx.core as mx

    states = initial_states
    for step in range(recurrent_steps):
        outputs = _advance_recurrent_states(
            model,
            prompts,
            states,
            anchors,
            spec,
            step,
            prelude_end,
            coda_start,
        )
        next_states = tuple(mx.stop_gradient(value) for value in outputs)
        mx.eval(next_states)
        if states is not initial_states:
            del states
        del outputs
        states = next_states
        mx.clear_cache()
    return states


def _cached_branch_loss_from_final_states(
    model: Any,
    prompt_embeddings: Any,
    seeds: tuple[Any, ...],
    final_states: tuple[Any, ...],
    answer_tokens: Sequence[int],
    *,
    branch_index: int,
    bridge_tokens: Sequence[int],
    rollin_tokens: Sequence[int],
    prelude_end: int,
    coda_start: int,
    weight_tensor: Any,
    weight_total: float,
) -> Any:
    import mlx.core as mx

    from core.brain.llm.decoder_topology import decoder_layers
    from core.learning.intrinsic_recurrence import model_layer_caches

    layers = tuple(decoder_layers(model))
    cache = model_layer_caches(model)
    _cached_causal_layers(model, prompt_embeddings, cache)
    with recurrent_branch_index(branch_index), recurrence_adapter_scope():
        _cached_causal_layers(
            model,
            seeds[branch_index],
            cache,
            start=0,
            end=prelude_end,
        )
        persisted = _cached_causal_layers(
            model,
            final_states[branch_index],
            cache,
            start=prelude_end,
            end=coda_start,
        )
        output = _cached_causal_layers(
            model,
            persisted,
            cache,
            start=coda_start,
            end=len(layers),
        )
    logits = _logits(model, output)[0, -1]
    targets = (*bridge_tokens, *answer_tokens)
    decoder_inputs = (*bridge_tokens, *rollin_tokens)
    answer_logprobs: list[Any] = []
    for position, (target, decoder_input) in enumerate(zip(targets, decoder_inputs, strict=True)):
        logprob = logits[target].astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32))
        if position >= len(bridge_tokens):
            answer_logprobs.append(logprob)
        if position + 1 == len(targets):
            continue
        hidden = model.model.embed_tokens(mx.array([[decoder_input]]))
        hidden = _cached_causal_layers(model, hidden, cache)
        logits = _logits(model, hidden)[0, -1]
    logprobs = mx.stack(answer_logprobs)
    return -mx.sum(logprobs * weight_tensor) / weight_total


def _cache_from_state(model: Any, state: tuple[tuple[Any, Any], ...]) -> list[Any]:
    """Rebuild a per-layer cache from a saved boundary.

    Takes the model because the cache OBJECT differs per layer on a hybrid
    decoder: an attention layer holds K/V, a gated-delta layer holds a
    recurrent state. Constructing KVCache for every entry restores the values
    into the wrong container at three layers in four, and nothing raises.
    """
    from core.learning.intrinsic_recurrence import model_layer_caches

    cache = model_layer_caches(model)
    for entry, saved in zip(cache, state, strict=True):
        entry.state = saved
    return cache


def _initial_decoder_state(
    model: Any,
    parameter_tree: Any,
    prompt_embeddings: Any,
    seeds: tuple[Any, ...],
    final_states: tuple[Any, ...],
    *,
    branch_index: int,
    prelude_end: int,
    coda_start: int,
) -> tuple[tuple[tuple[Any, Any], ...], Any]:
    """Build the cache and logits that predict the first tail token."""

    from core.brain.llm.decoder_topology import decoder_layers
    from core.learning.intrinsic_recurrence import model_layer_caches

    model.update(parameter_tree)
    layers = tuple(decoder_layers(model))
    cache = model_layer_caches(model)
    _cached_causal_layers(model, prompt_embeddings, cache)
    with recurrent_branch_index(branch_index), recurrence_adapter_scope():
        _cached_causal_layers(
            model,
            seeds[branch_index],
            cache,
            start=0,
            end=prelude_end,
        )
        persisted = _cached_causal_layers(
            model,
            final_states[branch_index],
            cache,
            start=prelude_end,
            end=coda_start,
        )
        output = _cached_causal_layers(
            model,
            persisted,
            cache,
            start=coda_start,
            end=len(layers),
        )
    return tuple(entry.state for entry in cache), _logits(model, output)[0, -1]


def _decoder_transition(
    model: Any,
    parameter_tree: Any,
    cache_state: tuple[tuple[Any, Any], ...],
    token: int,
) -> tuple[tuple[tuple[Any, Any], ...], Any]:
    """Advance one ordinary decode token from an explicit KV boundary."""

    import mlx.core as mx

    model.update(parameter_tree)
    cache = _cache_from_state(model, cache_state)
    hidden = model.model.embed_tokens(mx.array([[token]]))
    hidden = _cached_causal_layers(model, hidden, cache)
    return tuple(entry.state for entry in cache), _logits(model, hidden)[0, -1]


def _detached_decoder_trajectory(
    model: Any,
    parameters: Any,
    prompt_embeddings: Any,
    seeds: tuple[Any, ...],
    final_states: tuple[Any, ...],
    decoder_inputs: tuple[int, ...],
    *,
    branch_index: int,
    prelude_end: int,
    coda_start: int,
) -> tuple[tuple[tuple[tuple[Any, Any], ...], Any], ...]:
    """Materialize every decoder boundary without retaining producer graphs."""

    import mlx.core as mx

    cache_state, logits = _initial_decoder_state(
        model,
        parameters,
        prompt_embeddings,
        seeds,
        final_states,
        branch_index=branch_index,
        prelude_end=prelude_end,
        coda_start=coda_start,
    )
    cache_state = tuple(
        (mx.stop_gradient(keys), mx.stop_gradient(values))
        for keys, values in cache_state
    )
    logits = mx.stop_gradient(logits)
    mx.eval(cache_state, logits)
    mx.synchronize()
    mx.clear_cache()
    trajectory = [(cache_state, logits)]
    for token in decoder_inputs[:-1]:
        cache_state, logits = _decoder_transition(
            model,
            parameters,
            cache_state,
            token,
        )
        cache_state = tuple(
            (mx.stop_gradient(keys), mx.stop_gradient(values))
            for keys, values in cache_state
        )
        logits = mx.stop_gradient(logits)
        mx.eval(cache_state, logits)
        mx.synchronize()
        mx.clear_cache()
        trajectory.append((cache_state, logits))
    return tuple(trajectory)


def _tree_inner_product(left: Any, right: Any) -> Any:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    left_items = tree_flatten(left)
    right_items = tree_flatten(right)
    if [path for path, _value in left_items] != [path for path, _value in right_items]:
        raise RuntimeError("decoder adjoint cache topology drift")
    return sum(
        mx.sum(left_value * right_value)
        for (_path, left_value), (_other, right_value) in zip(
            left_items,
            right_items,
            strict=True,
        )
    )


def _weighted_token_loss(logits: Any, target: int, weight: float) -> Any:
    import mlx.core as mx

    if weight == 0.0:
        return mx.array(0.0, dtype=mx.float32)
    return -float(weight) * (
        logits[target].astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32))
    )


def _exact_adjoint_branch_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    bridge_tokens: Sequence[int],
    rollin_tokens: Sequence[int],
    weight_tensor: Any,
    weight_total: float,
) -> tuple[Any, Any]:
    """Differentiate one cached branch with O(1) recurrent-depth residency."""

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    parameters = model.trainable_parameters()
    (
        prompt_embeddings,
        seeds,
        prompts,
        initial_states,
        anchors,
        prelude_end,
        coda_start,
    ) = _frozen_recurrent_prefix(model, prompt_tokens, spec=spec)
    final_states = _recomputed_states(
        model,
        prompts,
        anchors,
        initial_states,
        spec=spec,
        recurrent_steps=spec.recurrent_steps,
        prelude_end=prelude_end,
        coda_start=coda_start,
    )

    targets = (*bridge_tokens, *answer_tokens)
    decoder_inputs = (*bridge_tokens, *rollin_tokens)
    position_weights = (
        *(0.0 for _token in bridge_tokens),
        *(float(weight) / weight_total for weight in weight_tensor.tolist()),
    )
    if len(targets) != len(decoder_inputs) or len(targets) != len(position_weights):
        raise RuntimeError("decoder adjoint token alignment drift")

    trajectory = _detached_decoder_trajectory(
        model,
        parameters,
        prompt_embeddings,
        seeds,
        final_states,
        decoder_inputs,
        branch_index=branch_index,
        prelude_end=prelude_end,
        coda_start=coda_start,
    )
    if len(trajectory) != len(targets):
        raise RuntimeError("decoder adjoint trajectory length drift")
    token_losses = tuple(
        _weighted_token_loss(logits, target, weight)
        for (_cache, logits), target, weight in zip(
            trajectory,
            targets,
            position_weights,
            strict=True,
        )
    )
    mx.eval(token_losses)
    value_total = sum(float(loss) for loss in token_losses)
    final_cache = trajectory[-1][0]
    cache_cotangents = tuple(
        (mx.zeros_like(keys), mx.zeros_like(values)) for keys, values in final_cache
    )
    mx.eval(cache_cotangents)
    for position in range(len(targets) - 1, 0, -1):
        prior_cache = trajectory[position - 1][0]

        def decoder_pullback(
            cache_state: tuple[tuple[Any, Any], ...],
            _token: int = decoder_inputs[position - 1],
            _target: int = targets[position],
            _weight: float = position_weights[position],
            _cotangents: tuple[tuple[Any, Any], ...] = cache_cotangents,
        ) -> Any:
            with transformer_layer_group_checkpointing(model, parameters, group_size=1):
                next_cache, logits = _decoder_transition(
                    model,
                    parameters,
                    cache_state,
                    _token,
                )
            return _weighted_token_loss(logits, _target, _weight) + _tree_inner_product(
                next_cache,
                _cotangents,
            )

        decoder_value, incoming_cache = mx.value_and_grad(decoder_pullback)(prior_cache)
        mx.eval(decoder_value, incoming_cache)
        cache_cotangents = tree_map(mx.stop_gradient, incoming_cache)
        mx.eval(cache_cotangents)
        mx.synchronize()
        del prior_cache, incoming_cache, decoder_value
        mx.clear_cache()

    def initial_tail_pullback(parameter_tree: Any, states: tuple[Any, ...]) -> Any:
        with transformer_layer_group_checkpointing(model, parameter_tree, group_size=1):
            cache_state, logits = _initial_decoder_state(
                model,
                parameter_tree,
                prompt_embeddings,
                seeds,
                states,
                branch_index=branch_index,
                prelude_end=prelude_end,
                coda_start=coda_start,
            )
        return _weighted_token_loss(
            logits,
            targets[0],
            position_weights[0],
        ) + _tree_inner_product(cache_state, cache_cotangents)

    initial_value, (accumulated, cotangents) = mx.value_and_grad(
        initial_tail_pullback,
        argnums=(0, 1),
    )(parameters, final_states)
    mx.eval(initial_value, accumulated, cotangents)
    value = mx.array(value_total, dtype=mx.float32)
    mx.eval(value)
    cotangents = tuple(mx.stop_gradient(item) for item in cotangents)
    mx.eval(cotangents)
    finite = [mx.all(mx.isfinite(item)) for _path, item in tree_flatten(accumulated)]
    mx.eval(finite)
    if not finite or not all(bool(flag) for flag in finite):
        raise FloatingPointError("decoder adjoint parameter gradient is non-finite")
    del final_states, trajectory, token_losses
    mx.synchronize()
    mx.clear_cache()

    for step in range(spec.recurrent_steps - 1, -1, -1):
        prior_states = _recomputed_states(
            model,
            prompts,
            anchors,
            initial_states,
            spec=spec,
            recurrent_steps=step,
            prelude_end=prelude_end,
            coda_start=coda_start,
        )

        def transition_pullback(
            parameter_tree: Any,
            states: tuple[Any, ...],
            _step: int = step,
            _cotangents: tuple[Any, ...] = cotangents,
        ) -> Any:
            model.update(parameter_tree)
            with transformer_layer_group_checkpointing(model, parameter_tree, group_size=1):
                outputs = _advance_recurrent_states(
                    model,
                    prompts,
                    states,
                    anchors,
                    spec,
                    _step,
                    prelude_end,
                    coda_start,
                )
            return sum(
                mx.sum(output * cotangent)
                for output, cotangent in zip(outputs, _cotangents, strict=True)
            )

        _pullback, (parameter_gradient, incoming) = mx.value_and_grad(
            transition_pullback,
            argnums=(0, 1),
        )(parameters, prior_states)
        mx.eval(parameter_gradient, incoming)
        accumulated = tree_map(
            lambda left, right: left + right,
            accumulated,
            parameter_gradient,
        )
        mx.eval(accumulated)
        cotangents = tuple(mx.stop_gradient(item) for item in incoming)
        mx.eval(cotangents)
        if prior_states is not initial_states:
            del prior_states
        del parameter_gradient, incoming
        mx.clear_cache()
    return value, accumulated


def generated_rollin_live_path_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    base_seed: int,
    config: GeneratedRollinSelectionConfig | None = None,
    bridge_tokens: Sequence[int] = (),
    token_loss_weights: Sequence[float] | None = None,
    branch_indices: Sequence[int] | None = None,
) -> GeneratedRollinLivePathResult:
    """Differentiate generated-prefix soft branch selection with bounded memory."""

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    resolved = config or GeneratedRollinSelectionConfig()
    answer, bridge, indices = _inputs(
        answer_tokens,
        bridge_tokens,
        branch_indices,
        spec=spec,
        base_seed=base_seed,
    )
    weights = (
        tuple(1.0 for _ in answer)
        if token_loss_weights is None
        else tuple(float(value) for value in token_loss_weights)
    )
    weight_total = sum(weights)
    if (
        len(weights) != len(answer)
        or any(not math.isfinite(value) or value < 0.0 for value in weights)
        or weight_total <= 0.0
    ):
        raise ValueError("token loss weights must be finite and answer-aligned")

    # With one branch and no generated-prefix positions, v5 is definitionally
    # the v2 teacher-forced objective. Delegating that exact boundary avoids a
    # second mathematically equivalent float32 adjoint accumulating different
    # rounding error based on prior MLX allocator/RNG history.
    if len(indices) == 1 and resolved.student_forcing_probability == 0.0:
        branch_index = indices[0]
        rollin, positions, branch_seed, generated_sha256 = _branch_rollin(
            model,
            prompt_tokens,
            answer,
            bridge,
            spec=spec,
            branch_index=branch_index,
            base_seed=base_seed,
            config=resolved,
        )
        if rollin != answer or positions:
            raise RuntimeError("zero-forcing roll-in diverged from teacher tokens")
        exact = cached_supervised_live_path_value_and_grad(
            model,
            prompt_tokens,
            answer,
            spec=spec,
            bridge_tokens=bridge,
            token_loss_weights=weights,
            branch_indices=indices,
        )
        branch_value = float(exact.branch_values[0])
        evaluation = GeneratedRollinLivePathEvaluation(
            value=branch_value,
            branches=(
                GeneratedRollinBranchEvidence(
                    branch_index=branch_index,
                    branch_seed=branch_seed,
                    loss=branch_value,
                    selection_weight=1.0,
                    generated_tokens_sha256=generated_sha256,
                    effective_rollin_sha256=_sha256_tokens(rollin),
                    student_forced_positions=positions,
                ),
            ),
            answer_token_count=len(answer),
            execution_spec_sha256=spec.sha256,
            prompt_tokens_sha256=_sha256_tokens(prompt_tokens),
            answer_tokens_sha256=_sha256_tokens(answer),
            bridge_tokens_sha256=_sha256_tokens(bridge, allow_empty=True),
            config=resolved,
            base_seed=base_seed,
        )
        return GeneratedRollinLivePathResult(
            evaluation=evaluation,
            gradients=exact.gradients,
        )
    weight_tensor = mx.array(weights, dtype=mx.float32)

    gradients_numerator: Any | None = None
    denominator = 0.0
    reference: float | None = None
    records: list[dict[str, Any]] = []
    for branch_index in indices:
        rollin, positions, branch_seed, generated_sha256 = _branch_rollin(
            model,
            prompt_tokens,
            answer,
            bridge,
            spec=spec,
            branch_index=branch_index,
            base_seed=base_seed,
            config=resolved,
        )

        value, gradients = _exact_adjoint_branch_value_and_grad(
            model,
            prompt_tokens,
            answer,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge,
            rollin_tokens=rollin,
            weight_tensor=weight_tensor,
            weight_total=weight_total,
        )
        finite_flags = [
            mx.all(mx.isfinite(gradient)) for _path, gradient in tree_flatten(gradients)
        ]
        mx.eval(value, gradients, finite_flags)
        gradient_value = float(value)
        if (
            not math.isfinite(gradient_value)
            or gradient_value < 0.0
            or not finite_flags
            or not all(bool(flag) for flag in finite_flags)
        ):
            raise FloatingPointError("generated roll-in branch gradient is non-finite")

        canonical_value = _rematerialized_branch_loss(
            model,
            prompt_tokens,
            answer,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge,
            rollin_tokens=rollin,
            weight_tensor=weight_tensor,
            weight_total=weight_total,
        )
        mx.eval(canonical_value)
        branch_value = float(canonical_value)
        del canonical_value

        if resolved.branch_aggregation == "equal_mean":
            raw_weight = 1.0
            if reference is None:
                reference = branch_value
        elif reference is None:
            reference = branch_value
            raw_weight = 1.0
        elif branch_value < reference:
            exponent = -(reference - branch_value) / resolved.branch_softmin_temperature
            if exponent < -700.0:
                raise FloatingPointError("branch loss spread exceeds selection envelope")
            rescale = math.exp(exponent)
            if gradients_numerator is not None:
                gradients_numerator = tree_map(
                    lambda total, factor=rescale: total * factor,
                    gradients_numerator,
                )
                mx.eval(gradients_numerator)
            denominator *= rescale
            reference = branch_value
            raw_weight = 1.0
        else:
            exponent = -(branch_value - reference) / resolved.branch_softmin_temperature
            if exponent < -700.0:
                raise FloatingPointError("branch loss spread exceeds selection envelope")
            raw_weight = math.exp(exponent)
        scaled = tree_map(
            lambda gradient, factor=raw_weight: factor * gradient,
            gradients,
        )
        gradients_numerator = (
            scaled
            if gradients_numerator is None
            else tree_map(
                lambda total, gradient: total + gradient,
                gradients_numerator,
                scaled,
            )
        )
        denominator += raw_weight
        mx.eval(gradients_numerator)
        records.append(
            {
                "branch_index": branch_index,
                "branch_seed": branch_seed,
                "loss": branch_value,
                "generated_tokens_sha256": generated_sha256,
                "effective_rollin_sha256": _sha256_tokens(rollin),
                "student_forced_positions": positions,
            }
        )
        del value, gradients, scaled
        mx.clear_cache()

    if (
        gradients_numerator is None
        or reference is None
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        raise RuntimeError("generated roll-in objective produced no gradient")
    gradients = tree_map(lambda value: value / denominator, gradients_numerator)
    mx.eval(gradients)
    branch_values = tuple(record["loss"] for record in records)
    selection_weights = _branch_objective_weights(
        branch_values,
        config=resolved,
    )
    branches = tuple(
        GeneratedRollinBranchEvidence(
            branch_index=record["branch_index"],
            branch_seed=record["branch_seed"],
            loss=record["loss"],
            selection_weight=selection_weight,
            generated_tokens_sha256=record["generated_tokens_sha256"],
            effective_rollin_sha256=record["effective_rollin_sha256"],
            student_forced_positions=record["student_forced_positions"],
        )
        for record, selection_weight in zip(records, selection_weights, strict=True)
    )
    evaluation = GeneratedRollinLivePathEvaluation(
        value=_branch_objective_value(
            branch_values,
            config=resolved,
        ),
        branches=branches,
        answer_token_count=len(answer),
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=_sha256_tokens(prompt_tokens),
        answer_tokens_sha256=_sha256_tokens(answer),
        bridge_tokens_sha256=_sha256_tokens(bridge, allow_empty=True),
        config=resolved,
        base_seed=base_seed,
    )
    return GeneratedRollinLivePathResult(
        evaluation=evaluation,
        gradients=gradients,
    )


def generated_rollin_live_path_loss(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    base_seed: int,
    config: GeneratedRollinSelectionConfig | None = None,
    bridge_tokens: Sequence[int] = (),
    token_loss_weights: Sequence[float] | None = None,
    branch_indices: Sequence[int] | None = None,
) -> GeneratedRollinLivePathEvaluation:
    """Evaluate the exact generated-prefix branch objective without mutation."""

    import mlx.core as mx

    resolved = config or GeneratedRollinSelectionConfig()
    answer, bridge, indices = _inputs(
        answer_tokens,
        bridge_tokens,
        branch_indices,
        spec=spec,
        base_seed=base_seed,
    )
    weights = (
        tuple(1.0 for _ in answer)
        if token_loss_weights is None
        else tuple(float(value) for value in token_loss_weights)
    )
    weight_total = sum(weights)
    if (
        len(weights) != len(answer)
        or any(not math.isfinite(value) or value < 0.0 for value in weights)
        or weight_total <= 0.0
    ):
        raise ValueError("token loss weights must be finite and answer-aligned")
    weight_tensor = mx.array(weights, dtype=mx.float32)
    records: list[dict[str, Any]] = []
    for branch_index in indices:
        rollin, positions, branch_seed, generated_sha256 = _branch_rollin(
            model,
            prompt_tokens,
            answer,
            bridge,
            spec=spec,
            branch_index=branch_index,
            base_seed=base_seed,
            config=resolved,
        )
        value = _rematerialized_branch_loss(
            model,
            prompt_tokens,
            answer,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge,
            rollin_tokens=rollin,
            weight_tensor=weight_tensor,
            weight_total=weight_total,
        )
        mx.eval(value)
        branch_value = float(value)
        del value
        mx.clear_cache()
        if not math.isfinite(branch_value) or branch_value < 0.0:
            raise FloatingPointError("generated roll-in branch loss is non-finite")
        records.append(
            {
                "branch_index": branch_index,
                "branch_seed": branch_seed,
                "loss": branch_value,
                "generated_tokens_sha256": generated_sha256,
                "effective_rollin_sha256": _sha256_tokens(rollin),
                "student_forced_positions": positions,
            }
        )
    branch_values = tuple(record["loss"] for record in records)
    selection_weights = _branch_objective_weights(
        branch_values,
        config=resolved,
    )
    return GeneratedRollinLivePathEvaluation(
        value=_branch_objective_value(
            branch_values,
            config=resolved,
        ),
        branches=tuple(
            GeneratedRollinBranchEvidence(
                branch_index=record["branch_index"],
                branch_seed=record["branch_seed"],
                loss=record["loss"],
                selection_weight=selection_weight,
                generated_tokens_sha256=record["generated_tokens_sha256"],
                effective_rollin_sha256=record["effective_rollin_sha256"],
                student_forced_positions=record["student_forced_positions"],
            )
            for record, selection_weight in zip(
                records,
                selection_weights,
                strict=True,
            )
        ),
        answer_token_count=len(answer),
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=_sha256_tokens(prompt_tokens),
        answer_tokens_sha256=_sha256_tokens(answer),
        bridge_tokens_sha256=_sha256_tokens(bridge, allow_empty=True),
        config=resolved,
        base_seed=base_seed,
    )


def validate_generated_rollin_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Replay all producer arithmetic and reject a merely rehashed receipt."""

    required = {
        "schema",
        "objective_schema",
        "trust_boundary",
        "value",
        "branches",
        "answer_token_count",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "answer_tokens_sha256",
        "bridge_tokens_sha256",
        "config",
        "config_sha256",
        "base_seed",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("generated roll-in receipt fields do not match")
    normalized = dict(value)
    if (
        normalized["schema"] != GENERATED_ROLLIN_RECEIPT_SCHEMA
        or normalized["objective_schema"] != RECURRENCE_NATIVE_OBJECTIVE_V5_SCHEMA
        or normalized["trust_boundary"] != GENERATED_ROLLIN_TRUST_BOUNDARY
        or type(normalized["answer_token_count"]) is not int
        or normalized["answer_token_count"] < 1
        or type(normalized["base_seed"]) is not int
        or not 0 <= normalized["base_seed"] <= 0xFFFFFFFF
    ):
        raise ValueError("generated roll-in receipt structure is invalid")
    for role in (
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "answer_tokens_sha256",
        "bridge_tokens_sha256",
        "config_sha256",
        "receipt_sha256",
    ):
        candidate = normalized[role]
        if (
            not isinstance(candidate, str)
            or len(candidate) != 64
            or any(character not in "0123456789abcdef" for character in candidate)
        ):
            raise ValueError(f"generated roll-in receipt {role} is invalid")
    branches = normalized["branches"]
    if not isinstance(branches, list) or not branches:
        raise ValueError("generated roll-in receipt branches are invalid")
    branch_fields = {
        "branch_index",
        "branch_seed",
        "loss",
        "selection_weight",
        "generated_tokens_sha256",
        "effective_rollin_sha256",
        "student_forced_positions",
    }
    losses: list[float] = []
    weights: list[float] = []
    indices: list[int] = []
    for branch in branches:
        if not isinstance(branch, Mapping) or set(branch) != branch_fields:
            raise ValueError("generated roll-in branch receipt is invalid")
        index = branch["branch_index"]
        seed = branch["branch_seed"]
        loss = branch["loss"]
        weight = branch["selection_weight"]
        positions = branch["student_forced_positions"]
        if (
            type(index) is not int
            or index < 0
            or type(seed) is not int
            or not 0 <= seed <= 0xFFFFFFFF
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or not 0.0 < float(weight) <= 1.0
            or not isinstance(positions, list)
            or any(
                type(position) is not int
                or not 0 <= position < normalized["answer_token_count"] - 1
                for position in positions
            )
            or positions != sorted(set(positions))
        ):
            raise ValueError("generated roll-in branch values are invalid")
        for role in ("generated_tokens_sha256", "effective_rollin_sha256"):
            digest = branch[role]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("generated roll-in branch digest is invalid")
        indices.append(index)
        losses.append(float(loss))
        weights.append(float(weight))
    if len(indices) != len(set(indices)):
        raise ValueError("generated roll-in branch indices repeat")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("generated roll-in branch weights do not normalize")
    config = GeneratedRollinSelectionConfig.from_dict(normalized["config"])
    if config.sha256 != normalized["config_sha256"]:
        raise ValueError("generated roll-in config commitment mismatch")
    replayed_weights = _branch_objective_weights(
        losses,
        config=config,
    )
    if any(
        not math.isclose(observed, replayed, rel_tol=0.0, abs_tol=1e-12)
        for observed, replayed in zip(weights, replayed_weights, strict=True)
    ):
        raise ValueError("generated roll-in branch weights do not replay")
    replayed_value = _branch_objective_value(
        losses,
        config=config,
    )
    if (
        isinstance(normalized["value"], bool)
        or not isinstance(normalized["value"], (int, float))
        or not math.isfinite(float(normalized["value"]))
        or not math.isclose(
            float(normalized["value"]),
            replayed_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("generated roll-in objective value does not replay")
    body = {key: normalized[key] for key in required - {"receipt_sha256"}}
    if _sha256_json(body) != normalized["receipt_sha256"]:
        raise ValueError("generated roll-in receipt commitment mismatch")
    return normalized


__all__ = [
    "GENERATED_ROLLIN_CONFIG_SCHEMA",
    "GENERATED_ROLLIN_CONFIG_SCHEMA_V1",
    "GENERATED_ROLLIN_RECEIPT_SCHEMA",
    "GENERATED_ROLLIN_TRUST_BOUNDARY",
    "RECURRENCE_NATIVE_OBJECTIVE_V5_SCHEMA",
    "GeneratedRollinBranchEvidence",
    "GeneratedRollinLivePathEvaluation",
    "GeneratedRollinLivePathResult",
    "GeneratedRollinSelectionConfig",
    "derive_rollin_seed",
    "detached_softmin_weights",
    "deterministic_mixed_rollin",
    "generated_rollin_live_path_loss",
    "generated_rollin_live_path_value_and_grad",
    "validate_generated_rollin_receipt",
]

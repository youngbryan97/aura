"""Live-path recurrence-native objective over latent slots.

Unlike v1, this objective never recurs lexical prompt or answer states. It
reproduces the live causal layout in a differentiable no-cache view:

1. embed and prefill the prompt;
2. create deterministic role-seeded latent slots;
3. recur only those slots against the fixed prompt prefix;
4. exchange branch consensus and apply the live anti-collapse perturbation;
5. persist each final branch through prelude/window/coda at slot positions;
6. score teacher-forced answer tokens after the persisted slots.

The no-cache view remains useful for structural objectives. Policy-gradient
probabilities use the differentiable KV-cached path below: resident quantized
kernels are shape-dependent, so mathematical graph equivalence does not imply
numerically interchangeable behavior probabilities at 32B scale.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from core.brain.llm.latent_cortex.branch_exchange import private_exchange_slots
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.loop_core import (
    alpha_for_step,
    build_loop_core_contract,
    controlled_recurrent_update,
)
from core.brain.llm.latent_cortex.recurrence_adapter import (
    coda_adapter_scope,
    current_recurrence_adapter_scope,
    recurrence_adapter_disabled,
    recurrence_adapter_scope,
)
from core.brain.llm.latent_cortex.types import WorkspaceConfig
from core.brain.llm.latent_cortex.workspace import LatentWorkspace, per_position_rms
from core.learning.depth_conditioned_lora import (
    current_depth_index,
    recurrent_depth_index,
)
from core.learning.role_conditioned_lora import (
    current_branch_index,
    recurrent_branch_index,
)

RECURRENCE_NATIVE_SCHEMA_V2 = "aura.recurrence_native_objective.v2"
RECURRENT_TRANSITION_STATE_SCHEMA = "aura.recurrent_transition_state.v1"
RECURRENT_TRANSITION_INPUT_SCHEMA = "aura.recurrent_transition_input.v1"
RECURRENT_STATE_TRAIL_SCHEMA = "aura.recurrent_state_trail.v1"
EXACT_ADJOINT_TRAJECTORY_SCHEMA = "aura.exact_adjoint_trajectory_objective.v1"
EXACT_ADJOINT_TRAJECTORY_RECEIPT_SCHEMA = "aura.exact_adjoint_trajectory_objective_receipt.v2"
EXACT_ADJOINT_INTERVENTION_SCHEMA = "aura.exact_adjoint_intervention_objective.v1"
EXACT_ADJOINT_INTERVENTION_RECEIPT_SCHEMA = "aura.exact_adjoint_trajectory_objective_receipt.v3"
EXACT_ADJOINT_AUXILIARY_RECEIPT_SCHEMA = "aura.exact_adjoint_trajectory_objective_receipt.v4"
INTERVENTION_MEASUREMENT_TRUST_BOUNDARY = (
    "producer_sealed_arithmetic_external_state_replay_required"
)
_EXACT_ADJOINT_INPUT_DOMAIN = b"aura.exact_adjoint_input.v1\0"


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_tokens_sha256(tokens: Sequence[int], *, role: str) -> str:
    normalized = list(tokens)
    if not normalized or any(type(token) is not int or token < 0 for token in normalized):
        raise ValueError(f"{role} must contain non-negative integer tokens")
    return hashlib.sha256(
        json.dumps(
            normalized,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _canonical_optional_tokens_sha256(tokens: Sequence[int], *, role: str) -> str:
    normalized = list(tokens)
    if any(type(token) is not int or token < 0 for token in normalized):
        raise ValueError(f"{role} must contain non-negative integer tokens")
    return hashlib.sha256(
        json.dumps(
            normalized,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _exact_adjoint_input_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(_EXACT_ADJOINT_INPUT_DOMAIN)
    digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExactAdjointTrajectoryConfig:
    """Auxiliary trajectory terms replayed through the bounded exact adjoint.

    The terminal policy objective and these terms remain separate. Callers can
    therefore measure each term's gradient in isolation before admitting a
    composite, while the resident path still keeps only one recurrent
    transition graph live at a time.
    """

    probe_steps: tuple[int, ...] = (1, 2)
    improvement_weight: float = 0.0
    improvement_margin: float = 0.02
    displacement_weight: float = 0.0
    displacement_floor: float = 0.01
    oscillation_weight: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.probe_steps
            or any(type(step) is not int or step < 1 for step in self.probe_steps)
            or tuple(sorted(set(self.probe_steps))) != self.probe_steps
        ):
            raise ValueError("probe_steps must be strictly increasing positive integers")
        for name, value, high in (
            ("improvement_weight", self.improvement_weight, 100.0),
            ("improvement_margin", self.improvement_margin, 10.0),
            ("displacement_weight", self.displacement_weight, 100.0),
            ("displacement_floor", self.displacement_floor, 1.0),
            ("oscillation_weight", self.oscillation_weight, 100.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= high
            ):
                raise ValueError(f"{name} must be finite inside [0, {high:g}]")
        if float(self.improvement_weight) > 0.0 and len(self.probe_steps) < 2:
            raise ValueError("improvement requires at least two probe steps")
        if not any(
            float(weight) > 0.0
            for weight in (
                self.improvement_weight,
                self.displacement_weight,
                self.oscillation_weight,
            )
        ):
            raise ValueError("trajectory objective must enable at least one term")

    def validate_depth(self, depth: int) -> None:
        if self.probe_steps[-1] > depth:
            raise ValueError("trajectory probe step exceeds recurrent depth")
        if float(self.oscillation_weight) > 0.0 and depth < 2:
            raise ValueError("oscillation objective requires at least two transitions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXACT_ADJOINT_TRAJECTORY_SCHEMA,
            "probe_steps": list(self.probe_steps),
            "improvement_weight": float(self.improvement_weight),
            "improvement_margin": float(self.improvement_margin),
            "displacement_weight": float(self.displacement_weight),
            "displacement_floor": float(self.displacement_floor),
            "oscillation_weight": float(self.oscillation_weight),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ExactAdjointTrajectoryConfig:
        required = {
            "schema",
            "probe_steps",
            "improvement_weight",
            "improvement_margin",
            "displacement_weight",
            "displacement_floor",
            "oscillation_weight",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("trajectory objective config fields do not match")
        if value.get("schema") != EXACT_ADJOINT_TRAJECTORY_SCHEMA:
            raise ValueError("trajectory objective config schema is unsupported")
        probe_steps = value.get("probe_steps")
        if not isinstance(probe_steps, list):
            raise ValueError("trajectory objective probe_steps must be a list")
        return cls(
            probe_steps=tuple(probe_steps),
            improvement_weight=value["improvement_weight"],
            improvement_margin=value["improvement_margin"],
            displacement_weight=value["displacement_weight"],
            displacement_floor=value["displacement_floor"],
            oscillation_weight=value["oscillation_weight"],
        )


@dataclass(frozen=True, slots=True)
class ExactAdjointInterventionConfig:
    """Causal-necessity and cost-aware stopping terms for verified answers."""

    lesion_steps: tuple[int, ...] = (1,)
    causality_weight: float = 0.0
    causality_margin: float = 0.02
    stopping_steps: tuple[int, ...] = (1, 2)
    stopping_weight: float = 0.0
    stopping_ponder_cost: float = 0.01
    stopping_temperature: float = 0.1

    def __post_init__(self) -> None:
        for name, steps in (
            ("lesion_steps", self.lesion_steps),
            ("stopping_steps", self.stopping_steps),
        ):
            if (
                not steps
                or any(type(step) is not int or step < 1 for step in steps)
                or tuple(sorted(set(steps))) != steps
            ):
                raise ValueError(f"{name} must be strictly increasing positive integers")
        for name, value, high in (
            ("causality_weight", self.causality_weight, 100.0),
            ("causality_margin", self.causality_margin, 10.0),
            ("stopping_weight", self.stopping_weight, 100.0),
            ("stopping_ponder_cost", self.stopping_ponder_cost, 1.0),
            ("stopping_temperature", self.stopping_temperature, 10.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= high
            ):
                raise ValueError(f"{name} must be finite inside [0, {high:g}]")
        if not 1e-4 <= float(self.stopping_temperature) <= 10.0:
            raise ValueError("stopping_temperature must be inside [1e-4, 10]")
        if float(self.stopping_weight) > 0.0 and len(self.stopping_steps) < 2:
            raise ValueError("stopping objective requires at least two candidate depths")
        if not any(float(weight) > 0.0 for weight in (self.causality_weight, self.stopping_weight)):
            raise ValueError("intervention objective must enable at least one term")

    def validate_depth(self, depth: int) -> None:
        if float(self.causality_weight) > 0.0 and self.lesion_steps[-1] > depth:
            raise ValueError("causal lesion step exceeds recurrent depth")
        if float(self.stopping_weight) > 0.0 and self.stopping_steps[-1] > depth:
            raise ValueError("stopping step exceeds recurrent depth")
        if float(self.stopping_weight) > 0.0 and self.stopping_steps[-1] != depth:
            raise ValueError("stopping objective must include the terminal depth")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXACT_ADJOINT_INTERVENTION_SCHEMA,
            "lesion_steps": list(self.lesion_steps),
            "causality_weight": float(self.causality_weight),
            "causality_margin": float(self.causality_margin),
            "stopping_steps": list(self.stopping_steps),
            "stopping_weight": float(self.stopping_weight),
            "stopping_ponder_cost": float(self.stopping_ponder_cost),
            "stopping_temperature": float(self.stopping_temperature),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ExactAdjointInterventionConfig:
        required = {
            "schema",
            "lesion_steps",
            "causality_weight",
            "causality_margin",
            "stopping_steps",
            "stopping_weight",
            "stopping_ponder_cost",
            "stopping_temperature",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("intervention objective config fields do not match")
        if value.get("schema") != EXACT_ADJOINT_INTERVENTION_SCHEMA:
            raise ValueError("intervention objective config schema is unsupported")
        if not isinstance(value["lesion_steps"], list) or not isinstance(
            value["stopping_steps"], list
        ):
            raise ValueError("intervention objective steps must be lists")
        return cls(
            lesion_steps=tuple(value["lesion_steps"]),
            causality_weight=value["causality_weight"],
            causality_margin=value["causality_margin"],
            stopping_steps=tuple(value["stopping_steps"]),
            stopping_weight=value["stopping_weight"],
            stopping_ponder_cost=value["stopping_ponder_cost"],
            stopping_temperature=value["stopping_temperature"],
        )


@dataclass(frozen=True, slots=True)
class ExactAdjointLivePathResult:
    """One exact-adjoint value/gradient result with replayable term telemetry."""

    value: float
    gradients: Any
    terminal_value: float
    diversity_value: float
    trajectory_values: Mapping[str, float]
    step_losses: Mapping[int, tuple[float, ...]]
    displacements: tuple[float, ...]
    oscillation_cosines: tuple[float, ...]
    diversity_cosines: tuple[float, ...]
    branch_indices: tuple[int, ...]
    trajectory_config: ExactAdjointTrajectoryConfig | None
    execution_spec_sha256: str
    recurrent_depth: int
    execution_branch_count: int
    diversity_weight: float
    diversity_target_cos: float
    policy_sha256: str | None
    prompt_tokens_sha256: str
    prompt_token_count: int
    answer_tokens_sha256: str
    answer_token_count: int
    bridge_tokens_sha256: str
    bridge_token_count: int
    token_loss_weights: tuple[float, ...]
    terminal_objective_weight: float = 1.0
    lesion_losses: Mapping[int, tuple[float, ...]] = field(default_factory=dict)
    stopping_teacher_receipts: tuple[Mapping[str, Any], ...] = ()
    intervention_config: ExactAdjointInterventionConfig | None = None

    def receipt(self) -> dict[str, Any]:
        if not _valid_sha256(self.policy_sha256):
            raise ValueError("proof receipt requires a valid policy_sha256")
        intervention_enabled = self.intervention_config is not None
        auxiliary_only = float(self.terminal_objective_weight) != 1.0
        if auxiliary_only and intervention_enabled:
            raise ValueError("auxiliary trajectory receipts cannot include interventions")
        payload = {
            "schema": (
                EXACT_ADJOINT_AUXILIARY_RECEIPT_SCHEMA
                if auxiliary_only
                else EXACT_ADJOINT_INTERVENTION_RECEIPT_SCHEMA
                if intervention_enabled
                else EXACT_ADJOINT_TRAJECTORY_RECEIPT_SCHEMA
            ),
            "value": float(self.value),
            "terminal_value": float(self.terminal_value),
            "diversity_value": float(self.diversity_value),
            "trajectory_values": {
                name: float(value) for name, value in sorted(self.trajectory_values.items())
            },
            "step_losses": {
                str(step): [float(value) for value in values]
                for step, values in sorted(self.step_losses.items())
            },
            "displacements": [float(value) for value in self.displacements],
            "oscillation_cosines": [float(value) for value in self.oscillation_cosines],
            "diversity_cosines": [float(value) for value in self.diversity_cosines],
            "branch_indices": list(self.branch_indices),
            "execution_spec_sha256": self.execution_spec_sha256,
            "recurrent_depth": self.recurrent_depth,
            "execution_branch_count": self.execution_branch_count,
            "diversity_weight": self.diversity_weight,
            "diversity_target_cos": self.diversity_target_cos,
            "policy_sha256": self.policy_sha256,
            "prompt_tokens_sha256": self.prompt_tokens_sha256,
            "prompt_token_count": self.prompt_token_count,
            "answer_tokens_sha256": self.answer_tokens_sha256,
            "answer_token_count": self.answer_token_count,
            "bridge_tokens_sha256": self.bridge_tokens_sha256,
            "bridge_token_count": self.bridge_token_count,
            "token_loss_weights": [float(value) for value in self.token_loss_weights],
            "trajectory_config": (
                self.trajectory_config.to_dict() if self.trajectory_config is not None else None
            ),
        }
        if auxiliary_only:
            payload["terminal_objective_weight"] = float(self.terminal_objective_weight)
        if intervention_enabled:
            payload.update(
                {
                    "lesion_losses": {
                        str(step): [float(value) for value in values]
                        for step, values in sorted(self.lesion_losses.items())
                    },
                    "stopping_teacher_receipts": [
                        dict(receipt) for receipt in self.stopping_teacher_receipts
                    ],
                    "intervention_config": self.intervention_config.to_dict(),
                    "measurement_trust_boundary": (INTERVENTION_MEASUREMENT_TRUST_BOUNDARY),
                }
            )
        input_payload = {
            key: payload[key]
            for key in (
                "policy_sha256",
                "prompt_tokens_sha256",
                "prompt_token_count",
                "answer_tokens_sha256",
                "answer_token_count",
                "bridge_tokens_sha256",
                "bridge_token_count",
                "token_loss_weights",
                "execution_spec_sha256",
                "recurrent_depth",
                "execution_branch_count",
                "branch_indices",
                "diversity_weight",
                "diversity_target_cos",
                "trajectory_config",
            )
        }
        if intervention_enabled:
            input_payload["intervention_config"] = payload["intervention_config"]
            input_payload["measurement_trust_boundary"] = payload["measurement_trust_boundary"]
        if auxiliary_only:
            input_payload["terminal_objective_weight"] = payload[
                "terminal_objective_weight"
            ]
        payload["objective_input_sha256"] = _exact_adjoint_input_sha256(input_payload)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return {**payload, "receipt_sha256": hashlib.sha256(encoded).hexdigest()}


def validate_exact_adjoint_live_path_receipt(value: Any) -> dict[str, Any]:
    """Validate custody and arithmetic over producer-sealed measurement atoms."""

    base_required = {
        "schema",
        "value",
        "terminal_value",
        "diversity_value",
        "trajectory_values",
        "step_losses",
        "displacements",
        "oscillation_cosines",
        "diversity_cosines",
        "branch_indices",
        "execution_spec_sha256",
        "recurrent_depth",
        "execution_branch_count",
        "diversity_weight",
        "diversity_target_cos",
        "policy_sha256",
        "prompt_tokens_sha256",
        "prompt_token_count",
        "answer_tokens_sha256",
        "answer_token_count",
        "bridge_tokens_sha256",
        "bridge_token_count",
        "token_loss_weights",
        "objective_input_sha256",
        "trajectory_config",
        "receipt_sha256",
    }
    intervention_fields = {
        "lesion_losses",
        "stopping_teacher_receipts",
        "intervention_config",
        "measurement_trust_boundary",
    }
    schema = value.get("schema") if isinstance(value, Mapping) else None
    auxiliary_only = schema == EXACT_ADJOINT_AUXILIARY_RECEIPT_SCHEMA
    intervention_enabled = schema == EXACT_ADJOINT_INTERVENTION_RECEIPT_SCHEMA
    required = base_required | intervention_fields if intervention_enabled else base_required
    if auxiliary_only:
        required = required | {"terminal_objective_weight"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("exact-adjoint trajectory receipt fields do not match")
    receipt = dict(value)
    observed = receipt.pop("receipt_sha256")
    encoded = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if not isinstance(observed, str) or observed != hashlib.sha256(encoded).hexdigest():
        raise ValueError("exact-adjoint trajectory receipt commitment mismatch")
    if receipt["schema"] not in {
        EXACT_ADJOINT_TRAJECTORY_RECEIPT_SCHEMA,
        EXACT_ADJOINT_INTERVENTION_RECEIPT_SCHEMA,
        EXACT_ADJOINT_AUXILIARY_RECEIPT_SCHEMA,
    }:
        raise ValueError("exact-adjoint trajectory receipt schema is unsupported")
    for role in (
        "execution_spec_sha256",
        "policy_sha256",
        "prompt_tokens_sha256",
        "answer_tokens_sha256",
        "bridge_tokens_sha256",
        "objective_input_sha256",
    ):
        if not _valid_sha256(receipt[role]):
            raise ValueError(f"exact-adjoint {role} is invalid")
    depth = receipt["recurrent_depth"]
    branch_count = receipt["execution_branch_count"]
    branches = receipt["branch_indices"]
    prompt_count = receipt["prompt_token_count"]
    answer_count = receipt["answer_token_count"]
    bridge_count = receipt["bridge_token_count"]
    if (
        type(depth) is not int
        or depth < 1
        or type(branch_count) is not int
        or branch_count < 1
        or not isinstance(branches, list)
        or not branches
        or any(type(index) is not int or index < 0 for index in branches)
        or any(index >= branch_count for index in branches)
        or len(set(branches)) != len(branches)
        or type(prompt_count) is not int
        or prompt_count < 1
        or type(answer_count) is not int
        or answer_count < 1
        or type(bridge_count) is not int
        or bridge_count < 0
    ):
        raise ValueError("exact-adjoint input cardinality or branch identity is invalid")

    def finite_number(item: Any, *, role: str) -> float:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"exact-adjoint {role} is not finite")
        return float(item)

    terminal = finite_number(receipt["terminal_value"], role="terminal value")
    diversity = finite_number(receipt["diversity_value"], role="diversity value")
    diversity_weight = finite_number(receipt["diversity_weight"], role="diversity weight")
    diversity_target = finite_number(receipt["diversity_target_cos"], role="diversity target")
    if not 0.0 <= diversity_weight <= 10.0 or not 0.0 <= diversity_target <= 1.0:
        raise ValueError("exact-adjoint diversity configuration is invalid")
    weights = receipt["token_loss_weights"]
    if not isinstance(weights, list) or len(weights) != answer_count:
        raise ValueError("exact-adjoint token loss weights do not align")
    normalized_weights = [finite_number(item, role="token loss weight") for item in weights]
    if any(item < 0.0 for item in normalized_weights):
        raise ValueError("exact-adjoint token loss weight is negative")
    total = finite_number(receipt["value"], role="total value")
    terms = receipt["trajectory_values"]
    expected_term_names = {
        "improvement",
        "displacement",
        "oscillation",
        *(("causality", "stopping") if intervention_enabled else ()),
    }
    if not isinstance(terms, Mapping) or set(terms) != expected_term_names:
        raise ValueError("exact-adjoint trajectory term set is invalid")
    term_values = {
        str(name): finite_number(number, role=f"{name} value") for name, number in terms.items()
    }
    terminal_weight = (
        finite_number(
            receipt["terminal_objective_weight"],
            role="terminal objective weight",
        )
        if auxiliary_only
        else 1.0
    )
    if not 0.0 <= terminal_weight <= 1.0:
        raise ValueError("exact-adjoint terminal objective weight is invalid")
    expected_total = terminal_weight * terminal + diversity + sum(term_values.values())
    if not math.isclose(total, expected_total, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("exact-adjoint total does not replay from its terms")

    config_value = receipt["trajectory_config"]
    config = (
        ExactAdjointTrajectoryConfig.from_dict(config_value) if config_value is not None else None
    )
    if config is not None:
        config.validate_depth(depth)
    intervention_config = (
        ExactAdjointInterventionConfig.from_dict(receipt["intervention_config"])
        if intervention_enabled
        else None
    )
    if intervention_config is not None:
        intervention_config.validate_depth(depth)
    if intervention_enabled and receipt["measurement_trust_boundary"] != (
        INTERVENTION_MEASUREMENT_TRUST_BOUNDARY
    ):
        raise ValueError("exact-adjoint intervention measurement boundary is invalid")
    input_payload = {
        key: receipt[key]
        for key in (
            "policy_sha256",
            "prompt_tokens_sha256",
            "prompt_token_count",
            "answer_tokens_sha256",
            "answer_token_count",
            "bridge_tokens_sha256",
            "bridge_token_count",
            "token_loss_weights",
            "execution_spec_sha256",
            "recurrent_depth",
            "execution_branch_count",
            "branch_indices",
            "diversity_weight",
            "diversity_target_cos",
            "trajectory_config",
        )
    }
    if intervention_enabled:
        input_payload["intervention_config"] = receipt["intervention_config"]
        input_payload["measurement_trust_boundary"] = receipt["measurement_trust_boundary"]
    if auxiliary_only:
        input_payload["terminal_objective_weight"] = terminal_weight
    if receipt["objective_input_sha256"] != _exact_adjoint_input_sha256(input_payload):
        raise ValueError("exact-adjoint objective input commitment mismatch")
    step_losses = receipt["step_losses"]
    if not isinstance(step_losses, Mapping):
        raise ValueError("exact-adjoint step losses must be a mapping")
    normalized_steps: dict[int, list[Any]] = {}
    for key, losses in step_losses.items():
        if (
            not isinstance(key, str)
            or not key.isdigit()
            or key != str(int(key))
            or int(key) < 1
            or not isinstance(losses, list)
        ):
            raise ValueError("exact-adjoint step-loss row is invalid")
        normalized_steps[int(key)] = losses
        if len(losses) != len(branches):
            raise ValueError("exact-adjoint step-loss branches do not align")
        for loss in losses:
            if finite_number(loss, role="step loss") < 0.0:
                raise ValueError("exact-adjoint step loss is negative")
    expected_steps: set[int] = set()
    if config is not None and float(config.improvement_weight) > 0.0:
        expected_steps.update(config.probe_steps)
    if intervention_config is not None:
        if float(intervention_config.causality_weight) > 0.0:
            expected_steps.add(depth)
        if float(intervention_config.stopping_weight) > 0.0:
            expected_steps.update(intervention_config.stopping_steps)
    if set(normalized_steps) != expected_steps:
        raise ValueError("exact-adjoint step-loss probes do not match the config")

    normalized_lesions: dict[int, list[Any]] = {}
    stopping_receipts: list[Any] = []
    if intervention_enabled:
        lesion_losses = receipt["lesion_losses"]
        if not isinstance(lesion_losses, Mapping):
            raise ValueError("exact-adjoint lesion losses must be a mapping")
        for key, losses in lesion_losses.items():
            if (
                not isinstance(key, str)
                or not key.isdigit()
                or key != str(int(key))
                or int(key) < 1
                or not isinstance(losses, list)
                or len(losses) != len(branches)
            ):
                raise ValueError("exact-adjoint lesion-loss row is invalid")
            for loss in losses:
                if finite_number(loss, role="lesion loss") < 0.0:
                    raise ValueError("exact-adjoint lesion loss is negative")
            normalized_lesions[int(key)] = losses
        expected_lesions = (
            set(intervention_config.lesion_steps)
            if intervention_config is not None and float(intervention_config.causality_weight) > 0.0
            else set()
        )
        if set(normalized_lesions) != expected_lesions:
            raise ValueError("exact-adjoint lesion steps do not match the config")
        stopping_receipts = receipt["stopping_teacher_receipts"]
        expected_teacher_count = (
            len(branches)
            if intervention_config is not None and float(intervention_config.stopping_weight) > 0.0
            else 0
        )
        if (
            not isinstance(stopping_receipts, list)
            or len(stopping_receipts) != expected_teacher_count
            or any(not isinstance(item, Mapping) for item in stopping_receipts)
        ):
            raise ValueError("exact-adjoint stopping teacher cardinality is invalid")

    for role in ("displacements", "oscillation_cosines", "diversity_cosines"):
        sequence = receipt[role]
        if not isinstance(sequence, list):
            raise ValueError(f"exact-adjoint {role} must be a list")
        for item in sequence:
            normalized = finite_number(item, role=role)
            if role == "displacements" and normalized < 0.0:
                raise ValueError("exact-adjoint displacement is negative")
            if role != "displacements" and not -1.000001 <= normalized <= 1.000001:
                raise ValueError(f"exact-adjoint {role} is outside cosine range")
    expected_diversity_count = branch_count * (branch_count - 1) // 2
    if len(receipt["diversity_cosines"]) != expected_diversity_count:
        raise ValueError("exact-adjoint diversity cardinality is invalid")
    replayed_diversity = (
        diversity_weight
        * sum(
            max(float(cosine) - diversity_target, 0.0) ** 2
            for cosine in receipt["diversity_cosines"]
        )
        / expected_diversity_count
        if expected_diversity_count
        else 0.0
    )
    if not math.isclose(
        diversity,
        replayed_diversity,
        rel_tol=0.0,
        # The producer evaluates the penalty in MLX float32 while this replay
        # uses the sealed Python floats. The tolerance covers that one
        # representation crossing, not a statistical or model-level margin.
        abs_tol=1e-6,
    ):
        raise ValueError("exact-adjoint diversity does not replay")
    expected_displacements = (
        depth * len(branches)
        if config is not None and float(config.displacement_weight) > 0.0
        else 0
    )
    expected_oscillations = (
        (depth - 1) * len(branches)
        if config is not None and float(config.oscillation_weight) > 0.0
        else 0
    )
    if len(receipt["displacements"]) != expected_displacements:
        raise ValueError("exact-adjoint displacement cardinality is invalid")
    if len(receipt["oscillation_cosines"]) != expected_oscillations:
        raise ValueError("exact-adjoint oscillation cardinality is invalid")
    if config is None and any(
        abs(term_values[name]) > 0.0 for name in ("improvement", "displacement", "oscillation")
    ):
        raise ValueError("exact-adjoint receipt has terms without a trajectory config")
    if intervention_config is None and any(
        abs(term_values.get(name, 0.0)) > 0.0 for name in ("causality", "stopping")
    ):
        raise ValueError("exact-adjoint receipt has terms without an intervention config")
    replayed_improvement = 0.0
    replayed_displacement = 0.0
    replayed_oscillation = 0.0
    replayed_causality = 0.0
    replayed_stopping = 0.0
    if config is not None:
        if float(config.improvement_weight) > 0.0:
            hinges = [
                max(
                    float(normalized_steps[current][branch])
                    - float(normalized_steps[previous][branch])
                    + float(config.improvement_margin),
                    0.0,
                )
                for previous, current in zip(
                    config.probe_steps,
                    config.probe_steps[1:],
                    strict=False,
                )
                for branch in range(len(branches))
            ]
            replayed_improvement = float(config.improvement_weight) * sum(hinges) / len(hinges)
        if float(config.displacement_weight) > 0.0:
            replayed_displacement = (
                float(config.displacement_weight)
                * sum(
                    max(
                        float(config.displacement_floor) - float(displacement),
                        0.0,
                    )
                    for displacement in receipt["displacements"]
                )
                / len(receipt["displacements"])
            )
        if float(config.oscillation_weight) > 0.0:
            replayed_oscillation = (
                float(config.oscillation_weight)
                * sum(max(-float(cosine), 0.0) for cosine in receipt["oscillation_cosines"])
                / len(receipt["oscillation_cosines"])
            )
    if intervention_config is not None:
        if float(intervention_config.causality_weight) > 0.0:
            intact = normalized_steps[depth]
            hinges = [
                max(
                    float(intact[branch])
                    - float(normalized_lesions[step][branch])
                    + float(intervention_config.causality_margin),
                    0.0,
                )
                for step in intervention_config.lesion_steps
                for branch in range(len(branches))
            ]
            replayed_causality = (
                float(intervention_config.causality_weight) * sum(hinges) / len(hinges)
            )
        if float(intervention_config.stopping_weight) > 0.0:
            from core.learning.adaptive_halting import (
                validate_verified_stopping_teacher_receipt,
            )

            validated_teachers = [
                validate_verified_stopping_teacher_receipt(dict(item)) for item in stopping_receipts
            ]
            for branch, teacher in enumerate(validated_teachers):
                if (
                    teacher["steps"] != list(intervention_config.stopping_steps)
                    or any(
                        not math.isclose(
                            float(loss),
                            float(normalized_steps[step][branch]),
                            rel_tol=0.0,
                            abs_tol=1e-6,
                        )
                        for step, loss in zip(
                            intervention_config.stopping_steps,
                            teacher["losses"],
                            strict=True,
                        )
                    )
                    or not math.isclose(
                        float(teacher["ponder_cost"]),
                        float(intervention_config.stopping_ponder_cost),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        float(teacher["temperature"]),
                        float(intervention_config.stopping_temperature),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError("exact-adjoint stopping teacher differs from config")
            replayed_stopping = (
                float(intervention_config.stopping_weight)
                * sum(float(teacher["expected_risk"]) for teacher in validated_teachers)
                / len(validated_teachers)
            )
    for name, replayed in (
        ("improvement", replayed_improvement),
        ("displacement", replayed_displacement),
        ("oscillation", replayed_oscillation),
        *(
            (
                ("causality", replayed_causality),
                ("stopping", replayed_stopping),
            )
            if intervention_enabled
            else ()
        ),
    ):
        if not math.isclose(
            term_values[name],
            replayed,
            rel_tol=0.0,
            # State-derived atoms cross MLX float32 before Python replay.
            abs_tol=1e-6,
        ):
            raise ValueError(f"exact-adjoint {name} term does not replay")
    return dict(value)


@dataclass
class _LayerCheckpointState:
    model: Any
    parameters: Any
    group_size: int
    wrappers: dict[
        tuple[Any, ...],
        Callable[..., Any],
    ]
    cached_wrappers: dict[tuple[Any, ...], Callable[..., Any]]
    transition_wrappers: dict[int, Callable[..., Any]]


_LAYER_CHECKPOINTS: ContextVar[_LayerCheckpointState | None] = ContextVar(
    "aura_recurrence_layer_checkpoints",
    default=None,
)

# Per-step phase scale. A ContextVar rather than an RLCExecutionSpec field
# on purpose: the spec is hash-bound into adapter identity receipts, so
# adding a key would invalidate every existing bundle. Default 0.0 keeps
# behavior bit-identical unless a caller explicitly opts in.
_PHASE_SCALE: ContextVar[float] = ContextVar(
    "aura_recurrence_phase_scale",
    default=0.0,
)


@contextmanager
def recurrent_phase(scale: float) -> Iterator[None]:
    """Give each recurrent step an identity for the duration of a forward."""
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not 0.0 <= float(scale) <= 1.0
    ):
        raise ValueError("phase scale must be inside [0, 1]")
    token = _PHASE_SCALE.set(float(scale))
    try:
        yield
    finally:
        _PHASE_SCALE.reset(token)


@dataclass(frozen=True)
class LivePathForward:
    """Differentiable outputs and structural evidence from one depth."""

    branch_logits: tuple[Any, ...]
    branch_states: tuple[Any, ...]
    exchanges: int
    prompt_tokens: int
    answer_tokens: int
    bridge_tokens: int
    loop_core: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CachedSupervisedLivePathResult:
    """One streamed teacher-forced update on the resident cached backend."""

    value: float
    gradients: Any
    branch_values: tuple[float, ...]
    branch_indices: tuple[int, ...]
    answer_token_count: int
    execution_spec_sha256: str
    prompt_tokens_sha256: str
    answer_tokens_sha256: str
    bridge_tokens_sha256: str


@dataclass(frozen=True, slots=True)
class CachedSupervisedLivePathEvaluation:
    """Teacher-forced CE measured on the exact resident cached backend."""

    value: float
    branch_values: tuple[float, ...]
    branch_indices: tuple[int, ...]
    answer_token_count: int
    execution_spec_sha256: str
    prompt_tokens_sha256: str
    answer_tokens_sha256: str
    bridge_tokens_sha256: str


@dataclass(frozen=True, slots=True)
class CachedLivePathRollin:
    """Deterministic generated prefix from one resident cached branch."""

    tokens: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    branch_index: int
    seed: int
    temperature: float
    execution_spec_sha256: str
    prompt_tokens_sha256: str
    tokens_sha256: str


@dataclass(frozen=True)
class _PreparedLivePath:
    prompt_embeddings: Any
    tail_embeddings: Any
    seeds: tuple[Any, ...]
    prompts_at_window: tuple[Any, ...]
    states: tuple[Any, ...]
    anchors: tuple[Any, ...]
    prelude_end: int
    coda_start: int
    prompt_count: int
    bridge_count: int
    answer_count: int


@dataclass(frozen=True, slots=True)
class PreparedFinalRecurrentTransition:
    """Frozen parent and child ensembles around the final recurrent update.

    The object carries tensors for immediate decode and a tensor-free receipt
    for durable custody.  ``child_states`` are computed from ``parent_states``
    by exactly one invocation of the live transition operator; neither state
    is reconstructed from text or from a second independent episode.
    """

    prompt_embeddings: Any
    seeds: tuple[Any, ...]
    parent_states: tuple[Any, ...]
    child_states: tuple[Any, ...]
    prelude_end: int
    coda_start: int
    transition_index: int
    execution_spec_sha256: str
    prompt_tokens_sha256: str
    parent_branch_sha256s: tuple[str, ...]
    child_branch_sha256s: tuple[str, ...]
    parent_ensemble_sha256: str
    child_ensemble_sha256: str
    transition_source_sha256: str
    receipt_sha256: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": RECURRENT_TRANSITION_STATE_SCHEMA,
            "execution_spec_sha256": self.execution_spec_sha256,
            "prompt_tokens_sha256": self.prompt_tokens_sha256,
            "transition_index": self.transition_index,
            "parent_depth": self.transition_index,
            "child_depth": self.transition_index + 1,
            "branch_count": len(self.parent_states),
            "parent_branch_sha256s": list(self.parent_branch_sha256s),
            "child_branch_sha256s": list(self.child_branch_sha256s),
            "parent_ensemble_sha256": self.parent_ensemble_sha256,
            "child_ensemble_sha256": self.child_ensemble_sha256,
            "transition_source_sha256": self.transition_source_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class PreparedRecurrentStateTrail:
    """Frozen live-path states at depth zero and every recurrent boundary."""

    states_by_depth: tuple[tuple[Any, ...], ...]
    execution_spec_sha256: str
    prompt_tokens_sha256: str
    depth_branch_sha256s: tuple[tuple[str, ...], ...]
    transition_source_sha256: str
    receipt_sha256: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": RECURRENT_STATE_TRAIL_SCHEMA,
            "execution_spec_sha256": self.execution_spec_sha256,
            "prompt_tokens_sha256": self.prompt_tokens_sha256,
            "recurrent_steps": len(self.states_by_depth) - 1,
            "branch_count": len(self.states_by_depth[0]),
            "depth_branch_sha256s": [
                list(branches) for branches in self.depth_branch_sha256s
            ],
            "transition_source_sha256": self.transition_source_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class PreparedRecurrentTransitionInput:
    """Student-induced parent state for one exact live recurrent update."""

    prompts_at_window: tuple[Any, ...]
    parent_states: tuple[Any, ...]
    anchors: tuple[Any, ...]
    prelude_end: int
    coda_start: int
    transition_index: int
    execution_spec_sha256: str
    prompt_tokens_sha256: str
    parent_branch_sha256s: tuple[str, ...]
    transition_source_sha256: str
    receipt_sha256: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": RECURRENT_TRANSITION_INPUT_SCHEMA,
            "execution_spec_sha256": self.execution_spec_sha256,
            "prompt_tokens_sha256": self.prompt_tokens_sha256,
            "transition_index": self.transition_index,
            "branch_count": len(self.parent_states),
            "parent_branch_sha256s": list(self.parent_branch_sha256s),
            "transition_source_sha256": self.transition_source_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


def _boundaries(model: Any, spec: RLCExecutionSpec) -> tuple[int, int, int]:
    n_layers = len(model.model.layers)
    prelude_end = max(1, int(n_layers * spec.prelude_frac))
    coda_start = min(n_layers - 1, n_layers - int(n_layers * spec.coda_frac))
    if coda_start - prelude_end < 1:
        raise ValueError("execution spec leaves no recurrent layer window")
    return n_layers, prelude_end, coda_start


def _logits(model: Any, hidden: Any) -> Any:
    inner = model.model
    hidden = inner.norm(hidden)
    head = getattr(model, "lm_head", None)
    if head is not None and not isinstance(head, type(inner.embed_tokens)):
        return head(hidden)
    return inner.embed_tokens.as_linear(hidden)


@contextmanager
def transformer_layer_group_checkpointing(
    model: Any,
    parameters: Any,
    *,
    group_size: int = 4,
) -> Iterator[None]:
    """Rematerialize bounded layer groups while preserving graph semantics."""

    layers = tuple(model.model.layers)
    if not layers:
        raise ValueError("model has no transformer layers")
    if type(group_size) is not int or not 1 <= group_size <= len(layers):
        raise ValueError("group_size must be inside [1, model layer count]")
    token = _LAYER_CHECKPOINTS.set(
        _LayerCheckpointState(
            model=model,
            parameters=parameters,
            group_size=group_size,
            wrappers={},
            cached_wrappers={},
            transition_wrappers={},
        )
    )
    try:
        yield
    finally:
        _LAYER_CHECKPOINTS.reset(token)


def _causal_layers(layers: Sequence[Any], hidden: Any) -> Any:
    from mlx_lm.models.base import create_attention_mask

    layer_sequence = tuple(layers)
    if not layer_sequence:
        return hidden
    checkpointed = _LAYER_CHECKPOINTS.get()
    if checkpointed is None:
        for layer in layer_sequence:
            hidden = layer(hidden, create_attention_mask(hidden, None), None)
        return hidden
    activation = current_recurrence_adapter_scope()
    start = activation.start if activation is not None else None
    stop = activation.stop if activation is not None else None
    branch_index = current_branch_index()
    depth_index = current_depth_index()
    import mlx.core as mx

    for offset in range(0, len(layer_sequence), checkpointed.group_size):
        group = layer_sequence[offset : offset + checkpointed.group_size]
        key = (
            tuple(id(layer) for layer in group),
            start,
            stop,
            branch_index,
            depth_index,
        )
        call = checkpointed.wrappers.get(key)
        if call is None:

            def layer_group_call(
                all_parameters: Any,
                value: Any,
                _group: tuple[Any, ...] = group,
                _start: int | None = start,
                _stop: int | None = stop,
                _branch_index: int | None = branch_index,
                _depth_index: int = depth_index,
            ) -> Any:
                checkpointed.model.update(all_parameters)

                def run(current: Any) -> Any:
                    for member in _group:
                        current = member(
                            current,
                            create_attention_mask(current, None),
                            None,
                        )
                    return current

                adapter_scope = (
                    nullcontext()
                    if _start is None or _stop is None
                    else recurrence_adapter_scope(start=_start, stop=_stop)
                )
                branch_scope = (
                    nullcontext()
                    if _branch_index is None
                    else recurrent_branch_index(_branch_index)
                )
                with adapter_scope, branch_scope, recurrent_depth_index(_depth_index):
                    return run(value)

            call = mx.checkpoint(layer_group_call)
            checkpointed.wrappers[key] = call
        hidden = call(checkpointed.parameters, hidden)
    return hidden


def _cached_causal_layers(
    model: Any,
    hidden: Any,
    cache: Sequence[Any],
    *,
    start: int = 0,
    end: int | None = None,
) -> Any:
    """Run the exact cached kernel with explicit rematerializable KV state."""

    from mlx_lm.models.base import create_attention_mask

    stop = len(model.model.layers) if end is None else end
    if not 0 <= start < stop <= len(model.model.layers) or len(cache) != len(model.model.layers):
        raise ValueError("cached layer window is invalid")
    checkpointed = _LAYER_CHECKPOINTS.get()
    mask = create_attention_mask(hidden, cache[start:stop])
    if checkpointed is None:
        for index in range(start, stop):
            hidden = model.model.layers[index](hidden, mask, cache[index])
        return hidden

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    activation = current_recurrence_adapter_scope()
    adapter_active = activation is not None
    scope_start = activation.start if activation is not None else None
    scope_stop = activation.stop if activation is not None else None
    branch_index = current_branch_index()
    depth_index = current_depth_index()
    for index in range(start, stop):
        layer = model.model.layers[index]
        prior = None if cache[index].empty() else cache[index].state
        key = (
            "cached",
            id(layer),
            prior is not None,
            adapter_active,
            scope_start,
            scope_stop,
            branch_index,
            depth_index,
        )
        call = checkpointed.cached_wrappers.get(key)
        if call is None:

            def cached_layer_call(
                all_parameters: Any,
                value: Any,
                *cache_state: Any,
                _layer: Any = layer,
                _adapter_active: bool = adapter_active,
                _scope_start: int | None = scope_start,
                _scope_stop: int | None = scope_stop,
                _branch_index: int | None = branch_index,
                _depth_index: int = depth_index,
                _mask: Any = mask,
            ) -> tuple[Any, Any, Any]:
                checkpointed.model.update(all_parameters)
                local_cache = KVCache()
                if cache_state:
                    local_cache.state = (cache_state[0], cache_state[1])
                if not _adapter_active:
                    adapter_scope = nullcontext()
                elif _scope_start is None or _scope_stop is None:
                    adapter_scope = recurrence_adapter_scope()
                else:
                    adapter_scope = recurrence_adapter_scope(
                        start=_scope_start,
                        stop=_scope_stop,
                    )
                branch_scope = (
                    nullcontext()
                    if _branch_index is None
                    else recurrent_branch_index(_branch_index)
                )
                with adapter_scope, branch_scope, recurrent_depth_index(_depth_index):
                    output = _layer(value, _mask, local_cache)
                keys, values = local_cache.state
                return output, keys, values

            call = mx.checkpoint(cached_layer_call)
            checkpointed.cached_wrappers[key] = call
        output = call(
            checkpointed.parameters,
            hidden,
            *(prior or ()),
        )
        hidden, keys, values = output
        cache[index].state = (keys, values)
    return hidden


def _seed_branch(
    prompt_embeddings: Any,
    spec: RLCExecutionSpec,
    branch_role: str,
) -> Any:
    workspace = LatentWorkspace.from_prompt_embeddings(
        prompt_embeddings,
        WorkspaceConfig(
            n_slots=spec.n_slots,
            seed=spec.slot_seed,
            roles=spec.slot_roles,
            anchor_scale=spec.anchor_scale,
        ),
        branch_role=branch_role,
    )
    return workspace.seed_z


def _prelude_prompt_and_slots(
    model: Any,
    prompt_embeddings: Any,
    slot_seed: Any,
    prelude_end: int,
) -> tuple[Any, Any]:
    import mlx.core as mx

    prompt_length = int(prompt_embeddings.shape[1])
    hidden = mx.concatenate([prompt_embeddings, slot_seed], axis=1)
    hidden = _causal_layers(model.model.layers[:prelude_end], hidden)
    return hidden[:, :prompt_length, :], hidden[:, prompt_length:, :]


def recurrent_phase_code(step: int, hidden: int) -> Any:
    """Parameter-free sinusoidal code identifying a recurrent step (CP210).

    The recurrence applies the SAME operator every step, so no step can
    know which step it is and no staged algorithm (encode -> retrieve ->
    compare -> verify) is expressible. Measured consequence: the operator
    is a contraction (residual 0.302 -> 0.026, asymptoting) that reaches a
    fixed point by step ~10 and stops computing — which is why depth
    saturates at 8, why deeper mildly hurts, and why branches (all falling
    into the same fixed point) collapse.

    Injecting this code gives each step an identity, exactly as positional
    encoding differentiates otherwise-identical tokens. Measured on the
    untrained 1.5B over khop: best-depth CE 1.8072 -> 1.6958 (-6.2%), with
    the gain GROWING at depth (d4 -2.4%, d8 -6.2%, d16 -7.6%).

    It is an input-side signal, so it does not by itself break the
    contraction (residual ratio moved only 0.1142 -> 0.1048); a trained
    phase-conditioned OPERATOR is required for that. This is the free part.
    """
    import mlx.core as mx

    positions = mx.arange(hidden, dtype=mx.float32)
    frequency = mx.exp(-math.log(10000.0) * (2 * mx.floor(positions / 2)) / hidden)
    angle = float(step) * frequency
    return mx.where(positions % 2 == 0, mx.sin(angle), mx.cos(angle))


def _window_pass(
    model: Any,
    prompt_at_window: Any,
    slots: Any,
    prelude_end: int,
    coda_start: int,
    *,
    phase_step: int | None = None,
) -> Any:
    import mlx.core as mx

    phase_scale = _PHASE_SCALE.get()
    if phase_step is not None and phase_scale > 0.0:
        rms = mx.sqrt(mx.mean(mx.square(slots)) + 1e-9)
        code = recurrent_phase_code(phase_step, int(slots.shape[-1]))
        slots = slots + phase_scale * rms * code[None, None, :]
    prompt_length = int(prompt_at_window.shape[1])
    slot_count = int(slots.shape[1])
    prompt_hidden = prompt_at_window
    slot_hidden = slots
    with recurrence_adapter_scope(
        start=prompt_length,
        stop=prompt_length + slot_count,
    ):
        joined = mx.concatenate([prompt_hidden, slot_hidden], axis=1)
        joined = _causal_layers(model.model.layers[prelude_end:coda_start], joined)
        prompt_hidden = joined[:, :prompt_length, :]
        slot_hidden = joined[:, prompt_length:, :]
    return slot_hidden


def _alpha_at(spec: RLCExecutionSpec, step: int) -> float:
    return alpha_for_step(
        alpha=spec.alpha,
        schedule=spec.alpha_schedule,
        max_steps=spec.recurrent_steps,
        step=step,
    )


def _exchange_and_decorrelate(
    states: list[Any],
    spec: RLCExecutionSpec,
    step_number: int,
) -> list[Any]:
    import mlx.core as mx

    if len(states) < 2:
        return states
    source_slots = private_exchange_slots(
        n_slots=int(states[0].shape[1]),
        comm_slot=int(spec.comm_slot),
        context_slots=(),
    )
    if len(source_slots) > spec.exchange_source_slot_limit:
        raise ValueError("training exchange source exceeds execution spec")
    summaries = [
        mx.mean(
            mx.concatenate(
                [state[:, index : index + 1, :] for index in source_slots],
                axis=1,
            ),
            axis=1,
            keepdims=True,
        )
        for state in states
    ]
    stack = mx.concatenate(summaries, axis=1)
    mean = mx.mean(stack, axis=1, keepdims=True)

    def cosine(left: Any, right: Any) -> Any:
        denominator = mx.maximum(mx.linalg.norm(left) * mx.linalg.norm(right), 1e-6)
        return mx.sum(left * right) / denominator

    agreements = mx.stack([cosine(summary, mean) for summary in summaries])
    weights = mx.softmax(agreements, axis=0)
    consensus = sum(weight * summary for weight, summary in zip(weights, summaries, strict=True))
    slot = spec.comm_slot
    exchanged: list[Any] = []
    for state in states:
        comm = (1.0 - spec.exchange_gamma) * state[
            :, slot : slot + 1, :
        ] + spec.exchange_gamma * consensus
        exchanged.append(
            mx.concatenate(
                [state[:, :slot, :], comm, state[:, slot + 1 :, :]],
                axis=1,
            )
        )

    for left_index in range(len(exchanged)):
        for right_index in range(left_index + 1, len(exchanged)):
            left = exchanged[left_index]
            right = exchanged[right_index]
            similarity = cosine(
                mx.mean(left, axis=1, keepdims=True),
                mx.mean(right, axis=1, keepdims=True),
            )
            gate = (similarity > spec.collapse_cos_threshold).astype(right.dtype)
            key = mx.random.key(1000 + 31 * left_index + right_index + step_number)
            jitter = mx.random.normal(right.shape, key=key)
            jitter = jitter * (
                spec.jitter_scale
                * per_position_rms(right)
                / mx.maximum(per_position_rms(jitter), 1e-6)
            )
            exchanged[right_index] = right + gate * jitter
    return exchanged


def _advance_recurrent_states(
    model: Any,
    prompts_at_window: Sequence[Any],
    states: Sequence[Any],
    anchors: Sequence[Any],
    spec: RLCExecutionSpec,
    step: int,
    prelude_end: int,
    coda_start: int,
) -> list[Any]:
    updated: list[Any] = []
    alpha = _alpha_at(spec, step)
    for branch_index, (prompt_at_window, state, anchor) in enumerate(
        zip(
            prompts_at_window,
            states,
            anchors,
            strict=True,
        )
    ):
        # Publish the recurrent step so any attached depth-conditioned
        # operator bank selects this step's effective transform. A no-op
        # when no bank is attached.
        from core.learning.depth_conditioned_lora import recurrent_depth_index

        with recurrent_branch_index(branch_index), recurrent_depth_index(step):
            candidate = _window_pass(
                model,
                prompt_at_window,
                state,
                prelude_end,
                coda_start,
                phase_step=step,
            )
        updated.append(
            controlled_recurrent_update(
                state,
                candidate,
                anchor,
                alpha=alpha,
                clip_ratio=spec.rms_clip_ratio,
            )
        )
    if len(updated) > 1 and (step + 1) % spec.exchange_interval == 0:
        return _exchange_and_decorrelate(updated, spec, step + 1)
    return updated


def _checkpointed_recurrent_transition(
    model: Any,
    prompts_at_window: Sequence[Any],
    states: Sequence[Any],
    anchors: Sequence[Any],
    spec: RLCExecutionSpec,
    step: int,
    prelude_end: int,
    coda_start: int,
) -> list[Any]:
    checkpointed = _LAYER_CHECKPOINTS.get()
    if checkpointed is None:
        return _advance_recurrent_states(
            model,
            prompts_at_window,
            states,
            anchors,
            spec,
            step,
            prelude_end,
            coda_start,
        )
    branch_count = len(states)
    call = checkpointed.transition_wrappers.get(step)
    if call is None:
        import mlx.core as mx

        def transition(all_parameters: Any, *values: Any) -> tuple[Any, ...]:
            checkpointed.model.update(all_parameters)
            prompts = values[:branch_count]
            current_states = values[branch_count : 2 * branch_count]
            current_anchors = values[2 * branch_count :]
            # Keep layer checkpointing active inside the transition
            # checkpoint. The outer boundary bounds recurrence-depth
            # residency; the nested layer boundaries bound each transformer's
            # activation residency during backward replay.
            return tuple(
                _advance_recurrent_states(
                    checkpointed.model,
                    prompts,
                    current_states,
                    current_anchors,
                    spec,
                    step,
                    prelude_end,
                    coda_start,
                )
            )

        call = mx.checkpoint(transition)
        checkpointed.transition_wrappers[step] = call
    return list(
        call(
            checkpointed.parameters,
            *prompts_at_window,
            *states,
            *anchors,
        )
    )


def _persist_and_score(
    model: Any,
    prompt_embeddings: Any,
    slot_seed: Any,
    final_slots: Any,
    tail_embeddings: Any,
    *,
    branch_index: int = 0,
    bridge_count: int,
    answer_count: int,
    prelude_end: int,
    coda_start: int,
) -> Any:
    import mlx.core as mx

    prompt_length = int(prompt_embeddings.shape[1])
    slot_count = int(slot_seed.shape[1])
    hidden = mx.concatenate([prompt_embeddings, slot_seed, tail_embeddings], axis=1)
    hidden = _causal_layers(model.model.layers[:prelude_end], hidden)
    slot_start = prompt_length
    slot_stop = prompt_length + slot_count
    hidden = mx.concatenate(
        [hidden[:, :slot_start, :], final_slots, hidden[:, slot_stop:, :]],
        axis=1,
    )
    with (
        recurrent_branch_index(branch_index),
        recurrence_adapter_scope(
            start=slot_start,
            stop=slot_stop,
        ),
    ):
        hidden = _causal_layers(model.model.layers[prelude_end:coda_start], hidden)
    with coda_adapter_scope(start=slot_start, stop=int(hidden.shape[1])):
        hidden = _causal_layers(model.model.layers[coda_start:], hidden)
    all_logits = _logits(model, hidden)
    answer_start = prompt_length + slot_count + bridge_count
    prediction_start = answer_start - 1
    return all_logits[:, prediction_start : prediction_start + answer_count, :]


def _token_sequence_sha256(tokens: Sequence[int]) -> str:
    encoded = json.dumps(list(tokens), separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: Any) -> str:
    import mlx.core as mx
    import numpy as np

    mx.eval(value)
    try:
        array = np.asarray(value)
    except RuntimeError:
        array = np.asarray(value.astype(mx.float32))
    digest = hashlib.sha256()
    for part in (
        str(value.dtype).encode("ascii"),
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _ensemble_sha256(branch_sha256s: Sequence[str]) -> str:
    encoded = json.dumps(list(branch_sha256s), separators=(",", ":"), allow_nan=False).encode(
        "ascii"
    )
    return hashlib.sha256(b"aura.recurrent_ensemble.v1\0" + encoded).hexdigest()


def _seal_transition_receipt(body: dict[str, Any]) -> str:
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_recurrent_prefix(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
) -> tuple[
    Any,
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
    int,
    int,
]:
    import mlx.core as mx

    problems = spec.validate()
    if problems:
        raise ValueError(f"invalid execution spec: {problems}")
    prompt = list(prompt_tokens)
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("prompt_tokens must contain non-negative integers")
    _n_layers, prelude_end, coda_start = _boundaries(model, spec)
    prompt_embeddings = model.model.embed_tokens(mx.array([prompt]))
    seeds: list[Any] = []
    prompts_at_window: list[Any] = []
    states: list[Any] = []
    anchors: list[Any] = []
    for role in spec.branch_roles:
        seed = _seed_branch(prompt_embeddings, spec, role)
        prompt_at_window, state = _prelude_prompt_and_slots(
            model,
            prompt_embeddings,
            seed,
            prelude_end,
        )
        seeds.append(seed)
        prompts_at_window.append(prompt_at_window)
        states.append(state)
        anchors.append(state)
    return (
        prompt_embeddings,
        tuple(seeds),
        tuple(prompts_at_window),
        tuple(states),
        tuple(anchors),
        prelude_end,
        coda_start,
    )


def _prepare_live_path(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int],
) -> _PreparedLivePath:
    import mlx.core as mx

    prompt = list(prompt_tokens)
    answer = list(answer_tokens)
    bridge = list(bridge_tokens)
    if not answer or any(type(token) is not int or token < 0 for token in answer):
        raise ValueError("answer_tokens must contain non-negative integers")
    if any(type(token) is not int or token < 0 for token in bridge):
        raise ValueError("bridge_tokens must contain non-negative integers")
    if spec.decode_bridge_policy == "none" and bridge:
        raise ValueError("bridge tokens supplied while decode bridge is disabled")
    if spec.decode_bridge_policy != "none" and not bridge:
        raise ValueError("execution spec requires decode bridge tokens")

    (
        prompt_embeddings,
        seeds,
        prompts_at_window,
        states,
        anchors,
        prelude_end,
        coda_start,
    ) = _prepare_recurrent_prefix(model, prompt, spec=spec)
    tail_embeddings = model.model.embed_tokens(mx.array([bridge + answer]))
    return _PreparedLivePath(
        prompt_embeddings=prompt_embeddings,
        tail_embeddings=tail_embeddings,
        seeds=seeds,
        prompts_at_window=prompts_at_window,
        states=states,
        anchors=anchors,
        prelude_end=prelude_end,
        coda_start=coda_start,
        prompt_count=len(prompt),
        bridge_count=len(bridge),
        answer_count=len(answer),
    )


def prepare_final_recurrent_transition(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
) -> PreparedFinalRecurrentTransition:
    """Freeze ``S[k]`` and ``S[k+1]`` around the final configured update."""

    import mlx.core as mx

    (
        prompt_embeddings,
        seeds,
        prompts_at_window,
        initial_states,
        anchors,
        prelude_end,
        coda_start,
    ) = _prepare_recurrent_prefix(model, prompt_tokens, spec=spec)
    states = list(initial_states)
    transition_index = spec.recurrent_steps - 1
    for step in range(transition_index):
        states = _checkpointed_recurrent_transition(
            model,
            prompts_at_window,
            states,
            anchors,
            spec,
            step,
            prelude_end,
            coda_start,
        )
    parent_states = tuple(mx.stop_gradient(state) for state in states)
    mx.eval(parent_states)
    child = _checkpointed_recurrent_transition(
        model,
        prompts_at_window,
        parent_states,
        anchors,
        spec,
        transition_index,
        prelude_end,
        coda_start,
    )
    child_states = tuple(mx.stop_gradient(state) for state in child)
    mx.eval(child_states)
    parent_branch_sha256s = tuple(_tensor_sha256(state) for state in parent_states)
    child_branch_sha256s = tuple(_tensor_sha256(state) for state in child_states)
    transition_source_sha256 = hashlib.sha256(
        inspect.getsource(_advance_recurrent_states).encode("utf-8")
    ).hexdigest()
    body = {
        "schema": RECURRENT_TRANSITION_STATE_SCHEMA,
        "execution_spec_sha256": spec.sha256,
        "prompt_tokens_sha256": _token_sequence_sha256(prompt_tokens),
        "transition_index": transition_index,
        "parent_depth": transition_index,
        "child_depth": transition_index + 1,
        "branch_count": len(parent_states),
        "parent_branch_sha256s": list(parent_branch_sha256s),
        "child_branch_sha256s": list(child_branch_sha256s),
        "parent_ensemble_sha256": _ensemble_sha256(parent_branch_sha256s),
        "child_ensemble_sha256": _ensemble_sha256(child_branch_sha256s),
        "transition_source_sha256": transition_source_sha256,
    }
    receipt_sha256 = _seal_transition_receipt(body)
    return PreparedFinalRecurrentTransition(
        prompt_embeddings=mx.stop_gradient(prompt_embeddings),
        seeds=tuple(mx.stop_gradient(seed) for seed in seeds),
        parent_states=parent_states,
        child_states=child_states,
        prelude_end=prelude_end,
        coda_start=coda_start,
        transition_index=transition_index,
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=body["prompt_tokens_sha256"],
        parent_branch_sha256s=parent_branch_sha256s,
        child_branch_sha256s=child_branch_sha256s,
        parent_ensemble_sha256=body["parent_ensemble_sha256"],
        child_ensemble_sha256=body["child_ensemble_sha256"],
        transition_source_sha256=transition_source_sha256,
        receipt_sha256=receipt_sha256,
    )


def prepare_recurrent_state_trail(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
) -> PreparedRecurrentStateTrail:
    """Capture the exact shared live transition trajectory without decoding."""

    import mlx.core as mx

    (
        _prompt_embeddings,
        _seeds,
        prompts_at_window,
        initial_states,
        anchors,
        prelude_end,
        coda_start,
    ) = _prepare_recurrent_prefix(model, prompt_tokens, spec=spec)

    def freeze(states: Sequence[Any]) -> tuple[Any, ...]:
        frozen = tuple(mx.stop_gradient(state) for state in states)
        mx.eval(frozen)
        return frozen

    current = freeze(initial_states)
    history: list[tuple[Any, ...]] = [current]
    for step in range(spec.recurrent_steps):
        current = freeze(
            _checkpointed_recurrent_transition(
                model,
                prompts_at_window,
                current,
                anchors,
                spec,
                step,
                prelude_end,
                coda_start,
            )
        )
        history.append(current)
    depth_hashes = tuple(
        tuple(_tensor_sha256(state) for state in states) for states in history
    )
    transition_source_sha256 = hashlib.sha256(
        inspect.getsource(_advance_recurrent_states).encode("utf-8")
    ).hexdigest()
    body = {
        "schema": RECURRENT_STATE_TRAIL_SCHEMA,
        "execution_spec_sha256": spec.sha256,
        "prompt_tokens_sha256": _token_sequence_sha256(prompt_tokens),
        "recurrent_steps": spec.recurrent_steps,
        "branch_count": len(current),
        "depth_branch_sha256s": [list(branches) for branches in depth_hashes],
        "transition_source_sha256": transition_source_sha256,
    }
    receipt_sha256 = _seal_transition_receipt(body)
    return PreparedRecurrentStateTrail(
        states_by_depth=tuple(history),
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=body["prompt_tokens_sha256"],
        depth_branch_sha256s=depth_hashes,
        transition_source_sha256=transition_source_sha256,
        receipt_sha256=receipt_sha256,
    )


def prepare_recurrent_transition_input(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    transition_index: int,
) -> PreparedRecurrentTransitionInput:
    """Freeze the state reached by the current policy before one transition.

    Recomputing this boundary after each optimizer update yields a
    student-induced roll-in rather than teacher forcing on stale latent states.
    The recurrent adapters are the only trainable model surface, so the frozen
    prelude/context tensors remain valid for the immediately following update.
    """

    import mlx.core as mx

    if (
        type(transition_index) is not int
        or transition_index < 0
        or transition_index >= spec.recurrent_steps
    ):
        raise ValueError("transition index is outside the execution spec")
    (
        _prompt_embeddings,
        _seeds,
        prompts_at_window,
        initial_states,
        anchors,
        prelude_end,
        coda_start,
    ) = _prepare_recurrent_prefix(model, prompt_tokens, spec=spec)

    def freeze(values: Sequence[Any]) -> tuple[Any, ...]:
        frozen = tuple(mx.stop_gradient(value) for value in values)
        mx.eval(frozen)
        return frozen

    prompts = freeze(prompts_at_window)
    current = freeze(initial_states)
    stable_anchors = freeze(anchors)
    for step in range(transition_index):
        current = freeze(
            _checkpointed_recurrent_transition(
                model,
                prompts,
                current,
                stable_anchors,
                spec,
                step,
                prelude_end,
                coda_start,
            )
        )
    parent_hashes = tuple(_tensor_sha256(state) for state in current)
    transition_source_sha256 = hashlib.sha256(
        inspect.getsource(_advance_recurrent_states).encode("utf-8")
    ).hexdigest()
    body = {
        "schema": RECURRENT_TRANSITION_INPUT_SCHEMA,
        "execution_spec_sha256": spec.sha256,
        "prompt_tokens_sha256": _token_sequence_sha256(prompt_tokens),
        "transition_index": transition_index,
        "branch_count": len(current),
        "parent_branch_sha256s": list(parent_hashes),
        "transition_source_sha256": transition_source_sha256,
    }
    return PreparedRecurrentTransitionInput(
        prompts_at_window=prompts,
        parent_states=current,
        anchors=stable_anchors,
        prelude_end=prelude_end,
        coda_start=coda_start,
        transition_index=transition_index,
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=body["prompt_tokens_sha256"],
        parent_branch_sha256s=parent_hashes,
        transition_source_sha256=transition_source_sha256,
        receipt_sha256=_seal_transition_receipt(body),
    )


def execute_prepared_recurrent_transition(
    model: Any,
    prepared: PreparedRecurrentTransitionInput,
    *,
    spec: RLCExecutionSpec,
) -> tuple[Any, ...]:
    """Differentiate one live transition from a sealed student roll-in."""

    if not isinstance(prepared, PreparedRecurrentTransitionInput):
        raise TypeError("prepared transition input has the wrong type")
    if prepared.execution_spec_sha256 != spec.sha256:
        raise ValueError("prepared transition execution spec differs")
    if prepared.transition_index >= spec.recurrent_steps:
        raise ValueError("prepared transition index exceeds execution depth")
    source_sha256 = hashlib.sha256(
        inspect.getsource(_advance_recurrent_states).encode("utf-8")
    ).hexdigest()
    if prepared.transition_source_sha256 != source_sha256:
        raise ValueError("prepared transition implementation differs")
    if not (
        len(prepared.prompts_at_window)
        == len(prepared.parent_states)
        == len(prepared.anchors)
        == len(spec.branch_roles)
    ):
        raise ValueError("prepared transition branches differ")
    return tuple(
        _checkpointed_recurrent_transition(
            model,
            prepared.prompts_at_window,
            prepared.parent_states,
            prepared.anchors,
            spec,
            prepared.transition_index,
            prepared.prelude_end,
            prepared.coda_start,
        )
    )


def validate_recurrent_transition_input_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the tensor-free commitment to a prepared transition parent."""

    required = {
        "schema",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "transition_index",
        "branch_count",
        "parent_branch_sha256s",
        "transition_source_sha256",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise ValueError("prepared transition receipt fields differ")
    normalized = json.loads(
        json.dumps(
            dict(receipt),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    observed = normalized.pop("receipt_sha256")
    if (
        normalized.get("schema") != RECURRENT_TRANSITION_INPUT_SCHEMA
        or not _valid_sha256(normalized.get("execution_spec_sha256"))
        or not _valid_sha256(normalized.get("prompt_tokens_sha256"))
        or not _valid_sha256(normalized.get("transition_source_sha256"))
        or type(normalized.get("transition_index")) is not int
        or normalized["transition_index"] < 0
        or type(normalized.get("branch_count")) is not int
        or normalized["branch_count"] < 1
        or not isinstance(normalized.get("parent_branch_sha256s"), list)
        or len(normalized["parent_branch_sha256s"]) != normalized["branch_count"]
        or any(
            not _valid_sha256(value)
            for value in normalized["parent_branch_sha256s"]
        )
        or not _valid_sha256(observed)
        or _seal_transition_receipt(normalized) != observed
    ):
        raise ValueError("prepared transition receipt is invalid")
    return {**normalized, "receipt_sha256": observed}


def validate_recurrent_state_trail_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay structural and digest integrity for a tensor-free state trail."""

    required = {
        "schema",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "recurrent_steps",
        "branch_count",
        "depth_branch_sha256s",
        "transition_source_sha256",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise ValueError("recurrent state trail receipt fields differ")
    normalized = json.loads(
        json.dumps(
            dict(receipt),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    observed = normalized.pop("receipt_sha256")
    steps = normalized.get("recurrent_steps")
    branch_count = normalized.get("branch_count")
    depth_hashes = normalized.get("depth_branch_sha256s")
    if (
        normalized.get("schema") != RECURRENT_STATE_TRAIL_SCHEMA
        or not _valid_sha256(normalized.get("execution_spec_sha256"))
        or not _valid_sha256(normalized.get("prompt_tokens_sha256"))
        or not _valid_sha256(normalized.get("transition_source_sha256"))
        or type(steps) is not int
        or steps < 1
        or type(branch_count) is not int
        or branch_count < 1
        or not isinstance(depth_hashes, list)
        or len(depth_hashes) != steps + 1
        or any(
            not isinstance(branches, list)
            or len(branches) != branch_count
            or any(not _valid_sha256(value) for value in branches)
            for branches in depth_hashes
        )
        or not _valid_sha256(observed)
        or _seal_transition_receipt(normalized) != observed
    ):
        raise ValueError("recurrent state trail receipt is invalid")
    return {**normalized, "receipt_sha256": observed}


def validate_final_recurrent_transition_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the durable, tensor-free edge certificate."""

    required = {
        "schema",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "transition_index",
        "parent_depth",
        "child_depth",
        "branch_count",
        "parent_branch_sha256s",
        "child_branch_sha256s",
        "parent_ensemble_sha256",
        "child_ensemble_sha256",
        "transition_source_sha256",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise ValueError("recurrent_transition_receipt_schema_invalid")
    normalized = dict(receipt)
    if normalized.get("schema") != RECURRENT_TRANSITION_STATE_SCHEMA:
        raise ValueError("recurrent_transition_receipt_version_invalid")
    branch_count = normalized.get("branch_count")
    transition_index = normalized.get("transition_index")
    parent = normalized.get("parent_branch_sha256s")
    child = normalized.get("child_branch_sha256s")
    digests = (
        normalized.get("execution_spec_sha256"),
        normalized.get("prompt_tokens_sha256"),
        normalized.get("parent_ensemble_sha256"),
        normalized.get("child_ensemble_sha256"),
        normalized.get("transition_source_sha256"),
        normalized.get("receipt_sha256"),
    )
    if (
        type(branch_count) is not int
        or branch_count < 1
        or type(transition_index) is not int
        or transition_index < 0
        or normalized.get("parent_depth") != transition_index
        or normalized.get("child_depth") != transition_index + 1
        or not isinstance(parent, list)
        or not isinstance(child, list)
        or len(parent) != branch_count
        or len(child) != branch_count
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (*digests, *parent, *child)
        )
        or normalized["parent_ensemble_sha256"] != _ensemble_sha256(parent)
        or normalized["child_ensemble_sha256"] != _ensemble_sha256(child)
    ):
        raise ValueError("recurrent_transition_receipt_identity_invalid")
    unsigned = dict(normalized)
    observed = unsigned.pop("receipt_sha256")
    if _seal_transition_receipt(unsigned) != observed:
        raise ValueError("recurrent_transition_receipt_digest_mismatch")
    return normalized


def live_path_forward(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
) -> LivePathForward:
    """Run the differentiable latent-slot path and return per-branch logits."""

    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
    )
    states = list(prepared.states)

    exchanges = 0
    for step in range(spec.recurrent_steps):
        states = _checkpointed_recurrent_transition(
            model,
            prepared.prompts_at_window,
            states,
            prepared.anchors,
            spec,
            step,
            prepared.prelude_end,
            prepared.coda_start,
        )
        if len(states) > 1 and (step + 1) % spec.exchange_interval == 0:
            exchanges += 1

    branch_logits = tuple(
        _persist_and_score(
            model,
            prepared.prompt_embeddings,
            seed,
            state,
            prepared.tail_embeddings,
            branch_index=branch_index,
            bridge_count=prepared.bridge_count,
            answer_count=prepared.answer_count,
            prelude_end=prepared.prelude_end,
            coda_start=prepared.coda_start,
        )
        for branch_index, (seed, state) in enumerate(zip(prepared.seeds, states, strict=True))
    )
    return LivePathForward(
        branch_logits=branch_logits,
        branch_states=tuple(states),
        exchanges=exchanges,
        prompt_tokens=prepared.prompt_count,
        answer_tokens=prepared.answer_count,
        bridge_tokens=prepared.bridge_count,
        loop_core=build_loop_core_contract(
            prelude_end=prepared.prelude_end,
            coda_start=prepared.coda_start,
            max_steps=spec.recurrent_steps,
            min_steps=spec.recurrent_steps,
            alpha=spec.alpha,
            alpha_schedule=spec.alpha_schedule,
            rms_clip_ratio=spec.rms_clip_ratio,
            convergence_eps=1e-9,
            divergence_ratio=1000.0,
            fixed_depth=True,
        ),
    )


def cached_live_path_token_logprobs(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    bridge_tokens: Sequence[int] = (),
    adapters_on: bool = True,
    rollin_tokens: Sequence[int] | None = None,
) -> Any:
    """Score fixed labels through the resident KV-cached recurrent policy.

    Quantized matrix kernels can produce materially different logits for a
    batched no-cache forward and token-at-a-time KV decoding. PPO behavior
    probabilities must therefore be recomputed by the exact backend that
    generated them. This path remains differentiable through the recurrent
    states, persisted slot window, cache prefill, and lexical decode.

    ``rollin_tokens`` separates labels from decoder inputs. When omitted, this
    is the historical teacher-forced objective. When supplied, answer token
    ``i`` is still the label at position ``i`` while roll-in token ``i`` is fed
    to produce the next position. This is the exact cached scheduled-sampling
    primitive; no generated token is silently relabeled as correct.
    """

    import mlx.core as mx

    answer = tuple(answer_tokens)
    bridge = tuple(bridge_tokens)
    rollin = answer if rollin_tokens is None else tuple(rollin_tokens)
    if not answer or any(type(token) is not int or token < 0 for token in answer):
        raise ValueError("answer_tokens must contain non-negative integers")
    if any(type(token) is not int or token < 0 for token in bridge):
        raise ValueError("bridge_tokens must contain non-negative integers")
    if len(rollin) != len(answer) or any(type(token) is not int or token < 0 for token in rollin):
        raise ValueError("rollin_tokens must be non-negative and answer-aligned")
    if spec.decode_bridge_policy == "none" and bridge:
        raise ValueError("bridge tokens supplied while decode bridge is disabled")
    if spec.decode_bridge_policy != "none" and not bridge:
        raise ValueError("execution spec requires decode bridge tokens")

    layers, cache, logits = _cached_live_path_initial_logits(
        model,
        prompt_tokens,
        spec=spec,
        branch_index=branch_index,
        decode_token_budget=len(bridge) + len(answer),
        adapters_on=adapters_on,
    )
    targets = (*bridge, *answer)
    decoder_inputs = (*bridge, *rollin)
    answer_logprobs: list[Any] = []
    for position, (target, decoder_input) in enumerate(zip(targets, decoder_inputs, strict=True)):
        logprob = logits[target].astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32))
        if position >= len(bridge):
            answer_logprobs.append(logprob)
        if position + 1 == len(targets):
            continue
        hidden = model.model.embed_tokens(mx.array([[decoder_input]]))
        with coda_adapter_scope():
            hidden = _cached_causal_layers(model, hidden, cache)
        logits = _logits(model, hidden)[0, -1]
    if len(answer_logprobs) != len(answer):
        raise RuntimeError("cached answer log-probabilities do not align")
    return mx.stack(answer_logprobs)


def _cached_live_path_initial_logits(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    decode_token_budget: int,
    adapters_on: bool,
) -> tuple[tuple[Any, ...], list[Any], Any]:
    """Create the exact resident cache and first lexical distribution."""


    from core.brain.llm.decoder_topology import decoder_layers
    from core.brain.llm.latent_cortex.recurrence import WindowRunner
    from core.brain.llm.latent_cortex.types import ComputeBudget
    from core.learning.intrinsic_recurrence import model_layer_caches

    if type(decode_token_budget) is not int or not 1 <= decode_token_budget <= 32_768:
        raise ValueError("decode_token_budget must be inside [1, 32768]")
    if type(branch_index) is not int or not 0 <= branch_index < len(spec.branch_roles):
        raise ValueError("branch_index is outside the live-path branch set")
    boundary = nullcontext() if adapters_on else recurrence_adapter_disabled()
    with boundary, recurrent_branch_index(branch_index):
        (
            prompt_embeddings,
            seeds,
            prompts_at_window,
            initial_states,
            anchors,
            prelude_end,
            coda_start,
        ) = _prepare_recurrent_prefix(model, prompt_tokens, spec=spec)
        if not 0 <= branch_index < len(seeds):
            raise ValueError("branch_index is outside the live-path branch set")
        states = list(initial_states)
        for step in range(spec.recurrent_steps):
            states = _checkpointed_recurrent_transition(
                model,
                prompts_at_window,
                states,
                anchors,
                spec,
                step,
                prelude_end,
                coda_start,
            )

        layers = tuple(decoder_layers(model))
        cache = model_layer_caches(model)
        _cached_causal_layers(model, prompt_embeddings, cache)

        budget = ComputeBudget(
            max_layer_apps=max(
                1,
                (len(prompt_tokens) + 2 * int(seeds[branch_index].shape[1]) + decode_token_budget)
                * len(layers),
            ),
            wall_clock_s=600.0,
        )
        if _LAYER_CHECKPOINTS.get() is None:
            runner = WindowRunner(model.model, budget)
            runner.run(seeds[branch_index], cache, 0, prelude_end, persist=True)
            persisted = runner.run(
                states[branch_index],
                cache,
                prelude_end,
                coda_start,
                persist=True,
            )
            with coda_adapter_scope():
                output = runner.run(
                    persisted,
                    cache,
                    coda_start,
                    len(layers),
                    persist=True,
                )
        else:
            with recurrence_adapter_scope():
                _cached_causal_layers(
                    model,
                    seeds[branch_index],
                    cache,
                    start=0,
                    end=prelude_end,
                )
                persisted = _cached_causal_layers(
                    model,
                    states[branch_index],
                    cache,
                    start=prelude_end,
                    end=coda_start,
                )
                with coda_adapter_scope():
                    output = _cached_causal_layers(
                        model,
                        persisted,
                        cache,
                        start=coda_start,
                        end=len(layers),
                    )
        logits = _logits(model, output)[0, -1]
    return layers, cache, logits


def generate_cached_live_path_rollin(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    token_count: int,
    seed: int,
    temperature: float = 1.0,
    bridge_tokens: Sequence[int] = (),
) -> CachedLivePathRollin:
    """Generate a deterministic branch-local roll-in on the resident backend."""

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    bridge = tuple(bridge_tokens)
    if type(token_count) is not int or not 1 <= token_count <= 32_768:
        raise ValueError("token_count must be inside [1, 32768]")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be inside [0, 2^32-1]")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 0.0 <= float(temperature) <= 10.0
    ):
        raise ValueError("temperature must be inside [0, 10]")
    if any(type(token) is not int or token < 0 for token in bridge):
        raise ValueError("bridge_tokens must contain non-negative integers")
    if spec.decode_bridge_policy == "none" and bridge:
        raise ValueError("bridge tokens supplied while decode bridge is disabled")
    if spec.decode_bridge_policy != "none" and not bridge:
        raise ValueError("execution spec requires decode bridge tokens")

    layers, cache, logits = _cached_live_path_initial_logits(
        model,
        prompt_tokens,
        spec=spec,
        branch_index=branch_index,
        decode_token_budget=len(bridge) + token_count,
        adapters_on=True,
    )
    for token in bridge:
        hidden = model.model.embed_tokens(mx.array([[token]]))
        mask = create_attention_mask(hidden, cache)
        with coda_adapter_scope():
            for index, layer in enumerate(layers):
                hidden = layer(hidden, mask, cache[index])
        logits = _logits(model, hidden)[0, -1]

    tokens: list[int] = []
    behavior_logprobs: list[float] = []
    for draw in range(token_count):
        detached_logits = mx.stop_gradient(logits.astype(mx.float32))
        if float(temperature) == 0.0:
            token = int(mx.argmax(detached_logits))
            logprob = mx.array(0.0, dtype=mx.float32)
        else:
            key = mx.random.key((seed + draw * 0x9E3779B1) & 0x7FFFFFFF)
            sampling_logits = detached_logits / float(temperature)
            token = int(mx.random.categorical(sampling_logits, key=key))
            logprob = sampling_logits[token] - mx.logsumexp(sampling_logits)
        mx.eval(logprob)
        tokens.append(token)
        behavior_logprobs.append(float(logprob))
        if draw + 1 == token_count:
            continue
        hidden = model.model.embed_tokens(mx.array([[token]]))
        mask = create_attention_mask(hidden, cache)
        with coda_adapter_scope():
            for index, layer in enumerate(layers):
                hidden = layer(hidden, mask, cache[index])
        logits = _logits(model, hidden)[0, -1]
    normalized = tuple(tokens)
    result = CachedLivePathRollin(
        tokens=normalized,
        behavior_logprobs=tuple(behavior_logprobs),
        branch_index=branch_index,
        seed=seed,
        temperature=float(temperature),
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=_canonical_tokens_sha256(
            prompt_tokens,
            role="prompt_tokens",
        ),
        tokens_sha256=_canonical_tokens_sha256(
            normalized,
            role="rollin_tokens",
        ),
    )
    del layers, cache, logits
    mx.clear_cache()
    return result


def cached_supervised_live_path_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
    token_loss_weights: Sequence[float] | None = None,
    branch_indices: Sequence[int] | None = None,
) -> CachedSupervisedLivePathResult:
    """Differentiate teacher-forced CE through the exact cached live policy.

    Each branch is differentiated and materialized independently before its
    scaled gradient is accumulated. This keeps resident graph residency bounded
    while preserving the exact branch-ensemble objective used by live RLC
    execution. Unlike the historical no-cache SFT objective, lexical scoring
    uses the same token-at-a-time quantized kernels as resident generation.
    """

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map

    answer, bridge, weights, indices, weight_total = _cached_supervised_inputs(
        answer_tokens,
        bridge_tokens=bridge_tokens,
        token_loss_weights=token_loss_weights,
        branch_indices=branch_indices,
        spec=spec,
    )

    weight_tensor = mx.array(weights, dtype=mx.float32)
    branch_scale = 1.0 / len(indices)
    accumulated: Any | None = None
    branch_values: list[float] = []
    for branch_index in indices:

        def objective(current_model: Any, _branch_index: int = branch_index) -> Any:
            logprobs = cached_live_path_token_logprobs(
                current_model,
                prompt_tokens,
                answer,
                spec=spec,
                branch_index=_branch_index,
                bridge_tokens=bridge,
                adapters_on=True,
            )
            return -mx.sum(logprobs * weight_tensor) / weight_total

        value, gradients = nn.value_and_grad(model, objective)(model)
        finite_flags = [
            mx.all(mx.isfinite(gradient)) for _path, gradient in tree_flatten(gradients)
        ]
        mx.eval(value, gradients, finite_flags)
        branch_value = float(value)
        if (
            not math.isfinite(branch_value)
            or not finite_flags
            or not all(bool(flag) for flag in finite_flags)
        ):
            raise FloatingPointError("cached supervised live-path gradient is non-finite")
        scaled = tree_map(lambda gradient: branch_scale * gradient, gradients)
        accumulated = (
            scaled
            if accumulated is None
            else tree_map(lambda total, gradient: total + gradient, accumulated, scaled)
        )
        mx.eval(accumulated)
        branch_values.append(branch_value)
        del value, gradients, scaled
        mx.clear_cache()
    if accumulated is None:
        raise RuntimeError("cached supervised live-path gradient is empty")
    return CachedSupervisedLivePathResult(
        value=sum(branch_values) / len(branch_values),
        gradients=accumulated,
        branch_values=tuple(branch_values),
        branch_indices=indices,
        answer_token_count=len(answer),
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=_canonical_tokens_sha256(
            prompt_tokens,
            role="prompt_tokens",
        ),
        answer_tokens_sha256=_canonical_tokens_sha256(
            answer,
            role="answer_tokens",
        ),
        bridge_tokens_sha256=_canonical_optional_tokens_sha256(
            bridge,
            role="bridge_tokens",
        ),
    )


def cached_supervised_live_path_loss(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
    token_loss_weights: Sequence[float] | None = None,
    branch_indices: Sequence[int] | None = None,
) -> CachedSupervisedLivePathEvaluation:
    """Evaluate teacher-forced CE through the exact cached live policy."""

    import mlx.core as mx

    answer, bridge, weights, indices, weight_total = _cached_supervised_inputs(
        answer_tokens,
        bridge_tokens=bridge_tokens,
        token_loss_weights=token_loss_weights,
        branch_indices=branch_indices,
        spec=spec,
    )
    weight_tensor = mx.array(weights, dtype=mx.float32)
    branch_values: list[float] = []
    for branch_index in indices:
        logprobs = cached_live_path_token_logprobs(
            model,
            prompt_tokens,
            answer,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge,
            adapters_on=True,
        )
        value = -mx.sum(logprobs * weight_tensor) / weight_total
        try:
            mx.eval(value)
            branch_value = float(value)
        finally:
            del value, logprobs
            mx.clear_cache()
        if not math.isfinite(branch_value) or branch_value < 0.0:
            raise FloatingPointError("cached supervised live-path loss is non-finite")
        branch_values.append(branch_value)
    return CachedSupervisedLivePathEvaluation(
        value=sum(branch_values) / len(branch_values),
        branch_values=tuple(branch_values),
        branch_indices=indices,
        answer_token_count=len(answer),
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=_canonical_tokens_sha256(
            prompt_tokens,
            role="prompt_tokens",
        ),
        answer_tokens_sha256=_canonical_tokens_sha256(
            answer,
            role="answer_tokens",
        ),
        bridge_tokens_sha256=_canonical_optional_tokens_sha256(
            bridge,
            role="bridge_tokens",
        ),
    )


def _cached_supervised_inputs(
    answer_tokens: Sequence[int],
    *,
    bridge_tokens: Sequence[int],
    token_loss_weights: Sequence[float] | None,
    branch_indices: Sequence[int] | None,
    spec: RLCExecutionSpec,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...], tuple[int, ...], float]:
    answer = tuple(answer_tokens)
    bridge = tuple(bridge_tokens)
    if not answer or any(type(token) is not int or token < 0 for token in answer):
        raise ValueError("answer_tokens must contain non-negative integers")
    if any(type(token) is not int or token < 0 for token in bridge):
        raise ValueError("bridge_tokens must contain non-negative integers")
    weights = (
        tuple(1.0 for _ in answer)
        if token_loss_weights is None
        else tuple(float(value) for value in token_loss_weights)
    )
    weight_total = float(sum(weights))
    if (
        len(weights) != len(answer)
        or any(not math.isfinite(value) or value < 0.0 for value in weights)
        or weight_total <= 0.0
    ):
        raise ValueError("token loss weights must be finite, non-negative, and token-aligned")
    indices = (
        tuple(range(len(spec.branch_roles))) if branch_indices is None else tuple(branch_indices)
    )
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            type(index) is not int or not 0 <= index < len(spec.branch_roles) for index in indices
        )
    ):
        raise ValueError("branch indices must be unique members of the live branch set")
    return answer, bridge, weights, indices, weight_total


def live_path_branch_answer_ce_trail(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    bridge_tokens: Sequence[int] = (),
) -> list[float]:
    """Answer CE after each recurrent step for one live-path branch.

    GRPO's final verifier can mark an entire sampled group wrong, leaving
    zero group-relative advantage. For recurrence-native training, that wastes
    the most important early signal: which internal state trajectories moved
    toward the known correct answer before the sampled decode missed. This
    function measures that signal on the same live recurrent graph used by the
    exact-adjoint objective. It is telemetry/credit assignment only; it does
    not replace the external verifier.
    """

    import mlx.core as mx
    import mlx.nn as nn

    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
    )
    if type(branch_index) is not int or not 0 <= branch_index < len(prepared.states):
        raise ValueError("branch_index is outside the live-path branch set")

    targets = mx.array(list(answer_tokens))[None, :]
    states = list(prepared.states)
    trail: list[float] = []
    for step in range(spec.recurrent_steps):
        states = _checkpointed_recurrent_transition(
            model,
            prepared.prompts_at_window,
            states,
            prepared.anchors,
            spec,
            step,
            prepared.prelude_end,
            prepared.coda_start,
        )
        logits = _persist_and_score(
            model,
            prepared.prompt_embeddings,
            prepared.seeds[branch_index],
            states[branch_index],
            prepared.tail_embeddings,
            branch_index=branch_index,
            bridge_count=prepared.bridge_count,
            answer_count=prepared.answer_count,
            prelude_end=prepared.prelude_end,
            coda_start=prepared.coda_start,
        )
        losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
        value = mx.mean(losses)
        mx.eval(value)
        trail.append(float(value))
        del logits, losses, value
        mx.clear_cache()
    return trail


def branch_mean_answer_loss(forward: LivePathForward, answer_tokens: Sequence[int]) -> Any:
    """Mean answer CE: every role must remain competent, not only an oracle arm."""

    import mlx.core as mx
    import mlx.nn as nn

    targets = mx.array(list(answer_tokens))[None, :]
    losses = [
        nn.losses.cross_entropy(logits, targets, reduction="mean")
        for logits in forward.branch_logits
    ]
    return sum(losses) / len(losses)


def _exact_adjoint_live_path_result(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
    diversity_weight: float = 0.0,
    diversity_target_cos: float = 0.98,
    token_loss_weights: Sequence[float] | None = None,
    branch_index: int | None = None,
    trajectory_config: ExactAdjointTrajectoryConfig | None = None,
    intervention_config: ExactAdjointInterventionConfig | None = None,
    policy_sha256: str | None = None,
    allow_signed_token_loss_weights: bool = False,
    terminal_objective_weight: float = 1.0,
) -> ExactAdjointLivePathResult:
    """Compute the exact live-path gradient with bounded graph residency.

    Only recurrent LoRA parameters are trainable. Prelude outputs are therefore
    parameter-independent boundary values. Recurrence states are materialized
    between transitions, then exact vector-Jacobian products replay those
    transitions in reverse. Terminal branch losses are differentiated one at a
    time and their parameter/state gradients are accumulated algebraically.
    """

    import math
    import re

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map

    if (
        isinstance(diversity_weight, bool)
        or not isinstance(diversity_weight, (int, float))
        or not math.isfinite(float(diversity_weight))
        or not 0.0 <= float(diversity_weight) <= 10.0
    ):
        raise ValueError("diversity_weight must be inside [0, 10]")
    if (
        isinstance(terminal_objective_weight, bool)
        or not isinstance(terminal_objective_weight, (int, float))
        or not math.isfinite(float(terminal_objective_weight))
        or not 0.0 <= float(terminal_objective_weight) <= 1.0
    ):
        raise ValueError("terminal_objective_weight must be inside [0, 1]")
    if token_loss_weights is None:
        normalized_token_weights = (1.0,) * len(answer_tokens)
    else:
        if any(isinstance(value, bool) for value in token_loss_weights):
            raise ValueError("token_loss_weights must align and be finite")
        normalized_token_weights = tuple(float(value) for value in token_loss_weights)
        if len(normalized_token_weights) != len(answer_tokens) or any(
            not math.isfinite(value) for value in normalized_token_weights
        ):
            raise ValueError("token_loss_weights must align and be finite")
        if not allow_signed_token_loss_weights and any(
            value < 0.0 for value in normalized_token_weights
        ):
            raise ValueError("proof receipt token_loss_weights must be non-negative")
    prompt_tokens_sha256 = _canonical_tokens_sha256(prompt_tokens, role="prompt_tokens")
    answer_tokens_sha256 = _canonical_tokens_sha256(answer_tokens, role="answer_tokens")
    bridge_tokens_sha256 = _canonical_optional_tokens_sha256(
        bridge_tokens,
        role="bridge_tokens",
    )
    if policy_sha256 is not None and not _valid_sha256(policy_sha256):
        raise ValueError("policy_sha256 must be a lowercase SHA-256 digest")
    parameters = model.trainable_parameters()
    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
    )
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.")
    layer_count = len(model.model.layers)
    for path, _value in tree_flatten(parameters):
        match = layer_pattern.match(path)
        if match is None or not (
            prepared.prelude_end <= int(match.group(1)) < layer_count
        ):
            raise RuntimeError("exact_adjoint_requires_recurrent_or_coda_trainables")
    if branch_index is not None and (
        type(branch_index) is not int or not 0 <= branch_index < len(prepared.states)
    ):
        raise ValueError("branch_index is outside the live-path branch set")
    if trajectory_config is not None:
        if not isinstance(trajectory_config, ExactAdjointTrajectoryConfig):
            raise TypeError("trajectory_config must be an ExactAdjointTrajectoryConfig")
        trajectory_config.validate_depth(spec.recurrent_steps)
    if intervention_config is not None:
        if not isinstance(intervention_config, ExactAdjointInterventionConfig):
            raise TypeError("intervention_config must be an ExactAdjointInterventionConfig")
        intervention_config.validate_depth(spec.recurrent_steps)

    def detached(values: Sequence[Any]) -> tuple[Any, ...]:
        result = tuple(mx.stop_gradient(value) for value in values)
        mx.eval(result)
        return result

    prompt_embeddings = mx.stop_gradient(prepared.prompt_embeddings)
    tail_embeddings = mx.stop_gradient(prepared.tail_embeddings)
    seeds = detached(prepared.seeds)
    mx.eval(prompt_embeddings, tail_embeddings)
    prompts = detached(prepared.prompts_at_window)
    anchors = detached(prepared.anchors)
    history: list[tuple[Any, ...]] = [detached(prepared.states)]
    current = history[0]
    for step in range(spec.recurrent_steps):
        outputs = _advance_recurrent_states(
            model,
            prompts,
            current,
            anchors,
            spec,
            step,
            prepared.prelude_end,
            prepared.coda_start,
        )
        current = detached(outputs)
        history.append(current)
        del outputs
        mx.clear_cache()

    accumulated: Any | None = None

    def add_parameter_gradient(gradient: Any, scale: float = 1.0) -> None:
        nonlocal accumulated
        scaled = tree_map(lambda value: scale * value, gradient)
        accumulated = (
            scaled
            if accumulated is None
            else tree_map(lambda left, right: left + right, accumulated, scaled)
        )
        mx.eval(accumulated)

    targets = mx.array(list(answer_tokens))[None, :]
    token_weights = mx.array(normalized_token_weights, dtype=mx.float32)[None, :]
    selected_indices = tuple(range(len(current))) if branch_index is None else (branch_index,)
    branch_scale = 1.0 / len(selected_indices)
    branch_values: list[float] = []
    direct_cotangents: list[list[Any]] = [
        [mx.zeros_like(state) for state in states] for states in history
    ]
    for selected_index in selected_indices:
        seed = seeds[selected_index]
        state = current[selected_index]

        def terminal_loss(
            parameter_tree: Any,
            final_state: Any,
            _seed: Any = seed,
            _branch_index: int = selected_index,
        ) -> Any:
            model.update(parameter_tree)
            logits = _persist_and_score(
                model,
                prompt_embeddings,
                _seed,
                final_state,
                tail_embeddings,
                branch_index=_branch_index,
                bridge_count=prepared.bridge_count,
                answer_count=prepared.answer_count,
                prelude_end=prepared.prelude_end,
                coda_start=prepared.coda_start,
            )
            token_losses = nn.losses.cross_entropy(
                logits.astype(mx.float32), targets, reduction="none"
            )
            return mx.mean(token_losses * token_weights)

        value, (parameter_gradient, state_gradient) = mx.value_and_grad(
            terminal_loss,
            argnums=(0, 1),
        )(parameters, state)
        mx.eval(value, parameter_gradient, state_gradient)
        branch_values.append(float(value))
        terminal_scale = branch_scale * float(terminal_objective_weight)
        add_parameter_gradient(parameter_gradient, terminal_scale)
        direct_cotangents[-1][selected_index] = mx.stop_gradient(
            direct_cotangents[-1][selected_index] + terminal_scale * state_gradient
        )
        del value, parameter_gradient, state_gradient
        mx.clear_cache()

    from core.learning.recurrence_native_objective_v3 import (
        branch_diversity_penalty,
    )

    terminal_forward = LivePathForward(
        branch_logits=(),
        branch_states=current,
        exchanges=0,
        prompt_tokens=prepared.prompt_count,
        answer_tokens=prepared.answer_count,
        bridge_tokens=prepared.bridge_count,
    )
    diversity_penalty, cosines = branch_diversity_penalty(
        terminal_forward,
        target_cos=diversity_target_cos,
    )
    mx.eval(diversity_penalty)
    diversity_value = float(diversity_penalty)
    if float(diversity_weight) > 0.0:

        def diversity_loss(final_states: tuple[Any, ...]) -> Any:
            forward = LivePathForward(
                branch_logits=(),
                branch_states=final_states,
                exchanges=0,
                prompt_tokens=prepared.prompt_count,
                answer_tokens=prepared.answer_count,
                bridge_tokens=prepared.bridge_count,
            )
            penalty, _cosines = branch_diversity_penalty(
                forward,
                target_cos=diversity_target_cos,
            )
            return penalty

        _value, diversity_gradients = mx.value_and_grad(diversity_loss)(current)
        mx.eval(diversity_gradients)
        direct_cotangents[-1] = [
            mx.stop_gradient(existing + float(diversity_weight) * diversity)
            for existing, diversity in zip(direct_cotangents[-1], diversity_gradients, strict=True)
        ]
        mx.eval(direct_cotangents[-1])
        del diversity_gradients
        mx.clear_cache()

    trajectory_values = {
        "improvement": 0.0,
        "displacement": 0.0,
        "oscillation": 0.0,
    }
    if intervention_config is not None:
        trajectory_values.update(
            {
                "causality": 0.0,
                "stopping": 0.0,
            }
        )
    step_loss_values: dict[int, tuple[float, ...]] = {}
    lesion_loss_values: dict[int, tuple[float, ...]] = {}
    stopping_teacher_receipts: list[Mapping[str, Any]] = []
    measured_displacements: list[float] = []
    measured_oscillation_cosines: list[float] = []
    probe_steps: set[int] = set()
    if trajectory_config is not None and float(trajectory_config.improvement_weight) > 0.0:
        probe_steps.update(trajectory_config.probe_steps)
    if intervention_config is not None:
        if float(intervention_config.causality_weight) > 0.0:
            probe_steps.add(spec.recurrent_steps)
        if float(intervention_config.stopping_weight) > 0.0:
            probe_steps.update(intervention_config.stopping_steps)
    probe_losses: dict[int, list[float]] = {}
    probe_gradients: dict[int, list[tuple[Any, Any]]] = {}
    for depth in sorted(probe_steps):
        depth_losses: list[float] = []
        depth_gradients: list[tuple[Any, Any]] = []
        for selected_index in selected_indices:
            seed = seeds[selected_index]

            def intermediate_loss(
                parameter_tree: Any,
                state: Any,
                _seed: Any = seed,
                _branch_index: int = selected_index,
            ) -> Any:
                model.update(parameter_tree)
                logits = _persist_and_score(
                    model,
                    prompt_embeddings,
                    _seed,
                    state,
                    tail_embeddings,
                    branch_index=_branch_index,
                    bridge_count=prepared.bridge_count,
                    answer_count=prepared.answer_count,
                    prelude_end=prepared.prelude_end,
                    coda_start=prepared.coda_start,
                )
                return nn.losses.cross_entropy(
                    logits.astype(mx.float32),
                    targets,
                    reduction="mean",
                )

            loss, (parameter_gradient, state_gradient) = mx.value_and_grad(
                intermediate_loss,
                argnums=(0, 1),
            )(parameters, history[depth][selected_index])
            mx.eval(loss, parameter_gradient, state_gradient)
            depth_losses.append(float(loss))
            depth_gradients.append(
                (
                    parameter_gradient,
                    mx.stop_gradient(state_gradient),
                )
            )
        probe_losses[depth] = depth_losses
        probe_gradients[depth] = depth_gradients
        step_loss_values[depth] = tuple(depth_losses)

    if trajectory_config is not None:
        if float(trajectory_config.improvement_weight) > 0.0:
            hinge_count = (len(trajectory_config.probe_steps) - 1) * len(selected_indices)
            hinge_scale = float(trajectory_config.improvement_weight) / hinge_count
            improvement_value = 0.0
            for previous_depth, current_depth in zip(
                trajectory_config.probe_steps,
                trajectory_config.probe_steps[1:],
                strict=False,
            ):
                for offset, selected_index in enumerate(selected_indices):
                    hinge = max(
                        probe_losses[current_depth][offset]
                        - probe_losses[previous_depth][offset]
                        + float(trajectory_config.improvement_margin),
                        0.0,
                    )
                    improvement_value += hinge_scale * hinge
                    if hinge <= 0.0:
                        continue
                    parameter_gradient, state_gradient = probe_gradients[current_depth][offset]
                    add_parameter_gradient(parameter_gradient, hinge_scale)
                    direct_cotangents[current_depth][selected_index] = mx.stop_gradient(
                        direct_cotangents[current_depth][selected_index]
                        + hinge_scale * state_gradient
                    )
            trajectory_values["improvement"] = improvement_value

        if float(trajectory_config.displacement_weight) > 0.0:
            term_count = spec.recurrent_steps * len(selected_indices)
            term_scale = float(trajectory_config.displacement_weight) / term_count
            displacement_value = 0.0
            for depth in range(1, spec.recurrent_steps + 1):
                for selected_index in selected_indices:

                    def displacement_loss(previous: Any, current_state: Any) -> Any:
                        numerator = mx.linalg.norm(mx.reshape(current_state - previous, (-1,)))
                        denominator = mx.maximum(
                            mx.linalg.norm(mx.reshape(previous, (-1,))),
                            1e-9,
                        )
                        return mx.maximum(
                            float(trajectory_config.displacement_floor) - numerator / denominator,
                            0.0,
                        )

                    value, (previous_gradient, current_gradient) = mx.value_and_grad(
                        displacement_loss,
                        argnums=(0, 1),
                    )(
                        history[depth - 1][selected_index],
                        history[depth][selected_index],
                    )
                    mx.eval(value, previous_gradient, current_gradient)
                    previous_state = history[depth - 1][selected_index]
                    current_state = history[depth][selected_index]
                    displacement = float(
                        mx.linalg.norm(mx.reshape(current_state - previous_state, (-1,)))
                        / mx.maximum(
                            mx.linalg.norm(mx.reshape(previous_state, (-1,))),
                            1e-9,
                        )
                    )
                    measured_displacements.append(displacement)
                    displacement_value += term_scale * float(value)
                    direct_cotangents[depth - 1][selected_index] = mx.stop_gradient(
                        direct_cotangents[depth - 1][selected_index]
                        + term_scale * previous_gradient
                    )
                    direct_cotangents[depth][selected_index] = mx.stop_gradient(
                        direct_cotangents[depth][selected_index] + term_scale * current_gradient
                    )
            trajectory_values["displacement"] = displacement_value

        if float(trajectory_config.oscillation_weight) > 0.0:
            pair_count = (spec.recurrent_steps - 1) * len(selected_indices)
            pair_scale = float(trajectory_config.oscillation_weight) / pair_count
            oscillation_value = 0.0
            for depth in range(2, spec.recurrent_steps + 1):
                for selected_index in selected_indices:

                    def oscillation_loss(
                        previous: Any,
                        middle: Any,
                        current_state: Any,
                    ) -> Any:
                        first = mx.reshape(middle - previous, (-1,))
                        second = mx.reshape(current_state - middle, (-1,))
                        denominator = mx.maximum(
                            mx.linalg.norm(first) * mx.linalg.norm(second),
                            1e-9,
                        )
                        cosine = mx.sum(first * second) / denominator
                        return mx.maximum(-cosine, 0.0)

                    value, gradients = mx.value_and_grad(
                        oscillation_loss,
                        argnums=(0, 1, 2),
                    )(
                        history[depth - 2][selected_index],
                        history[depth - 1][selected_index],
                        history[depth][selected_index],
                    )
                    mx.eval(value, gradients)
                    first = mx.reshape(
                        history[depth - 1][selected_index] - history[depth - 2][selected_index],
                        (-1,),
                    )
                    second = mx.reshape(
                        history[depth][selected_index] - history[depth - 1][selected_index],
                        (-1,),
                    )
                    cosine = float(
                        mx.sum(first * second)
                        / mx.maximum(
                            mx.linalg.norm(first) * mx.linalg.norm(second),
                            1e-9,
                        )
                    )
                    measured_oscillation_cosines.append(cosine)
                    oscillation_value += pair_scale * float(value)
                    for state_depth, gradient in zip(
                        (depth - 2, depth - 1, depth),
                        gradients,
                        strict=True,
                    ):
                        direct_cotangents[state_depth][selected_index] = mx.stop_gradient(
                            direct_cotangents[state_depth][selected_index] + pair_scale * gradient
                        )
            trajectory_values["oscillation"] = oscillation_value

    if intervention_config is not None:
        if float(intervention_config.causality_weight) > 0.0:
            for lesion_step in intervention_config.lesion_steps:
                lesion_states = detached(history[lesion_step - 1])
                for replay_step in range(lesion_step, spec.recurrent_steps):
                    outputs = _advance_recurrent_states(
                        model,
                        prompts,
                        lesion_states,
                        anchors,
                        spec,
                        replay_step,
                        prepared.prelude_end,
                        prepared.coda_start,
                    )
                    lesion_states = detached(outputs)
                    del outputs
                    mx.clear_cache()
                losses: list[float] = []
                for selected_index in selected_indices:
                    logits = _persist_and_score(
                        model,
                        prompt_embeddings,
                        seeds[selected_index],
                        lesion_states[selected_index],
                        tail_embeddings,
                        branch_index=selected_index,
                        bridge_count=prepared.bridge_count,
                        answer_count=prepared.answer_count,
                        prelude_end=prepared.prelude_end,
                        coda_start=prepared.coda_start,
                    )
                    loss = nn.losses.cross_entropy(
                        logits.astype(mx.float32),
                        targets,
                        reduction="mean",
                    )
                    mx.eval(loss)
                    losses.append(float(loss))
                    del logits, loss
                    mx.clear_cache()
                lesion_loss_values[lesion_step] = tuple(losses)

            hinge_count = len(intervention_config.lesion_steps) * len(selected_indices)
            hinge_scale = float(intervention_config.causality_weight) / hinge_count
            causality_value = 0.0
            intact_losses = probe_losses[spec.recurrent_steps]
            for lesion_step in intervention_config.lesion_steps:
                for offset, selected_index in enumerate(selected_indices):
                    hinge = max(
                        intact_losses[offset]
                        - lesion_loss_values[lesion_step][offset]
                        + float(intervention_config.causality_margin),
                        0.0,
                    )
                    causality_value += hinge_scale * hinge
                    if hinge <= 0.0:
                        continue
                    parameter_gradient, state_gradient = probe_gradients[spec.recurrent_steps][
                        offset
                    ]
                    add_parameter_gradient(parameter_gradient, hinge_scale)
                    direct_cotangents[spec.recurrent_steps][selected_index] = mx.stop_gradient(
                        direct_cotangents[spec.recurrent_steps][selected_index]
                        + hinge_scale * state_gradient
                    )
            trajectory_values["causality"] = causality_value

        if float(intervention_config.stopping_weight) > 0.0:
            from core.learning.adaptive_halting import verified_stopping_teacher

            stopping_scale = float(intervention_config.stopping_weight) / len(selected_indices)
            stopping_value = 0.0
            for offset, selected_index in enumerate(selected_indices):
                teacher = verified_stopping_teacher(
                    [probe_losses[step][offset] for step in intervention_config.stopping_steps],
                    intervention_config.stopping_steps,
                    ponder_cost=intervention_config.stopping_ponder_cost,
                    temperature=intervention_config.stopping_temperature,
                )
                stopping_teacher_receipts.append(teacher.receipt())
                stopping_value += stopping_scale * teacher.expected_risk
                for step, probability in zip(
                    intervention_config.stopping_steps,
                    teacher.probabilities,
                    strict=True,
                ):
                    parameter_gradient, state_gradient = probe_gradients[step][offset]
                    scale = stopping_scale * probability
                    add_parameter_gradient(parameter_gradient, scale)
                    direct_cotangents[step][selected_index] = mx.stop_gradient(
                        direct_cotangents[step][selected_index] + scale * state_gradient
                    )
            trajectory_values["stopping"] = stopping_value

    cotangents = tuple(direct_cotangents[-1])
    for step in range(spec.recurrent_steps - 1, -1, -1):
        input_states = history[step]

        def transition_pullback(
            parameter_tree: Any,
            prior_states: tuple[Any, ...],
            _step: int = step,
            _cotangents: tuple[Any, ...] = cotangents,
        ) -> Any:
            model.update(parameter_tree)
            outputs = _advance_recurrent_states(
                model,
                prompts,
                prior_states,
                anchors,
                spec,
                _step,
                prepared.prelude_end,
                prepared.coda_start,
            )
            return sum(
                mx.sum(output * cotangent)
                for output, cotangent in zip(outputs, _cotangents, strict=True)
            )

        _pullback, (parameter_gradient, input_cotangents) = mx.value_and_grad(
            transition_pullback,
            argnums=(0, 1),
        )(parameters, input_states)
        mx.eval(parameter_gradient, input_cotangents)
        add_parameter_gradient(parameter_gradient)
        cotangents = tuple(
            mx.stop_gradient(incoming + direct)
            for incoming, direct in zip(
                input_cotangents,
                direct_cotangents[step],
                strict=True,
            )
        )
        mx.eval(cotangents)
        del parameter_gradient, input_cotangents
        mx.clear_cache()

    if accumulated is None:
        raise RuntimeError("exact adjoint parameter gradient is empty")
    base_value = sum(branch_values) / len(branch_values)
    total_value = (
        float(terminal_objective_weight) * base_value
        + float(diversity_weight) * diversity_value
        + sum(trajectory_values.values())
    )
    return ExactAdjointLivePathResult(
        value=total_value,
        gradients=accumulated,
        terminal_value=base_value,
        diversity_value=float(diversity_weight) * diversity_value,
        trajectory_values=trajectory_values,
        step_losses=step_loss_values,
        lesion_losses=lesion_loss_values,
        stopping_teacher_receipts=tuple(stopping_teacher_receipts),
        displacements=tuple(measured_displacements),
        oscillation_cosines=tuple(measured_oscillation_cosines),
        diversity_cosines=tuple(float(value) for value in cosines),
        branch_indices=selected_indices,
        trajectory_config=trajectory_config,
        intervention_config=intervention_config,
        execution_spec_sha256=spec.sha256,
        recurrent_depth=spec.recurrent_steps,
        execution_branch_count=len(current),
        diversity_weight=float(diversity_weight),
        diversity_target_cos=float(diversity_target_cos),
        policy_sha256=policy_sha256,
        prompt_tokens_sha256=prompt_tokens_sha256,
        prompt_token_count=len(prompt_tokens),
        answer_tokens_sha256=answer_tokens_sha256,
        answer_token_count=len(answer_tokens),
        bridge_tokens_sha256=bridge_tokens_sha256,
        bridge_token_count=len(bridge_tokens),
        token_loss_weights=normalized_token_weights,
        terminal_objective_weight=float(terminal_objective_weight),
    )


def exact_adjoint_live_path_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
    diversity_weight: float = 0.0,
    diversity_target_cos: float = 0.98,
    token_loss_weights: Sequence[float] | None = None,
    branch_index: int | None = None,
) -> tuple[float, Any, float, list[float]]:
    """Compatibility surface for the terminal exact-adjoint objective."""

    result = _exact_adjoint_live_path_result(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
        diversity_weight=diversity_weight,
        diversity_target_cos=diversity_target_cos,
        token_loss_weights=token_loss_weights,
        branch_index=branch_index,
        allow_signed_token_loss_weights=True,
    )
    return (
        result.value,
        result.gradients,
        result.terminal_value,
        list(result.diversity_cosines),
    )


def exact_adjoint_trajectory_live_path_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    trajectory_config: ExactAdjointTrajectoryConfig,
    policy_sha256: str,
    bridge_tokens: Sequence[int] = (),
    branch_index: int | None = None,
    diversity_weight: float = 0.0,
    diversity_target_cos: float = 0.98,
    token_loss_weights: Sequence[float] | None = None,
) -> ExactAdjointLivePathResult:
    """Compute terminal and trajectory gradients with bounded graph residency."""

    return exact_adjoint_composite_live_path_value_and_grad(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        trajectory_config=trajectory_config,
        policy_sha256=policy_sha256,
        bridge_tokens=bridge_tokens,
        branch_index=branch_index,
        diversity_weight=diversity_weight,
        diversity_target_cos=diversity_target_cos,
        token_loss_weights=token_loss_weights,
    )


def exact_adjoint_trajectory_auxiliary_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    trajectory_config: ExactAdjointTrajectoryConfig,
    policy_sha256: str,
    bridge_tokens: Sequence[int] = (),
    branch_index: int | None = None,
    token_loss_weights: Sequence[float] | None = None,
) -> ExactAdjointLivePathResult:
    """Differentiate trajectory terms without counting terminal CE twice.

    Generated-prefix objectives already own the terminal policy loss. This
    surface reuses the exact-adjoint depth probes while assigning the measured
    terminal loss zero objective weight; the v4 receipt makes that distinction
    machine-verifiable rather than implicit in the caller.
    """

    return _exact_adjoint_live_path_result(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
        token_loss_weights=token_loss_weights,
        branch_index=branch_index,
        trajectory_config=trajectory_config,
        policy_sha256=policy_sha256,
        terminal_objective_weight=0.0,
    )


def exact_adjoint_trajectory_auxiliary_loss(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    trajectory_config: ExactAdjointTrajectoryConfig,
    policy_sha256: str,
    bridge_tokens: Sequence[int] = (),
    branch_index: int | None = None,
) -> ExactAdjointLivePathResult:
    """Measure the detached depth-improvement auxiliary without an adjoint."""

    import mlx.core as mx
    import mlx.nn as nn

    if (
        float(trajectory_config.displacement_weight) > 0.0
        or float(trajectory_config.oscillation_weight) > 0.0
    ):
        raise ValueError("loss-only trajectory evaluation supports improvement terms only")
    trajectory_config.validate_depth(spec.recurrent_steps)
    if not _valid_sha256(policy_sha256):
        raise ValueError("policy_sha256 must be a lowercase SHA-256 digest")
    branch_indices = (
        tuple(range(len(spec.branch_roles)))
        if branch_index is None
        else (branch_index,)
    )
    if any(
        type(index) is not int or not 0 <= index < len(spec.branch_roles)
        for index in branch_indices
    ):
        raise ValueError("branch_index is outside the live-path branch set")
    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
    )
    targets = mx.array(list(answer_tokens))[None, :]
    states = tuple(mx.stop_gradient(state) for state in prepared.states)
    mx.eval(states)
    trails_by_branch = {index: [] for index in branch_indices}
    for step in range(spec.recurrent_steps):
        outputs = _advance_recurrent_states(
            model,
            prepared.prompts_at_window,
            states,
            prepared.anchors,
            spec,
            step,
            prepared.prelude_end,
            prepared.coda_start,
        )
        states = tuple(mx.stop_gradient(state) for state in outputs)
        mx.eval(states)
        for index in branch_indices:
            logits = _persist_and_score(
                model,
                prepared.prompt_embeddings,
                prepared.seeds[index],
                states[index],
                prepared.tail_embeddings,
                branch_index=index,
                bridge_count=prepared.bridge_count,
                answer_count=prepared.answer_count,
                prelude_end=prepared.prelude_end,
                coda_start=prepared.coda_start,
            )
            loss = nn.losses.cross_entropy(
                logits.astype(mx.float32),
                targets,
                reduction="mean",
            )
            mx.eval(loss)
            trails_by_branch[index].append(float(loss))
            del logits, loss
        del outputs
        mx.clear_cache()
    trails = tuple(tuple(trails_by_branch[index]) for index in branch_indices)
    step_losses = {
        step: tuple(trail[step - 1] for trail in trails)
        for step in trajectory_config.probe_steps
    }
    pair_count = (len(trajectory_config.probe_steps) - 1) * len(branch_indices)
    improvement = 0.0
    if float(trajectory_config.improvement_weight) > 0.0:
        for previous, current in zip(
            trajectory_config.probe_steps,
            trajectory_config.probe_steps[1:],
            strict=False,
        ):
            improvement += sum(
                max(
                    step_losses[current][offset]
                    - step_losses[previous][offset]
                    + float(trajectory_config.improvement_margin),
                    0.0,
                )
                for offset in range(len(branch_indices))
            )
        improvement *= float(trajectory_config.improvement_weight) / pair_count
    terminal_value = sum(trail[-1] for trail in trails) / len(trails)
    from core.learning.recurrence_native_objective_v3 import (
        branch_diversity_penalty,
    )

    terminal_forward = LivePathForward(
        branch_logits=(),
        branch_states=states,
        exchanges=0,
        prompt_tokens=prepared.prompt_count,
        answer_tokens=prepared.answer_count,
        bridge_tokens=prepared.bridge_count,
    )
    _unused_diversity_penalty, diversity_cosines = branch_diversity_penalty(
        terminal_forward,
        target_cos=0.98,
    )
    return ExactAdjointLivePathResult(
        value=improvement,
        gradients=None,
        terminal_value=terminal_value,
        diversity_value=0.0,
        trajectory_values={
            "improvement": improvement,
            "displacement": 0.0,
            "oscillation": 0.0,
        },
        step_losses=step_losses,
        displacements=(),
        oscillation_cosines=(),
        diversity_cosines=tuple(diversity_cosines),
        branch_indices=branch_indices,
        trajectory_config=trajectory_config,
        execution_spec_sha256=spec.sha256,
        recurrent_depth=spec.recurrent_steps,
        execution_branch_count=len(spec.branch_roles),
        diversity_weight=0.0,
        diversity_target_cos=0.98,
        policy_sha256=policy_sha256,
        prompt_tokens_sha256=_canonical_tokens_sha256(
            prompt_tokens,
            role="prompt_tokens",
        ),
        prompt_token_count=len(prompt_tokens),
        answer_tokens_sha256=_canonical_tokens_sha256(
            answer_tokens,
            role="answer_tokens",
        ),
        answer_token_count=len(answer_tokens),
        bridge_tokens_sha256=_canonical_optional_tokens_sha256(
            bridge_tokens,
            role="bridge_tokens",
        ),
        bridge_token_count=len(bridge_tokens),
        token_loss_weights=(1.0,) * len(answer_tokens),
        terminal_objective_weight=0.0,
    )


def exact_adjoint_composite_live_path_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    trajectory_config: ExactAdjointTrajectoryConfig | None = None,
    intervention_config: ExactAdjointInterventionConfig | None = None,
    policy_sha256: str,
    bridge_tokens: Sequence[int] = (),
    branch_index: int | None = None,
    diversity_weight: float = 0.0,
    diversity_target_cos: float = 0.98,
    token_loss_weights: Sequence[float] | None = None,
) -> ExactAdjointLivePathResult:
    """Return rich exact-adjoint telemetry for any admitted objective mix.

    ``trajectory_config=None`` is intentionally supported here so a caller can
    train branch diversity without inventing a fake trajectory term.  The
    narrower ``exact_adjoint_trajectory_live_path_value_and_grad`` surface
    continues to require a real trajectory configuration.
    """

    return _exact_adjoint_live_path_result(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
        diversity_weight=diversity_weight,
        diversity_target_cos=diversity_target_cos,
        token_loss_weights=token_loss_weights,
        branch_index=branch_index,
        trajectory_config=trajectory_config,
        intervention_config=intervention_config,
        policy_sha256=policy_sha256,
    )


def live_path_loss(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
) -> Any:
    forward = live_path_forward(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
    )
    return branch_mean_answer_loss(forward, answer_tokens)


def detached_monotonicity_penalty(losses: Sequence[Any]) -> Any:
    """Penalize deeper regression without rewarding damage to shallow depth."""

    import mlx.core as mx

    if len(losses) < 2:
        raise ValueError("monotonicity penalty needs at least two depths")
    penalty = mx.zeros(())
    for shallow, deep in zip(losses, losses[1:], strict=False):
        penalty = penalty + mx.maximum(deep - mx.stop_gradient(shallow), 0.0)
    return penalty


def depth_curriculum_loss_v2(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    depths: tuple[int, ...] = (1, 2, 4),
    monotonicity_weight: float = 0.5,
    bridge_tokens: Sequence[int] = (),
) -> Any:
    """Answer CE over a depth ladder plus a shallow-detached monotonic hinge."""

    if (
        len(depths) < 2
        or any(type(depth) is not int or depth < 1 for depth in depths)
        or tuple(sorted(set(depths))) != depths
    ):
        raise ValueError("depths must be a strictly increasing tuple")
    if (
        isinstance(monotonicity_weight, bool)
        or not isinstance(monotonicity_weight, (int, float))
        or not 0.0 <= float(monotonicity_weight) <= 10.0
    ):
        raise ValueError("monotonicity_weight must be inside [0, 10]")
    losses = [
        live_path_loss(
            model,
            prompt_tokens,
            answer_tokens,
            spec=spec.with_depth(depth),
            bridge_tokens=bridge_tokens,
        )
        for depth in depths
    ]
    return sum(losses) / len(losses) + float(monotonicity_weight) * detached_monotonicity_penalty(
        losses
    )


__all__ = [
    "CachedLivePathRollin",
    "CachedSupervisedLivePathEvaluation",
    "CachedSupervisedLivePathResult",
    "EXACT_ADJOINT_AUXILIARY_RECEIPT_SCHEMA",
    "EXACT_ADJOINT_INTERVENTION_RECEIPT_SCHEMA",
    "EXACT_ADJOINT_INTERVENTION_SCHEMA",
    "EXACT_ADJOINT_TRAJECTORY_SCHEMA",
    "EXACT_ADJOINT_TRAJECTORY_RECEIPT_SCHEMA",
    "ExactAdjointInterventionConfig",
    "ExactAdjointLivePathResult",
    "ExactAdjointTrajectoryConfig",
    "INTERVENTION_MEASUREMENT_TRUST_BOUNDARY",
    "LivePathForward",
    "PreparedFinalRecurrentTransition",
    "PreparedRecurrentStateTrail",
    "PreparedRecurrentTransitionInput",
    "RECURRENCE_NATIVE_SCHEMA_V2",
    "RECURRENT_TRANSITION_INPUT_SCHEMA",
    "RECURRENT_TRANSITION_STATE_SCHEMA",
    "RECURRENT_STATE_TRAIL_SCHEMA",
    "branch_mean_answer_loss",
    "cached_live_path_token_logprobs",
    "cached_supervised_live_path_loss",
    "cached_supervised_live_path_value_and_grad",
    "depth_curriculum_loss_v2",
    "detached_monotonicity_penalty",
    "exact_adjoint_composite_live_path_value_and_grad",
    "exact_adjoint_live_path_value_and_grad",
    "exact_adjoint_trajectory_auxiliary_loss",
    "exact_adjoint_trajectory_auxiliary_value_and_grad",
    "exact_adjoint_trajectory_live_path_value_and_grad",
    "execute_prepared_recurrent_transition",
    "generate_cached_live_path_rollin",
    "live_path_branch_answer_ce_trail",
    "live_path_forward",
    "live_path_loss",
    "prepare_final_recurrent_transition",
    "prepare_recurrent_state_trail",
    "prepare_recurrent_transition_input",
    "validate_exact_adjoint_live_path_receipt",
    "validate_final_recurrent_transition_receipt",
    "validate_recurrent_state_trail_receipt",
    "validate_recurrent_transition_input_receipt",
]

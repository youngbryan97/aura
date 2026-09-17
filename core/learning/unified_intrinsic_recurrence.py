"""One intrinsic recurrent forward with depth, memory, correction, and halting.

The repository previously held these mechanisms in separate execution
architectures. This module makes them act on the same resident-transformer
trajectory. It is additive and identity-initialized: one iteration remains the
base forward until learned controller parameters are admitted.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import mlx.core as mx
import mlx.nn as nn

from core.learning.intrinsic_recurrence import (
    RecurrentDepthPlan,
    _rms,
    _run,
    recurrent_iteration,
)
from core.learning.protected_memory import (
    MemoryLayout,
    apply_protected_transition,
    memory_retention,
    semantic_convergence,
)
from core.learning.recurrent_action_schema import (
    ACTION_CARDINALITY,
    ACTION_NULL,
    ACTION_SLOT_NAMES,
    MAX_RECURRENT_OPCODE,
    OP_ADD_MOD,
    OP_BOOL_AND,
    OP_BOOL_NOT,
    OP_BOOL_OR,
    OP_BOOL_XOR,
    OP_CAUSAL_CHAIN,
    OP_COPY_VALUE,
    OP_FRONTIER_AUDIT,
    OP_FRONTIER_CALIBRATE,
    OP_FRONTIER_ENUMERATE,
    OP_FRONTIER_INFER,
    OP_FRONTIER_SCHEDULE,
    OP_FRONTIER_SIMULATE,
    OP_FRONTIER_TRAVERSE,
    OP_MUL_MOD,
    OP_PAIR_ADD,
    OP_PAIR_COPY,
    OP_PAIR_DIV,
    OP_PAIR_EUCLID_STEP,
    OP_PAIR_MUL_IMMEDIATE,
    OP_PAIR_PRODUCT,
    OP_PAIR_SET,
    OP_PAIR_SIGNED_SUB_IMMEDIATE,
    OP_PAIR_SUB_IMMEDIATE,
    OP_RANKED_COMMIT,
    OP_RATIO_BAND,
    OP_RATIO_CHOICE,
    OP_REGISTER_AFFINE,
    OP_SET_SCALAR,
    OP_SIGNED_PAIR_ADD_IMMEDIATE,
    OP_SIGNED_RANKED_GREATER,
    OP_SUB_MOD,
)
from core.learning.recurrent_answer_emission import RecurrentAnswerEmissionContract
from core.learning.recurrent_literal_grounding import (
    LITERAL_MAX_VALUE,
    LiteralObservationContract,
)
from core.learning.recurrent_opcode_grounding import (
    FrontierFamilyObservationContract,
    OpcodeObservationContract,
)
from core.learning.recurrent_state_schema import (
    SEMANTIC_STATE_SLOT_NAMES,
    STATE_CARDINALITY,
    STATE_INVALID,
    STATE_SLOT_NAMES,
    state_slot_names,
)

UNIFIED_INTRINSIC_RECURRENCE_SCHEMA: Final = "aura.unified_intrinsic_recurrence.v1"
PROCESS_TAPE_SCHEMA: Final = "aura.unified_intrinsic.process_tape.v4"
PROCESS_READER_PARAMETER_NAMES: Final = (
    "process_reader_1_query",
    "process_reader_1_key",
    "process_reader_1_value",
    "process_reader_1_output",
    "process_reader_1_gate_logit",
    "process_reader_2_query",
    "process_reader_2_key",
    "process_reader_2_value",
    "process_reader_2_output",
    "process_reader_2_gate_logit",
)
INITIAL_STATE_PARAMETER_NAMES: Final = (
    "initial_state_query",
    "initial_state_key",
    "initial_state_value",
    "initial_state_output",
    "initial_state_bias",
    "initial_state_literal_copy_logit",
)
ACTION_WORKSPACE_PARAMETER_NAMES: Final = (
    "action_workspace_seed",
    "action_workspace_depth",
    "action_workspace_cross_query",
    "action_workspace_cross_key",
    "action_workspace_cross_value",
    "action_workspace_cross_output",
    "action_workspace_self_query",
    "action_workspace_self_key",
    "action_workspace_self_value",
    "action_workspace_self_output",
    "action_workspace_ff_up",
    "action_workspace_ff_down",
    "action_workspace_output",
)
CAUSAL_ACTION_PARAMETER_NAMES: Final = (
    "action_causal_value_embeddings",
    "action_causal_input_projection",
    "action_causal_state_projection",
    "action_causal_prior_projection",
    "action_causal_output",
)
FAMILY_ACTION_PARAMETER_NAMES: Final = (
    "action_family_output",
    "action_family_bias",
    "action_family_kernel_mean",
    "action_family_kernel_inv_scale",
    "action_family_kernel_prototypes",
    "action_family_kernel_coefficients",
    "action_family_kernel_mask",
    "action_family_kernel_gamma",
)
ACTION_LITERAL_BINDING_PARAMETER_NAMES: Final = (
    "action_literal_binding_query",
    "action_literal_binding_key",
    "action_literal_binding_output",
    "action_literal_binding_family_output",
)
TRANSITION_MEMORY_PARAMETER_NAMES: Final = (
    "transition_memory_input",
    "transition_memory_reset_input",
    "transition_memory_reset_recurrent",
    "transition_memory_reset_bias",
    "transition_memory_update_input",
    "transition_memory_update_recurrent",
    "transition_memory_update_bias",
    "transition_memory_candidate_input",
    "transition_memory_candidate_recurrent",
    "transition_memory_cross_input",
    "transition_memory_cross_recurrent",
    "transition_memory_candidate_bias",
    "transition_memory_depth",
    "transition_memory_output",
)
TRANSITION_TAPE_READER_PARAMETER_NAMES: Final = (
    "transition_tape_key",
    "transition_tape_value",
    "transition_tape_position_key",
    "transition_tape_position_value",
    "transition_tape_state_query",
    "transition_tape_action_query",
    "transition_tape_output",
)
TRANSITION_PROCESSOR_PARAMETER_NAMES: Final = (
    "transition_processor_state_projection",
    "transition_processor_state_cross_projection",
    "transition_processor_action_left",
    "transition_processor_action_right",
    "transition_processor_history_projection",
    "transition_processor_interaction_up",
    "transition_processor_interaction_bias",
    "transition_processor_interaction_down",
    "transition_processor_output",
)
TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES: Final = (
    "transition_processor_opcode_interaction_up",
    "transition_processor_opcode_interaction_down",
    "transition_processor_opcode_hidden",
    "transition_processor_opcode_output",
)
TRANSITION_REPLAY_PARAMETER_NAMES: Final = (
    "transition_replay_key",
    "transition_replay_value",
    "transition_replay_position_key",
    "transition_replay_position_value",
    "transition_replay_query",
    "transition_replay_projection",
    "transition_replay_output",
    "transition_replay_opcode_output",
    "transition_replay_gate_weight",
    "transition_replay_gate_bias",
)
TRANSITION_EXECUTION_DEPENDENCY_PARAMETER_NAMES: Final = (
    # The public action tape is embedded before it enters transition memory.
    # A frozen transition machine is not reproducible without this codebook.
    "action_value_embeddings",
)
TRANSITION_PROCESSOR_MODES: Final = (
    "residual",
    "authoritative",
    "copy_write",
    "masked_copy_write",
)
TRANSITION_COPY_PRIOR_LOGIT_BIAS: Final = 2.0
TRANSITION_OPCODE_EXPERT_ROUTING_MODES: Final = (
    "opcode",
    "uniform",
    "lesion",
)
TRANSITION_REPLAY_MODES: Final = (
    "disabled",
    "active",
    "forced",
    "lesion",
)
ACTION_LITERAL_BINDING_TRANSFORMS: Final = (
    "identity",
    "unsigned_radix_low",
    "unsigned_radix_high",
    "positive_signed_radix_low",
    "positive_signed_radix_high",
    "offset_plus_three",
)
FRONTIER_ACTION_EXPERT_COUNT: Final = OP_FRONTIER_AUDIT - OP_FRONTIER_TRAVERSE + 1
PROCESS_RADIX: Final = 31
MAX_PROCESS_INTEGER: Final = PROCESS_RADIX**2 - 1


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


#: Opcodes the categorical microcode ACTUALLY executes. Recognition used to be
#: a range check against MAX_RECURRENT_OPCODE, which meant every opcode added
#: to the schema was declared recognized here before anyone wrote a branch for
#: it — a check reporting yes when it meant "the number is in range". Adding an
#: opcode to core/learning/recurrent_action_schema.py now leaves it
#: unrecognized until it is implemented below AND listed here, so the failure
#: names the gap instead of surfacing as a wrong next state.
MICROCODE_IMPLEMENTED_OPCODES: Final = frozenset(
    {
        OP_COPY_VALUE,
        OP_ADD_MOD,
        OP_MUL_MOD,
        OP_SUB_MOD,
        OP_BOOL_NOT,
        OP_BOOL_AND,
        OP_BOOL_OR,
        OP_BOOL_XOR,
        OP_REGISTER_AFFINE,
        OP_FRONTIER_TRAVERSE,
        OP_FRONTIER_ENUMERATE,
        OP_FRONTIER_SIMULATE,
        OP_FRONTIER_INFER,
        OP_FRONTIER_SCHEDULE,
        OP_FRONTIER_CALIBRATE,
        OP_FRONTIER_AUDIT,
        OP_PAIR_SET,
        OP_PAIR_ADD,
        OP_PAIR_MUL_IMMEDIATE,
        OP_PAIR_SUB_IMMEDIATE,
        OP_PAIR_SIGNED_SUB_IMMEDIATE,
        OP_PAIR_DIV,
        OP_RATIO_CHOICE,
        OP_RATIO_BAND,
        OP_SIGNED_PAIR_ADD_IMMEDIATE,
        OP_SIGNED_RANKED_GREATER,
        OP_RANKED_COMMIT,
        OP_SET_SCALAR,
        OP_PAIR_COPY,
        OP_PAIR_EUCLID_STEP,
        OP_PAIR_PRODUCT,
        OP_CAUSAL_CHAIN,
    }
)
_MICROCODE_OPCODE_MASK: Any = None


def _microcode_implemented_opcode_mask() -> Any:
    """A 0/1 lookup over the opcode vocabulary, built once."""
    global _MICROCODE_OPCODE_MASK
    if _MICROCODE_OPCODE_MASK is None:
        _MICROCODE_OPCODE_MASK = mx.array(
            [
                1 if code in MICROCODE_IMPLEMENTED_OPCODES else 0
                for code in range(ACTION_NULL + 1)
            ],
            dtype=mx.int32,
        )
    return _MICROCODE_OPCODE_MASK


@dataclass(frozen=True, slots=True)
class UnifiedRecurrenceConfig:
    hidden_size: int
    correction_rank: int = 8
    depth_basis_size: int = 4
    memory_write_threshold: float = 0.5
    halt_threshold: float = 0.9
    minimum_iterations: int = 2
    state_slots: int = len(STATE_SLOT_NAMES)
    state_cardinality: int = STATE_CARDINALITY
    action_slots: int = len(ACTION_SLOT_NAMES)
    action_cardinality: int = ACTION_CARDINALITY
    literal_digit_token_ids: tuple[int, ...] = ()
    numeric_observation_max_value: int = MAX_PROCESS_INTEGER
    opcode_token_patterns: tuple[tuple[int, tuple[int, ...]], ...] = ()
    opcode_context_patterns: tuple[tuple[str, tuple[int, ...]], ...] = ()
    frontier_family_token_patterns: tuple[tuple[int, tuple[int, ...]], ...] = ()
    initialization_seed: int = 20260810198
    schema: str = UNIFIED_INTRINSIC_RECURRENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != UNIFIED_INTRINSIC_RECURRENCE_SCHEMA:
            raise ValueError("unified recurrence schema differs")
        for name in (
            "hidden_size",
            "correction_rank",
            "depth_basis_size",
            "state_slots",
            "state_cardinality",
            "action_slots",
            "action_cardinality",
            "numeric_observation_max_value",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.correction_rank > self.hidden_size:
            raise ValueError("correction rank exceeds hidden size")
        if self.state_slots not in {
            len(STATE_SLOT_NAMES),
            len(SEMANTIC_STATE_SLOT_NAMES),
        }:
            raise ValueError("state slot count differs from the canonical schema")
        if self.state_cardinality != STATE_CARDINALITY:
            raise ValueError("state cardinality differs from the canonical schema")
        if self.action_slots != len(ACTION_SLOT_NAMES):
            raise ValueError("action slot count differs from the canonical schema")
        if self.action_cardinality != ACTION_CARDINALITY:
            raise ValueError("action cardinality differs from the canonical schema")
        if self.numeric_observation_max_value < LITERAL_MAX_VALUE:
            raise ValueError("numeric observation range is smaller than literal grounding")
        if self.literal_digit_token_ids and (
            len(self.literal_digit_token_ids) != 10
            or len(set(self.literal_digit_token_ids)) != 10
            or any(
                type(value) is not int or value < 0
                for value in self.literal_digit_token_ids
            )
        ):
            raise ValueError("literal digit token identity is invalid")
        if bool(self.opcode_token_patterns) != bool(self.opcode_context_patterns):
            raise ValueError("opcode pattern and context contracts differ")
        if self.opcode_token_patterns:
            OpcodeObservationContract(
                self.opcode_token_patterns,
                self.opcode_context_patterns,
            )
        if self.frontier_family_token_patterns:
            FrontierFamilyObservationContract(self.frontier_family_token_patterns)
        if type(self.minimum_iterations) is not int or self.minimum_iterations < 1:
            raise ValueError("minimum_iterations must be positive")
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            raise ValueError("initialization_seed must be non-negative")
        for name in ("memory_write_threshold", "halt_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 < float(value) < 1.0
            ):
                raise ValueError(f"{name} must be inside (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UnifiedRecurrenceTelemetry:
    configured_iterations: int
    executed_iterations: int
    halt_probabilities: tuple[float, ...]
    memory_write_means: tuple[float, ...]
    transport_gates: tuple[float, ...]
    halted: bool
    halt_reason: str
    memory_retention: dict[str, float] | None
    semantic_residuals: tuple[float, ...]
    process_tape_entries: int
    process_tape_active_entries: int
    controller_sha256: str

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": UNIFIED_INTRINSIC_RECURRENCE_SCHEMA,
            "configured_iterations": self.configured_iterations,
            "executed_iterations": self.executed_iterations,
            "halt_probabilities": list(self.halt_probabilities),
            "memory_write_means": list(self.memory_write_means),
            "transport_gates": list(self.transport_gates),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "memory_retention": self.memory_retention,
            "semantic_residuals": list(self.semantic_residuals),
            "process_tape_entries": self.process_tape_entries,
            "process_tape_active_entries": self.process_tape_active_entries,
            "controller_sha256": self.controller_sha256,
            "teacher_available": False,
            "solver_available": False,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


class UnifiedRecurrentController(nn.Module):
    """Small trainable control tissue acting on the real recurrent stream.

    A bounded rational depth basis ``u=step/(step+1)`` is used instead of a
    per-depth lookup bank. The resulting operator is defined at every unseen
    depth and converges rather than growing without bound.
    """

    def __init__(self, config: UnifiedRecurrenceConfig) -> None:
        super().__init__()
        self.config = config
        (
            key_a,
            key_b,
            key_depth,
            key_memory,
            key_halt,
            key_state,
            key_state_slots,
            key_state_values,
            key_transition_query,
            key_transition_key,
            key_transition_value,
            key_transition_self,
            key_transition_output,
            key_transition_depth,
            key_action_slots,
            key_action_values,
            key_action_query,
            key_action_key,
            key_action_value,
            key_action_output,
            key_action_depth,
            key_state_action,
            key_literal_values,
            key_answer_query,
            key_answer_key,
            key_answer_value,
            key_answer_output,
        ) = mx.random.split(
            mx.random.key(config.initialization_seed),
            num=27,
        )
        scale = 1.0 / math.sqrt(config.hidden_size)
        self._init_correction_and_halt(
            config,
            scale=scale,
            key_a=key_a,
            key_depth=key_depth,
            key_memory=key_memory,
            key_halt=key_halt,
        )
        self._init_state_registers_and_actions(
            config,
            scale=scale,
            key_state=key_state,
            key_state_slots=key_state_slots,
            key_state_values=key_state_values,
            key_transition_query=key_transition_query,
            key_transition_key=key_transition_key,
            key_transition_value=key_transition_value,
            key_transition_self=key_transition_self,
            key_transition_output=key_transition_output,
            key_transition_depth=key_transition_depth,
            key_action_slots=key_action_slots,
            key_action_values=key_action_values,
            key_action_query=key_action_query,
            key_action_key=key_action_key,
            key_action_value=key_action_value,
            key_action_output=key_action_output,
            key_action_depth=key_action_depth,
            key_state_action=key_state_action,
        )
        self._init_transition_memory(config, scale=scale)
        self._init_transition_tape(config)
        self._init_transition_processor(config)
        self._init_transition_replay(config)
        workspace_scale, workspace_width = self._init_action_workspace_and_family(config)
        self._init_action_causal_and_answer(
            config,
            workspace_width=workspace_width,
            workspace_scale=workspace_scale,
            key_literal_values=key_literal_values,
            scale=scale,
            key_answer_query=key_answer_query,
            key_answer_key=key_answer_key,
            key_answer_value=key_answer_value,
            key_answer_output=key_answer_output,
        )
        self._init_process_reader_and_answer_gates(config, scale=scale)

    def _init_correction_and_halt(
        self,
        config: UnifiedRecurrenceConfig,
        scale,
        key_a,
        key_depth,
        key_memory,
        key_halt,
    ) -> None:
        self.correction_a = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_a,
            ).astype(mx.float32)
            * scale
        )
        self.correction_b = mx.zeros(
            (config.correction_rank, config.hidden_size),
            dtype=mx.float32,
        )
        self.depth_scale = (
            mx.random.normal(
                (config.depth_basis_size, config.correction_rank),
                key=key_depth,
            ).astype(mx.float32)
            * 0.01
        )
        self.memory_write_weight = (
            mx.random.normal((config.hidden_size,), key=key_memory).astype(mx.float32)
            * scale
        )
        self.memory_write_bias = mx.array(-6.0, dtype=mx.float32)
        self.transport_depth_weight = mx.zeros(
            (config.depth_basis_size,),
            dtype=mx.float32,
        )
        # Zero initialization preserves the CP203 depth-only operator exactly.
        # Training can then learn to accept a useful recurrent proposal for one
        # token/task while rejecting destructive motion for another.
        self.transport_state_weight = mx.zeros(
            (config.hidden_size,),
            dtype=mx.float32,
        )
        self.transport_motion_weight = mx.zeros(
            (config.hidden_size,),
            dtype=mx.float32,
        )
        # Re-entry starts conservative. Step zero bypasses this gate exactly,
        # retaining base-forward parity; later steps learn how much of the new
        # window state can be admitted without leaving the coda's manifold.
        self.transport_bias = mx.array(-0.5, dtype=mx.float32)
        # p stays inside (0.5, 1.0). The cumulative displacement of unseen
        # passes therefore grows sublinearly instead of linearly with depth,
        # while gradients can still relax the schedule toward sqrt decay.
        self.transport_decay_logit = mx.array(4.0, dtype=mx.float32)
        self.halt_state_weight = (
            mx.random.normal((config.hidden_size,), key=key_halt).astype(mx.float32)
            * scale
        )
        self.halt_motion_weight = mx.array(0.0, dtype=mx.float32)
        self.halt_bias = mx.array(-6.0, dtype=mx.float32)

    def _init_state_registers_and_actions(
        self,
        config: UnifiedRecurrenceConfig,
        scale,
        key_state,
        key_state_slots,
        key_state_values,
        key_transition_query,
        key_transition_key,
        key_transition_value,
        key_transition_self,
        key_transition_output,
        key_transition_depth,
        key_action_slots,
        key_action_values,
        key_action_query,
        key_action_key,
        key_action_value,
        key_action_output,
        key_action_depth,
        key_state_action,
    ) -> None:
        self.state_readout_weight = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.hidden_size,
                    config.state_cardinality,
                ),
                key=key_state,
            ).astype(mx.float32)
            * scale
        )
        self.state_readout_bias = mx.zeros(
            (config.state_slots, config.state_cardinality),
            dtype=mx.float32,
        )
        self.state_slot_embeddings = (
            mx.random.normal(
                (config.state_slots, config.hidden_size),
                key=key_state_slots,
            ).astype(mx.float32)
            * 0.02
        )
        self.state_value_embeddings = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.state_cardinality,
                    config.hidden_size,
                ),
                key=key_state_values,
            ).astype(mx.float32)
            * 0.02
        )
        self.state_transition_query = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.hidden_size,
                    config.correction_rank,
                ),
                key=key_transition_query,
            ).astype(mx.float32)
            * scale
        )
        self.state_transition_key = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_transition_key,
            ).astype(mx.float32)
            * scale
        )
        self.state_transition_value = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_transition_value,
            ).astype(mx.float32)
            * scale
        )
        self.state_transition_self = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.hidden_size,
                    config.correction_rank,
                ),
                key=key_transition_self,
            ).astype(mx.float32)
            * scale
        )
        self.state_transition_output = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.correction_rank,
                    config.state_cardinality,
                ),
                key=key_transition_output,
            ).astype(mx.float32)
            / math.sqrt(config.correction_rank)
        )
        self.state_transition_depth = (
            mx.random.normal(
                (config.depth_basis_size, config.correction_rank),
                key=key_transition_depth,
            ).astype(mx.float32)
            * 0.01
        )
        self.state_transition_bias = mx.zeros(
            (config.state_slots, config.state_cardinality),
            dtype=mx.float32,
        )
        # Initial parsing and recurrent transitions are separate causal roles.
        # Start from identical tensors for behavior parity, then let their
        # independently owned objectives train without overwriting each other.
        self.initial_state_query = mx.array(self.state_transition_query)
        self.initial_state_key = mx.array(self.state_transition_key)
        self.initial_state_value = mx.array(self.state_transition_value)
        self.initial_state_output = mx.array(self.state_transition_output)
        self.initial_state_bias = mx.array(self.state_transition_bias)
        self.action_slot_embeddings = (
            mx.random.normal(
                (config.action_slots, config.hidden_size), key=key_action_slots
            ).astype(mx.float32)
            * 0.02
        )
        self.action_value_embeddings = (
            mx.random.normal(
                (
                    config.action_slots,
                    config.action_cardinality,
                    config.hidden_size,
                ),
                key=key_action_values,
            ).astype(mx.float32)
            * 0.02
        )
        self.action_query = (
            mx.random.normal(
                (config.action_slots, config.hidden_size, config.correction_rank),
                key=key_action_query,
            ).astype(mx.float32)
            * scale
        )
        self.action_key = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank), key=key_action_key
            ).astype(mx.float32)
            * scale
        )
        self.action_value = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank), key=key_action_value
            ).astype(mx.float32)
            * scale
        )
        self.action_output = (
            mx.random.normal(
                (config.action_slots, config.correction_rank, config.action_cardinality),
                key=key_action_output,
            ).astype(mx.float32)
            / math.sqrt(config.correction_rank)
        )
        self.action_depth = (
            mx.random.normal(
                (config.depth_basis_size, config.correction_rank), key=key_action_depth
            ).astype(mx.float32)
            * 0.01
        )
        self.action_bias = mx.zeros(
            (config.action_slots, config.action_cardinality), dtype=mx.float32
        )
        self.state_action_projection = (
            mx.random.normal(
                (config.action_slots, config.hidden_size, config.correction_rank),
                key=key_state_action,
            ).astype(mx.float32)
            * scale
        )

    def _init_transition_memory(self, config: UnifiedRecurrenceConfig, scale) -> None:
        (
            key_transition_memory_input,
            key_transition_memory_reset_input,
            key_transition_memory_reset_recurrent,
            key_transition_memory_update_input,
            key_transition_memory_update_recurrent,
            key_transition_memory_candidate_input,
            key_transition_memory_candidate_recurrent,
            key_transition_memory_cross_input,
            key_transition_memory_cross_recurrent,
            key_transition_memory_depth,
        ) = mx.random.split(
            mx.random.key(config.initialization_seed ^ 0x54524D45),
            num=10,
        )
        # Keep every typed action field in its own recurrent cell. The former
        # history summary compressed the complete instruction into one vector
        # with a fixed 0.5 decay, which made order observable but did not give
        # the transition a durable representation of each operand. This GRU-
        # style tape retains field identity and learns what to keep, replace,
        # or expose to each state register. Its output is exactly zero at
        # attachment, preserving parent behavior until transition supervision
        # trains the new tissue.
        transition_scale = 1.0 / math.sqrt(config.correction_rank)
        transition_shape = (
            config.action_slots,
            config.correction_rank,
            config.correction_rank,
        )
        self.transition_memory_input = (
            mx.random.normal(
                (config.action_slots, config.hidden_size, config.correction_rank),
                key=key_transition_memory_input,
            ).astype(mx.float32)
            * scale
        )
        self.transition_memory_reset_input = (
            mx.random.normal(
                transition_shape,
                key=key_transition_memory_reset_input,
            ).astype(mx.float32)
            * transition_scale
        )
        self.transition_memory_reset_recurrent = (
            mx.random.normal(
                transition_shape,
                key=key_transition_memory_reset_recurrent,
            ).astype(mx.float32)
            * transition_scale
        )
        self.transition_memory_reset_bias = mx.zeros(
            (config.action_slots, config.correction_rank),
            dtype=mx.float32,
        )
        self.transition_memory_update_input = (
            mx.random.normal(
                transition_shape,
                key=key_transition_memory_update_input,
            ).astype(mx.float32)
            * transition_scale
        )
        self.transition_memory_update_recurrent = (
            mx.random.normal(
                transition_shape,
                key=key_transition_memory_update_recurrent,
            ).astype(mx.float32)
            * transition_scale
        )
        self.transition_memory_update_bias = mx.zeros(
            (config.action_slots, config.correction_rank),
            dtype=mx.float32,
        )
        self.transition_memory_candidate_input = (
            mx.random.normal(
                transition_shape,
                key=key_transition_memory_candidate_input,
            ).astype(mx.float32)
            * transition_scale
        )
        self.transition_memory_candidate_recurrent = (
            mx.random.normal(
                transition_shape,
                key=key_transition_memory_candidate_recurrent,
            ).astype(mx.float32)
            * transition_scale
        )
        transition_cross_scale = 1.0 / math.sqrt(
            config.action_slots * config.correction_rank
        )
        transition_cross_shape = (
            config.action_slots,
            config.action_slots,
            config.correction_rank,
            config.correction_rank,
        )
        self.transition_memory_cross_input = (
            mx.random.normal(
                transition_cross_shape,
                key=key_transition_memory_cross_input,
            ).astype(mx.float32)
            * transition_cross_scale
        )
        self.transition_memory_cross_recurrent = (
            mx.random.normal(
                transition_cross_shape,
                key=key_transition_memory_cross_recurrent,
            ).astype(mx.float32)
            * transition_cross_scale
        )
        self.transition_memory_candidate_bias = mx.zeros(
            (config.action_slots, config.correction_rank),
            dtype=mx.float32,
        )
        self.transition_memory_depth = (
            mx.random.normal(
                (config.depth_basis_size, config.correction_rank),
                key=key_transition_memory_depth,
            ).astype(mx.float32)
            * 0.01
        )
        self.transition_memory_output = mx.zeros(
            (
                config.state_slots,
                config.action_slots,
                config.correction_rank,
                config.correction_rank,
            ),
            dtype=mx.float32,
        )

    def _init_transition_tape(self, config: UnifiedRecurrenceConfig) -> None:
        (
            key_transition_tape_key,
            key_transition_tape_value,
            key_transition_tape_position_key,
            key_transition_tape_position_value,
            key_transition_tape_state_query,
            key_transition_tape_action_query,
        ) = mx.random.split(
            mx.random.key(config.initialization_seed ^ 0x54504552),
            num=6,
        )
        # The gated memory above is a useful learned summary, but it is not an
        # information-preserving view of the public program.  Three frontier
        # families have legal transitions that are ambiguous from local state
        # and current action yet identifiable from the complete public prefix.
        # Keep that prefix intact and let every state register query it at the
        # moment of transition.  Only the final projection is zero-attached,
        # preserving exact parent behavior while avoiding an irreversible GRU
        # bottleneck once this reader is trained.
        tape_scale = 1.0 / math.sqrt(config.correction_rank)
        self.transition_tape_key = (
            mx.random.normal(
                (
                    config.action_slots,
                    config.action_cardinality,
                    config.correction_rank,
                ),
                key=key_transition_tape_key,
            ).astype(mx.float32)
            * tape_scale
        )
        self.transition_tape_value = (
            mx.random.normal(
                (
                    config.action_slots,
                    config.action_cardinality,
                    config.correction_rank,
                ),
                key=key_transition_tape_value,
            ).astype(mx.float32)
            * tape_scale
        )
        self.transition_tape_position_key = (
            mx.random.normal(
                (config.depth_basis_size, config.correction_rank),
                key=key_transition_tape_position_key,
            ).astype(mx.float32)
            * 0.01
        )
        self.transition_tape_position_value = (
            mx.random.normal(
                (config.depth_basis_size, config.correction_rank),
                key=key_transition_tape_position_value,
            ).astype(mx.float32)
            * 0.01
        )
        self.transition_tape_state_query = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.state_cardinality,
                    config.correction_rank,
                ),
                key=key_transition_tape_state_query,
            ).astype(mx.float32)
            * tape_scale
        )
        self.transition_tape_action_query = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.action_slots,
                    config.action_cardinality,
                    config.correction_rank,
                ),
                key=key_transition_tape_action_query,
            ).astype(mx.float32)
            / math.sqrt(config.action_slots * config.correction_rank)
        )
        self.transition_tape_output = mx.zeros(
            (
                config.state_slots,
                config.correction_rank,
                config.correction_rank,
            ),
            dtype=mx.float32,
        )

    def _init_transition_processor(self, config: UnifiedRecurrenceConfig) -> None:
        (
            key_transition_processor_state,
            key_transition_processor_action_left,
            key_transition_processor_action_right,
            key_transition_processor_history,
            key_transition_processor_up,
            key_transition_processor_down,
        ) = mx.random.split(
            mx.random.key(config.initialization_seed ^ 0x54505243),
            num=6,
        )
        (key_transition_opcode_interaction_up,) = mx.random.split(
            mx.random.key(config.initialization_seed ^ 0x4F504958),
            num=1,
        )
        # Preserve exact categorical identity before learning the transition
        # algebra. The old state head added independently projected state and
        # action summaries, forcing one tanh to discover every operand
        # interaction. This processor receives deterministic category features
        # and exposes state/action/history products explicitly. Its final head
        # starts at exact zero, so attaching it cannot change parent behavior.
        processor_scale = 1.0 / math.sqrt(config.correction_rank)
        self.transition_processor_state_projection = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.correction_rank,
                    config.correction_rank,
                ),
                key=key_transition_processor_state,
            ).astype(mx.float32)
            * processor_scale
        )
        # Existing processor checkpoints only let an output register inspect
        # its own prior value. That makes cross-register predicates (for
        # example comparing a candidate against a two-register score)
        # structurally unrepresentable. This bank starts at exact zero so it
        # can be attached to trained parents without changing one logit.
        self.transition_processor_state_cross_projection = mx.zeros(
            (
                config.state_slots,
                config.state_slots,
                config.correction_rank,
                config.correction_rank,
            ),
            dtype=mx.float32,
        )
        processor_action_shape = (
            config.state_slots,
            config.action_slots,
            config.correction_rank,
            config.correction_rank,
        )
        self.transition_processor_action_left = (
            mx.random.normal(
                processor_action_shape,
                key=key_transition_processor_action_left,
            ).astype(mx.float32)
            / math.sqrt(config.action_slots * config.correction_rank)
        )
        self.transition_processor_action_right = (
            mx.random.normal(
                processor_action_shape,
                key=key_transition_processor_action_right,
            ).astype(mx.float32)
            / math.sqrt(config.action_slots * config.correction_rank)
        )
        self.transition_processor_history_projection = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.correction_rank,
                    config.correction_rank,
                ),
                key=key_transition_processor_history,
            ).astype(mx.float32)
            * processor_scale
        )
        processor_width = 4 * config.correction_rank
        interaction_width = 9 * config.correction_rank
        self.transition_processor_interaction_up = (
            mx.random.normal(
                (config.state_slots, interaction_width, processor_width),
                key=key_transition_processor_up,
            ).astype(mx.float32)
            / math.sqrt(interaction_width)
        )
        self.transition_processor_interaction_bias = mx.zeros(
            (config.state_slots, processor_width),
            dtype=mx.float32,
        )
        self.transition_processor_interaction_down = (
            mx.random.normal(
                (
                    config.state_slots,
                    processor_width,
                    config.correction_rank,
                ),
                key=key_transition_processor_down,
            ).astype(mx.float32)
            / math.sqrt(processor_width)
        )
        self.transition_processor_output = mx.zeros(
            (
                config.state_slots,
                config.correction_rank,
                config.state_cardinality,
            ),
            dtype=mx.float32,
        )
        opcode_interaction_rank = max(4, config.correction_rank // 8)
        self.transition_processor_opcode_interaction_up = (
            mx.random.normal(
                (
                    config.action_cardinality,
                    config.state_slots,
                    interaction_width,
                    opcode_interaction_rank,
                ),
                key=key_transition_opcode_interaction_up,
            ).astype(mx.float32)
            / math.sqrt(interaction_width)
        )
        self.transition_processor_opcode_interaction_down = mx.zeros(
            (
                config.action_cardinality,
                config.state_slots,
                opcode_interaction_rank,
                config.correction_rank,
            ),
            dtype=mx.float32,
        )
        # Public opcodes select distinct transition algebras.  Specializing
        # only the categorical head leaves every opcode sharing the same
        # hidden computation, so incompatible algorithms can still overwrite
        # one another before decoding.  This residual bank operates inside
        # the processor and attaches as an exact no-op.  A uniform router uses
        # the same tensor inventory as a matched-capacity control.
        self.transition_processor_opcode_hidden = mx.zeros(
            (
                config.action_cardinality,
                config.state_slots,
                config.correction_rank,
                config.correction_rank,
            ),
            dtype=mx.float32,
        )
        # Different public opcodes implement genuinely different state
        # machines. A shared head made sparse examples from one family rewrite
        # another family's readout. This zero-attached expert bank preserves
        # parent behavior while giving each opcode an independently trainable
        # categorical transition head over the shared processor features.
        self.transition_processor_opcode_output = mx.zeros(
            (
                config.action_cardinality,
                config.state_slots,
                config.correction_rank,
                config.state_cardinality,
            ),
            dtype=mx.float32,
        )

    def _init_transition_replay(self, config: UnifiedRecurrenceConfig) -> None:
        (
            key_transition_replay_key,
            key_transition_replay_value,
            key_transition_replay_position_key,
            key_transition_replay_position_value,
            key_transition_replay_query,
            key_transition_replay_projection,
        ) = mx.random.split(
            mx.random.key(config.initialization_seed ^ 0x52504C59),
            num=6,
        )
        # A recurrent state cannot repair itself if every recovery feature is
        # queried through that same state.  This separate causal reader sees
        # only the public action prefix, preserving field and order identity.
        # Its candidate heads attach at exact zero, so old checkpoints and the
        # disabled/lesioned arm remain behavior-identical before training.
        replay_scale = 1.0 / math.sqrt(config.correction_rank)
        self.transition_replay_key = (
            mx.random.normal(
                (
                    config.action_slots,
                    config.action_cardinality,
                    config.correction_rank,
                ),
                key=key_transition_replay_key,
            ).astype(mx.float32)
            * replay_scale
        )
        self.transition_replay_value = (
            mx.random.normal(
                (
                    config.action_slots,
                    config.action_cardinality,
                    config.correction_rank,
                ),
                key=key_transition_replay_value,
            ).astype(mx.float32)
            * replay_scale
        )
        self.transition_replay_position_key = (
            mx.random.normal(
                (config.depth_basis_size, config.correction_rank),
                key=key_transition_replay_position_key,
            ).astype(mx.float32)
            * 0.01
        )
        self.transition_replay_position_value = (
            mx.random.normal(
                (config.depth_basis_size, config.correction_rank),
                key=key_transition_replay_position_value,
            ).astype(mx.float32)
            * 0.01
        )
        self.transition_replay_query = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.action_slots,
                    config.correction_rank,
                ),
                key=key_transition_replay_query,
            ).astype(mx.float32)
            * replay_scale
        )
        self.transition_replay_projection = (
            mx.random.normal(
                (
                    config.state_slots,
                    config.action_slots,
                    config.correction_rank,
                    config.correction_rank,
                ),
                key=key_transition_replay_projection,
            ).astype(mx.float32)
            / math.sqrt(config.action_slots * config.correction_rank)
        )
        self.transition_replay_output = mx.zeros(
            (
                config.state_slots,
                config.correction_rank,
                config.state_cardinality,
            ),
            dtype=mx.float32,
        )
        self.transition_replay_opcode_output = mx.zeros(
            (
                config.action_cardinality,
                config.state_slots,
                config.correction_rank,
                config.state_cardinality,
            ),
            dtype=mx.float32,
        )
        # Gate features are local confidence, replay confidence, categorical
        # disagreement and normalized prefix depth.  Zero replay logits
        # makes attachment exact even though the gate itself remains trainable.
        self.transition_replay_gate_weight = mx.zeros(
            (config.state_slots, 4),
            dtype=mx.float32,
        )
        self.transition_replay_gate_bias = mx.zeros(
            (config.state_slots,),
            dtype=mx.float32,
        )

    def _init_action_workspace_and_family(self, config: UnifiedRecurrenceConfig) -> tuple[Any, ...]:
        (
            key_action_workspace_seed,
            key_action_workspace_depth,
            key_action_workspace_cross_query,
            key_action_workspace_cross_key,
            key_action_workspace_cross_value,
            key_action_workspace_cross_output,
            key_action_workspace_self_query,
            key_action_workspace_self_key,
            key_action_workspace_self_value,
            key_action_workspace_self_output,
            key_action_workspace_ff_up,
            key_action_workspace_ff_down,
        ) = mx.random.split(
            mx.random.key(config.initialization_seed ^ 0x4143544E),
            num=12,
        )
        # The legacy action head is a single cross-attention read. Broad
        # programs require several action fields to jointly retain evidence,
        # state and execution order. This fixed-cost workspace refines the
        # eight action slots without quadratic prompt attention. Its final
        # projection starts at exact zero, so attaching it to proven parent
        # tissue is behavior-identical until the action objective trains it.
        workspace_width = min(config.hidden_size, 4 * config.correction_rank)
        workspace_scale = 1.0 / math.sqrt(workspace_width)
        hidden_scale = 1.0 / math.sqrt(config.hidden_size)
        self.action_workspace_seed = (
            mx.random.normal(
                (config.correction_rank, workspace_width),
                key=key_action_workspace_seed,
            ).astype(mx.float32)
            / math.sqrt(config.correction_rank)
        )
        self.action_workspace_depth = (
            mx.random.normal(
                (config.depth_basis_size, workspace_width),
                key=key_action_workspace_depth,
            ).astype(mx.float32)
            * 0.01
        )
        self.action_workspace_cross_query = (
            mx.random.normal(
                (2, workspace_width, workspace_width),
                key=key_action_workspace_cross_query,
            ).astype(mx.float32)
            * workspace_scale
        )
        self.action_workspace_cross_key = (
            mx.random.normal(
                (2, config.hidden_size, workspace_width),
                key=key_action_workspace_cross_key,
            ).astype(mx.float32)
            * hidden_scale
        )
        self.action_workspace_cross_value = (
            mx.random.normal(
                (2, config.hidden_size, workspace_width),
                key=key_action_workspace_cross_value,
            ).astype(mx.float32)
            * hidden_scale
        )
        self.action_workspace_cross_output = (
            mx.random.normal(
                (2, workspace_width, workspace_width),
                key=key_action_workspace_cross_output,
            ).astype(mx.float32)
            * workspace_scale
        )
        self.action_workspace_self_query = (
            mx.random.normal(
                (2, workspace_width, workspace_width),
                key=key_action_workspace_self_query,
            ).astype(mx.float32)
            * workspace_scale
        )
        self.action_workspace_self_key = (
            mx.random.normal(
                (2, workspace_width, workspace_width),
                key=key_action_workspace_self_key,
            ).astype(mx.float32)
            * workspace_scale
        )
        self.action_workspace_self_value = (
            mx.random.normal(
                (2, workspace_width, workspace_width),
                key=key_action_workspace_self_value,
            ).astype(mx.float32)
            * workspace_scale
        )
        self.action_workspace_self_output = (
            mx.random.normal(
                (2, workspace_width, workspace_width),
                key=key_action_workspace_self_output,
            ).astype(mx.float32)
            * workspace_scale
        )
        self.action_workspace_ff_up = (
            mx.random.normal(
                (2, workspace_width, 4 * workspace_width),
                key=key_action_workspace_ff_up,
            ).astype(mx.float32)
            * workspace_scale
        )
        self.action_workspace_ff_down = (
            mx.random.normal(
                (2, 4 * workspace_width, workspace_width),
                key=key_action_workspace_ff_down,
            ).astype(mx.float32)
            / math.sqrt(4 * workspace_width)
        )
        self.action_workspace_output = mx.zeros(
            (config.action_slots, workspace_width, config.action_cardinality),
            dtype=mx.float32,
        )
        # Frontier families execute different algorithms and assign different
        # meanings to the same argument slots. One shared projection collapsed
        # to a cross-family class prior in CP425. Public family declarations may
        # select an isolated expert, but no private program or answer enters the
        # route. Zero outputs preserve every parent logit before training.
        self.action_family_output = mx.zeros(
            (
                FRONTIER_ACTION_EXPERT_COUNT,
                config.action_slots,
                workspace_width,
                config.action_cardinality,
            ),
            dtype=mx.float32,
        )
        self.action_family_bias = mx.zeros(
            (
                FRONTIER_ACTION_EXPERT_COUNT,
                config.action_slots,
                config.action_cardinality,
            ),
            dtype=mx.float32,
        )
        kernel_capacity = 128
        self.action_family_kernel_mean = mx.zeros(
            (FRONTIER_ACTION_EXPERT_COUNT, config.action_slots, workspace_width),
            dtype=mx.float32,
        )
        self.action_family_kernel_inv_scale = mx.zeros_like(
            self.action_family_kernel_mean
        )
        self.action_family_kernel_prototypes = mx.zeros(
            (
                FRONTIER_ACTION_EXPERT_COUNT,
                config.action_slots,
                kernel_capacity,
                workspace_width,
            ),
            dtype=mx.float32,
        )
        self.action_family_kernel_coefficients = mx.zeros(
            (
                FRONTIER_ACTION_EXPERT_COUNT,
                config.action_slots,
                kernel_capacity,
                config.action_cardinality,
            ),
            dtype=mx.float32,
        )
        self.action_family_kernel_mask = mx.zeros(
            (FRONTIER_ACTION_EXPERT_COUNT, config.action_slots, kernel_capacity),
            dtype=mx.float32,
        )
        self.action_family_kernel_gamma = mx.zeros(
            (FRONTIER_ACTION_EXPERT_COUNT, config.action_slots),
            dtype=mx.float32,
        )
        return workspace_scale, workspace_width

    def _init_action_causal_and_answer(
        self,
        config: UnifiedRecurrenceConfig,
        workspace_width,
        workspace_scale,
        key_literal_values,
        scale,
        key_answer_query,
        key_answer_key,
        key_answer_value,
        key_answer_output,
    ) -> None:
        (
            key_action_causal_values,
            key_action_causal_input,
            key_action_causal_state,
            key_action_causal_prior,
        ) = mx.random.split(
            mx.random.key(config.initialization_seed ^ 0x43415553),
            num=4,
        )
        (
            key_action_literal_binding_query,
            key_action_literal_binding_key,
        ) = mx.random.split(
            mx.random.key(config.initialization_seed ^ 0x42494E44),
            num=2,
        )
        # A program instruction is not eight independent labels. The opcode
        # constrains its arguments, each emitted argument constrains those that
        # follow, and the previous instruction can matter to the next recurrent
        # step. This bounded autoregressive extension supplies that causal graph.
        # Its output starts at exact zero, preserving every parent logit until
        # the typed-action objective trains it.
        self.action_causal_value_embeddings = (
            mx.random.normal(
                (
                    config.action_slots,
                    config.action_cardinality,
                    workspace_width,
                ),
                key=key_action_causal_values,
            ).astype(mx.float32)
            * 0.02
        )
        self.action_causal_input_projection = (
            mx.random.normal(
                (workspace_width, workspace_width),
                key=key_action_causal_input,
            ).astype(mx.float32)
            * workspace_scale
        )
        self.action_causal_state_projection = (
            mx.random.normal(
                (workspace_width, workspace_width),
                key=key_action_causal_state,
            ).astype(mx.float32)
            * workspace_scale
        )
        self.action_causal_prior_projection = (
            mx.random.normal(
                (workspace_width, workspace_width),
                key=key_action_causal_prior,
            ).astype(mx.float32)
            * workspace_scale
        )
        self.action_causal_output = mx.zeros(
            (config.action_slots, workspace_width, config.action_cardinality),
            dtype=mx.float32,
        )
        # Prompt numbers remain exact observations, but their roles are learned.
        # This pointer surface lets each action field bind one ordered public
        # literal and choose a general numeric representation without requiring
        # the transformer to reconstruct radix arithmetic in a dense vector.
        # The output weights start at zero, so adding the surface to a parent
        # checkpoint is exactly behavior-preserving until process supervision
        # establishes useful bindings.
        binding_rank = min(workspace_width, 2 * config.correction_rank)
        self.action_literal_binding_query = (
            mx.random.normal(
                (config.action_slots, workspace_width, binding_rank),
                key=key_action_literal_binding_query,
            ).astype(mx.float32)
            / math.sqrt(workspace_width)
        )
        self.action_literal_binding_key = (
            mx.random.normal(
                (config.hidden_size, binding_rank),
                key=key_action_literal_binding_key,
            ).astype(mx.float32)
            / math.sqrt(config.hidden_size)
        )
        self.action_literal_binding_output = mx.zeros(
            (config.action_slots, len(ACTION_LITERAL_BINDING_TRANSFORMS)),
            dtype=mx.float32,
        )
        self.action_literal_binding_family_output = mx.zeros(
            (
                FRONTIER_ACTION_EXPERT_COUNT,
                config.action_slots,
                len(ACTION_LITERAL_BINDING_TRANSFORMS),
            ),
            dtype=mx.float32,
        )
        self.literal_value_embeddings = (
            mx.random.normal(
                (LITERAL_MAX_VALUE + 1, config.hidden_size), key=key_literal_values
            ).astype(mx.float32)
            * 0.02
        )
        self.answer_query = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_answer_query,
            ).astype(mx.float32)
            * scale
        )
        self.answer_key = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_answer_key,
            ).astype(mx.float32)
            * scale
        )
        self.answer_value = (
            mx.random.normal(
                (config.hidden_size, config.correction_rank),
                key=key_answer_value,
            ).astype(mx.float32)
            * scale
        )
        self.answer_output = (
            mx.random.normal(
                (config.correction_rank, config.hidden_size),
                key=key_answer_output,
            ).astype(mx.float32)
            / math.sqrt(config.correction_rank)
        )

    def _init_process_reader_and_answer_gates(self, config: UnifiedRecurrenceConfig, scale) -> None:
        (
            key_process_reader_1_query,
            key_process_reader_1_key,
            key_process_reader_1_value,
            key_process_reader_1_output,
            key_process_reader_2_query,
            key_process_reader_2_key,
            key_process_reader_2_value,
            key_process_reader_2_output,
        ) = mx.random.split(
            mx.random.key(config.initialization_seed ^ 0x50524F43),
            num=8,
        )
        # Causal process composition and answer extraction are distinct roles.
        # Sharing these projections made one objective erase progress on the
        # other during CP411. This small reader is independently trainable and
        # migrates into proven parent tissue under an exact inventory receipt.
        process_reader_rank = min(config.hidden_size, 4 * config.correction_rank)
        self.process_reader_1_query = (
            mx.random.normal(
                (config.hidden_size, process_reader_rank),
                key=key_process_reader_1_query,
            ).astype(mx.float32)
            * scale
        )
        self.process_reader_1_key = (
            mx.random.normal(
                (config.hidden_size, process_reader_rank),
                key=key_process_reader_1_key,
            ).astype(mx.float32)
            * scale
        )
        self.process_reader_1_value = (
            mx.random.normal(
                (config.hidden_size, process_reader_rank),
                key=key_process_reader_1_value,
            ).astype(mx.float32)
            * scale
        )
        self.process_reader_1_output = (
            mx.random.normal(
                (process_reader_rank, config.hidden_size),
                key=key_process_reader_1_output,
            ).astype(mx.float32)
            / math.sqrt(process_reader_rank)
        )
        self.process_reader_1_gate_logit = mx.array(-2.0, dtype=mx.float32)
        self.process_reader_2_query = (
            mx.random.normal(
                (config.hidden_size, process_reader_rank),
                key=key_process_reader_2_query,
            ).astype(mx.float32)
            * scale
        )
        self.process_reader_2_key = (
            mx.random.normal(
                (config.hidden_size, process_reader_rank),
                key=key_process_reader_2_key,
            ).astype(mx.float32)
            * scale
        )
        self.process_reader_2_value = (
            mx.random.normal(
                (config.hidden_size, process_reader_rank),
                key=key_process_reader_2_value,
            ).astype(mx.float32)
            * scale
        )
        self.process_reader_2_output = (
            mx.random.normal(
                (process_reader_rank, config.hidden_size),
                key=key_process_reader_2_output,
            ).astype(mx.float32)
            / math.sqrt(process_reader_rank)
        )
        self.process_reader_2_gate_logit = mx.array(-2.0, dtype=mx.float32)
        self.answer_gate_query = mx.zeros(
            (config.hidden_size, 1), dtype=mx.float32
        )
        self.answer_gate_logit = mx.array(-2.0, dtype=mx.float32)
        self.answer_role_projection = mx.zeros(
            (config.hidden_size, config.state_slots + 1), dtype=mx.float32
        )
        self.answer_role_bias = mx.concatenate(
            [mx.array([6.0], dtype=mx.float32), mx.zeros((config.state_slots,))]
        )
        self.answer_place_projection = mx.zeros(
            (config.hidden_size, 3), dtype=mx.float32
        )
        self.answer_place_state_projection = mx.zeros(
            (config.hidden_size, 3), dtype=mx.float32
        )
        self.answer_place_width_projection = mx.zeros(
            (2, 3), dtype=mx.float32
        )
        self.answer_place_bias = mx.array((6.0, 0.0, 0.0), dtype=mx.float32)
        # A high-confidence learned role/place decision must be authoritative.
        # The no-op prior is carried by the role/place ``none`` classes, whose
        # mass makes pointer confidence structurally zero on syntax positions.
        self.answer_digit_gate_logit = mx.array(4.0, dtype=mx.float32)
        self.literal_grounding_logit = mx.array(-1.1, dtype=mx.float32)
        self.state_literal_copy_logit = mx.array(
            (-4.0, *([0.5] * (config.state_slots - 2)), -4.0),
            dtype=mx.float32,
        )
        self.initial_state_literal_copy_logit = mx.array(
            self.state_literal_copy_logit
        )
        self.action_literal_copy_logit = mx.array(
            (-4.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -4.0),
            dtype=mx.float32,
        )
        self.opcode_copy_logit = mx.array(1.5, dtype=mx.float32)

    def depth_features(self, step: int) -> Any:
        if type(step) is not int or step < 0:
            raise ValueError("recurrent step must be a non-negative integer")
        u = float(step) / float(step + 1) if step else 0.0
        return mx.array(
            [u ** (index + 1) for index in range(self.config.depth_basis_size)],
            dtype=mx.float32,
        )

    def correction(self, hidden: Any, step: int) -> Any:
        if type(step) is not int or step < 0:
            raise ValueError("correction step must be a non-negative integer")
        # T=1 is the semantic anchor.  Recurrent control tissue must earn an
        # improvement through additional passes, never by silently rewriting
        # the baseline against which those passes are compared.
        if step == 0:
            return hidden
        features = self.depth_features(step)
        scale = 1.0 + features @ self.depth_scale
        low_rank = (hidden.astype(mx.float32) @ self.correction_a) * scale
        delta = low_rank @ self.correction_b
        return hidden + delta.astype(hidden.dtype)

    def memory_write_probabilities(self, previous: Any, candidate: Any) -> Any:
        disagreement = (candidate - previous).astype(mx.float32)
        logits = disagreement @ self.memory_write_weight + self.memory_write_bias
        return mx.sigmoid(logits)[..., None]

    def transport_gate(
        self,
        step: int,
        previous: Any | None = None,
        candidate: Any | None = None,
    ) -> Any:
        if type(step) is not int or step < 0:
            raise ValueError("transport step must be a non-negative integer")
        if (previous is None) != (candidate is None):
            raise ValueError("transport state and candidate must be supplied together")
        if step == 0:
            return mx.array(1.0, dtype=mx.float32)
        logit = (
            self.transport_bias
            + self.depth_features(step) @ self.transport_depth_weight
        )
        if previous is not None and candidate is not None:
            if previous.shape != candidate.shape:
                raise ValueError("transport state and candidate shapes differ")
            previous_wide = previous.astype(mx.float32)
            motion = candidate.astype(mx.float32) - previous_wide
            previous_features = previous_wide / _rms(previous_wide)
            motion_features = motion / _rms(motion)
            feature_scale = 1.0 / math.sqrt(self.config.hidden_size)
            logit = logit + feature_scale * (
                previous_features @ self.transport_state_weight
                + motion_features @ self.transport_motion_weight
            )
        base = mx.sigmoid(logit)
        exponent = self.transport_decay_exponent()
        gate = base / mx.power(
            mx.array(float(step), dtype=mx.float32),
            exponent,
        )
        return gate[..., None] if previous is not None else gate

    def transport_decay_exponent(self) -> Any:
        return 0.5 + 0.5 * mx.sigmoid(self.transport_decay_logit)

    def transport(self, previous: Any, candidate: Any, step: int) -> tuple[Any, Any]:
        if step == 0:
            return candidate, self.transport_gate(step)
        matched = candidate * (_rms(previous) / _rms(candidate)).astype(
            candidate.dtype
        )
        gate = self.transport_gate(step, previous, matched)
        transported = previous + gate.astype(previous.dtype) * (matched - previous)
        return transported, gate

    def halt_probability(self, previous: Any, candidate: Any) -> Any:
        current = candidate.astype(mx.float32)
        previous_wide = previous.astype(mx.float32)
        pooled = mx.mean(current, axis=(0, 1))
        motion = mx.mean(mx.abs(current - previous_wide)) / mx.maximum(
            mx.mean(mx.abs(previous_wide)),
            1e-6,
        )
        logit = (
            pooled @ self.halt_state_weight
            + self.halt_motion_weight * (1.0 - mx.minimum(motion, 1.0))
            + self.halt_bias
        )
        return mx.sigmoid(logit)

    def initial_state_slots(self, batch_size: int, dtype: Any) -> Any:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("state slot batch size must be positive")
        return mx.broadcast_to(
            self.state_slot_embeddings.astype(dtype)[None, :, :],
            (batch_size, self.config.state_slots, self.config.hidden_size),
        )

    def encode_process_tape_entry(
        self,
        value: Any,
        *,
        step: int,
        kind: str,
    ) -> Any:
        """Attach deterministic temporal and type identity to one tape entry."""

        if type(step) is not int or step < 0:
            raise ValueError("process tape step must be a non-negative integer")
        kind_offsets = {
            "state_pre": 0,
            "action": 1,
            "state_post": 2,
            "state_delta": 3,
        }
        if kind not in kind_offsets:
            raise ValueError("process tape kind differs from the transition schema")
        if len(value.shape) != 3 or int(value.shape[-1]) != self.config.hidden_size:
            raise ValueError("process tape entry differs from the residual layout")
        # Content-only attention treats execution history as a set. Encode the
        # ordinal and event type so non-commutative traces remain distinguishable.
        ordinal = float(4 * step + kind_offsets[kind] + 1)
        dimensions = mx.arange(self.config.hidden_size, dtype=mx.float32)
        pairs = mx.floor(dimensions / 2.0)
        frequencies = mx.exp(
            -math.log(10_000.0)
            * (2.0 * pairs / float(self.config.hidden_size))
        )
        slot_offsets = mx.arange(int(value.shape[1]), dtype=mx.float32) + 1.0
        angles = (ordinal + 0.125 * slot_offsets[:, None]) * frequencies[None, :]
        even = (mx.arange(self.config.hidden_size) % 2) == 0
        identity = mx.where(even, mx.sin(angles), mx.cos(angles))
        scale = 0.25 * mx.stop_gradient(_rms(value.astype(mx.float32)))
        return value + (scale * identity[None, :, :]).astype(value.dtype)

    def contextualize_process_tape(
        self,
        memory: Any,
        mask: Any | None = None,
    ) -> Any:
        """Compose an ordered trace with causal self-attention.

        Positional identity makes order observable, but direct cross-attention
        still asks one read to infer every intermediate dependency.  This scan
        gives each live event a representation of its complete causal prefix.
        Its projections are independent from answer extraction: process
        composition and answer selection otherwise send competing gradients
        through the same parameters. Two independent causal blocks let later
        events compose already-contextualized transitions without adding a
        parallel answer model.
        """

        if (
            len(memory.shape) != 3
            or int(memory.shape[-1]) != self.config.hidden_size
            or int(memory.shape[1]) < 1
        ):
            raise ValueError("process tape memory differs from the residual layout")
        if mask is None:
            mask = mx.ones(memory.shape[:2], dtype=mx.bool_)
        if len(mask.shape) != 2 or mask.shape != memory.shape[:2]:
            raise ValueError("process tape causal mask differs")
        contextual = memory.astype(mx.float32)
        count = int(memory.shape[1])
        indices = mx.arange(count)
        causal = indices[:, None] >= indices[None, :]
        allowed = causal[None, :, :] & mask[:, :, None] & mask[:, None, :]
        layers = (
            (
                self.process_reader_1_query,
                self.process_reader_1_key,
                self.process_reader_1_value,
                self.process_reader_1_output,
                self.process_reader_1_gate_logit,
            ),
            (
                self.process_reader_2_query,
                self.process_reader_2_key,
                self.process_reader_2_value,
                self.process_reader_2_output,
                self.process_reader_2_gate_logit,
            ),
        )
        for reader_query, reader_key, reader_value, reader_output, gate_logit in layers:
            normalized = contextual / _rms(contextual)
            query = normalized @ reader_query
            key = normalized @ reader_key
            value = normalized @ reader_value
            logits = (
                mx.einsum("bnr,bmr->bnm", query, key)
                / math.sqrt(int(reader_query.shape[-1]))
            )
            logits = mx.where(allowed, logits, mx.array(-1e9, dtype=logits.dtype))
            attention = mx.softmax(logits, axis=-1)
            context = mx.einsum("bnm,bmr->bnr", attention, value)
            delta = context @ reader_output
            contextual_rms = _rms(contextual)
            delta_rms = _rms(delta)
            delta = delta * mx.minimum(1.0, contextual_rms / delta_rms)
            contextual = mx.where(
                mask[:, :, None],
                contextual + mx.sigmoid(gate_logit) * delta,
                contextual,
            )
        return contextual.astype(memory.dtype)

    def attend_answer_to_state(
        self,
        candidate: Any,
        committed_state: Any,
        *,
        state_slot_start: int,
        state_probabilities: Any | None = None,
        process_memory: Any | None = None,
        process_memory_mask: Any | None = None,
        role_logit_trajectory: list[Any] | None = None,
        place_logit_trajectory: list[Any] | None = None,
        binding_feature_trajectory: list[tuple[Any, Any, Any]] | None = None,
    ) -> Any:
        """Give answer positions a direct neural read path to typed registers."""

        stop = state_slot_start + self.config.state_slots
        answer_start = stop - 1
        if (
            len(candidate.shape) != 3
            or candidate.shape != committed_state.shape
            or int(candidate.shape[-1]) != self.config.hidden_size
            or not 0 <= state_slot_start < stop <= int(candidate.shape[1])
        ):
            raise ValueError("state-to-answer bridge layout differs")
        answer = candidate[:, answer_start:, :].astype(mx.float32)
        state = committed_state[:, state_slot_start:stop, :].astype(mx.float32)
        memory = state if process_memory is None else process_memory.astype(mx.float32)
        if (
            len(memory.shape) != 3
            or int(memory.shape[0]) != int(answer.shape[0])
            or int(memory.shape[-1]) != self.config.hidden_size
            or int(memory.shape[1]) < self.config.state_slots
        ):
            raise ValueError("answer process-memory layout differs")
        if process_memory_mask is not None and (
            len(process_memory_mask.shape) != 2
            or process_memory_mask.shape != memory.shape[:2]
        ):
            raise ValueError("answer process-memory mask differs")
        if process_memory is not None:
            memory = self.contextualize_process_tape(memory, process_memory_mask)
        query = answer @ self.answer_query
        key = memory @ self.answer_key
        value = memory @ self.answer_value
        attention_logits = (
            mx.einsum("bar,bmr->bam", query, key)
            / math.sqrt(self.config.correction_rank)
        )
        if process_memory_mask is not None:
            attention_logits = mx.where(
                process_memory_mask[:, None, :],
                attention_logits,
                mx.array(-1e9, dtype=attention_logits.dtype),
            )
        attention = mx.softmax(attention_logits, axis=-1)
        context = mx.einsum("bam,bmr->bar", attention, value)
        delta = context @ self.answer_output
        # State is useful at value-bearing positions and harmful when a frozen
        # language manifold is already emitting syntax or EOS. A global gate
        # applied the same correction everywhere and eventually produced
        # repeated braces under generated-history training. Keep the write
        # token-conditioned and no larger than the local residual scale.
        answer_rms = mx.sqrt(mx.mean(answer**2, axis=-1, keepdims=True) + 1e-6)
        delta_rms = mx.sqrt(mx.mean(delta**2, axis=-1, keepdims=True) + 1e-6)
        delta = delta * mx.minimum(1.0, answer_rms / delta_rms)
        gate = mx.sigmoid(answer @ self.answer_gate_query + self.answer_gate_logit)
        bridged = answer + gate * delta

        role_logits, place_logits = self.answer_binding_logits(
            answer,
            state,
            state_probabilities,
        )
        if binding_feature_trajectory is not None:
            if state_probabilities is None:
                raise ValueError("cached answer binding requires typed probabilities")
            binding_feature_trajectory.append(
                (
                    mx.stop_gradient(answer),
                    mx.stop_gradient(state),
                    mx.stop_gradient(state_probabilities.astype(mx.float32)),
                )
            )
        if role_logit_trajectory is not None:
            role_logit_trajectory.append(role_logits)
        if place_logit_trajectory is not None:
            place_logit_trajectory.append(place_logits)
        return mx.concatenate(
            [candidate[:, :answer_start, :], bridged.astype(candidate.dtype)],
            axis=1,
        )

    def answer_binding_logits(
        self,
        answer: Any,
        state: Any,
        state_probabilities: Any | None,
    ) -> tuple[Any, Any]:
        """Decode answer role and decimal place from reusable causal features."""

        if (
            len(answer.shape) != 3
            or len(state.shape) != 3
            or int(answer.shape[0]) != int(state.shape[0])
            or int(answer.shape[-1]) != self.config.hidden_size
            or state.shape[1:] != (
                self.config.state_slots,
                self.config.hidden_size,
            )
        ):
            raise ValueError("answer binding feature layout differs")
        role_logits = answer @ self.answer_role_projection + self.answer_role_bias
        role_probabilities = mx.softmax(role_logits.astype(mx.float32), axis=-1)
        selected_state = mx.einsum(
            "bar,brh->bah",
            role_probabilities[..., 1:],
            state.astype(mx.float32),
        )
        width_logits = mx.zeros((*answer.shape[:2], 3), dtype=mx.float32)
        if state_probabilities is not None:
            if state_probabilities.shape != (
                int(answer.shape[0]),
                self.config.state_slots,
                self.config.state_cardinality,
            ):
                raise ValueError("answer bridge state probabilities differ")
            selected_values = mx.einsum(
                "bar,brv->bav",
                role_probabilities[..., 1:],
                state_probabilities.astype(mx.float32),
            )
            width_features = mx.stack(
                (
                    mx.sum(selected_values[..., :10], axis=-1),
                    mx.sum(selected_values[..., 10:], axis=-1),
                ),
                axis=-1,
            )
            width_logits = width_features @ self.answer_place_width_projection
        # Prefix position identifies the output field, but it cannot reveal
        # whether that field's value has one or two digits. The selected typed
        # register makes digit width causally observable to the neural head.
        place_logits = (
            answer @ self.answer_place_projection
            + selected_state @ self.answer_place_state_projection
            + width_logits
            + self.answer_place_bias
        )
        return role_logits, place_logits

    def apply_answer_digit_pointer(
        self,
        logits: Any,
        role_logits: Any,
        place_logits: Any,
        state_probabilities: Any,
    ) -> Any:
        """Mix a learned terminal-register digit pointer into frozen logits."""

        if not self.config.literal_digit_token_ids:
            raise ValueError("answer digit pointer requires bound tokenizer digits")
        if (
            len(logits.shape) != 3
            or len(role_logits.shape) != 3
            or len(place_logits.shape) != 3
            or len(state_probabilities.shape) != 3
            or logits.shape[:2] != role_logits.shape[:2]
            or logits.shape[:2] != place_logits.shape[:2]
            or int(role_logits.shape[-1]) != self.config.state_slots + 1
            or int(place_logits.shape[-1]) != 3
            or state_probabilities.shape[1:] != (
                self.config.state_slots,
                self.config.state_cardinality,
            )
        ):
            raise ValueError("answer digit pointer tensor layout differs")
        vocabulary_size = int(logits.shape[-1])
        if max(self.config.literal_digit_token_ids) >= vocabulary_size:
            raise ValueError("answer digit pointer token is outside the vocabulary")

        raw_roles = mx.softmax(role_logits.astype(mx.float32), axis=-1)
        raw_places = mx.softmax(place_logits.astype(mx.float32), axis=-1)
        value_indices = mx.arange(self.config.state_cardinality)
        digit_indices = mx.arange(10)
        ones = ((value_indices[:, None] % 10) == digit_indices[None, :]).astype(
            mx.float32
        )
        tens = (
            ((value_indices[:, None] // 10) == digit_indices[None, :])
            * (value_indices[:, None] >= 10)
        ).astype(mx.float32)
        digit_probabilities_by_position: list[Any] = []
        confidences: list[Any] = []
        previous_role = mx.concatenate(
            (
                mx.ones((*raw_roles.shape[:1], 1), dtype=mx.float32),
                mx.zeros(
                    (*raw_roles.shape[:1], self.config.state_slots),
                    dtype=mx.float32,
                ),
            ),
            axis=-1,
        )
        previous_two_digit_start = mx.zeros(
            (*raw_roles.shape[:1], 1), dtype=mx.float32
        )
        previous_value_complete = mx.zeros(
            (*raw_roles.shape[:1], 1), dtype=mx.float32
        )
        for position in range(int(raw_roles.shape[1])):
            raw_role = raw_roles[:, position, :]
            raw_place = raw_places[:, position, :]
            learned_digit_mass = (
                mx.sum(raw_place[:, 1:], axis=-1, keepdims=True)
                * (1.0 - previous_value_complete)
            )
            # Once exact typed state establishes a two-digit value, its ones
            # token is mandatory. The frozen language prior often calls that
            # position syntax after seeing the tens token; allowing that local
            # guess to disengage the pointer produced plausible but wrong 19s.
            digit_mass = mx.maximum(learned_digit_mass, previous_two_digit_start)
            previous_role_mass = mx.sum(
                previous_role[:, 1:], axis=-1, keepdims=True
            )
            # A second digit is still part of the field selected at the first
            # digit.  Letting an independently classified role replace it made
            # multi-register answers read the ones digit from another slot.
            continuation = (
                previous_role_mass * previous_two_digit_start
            )
            role = (1.0 - continuation) * raw_role + continuation * previous_role
            role_mass = mx.sum(role[:, 1:], axis=-1, keepdims=True)
            selected_values = mx.einsum(
                "br,brv->bv",
                role[:, 1:],
                state_probabilities.astype(mx.float32),
            ) / mx.maximum(role_mass, 1e-8)
            one_digit_mass = mx.sum(selected_values[:, :10], axis=-1, keepdims=True)
            two_digit_mass = mx.sum(selected_values[:, 10:], axis=-1, keepdims=True)
            start = 1.0 - previous_role_mass
            tens_mass = digit_mass * start * two_digit_mass
            ones_mass = digit_mass * (
                previous_role_mass + start * one_digit_mass
            )
            digit_probabilities_by_position.append(
                tens_mass
                * ((selected_values @ tens) / mx.maximum(two_digit_mass, 1e-8))
                + ones_mass * (selected_values @ ones)
            )
            confidences.append(
                role_mass
                * digit_mass
                * mx.sigmoid(self.answer_digit_gate_logit)
            )
            previous_value_complete = (
                digit_mass * start * one_digit_mass + continuation
            )
            previous_two_digit_start = digit_mass * start * two_digit_mass
            previous_role = role
        digit_probabilities = mx.stack(digit_probabilities_by_position, axis=1)
        vocabulary = mx.arange(vocabulary_size)
        digit_tokens = mx.array(self.config.literal_digit_token_ids)
        digit_to_vocabulary = (digit_tokens[:, None] == vocabulary[None, :]).astype(
            mx.float32
        )
        pointer_probabilities = digit_probabilities @ digit_to_vocabulary
        confidence = mx.stack(confidences, axis=1)
        base_probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
        mixed = (
            (1.0 - confidence) * base_probabilities
            + confidence * pointer_probabilities
        )
        return mx.log(mx.maximum(mixed, 1e-12)).astype(logits.dtype)

    def state_logits(
        self,
        hidden: Any,
        *,
        public_token_count: int | None = None,
        state_slot_start: int | None = None,
    ) -> Any:
        """Decode typed state from causal slots, or a public-only diagnostic."""

        if len(hidden.shape) != 3 or int(hidden.shape[-1]) != self.config.hidden_size:
            raise ValueError("recurrent state readout hidden shape differs")
        if (public_token_count is None) == (state_slot_start is None):
            raise ValueError("exactly one recurrent state readout source is required")
        if state_slot_start is not None:
            if (
                type(state_slot_start) is not int
                or state_slot_start < 0
                or state_slot_start + self.config.state_slots > int(hidden.shape[1])
            ):
                raise ValueError("state slot range is outside the recurrent state")
            selected = hidden[
                :,
                state_slot_start : state_slot_start + self.config.state_slots,
                :,
            ].astype(mx.float32)
            return mx.einsum(
                "bsh,shc->bsc",
                selected,
                self.state_readout_weight,
            ) + self.state_readout_bias
        if (
            type(public_token_count) is not int
            or not 1 <= public_token_count <= int(hidden.shape[1])
        ):
            raise ValueError("public token count is outside the recurrent state")
        public_summary = hidden[:, public_token_count - 1, :].astype(mx.float32)
        return mx.einsum(
            "bh,shc->bsc",
            public_summary,
            self.state_readout_weight,
        ) + self.state_readout_bias

    def initial_state_logits(
        self,
        problem_evidence: Any,
        token_ids: Any | None = None,
    ) -> Any:
        """Decode slot-specific initial state from the complete public prefix."""

        if (
            len(problem_evidence.shape) != 3
            or int(problem_evidence.shape[-1]) != self.config.hidden_size
            or int(problem_evidence.shape[1]) < 1
        ):
            raise ValueError("initial state problem evidence shape differs")
        slot_queries = mx.einsum(
            "sh,shr->sr",
            self.state_slot_embeddings.astype(mx.float32),
            self.initial_state_query,
        )
        keys = problem_evidence.astype(mx.float32) @ self.initial_state_key
        values = problem_evidence.astype(mx.float32) @ self.initial_state_value
        attention = mx.softmax(
            mx.einsum("sr,bnr->bsn", slot_queries, keys)
            / math.sqrt(self.config.correction_rank),
            axis=-1,
        )
        context = mx.einsum("bsn,bnr->bsr", attention, values)
        logits = mx.einsum(
            "bsr,src->bsc", context, self.initial_state_output
        ) + self.initial_state_bias
        if token_ids is not None and self.config.literal_digit_token_ids:
            pointer = self._literal_pointer_logits(
                problem_evidence,
                token_ids,
                mx.broadcast_to(
                    slot_queries[None, :, :],
                    (int(problem_evidence.shape[0]),) + tuple(slot_queries.shape),
                ),
                self.initial_state_key,
                self.config.state_cardinality,
            )
            copy_strength = mx.logaddexp(
                self.initial_state_literal_copy_logit,
                mx.zeros_like(self.initial_state_literal_copy_logit),
            )
            logits = logits + copy_strength[None, :, None] * pointer
        if (
            token_ids is not None
            and self.config.literal_digit_token_ids
            and self.config.opcode_token_patterns
        ):
            opcode_contract = OpcodeObservationContract(
                self.config.opcode_token_patterns,
                self.config.opcode_context_patterns,
            )
            literal_contract = LiteralObservationContract(
                self.config.literal_digit_token_ids
            )
            public_values, recognized = opcode_contract.public_initial_states(
                token_ids.tolist(),
                literal_contract,
                slots=self.config.state_slots,
            )
            exact = self._exact_categorical_logits(
                public_values,
                slots=self.config.state_slots,
                cardinality=self.config.state_cardinality,
            )
            known = mx.array(recognized, dtype=mx.bool_)
            logits = mx.where(known[:, None, None], exact, logits)
        if (
            token_ids is not None
            and self.config.literal_digit_token_ids
            and self.config.frontier_family_token_patterns
        ):
            family_contract = FrontierFamilyObservationContract(
                self.config.frontier_family_token_patterns
            )
            literal_contract = LiteralObservationContract(
                self.config.literal_digit_token_ids,
                max_value=self.config.numeric_observation_max_value,
            )
            public_values, recognized = family_contract.public_initial_states(
                token_ids.tolist(),
                literal_contract,
                slots=self.config.state_slots,
            )
            exact = self._exact_categorical_logits(
                public_values,
                slots=self.config.state_slots,
                cardinality=self.config.state_cardinality,
            )
            known = mx.array(recognized, dtype=mx.bool_)
            logits = mx.where(known[:, None, None], exact, logits)
        return logits

    def ground_literal_evidence(self, problem_evidence: Any, token_ids: Any) -> Any:
        """Add exact bounded integers without assigning them semantic roles.

        Values inside the categorical vocabulary retain their established
        embedding exactly. Larger values are represented by an ordered radix
        pair using the same grounded codebook. This makes the full process
        machine range observable without adding one unrelated learned row per
        integer or asking the transformer to rediscover token arithmetic.
        """

        if not self.config.literal_digit_token_ids:
            return problem_evidence
        if (
            len(problem_evidence.shape) != 3
            or len(token_ids.shape) != 2
            or tuple(problem_evidence.shape[:2]) != tuple(token_ids.shape)
            or int(problem_evidence.shape[-1]) != self.config.hidden_size
        ):
            raise ValueError("literal grounding evidence and tokens differ")
        contract = LiteralObservationContract(
            self.config.literal_digit_token_ids,
            max_value=self.config.numeric_observation_max_value,
        )
        values, masks = contract.observe(token_ids.tolist())
        value_ids = mx.array(values, dtype=mx.int32)
        observed = mx.array(masks, dtype=mx.float32)[..., None]
        direct_ids = mx.minimum(value_ids, LITERAL_MAX_VALUE)
        direct = self.literal_value_embeddings[direct_ids]
        low = self.literal_value_embeddings[value_ids % PROCESS_RADIX]
        high = self.literal_value_embeddings[value_ids // PROCESS_RADIX]
        # A one-coordinate roll gives the high digit a distinct role while
        # retaining the frozen prelude's native numeric manifold.
        wide = low + mx.roll(high, shift=1, axis=-1)
        literal = mx.where(
            (value_ids <= LITERAL_MAX_VALUE)[..., None],
            direct,
            wide,
        )
        scale = mx.sigmoid(self.literal_grounding_logit)
        return problem_evidence + (
            observed * scale * literal.astype(problem_evidence.dtype)
        )

    def _literal_pointer_logits(
        self,
        problem_evidence: Any,
        token_ids: Any,
        query: Any,
        key_projection: Any,
        cardinality: int,
    ) -> Any:
        """Map learned role attention onto exact observed numeric categories."""

        contract = LiteralObservationContract(self.config.literal_digit_token_ids)
        values, masks = contract.observe(token_ids.tolist())
        return self._categorical_pointer_logits(
            problem_evidence,
            values,
            masks,
            query,
            key_projection,
            cardinality,
        )

    def _categorical_pointer_logits(
        self,
        problem_evidence: Any,
        values: Sequence[Sequence[int]],
        masks: Sequence[Sequence[bool]],
        query: Any,
        key_projection: Any,
        cardinality: int,
    ) -> Any:
        value_ids = mx.array(values, dtype=mx.int32)
        observed = mx.array(masks, dtype=mx.float32)
        keys = problem_evidence.astype(mx.float32) @ key_projection
        attention_logits = mx.einsum("bsr,bnr->bsn", query, keys) / math.sqrt(
            self.config.correction_rank
        )
        attention = mx.softmax(attention_logits, axis=-1) * observed[:, None, :]
        attention = attention / mx.maximum(
            mx.sum(attention, axis=-1, keepdims=True),
            1e-12,
        )
        categories = mx.arange(cardinality)[None, None, :]
        one_hot = (value_ids[..., None] == categories).astype(mx.float32)
        probabilities = mx.einsum("bsn,bnc->bsc", attention, one_hot)
        pointer = mx.log(mx.maximum(probabilities, 1e-6))
        has_observation = mx.sum(observed, axis=-1) > 0.0
        return mx.where(has_observation[:, None, None], pointer, mx.zeros_like(pointer))

    def typed_state_transition(
        self,
        hidden: Any,
        *,
        state_slot_start: int,
    ) -> tuple[Any, Any]:
        """Commit one explicit categorical state for the next recurrent step.

        The straight-through categorical bottleneck makes the forward path
        discrete while retaining gradients through the decision probabilities.
        This prevents the auxiliary state head from merely describing a drifting
        residual: the state it predicts is the state the next pass receives.
        """

        logits = self.state_logits(hidden, state_slot_start=state_slot_start)
        return self.commit_state_logits(
            hidden,
            state_slot_start=state_slot_start,
            logits=logits,
        ), logits

    def _typed_transition_memory(
        self,
        action_probability_history: Sequence[Any],
        *,
        state_probabilities: Any | None = None,
        action_probabilities: Any | None = None,
    ) -> Any:
        """Compose the public action tape without collapsing typed fields.

        Memory is recurrent across instructions and remains separate across
        opcode, operands, and terminal state. The final projection is also
        state-slot specific, so a retained operand can affect one register
        without being forced through the same readout as every other field.
        """

        batch_size = int(action_probability_history[0].shape[0])
        memory = mx.zeros(
            (
                batch_size,
                self.config.action_slots,
                self.config.correction_rank,
            ),
            dtype=mx.float32,
        )
        for history_step, probabilities in enumerate(action_probability_history):
            committed = self.commit_action_probabilities(probabilities).astype(
                mx.float32
            )
            incoming = mx.einsum(
                "bah,ahr->bar",
                committed,
                self.transition_memory_input,
            )
            reset = mx.sigmoid(
                mx.einsum(
                    "bai,aio->bao",
                    incoming,
                    self.transition_memory_reset_input,
                )
                + mx.einsum(
                    "bai,aio->bao",
                    memory,
                    self.transition_memory_reset_recurrent,
                )
                + self.transition_memory_reset_bias[None, :, :]
            )
            update = mx.sigmoid(
                mx.einsum(
                    "bai,aio->bao",
                    incoming,
                    self.transition_memory_update_input,
                )
                + mx.einsum(
                    "bai,aio->bao",
                    memory,
                    self.transition_memory_update_recurrent,
                )
                + self.transition_memory_update_bias[None, :, :]
            )
            recurrent_candidate = mx.einsum(
                "bai,aio->bao",
                memory,
                self.transition_memory_candidate_recurrent,
            )
            cross_input = mx.einsum(
                "bsi,asio->bao",
                incoming,
                self.transition_memory_cross_input,
            )
            cross_recurrent = mx.einsum(
                "bsi,asio->bao",
                memory,
                self.transition_memory_cross_recurrent,
            )
            depth = self.depth_features(history_step) @ self.transition_memory_depth
            candidate = mx.tanh(
                mx.einsum(
                    "bai,aio->bao",
                    incoming,
                    self.transition_memory_candidate_input,
                )
                + cross_input
                + reset * (recurrent_candidate + cross_recurrent)
                + self.transition_memory_candidate_bias[None, :, :]
                + depth[None, None, :]
            )
            memory = (1.0 - update) * memory + update * candidate
        summary = mx.einsum(
            "bar,saro->bso",
            memory,
            self.transition_memory_output,
        )
        if state_probabilities is None and action_probabilities is None:
            return summary
        if state_probabilities is None or action_probabilities is None:
            raise ValueError("typed transition tape query is incomplete")
        return summary + self._typed_transition_tape_read(
            action_probability_history,
            state_probabilities=state_probabilities,
            action_probabilities=action_probabilities,
        )

    def _typed_transition_tape_read(
        self,
        action_probability_history: Sequence[Any],
        *,
        state_probabilities: Any,
        action_probabilities: Any,
    ) -> Any:
        """Query the intact causal public-action prefix without future access."""

        batch_size = int(state_probabilities.shape[0])
        if state_probabilities.shape != (
            batch_size,
            self.config.state_slots,
            self.config.state_cardinality,
        ) or action_probabilities.shape != (
            batch_size,
            self.config.action_slots,
            self.config.action_cardinality,
        ):
            raise ValueError("typed transition tape query shape differs")
        if not action_probability_history or any(
            probabilities.shape
            != (
                batch_size,
                self.config.action_slots,
                self.config.action_cardinality,
            )
            for probabilities in action_probability_history
        ):
            raise ValueError("typed transition tape prefix differs")

        committed_tape = mx.stack(
            tuple(
                probabilities.astype(mx.float32)
                for probabilities in action_probability_history
            ),
            axis=1,
        )
        keys = mx.einsum(
            "btac,acr->btar",
            committed_tape,
            self.transition_tape_key,
        )
        values = mx.einsum(
            "btac,acr->btar",
            committed_tape,
            self.transition_tape_value,
        )
        positions = mx.stack(
            tuple(self.depth_features(step) for step in range(len(action_probability_history))),
            axis=0,
        )
        keys = keys + (positions @ self.transition_tape_position_key)[None, :, None, :]
        values = values + (
            positions @ self.transition_tape_position_value
        )[None, :, None, :]
        query = mx.einsum(
            "bsc,scr->bsr",
            state_probabilities.astype(mx.float32),
            self.transition_tape_state_query,
        ) + mx.einsum(
            "bac,sacr->bsr",
            action_probabilities.astype(mx.float32),
            self.transition_tape_action_query,
        )
        attention_logits = mx.einsum("bsr,btar->bsta", query, keys) / math.sqrt(
            self.config.correction_rank
        )
        flattened = attention_logits.reshape(
            batch_size,
            self.config.state_slots,
            -1,
        )
        attention = mx.softmax(flattened, axis=-1).reshape(attention_logits.shape)
        read = mx.einsum("bsta,btar->bsr", attention, values)
        return mx.einsum(
            "bsi,sio->bso",
            read,
            self.transition_tape_output,
        )

    def _categorical_identity_features(self, probabilities: Any) -> Any:
        """Encode committed categories without learning away their identity."""

        if len(probabilities.shape) != 3:
            raise ValueError("typed categorical probabilities differ")
        cardinality = int(probabilities.shape[-1])
        rank = self.config.correction_rank
        if rank >= cardinality:
            padding = mx.zeros(
                (*probabilities.shape[:-1], rank - cardinality),
                dtype=probabilities.dtype,
            )
            return mx.concatenate((probabilities, padding), axis=-1).astype(mx.float32)
        categories = mx.arange(cardinality, dtype=mx.float32)[:, None]
        dimensions = mx.arange(rank, dtype=mx.float32)[None, :]
        frequencies = mx.floor(dimensions / 2.0) + 1.0
        angles = (
            2.0
            * math.pi
            * categories
            * frequencies
            / float(cardinality)
        )
        even = (mx.arange(rank) % 2) == 0
        basis = mx.where(even[None, :], mx.sin(angles), mx.cos(angles))
        return mx.einsum(
            "bnc,cr->bnr",
            probabilities.astype(mx.float32),
            basis,
        )

    def typed_transition_replay_logits(
        self,
        local_logits: Any,
        action_probability_history: Sequence[Any],
        *,
        action_probabilities: Any,
        replay_mode: str = "active",
        opcode_expert_routing: str = "opcode",
    ) -> tuple[Any, Any, Any]:
        """Recover from the public action prefix without reading recurrent state.

        The replay candidate is trained against verified next-state labels, but
        those labels are never runtime inputs.  Its query is fixed per state
        register and action field, so a wrong recurrent state cannot corrupt
        the evidence used to propose a repair.
        """

        if replay_mode not in TRANSITION_REPLAY_MODES:
            raise ValueError("typed transition replay mode differs")
        if opcode_expert_routing not in TRANSITION_OPCODE_EXPERT_ROUTING_MODES:
            raise ValueError("typed transition replay routing differs")
        batch_size = int(local_logits.shape[0])
        if local_logits.shape != (
            batch_size,
            self.config.state_slots,
            self.config.state_cardinality,
        ) or action_probabilities.shape != (
            batch_size,
            self.config.action_slots,
            self.config.action_cardinality,
        ):
            raise ValueError("typed transition replay categories differ")
        if not action_probability_history or any(
            item.shape
            != (
                batch_size,
                self.config.action_slots,
                self.config.action_cardinality,
            )
            for item in action_probability_history
        ):
            raise ValueError("typed transition replay prefix differs")
        if replay_mode in ("disabled", "lesion"):
            zero_gate = mx.zeros(
                (batch_size, self.config.state_slots), dtype=mx.float32
            )
            return local_logits, mx.stop_gradient(local_logits), zero_gate

        committed = mx.stack(
            tuple(item.astype(mx.float32) for item in action_probability_history),
            axis=1,
        )
        keys = mx.einsum("btac,acr->btar", committed, self.transition_replay_key)
        values = mx.einsum("btac,acr->btar", committed, self.transition_replay_value)
        positions = mx.stack(
            tuple(self.depth_features(step) for step in range(len(action_probability_history))),
            axis=0,
        )
        keys = keys + (
            positions @ self.transition_replay_position_key
        )[None, :, None, :]
        values = values + (
            positions @ self.transition_replay_position_value
        )[None, :, None, :]
        attention = mx.softmax(
            mx.einsum(
                "sar,btar->bsat",
                self.transition_replay_query,
                keys,
            )
            / math.sqrt(self.config.correction_rank),
            axis=-1,
        )
        read = mx.einsum("bsat,btar->bsar", attention, values)
        replay_features = nn.gelu(
            mx.einsum(
                "bsai,saio->bso",
                read,
                self.transition_replay_projection,
            )
        )
        opcode_probabilities = action_probabilities[:, 0, :].astype(mx.float32)
        if opcode_expert_routing == "uniform":
            routing = mx.full_like(
                opcode_probabilities,
                1.0 / float(self.config.action_cardinality),
            )
        elif opcode_expert_routing == "lesion":
            routing = mx.zeros_like(opcode_probabilities)
        else:
            routing = opcode_probabilities
        replay_candidate = mx.einsum(
            "bsr,src->bsc",
            replay_features,
            self.transition_replay_output,
        ) + mx.einsum(
            "bo,osrc,bsr->bsc",
            routing,
            self.transition_replay_opcode_output,
            replay_features,
        )
        local_probabilities = mx.softmax(local_logits.astype(mx.float32), axis=-1)
        replay_probabilities = mx.softmax(replay_candidate.astype(mx.float32), axis=-1)
        gate_features = mx.stack(
            (
                mx.max(local_probabilities, axis=-1),
                mx.max(replay_probabilities, axis=-1),
                1.0 - mx.sum(local_probabilities * replay_probabilities, axis=-1),
                mx.full(
                    (batch_size, self.config.state_slots),
                    min(1.0, len(action_probability_history) / 16.0),
                    dtype=mx.float32,
                ),
            ),
            axis=-1,
        )
        gate = mx.sigmoid(
            mx.einsum(
                "bsf,sf->bs",
                gate_features,
                self.transition_replay_gate_weight,
            )
            + self.transition_replay_gate_bias[None, :]
        )
        if replay_mode == "forced":
            return replay_candidate, replay_candidate, mx.ones_like(gate)
        return (
            local_logits + gate[..., None] * replay_candidate,
            replay_candidate,
            gate,
        )

    def typed_transition_processor_logits(
        self,
        state_probabilities: Any,
        action_probabilities: Any,
        history_memory: Any | None,
        *,
        opcode_expert_routing: str = "opcode",
    ) -> Any:
        """Compute one register transition from typed state, action and history."""

        if state_probabilities.shape[1:] != (
            self.config.state_slots,
            self.config.state_cardinality,
        ) or action_probabilities.shape[1:] != (
            self.config.action_slots,
            self.config.action_cardinality,
        ):
            raise ValueError("typed transition processor categories differ")
        if int(state_probabilities.shape[0]) != int(action_probabilities.shape[0]):
            raise ValueError("typed transition processor batch differs")
        if opcode_expert_routing not in TRANSITION_OPCODE_EXPERT_ROUTING_MODES:
            raise ValueError("typed transition processor routing differs")
        if history_memory is None:
            history_memory = mx.zeros(
                (
                    int(state_probabilities.shape[0]),
                    self.config.state_slots,
                    self.config.correction_rank,
                ),
                dtype=mx.float32,
            )
        if history_memory.shape != (
            int(state_probabilities.shape[0]),
            self.config.state_slots,
            self.config.correction_rank,
        ):
            raise ValueError("typed transition processor history differs")

        state_identity = self._categorical_identity_features(state_probabilities)
        action_identity = self._categorical_identity_features(action_probabilities)
        local_state = mx.einsum(
            "bsi,sio->bso",
            state_identity,
            self.transition_processor_state_projection,
        )
        cross_state = mx.einsum(
            "bni,snio->bso",
            state_identity,
            self.transition_processor_state_cross_projection,
        )
        state = local_state + cross_state
        action_left = mx.einsum(
            "bai,saio->bso",
            action_identity,
            self.transition_processor_action_left,
        )
        action_right = mx.einsum(
            "bai,saio->bso",
            action_identity,
            self.transition_processor_action_right,
        )
        history = mx.einsum(
            "bsi,sio->bso",
            history_memory.astype(mx.float32),
            self.transition_processor_history_projection,
        )
        interactions = mx.concatenate(
            (
                state,
                action_left,
                action_right,
                history,
                state * action_left,
                state * action_right,
                action_left * action_right,
                state * history,
                action_left * action_right * history,
            ),
            axis=-1,
        )
        expanded = nn.gelu(
            mx.einsum(
                "bsi,sio->bso",
                interactions,
                self.transition_processor_interaction_up,
            )
            + self.transition_processor_interaction_bias[None, :, :]
        )
        processed = nn.gelu(
            mx.einsum(
                "bsi,sio->bso",
                expanded,
                self.transition_processor_interaction_down,
            )
        )
        opcode_probabilities = action_probabilities[:, 0, :].astype(mx.float32)
        if opcode_expert_routing == "uniform":
            routing = mx.full_like(
                opcode_probabilities,
                1.0 / float(self.config.action_cardinality),
            )
        elif opcode_expert_routing == "lesion":
            routing = mx.zeros_like(opcode_probabilities)
        else:
            routing = opcode_probabilities
        interaction_expert = nn.gelu(
            mx.einsum(
                "osie,bsi->bose",
                self.transition_processor_opcode_interaction_up,
                interactions,
            )
        )
        processed = processed + mx.einsum(
            "bo,oser,bose->bsr",
            routing,
            self.transition_processor_opcode_interaction_down,
            interaction_expert,
        )
        hidden_residual = mx.einsum(
            "bo,osri,bsi->bsr",
            routing,
            self.transition_processor_opcode_hidden,
            processed,
        )
        processed = processed + hidden_residual
        shared_logits = mx.einsum(
            "bsr,src->bsc",
            processed,
            self.transition_processor_output,
        )
        expert_logits = mx.einsum(
            "bo,osrc,bsr->bsc",
            routing,
            self.transition_processor_opcode_output,
            processed,
        )
        return shared_logits + expert_logits

    def resolve_transition_processor_logits(
        self,
        legacy_logits: Any | None,
        state_probabilities: Any,
        action_probabilities: Any,
        history_memory: Any | None,
        *,
        transition_processor_mode: str,
        opcode_expert_routing: str = "opcode",
        transition_copy_prior_logit_bias: float = TRANSITION_COPY_PRIOR_LOGIT_BIAS,
    ) -> Any:
        """Apply one declared processor policy to the deployed transition.

        Direct processor training has no transformer or legacy transition graph.
        Its certified runtime counterpart must therefore use the processor as
        the authoritative transition, rather than silently adding a separately
        learned legacy head.  ``residual`` remains available for historical
        checkpoints, while evidence campaigns bind ``authoritative`` into their
        identity.
        """

        if transition_processor_mode not in TRANSITION_PROCESSOR_MODES:
            raise ValueError("typed transition processor mode differs")
        if (
            isinstance(transition_copy_prior_logit_bias, bool)
            or not isinstance(transition_copy_prior_logit_bias, (int, float))
            or not math.isfinite(float(transition_copy_prior_logit_bias))
            or not 0.0 <= float(transition_copy_prior_logit_bias) <= 8.0
        ):
            raise ValueError("transition copy prior logit bias differs")
        processor_logits = self.typed_transition_processor_logits(
            state_probabilities,
            action_probabilities,
            history_memory,
            opcode_expert_routing=opcode_expert_routing,
        )
        if transition_processor_mode == "authoritative":
            resolved = processor_logits
        elif transition_processor_mode in {"copy_write", "masked_copy_write"}:
            # A register machine is sparse by construction: most operations
            # retain most of the committed state. Predicting every register
            # from scratch spends gradient on relearning identity and lets an
            # uncertain head erase valid state. The finite prior is strong
            # enough to make a zero-attached processor an exact copy, while a
            # learned candidate can still override any register. It carries no
            # target, private trace, opcode semantics, or future information.
            writable_logits = processor_logits + (
                float(transition_copy_prior_logit_bias)
                * state_probabilities.astype(mx.float32)
            )
            if transition_processor_mode == "copy_write":
                resolved = writable_logits
            else:
                # Sparse register updates are part of the public instruction
                # contract. A learned operation may choose the new value of a
                # writable register, but it may not corrupt state that the
                # instruction does not address. This mask depends only on the
                # committed state and canonical public action, never on a
                # private trace or future target.
                write_authority = self.transition_write_authority_mask(
                    state_probabilities,
                    action_probabilities,
                )
                copied_logits = mx.log(
                    mx.maximum(state_probabilities.astype(mx.float32), 1e-6)
                )
                resolved = mx.where(
                    write_authority[..., None],
                    writable_logits,
                    copied_logits,
                )
                # Program position and termination are control-plane facts,
                # not task semantics.  For a recognized public instruction the
                # next PC and terminal latch are fully determined by the
                # committed state and action.  Asking learned tissue to infer
                # them introduces geometric trajectory failures while adding
                # no reasoning capacity.  Value registers remain learned.
                state_values = mx.argmax(state_probabilities, axis=-1).astype(
                    mx.int32
                )
                action_values = mx.argmax(action_probabilities, axis=-1).astype(
                    mx.int32
                )
                opcode = action_values[:, 0]
                # Same membership rule as microcode_transition_logits: an
                # opcode is recognized because a branch executes it, not
                # because its number is inside a range.
                recognized = mx.take(
                    _microcode_implemented_opcode_mask(),
                    mx.minimum(opcode, ACTION_NULL),
                ).astype(mx.bool_)
                next_pc = mx.minimum(
                    state_values[:, 0] + 1,
                    self.config.state_cardinality - 1,
                )
                control_values = mx.concatenate(
                    (
                        next_pc[:, None],
                        state_values[:, 1:-1],
                        action_values[:, 7, None],
                    ),
                    axis=1,
                )
                categories = mx.arange(self.config.state_cardinality)[None, None, :]
                control_logits = mx.log(
                    mx.maximum(
                        (control_values[..., None] == categories).astype(mx.float32),
                        1e-6,
                    )
                )
                slots = mx.arange(self.config.state_slots)[None, :]
                control_authority = recognized[:, None] & (
                    (slots == 0) | (slots == self.config.state_slots - 1)
                )
                resolved = mx.where(
                    control_authority[..., None],
                    control_logits,
                    resolved,
                )
        else:
            if legacy_logits is None:
                raise ValueError("residual transition processor requires legacy logits")
            if legacy_logits.shape != processor_logits.shape:
                raise ValueError("legacy and typed transition logits differ")
            resolved = legacy_logits + processor_logits
        # STATE_INVALID is an explicit absorbing fault state, not another value
        # the learned processor may reinterpret. This structural latch carries
        # no task answer and prevents one corrupt register from being laundered
        # into apparently valid recurrent execution on the next step.
        state_values = mx.argmax(state_probabilities, axis=-1)
        invalid = mx.any(state_values == STATE_INVALID, axis=-1)
        categories = mx.arange(self.config.state_cardinality)[None, None, :]
        invalid_logits = mx.log(
            mx.maximum(
                (categories == STATE_INVALID).astype(mx.float32),
                1e-6,
            )
        )
        return mx.where(invalid[:, None, None], invalid_logits, resolved)

    def transition_write_authority_mask(
        self,
        state_probabilities: Any,
        action_probabilities: Any,
    ) -> Any:
        """Return public opcode-qualified register write authority.

        The result is conservative: a register is writable when the canonical
        instruction can change it for the current public operands and committed
        state. Non-writable registers are exact copies in ``masked_copy_write``.
        """

        if state_probabilities.shape[1:] != (
            self.config.state_slots,
            self.config.state_cardinality,
        ) or action_probabilities.shape[1:] != (
            self.config.action_slots,
            self.config.action_cardinality,
        ):
            raise ValueError("transition write authority differs from its schema")
        state = mx.argmax(state_probabilities, axis=-1).astype(mx.int32)
        action = mx.argmax(action_probabilities, axis=-1).astype(mx.int32)
        opcode = action[:, 0]
        arg0, arg1, arg2, arg3, arg4, _arg5 = (
            action[:, index] for index in range(1, 7)
        )
        slots = mx.arange(self.config.state_slots)[None, :]
        # PC and done are resolved by the structural control plane after learned
        # value writes. They are deliberately absent from learned authority.
        writable = mx.zeros(
            (state_probabilities.shape[0], self.config.state_slots),
            dtype=mx.bool_,
        )

        scalar = (opcode >= OP_COPY_VALUE) & (opcode <= OP_BOOL_XOR)
        writable = writable | (scalar[:, None] & (slots == 1))

        register = (opcode == OP_REGISTER_AFFINE) & (arg0 < 3)
        writable = writable | (
            register[:, None] & (slots == (arg0[:, None] + 1))
        )

        first_three = (
            (opcode == OP_FRONTIER_TRAVERSE)
            | (opcode == OP_FRONTIER_ENUMERATE)
            | (opcode == OP_FRONTIER_SCHEDULE)
        )
        writable = writable | (
            first_three[:, None] & (slots >= 1) & (slots <= 3)
        )

        infer = opcode == OP_FRONTIER_INFER
        writable = writable | (infer[:, None] & (slots == 1))
        writable = writable | (
            (infer & (arg0 == 3))[:, None] & (slots >= 2) & (slots <= 3)
        )

        simulate = opcode == OP_FRONTIER_SIMULATE
        writable = writable | (simulate[:, None] & (slots == 1))
        same_case = state[:, 1] == arg0
        if self.config.state_slots >= len(SEMANTIC_STATE_SLOT_NAMES):
            reset_balances = simulate & ~same_case
            writable = writable | (
                reset_balances[:, None] & (slots >= 2) & (slots <= 9)
            )
            selected_balance = simulate & same_case & (arg1 < 4)
            selected_low_slot = 2 + (2 * arg1)
            writable = writable | (
                selected_balance[:, None]
                & (
                    (slots == selected_low_slot[:, None])
                    | (slots == (selected_low_slot + 1)[:, None])
                )
            )
        else:
            writable = writable | (
                simulate[:, None] & (slots >= 2) & (slots <= 3)
            )

        calibrate = opcode == OP_FRONTIER_CALIBRATE
        pc = state[:, 0]
        if self.config.state_slots >= len(SEMANTIC_STATE_SLOT_NAMES):
            writable = writable | (
                (calibrate & (pc == 0))[:, None] & (slots >= 1) & (slots <= 4)
            )
            writable = writable | (
                (calibrate & (pc == 1))[:, None] & (slots == 5)
            )
            writable = writable | (
                (calibrate & (pc >= 2))[:, None] & (slots == 6)
            )
        else:
            writable = writable | (
                (calibrate & (pc == 0))[:, None] & (slots >= 1) & (slots <= 2)
            )
            writable = writable | (
                (calibrate & (pc >= 1))[:, None] & (slots >= 1) & (slots <= 3)
            )

        audit = opcode == OP_FRONTIER_AUDIT
        encoded_score = state[:, 2] + PROCESS_RADIX * state[:, 3]
        current_score = mx.where(
            (encoded_score % 2) == 0,
            encoded_score // 2,
            -((encoded_score + 1) // 2),
        )
        candidate_score = arg1 * arg2 - arg3
        if self.config.state_slots >= len(SEMANTIC_STATE_SLOT_NAMES):
            candidate_wins = (
                (state[:, 5] == 0)
                | (candidate_score > current_score)
                | ((candidate_score == current_score) & (arg4 < state[:, 4]))
            )
            audit_last_slot = 5
        else:
            candidate_wins = (pc == 0) | (candidate_score > current_score)
            audit_last_slot = 3
        writable = writable | (
            (audit & candidate_wins)[:, None]
            & (slots >= 1)
            & (slots <= audit_last_slot)
        )

        pair_destination = (
            (opcode == OP_PAIR_SET)
            | (opcode == OP_PAIR_ADD)
            | (opcode == OP_PAIR_MUL_IMMEDIATE)
            | (opcode == OP_PAIR_PRODUCT)
            | (opcode == OP_PAIR_SUB_IMMEDIATE)
            | (opcode == OP_PAIR_SIGNED_SUB_IMMEDIATE)
            | (opcode == OP_PAIR_DIV)
            | (opcode == OP_PAIR_COPY)
            | (opcode == OP_SIGNED_PAIR_ADD_IMMEDIATE)
        ) & (arg0 < self.config.state_slots - 2)
        pair_low_slot = arg0 + 1
        writable = writable | (
            pair_destination[:, None]
            & (
                (slots == pair_low_slot[:, None])
                | (slots == (pair_low_slot + 1)[:, None])
            )
        )

        euclid = (
            (opcode == OP_PAIR_EUCLID_STEP)
            & (arg0 < self.config.state_slots - 2)
            & (arg1 < self.config.state_slots - 2)
        )
        euclid_left = arg0 + 1
        euclid_right = arg1 + 1
        writable = writable | (
            euclid[:, None]
            & (
                (slots == euclid_left[:, None])
                | (slots == (euclid_left + 1)[:, None])
                | (slots == euclid_right[:, None])
                | (slots == (euclid_right + 1)[:, None])
            )
        )

        scalar_destination = (
            (opcode == OP_RATIO_CHOICE)
            | (opcode == OP_RATIO_BAND)
            | (opcode == OP_SIGNED_RANKED_GREATER)
            | (opcode == OP_SET_SCALAR)
        ) & (arg0 < self.config.state_slots - 2)
        writable = writable | (
            scalar_destination[:, None] & (slots == (arg0 + 1)[:, None])
        )

        ranked_commit = opcode == OP_RANKED_COMMIT
        valid_flag = arg0 < self.config.state_slots - 2
        flag_value = mx.take_along_axis(
            state,
            mx.minimum(arg0 + 1, self.config.state_slots - 1)[:, None],
            axis=1,
        )[:, 0]
        commit_selected = ranked_commit & valid_flag & (flag_value != 0)
        writable = writable | (
            commit_selected[:, None] & (slots >= 1) & (slots <= 4)
        )
        writable = writable | (ranked_commit[:, None] & (slots == 5))
        return writable

    def state_transition_logits(
        self,
        problem_evidence: Any,
        hidden: Any,
        *,
        state_slot_start: int,
        step: int,
        action_state: Any | None = None,
        state_probabilities: Any | None = None,
        action_probabilities: Any | None = None,
        action_probability_history: Sequence[Any] | None = None,
        microcode_lesion: bool = False,
        transition_processor_lesion: bool = False,
        transition_processor_mode: str = "residual",
        transition_copy_prior_logit_bias: float = TRANSITION_COPY_PRIOR_LOGIT_BIAS,
        opcode_expert_routing: str = "opcode",
        transition_replay_mode: str = "disabled",
    ) -> Any:
        """Predict one shared typed transition from state plus immutable evidence."""

        if (
            type(microcode_lesion) is not bool
            or type(transition_processor_lesion) is not bool
        ):
            raise TypeError("transition lesion flags must be bools")
        if transition_processor_mode not in TRANSITION_PROCESSOR_MODES:
            raise ValueError("state transition processor mode differs")
        if opcode_expert_routing not in TRANSITION_OPCODE_EXPERT_ROUTING_MODES:
            raise ValueError("state transition opcode routing differs")
        if transition_replay_mode not in TRANSITION_REPLAY_MODES:
            raise ValueError("state transition replay mode differs")

        if (
            len(problem_evidence.shape) != 3
            or int(problem_evidence.shape[0]) != int(hidden.shape[0])
            or int(problem_evidence.shape[-1]) != self.config.hidden_size
            or int(problem_evidence.shape[1]) < 1
        ):
            raise ValueError("state transition problem evidence shape differs")
        stop = state_slot_start + self.config.state_slots
        if not 0 <= state_slot_start < stop <= int(hidden.shape[1]):
            raise ValueError("state transition slots are outside the recurrent state")
        state = hidden[:, state_slot_start:stop, :].astype(mx.float32)
        evidence = problem_evidence.astype(mx.float32)
        query = mx.einsum("bsh,shr->bsr", state, self.state_transition_query)
        keys = evidence @ self.state_transition_key
        values = evidence @ self.state_transition_value
        attention = mx.softmax(
            mx.einsum("bsr,bnr->bsn", query, keys)
            / math.sqrt(self.config.correction_rank),
            axis=-1,
        )
        context = mx.einsum("bsn,bnr->bsr", attention, values)
        self_state = mx.einsum(
            "bsh,shr->bsr",
            state,
            self.state_transition_self,
        )
        depth = self.depth_features(step) @ self.state_transition_depth
        features = mx.tanh(context + self_state + depth[None, None, :])
        typed_memory = None
        if action_probability_history is not None:
            if not action_probability_history or any(
                tuple(item.shape)
                != (
                    int(hidden.shape[0]),
                    self.config.action_slots,
                    self.config.action_cardinality,
                )
                for item in action_probability_history
            ):
                raise ValueError("state transition action history differs")
            # The current instruction has its own slot-aware projection below.
            # Fold only prior instructions into an ordered recurrent summary so
            # stateful programs do not have to rediscover their execution tape
            # from prompt text at every step. Reusing the existing transition
            # projections preserves checkpoint topology while making order
            # causal: permuting two prior instructions changes this state.
            history = mx.zeros(
                (int(hidden.shape[0]), self.config.correction_rank),
                dtype=mx.float32,
            )
            for history_step, probabilities in enumerate(
                action_probability_history[:-1]
            ):
                historical_state = self.commit_action_probabilities(probabilities)
                historical_action = mx.mean(
                    mx.einsum(
                        "bah,ahr->bar",
                        historical_state.astype(mx.float32),
                        self.state_action_projection,
                    ),
                    axis=1,
                )
                historical_depth = (
                    self.depth_features(history_step) @ self.state_transition_depth
                )
                history = mx.tanh(
                    0.5 * history + historical_action + historical_depth[None, :]
                )
            typed_memory = self._typed_transition_memory(
                action_probability_history,
                state_probabilities=state_probabilities,
                action_probabilities=action_probabilities,
            )
            features = mx.tanh(
                features + history[:, None, :] + typed_memory
            )
        if action_state is not None:
            if action_state.shape != (
                int(hidden.shape[0]),
                self.config.action_slots,
                self.config.hidden_size,
            ):
                raise ValueError("typed action state shape differs")
            action = mx.mean(
                mx.einsum(
                    "bah,ahr->bar",
                    action_state.astype(mx.float32),
                    self.state_action_projection,
                ),
                axis=1,
            )
            features = mx.tanh(features + action[:, None, :])
        learned_logits = mx.einsum(
            "bsr,src->bsc",
            features,
            self.state_transition_output,
        ) + self.state_transition_bias
        if (
            not transition_processor_lesion
            and state_probabilities is not None
            and action_probabilities is not None
        ):
            learned_logits = self.resolve_transition_processor_logits(
                learned_logits,
                state_probabilities,
                action_probabilities,
                typed_memory,
                transition_processor_mode=transition_processor_mode,
                opcode_expert_routing=opcode_expert_routing,
                transition_copy_prior_logit_bias=(
                    transition_copy_prior_logit_bias
                ),
            )
            if transition_replay_mode not in ("disabled", "lesion"):
                if action_probability_history is None:
                    raise ValueError("state transition replay has no public prefix")
                learned_logits, _replay_candidate, _replay_gate = (
                    self.typed_transition_replay_logits(
                        learned_logits,
                        action_probability_history,
                        action_probabilities=action_probabilities,
                        replay_mode=transition_replay_mode,
                        opcode_expert_routing=opcode_expert_routing,
                    )
                )
        if state_probabilities is None or action_probabilities is None:
            return learned_logits
        state_values = mx.argmax(state_probabilities, axis=-1).astype(mx.int32)
        terminal = state_values[:, -1] == 1
        state_categories = mx.arange(self.config.state_cardinality)[None, None, :]
        terminal_exact = (state_values[..., None] == state_categories).astype(mx.float32)
        terminal_logits = mx.log(mx.maximum(terminal_exact, 1e-6))
        if microcode_lesion:
            return mx.where(terminal[:, None, None], terminal_logits, learned_logits)
        microcode_logits, recognized = self.microcode_transition_logits(
            state_probabilities,
            action_probabilities,
            action_probability_history=action_probability_history,
        )
        return mx.where(
            terminal[:, None, None],
            terminal_logits,
            mx.where(recognized[:, None, None], microcode_logits, learned_logits),
        )

    def microcode_transition_logits(
        self,
        state_probabilities: Any,
        action_probabilities: Any,
        *,
        action_probability_history: Sequence[Any] | None = None,
    ) -> tuple[Any, Any]:
        """Execute recognized canonical instructions on categorical registers."""

        if state_probabilities.shape[1:] != (
            self.config.state_slots,
            self.config.state_cardinality,
        ) or action_probabilities.shape[1:] != (
            self.config.action_slots,
            self.config.action_cardinality,
        ):
            raise ValueError("microcode categorical state differs from its schema")
        state = mx.argmax(state_probabilities, axis=-1).astype(mx.int32)
        action = mx.argmax(action_probabilities, axis=-1).astype(mx.int32)
        if action_probability_history is not None and (
            not action_probability_history
            or any(
                tuple(item.shape) != tuple(action_probabilities.shape)
                for item in action_probability_history
            )
        ):
            raise ValueError("microcode action history differs from its schema")
        opcode = action[:, 0]
        arguments = action[:, 1:7]
        terminal = action[:, 7]
        # RECOGNITION IS MEMBERSHIP, NOT RANGE. This was
        # `opcode <= MAX_RECURRENT_OPCODE`, so every opcode added to the
        # schema was declared recognized here the moment the constant moved —
        # implemented or not. OP_CAUSAL_CHAIN arrived that way: the microcode
        # reported success, executed no branch, returned the registers
        # unchanged, and three process-supervision tests failed with a state
        # mismatch instead of "this opcode is not implemented".
        recognized = mx.take(
            _microcode_implemented_opcode_mask(), mx.minimum(opcode, ACTION_NULL)
        ).astype(mx.bool_)

        pc = mx.minimum(state[:, 0] + 1, self.config.state_cardinality - 1)
        values = [state[:, index] for index in range(1, self.config.state_slots - 1)]
        value0, value1, value2 = values[:3]
        arg0, arg1, arg2, arg3, arg4, arg5 = (
            arguments[:, index] for index in range(6)
        )
        modulus = mx.maximum(arg1, 1)
        value0 = mx.where(opcode == OP_COPY_VALUE, arg0, value0)
        value0 = mx.where(opcode == OP_ADD_MOD, (value0 + arg0) % modulus, value0)
        value0 = mx.where(opcode == OP_MUL_MOD, (value0 * arg0) % modulus, value0)
        value0 = mx.where(opcode == OP_SUB_MOD, (value0 - arg0) % modulus, value0)
        value0 = mx.where(opcode == OP_BOOL_NOT, 1 - mx.minimum(value0, 1), value0)
        value0 = mx.where(
            opcode == OP_BOOL_AND,
            mx.minimum(value0, 1) & mx.minimum(arg0, 1),
            value0,
        )
        value0 = mx.where(
            opcode == OP_BOOL_OR,
            mx.minimum(value0, 1) | mx.minimum(arg0, 1),
            value0,
        )
        value0 = mx.where(
            opcode == OP_BOOL_XOR,
            mx.minimum(value0, 1) ^ mx.minimum(arg0, 1),
            value0,
        )

        registers = mx.stack((state[:, 1], state[:, 2], state[:, 3]), axis=1)
        left = mx.take_along_axis(
            registers, mx.minimum(arg1, 2)[:, None], axis=1
        )[:, 0]
        right = mx.take_along_axis(
            registers, mx.minimum(arg2, 2)[:, None], axis=1
        )[:, 0]
        register_modulus = mx.maximum(arg5, 1)
        register_result = (left + arg3 * right + arg4) % register_modulus
        is_register = opcode == OP_REGISTER_AFFINE
        value0 = mx.where(is_register & (arg0 == 0), register_result, value0)
        value1 = mx.where(is_register & (arg0 == 1), register_result, value1)
        value2 = mx.where(is_register & (arg0 == 2), register_result, value2)
        values[:3] = (value0, value1, value2)

        # Broad process instructions use the same finite categorical machine as
        # the original executable curriculum.  Their private training traces
        # teach action selection; once selected, these transitions are exact and
        # require neither a runtime teacher nor a task-specific answer producer.
        is_traverse = opcode == OP_FRONTIER_TRAVERSE
        selected_value = arg1 + PROCESS_RADIX * arg2
        next_checksum = (
            value2 + (state[:, 0] + 1) * selected_value
        ) % PROCESS_RADIX
        values[0] = mx.where(is_traverse, arg0, values[0])
        values[1] = mx.where(is_traverse, mx.maximum(values[1] - 1, 0), values[1])
        values[2] = mx.where(is_traverse, next_checksum, values[2])

        is_enumerate = opcode == OP_FRONTIER_ENUMERATE
        if action_probability_history is not None:
            history = mx.stack(
                [
                    mx.argmax(mx.stop_gradient(item), axis=-1).astype(mx.int32)
                    for item in action_probability_history
                ],
                axis=1,
            )
            configuration = history[:, 0, :]
            choose = configuration[:, 1]
            gap = configuration[:, 2]

            def decode_signed(encoded: Any) -> Any:
                return mx.where(
                    (encoded % 2) == 0,
                    encoded // 2,
                    -((encoded + 1) // 2),
                )

            low = decode_signed(
                configuration[:, 3] + PROCESS_RADIX * configuration[:, 4]
            )
            high = decode_signed(
                configuration[:, 5] + PROCESS_RADIX * configuration[:, 6]
            )
            events = history[:, 1:, :]
            event_indices = events[:, :, 1]
            event_values = decode_signed(
                events[:, :, 2] + PROCESS_RADIX * events[:, :, 3]
            )

            def evaluate_combinations(width: int) -> tuple[Any, Any]:
                combinations = tuple(
                    itertools.combinations(range(int(events.shape[1])), width)
                )
                if not combinations:
                    zeros = mx.zeros_like(choose)
                    return zeros, zeros
                combination_indices = mx.array(combinations, dtype=mx.int32)
                selected = mx.take(event_values, combination_indices, axis=1)
                separated = mx.all(
                    (selected[:, :, 1:] - selected[:, :, :-1])
                    >= gap[:, None, None],
                    axis=-1,
                )
                sums = mx.sum(selected, axis=-1)
                valid = (
                    separated
                    & (sums >= low[:, None])
                    & (sums <= high[:, None])
                )
                count = mx.sum(valid, axis=1).astype(mx.int32)
                heads = mx.take(
                    event_indices,
                    combination_indices[:, 0],
                    axis=1,
                ) + 1
                sentinel = mx.full_like(heads, self.config.action_cardinality)
                witness = mx.min(mx.where(valid, heads, sentinel), axis=1)
                witness = mx.where(
                    witness == self.config.action_cardinality,
                    0,
                    witness,
                )
                return count, witness

            count_three, witness_three = evaluate_combinations(3)
            count_four, witness_four = evaluate_combinations(4)
            next_count = mx.where(choose == 3, count_three, count_four)
            next_witness = mx.where(choose == 3, witness_three, witness_four)
            values[0] = mx.where(is_enumerate, next_count % PROCESS_RADIX, values[0])
            values[1] = mx.where(is_enumerate, next_count // PROCESS_RADIX, values[1])
            values[2] = mx.where(is_enumerate, next_witness, values[2])

        is_simulate = opcode == OP_FRONTIER_SIMULATE
        if len(values) >= 9:
            same_case = values[0] == arg0
            for name_index in range(4):
                lo_index = 1 + (2 * name_index)
                hi_index = lo_index + 1
                encoded = mx.where(
                    same_case,
                    values[lo_index] + PROCESS_RADIX * values[hi_index],
                    0,
                )
                balance = mx.where(
                    (encoded % 2) == 0,
                    encoded // 2,
                    -((encoded + 1) // 2),
                )
                next_balance = balance + (arg2 - 3)
                next_encoded = mx.where(
                    next_balance >= 0,
                    next_balance * 2,
                    (-next_balance * 2) - 1,
                )
                selected_name = is_simulate & (arg1 == name_index)
                values[lo_index] = mx.where(
                    selected_name,
                    next_encoded % PROCESS_RADIX,
                    mx.where(is_simulate & ~same_case, 0, values[lo_index]),
                )
                values[hi_index] = mx.where(
                    selected_name,
                    next_encoded // PROCESS_RADIX,
                    mx.where(is_simulate & ~same_case, 0, values[hi_index]),
                )
            values[0] = mx.where(is_simulate, arg0, values[0])
        elif action_probability_history is not None:
            history = mx.stack(
                [
                    mx.argmax(mx.stop_gradient(item), axis=-1).astype(mx.int32)
                    for item in action_probability_history
                ],
                axis=1,
            )
            same_case_event = (
                (history[:, :, 0] == OP_FRONTIER_SIMULATE)
                & (history[:, :, 1] == arg0[:, None])
            )
            names = mx.arange(self.config.action_cardinality)[None, None, :]
            matching_name = history[:, :, 2, None] == names
            deltas = history[:, :, 3] - 3
            balances = mx.sum(
                mx.where(
                    same_case_event[:, :, None] & matching_name,
                    deltas[:, :, None],
                    0,
                ),
                axis=1,
            )
            values[0] = mx.where(is_simulate, arg0, values[0])
            values[1] = mx.where(
                is_simulate,
                mx.sum(balances != 0, axis=1).astype(mx.int32),
                values[1],
            )
            values[2] = mx.where(
                is_simulate,
                mx.sum(mx.abs(balances), axis=1).astype(mx.int32),
                values[2],
            )

        is_infer = opcode == OP_FRONTIER_INFER
        inferred_role = mx.where(
            arg0 == 0,
            arg1 + 1,
            mx.where(
                arg0 == 1,
                arg1 * 3 + arg2 + 1,
                value0,
            ),
        )
        downstream_base = arg1 + PROCESS_RADIX * arg2
        inferred_prediction = downstream_base + arg3 * arg4 * arg5
        inferred_prediction_encoded = inferred_prediction * 2
        values[0] = mx.where(is_infer, inferred_role, values[0])
        values[1] = mx.where(
            is_infer & (arg0 == 3),
            inferred_prediction_encoded % PROCESS_RADIX,
            values[1],
        )
        values[2] = mx.where(
            is_infer & (arg0 == 3),
            inferred_prediction_encoded // PROCESS_RADIX,
            values[2],
        )

        is_schedule = opcode == OP_FRONTIER_SCHEDULE
        reward = state[:, 2] + PROCESS_RADIX * state[:, 3]
        next_reward = mx.minimum(reward + arg3, MAX_PROCESS_INTEGER)
        values[0] = mx.where(
            is_schedule,
            mx.minimum(values[0] + arg1, self.config.state_cardinality - 1),
            values[0],
        )
        values[1] = mx.where(is_schedule, next_reward % PROCESS_RADIX, values[1])
        values[2] = mx.where(is_schedule, next_reward // PROCESS_RADIX, values[2])

        is_calibrate = opcode == OP_FRONTIER_CALIBRATE
        if action_probability_history is not None:
            calibration_input = mx.argmax(
                mx.stop_gradient(action_probability_history[0]), axis=-1
            ).astype(mx.int32)
            prior_num = calibration_input[:, 1]
            prior_den = calibration_input[:, 2]
            likelihood_h_num = calibration_input[:, 3]
            likelihood_h_den = calibration_input[:, 4]
            likelihood_not_h_num = calibration_input[:, 5]
            likelihood_not_h_den = calibration_input[:, 6]
            posterior_num = likelihood_h_num * prior_num * likelihood_not_h_den
            posterior_den = posterior_num + (
                likelihood_not_h_num
                * (prior_den - prior_num)
                * likelihood_h_den
            )
            divisor_left = posterior_num
            divisor_right = posterior_den
            for _ in range(12):
                remainder = divisor_left % mx.maximum(divisor_right, 1)
                next_left = mx.where(divisor_right == 0, divisor_left, divisor_right)
                next_right = mx.where(divisor_right == 0, 0, remainder)
                divisor_left, divisor_right = next_left, next_right
            divisor = mx.maximum(divisor_left, 1)
            reduced_num = posterior_num // divisor
            reduced_den = posterior_den // divisor
            choose_h = (2 * reduced_num) >= reduced_den
            percentage = (100 * reduced_num) // mx.maximum(reduced_den, 1)
            confidence_band = mx.where(
                percentage < 50,
                0,
                mx.where(percentage < 70, 1, mx.where(percentage < 90, 2, 3)),
            )
            calibration_step = state[:, 0]
            values[0] = mx.where(
                is_calibrate & (calibration_step == 0),
                reduced_num % PROCESS_RADIX,
                values[0],
            )
            values[1] = mx.where(
                is_calibrate & (calibration_step == 0),
                reduced_num // PROCESS_RADIX,
                values[1],
            )
            if len(values) >= 6:
                values[2] = mx.where(
                    is_calibrate & (calibration_step == 0),
                    reduced_den % PROCESS_RADIX,
                    values[2],
                )
                values[3] = mx.where(
                    is_calibrate & (calibration_step == 0),
                    reduced_den // PROCESS_RADIX,
                    values[3],
                )
                values[4] = mx.where(
                    is_calibrate & (calibration_step == 1),
                    choose_h.astype(mx.int32) + 1,
                    values[4],
                )
                values[5] = mx.where(
                    is_calibrate & (calibration_step >= 2),
                    confidence_band.astype(mx.int32) + 1,
                    values[5],
                )
            else:
                values[0] = mx.where(
                    is_calibrate & (calibration_step >= 1),
                    reduced_den % PROCESS_RADIX,
                    values[0],
                )
                values[1] = mx.where(
                    is_calibrate & (calibration_step >= 1),
                    reduced_den // PROCESS_RADIX,
                    values[1],
                )
                values[2] = mx.where(
                    is_calibrate & (calibration_step == 1),
                    choose_h.astype(mx.int32) + 1,
                    values[2],
                )
                values[2] = mx.where(
                    is_calibrate & (calibration_step >= 2),
                    confidence_band.astype(mx.int32) + 1,
                    values[2],
                )

        is_audit = opcode == OP_FRONTIER_AUDIT
        encoded_score = values[1] + PROCESS_RADIX * values[2]
        current_score = mx.where(
            (encoded_score % 2) == 0,
            encoded_score // 2,
            -((encoded_score + 1) // 2),
        )
        candidate_score = arg1 * arg2 - arg3
        candidate_encoded = mx.where(
            candidate_score >= 0,
            candidate_score * 2,
            (-candidate_score * 2) - 1,
        )
        candidate_wins = (
            (values[4] == 0)
            | (candidate_score > current_score)
            | ((candidate_score == current_score) & (arg4 < values[3]))
            if len(values) >= 5
            else (state[:, 0] == 0) | (candidate_score > current_score)
        )
        values[0] = mx.where(
            is_audit & candidate_wins,
            arg0,
            values[0],
        )
        values[1] = mx.where(
            is_audit & candidate_wins,
            candidate_encoded % PROCESS_RADIX,
            values[1],
        )
        values[2] = mx.where(
            is_audit & candidate_wins,
            candidate_encoded // PROCESS_RADIX,
            values[2],
        )
        if len(values) >= 5:
            values[3] = mx.where(is_audit & candidate_wins, arg4, values[3])
            values[4] = mx.where(is_audit, 1, values[4])

        semantic_opcode = (opcode >= OP_PAIR_SET) & (
            opcode <= MAX_RECURRENT_OPCODE
        )
        semantic_invalid = semantic_opcode & (len(values) < 8)

        def valid_value_address(index: Any) -> Any:
            return index < len(values)

        def valid_pair_address(index: Any) -> Any:
            return index < len(values) - 1

        pair_set = opcode == OP_PAIR_SET
        pair_add = opcode == OP_PAIR_ADD
        pair_multiply = opcode == OP_PAIR_MUL_IMMEDIATE
        pair_subtract = opcode == OP_PAIR_SUB_IMMEDIATE
        pair_signed_subtract = opcode == OP_PAIR_SIGNED_SUB_IMMEDIATE
        pair_divide = opcode == OP_PAIR_DIV
        ratio_choice = opcode == OP_RATIO_CHOICE
        ratio_band = opcode == OP_RATIO_BAND
        signed_add = opcode == OP_SIGNED_PAIR_ADD_IMMEDIATE
        ranked_greater = opcode == OP_SIGNED_RANKED_GREATER
        ranked_commit = opcode == OP_RANKED_COMMIT
        set_scalar = opcode == OP_SET_SCALAR
        pair_copy = opcode == OP_PAIR_COPY
        pair_euclid = opcode == OP_PAIR_EUCLID_STEP
        pair_product = opcode == OP_PAIR_PRODUCT
        semantic_invalid = semantic_invalid | (
            pair_set & ~valid_pair_address(arg0)
        ) | (
            pair_add
            & ~(
                valid_pair_address(arg0)
                & valid_pair_address(arg1)
                & valid_pair_address(arg2)
            )
        ) | (
            (pair_multiply | pair_subtract | pair_signed_subtract | signed_add)
            & ~valid_pair_address(arg0)
        ) | (
            pair_product & ~valid_pair_address(arg0)
        ) | (
            pair_copy
            & ~(valid_pair_address(arg0) & valid_pair_address(arg1))
        ) | (
            pair_euclid
            & ~(
                valid_pair_address(arg0)
                & valid_pair_address(arg1)
                & (arg0 != arg1)
            )
        ) | (
            pair_divide
            & ~(
                valid_pair_address(arg0)
                & valid_pair_address(arg1)
                & valid_pair_address(arg2)
            )
        ) | (
            (ratio_choice | ratio_band)
            & ~(
                valid_value_address(arg0)
                & valid_pair_address(arg1)
                & valid_pair_address(arg2)
            )
        ) | (
            ranked_greater
            & ~(
                valid_value_address(arg0)
                & valid_pair_address(arg1)
                & valid_pair_address(arg2)
                & valid_value_address(arg4)
                & valid_value_address(arg5)
            )
        ) | (
            ranked_commit
            & ~(valid_value_address(arg0) & valid_pair_address(arg1))
        ) | (
            set_scalar & ~valid_value_address(arg0)
        )

        def read_value_slot(index: Any) -> Any:
            bank = mx.stack(values, axis=1)
            bounded = mx.minimum(index, len(values) - 1)
            return mx.take_along_axis(bank, bounded[:, None], axis=1)[:, 0]

        def write_value_slot(index: Any, result: Any, authority: Any) -> None:
            for slot_index in range(len(values)):
                values[slot_index] = mx.where(
                    authority & (index == slot_index),
                    result,
                    values[slot_index],
                )

        def read_pair(low_index: Any) -> Any:
            return read_value_slot(low_index) + PROCESS_RADIX * read_value_slot(
                low_index + 1
            )

        def write_pair(low_index: Any, result: Any, authority: Any) -> None:
            valid_address = low_index < len(values) - 1
            write_value_slot(
                low_index,
                result % PROCESS_RADIX,
                authority & valid_address,
            )
            write_value_slot(
                low_index + 1,
                result // PROCESS_RADIX,
                authority & valid_address,
            )

        pair_set_result = arg1 + PROCESS_RADIX * arg2
        write_pair(arg0, pair_set_result, pair_set)

        pair_add_result = read_pair(arg1) + read_pair(arg2)
        write_pair(arg0, pair_add_result, pair_add)

        pair_multiply_result = read_pair(arg0) * arg1
        write_pair(arg0, pair_multiply_result, pair_multiply)

        write_pair(arg0, arg1 * arg2, pair_product)

        pair_subtract_result = read_pair(arg0) - arg1
        semantic_invalid = semantic_invalid | (
            pair_subtract & (pair_subtract_result < 0)
        )
        write_pair(arg0, mx.maximum(pair_subtract_result, 0), pair_subtract)

        signed_difference = read_pair(arg0) - arg1
        signed_difference_encoded = mx.where(
            signed_difference >= 0,
            2 * signed_difference,
            (-2 * signed_difference) - 1,
        )
        write_pair(arg0, signed_difference_encoded, pair_signed_subtract)

        write_pair(arg0, read_pair(arg1), pair_copy)

        euclid_left = read_pair(arg0)
        euclid_right = read_pair(arg1)
        euclid_next_left = mx.where(euclid_right == 0, euclid_left, euclid_right)
        euclid_next_right = mx.where(
            euclid_right == 0,
            0,
            euclid_left % mx.maximum(euclid_right, 1),
        )
        write_pair(arg0, euclid_next_left, pair_euclid)
        write_pair(arg1, euclid_next_right, pair_euclid)

        divide_numerator = read_pair(arg1)
        divide_denominator = read_pair(arg2)
        semantic_invalid = semantic_invalid | (
            pair_divide
            & (
                (divide_denominator == 0)
                | (
                    divide_numerator % mx.maximum(divide_denominator, 1)
                    != 0
                )
            )
        )
        divide_result = divide_numerator // mx.maximum(divide_denominator, 1)
        write_pair(arg0, divide_result, pair_divide)

        ratio_numerator = read_pair(arg1)
        ratio_denominator = read_pair(arg2)
        semantic_invalid = semantic_invalid | (
            (ratio_choice | ratio_band) & (ratio_denominator == 0)
        )
        choice_result = (2 * ratio_numerator >= ratio_denominator).astype(
            mx.int32
        ) + 1
        percentage = (100 * ratio_numerator) // mx.maximum(ratio_denominator, 1)
        band_result = mx.where(
            percentage < 50,
            1,
            mx.where(percentage < 70, 2, mx.where(percentage < 90, 3, 4)),
        )
        write_value_slot(arg0, choice_result, ratio_choice)
        write_value_slot(arg0, band_result, ratio_band)

        signed_current_encoded = read_pair(arg0)
        signed_current = mx.where(
            (signed_current_encoded % 2) == 0,
            signed_current_encoded // 2,
            -((signed_current_encoded + 1) // 2),
        )
        signed_immediate = mx.where(
            (arg1 % 2) == 0,
            arg1 // 2,
            -((arg1 + 1) // 2),
        )
        signed_result = signed_current + signed_immediate
        signed_result_encoded = mx.where(
            signed_result >= 0,
            2 * signed_result,
            (-2 * signed_result) - 1,
        )
        write_pair(arg0, signed_result_encoded, signed_add)

        candidate_encoded = read_pair(arg1)
        incumbent_encoded = read_pair(arg2)
        candidate_score = mx.where(
            (candidate_encoded % 2) == 0,
            candidate_encoded // 2,
            -((candidate_encoded + 1) // 2),
        )
        incumbent_score = mx.where(
            (incumbent_encoded % 2) == 0,
            incumbent_encoded // 2,
            -((incumbent_encoded + 1) // 2),
        )
        incumbent_rank = read_value_slot(arg4)
        has_incumbent = read_value_slot(arg5)
        ranked_result = (
            (has_incumbent == 0)
            | (candidate_score > incumbent_score)
            | ((candidate_score == incumbent_score) & (arg3 < incumbent_rank))
        ).astype(mx.int32)
        write_value_slot(arg0, ranked_result, ranked_greater)

        commit_selected = ranked_commit & (read_value_slot(arg0) != 0)
        candidate_pair = read_pair(arg1)
        if len(values) >= 5:
            values[0] = mx.where(commit_selected, arg2, values[0])
            values[1] = mx.where(
                commit_selected,
                candidate_pair % PROCESS_RADIX,
                values[1],
            )
            values[2] = mx.where(
                commit_selected,
                candidate_pair // PROCESS_RADIX,
                values[2],
            )
            values[3] = mx.where(commit_selected, arg3, values[3])
            values[4] = mx.where(ranked_commit, 1, values[4])


        # ── OP_CAUSAL_CHAIN ────────────────────────────────────────────────
        # The public intervention edges arrive before the baselines, so the
        # machine — not the compiler — works out which variable is the root
        # (two edges), which is the mediator (one) and which is downstream
        # (none). Six stages, held in values[8].
        #
        # This mirrors _semantic_micro_states in
        # core/learning/frontier_process_supervision.py line for line. Where
        # the reference raises, this sets semantic_invalid: the microcode has
        # no exception path, and a row that would have raised must become the
        # invalid state rather than a plausible-looking wrong one.
        causal_chain = opcode == OP_CAUSAL_CHAIN
        if len(values) >= 9:
            v0, v1, v2, v3, v4, v5, v6, v7, stage = (values[i] for i in range(9))
            change = arg3 + PROCESS_RADIX * arg4

            # Branch A: an intervention edge between two measured variables.
            edge = causal_chain & (arg0 <= 2) & (arg1 <= 2)
            first_edge = edge & (stage == 0)
            second_edge = edge & (stage == 1)
            mediator_edge = edge & (stage == 2)
            semantic_invalid = semantic_invalid | (
                edge & (stage >= 3)
            ) | (
                first_edge & ((arg0 == arg1) | (arg5 != 0))
            ) | (
                second_edge
                & (
                    (arg0 != v0)
                    | (arg1 == arg0)
                    | (arg1 == v1)
                    | (arg2 != v2)
                    | (arg5 != 1)
                )
            )
            # The mediator is whichever of the root's two targets this edge
            # starts from; its recorded change is the one the root produced
            # in it.
            arg0_is_first_target = arg0 == v1
            root_mediator_change = mx.where(
                arg0_is_first_target,
                v2 + PROCESS_RADIX * v3,
                v5 + PROCESS_RADIX * v6,
            )
            semantic_invalid = semantic_invalid | (
                mediator_edge
                & (
                    (arg0 == v0)
                    | ~((arg0 == v1) | (arg0 == v4))
                    | (arg1 == arg0)
                    | ~((arg1 == v1) | (arg1 == v4))
                    | (arg5 != 1)
                    | (root_mediator_change >= ACTION_NULL)
                    | (change >= ACTION_NULL)
                )
            )

            # Branch B: the downstream variable moves nothing.
            downstream_null = causal_chain & (arg0 <= 2) & (arg1 == 3)
            semantic_invalid = semantic_invalid | (
                downstream_null
                & (
                    (stage != 3)
                    | (arg0 != v2)
                    | (arg2 != 0)
                    | (arg3 != 0)
                    | (arg4 != 0)
                    | (arg5 != 1)
                )
            )

            # Branch C: predict the downstream value at the queried root delta.
            prediction = causal_chain & (arg0 == 3) & (arg1 == 3)
            exact_effects = (v4 > 0) & (v6 > 0)
            safe_v4 = mx.maximum(v4, 1)
            safe_v6 = mx.maximum(v6, 1)
            semantic_invalid = semantic_invalid | (
                prediction
                & (
                    (stage != 4)
                    | (arg2 < 1)
                    | (arg2 >= ACTION_NULL)
                    | (arg3 != 0)
                    | (arg4 != 0)
                    | (arg5 != 1)
                    | ~exact_effects
                    | (v3 % safe_v4 != 0)
                    | (v5 % safe_v6 != 0)
                )
            )
            effect = arg2 * (v3 // safe_v4) * (v5 // safe_v6)

            # Branch D: add the downstream baseline once its variable arrives.
            baseline_commit = causal_chain & (arg0 == 4) & (arg1 == 4)
            semantic_invalid = semantic_invalid | (
                baseline_commit
                & (
                    ((stage != 5) & (stage != 6))
                    | (arg2 != 0)
                    | (arg3 != 0)
                    | (arg4 != 0)
                    | (arg5 != 1)
                )
            )
            commit_now = baseline_commit & (v7 == v2)
            committed = (v3 + PROCESS_RADIX * v4) + (v5 + PROCESS_RADIX * v6)

            semantic_invalid = semantic_invalid | (
                causal_chain & ~(edge | downstream_null | prediction | baseline_commit)
            )

            # Every slot in one pass, from the PRE-instruction snapshot.
            values[0] = mx.where(first_edge | second_edge, arg0, values[0])
            values[1] = mx.where(
                first_edge, arg1, mx.where(mediator_edge, arg0, values[1])
            )
            values[2] = mx.where(
                first_edge,
                arg2,
                mx.where(second_edge, v3, mx.where(mediator_edge, arg1, values[2])),
            )
            values[3] = mx.where(
                first_edge,
                arg3,
                mx.where(
                    second_edge,
                    v4,
                    mx.where(
                        mediator_edge,
                        root_mediator_change,
                        mx.where(
                            prediction,
                            effect % PROCESS_RADIX,
                            mx.where(
                                commit_now, committed % PROCESS_RADIX, values[3]
                            ),
                        ),
                    ),
                ),
            )
            values[4] = mx.where(
                first_edge,
                arg4,
                mx.where(
                    second_edge,
                    arg1,
                    mx.where(
                        mediator_edge,
                        v7,
                        mx.where(
                            prediction,
                            effect // PROCESS_RADIX,
                            mx.where(
                                commit_now, committed // PROCESS_RADIX, values[4]
                            ),
                        ),
                    ),
                ),
            )
            values[5] = mx.where(
                first_edge,
                0,
                mx.where(
                    second_edge,
                    arg3,
                    mx.where(
                        mediator_edge, change, mx.where(prediction, 0, values[5])
                    ),
                ),
            )
            values[6] = mx.where(
                first_edge,
                0,
                mx.where(
                    second_edge,
                    arg4,
                    mx.where(mediator_edge, arg2, mx.where(prediction, 0, values[6])),
                ),
            )
            values[7] = mx.where(
                first_edge,
                0,
                mx.where(
                    second_edge,
                    arg2,
                    mx.where(mediator_edge, 0, mx.where(prediction, 0, values[7])),
                ),
            )
            values[8] = mx.where(
                first_edge,
                1,
                mx.where(
                    second_edge,
                    2,
                    mx.where(
                        mediator_edge,
                        3,
                        mx.where(
                            downstream_null,
                            4,
                            mx.where(
                                prediction, 5, mx.where(commit_now, 6, values[8])
                            ),
                        ),
                    ),
                ),
            )
        else:
            semantic_invalid = semantic_invalid | causal_chain

        write_value_slot(arg0, arg1, set_scalar)
        next_values = mx.stack((pc, *values, terminal), axis=1)
        invalid = semantic_invalid | mx.any(state == STATE_INVALID, axis=1) | mx.any(
            (next_values < 0) | (next_values >= self.config.state_cardinality),
            axis=1,
        )
        next_values = mx.where(
            invalid[:, None],
            mx.full_like(next_values, STATE_INVALID),
            next_values,
        )
        recognized = recognized | invalid
        categories = mx.arange(self.config.state_cardinality)[None, None, :]
        exact = (next_values[..., None] == categories).astype(mx.float32)
        return mx.log(mx.maximum(exact, 1e-6)), recognized

    def action_logits(
        self,
        problem_evidence: Any,
        hidden: Any,
        *,
        state_slot_start: int,
        step: int,
        token_ids: Any | None = None,
        state_probabilities: Any | None = None,
        teacher_values: Sequence[int] | None = None,
        teacher_forcing_probability: float = 0.0,
        prior_action_probabilities: Any | None = None,
        process_memory: Any | None = None,
        causal_action_lesion: bool = False,
        action_feedback_lesion: bool = False,
        family_action_lesion: bool = False,
        action_literal_binding_lesion: bool = False,
        action_workspace_trajectory: list[Any] | None = None,
        action_kernel_feature_trajectory: list[Any] | None = None,
    ) -> Any:
        """Decode the next typed operation from public evidence and current state."""

        stop = state_slot_start + self.config.state_slots
        if not 0 <= state_slot_start < stop <= int(hidden.shape[1]):
            raise ValueError("action decoder state slots are outside the recurrent state")
        state = mx.mean(
            hidden[:, state_slot_start:stop, :].astype(mx.float32), axis=1
        )
        query_input = state[:, None, :] + self.action_slot_embeddings[None, :, :]
        query = mx.einsum("bah,ahr->bar", query_input, self.action_query)
        keys = problem_evidence.astype(mx.float32) @ self.action_key
        values = problem_evidence.astype(mx.float32) @ self.action_value
        attention = mx.softmax(
            mx.einsum("bar,bnr->ban", query, keys)
            / math.sqrt(self.config.correction_rank),
            axis=-1,
        )
        context = mx.einsum("ban,bnr->bar", attention, values)
        depth = self.depth_features(step) @ self.action_depth
        features = mx.tanh(context + depth[None, None, :])
        logits = (
            mx.einsum("bar,arc->bac", features, self.action_output)
            + self.action_bias
        )
        workspace = self._action_workspace(
            problem_evidence,
            query,
            step=step,
            process_memory=process_memory,
        )
        if action_workspace_trajectory is not None:
            action_workspace_trajectory.append(workspace)
        kernel_feature = self._public_action_signature(
            token_ids,
            state_probabilities,
            step=step,
            width=int(workspace.shape[-1]),
        )
        if kernel_feature is None:
            kernel_feature = workspace
        if action_kernel_feature_trajectory is not None:
            action_kernel_feature_trajectory.append(kernel_feature)
        logits = logits + mx.einsum(
            "baw,awc->bac",
            workspace,
            self.action_workspace_output,
        )
        logits = self._family_action_logits(
            logits,
            workspace,
            kernel_feature,
            token_ids=token_ids,
            family_action_lesion=family_action_lesion,
        )
        logits = self._causal_action_logits(
            logits,
            workspace,
            problem_evidence=problem_evidence,
            token_ids=token_ids,
            teacher_values=teacher_values,
            teacher_forcing_probability=teacher_forcing_probability,
            prior_action_probabilities=prior_action_probabilities,
            causal_action_lesion=causal_action_lesion,
            action_feedback_lesion=action_feedback_lesion,
            action_literal_binding_lesion=action_literal_binding_lesion,
        )
        # Observation priors help before a recurrent state exists, but they are
        # not licensed semantics. Once execution has a state, derived values
        # need not occur literally in the prompt. Applying the copy prior there
        # assigns every absent category a large negative logit and makes broad
        # programs fight the parser merely to express a computed value. Exact
        # recognized public instructions remain authoritative below.
        use_observation_priors = state_probabilities is None
        if (
            use_observation_priors
            and token_ids is not None
            and self.config.literal_digit_token_ids
        ):
            pointer = self._literal_pointer_logits(
                problem_evidence,
                token_ids,
                query,
                self.action_key,
                self.config.action_cardinality,
            )
            copy_strength = mx.logaddexp(
                self.action_literal_copy_logit,
                mx.zeros_like(self.action_literal_copy_logit),
            )
            logits = logits + copy_strength[None, :, None] * pointer
        if (
            use_observation_priors
            and token_ids is not None
            and self.config.opcode_token_patterns
        ):
            contract = OpcodeObservationContract(
                self.config.opcode_token_patterns,
                self.config.opcode_context_patterns,
            )
            opcode_values, opcode_masks = contract.observe(token_ids.tolist())
            opcode_pointer = self._categorical_pointer_logits(
                problem_evidence,
                opcode_values,
                opcode_masks,
                query[:, :1, :],
                self.action_key,
                self.config.action_cardinality,
            )
            opcode_strength = mx.logaddexp(
                self.opcode_copy_logit,
                mx.zeros_like(self.opcode_copy_logit),
            )
            logits = mx.concatenate(
                [
                    logits[:, :1, :] + opcode_strength * opcode_pointer,
                    logits[:, 1:, :],
                ],
                axis=1,
            )
        if (
            token_ids is not None
            and state_probabilities is not None
            and self.config.literal_digit_token_ids
            and self.config.opcode_token_patterns
        ):
            contract = OpcodeObservationContract(
                self.config.opcode_token_patterns,
                self.config.opcode_context_patterns,
            )
            literal_contract = LiteralObservationContract(
                self.config.literal_digit_token_ids
            )
            state_rows = mx.argmax(state_probabilities, axis=-1).tolist()
            public_values, recognized = contract.public_instructions(
                token_ids.tolist(),
                literal_contract,
                state_rows,
            )
            exact = self._exact_categorical_logits(
                public_values,
                slots=self.config.action_slots,
                cardinality=self.config.action_cardinality,
            )
            known = mx.array(recognized, dtype=mx.bool_)
            logits = mx.where(known[:, None, None], exact, logits)
        return logits

    @staticmethod
    def _workspace_norm(value: Any) -> Any:
        return value / mx.sqrt(
            mx.mean(value.astype(mx.float32) ** 2, axis=-1, keepdims=True) + 1e-6
        )

    @staticmethod
    def _workspace_attention(query: Any, key: Any, value: Any) -> Any:
        width = int(query.shape[-1])
        heads = math.gcd(width, 4)
        head_width = width // heads

        def split(tensor: Any) -> Any:
            batch, tokens, _width = tensor.shape
            return tensor.reshape(batch, tokens, heads, head_width).transpose(0, 2, 1, 3)

        query_heads = split(query)
        key_heads = split(key)
        value_heads = split(value)
        weights = mx.softmax(
            (query_heads @ key_heads.transpose(0, 1, 3, 2))
            / math.sqrt(head_width),
            axis=-1,
        )
        attended = weights @ value_heads
        return attended.transpose(0, 2, 1, 3).reshape(
            query.shape[0], query.shape[1], width
        )

    def _action_workspace(
        self,
        problem_evidence: Any,
        query: Any,
        *,
        step: int,
        process_memory: Any | None = None,
    ) -> Any:
        """Compose public evidence and prior causal state into the next action."""

        depth = self.depth_features(step) @ self.action_workspace_depth
        workspace = (
            query @ self.action_workspace_seed
            + depth[None, None, :]
        )
        evidence = problem_evidence.astype(mx.float32)
        if process_memory is not None:
            if (
                len(process_memory.shape) != 3
                or int(process_memory.shape[0]) != int(evidence.shape[0])
                or int(process_memory.shape[-1]) != self.config.hidden_size
                or int(process_memory.shape[1]) < 1
            ):
                raise ValueError("action process memory differs from the recurrent schema")
            evidence = mx.concatenate(
                (evidence, process_memory.astype(mx.float32)),
                axis=1,
            )
        for layer in range(2):
            normalized = self._workspace_norm(workspace)
            cross_query = normalized @ self.action_workspace_cross_query[layer]
            cross_key = evidence @ self.action_workspace_cross_key[layer]
            cross_value = evidence @ self.action_workspace_cross_value[layer]
            cross = self._workspace_attention(
                cross_query,
                cross_key,
                cross_value,
            ) @ self.action_workspace_cross_output[layer]
            workspace = workspace + cross

            normalized = self._workspace_norm(workspace)
            reflected = self._workspace_attention(
                normalized @ self.action_workspace_self_query[layer],
                normalized @ self.action_workspace_self_key[layer],
                normalized @ self.action_workspace_self_value[layer],
            ) @ self.action_workspace_self_output[layer]
            workspace = workspace + reflected

            normalized = self._workspace_norm(workspace)
            expanded = nn.gelu(normalized @ self.action_workspace_ff_up[layer])
            workspace = workspace + expanded @ self.action_workspace_ff_down[layer]
        return self._workspace_norm(workspace)

    def _public_action_signature(
        self,
        token_ids: Any | None,
        state_probabilities: Any | None,
        *,
        step: int,
        width: int,
    ) -> Any | None:
        """Expose ordered public literals and causal state to learned tissue."""

        if (
            token_ids is None
            or state_probabilities is None
            or not self.config.literal_digit_token_ids
            or width < 19
        ):
            return None
        contract = LiteralObservationContract(
            self.config.literal_digit_token_ids,
            max_value=self.config.numeric_observation_max_value,
        )
        values, masks = contract.observe(token_ids.tolist())
        family_routes = self._frontier_family_routes(token_ids)
        family_width = FRONTIER_ACTION_EXPERT_COUNT
        fixed_width = (
            self.config.state_slots
            + self.config.depth_basis_size
            + family_width
            + 1
        )
        literal_slots = max(1, (width - fixed_width) // 2)
        literal_rows: list[list[float]] = []
        presence_rows: list[list[float]] = []
        for value_row, mask_row in zip(values, masks, strict=True):
            observed = [
                int(value)
                for value, present in zip(value_row, mask_row, strict=True)
                if present
            ][:literal_slots]
            literal_rows.append(
                [
                    value / float(self.config.numeric_observation_max_value)
                    for value in observed
                ]
                + [0.0] * (literal_slots - len(observed))
            )
            presence_rows.append(
                [1.0] * len(observed) + [0.0] * (literal_slots - len(observed))
            )
        literal_features = mx.array(literal_rows, dtype=mx.float32)
        presence_features = mx.array(presence_rows, dtype=mx.float32)
        categories = mx.arange(self.config.state_cardinality, dtype=mx.float32)
        state_features = (
            mx.sum(state_probabilities.astype(mx.float32) * categories, axis=-1)
            / float(self.config.state_cardinality - 1)
        )
        depth = mx.broadcast_to(
            self.depth_features(step)[None, :],
            (int(token_ids.shape[0]), self.config.depth_basis_size),
        )
        if family_routes is None:
            family_routes = mx.zeros(
                (int(token_ids.shape[0]), family_width), dtype=mx.float32
            )
        counts = mx.sum(presence_features, axis=-1, keepdims=True) / float(
            literal_slots
        )
        signature = mx.concatenate(
            (
                literal_features,
                presence_features,
                state_features,
                depth,
                family_routes,
                counts,
            ),
            axis=-1,
        )
        if int(signature.shape[-1]) < width:
            signature = mx.pad(
                signature,
                ((0, 0), (0, width - int(signature.shape[-1]))),
            )
        return mx.broadcast_to(
            signature[:, None, :],
            (int(signature.shape[0]), self.config.action_slots, width),
        )

    def _causal_action_logits(
        self,
        base_logits: Any,
        workspace: Any,
        *,
        problem_evidence: Any,
        token_ids: Any | None,
        teacher_values: Sequence[int] | None,
        teacher_forcing_probability: float,
        prior_action_probabilities: Any | None,
        causal_action_lesion: bool,
        action_feedback_lesion: bool,
        action_literal_binding_lesion: bool,
    ) -> Any:
        """Decode and ground one instruction left-to-right.

        Numeric grounding belongs inside the autoregressive action reader. A
        duration, score operand, or causal coefficient is often ambiguous until
        an earlier field has selected its row or role. Grounding every slot in
        parallel prevented that decision from moving later literal pointers.
        """

        expected_logits = (
            int(workspace.shape[0]),
            self.config.action_slots,
            self.config.action_cardinality,
        )
        if tuple(base_logits.shape) != expected_logits or tuple(workspace.shape[:2]) != (
            expected_logits[0],
            expected_logits[1],
        ):
            raise ValueError("causal action tensors differ from the canonical schema")
        if (
            type(causal_action_lesion) is not bool
            or type(action_feedback_lesion) is not bool
            or type(action_literal_binding_lesion) is not bool
        ):
            raise TypeError("causal action lesion flags must be bools")
        if (
            isinstance(teacher_forcing_probability, bool)
            or not isinstance(teacher_forcing_probability, (int, float))
            or not 0.0 <= float(teacher_forcing_probability) <= 1.0
        ):
            raise ValueError("action teacher-forcing probability must be inside [0, 1]")
        if teacher_values is not None and (
            len(teacher_values) != self.config.action_slots
            or any(
                type(value) is not int
                or not 0 <= value < self.config.action_cardinality
                for value in teacher_values
            )
        ):
            raise ValueError("action teacher values differ from the canonical schema")
        if prior_action_probabilities is not None and tuple(
            prior_action_probabilities.shape
        ) != expected_logits:
            raise ValueError("prior action probabilities differ from the canonical schema")
        if causal_action_lesion:
            return self._action_literal_binding_logits(
                base_logits,
                problem_evidence,
                workspace,
                token_ids=token_ids,
                lesion=action_literal_binding_lesion,
            )

        width = int(workspace.shape[-1])
        prefix = mx.zeros((expected_logits[0], width), dtype=mx.float32)
        if prior_action_probabilities is not None and not action_feedback_lesion:
            prior_values = mx.einsum(
                "bac,acw->baw",
                prior_action_probabilities.astype(mx.float32),
                self.action_causal_value_embeddings,
            )
            prefix = self._workspace_norm(
                mx.mean(prior_values, axis=1) @ self.action_causal_prior_projection
            )

        decisions: list[Any] = []
        categories = mx.arange(self.config.action_cardinality)[None, :]
        teacher_probability = float(teacher_forcing_probability)
        for slot in range(self.config.action_slots):
            feature = self._workspace_norm(
                workspace[:, slot, :].astype(mx.float32) + prefix
            )
            grounded = self._action_literal_binding_logits(
                base_logits,
                problem_evidence,
                workspace.astype(mx.float32) + prefix[:, None, :],
                token_ids=token_ids,
                lesion=action_literal_binding_lesion,
            )
            slot_logits = (
                grounded[:, slot, :]
                + feature @ self.action_causal_output[slot]
            )
            decisions.append(slot_logits)
            selected = self.straight_through_probabilities(
                slot_logits[:, None, :]
            )[:, 0, :]
            if teacher_values is not None and teacher_probability > 0.0:
                teacher = (
                    categories == int(teacher_values[slot])
                ).astype(mx.float32)
                selected = (
                    (1.0 - teacher_probability) * selected
                    + teacher_probability * teacher
                )
            emitted = selected @ self.action_causal_value_embeddings[slot]
            update = mx.tanh(
                emitted @ self.action_causal_input_projection
                + prefix @ self.action_causal_state_projection
            )
            prefix = self._workspace_norm(prefix + update)
        return mx.stack(decisions, axis=1)

    def _action_literal_binding_logits(
        self,
        base_logits: Any,
        problem_evidence: Any,
        workspace: Any,
        *,
        token_ids: Any | None,
        lesion: bool,
    ) -> Any:
        """Bind action fields to ordered public literals without assigning roles.

        A prompt literal may enter the process machine directly, as one radix
        digit, as the positive branch of the signed integer code, or through the
        small offset used by bounded signed deltas. The pointer learns which
        public position and representation a field needs. Computed values remain
        the responsibility of recurrent execution and the ordinary action path.
        """

        if type(lesion) is not bool:
            raise TypeError("action literal binding lesion flag must be bool")
        if lesion or token_ids is None or not self.config.literal_digit_token_ids:
            return base_logits
        if (
            len(problem_evidence.shape) != 3
            or len(workspace.shape) != 3
            or len(token_ids.shape) != 2
            or tuple(problem_evidence.shape[:2]) != tuple(token_ids.shape)
            or int(problem_evidence.shape[0]) != int(workspace.shape[0])
            or int(workspace.shape[1]) != self.config.action_slots
        ):
            raise ValueError("action literal binding tensors differ")

        contract = LiteralObservationContract(
            self.config.literal_digit_token_ids,
            max_value=self.config.numeric_observation_max_value,
        )
        values, masks = contract.observe(token_ids.tolist())
        value_ids = mx.array(values, dtype=mx.int32)
        observed = mx.array(masks, dtype=mx.bool_)
        positive_signed = 2 * value_ids
        candidates = mx.stack(
            (
                value_ids,
                value_ids % PROCESS_RADIX,
                value_ids // PROCESS_RADIX,
                positive_signed % PROCESS_RADIX,
                positive_signed // PROCESS_RADIX,
                value_ids + 3,
            ),
            axis=-1,
        )
        validity = mx.stack(
            (
                value_ids < self.config.action_cardinality,
                value_ids <= MAX_PROCESS_INTEGER,
                value_ids <= MAX_PROCESS_INTEGER,
                positive_signed <= MAX_PROCESS_INTEGER,
                positive_signed <= MAX_PROCESS_INTEGER,
                value_ids + 3 < self.config.action_cardinality,
            ),
            axis=-1,
        ) & observed[..., None]

        query = mx.einsum(
            "baw,awr->bar",
            self._workspace_norm(workspace.astype(mx.float32)),
            self.action_literal_binding_query,
        )
        key = problem_evidence.astype(mx.float32) @ self.action_literal_binding_key
        attention_logits = mx.einsum("bar,bnr->ban", query, key) / math.sqrt(
            int(key.shape[-1])
        )
        attention = mx.softmax(attention_logits, axis=-1)[..., None]
        weighted = attention * validity[:, None, :, :].astype(mx.float32)
        weighted = weighted / mx.maximum(mx.sum(weighted, axis=2, keepdims=True), 1e-12)
        categories = mx.arange(self.config.action_cardinality)[None, None, None, :]
        candidate_one_hot = (candidates[..., None] == categories).astype(mx.float32)
        probabilities = mx.einsum("bant,bntc->batc", weighted, candidate_one_hot)
        pointer_logits = mx.log(mx.maximum(probabilities, 1e-6))
        has_candidate = mx.any(validity, axis=1)
        pointer_logits = mx.where(
            has_candidate[:, None, :, None],
            pointer_logits,
            mx.zeros_like(pointer_logits),
        )
        binding_weights = mx.broadcast_to(
            self.action_literal_binding_output[None, :, :],
            (
                int(base_logits.shape[0]),
                self.config.action_slots,
                len(ACTION_LITERAL_BINDING_TRANSFORMS),
            ),
        )
        routes = self._frontier_family_routes(token_ids)
        if routes is not None:
            binding_weights = binding_weights + mx.einsum(
                "be,eat->bat",
                routes,
                self.action_literal_binding_family_output,
            )
        return base_logits + mx.einsum("bat,batc->bac", binding_weights, pointer_logits)

    def _frontier_family_routes(self, token_ids: Any | None) -> Any | None:
        """Return exact public family routes, or no route when unavailable."""

        if token_ids is None or not self.config.frontier_family_token_patterns:
            return None
        contract = FrontierFamilyObservationContract(
            self.config.frontier_family_token_patterns
        )
        opcodes, recognized = contract.observe(token_ids.tolist())
        return mx.array(
            [
                [
                    1.0
                    if known and opcode == OP_FRONTIER_TRAVERSE + expert
                    else 0.0
                    for expert in range(FRONTIER_ACTION_EXPERT_COUNT)
                ]
                for opcode, known in zip(opcodes, recognized, strict=True)
            ],
            dtype=mx.float32,
        )

    def _family_action_logits(
        self,
        base_logits: Any,
        workspace: Any,
        kernel_feature: Any,
        *,
        token_ids: Any | None,
        family_action_lesion: bool,
    ) -> Any:
        """Apply one public-family expert without exposing private supervision."""

        if type(family_action_lesion) is not bool:
            raise TypeError("family action lesion flag must be bool")
        if (
            family_action_lesion
            or token_ids is None
            or not self.config.frontier_family_token_patterns
        ):
            return base_logits
        routes = self._frontier_family_routes(token_ids)
        if routes is None:
            return base_logits
        expert_logits = mx.einsum(
            "baw,eawc->beac",
            workspace.astype(mx.float32),
            self.action_family_output,
        ) + self.action_family_bias[None, :, :, :]
        normalized = (
            kernel_feature[:, None, :, None, :].astype(mx.float32)
            - self.action_family_kernel_mean[None, :, :, None, :]
        ) * self.action_family_kernel_inv_scale[None, :, :, None, :]
        distance = mx.mean(
            mx.square(
                normalized
                - self.action_family_kernel_prototypes[None, :, :, :, :]
            ),
            axis=-1,
        )
        kernel = (
            mx.exp(
                -self.action_family_kernel_gamma[None, :, :, None]
                * distance
            )
            * self.action_family_kernel_mask[None, :, :, :]
        )
        expert_logits = expert_logits + mx.einsum(
            "beap,eapc->beac",
            kernel,
            self.action_family_kernel_coefficients,
        )
        return base_logits + mx.einsum("be,beac->bac", routes, expert_logits)

    @staticmethod
    def _exact_categorical_logits(
        values: Sequence[Sequence[int]],
        *,
        slots: int,
        cardinality: int,
    ) -> Any:
        if any(
            len(row) != slots
            or any(type(value) is not int or not 0 <= value < cardinality for value in row)
            for row in values
        ):
            raise ValueError("exact categorical values differ from their schema")
        labels = mx.array(values, dtype=mx.int32)[..., None]
        categories = mx.arange(cardinality)[None, None, :]
        exact = (labels == categories).astype(mx.float32)
        return mx.log(mx.maximum(exact, 1e-6))

    def commit_action_logits(self, logits: Any) -> Any:
        """Convert predicted categorical operations into causal action tissue."""

        if (
            len(logits.shape) != 3
            or int(logits.shape[1]) != self.config.action_slots
            or int(logits.shape[2]) != self.config.action_cardinality
        ):
            raise ValueError("typed action logits differ from the canonical schema")
        probabilities = self.straight_through_probabilities(logits)
        return self.commit_action_probabilities(probabilities)

    @staticmethod
    def straight_through_probabilities(logits: Any) -> Any:
        """Use hard categories in the forward pass and soft gradients backward."""

        probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
        selected = mx.argmax(probabilities, axis=-1)
        categories = mx.arange(int(logits.shape[-1]))[None, None, :]
        hard = (categories == selected[..., None]).astype(mx.float32)
        return probabilities + mx.stop_gradient(hard - probabilities)

    @staticmethod
    def exact_probabilities(
        values: Sequence[int],
        *,
        slots: int,
        cardinality: int,
    ) -> Any:
        if len(values) != slots or any(
            type(value) is not int or not 0 <= value < cardinality
            for value in values
        ):
            raise ValueError("exact categorical values differ from their schema")
        labels = mx.array(values, dtype=mx.int32)[None, :, None]
        categories = mx.arange(cardinality)[None, None, :]
        return (categories == labels).astype(mx.float32)

    def commit_action_probabilities(self, probabilities: Any) -> Any:
        if probabilities.shape[1:] != (
            self.config.action_slots,
            self.config.action_cardinality,
        ):
            raise ValueError("typed action probabilities differ from the schema")
        return mx.einsum(
            "bac,ach->bah", probabilities, self.action_value_embeddings
        )

    def teacher_action_state(self, values: Sequence[int]) -> Any:
        if len(values) != self.config.action_slots or any(
            type(value) is not int or not 0 <= value < self.config.action_cardinality
            for value in values
        ):
            raise ValueError("teacher action values differ from the canonical schema")
        exact = self.exact_probabilities(
            values,
            slots=self.config.action_slots,
            cardinality=self.config.action_cardinality,
        )
        return mx.einsum("bac,ach->bah", exact, self.action_value_embeddings)

    def commit_state_logits(
        self,
        hidden: Any,
        *,
        state_slot_start: int,
        logits: Any,
    ) -> Any:
        """Commit supplied transition logits through the categorical codebook."""

        if logits.shape != (
            int(hidden.shape[0]),
            self.config.state_slots,
            self.config.state_cardinality,
        ):
            raise ValueError("state transition logits differ from the canonical schema")
        probabilities = self.straight_through_probabilities(logits)
        return self.commit_state_probabilities(
            hidden,
            state_slot_start=state_slot_start,
            probabilities=probabilities,
        )

    def commit_state_probabilities(
        self,
        hidden: Any,
        *,
        state_slot_start: int,
        probabilities: Any,
    ) -> Any:
        if probabilities.shape[1:] != (
            self.config.state_slots,
            self.config.state_cardinality,
        ):
            raise ValueError("typed state probabilities differ from the schema")
        replacement = mx.einsum(
            "bsc,sch->bsh",
            probabilities,
            self.state_value_embeddings,
        )
        return self._replace_state_slots(hidden, state_slot_start, replacement)

    def teacher_state_transition(
        self,
        hidden: Any,
        *,
        state_slot_start: int,
        values: Sequence[int],
        probability: float,
    ) -> Any:
        """Blend an exact prior state into training roll-in without prompt leakage."""

        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("state teacher-forcing probability must be inside [0, 1]")
        if len(values) != self.config.state_slots or any(
            type(value) is not int or not 0 <= value < self.config.state_cardinality
            for value in values
        ):
            raise ValueError("teacher state values differ from the canonical schema")
        labels = mx.array(values, dtype=mx.int32)[None, :, None]
        categories = mx.arange(self.config.state_cardinality)[None, None, :]
        exact = (categories == labels).astype(mx.float32)
        replacement = mx.einsum(
            "bsc,sch->bsh",
            exact,
            self.state_value_embeddings,
        )
        teacher = self._replace_state_slots(hidden, state_slot_start, replacement)
        if float(probability) == 1.0:
            return teacher
        start = state_slot_start
        stop = start + self.config.state_slots
        mixed = (
            (1.0 - float(probability)) * hidden[:, start:stop, :]
            + float(probability) * teacher[:, start:stop, :]
        )
        return mx.concatenate(
            [hidden[:, :start, :], mixed, hidden[:, stop:, :]],
            axis=1,
        )

    def _replace_state_slots(
        self,
        hidden: Any,
        state_slot_start: int,
        replacement: Any,
    ) -> Any:
        """Replace typed slots at matched residual scale without scale gradients."""

        previous = hidden[
            :,
            state_slot_start : state_slot_start + self.config.state_slots,
            :,
        ]
        scale = mx.stop_gradient(
            _rms(previous.astype(mx.float32))
            / _rms(replacement.astype(mx.float32))
        )
        replacement = replacement * scale.astype(replacement.dtype)
        committed = mx.concatenate(
            [
                hidden[:, :state_slot_start, :],
                replacement.astype(hidden.dtype),
                hidden[:, state_slot_start + self.config.state_slots :, :],
            ],
            axis=1,
        )
        return committed

    def identity_initialized(self) -> bool:
        return bool(mx.all(self.correction_b == 0))

    def parameter_sha256(self) -> str:
        digest = hashlib.sha256()
        for name in (
            "correction_a",
            "correction_b",
            "depth_scale",
            "memory_write_weight",
            "memory_write_bias",
            "transport_depth_weight",
            "transport_state_weight",
            "transport_motion_weight",
            "transport_bias",
            "transport_decay_logit",
            "halt_state_weight",
            "halt_motion_weight",
            "halt_bias",
            "answer_query",
            "answer_key",
            "answer_value",
            "answer_output",
            *PROCESS_READER_PARAMETER_NAMES,
            "answer_gate_query",
            "answer_gate_logit",
            "answer_role_projection",
            "answer_role_bias",
            "answer_place_projection",
            "answer_place_state_projection",
            "answer_place_width_projection",
            "answer_place_bias",
            "answer_digit_gate_logit",
            "state_readout_weight",
            "state_readout_bias",
            "state_slot_embeddings",
            "state_value_embeddings",
            *INITIAL_STATE_PARAMETER_NAMES,
            "state_transition_query",
            "state_transition_key",
            "state_transition_value",
            "state_transition_self",
            "state_transition_output",
            "state_transition_depth",
            "state_transition_bias",
            "action_slot_embeddings",
            "action_value_embeddings",
            "action_query",
            "action_key",
            "action_value",
            "action_output",
            "action_depth",
            "action_bias",
            *ACTION_WORKSPACE_PARAMETER_NAMES,
            *CAUSAL_ACTION_PARAMETER_NAMES,
            *FAMILY_ACTION_PARAMETER_NAMES,
            *ACTION_LITERAL_BINDING_PARAMETER_NAMES,
            "state_action_projection",
            *TRANSITION_MEMORY_PARAMETER_NAMES,
            *TRANSITION_TAPE_READER_PARAMETER_NAMES,
            *TRANSITION_PROCESSOR_PARAMETER_NAMES,
            *TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES,
            *TRANSITION_REPLAY_PARAMETER_NAMES,
            "literal_value_embeddings",
            "literal_grounding_logit",
            "state_literal_copy_logit",
            "action_literal_copy_logit",
            "opcode_copy_logit",
        ):
            value = getattr(self, name)
            mx.eval(value)
            digest.update(name.encode("ascii"))
            digest.update(bytes(memoryview(value.astype(mx.float32))))
        return digest.hexdigest()

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": UNIFIED_INTRINSIC_RECURRENCE_SCHEMA,
            "config": self.config.to_dict(),
            "identity_initialized": self.identity_initialized(),
            "continuous_depth_basis": "bounded_rational_polynomial",
            "depth_extrapolation_defined": True,
            "typed_state_bottleneck": "straight_through_categorical",
            "predicted_state_is_next_step_input": True,
            "state_processor": "separate_initial_parser_and_recurrent_transition",
            "action_processor": (
                "public_evidence_bounded_autoregressive_typed_action_workspace"
            ),
            "action_process_memory": {
                "source": "complete_prior_typed_process_tape",
                "future_steps_visible": False,
                "private_answer_exposed": False,
            },
            "state_transition_memory": {
                "architecture": (
                    "slot_preserving_gated_summary_plus_causal_public_tape_reader"
                ),
                "field_order": list(ACTION_SLOT_NAMES),
                "state_register_order": list(
                    state_slot_names(self.config.state_slots)
                ),
                "current_and_prior_actions_visible": True,
                "future_actions_visible": False,
                "private_transition_trace_visible": False,
                "bootstrap_contract": "zero_output_exact_parent_noop",
                "output_active": bool(mx.any(self.transition_memory_output != 0)),
                "public_tape_reader": {
                    "prefix_retained_before_query": True,
                    "query_conditioning": "current_typed_state_and_action",
                    "position_encoding": "bounded_rational_polynomial",
                    "future_action_visible": False,
                    "private_transition_trace_visible": False,
                    "bootstrap_contract": "zero_output_exact_parent_noop",
                    "output_active": bool(mx.any(self.transition_tape_output != 0)),
                },
            },
            "state_transition_processor": {
                "architecture": "typed_categorical_higher_order_register_processor",
                "opcode_isolation": "independent_zero_attached_categorical_expert_heads",
                "category_identity": "exact_one_hot_or_deterministic_fourier",
                "interactions": [
                    "state_action",
                    "action_action",
                    "state_history",
                    "action_action_history",
                ],
                "bootstrap_contract": "zero_output_exact_parent_noop",
                "output_active": bool(mx.any(self.transition_processor_output != 0)),
                "opcode_expert_output_active": bool(
                    mx.any(self.transition_processor_opcode_output != 0)
                ),
                "opcode_interaction_expert_active": bool(
                    mx.any(self.transition_processor_opcode_interaction_down != 0)
                ),
                "lesionable": True,
                "private_transition_trace_visible": False,
            },
            "state_transition_replay": {
                "architecture": "state_independent_causal_public_prefix_reader",
                "query_conditioning": "fixed_state_register_and_action_field_queries",
                "future_action_visible": False,
                "private_transition_trace_visible": False,
                "bootstrap_contract": "zero_candidate_exact_parent_noop",
                "shared_output_active": bool(mx.any(self.transition_replay_output != 0)),
                "opcode_output_active": bool(
                    mx.any(self.transition_replay_opcode_output != 0)
                ),
                "learned_consistency_gate": True,
                "lesionable": True,
            },
            "numeric_observation": {
                "max_value": self.config.numeric_observation_max_value,
                "encoding": "direct_category_then_ordered_radix_pair",
                "radix": PROCESS_RADIX,
                "semantic_role_inferred": False,
            },
            "predicted_action_is_state_transition_input": True,
            "state_transition_action_history": {
                "source": "ordered_current_inclusive_typed_action_tape",
                "current_action_path": (
                    "legacy_slot_projection_plus_slot_preserving_gated_tape"
                ),
                "future_steps_visible": False,
                "private_transition_program_visible": False,
                "parameter_topology_changed": True,
                "lesionable": True,
            },
            "causal_action_decoder": {
                "field_order": list(ACTION_SLOT_NAMES),
                "future_field_teacher_leakage": False,
                "previous_action_feedback": True,
                "parent_attachment": "exact_zero_output_noop",
                "lesionable": True,
            },
            "frontier_action_experts": {
                "count": FRONTIER_ACTION_EXPERT_COUNT,
                "route": (
                    "tokenizer_bound_public_family_declaration"
                    if self.config.frontier_family_token_patterns
                    else "disabled"
                ),
                "private_transition_program_visible": False,
                "affine_feature_source": "learned_recurrent_workspace",
                "kernel_feature_source": (
                    "ordered_public_literals_current_state_depth_and_family"
                    if self.config.literal_digit_token_ids
                    else "learned_recurrent_workspace"
                ),
                "parent_attachment": "exact_zero_output_noop",
                "lesionable": True,
            },
            "action_literal_binding": {
                "source": "ordered_public_prompt_literals",
                "transforms": list(ACTION_LITERAL_BINDING_TRANSFORMS),
                "transform_selection": "public_family_conditioned",
                "field_conditioning": "prior_decoded_action_fields",
                "semantic_role_inferred": True,
                "private_transition_program_visible": False,
                "parent_attachment": "exact_zero_output_noop",
                "lesionable": True,
            },
            "terminal_state_semantics": "exact_idempotent_stutter",
            "terminal_decode_semantics": "first_terminal_state_preserved",
            "state_problem_evidence": "frozen_deep_prefix_no_decoder_suffix",
            "literal_grounding": (
                "tokenizer_bound_bounded_integer_observations"
                if self.config.literal_digit_token_ids
                else "disabled"
            ),
            "literal_role_binding": (
                "learned_attention_exact_category_pointer"
                if self.config.literal_digit_token_ids
                else "disabled"
            ),
            "opcode_grounding": (
                "tokenizer_bound_operation_observations"
                if self.config.opcode_token_patterns
                else "disabled"
            ),
            "public_instruction_decoder": (
                "tokenizer_bound_state_selected_exact_microcode"
                if self.config.opcode_token_patterns
                else "disabled"
            ),
            "transformer_answer_passes_per_state": 1,
            "state_to_answer_bridge": (
                "causally_contextualized_ordered_typed_masked_action_state_"
                "process_tape_attention_over_frozen_readout"
            ),
            "process_tape": {
                "schema": PROCESS_TAPE_SCHEMA,
                "ordering": "bounded_sinusoidal_step_and_transition_role",
                "reader": "two_independent_rank_expanded_causal_prefix_blocks",
                "reader_rank": min(
                    self.config.hidden_size, 4 * self.config.correction_rank
                ),
                "contents": [
                    "pre_state",
                    "typed_action",
                    "post_state",
                    "state_delta",
                ],
                "terminal_stutter_entries_masked": True,
            },
            "parameter_sha256": self.parameter_sha256(),
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


def _unified_recurrent_hidden_states_part_1(controller, detach_problem_evidence, initial_state_logit_trajectory, initial_state_teacher_values, problem_evidence, state_slot_start, state_teacher_forcing_probability, tokens):
    if detach_problem_evidence:
        problem_evidence = mx.stop_gradient(problem_evidence)
    problem_evidence = controller.ground_literal_evidence(
        problem_evidence,
        tokens[:, :state_slot_start],
    )
    # A recurrent machine must start from the problem's state, not from one
    # task-independent learned vector.  The prediction uses only the public
    # prefix and remains the sole source at inference; exact initial values
    # are an annealed training authority only.
    initial_state_logits = controller.initial_state_logits(
        problem_evidence,
        tokens[:, :state_slot_start],
    )
    state_probabilities = controller.straight_through_probabilities(
        initial_state_logits
    )
    if initial_state_logit_trajectory is not None:
        initial_state_logit_trajectory.append(initial_state_logits)
    if (
        initial_state_teacher_values is not None
        and state_teacher_forcing_probability > 0.0
    ):
        teacher_initial = controller.exact_probabilities(
            initial_state_teacher_values,
            slots=controller.config.state_slots,
            cardinality=controller.config.state_cardinality,
        )
        state_probabilities = (
            (1.0 - state_teacher_forcing_probability) * state_probabilities
            + state_teacher_forcing_probability * teacher_initial
        )
    return problem_evidence, state_probabilities

def _unified_recurrent_hidden_states_action_probabilities(action_logits, action_probability_history, action_teacher_values, controller, iteration, public_action_values, state_teacher_forcing_probability, typed_action_lesion):
    action_probabilities = controller.straight_through_probabilities(
        action_logits
    )
    if public_action_values is not None:
        action_probabilities = controller.exact_probabilities(
            public_action_values[iteration],
            slots=controller.config.action_slots,
            cardinality=controller.config.action_cardinality,
        )
    if typed_action_lesion:
        action_probabilities = controller.exact_probabilities(
            (ACTION_NULL,) * (controller.config.action_slots - 1) + (0,),
            slots=controller.config.action_slots,
            cardinality=controller.config.action_cardinality,
        )
    if (
        action_teacher_values is not None
        and state_teacher_forcing_probability > 0.0
    ):
        teacher_action = controller.exact_probabilities(
            action_teacher_values[iteration],
            slots=controller.config.action_slots,
            cardinality=controller.config.action_cardinality,
        )
        action_probabilities = (
            (1.0 - state_teacher_forcing_probability)
            * action_probabilities
            + state_teacher_forcing_probability * teacher_action
        )
    action_state = controller.commit_action_probabilities(
        action_probabilities
    )
    action_probability_history.append(action_probabilities)
    return action_probabilities, action_state

def _unified_recurrent_hidden_states_part_3(action_state, active, controller, hidden, iteration, post_state, pre_state, process_tape, process_tape_masks):
    process_tape.extend(
        (
            controller.encode_process_tape_entry(
                pre_state,
                step=iteration,
                kind="state_pre",
            ),
            controller.encode_process_tape_entry(
                action_state,
                step=iteration,
                kind="action",
            ),
            controller.encode_process_tape_entry(
                post_state,
                step=iteration,
                kind="state_post",
            ),
            controller.encode_process_tape_entry(
                post_state - pre_state,
                step=iteration,
                kind="state_delta",
            ),
        )
    )
    process_tape_masks.extend(
        (
            mx.broadcast_to(
                active[:, None],
                (
                    int(hidden.shape[0]),
                    controller.config.state_slots,
                ),
            ),
            mx.broadcast_to(
                active[:, None],
                (
                    int(hidden.shape[0]),
                    controller.config.action_slots,
                ),
            ),
            mx.broadcast_to(
                active[:, None],
                (
                    int(hidden.shape[0]),
                    controller.config.state_slots,
                ),
            ),
            mx.broadcast_to(
                active[:, None],
                (
                    int(hidden.shape[0]),
                    controller.config.state_slots,
                ),
            ),
        )
    )

def unified_recurrent_hidden_states(
    model: Any,
    tokens: Any,
    plan: RecurrentDepthPlan,
    controller: UnifiedRecurrentController,
    *,
    memory_layout: MemoryLayout | None = None,
    adaptive_halt: bool = False,
    soft_memory_writes: bool = False,
    state_slot_start: int | None = None,
    state_logit_trajectory: list[Any] | None = None,
    action_logit_trajectory: list[Any] | None = None,
    action_workspace_trajectory: list[Any] | None = None,
    action_kernel_feature_trajectory: list[Any] | None = None,
    initial_state_logit_trajectory: list[Any] | None = None,
    decode_state_trajectory: list[Any] | None = None,
    recurrent_input_trajectory: list[Any] | None = None,
    state_probability_trajectory: list[Any] | None = None,
    answer_role_logit_trajectory: list[Any] | None = None,
    answer_place_logit_trajectory: list[Any] | None = None,
    answer_binding_feature_trajectory: list[tuple[Any, Any, Any]] | None = None,
    state_teacher_values: Sequence[Sequence[int]] | None = None,
    action_teacher_values: Sequence[Sequence[int]] | None = None,
    public_action_values: Sequence[Sequence[int]] | None = None,
    initial_state_teacher_values: Sequence[int] | None = None,
    state_teacher_forcing_probability: float = 0.0,
    typed_action_lesion: bool = False,
    causal_action_lesion: bool = False,
    action_feedback_lesion: bool = False,
    family_action_lesion: bool = False,
    action_literal_binding_lesion: bool = False,
    microcode_lesion: bool = False,
    transition_processor_lesion: bool = False,
    transition_processor_mode: str = "residual",
    transition_copy_prior_logit_bias: float = TRANSITION_COPY_PRIOR_LOGIT_BIAS,
    transition_opcode_expert_routing: str = "opcode",
    transition_replay_mode: str = "disabled",
    transition_history_lesion: bool = False,
    process_tape_lesion: bool = False,
    process_only: bool = False,
    detach_problem_evidence: bool = True,
    caches: dict[str, Any] | None = None,
) -> tuple[Any, list[Any], UnifiedRecurrenceTelemetry]:
    """Run all Level-3 control mechanisms on one transformer trajectory."""

    if not isinstance(controller, UnifiedRecurrentController):
        raise TypeError("unified recurrence controller is invalid")
    if (
        type(adaptive_halt) is not bool
        or type(soft_memory_writes) is not bool
        or type(typed_action_lesion) is not bool
        or type(causal_action_lesion) is not bool
        or type(action_feedback_lesion) is not bool
        or type(family_action_lesion) is not bool
        or type(action_literal_binding_lesion) is not bool
        or type(microcode_lesion) is not bool
        or type(transition_processor_lesion) is not bool
        or type(transition_history_lesion) is not bool
        or type(process_tape_lesion) is not bool
        or type(process_only) is not bool
        or type(detach_problem_evidence) is not bool
    ):
        raise TypeError("unified recurrence mode flags must be bools")
    if transition_processor_mode not in TRANSITION_PROCESSOR_MODES:
        raise ValueError("unified recurrence transition processor mode differs")
    if transition_opcode_expert_routing not in TRANSITION_OPCODE_EXPERT_ROUTING_MODES:
        raise ValueError("unified recurrence opcode routing differs")
    if transition_replay_mode not in TRANSITION_REPLAY_MODES:
        raise ValueError("unified recurrence transition replay mode differs")
    if adaptive_halt and controller.config.minimum_iterations > plan.iterations:
        raise ValueError("minimum iterations exceed the recurrence plan")
    if caches is not None:
        if set(caches) != {"prelude", "window", "coda"}:
            raise ValueError("caches must come from make_recurrent_caches")
        if len(caches["window"]) != plan.iterations:
            raise ValueError("cache iteration count does not match the plan")
        expected_window_layers = plan.coda_start - plan.prelude_end
        if (
            len(caches["prelude"]) != plan.prelude_end
            or any(
                len(iteration_caches) != expected_window_layers
                for iteration_caches in caches["window"]
            )
        ):
            raise ValueError("cache layer topology does not match the plan")
        if any(
            value is not None
            for value in (
                memory_layout,
                state_slot_start,
                state_teacher_values,
                action_teacher_values,
                public_action_values,
                initial_state_teacher_values,
            )
        ) or adaptive_halt or soft_memory_writes:
            raise ValueError(
                "incremental recurrent caches require the untyped fixed-depth path"
            )
    if state_teacher_values is not None and state_slot_start is None:
        raise ValueError("state teacher roll-in requires typed state slots")
    if initial_state_teacher_values is not None and state_slot_start is None:
        raise ValueError("initial state teacher requires typed state slots")
    if action_teacher_values is not None and state_slot_start is None:
        raise ValueError("action teacher requires typed state slots")
    if public_action_values is not None and state_slot_start is None:
        raise ValueError("public actions require typed state slots")
    if public_action_values is not None and action_teacher_values is not None:
        raise ValueError("public actions and private action teacher are mutually exclusive")
    if typed_action_lesion and action_teacher_values is not None:
        raise ValueError("typed action lesion cannot accompany an action teacher")
    if typed_action_lesion and state_slot_start is None:
        raise ValueError("typed action lesion requires typed state slots")
    if process_only and state_slot_start is None:
        raise ValueError("process-only recurrence requires typed state slots")
    if process_only and any(
        value is not None
        for value in (
            decode_state_trajectory,
            answer_role_logit_trajectory,
            answer_place_logit_trajectory,
            answer_binding_feature_trajectory,
        )
    ):
        raise ValueError("process-only recurrence cannot emit answer trajectories")
    if state_teacher_values is not None and len(state_teacher_values) < plan.iterations:
        raise ValueError("state teacher roll-in is shorter than the recurrence plan")
    if action_teacher_values is not None and len(action_teacher_values) < plan.iterations:
        raise ValueError("action teacher is shorter than the recurrence plan")
    if public_action_values is not None and len(public_action_values) < plan.iterations:
        raise ValueError("public action program is shorter than the recurrence plan")
    if (
        isinstance(state_teacher_forcing_probability, bool)
        or not isinstance(state_teacher_forcing_probability, (int, float))
        or not 0.0 <= float(state_teacher_forcing_probability) <= 1.0
    ):
        raise ValueError("state teacher-forcing probability must be inside [0, 1]")
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers or plan.coda_start > len(layers):
        raise ValueError("model layers do not satisfy the recurrence plan")
    if caches is not None and len(caches["coda"]) != len(layers) - plan.coda_start:
        raise ValueError("cache layer topology does not match the plan")

    hidden = inner.embed_tokens(tokens)
    hidden = _run(
        layers[: plan.prelude_end],
        hidden,
        caches["prelude"] if caches else None,
    )
    if state_slot_start is not None:
        if (
            type(state_slot_start) is not int
            or not 0 <= state_slot_start <= int(hidden.shape[1])
        ):
            raise ValueError("state slot insertion is outside the token sequence")
        slots = controller.initial_state_slots(int(hidden.shape[0]), hidden.dtype)
        hidden = mx.concatenate(
            [hidden[:, :state_slot_start, :], slots, hidden[:, state_slot_start:, :]],
            axis=1,
        )
    anchor = hidden
    anchor_rms = _rms(anchor) if plan.renormalize else None
    if memory_layout is not None and memory_layout.n_slots != int(hidden.shape[1]):
        raise ValueError("protected memory layout differs from token positions")

    trajectory: list[Any] = []
    halt_probabilities: list[float] = []
    memory_write_means: list[float] = []
    transport_gates: list[float] = []
    process_tape: list[Any] = []
    process_tape_masks: list[Any] = []
    window = layers[plan.prelude_end : plan.coda_start]
    problem_evidence: Any | None = None
    state_probabilities: Any | None = None
    prior_action_probabilities: Any | None = None
    action_probability_history: list[Any] = []
    if state_slot_start is not None:
        # Prefix-only execution is causally identical to the same positions in
        # the complete sequence, but cannot expose teacher-forced answer tokens.
        # Inference detaches this evidence by default. Governed process
        # acquisition may differentiate it so the scoped transformer tissue,
        # rather than only its downstream categorical controller, can learn to
        # parse the public problem statement.
        with recurrent_iteration(0):
            problem_evidence = _run(window, anchor[:, :state_slot_start, :])
        problem_evidence, state_probabilities = _unified_recurrent_hidden_states_part_1(controller, detach_problem_evidence, initial_state_logit_trajectory, initial_state_teacher_values, problem_evidence, state_slot_start, state_teacher_forcing_probability, tokens)
        hidden = controller.commit_state_probabilities(
            hidden,
            state_slot_start=state_slot_start,
            probabilities=state_probabilities,
        )
    halted = False
    halt_reason = "configured_depth_exhausted"
    last_decode_state: Any | None = None
    for iteration in range(plan.iterations):
        if state_slot_start is not None:
            state_stop = state_slot_start + controller.config.state_slots
            prior_terminal_mask: Any | None = None
            if iteration > 0:
                hidden = mx.concatenate(
                    [
                        anchor[:, :state_slot_start, :],
                        hidden[:, state_slot_start:state_stop, :],
                        anchor[:, state_stop:, :],
                    ],
                    axis=1,
                )
            if last_decode_state is not None and state_probabilities is not None:
                prior_terminal_mask = (
                    mx.argmax(state_probabilities[:, -1, :], axis=-1) == 1
                )
            prior_state = hidden
            recurrent_input = hidden
            if recurrent_input_trajectory is not None:
                recurrent_input_trajectory.append(recurrent_input)
            with recurrent_iteration(iteration):
                action_logits = controller.action_logits(
                    problem_evidence,
                    recurrent_input,
                    state_slot_start=state_slot_start,
                    step=iteration,
                    token_ids=tokens[:, :state_slot_start],
                    state_probabilities=state_probabilities,
                    teacher_values=(
                        action_teacher_values[iteration]
                        if action_teacher_values is not None
                        else None
                    ),
                    teacher_forcing_probability=state_teacher_forcing_probability,
                    prior_action_probabilities=prior_action_probabilities,
                    process_memory=(
                        mx.concatenate(process_tape, axis=1)
                        if process_tape
                        else None
                    ),
                    causal_action_lesion=causal_action_lesion,
                    action_feedback_lesion=action_feedback_lesion,
                    family_action_lesion=family_action_lesion,
                    action_literal_binding_lesion=action_literal_binding_lesion,
                    action_workspace_trajectory=action_workspace_trajectory,
                    action_kernel_feature_trajectory=action_kernel_feature_trajectory,
                )
                action_probabilities, action_state = _unified_recurrent_hidden_states_action_probabilities(action_logits, action_probability_history, action_teacher_values, controller, iteration, public_action_values, state_teacher_forcing_probability, typed_action_lesion)
                prior_action_probabilities = action_probabilities
                state_logits = controller.state_transition_logits(
                    problem_evidence,
                    recurrent_input,
                    state_slot_start=state_slot_start,
                    step=iteration,
                    action_state=action_state,
                    state_probabilities=state_probabilities,
                    action_probabilities=action_probabilities,
                    action_probability_history=(
                        None
                        if transition_history_lesion
                        else action_probability_history
                    ),
                    microcode_lesion=microcode_lesion,
                    transition_processor_lesion=transition_processor_lesion,
                    transition_processor_mode=transition_processor_mode,
                    transition_copy_prior_logit_bias=(
                        transition_copy_prior_logit_bias
                    ),
                    opcode_expert_routing=transition_opcode_expert_routing,
                    transition_replay_mode=transition_replay_mode,
                )
                next_state_probabilities = (
                    controller.straight_through_probabilities(state_logits)
                )
                if (
                    state_teacher_values is not None
                    and state_teacher_forcing_probability > 0.0
                ):
                    teacher_state = controller.exact_probabilities(
                        state_teacher_values[iteration],
                        slots=controller.config.state_slots,
                        cardinality=controller.config.state_cardinality,
                    )
                    next_state_probabilities = (
                        (1.0 - state_teacher_forcing_probability)
                        * next_state_probabilities
                        + state_teacher_forcing_probability * teacher_state
                    )
                hidden = controller.commit_state_probabilities(
                    recurrent_input,
                    state_slot_start=state_slot_start,
                    probabilities=next_state_probabilities,
                )
                state_probabilities = next_state_probabilities
                active = (
                    mx.ones((int(hidden.shape[0]),), dtype=mx.bool_)
                    if prior_terminal_mask is None
                    else ~prior_terminal_mask
                )
                if not process_tape_lesion:
                    pre_state = recurrent_input[
                        :, state_slot_start:state_stop, :
                    ]
                    post_state = hidden[:, state_slot_start:state_stop, :]
                    _unified_recurrent_hidden_states_part_3(action_state, active, controller, hidden, iteration, post_state, pre_state, process_tape, process_tape_masks)
                if state_probability_trajectory is not None:
                    state_probability_trajectory.append(state_probabilities)
                # Process acquisition and answer emission are distinct graphs.
                # The typed transition depends on the prefix parser and prior
                # registers above; the candidate window/coda exists only to
                # emit language. Retaining it during process-only training used
                # tens of gigabytes while contributing no non-zero objective.
                if process_only:
                    candidate = hidden
                else:
                    candidate = _run(window, hidden)
                    candidate = controller.correction(candidate, iteration)
                    candidate = controller.attend_answer_to_state(
                        candidate,
                        hidden,
                        state_slot_start=state_slot_start,
                        state_probabilities=state_probabilities,
                        process_memory=(
                            mx.concatenate(process_tape, axis=1)
                            if process_tape
                            else None
                        ),
                        process_memory_mask=(
                            mx.concatenate(process_tape_masks, axis=1)
                            if process_tape_masks
                            else None
                        ),
                        role_logit_trajectory=answer_role_logit_trajectory,
                        place_logit_trajectory=answer_place_logit_trajectory,
                        binding_feature_trajectory=answer_binding_feature_trajectory,
                    )
                if prior_terminal_mask is not None:
                    # A completed program stutters semantically as well as in
                    # its typed registers. Re-running the transformer window
                    # after termination used to move an already-correct answer
                    # representation even though the machine state was fixed.
                    # Preserve the associated neural emission policy too;
                    # keeping only the hidden state let later no-op iterations
                    # overwrite role/place logits and corrupt deep decoding.
                    candidate = mx.where(
                        prior_terminal_mask[:, None, None],
                        last_decode_state,
                        candidate,
                    )
                    for binding_trajectory in (
                        answer_role_logit_trajectory,
                        answer_place_logit_trajectory,
                    ):
                        if (
                            binding_trajectory is not None
                            and len(binding_trajectory) >= 2
                        ):
                            binding_trajectory[-1] = mx.where(
                                prior_terminal_mask[:, None, None],
                                binding_trajectory[-2],
                                binding_trajectory[-1],
                            )
            if state_logit_trajectory is not None:
                state_logit_trajectory.append(state_logits)
            if action_logit_trajectory is not None:
                action_logit_trajectory.append(action_logits)
            last_decode_state = candidate
            if decode_state_trajectory is not None:
                decode_state_trajectory.append(candidate)
            probability = controller.halt_probability(prior_state, hidden)
            transport_gate = mx.array(1.0, dtype=mx.float32)
            mx.eval(hidden, candidate, probability, transport_gate)
            halt_probabilities.append(float(probability.item()))
            memory_write_means.append(0.0)
            transport_gates.append(1.0)
            trajectory.append(hidden)
            if (
                adaptive_halt
                and iteration + 1 >= controller.config.minimum_iterations
                and halt_probabilities[-1] >= controller.config.halt_threshold
            ):
                halted = True
                halt_reason = "learned_threshold"
                break
            continue

        prior_state = hidden
        if iteration > 0:
            if plan.anchor_injection > 0.0:
                hidden = hidden + plan.anchor_injection * anchor
            if plan.interpass_noise > 0.0:
                key = mx.random.key(plan.noise_seed * 1_000_003 + iteration)
                kick = mx.random.normal(hidden.shape, key=key).astype(mx.float32)
                hidden = hidden + (
                    kick * (plan.interpass_noise * _rms(hidden))
                ).astype(hidden.dtype)
            if plan.renormalize:
                hidden = hidden * (anchor_rms / _rms(hidden)).astype(hidden.dtype)
        recurrent_input = hidden
        if recurrent_input_trajectory is not None:
            recurrent_input_trajectory.append(recurrent_input)
        with recurrent_iteration(iteration):
            candidate = _run(
                window,
                recurrent_input,
                caches["window"][iteration] if caches else None,
            )
            candidate = controller.correction(candidate, iteration)
            candidate, transport_gate = controller.transport(
                prior_state,
                candidate,
                iteration,
            )

        if memory_layout is not None and iteration > 0:
            probabilities = controller.memory_write_probabilities(
                prior_state,
                candidate,
            )
            if soft_memory_writes:
                gates = probabilities
            else:
                gates = mx.where(
                    probabilities >= controller.config.memory_write_threshold,
                    probabilities,
                    mx.zeros_like(probabilities),
                )
            hidden, applied = apply_protected_transition(
                prior_state,
                candidate,
                memory_layout,
                write_gate=gates,
            )
            memory_write_means.append(float(mx.mean(applied)))
        else:
            hidden = candidate
            memory_write_means.append(0.0)

        probability = controller.halt_probability(prior_state, hidden)
        mx.eval(hidden, probability, transport_gate)
        halt_probabilities.append(float(probability.item()))
        transport_gates.append(float(mx.mean(transport_gate).item()))
        trajectory.append(hidden)
        if (
            adaptive_halt
            and iteration + 1 >= controller.config.minimum_iterations
            and halt_probabilities[-1] >= controller.config.halt_threshold
        ):
            halted = True
            halt_reason = "learned_threshold"
            break

    final_source = last_decode_state if last_decode_state is not None else hidden
    if process_only:
        final = final_source
    else:
        final = _run(
            layers[plan.coda_start :],
            final_source,
            caches["coda"] if caches else None,
        )
        final = inner.norm(final)
    retention = None
    residuals: tuple[float, ...] = ()
    if memory_layout is not None and trajectory:
        retention = memory_retention(trajectory[0], trajectory[-1], memory_layout)
        residuals = tuple(semantic_convergence(trajectory, memory_layout))
    telemetry = UnifiedRecurrenceTelemetry(
        configured_iterations=plan.iterations,
        executed_iterations=len(trajectory),
        halt_probabilities=tuple(halt_probabilities),
        memory_write_means=tuple(memory_write_means),
        transport_gates=tuple(transport_gates),
        halted=halted,
        halt_reason=halt_reason,
        memory_retention=retention,
        semantic_residuals=residuals,
        process_tape_entries=sum(int(value.shape[1]) for value in process_tape),
        process_tape_active_entries=(
            int(mx.sum(mx.concatenate(process_tape_masks, axis=1)).item())
            if process_tape_masks
            else 0
        ),
        controller_sha256=controller.parameter_sha256(),
    )
    return final, trajectory, telemetry


def apply_terminal_answer_grammar(
    logits: Any,
    tokens: Any,
    *,
    state_slot_start: int,
    state_probabilities: Any,
    contract: RecurrentAnswerEmissionContract,
) -> Any:
    """Constrain only canonical syntax around neurally emitted answer digits."""

    if (
        len(logits.shape) != 3
        or len(tokens.shape) != 2
        or len(state_probabilities.shape) != 3
        or int(logits.shape[0]) != int(tokens.shape[0])
        or int(logits.shape[0]) != int(state_probabilities.shape[0])
        or state_probabilities.shape[1:] != (
            len(STATE_SLOT_NAMES),
            STATE_CARDINALITY,
        )
        or type(state_slot_start) is not int
        or not 0 <= state_slot_start <= int(tokens.shape[1])
    ):
        raise ValueError("terminal answer grammar tensor layout differs")
    rows: list[Any] = []
    vocabulary = mx.arange(int(logits.shape[-1]))
    state_rows = mx.argmax(state_probabilities, axis=-1).tolist()
    token_rows = tokens.tolist()
    for batch_index, state_values in enumerate(state_rows):
        public_tokens = token_rows[batch_index][:state_slot_start]
        generated_tokens = token_rows[batch_index][state_slot_start:]
        template = contract.emission_template(public_tokens, state_values)
        row = logits[batch_index, -1, :]
        if template is not None:
            forced_token = contract.next_template_token(
                public_tokens,
                state_values,
                generated_tokens,
            )
            if forced_token is None:
                digit_tokens = mx.array(contract.digit_token_ids)
                allowed = mx.any(
                    digit_tokens[:, None] == vocabulary[None, :],
                    axis=0,
                )
                row = mx.where(allowed, row, -1e9).astype(logits.dtype)
            else:
                if forced_token >= int(logits.shape[-1]):
                    raise ValueError("answer grammar token is outside the vocabulary")
                row = mx.where(vocabulary == forced_token, 0.0, -1e9).astype(
                    logits.dtype
                )
        rows.append(row)
    constrained = mx.stack(rows, axis=0)
    return mx.concatenate([logits[:, :-1, :], constrained[:, None, :]], axis=1)


def unified_recurrent_logits(
    model: Any,
    tokens: Any,
    plan: RecurrentDepthPlan,
    controller: UnifiedRecurrentController,
    **kwargs: Any,
) -> tuple[Any, UnifiedRecurrenceTelemetry]:
    answer_emission_contract = kwargs.pop("answer_emission_contract", None)
    answer_digit_pointer_enabled = kwargs.pop("answer_digit_pointer_enabled", True)
    if answer_emission_contract is not None and not isinstance(
        answer_emission_contract, RecurrentAnswerEmissionContract
    ):
        raise TypeError("answer emission contract is invalid")
    if type(answer_digit_pointer_enabled) is not bool:
        raise TypeError("answer digit pointer flag must be bool")
    state_slot_start = kwargs.get("state_slot_start")
    pointer_enabled = (
        answer_digit_pointer_enabled
        and state_slot_start is not None
        and bool(controller.config.literal_digit_token_ids)
    )
    state_probabilities: list[Any] = kwargs.setdefault(
        "state_probability_trajectory", []
    )
    role_logits: list[Any] = kwargs.setdefault("answer_role_logit_trajectory", [])
    place_logits: list[Any] = kwargs.setdefault("answer_place_logit_trajectory", [])
    hidden, _trajectory, telemetry = unified_recurrent_hidden_states(
        model,
        tokens,
        plan,
        controller,
        **kwargs,
    )
    if getattr(model, "lm_head", None) is not None:
        logits = model.lm_head(hidden)
    else:
        logits = model.model.embed_tokens.as_linear(hidden)
    if pointer_enabled:
        if not state_probabilities or not role_logits or not place_logits:
            raise RuntimeError("answer digit pointer emitted no recurrent trajectory")
        answer_start = int(state_slot_start) + controller.config.state_slots - 1
        pointed = controller.apply_answer_digit_pointer(
            logits[:, answer_start:, :],
            role_logits[-1],
            place_logits[-1],
            state_probabilities[-1],
        )
        logits = mx.concatenate([logits[:, :answer_start, :], pointed], axis=1)
    if answer_emission_contract is not None:
        if state_slot_start is None or not state_probabilities:
            raise RuntimeError("answer grammar emitted no recurrent typed state")
        if (
            tuple(answer_emission_contract.digit_token_ids)
            != tuple(controller.config.literal_digit_token_ids)
        ):
            raise RuntimeError("answer grammar and neural digit contracts differ")
        logits = apply_terminal_answer_grammar(
            logits,
            tokens,
            state_slot_start=int(state_slot_start),
            state_probabilities=state_probabilities[-1],
            contract=answer_emission_contract,
        )
    return logits, telemetry


__all__ = [
    "ACTION_LITERAL_BINDING_PARAMETER_NAMES",
    "ACTION_LITERAL_BINDING_TRANSFORMS",
    "ACTION_WORKSPACE_PARAMETER_NAMES",
    "CAUSAL_ACTION_PARAMETER_NAMES",
    "FAMILY_ACTION_PARAMETER_NAMES",
    "FRONTIER_ACTION_EXPERT_COUNT",
    "MAX_PROCESS_INTEGER",
    "PROCESS_RADIX",
    "TRANSITION_EXECUTION_DEPENDENCY_PARAMETER_NAMES",
    "TRANSITION_MEMORY_PARAMETER_NAMES",
    "TRANSITION_TAPE_READER_PARAMETER_NAMES",
    "TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES",
    "TRANSITION_COPY_PRIOR_LOGIT_BIAS",
    "TRANSITION_PROCESSOR_MODES",
    "TRANSITION_PROCESSOR_PARAMETER_NAMES",
    "TRANSITION_REPLAY_MODES",
    "TRANSITION_REPLAY_PARAMETER_NAMES",
    "UNIFIED_INTRINSIC_RECURRENCE_SCHEMA",
    "UnifiedRecurrenceConfig",
    "UnifiedRecurrenceTelemetry",
    "UnifiedRecurrentController",
    "apply_terminal_answer_grammar",
    "unified_recurrent_hidden_states",
    "unified_recurrent_logits",
]

"""The registers, tapes and heads this controller is built from.

Lifted whole out of `unified_intrinsic_recurrence`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .unified_intrinsic_recurrence import (
        UnifiedRecurrenceConfig,
    )


class _BuildsItsRegisters:
    """Lifted whole out of UnifiedRecurrentController; see unified_intrinsic_recurrence.py."""

    def _init_correction_and_halt(
        self,
        config: UnifiedRecurrenceConfig,
        scale: Any,
        key_a: Any,
        key_depth: Any,
        key_memory: Any,
        key_halt: Any,
    ) -> None:
        from .unified_intrinsic_recurrence import (
            mx,
        )

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
        scale: Any,
        key_state: Any,
        key_state_slots: Any,
        key_state_values: Any,
        key_transition_query: Any,
        key_transition_key: Any,
        key_transition_value: Any,
        key_transition_self: Any,
        key_transition_output: Any,
        key_transition_depth: Any,
        key_action_slots: Any,
        key_action_values: Any,
        key_action_query: Any,
        key_action_key: Any,
        key_action_value: Any,
        key_action_output: Any,
        key_action_depth: Any,
        key_state_action: Any,
    ) -> None:
        from .unified_intrinsic_recurrence import (
            mx,
        )

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

    def _init_transition_memory(self, config: UnifiedRecurrenceConfig, scale: Any) -> None:
        from .unified_intrinsic_recurrence import (
            mx,
        )

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
        from .unified_intrinsic_recurrence import (
            mx,
        )

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
        from .unified_intrinsic_recurrence import (
            mx,
        )

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
        from .unified_intrinsic_recurrence import (
            mx,
        )

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
        from .unified_intrinsic_recurrence import (
            FRONTIER_ACTION_EXPERT_COUNT,
            mx,
        )

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
        workspace_width: Any,
        workspace_scale: Any,
        key_literal_values: Any,
        scale: Any,
        key_answer_query: Any,
        key_answer_key: Any,
        key_answer_value: Any,
        key_answer_output: Any,
    ) -> None:
        from .unified_intrinsic_recurrence import (
            ACTION_LITERAL_BINDING_TRANSFORMS,
            FRONTIER_ACTION_EXPERT_COUNT,
            LITERAL_MAX_VALUE,
            mx,
        )

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

    def _init_process_reader_and_answer_gates(
        self,
        config: UnifiedRecurrenceConfig,
        scale: Any,
    ) -> None:
        from .unified_intrinsic_recurrence import (
            mx,
        )

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


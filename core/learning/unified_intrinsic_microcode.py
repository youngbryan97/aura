"""The microcode opcodes a transition can take, and what each one does.

Five helpers of one function, called only by it, holding the per-opcode
logit rules. They are the part of the controller that changes when an
opcode changes rather than when the recurrence does.


Lifted whole out of `unified_intrinsic_recurrence`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _microcode_transition_logits_modulus(
    arg0: Any,
    arg1: Any,
    arg2: Any,
    arg3: Any,
    arg4: Any,
    arg5: Any,
    opcode: Any,
    state: Any,
    value0: Any,
) -> tuple[Any, Any]:
    from .unified_intrinsic_recurrence import (
        OP_ADD_MOD,
        OP_BOOL_AND,
        OP_BOOL_NOT,
        OP_BOOL_OR,
        OP_BOOL_XOR,
        OP_COPY_VALUE,
        OP_MUL_MOD,
        OP_SUB_MOD,
        mx,
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
    return register_result, value0

def _microcode_transition_logits_is_simulate(
    self,
    action_probability_history: Any,
    arg0: Sequence[Any] | None,
    arg1: Any,
    arg2: Any,
    opcode: Any,
    value0: Any,
    values: Any,
) -> tuple[Any, bool]:
    from .unified_intrinsic_recurrence import (
        OP_FRONTIER_INFER,
        OP_FRONTIER_SIMULATE,
        PROCESS_RADIX,
        mx,
    )

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
    return inferred_role, is_infer

def _microcode_transition_logits_calibration_input(
    action_probability_history: Sequence[Any] | None,
    is_calibrate: bool,
    state: Any,
    values: list[Any],
) -> Any:
    from .unified_intrinsic_recurrence import (
        PROCESS_RADIX,
        mx,
    )

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
    return percentage

def _microcode_transition_logits_is_audit(
    arg0: Any,
    arg1: Any,
    arg2: Any,
    arg3: Any,
    arg4: Any,
    opcode: Any,
    state: Any,
    values: list[Any],
) -> tuple[Any, Any]:
    from .unified_intrinsic_recurrence import (
        OP_FRONTIER_AUDIT,
        PROCESS_RADIX,
        mx,
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
    return candidate_encoded, candidate_score

def _microcode_transition_logits_op_causal_chain(
    arg0: Any,
    arg1: Any,
    arg2: Any,
    arg3: Any,
    arg4: Any,
    arg5: Any,
    opcode: Any,
    semantic_invalid: Any,
    values: list[Any],
) -> Any:
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
    from .unified_intrinsic_recurrence import (
        ACTION_NULL,
        OP_CAUSAL_CHAIN,
        PROCESS_RADIX,
        mx,
    )

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
    return semantic_invalid


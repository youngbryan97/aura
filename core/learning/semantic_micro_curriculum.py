"""Deterministic operand curriculum for semantic recurrent micro-operations.

The curriculum contains only committed categorical states and public
instructions.  It does not carry expected next states or answers.  Training
authority is supplied separately by the exact instruction semantics, while
fresh seeds and disjoint operand residues provide held-out combinations.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Final

from core.learning.recurrent_action_schema import (
    ACTION_NULL,
    ACTION_SLOT_NAMES,
    OP_CAUSAL_CHAIN,
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
    OP_SET_SCALAR,
    OP_SIGNED_PAIR_ADD_IMMEDIATE,
    OP_SIGNED_RANKED_GREATER,
    SEMANTIC_MICRO_ACTION_FIELD_NAMES,
    SEMANTIC_MICRO_OPCODES,
    canonical_instruction_from_public_fields,
)
from core.learning.recurrent_state_schema import SEMANTIC_STATE_SLOT_NAMES

SEMANTIC_MICRO_CURRICULUM_SCHEMA: Final = "aura.semantic_micro_curriculum.v1"
PROCESS_RADIX: Final = 31
MAX_PROCESS_INTEGER: Final = PROCESS_RADIX**2 - 1
_PAIR_ADDRESSES: Final = tuple(range(8))
_DISJOINT_PAIR_ADDRESSES: Final = (0, 2, 4, 6)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _signed_encode(value: int) -> int:
    encoded = 2 * value if value >= 0 else (-2 * value) - 1
    if not 0 <= encoded <= MAX_PROCESS_INTEGER:
        raise ValueError("semantic signed value exceeds the register pair")
    return encoded


def _write_pair(values: list[int], low: int, value: int) -> None:
    if not 0 <= low < len(values) - 1 or not 0 <= value <= MAX_PROCESS_INTEGER:
        raise ValueError("semantic primitive pair value is invalid")
    values[low] = value % PROCESS_RADIX
    values[low + 1] = value // PROCESS_RADIX


def _raw(opcode: int, *arguments: int) -> tuple[int, ...]:
    if (
        opcode not in SEMANTIC_MICRO_OPCODES
        or len(arguments) > 6
        or any(type(value) is not int or not 0 <= value < ACTION_NULL for value in arguments)
    ):
        raise ValueError("semantic primitive instruction is invalid")
    return (opcode, *arguments, *(0 for _index in range(6 - len(arguments))))


@dataclass(frozen=True, slots=True)
class SemanticMicroExample:
    """One answer-blind local transition input."""

    opcode: int
    state_values: tuple[int, ...]
    action_values: tuple[int, ...]
    sample_index: int

    def __post_init__(self) -> None:
        if (
            self.opcode not in SEMANTIC_MICRO_OPCODES
            or len(self.state_values) != len(SEMANTIC_STATE_SLOT_NAMES)
            or self.state_values[-1] != 0
            or any(type(value) is not int or not 0 <= value <= ACTION_NULL for value in self.state_values)
            or len(self.action_values) != len(ACTION_SLOT_NAMES)
            or self.action_values[0] != self.opcode
            or any(type(value) is not int or not 0 <= value <= ACTION_NULL for value in self.action_values)
            or type(self.sample_index) is not int
            or self.sample_index < 0
        ):
            raise ValueError("semantic micro example differs from its schema")

    @property
    def example_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": SEMANTIC_MICRO_CURRICULUM_SCHEMA,
                "opcode": self.opcode,
                "state_values": self.state_values,
                "action_values": self.action_values,
                "sample_index": self.sample_index,
            }
        )


def _primitive_example(opcode: int, sample_index: int, seed: int) -> SemanticMicroExample:
    rng = random.Random((seed << 32) ^ (opcode << 20) ^ sample_index)
    values = [rng.randrange(PROCESS_RADIX) for _slot in range(9)]
    destination = _PAIR_ADDRESSES[sample_index % len(_PAIR_ADDRESSES)]
    left_slot = _DISJOINT_PAIR_ADDRESSES[
        sample_index % len(_DISJOINT_PAIR_ADDRESSES)
    ]
    right_slot = _DISJOINT_PAIR_ADDRESSES[
        (sample_index + 1) % len(_DISJOINT_PAIR_ADDRESSES)
    ]

    if opcode == OP_PAIR_SET:
        action = _raw(opcode, destination, rng.randrange(PROCESS_RADIX), rng.randrange(PROCESS_RADIX))
    elif opcode == OP_PAIR_ADD:
        left = rng.randrange(MAX_PROCESS_INTEGER + 1)
        right = rng.randrange(MAX_PROCESS_INTEGER - left + 1)
        _write_pair(values, left_slot, left)
        _write_pair(values, right_slot, right)
        action = _raw(opcode, destination, left_slot, right_slot)
    elif opcode == OP_PAIR_MUL_IMMEDIATE:
        multiplier = rng.randrange(PROCESS_RADIX)
        current = rng.randrange(MAX_PROCESS_INTEGER // max(multiplier, 1) + 1)
        _write_pair(values, destination, current)
        action = _raw(opcode, destination, multiplier)
    elif opcode == OP_PAIR_SUB_IMMEDIATE:
        immediate = rng.randrange(PROCESS_RADIX)
        current = rng.randrange(immediate, MAX_PROCESS_INTEGER + 1)
        _write_pair(values, destination, current)
        action = _raw(opcode, destination, immediate)
    elif opcode == OP_PAIR_SIGNED_SUB_IMMEDIATE:
        immediate = rng.randrange(PROCESS_RADIX)
        current = rng.randrange(481)
        _write_pair(values, destination, current)
        action = _raw(opcode, destination, immediate)
    elif opcode == OP_PAIR_DIV:
        denominator = rng.randrange(1, PROCESS_RADIX)
        quotient = rng.randrange(PROCESS_RADIX)
        _write_pair(values, left_slot, denominator * quotient)
        _write_pair(values, right_slot, denominator)
        action = _raw(opcode, destination, left_slot, right_slot)
    elif opcode in {OP_RATIO_CHOICE, OP_RATIO_BAND}:
        denominator = rng.randrange(1, MAX_PROCESS_INTEGER + 1)
        numerator = rng.randrange(denominator + 1)
        _write_pair(values, left_slot, numerator)
        _write_pair(values, right_slot, denominator)
        scalar_destination = sample_index % len(values)
        action = _raw(opcode, scalar_destination, left_slot, right_slot)
    elif opcode == OP_SIGNED_PAIR_ADD_IMMEDIATE:
        current = rng.randrange(-420, 421)
        delta = rng.randrange(-15, 16)
        _write_pair(values, destination, _signed_encode(current))
        action = _raw(opcode, destination, _signed_encode(delta))
    elif opcode == OP_SIGNED_RANKED_GREATER:
        candidate = rng.randrange(-300, 301)
        incumbent = rng.randrange(-300, 301)
        _write_pair(values, left_slot, _signed_encode(candidate))
        _write_pair(values, right_slot, _signed_encode(incumbent))
        destination_scalar = sample_index % len(values)
        incumbent_rank_slot = (sample_index + 3) % len(values)
        has_incumbent_slot = (sample_index + 5) % len(values)
        values[incumbent_rank_slot] = rng.randrange(PROCESS_RADIX)
        values[has_incumbent_slot] = rng.randrange(2)
        action = _raw(
            opcode,
            destination_scalar,
            left_slot,
            right_slot,
            rng.randrange(PROCESS_RADIX),
            incumbent_rank_slot,
            has_incumbent_slot,
        )
    elif opcode == OP_RANKED_COMMIT:
        flag_slot = sample_index % len(values)
        values[flag_slot] = sample_index % 2
        candidate = rng.randrange(MAX_PROCESS_INTEGER + 1)
        _write_pair(values, left_slot, candidate)
        action = _raw(
            opcode,
            flag_slot,
            left_slot,
            rng.randrange(PROCESS_RADIX),
            rng.randrange(PROCESS_RADIX),
        )
    elif opcode == OP_SET_SCALAR:
        action = _raw(opcode, sample_index % len(values), rng.randrange(PROCESS_RADIX))
    elif opcode == OP_PAIR_COPY:
        _write_pair(values, left_slot, rng.randrange(MAX_PROCESS_INTEGER + 1))
        action = _raw(opcode, destination, left_slot)
    elif opcode == OP_PAIR_EUCLID_STEP:
        left = rng.randrange(MAX_PROCESS_INTEGER + 1)
        right = rng.randrange(1, MAX_PROCESS_INTEGER + 1)
        _write_pair(values, left_slot, left)
        _write_pair(values, right_slot, right)
        action = _raw(opcode, left_slot, right_slot)
    elif opcode == OP_PAIR_PRODUCT:
        action = _raw(opcode, destination, rng.randrange(PROCESS_RADIX), rng.randrange(PROCESS_RADIX))
    elif opcode == OP_CAUSAL_CHAIN:
        # The FIRST edge of the chain, which is the only step of it that is a
        # local transition. The rest of the protocol reads what the earlier
        # edges wrote in slot 8, so a sample of one of those would be a sample
        # of a state this curriculum never sets up.
        #
        # OP_CAUSAL_CHAIN was added to SEMANTIC_MICRO_OPCODES and not here, so
        # the generator reached its own "exhaustive opcode set is checked
        # above" branch and raised — which is the comment being false rather
        # than the opcode being unsupported.
        values[:] = [rng.randrange(PROCESS_RADIX) for _slot in range(8)] + [0]
        change_low = rng.randrange(PROCESS_RADIX)
        change_high = rng.randrange(PROCESS_RADIX)
        action = _raw(
            opcode,
            0,
            1,
            rng.randrange(PROCESS_RADIX),
            change_high,
            change_low,
            0,
        )
    else:  # pragma: no cover - exhaustive opcode set is checked above.
        raise ValueError("semantic primitive opcode is unsupported")

    terminal = sample_index % 7 == 0
    canonical = canonical_instruction_from_public_fields(
        "frontier_calibration",
        SEMANTIC_MICRO_ACTION_FIELD_NAMES,
        action,
        step=0,
        terminal=int(terminal),
    )
    state = (sample_index % 16, *values, 0)
    return SemanticMicroExample(
        opcode=opcode,
        state_values=state,
        action_values=canonical,
        sample_index=sample_index,
    )


def semantic_micro_batch(
    *,
    seed: int,
    batch_size: int,
    batch_index: int,
) -> tuple[SemanticMicroExample, ...]:
    """Return a deterministic balanced batch with disjoint seedable operands."""

    opcodes = tuple(sorted(SEMANTIC_MICRO_OPCODES))
    if (
        type(seed) is not int
        or seed < 0
        or type(batch_size) is not int
        or batch_size < len(opcodes)
        or type(batch_index) is not int
        or batch_index < 0
    ):
        raise ValueError("semantic primitive batch coordinates are invalid")
    start = batch_index * batch_size
    return tuple(
        _primitive_example(
            opcodes[(start + offset) % len(opcodes)],
            start + offset,
            seed,
        )
        for offset in range(batch_size)
    )


def semantic_micro_batch_receipt(
    examples: tuple[SemanticMicroExample, ...],
) -> dict[str, object]:
    if not examples:
        raise ValueError("semantic primitive receipt requires examples")
    body: dict[str, object] = {
        "schema": SEMANTIC_MICRO_CURRICULUM_SCHEMA,
        "examples": len(examples),
        "opcodes": sorted({example.opcode for example in examples}),
        "example_sha256s": [example.example_sha256 for example in examples],
        "answers_present": False,
        "expected_states_present": False,
    }
    return {**body, "receipt_sha256": _canonical_sha256(body)}


__all__ = [
    "SEMANTIC_MICRO_CURRICULUM_SCHEMA",
    "SemanticMicroExample",
    "semantic_micro_batch",
    "semantic_micro_batch_receipt",
]

"""Exact AVM1 addresses qualify branches; incomplete streams prove nothing."""
from __future__ import annotations

import struct
from dataclasses import FrozenInstanceError

import pytest

from core.perception.flash_actions_with_source import _action_address_invariant, parse_action_stream


def _action(code: int, payload: bytes = b"") -> bytes:
    return bytes((code,)) + (struct.pack("<H", len(payload)) + payload if code >= 0x80 else b"")


def _push(*values: str) -> bytes:
    return _action(0x96, b"".join(b"\0" + value.encode() + b"\0" for value in values))


def _function(name: str = "", body: bytes = b"\0", params: tuple[str, ...] = ()) -> bytes:
    header = name.encode() + b"\0" + struct.pack("<H", len(params))
    header += b"".join(param.encode() + b"\0" for param in params) + struct.pack("<H", len(body))
    return _action(0x9B, header) + body


@pytest.mark.parametrize('value', [0, 1])
def test_boolean_payload_retains_its_type_beside_equal_integer_values(value):
    got = parse_action_stream(_action(0x96, b'\x05'+bytes((value,))+b'\x07'+struct.pack('<i', value))+b'\0')
    assert got.complete
    flag, number = got.instructions[0].operands
    assert flag.kind == 'boolean' and type(flag.value) is bool and flag.value == bool(value)
    assert number.kind == 'integer' and type(number.value) is int and number.value == value


def test_a_nonboolean_byte_cannot_be_coerced_to_a_true_payload():
    got = parse_action_stream(_action(0x96, b'\x05\x02')+b'\0')
    assert not got.complete and any('invalid boolean' in issue.reason for issue in got.issues)


def test_branch_addresses_use_the_end_of_the_encoded_record() -> None:
    data = _action(0x9D, struct.pack("<h", 1)) + b"\x06\x07\0"
    got = parse_action_stream(data, source="frame-at-80", base_offset=80)
    assert got.complete
    first, play, stop, end = got.instructions
    assert (first.offset, first.payload_offset, first.payload_end, first.next_offset) == (80, 83, 85, 85)
    assert first.branch_target == stop.offset == 86
    assert first.successors == (play.offset, stop.offset)
    assert play.successors == (stop.offset,)
    assert stop.successors == (end.offset,)
    assert end.successors == ()
    assert first.payload == struct.pack("<h", 1)
    with pytest.raises(FrozenInstanceError):
        first.offset = 0


def test_backward_branch_retains_a_loop_without_unrolling_it() -> None:
    got = parse_action_stream(b"\x06" + _action(0x99, struct.pack("<h", -6)) + b"\0")
    assert got.complete
    assert got.instructions[1].branch_target == 0
    assert got.instructions[1].successors == (0,)


@pytest.mark.parametrize("delta", [-20, 2, 8])
def test_a_branch_cannot_target_payload_bytes_or_leave_its_stream(delta: int) -> None:
    got = parse_action_stream(_action(0x99, struct.pack("<h", delta)) + _push("x") + b"\0")
    assert not got.complete
    assert got.assignments == ()
    assert "boundary" in got.issues[0].reason


def test_parent_flow_skips_a_function_body_and_body_flow_keeps_its_addresses() -> None:
    body = b"\x06" + _action(0x99, struct.pack("<h", -6))
    data = _function("step", body, ("amount",)) + b"\x07\0"
    got = parse_action_stream(data, base_offset=20)
    assert got.complete
    function = got.functions[0]
    definition, stop, _end = got.instructions
    assert definition.next_offset == function.end_offset == stop.offset
    assert definition.payload_end == function.body_offset
    assert function.parameters == ("amount",)
    assert function.body.instructions[1].branch_target == function.body_offset
    assert definition.successors == (stop.offset,)
    assert got.assignments[0].target == ("step",)


def test_a_parent_jump_cannot_enter_a_function_body() -> None:
    function = _function("step", b"\x06\0")
    body_at = 5 + len(function) - 2
    got = parse_action_stream(_action(0x99, struct.pack("<h", body_at - 5)) + function + b"\0")
    assert not got.complete
    assert got.instructions[0].successors == ()


def test_anonymous_function_keeps_a_qualified_prototype_assignment() -> None:
    data = _push("Engine") + b"\x1c" + _push("prototype") + b"\x4e" + _push("advance")
    data += _function(body=b"\x07\0") + b"\x4f\0"
    got = parse_action_stream(data)
    assert got.complete
    assignment = got.assignments[0]
    assert assignment.target == ("Engine", "prototype", "advance")
    assert assignment.function_offset == got.functions[0].definition_offset
    assert assignment.instruction_offset == got.instructions[-2].offset


def test_constant_pool_and_register_targets_remain_source_derived() -> None:
    pool = _action(0x88, struct.pack("<H", 2) + b"Engine\0advance\0")
    data = pool + _action(0x96, b"\x08\0") + b"\x1c" + _action(0x87, b"\x01") + b"\x17"
    data += _action(0x96, b"\x04\x01\x08\x01") + _function() + b"\x4f\0"
    got = parse_action_stream(data)
    assert got.complete
    assert got.assignments[0].target == ("Engine", "advance")
    assert got.instructions[1].operands[0].kind == "constant"
    assert got.instructions[1].operands[0].value == 0


def test_function2_keeps_parameter_registers_and_preload_flags() -> None:
    body = _action(0x96, b"\x04\x02") + b"\x3e"
    header = b"advance\0" + struct.pack("<H", 1) + b"\x03" + struct.pack("<H", 0x41)
    header += b"\x02amount\0" + struct.pack("<H", len(body))
    got = parse_action_stream(_action(0x8E, header) + body + b"\0")
    assert got.complete
    function = got.functions[0]
    assert function.parameters == ("amount",)
    assert function.register_parameters == ((2, "amount"),)
    assert function.register_count == 3 and function.flags == 0x41


def test_wait_skip_uses_action_count_rather_than_byte_count() -> None:
    wait = _action(0x8A, struct.pack("<H", 2) + b"\x01")
    got = parse_action_stream(wait + _push("a much longer action") + b"\x06\0")
    assert got.complete
    assert got.instructions[0].successors == (got.instructions[1].offset, got.instructions[2].offset)


@pytest.mark.parametrize("data", [
    b"\x96", b"\x96\x02", b"\x96\x02\0\0", _action(0x96, b"\0unterminated"),
    _action(0x96, b"\x01\0"), _action(0x96, b"\xff"), _action(0x99, b"\0"),
    _action(0x88, b"\x01\0missing"), _action(0x94, b"\0\0"), b"\x70",
    _function()[:-1], b"\0\x06",
])
def test_truncation_unknown_instructions_and_hidden_trailing_code_are_unqualified(data: bytes) -> None:
    got = parse_action_stream(data)
    assert not got.complete
    assert got.assignments == ()


def test_incomplete_nested_function_invalidates_parent_ownership() -> None:
    got = parse_action_stream(_push("step") + _function(body=b"\x70") + b"\x1d\0")
    assert not got.complete
    assert got.assignments == ()
    assert not got.functions[0].body.complete


def test_assignment_values_are_not_carried_across_a_branch_join() -> None:
    prefix = _push("Engine") + b"\x1c" + _push("advance") + _function()
    branch = _action(0x9D, struct.pack("<h", 0))
    got = parse_action_stream(prefix + branch + b"\x4f\0")
    assert got.complete
    assert got.assignments == ()


def test_registered_address_invariant_checks_valid_and_invalid_branches() -> None:
    assert _action_address_invariant() == ()

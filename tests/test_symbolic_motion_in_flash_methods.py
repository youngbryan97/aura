"""Actual byte-addressed branch paths, not names, select the arithmetic."""
import struct
from dataclasses import replace

import pytest

from core.perception.flash_actions_with_source import parse_action_stream
from core.perception.symbolic_motion_in_flash_methods import (
    Arithmetic,
    _source_effect_prefix_invariant,
    bind_flash_methods,
    read_flash_method,
)


def _action(code, payload=b''):
    return bytes((code,)) + (struct.pack('<H', len(payload)) + payload if code >= 0x80 else b'')


def _push(*values):
    payload = b''
    for value in values:
        if isinstance(value, str):
            payload += b'\0' + value.encode() + b'\0'
        elif isinstance(value, bool):
            payload += b'\x05' + bytes((value,))
        elif value is None:
            payload += b'\x02'
        else:
            payload += b'\x07' + struct.pack('<i', value)
    return _action(0x96, payload)


def _function(body, params=()):
    header = b'\0' + struct.pack('<H', len(params))
    header += b''.join(param.encode() + b'\0' for param in params) + struct.pack('<H', len(body))
    return _action(0x9B, header) + body


def _assemble(parts):
    labels, offset = {}, 0
    for part in parts:
        if isinstance(part, tuple) and part[0] == 'label':
            labels[part[1]] = offset
        else:
            offset += 5 if isinstance(part, tuple) else len(part)
    data = b''
    for part in parts:
        if isinstance(part, tuple):
            if part[0] != 'label':
                data += _action(part[0], struct.pack('<h', labels[part[1]] - len(data) - 5))
        else:
            data += part
    return data


def _binding(body, params=(), before=b'', after=b''):
    parent = before + _push('Opaque') + b'\x1c' + _push('q') + _function(body, params) + after + b'\x4f\0'
    stream = parse_action_stream(parent, source='opaque-source', base_offset=80)
    assert stream.complete
    result = bind_flash_methods((stream,))
    assert not result.issues
    return result.method('Opaque.q')


def _write(value):
    return _push('this') + b'\x1c' + _push('pending', value) + b'\x4f'


def test_bound_parameter_resolves_actual_branches_without_key_name_inference():
    body = _assemble([_push('choice') + b'\x1c' + _push(0) + b'\x49\x12',
                      (0x9D, 'other'), _write(-10), (0x99, 'end'),
                      ('label', 'other'), _write(10), ('label', 'end'), b'\0'])
    binding = _binding(body, ('choice',))
    left = read_flash_method(binding, (0,))
    right = read_flash_method(binding, (1,))
    assert left.effects_composed and right.effects_composed
    assert len(left.paths) == len(right.paths) == 1
    assert left.paths[0].writes[0].target == 'Opaque.pending'
    assert left.paths[0].writes[0].value == Arithmetic('number', (-10.,))
    assert right.paths[0].writes[0].value == Arithmetic('number', (10.,))


def test_unknown_body_guard_produces_blocked_and_reachable_paths():
    body = _assemble([_push('this') + b'\x1c' + _push('available') + b'\x4e\x12',
                      (0x9D, 'end'), _write(10), ('label', 'end'), b'\0'])
    reading = read_flash_method(_binding(body))
    assert reading.complete
    assert {len(path.writes) for path in reading.paths} == {0, 1}
    live = next(path for path in reading.paths if path.writes)
    assert live.conditions[0].taken is False
    assert live.conditions[0].condition == Arithmetic('not', (Arithmetic('reference', ('Opaque.available',)),))
    assert live.conditions[0].source == 'opaque-source'
    assert live.conditions[0].offset >= 80


def test_source_constants_resolve_enum_comparison():
    body = _assemble([_push('choice') + b'\x1c' + _push('Opaque') + b'\x1c' + _push('ZZ') + b'\x4e\x49\x12',
                      (0x9D, 'end'), _write(7), ('label', 'end'), b'\0'])
    binding = _binding(body, ('choice',))
    reading = read_flash_method(binding, (3,), constants={'Opaque.ZZ': 3.})
    assert reading.complete and len(reading.paths) == 1
    assert reading.paths[0].writes[0].value.arguments == (7.,)
    assert reading.paths[0].assumptions == (('Opaque.ZZ', 3.),)
    assert not reading.effects_composed


def test_dominating_definition_pool_survives_later_pool_change():
    first = _action(0x88, struct.pack('<H', 1) + b'pending\0')
    second = _action(0x88, struct.pack('<H', 1) + b'other\0')
    body = _push('this') + b'\x1c' + _action(0x96, b'\x08\0') + _push(9) + b'\x4f\0'
    binding = _binding(body, before=first, after=second)
    assert binding.constant_pool == ('pending',)
    reading = read_flash_method(binding)
    assert reading.complete
    assert reading.paths[0].writes[0].target == 'Opaque.pending'


def test_missing_defining_pool_does_not_substitute_an_invented_name():
    body = _action(0x96, b'\x08\0') + b'\x1c\x3e'
    result = read_flash_method(_binding(body))
    assert not result.complete
    assert any('constant pool' in issue for issue in result.issues)


def test_opaque_source_call_is_recorded_and_requires_composition():
    body = _push('this') + b'\x1c' + _push(1, 'Elsewhere') + b'\x1c' + _push('apply') + b'\x52\x17\0'
    result = read_flash_method(_binding(body))
    assert result.complete
    assert not result.effects_composed
    call = result.paths[0].calls[0]
    assert call.target == 'Elsewhere.apply'
    assert call.arguments == (Arithmetic('reference', ('Opaque',)),)


def test_unimplemented_instruction_is_not_a_complete_model():
    result = read_flash_method(_binding(b'\x07\0'))
    assert not result.complete
    assert 'unsupported symbolic instruction Stop' in result.issues[0]


def test_backward_branch_is_bounded_and_unqualified():
    body = _assemble([('label', 'again'), _push(True) + b'\x17', (0x99, 'again'), b'\0'])
    result = read_flash_method(_binding(body))
    assert not result.complete
    assert result.issues == ('symbolic step or loop budget exceeded',)


def test_wrong_argument_count_is_refused():
    result = read_flash_method(_binding(b'\0', ('choice',)))
    assert not result.complete
    assert result.paths == ()


@pytest.mark.parametrize('opcode', [0x0A, 0x0B, 0x0C, 0x0D, 0x3F])
def test_numeric_operations_are_actual_computations(opcode):
    body = _push(8, 3) + bytes((opcode, 0x3E))
    result = read_flash_method(_binding(body))
    assert result.complete
    expected = {0x0A: 11., 0x0B: 5., 0x0C: 24., 0x0D: 8 / 3, 0x3F: 2.}[opcode]
    assert result.paths[0].returned == Arithmetic('number', (expected,))


def test_abstract_equality_null_undefined_is_distinct_from_strict_equality():
    abstract = read_flash_method(_binding(_push(None) + _action(0x96, b'\x03') + b'\x49\x3e'))
    strict = read_flash_method(_binding(_push(None) + _action(0x96, b'\x03') + b'\x66\x3e'))
    assert abstract.complete and strict.complete
    assert abstract.paths[0].returned == Arithmetic('boolean', (True,))
    assert strict.paths[0].returned == Arithmetic('boolean', (False,))


def test_abstract_boolean_numeric_equality_is_not_a_false_branch():
    reading = read_flash_method(_binding(_push(False, 0) + b'\x49\x3e'))
    assert reading.complete
    assert reading.paths[0].returned == Arithmetic('boolean', (True,))


def test_unsupported_coercion_cannot_report_a_complete_false_result():
    reading = read_flash_method(_binding(_push('0', 0) + b'\x49\x3e'))
    assert not reading.complete
    assert any('outside its domain' in issue for issue in reading.issues)


def test_assigning_an_existing_local_is_not_a_write_to_the_world():
    body = _push('local', 1) + b'\x3c' + _push('local', 2) + b'\x1d' + _push('local') + b'\x1c\x3e'
    reading = read_flash_method(_binding(body))
    assert reading.effects_composed
    assert reading.paths[0].writes == ()
    assert reading.paths[0].returned == Arithmetic('number', (2.,))


def test_a_nonfinite_argument_is_not_replaced_by_undefined():
    reading = read_flash_method(_binding(_push('arg') + b'\x1c\x3e', ('arg',)), (float('nan'),))
    assert not reading.complete
    assert reading.issues == ('nonfinite source number is unsupported',)


def _notify():
    return _push(0, 'notify')+b'\x3d\x17'


def test_a_later_branch_cannot_guard_an_earlier_write():
    body = _assemble([_write(2), _push('blocked')+b'\x1c', (0x9D, 'end'),
                      _write(3), ('label', 'end'), b'\0'])
    reading = read_flash_method(_binding(body))
    assert reading.complete and len(reading.paths) == 2
    for path in reading.paths:
        assert len(path.conditions) == 1 and path.writes[0].conditions == ()
        if len(path.writes) == 2:
            assert path.writes[1].conditions == path.conditions
    assert _source_effect_prefix_invariant() == ()


def test_a_call_keeps_only_the_guards_that_reached_it():
    body = _assemble([_push('available')+b'\x1c\x12', (0x9D, 'end'),
                      _notify(), _push('later')+b'\x1c', (0x9D, 'end'),
                      _write(3), ('label', 'end'), b'\0'])
    reading = read_flash_method(_binding(body))
    assert reading.complete and not reading.effects_composed
    reached = [path for path in reading.paths if path.calls]
    assert len(reached) == 2
    assert all(len(path.conditions) == 2 and len(path.calls[0].conditions) == 1 for path in reached)
    assert reached[0].calls[0] == reached[1].calls[0]


def test_later_numeric_assumptions_do_not_rewrite_the_earlier_effect_context():
    body = _push('this')+b'\x1c'+_push('pending', 'speed')+b'\x1c\x4f'
    body += _notify()+_push('later')+b'\x1c\x17\0'
    reading = read_flash_method(_binding(body), constants={'speed': 3., 'later': 8.})
    assert reading.complete and not reading.effects_composed
    path = reading.paths[0]
    assert path.assumptions == (('later', 8.), ('speed', 3.))
    assert path.writes[0].assumptions == path.calls[0].assumptions == (('speed', 3.),)


def test_known_key_and_parameter_context_travels_with_each_effect():
    body = _assemble([_push(38, 1, 'Key')+b'\x1c'+_push('isDown')+b'\x52\x12',
                      (0x9D, 'end'), _write(2), _notify(), ('label', 'end'), b'\0'])
    binding = _binding(body, ('amount',))
    pressed = read_flash_method(binding, (7,), held_codes=frozenset({38}))
    resting = read_flash_method(binding, (7,), held_codes=frozenset())
    assert pressed.complete and resting.complete
    assert pressed.invocation.held_codes == frozenset({38}) and resting.invocation.held_codes == frozenset()
    assert pressed.invocation.parameters == (('amount', Arithmetic('number', (7.,))),)
    path = pressed.paths[0]
    assert path.writes[0].invocation == path.calls[0].invocation == pressed.invocation
    assert not resting.paths[0].writes and not resting.paths[0].calls


@pytest.mark.parametrize('keys', [{38}, frozenset({True}), frozenset({300}), frozenset({'up'})])
def test_mutable_or_invalid_input_context_cannot_enter_an_effect_receipt(keys):
    reading = read_flash_method(_binding(_write(2)+b'\0'), held_codes=keys)
    assert not reading.complete and not reading.paths


def test_guarded_prefix_survives_an_unread_tail_without_qualifying_the_method():
    body = _assemble([_push('available')+b'\x1c\x12', (0x9D, 'end'),
                      _notify(), b'\x07', ('label', 'end'), b'\0'])
    reading = read_flash_method(_binding(body))
    assert not reading.complete and not reading.effects_composed
    path = next(path for path in reading.paths if path.calls)
    assert path.issues and path.calls[0].conditions == path.conditions


def test_a_free_callback_does_not_become_its_own_receiver():
    binding = replace(_binding(_write(2)+b'\0'), target='callback')
    unbound = read_flash_method(binding)
    supplied = read_flash_method(binding, receiver='vehicle')
    assert unbound.complete and supplied.complete
    assert unbound.paths[0].writes[0].target == 'this.pending'
    assert supplied.paths[0].writes[0].target == 'vehicle.pending'
    assert unbound.invocation.receiver_source == 'unresolved'
    assert supplied.invocation.receiver_source == 'caller supplied'
    assert supplied.paths[0].writes[0].invocation == supplied.invocation


@pytest.mark.parametrize('receiver', ['', 'object[2]', 'object;script()', 'a..b', True, 'x'*257])
def test_dynamic_or_unbounded_receiver_paths_remain_unread(receiver):
    reading = read_flash_method(_binding(_write(2)+b'\0'), receiver=receiver)
    assert not reading.complete and not reading.paths


def _delete_pending():
    return _push('this')+b'\x1c'+_push('pending')+b'\x3a'


def test_property_deletion_keeps_its_stack_result_and_uncomposed_mutation():
    from core.perception.symbolic_motion_in_flash_methods import _source_deletion_invariant

    reading = read_flash_method(_binding(_write(7)+_delete_pending()+b'\x3e'))
    assert reading.complete and not reading.effects_composed
    path, = reading.paths
    deletion, = path.deletions
    assert deletion.target == 'Opaque.pending'
    assert path.returned == deletion.result and deletion.result.operation == 'deletion result'
    assert deletion.source == 'opaque-source' and deletion.offset > path.writes[0].offset
    assert deletion.invocation == path.writes[0].invocation == reading.invocation
    assert _source_deletion_invariant() == ()


def test_consuming_a_deletion_result_leaves_the_following_write_readable():
    reading = read_flash_method(_binding(_delete_pending()+b'\x17'+_write(11)+b'\0'))
    assert reading.complete and not reading.effects_composed
    assert reading.paths[0].writes[0].value == Arithmetic('number', (11.,))
    assert len(reading.paths[0].deletions) == 1


def test_deletion_does_not_replay_an_old_write_or_a_source_constant():
    body = _write(7)+_delete_pending()+b'\x17'+_push('this')+b'\x1c'+_push('pending')+b'\x4e\x3e'
    reading = read_flash_method(_binding(body), constants={'Opaque.pending': 13.})
    assert reading.complete
    assert reading.paths[0].returned.operation == 'after deletion'
    assert not reading.paths[0].assumptions
    assert reading.paths[0].writes[0].value == Arithmetic('number', (7.,))


def test_a_value_saved_before_deletion_remains_a_previous_value():
    body = _write(7)+_push('this')+b'\x1c'+_push('pending')+b'\x4e'
    body += _delete_pending()+b'\x17\x3e'
    reading = read_flash_method(_binding(body))
    assert reading.complete and reading.paths[0].returned == Arithmetic('number', (7.,))


def test_a_later_assignment_can_supply_a_new_value_after_deletion():
    body = _write(7)+_delete_pending()+b'\x17'+_write(11)
    body += _push('this')+b'\x1c'+_push('pending')+b'\x4e\x3e'
    reading = read_flash_method(_binding(body))
    assert reading.complete and reading.paths[0].returned == Arithmetic('number', (11.,))
    assert not reading.effects_composed


def test_unknown_deletion_success_splits_branches_without_choosing_an_outcome():
    body = _assemble([_delete_pending(), (0x9D, 'yes'), _write(-1), (0x99, 'end'),
                      ('label', 'yes'), _write(1), ('label', 'end'), b'\0'])
    reading = read_flash_method(_binding(body))
    assert reading.complete and len(reading.paths) == 2 and not reading.effects_composed
    assert {path.writes[0].value.arguments[0] for path in reading.paths} == {-1., 1.}
    for path in reading.paths:
        assert path.deletions[0].conditions == ()
        assert path.writes[0].conditions == path.conditions
        assert path.conditions[0].condition == path.deletions[0].result


def test_guarded_deletion_survives_an_unread_tail_without_qualifying_the_method():
    body = _assemble([_push('available')+b'\x1c', (0x9D, 'end'),
                      _delete_pending()+b'\x17\x07', ('label', 'end'), b'\0'])
    reading = read_flash_method(_binding(body))
    assert not reading.complete and not reading.effects_composed
    path = next(path for path in reading.paths if path.deletions)
    assert path.issues and path.deletions[0].conditions == path.conditions


@pytest.mark.parametrize('opcode', [0x3A, 0x4E, 0x4F])
def test_an_unknown_member_value_cannot_be_treated_as_its_variable_name(opcode):
    body = _push('this')+b'\x1c'+_push('field')+b'\x1c'
    if opcode == 0x4F:
        body += _push(7)
    reading = read_flash_method(_binding(body+bytes((opcode,))+b'\0'))
    assert not reading.complete
    assert any('dynamic property name' in issue for issue in reading.issues)
    assert not reading.paths[0].writes and not reading.paths[0].deletions


def test_an_explicit_literal_member_computed_in_a_local_is_still_readable():
    body = _push('field', 'pending')+b'\x3c'+_write(7)
    body += _push('this')+b'\x1c'+_push('field')+b'\x1c\x4e\x3e'
    reading = read_flash_method(_binding(body))
    assert reading.complete and reading.paths[0].returned == Arithmetic('number', (7.,))


def test_scope_deletion_stays_unsupported_without_scope_resolution():
    reading = read_flash_method(_binding(_push('field')+b'\x3b\x17\0'))
    assert not reading.complete and any('Delete2' in issue for issue in reading.issues)


def test_a_delete_on_one_path_cannot_invalidate_the_sibling_path():
    body = _assemble([_write(7), _push('available')+b'\x1c', (0x9D, 'joined'),
                      _delete_pending()+b'\x17', ('label', 'joined'),
                      _push('this')+b'\x1c'+_push('pending')+b'\x4e\x3e'])
    reading = read_flash_method(_binding(body))
    assert reading.complete and len(reading.paths) == 2
    deleted = next(path for path in reading.paths if path.deletions)
    kept = next(path for path in reading.paths if not path.deletions)
    assert deleted.returned.operation == 'after deletion'
    assert kept.returned == Arithmetic('number', (7.,))
    assert deleted.deletions[0].conditions == deleted.conditions

"""Resolve parsed arithmetic paths and compare them with independent bytecode runs."""
import random
from dataclasses import replace

import pytest

from core.perception.a_source_method_with_resolved_values import (
    SourceValues, _resolved_source_step_invariant, resolve_source_method,
)
from core.perception.flash_action_sources import FlashActionSources
from core.perception.flash_actions_with_source import ActionIssue
from core.perception.reading_a_program import ProgramRead
from core.perception.symbolic_motion_in_flash_methods import (
    Arithmetic, BranchEvidence, FlashMethodReading, MethodPath, SourceWrite, read_flash_method,
)
from tests.test_guarded_source_effects_do_not_enumerate_paths import (
    _assemble, _binding, _diamonds, _initial, _push, _run_code, _source,
)

pytestmark = pytest.mark.unit


def values(**supplied):
    return SourceValues('fixture-bytes', ('observed-instance', 2, 1.), 5., supplied)


def program(body, params=()):
    return ProgramRead(action_sources=FlashActionSources('fixture-bytes', (_source(body, params),), ()))


def write(name, expression):
    return _push(name)+expression+b'\x1d'


def read(name):
    return _push(name)+b'\x1c'


def test_parsed_ordered_updates_match_independent_bytecode_for_100_states():
    body = _assemble([write('velocity', read('velocity')+read('force')+b'\x0a'),
        write('position', read('position')+read('velocity')+b'\x0a'),
        read('position')+read('limit')+b'\x48', (0x9D, 'done'),
        write('velocity', _push(0)), 'done', read('velocity')+b'\x3e'])
    binding, current = _binding(body, ('force',)), program(body, ('force',))
    rng = random.Random(20261020)
    for _ in range(100):
        supplied = dict(position=rng.randrange(-100, 100), velocity=rng.randrange(-10, 10),
                        limit=rng.randrange(-80, 80))
        force = rng.randrange(-5, 6)
        expected, memory = _run_code(binding, _initial(**supplied), (force,))
        step = current.resolved_method('operate', values(**supplied), arguments=(force,))
        assert step.resolved and step.returned.arguments == (float(expected),)
        actual = {name: float(value) for name, value in supplied.items()}
        for effect in step.writes:
            actual[effect.target] = effect.value.arguments[0]
            assert effect.source == 'fixture-bytes' and effect.offset >= 80
            assert effect.invocation == step.invocation
        assert actual['position'] == memory[0]['position']
        assert actual['velocity'] == memory[0]['velocity']
        assert step.owner == ('observed-instance', 2, 1.) and step.at == 5.
        assert step.conditions and step.invocation.parameters[0][1].arguments == (float(force),)


def test_known_physical_key_context_remains_in_the_resolved_result():
    key = _push(37, 1, 'Key')+b'\x1c'+_push('isDown')+b'\x52'
    body = _assemble([key, (0x9D, 'held'), write('velocity', _push(0)), (0x99, 'done'),
                      'held', write('velocity', _push(7)), 'done', read('velocity')+b'\x3e'])
    current = program(body)
    released = current.resolved_method('operate', values(), held_codes=frozenset())
    held = current.resolved_method('operate', values(), held_codes=frozenset({37}))
    assert released.resolved and held.resolved
    assert released.returned.arguments == (0.,) and held.returned.arguments == (7.,)
    assert released.invocation.held_codes == frozenset() and held.invocation.held_codes == frozenset({37})
    unknown = current.resolved_method('operate', values())
    assert not unknown.resolved and not unknown.writes


def test_missing_value_retains_the_actual_source_site_and_publishes_no_prefix():
    body = write('already_known', _push(3))+write('next', read('unobserved'))+b'\0'
    step = program(body).resolved_method('operate', values())
    assert not step.resolved and not step.writes and step.returned is None
    need, = step.needs
    assert (need.operation, need.target, need.source) == ('observe_source_value', 'unobserved', 'fixture-bytes')
    assert need.offset >= 80


def test_unknown_guard_cannot_choose_either_reachable_branch():
    body = _assemble([read('enabled'), (0x9D, 'yes'), write('v', _push(-2)), (0x99, 'done'),
                      'yes', write('v', _push(2)), 'done', b'\0'])
    step = program(body).resolved_method('operate', values())
    assert not step.resolved and not step.writes
    assert {need.target for need in step.needs} == {'enabled'}
    assert program(body).resolved_method('operate', values(enabled=False)).writes[0].value.arguments == (-2.,)


def test_a_known_false_guard_eliminates_a_path_with_another_unknown_condition():
    unknown = BranchEvidence('fixture-bytes', 81, Arithmetic('reference', ('missing',)), True)
    false = BranchEvidence('fixture-bytes', 82, Arithmetic('reference', ('known',)), True)
    fallback = replace(false, taken=False)
    reading = FlashMethodReading('m', (MethodPath((unknown, false), (), (), Arithmetic('number', (9.,))),
                                      MethodPath((fallback,), (), (), Arithmetic('number', (1.,)))))
    step = resolve_source_method(reading, values(known=False))
    assert step.resolved and step.returned.arguments == (1.,) and step.conditions == (fallback,)


def test_one_possible_unknown_rival_blocks_a_known_path():
    unknown = BranchEvidence('fixture-bytes', 81, Arithmetic('reference', ('missing',)), True)
    reading = FlashMethodReading('m', (MethodPath((), (), (), Arithmetic('number', (1.,))),
                                      MethodPath((unknown,), (), (), Arithmetic('number', (2.,)))))
    step = resolve_source_method(reading, values())
    assert not step.resolved and step.needs[0].target == 'missing'


def test_ambiguous_paths_and_foreign_instruction_sources_are_not_resolved():
    path = MethodPath((), (), (), Arithmetic('undefined'))
    step = resolve_source_method(FlashMethodReading('m', (path, path)), values())
    assert not step.resolved and step.needs[0].operation == 'distinguish_source_path'
    foreign = SourceWrite('v', Arithmetic('number', (2.,)), 'fixture-bytes-other', 9)
    step = resolve_source_method(FlashMethodReading('m', (replace(path, writes=(foreign,)),)), values())
    assert not step.resolved and step.needs[0].operation == 'bind_current_source'
    condition = BranchEvidence('other-source', 8, Arithmetic('boolean', (True,)), True)
    step = resolve_source_method(FlashMethodReading('m', (replace(path, conditions=(condition,)),)), values())
    assert not step.resolved and step.needs[0].operation == 'bind_current_source'


@pytest.mark.parametrize('body,operation', [
    (_push(0, 'notify')+b'\x3d\x17\0', 'resolve_call_effects'),
    (_push('object')+b'\x1c'+_push('value')+b'\x3a\x17\0', 'resolve_property_deletion'),
])
def test_uncomposed_effects_cannot_publish_partial_arithmetic(body, operation):
    step = program(write('prefix', _push(1))+body).resolved_method('operate', values())
    assert not step.resolved and not step.writes and step.returned is None
    assert step.needs[0].operation == operation and step.needs[0].offset >= 80


def test_assumed_constants_require_matching_explicit_values():
    reading = read_flash_method(_binding(read('external')+b'\x3e'), constants={'external': 4.})
    absent = resolve_source_method(reading, values())
    wrong = resolve_source_method(reading, values(external=5.))
    right = resolve_source_method(reading, values(external=4.))
    assert not absent.resolved and not wrong.resolved
    assert absent.needs[0].operation == 'verify_assumed_value'
    assert right.resolved and right.returned.arguments == (4.,)


def test_unsupported_arithmetic_and_path_budget_remain_explicit():
    division = program(read('numerator')+read('denominator')+b'\x0d\x3e')
    assert not division.resolved_method('operate', values(numerator=1., denominator=0.)).resolved
    assert division.resolved_method('operate', values(numerator=1., denominator=2.)).returned.arguments == (.5,)
    too_many = program(_diamonds(16)).resolved_method('operate', values())
    assert not too_many.resolved and too_many.needs[0].operation == 'complete_source_method'


@pytest.mark.parametrize('kind', ['number', 'boolean', 'string', 'null'])
def test_supplied_initial_state_selects_the_actual_path_before_the_path_budget_is_spent(kind):
    body = _diamonds(16)
    current, binding = program(body), _binding(body)
    randomizer = random.Random(20261020)
    for _ in range(12):
        flags = {f'flag{i}': bool(randomizer.getrandbits(1)) for i in range(16)}
        if kind == 'number':
            supplied = {name: int(value) for name, value in flags.items()}
        elif kind == 'string':
            supplied = {name: 'ready' if value else '' for name, value in flags.items()}
        elif kind == 'null':
            supplied = {name: 1 if value else None for name, value in flags.items()}
        else:
            supplied = flags
        expected, _memory = _run_code(binding, _initial(**supplied))
        step = current.resolved_method('operate', values(**supplied))
        assert step.resolved, step.needs
        assert step.returned.arguments == (float(expected),)
        assert len(step.conditions) == 16
        assert all(condition.source == 'fixture-bytes' and condition.offset >= 80 for condition in step.conditions)


def test_a_missing_initial_value_remains_unknown_after_other_guards_are_selected():
    supplied = {f'flag{i}': True for i in range(15)}
    step = program(_diamonds(16)).resolved_method('operate', values(**supplied))
    assert not step.resolved and not step.writes
    assert step.needs[0].operation == 'observe_source_value' and step.needs[0].target == 'flag15'


def test_a_cached_selected_path_cannot_be_resolved_with_changed_or_missing_inputs():
    before = values(**{f'flag{i}': True for i in range(16)})
    supplied = dict(before.values)
    reading = read_flash_method(_binding(_diamonds(16)), initial_values=supplied)
    supplied['flag0'] = Arithmetic('boolean', (False,))
    assert dict(reading.invocation.initial_values)['flag0'] == Arithmetic('boolean', (True,))
    assert resolve_source_method(reading, before).resolved
    changed = values(**{f'flag{i}': i != 0 for i in range(16)})
    wrong = resolve_source_method(reading, changed)
    absent = resolve_source_method(reading, values())
    assert not wrong.resolved and not wrong.writes and wrong.needs[0].target == 'flag0'
    assert wrong.needs[0].operation == 'verify_initial_source_value'
    assert not absent.resolved and not absent.writes


def test_source_writes_and_explicit_parameters_override_initial_field_values_in_order():
    body = write('v', read('v')+read('amount')+b'\x0a')+write('amount', _push(20))+read('v')+read('amount')+b'\x0a\x3e'
    binding, current = _binding(body, ('amount',)), program(body, ('amount',))
    supplied = dict(v=4., amount=90.)
    expected, _memory = _run_code(binding, _initial(**supplied), (3.,))
    step = current.resolved_method('operate', values(**supplied), arguments=(3.,))
    assert step.resolved and step.returned.arguments == (float(expected),) == (27.,)
    assert step.invocation.parameters == (('amount', Arithmetic('number', (3.,))),)


@pytest.mark.parametrize('initial', [
    {'flag0': Arithmetic('reference', ('hidden',))},
    {'flag0': Arithmetic('number', (float('inf'),))},
    {'flag0': Arithmetic('boolean', (1,))},
    {'this': Arithmetic('number', (3.,))},
    {'a[0]': Arithmetic('number', (3.,))},
    {f'flag{i}': Arithmetic('boolean', (True,)) for i in range(1025)},
])
def test_nonprimitive_or_unbounded_initial_context_cannot_choose_a_path(initial):
    reading = read_flash_method(_binding(_diamonds(2)), initial_values=initial)
    assert not reading.complete and reading.issues


def test_deleted_initial_values_cannot_reappear_from_the_supplied_snapshot():
    body = _push('object')+b'\x1c'+_push('value')+b'\x3a\x17'+read('object')+_push('value')+b'\x4e\x3e'
    step = program(body).resolved_method('operate', SourceValues('fixture-bytes', 3, 5., {'object.value': 7.}))
    assert not step.resolved and not step.writes and step.needs[0].operation == 'resolve_property_deletion'


def test_initial_source_context_has_a_registered_current_path_canary():
    from core.perception.symbolic_motion_in_flash_methods import _initial_source_path_invariant

    assert _initial_source_path_invariant() == ()


@pytest.mark.parametrize('context', [
    None,
    (('flag0', Arithmetic('reference', ('missing',))),),
    (('flag0', Arithmetic('number', (float('nan'),))),),
    (('flag0', Arithmetic('boolean', (True,))), ('flag0', Arithmetic('boolean', (False,)))),
    (('flag0',),),
])
def test_malformed_cached_initial_context_returns_an_evidence_need(context):
    reading = read_flash_method(_binding(_diamonds(1)), initial_values={'flag0': Arithmetic('boolean', (True,))})
    reading = replace(reading, invocation=replace(reading.invocation, initial_values=context))
    step = resolve_source_method(reading, values(flag0=True))
    assert not step.resolved and not step.writes
    assert step.needs[0].operation == 'verify_initial_source_context'


def test_program_requires_unique_current_complete_byte_sources():
    current = program(b'\0')
    assert current.resolved_method('missing', values()) is None
    assert current.resolved_method('operate', {'source': 'fixture-bytes'}) is None
    assert current.resolved_method('operate', replace(values(), source='older-source')) is None
    current.action_sources = replace(current.action_sources, issues=(ActionIssue(0, 'incomplete'),))
    assert current.resolved_method('operate', values()) is None
    current = program(b'\0')
    current.action_sources = replace(current.action_sources, streams=current.action_sources.streams*2)
    assert current.resolved_method('operate', values()) is None
    assert ProgramRead().resolved_method('operate', values()) is None


def test_supplied_values_are_frozen_snapshots_and_do_not_infer_hidden_state():
    supplied = {'v': 4.}
    snapshot = SourceValues('fixture-bytes', 3, 5., supplied)
    supplied['v'] = 90.
    assert snapshot.values['v'].arguments == (4.,)
    with pytest.raises(TypeError):
        snapshot.values['v'] = Arithmetic('number', (1.,))
    step = program(read('hidden')+b'\x3e').resolved_method('operate', snapshot)
    assert not step.resolved and step.needs[0].target == 'hidden'


@pytest.mark.parametrize('change', [
    {'at': True}, {'at': float('nan')}, {'at': -1.}, {'at': 10**400}, {'owner': []},
    {'source': ''}, {'values': {'v': 10**400}}, {'values': {'v': float('inf')}},
    {'values': {'v': object()}}, {'values': {'v': 'x'*4097}}, {'values': {'a[0]': 1.}},
    {'values': {f'v{i}': 1. for i in range(1025)}},
])
def test_invalid_or_unbounded_supplied_context_is_rejected(change):
    with pytest.raises(ValueError):
        replace(values(), **change)


@pytest.mark.parametrize('expression', [Arithmetic('add'), Arithmetic('reference'),
    Arithmetic('number', (float('nan'),)), Arithmetic('boolean', (3,)), Arithmetic('string'),
    Arithmetic('number', (10**400,)),
    Arithmetic('unsupported', (Arithmetic('number', (2.,)),)),
])
def test_malformed_expressions_cannot_become_a_complete_result(expression):
    reading = FlashMethodReading('m', (MethodPath((), (), (), expression),))
    step = resolve_source_method(reading, values())
    assert not step.resolved and not step.writes and step.needs


def test_deep_source_expressions_exhaust_the_bound_without_a_partial_update():
    expression = Arithmetic('number', (1.,))
    for _ in range(34):
        expression = Arithmetic('neg', (expression,))
    prefix = SourceWrite('prefix', Arithmetic('number', (1.,)), 'fixture-bytes', 9)
    reading = FlashMethodReading('m', (MethodPath((), (prefix,), (), expression),))
    step = resolve_source_method(reading, values())
    assert not step.resolved and not step.writes and step.needs[0].target == 'bounded_expression_required'


def test_bounded_source_resolution_has_a_registered_executable_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _resolved_source_step_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'source_arithmetic_requires_explicit_values_and_effects')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())

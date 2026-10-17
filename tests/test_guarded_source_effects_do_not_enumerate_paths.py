"""Compare guarded joins with concrete bytecode runs, including opaque effects."""
import copy
import itertools
import random
import struct
from dataclasses import dataclass, replace

import pytest

from core.perception.flash_action_sources import FlashActionSources
from core.perception.flash_actions_with_source import ActionIssue, parse_action_stream
from core.perception.guarded_effects_in_flash_methods import (
    _guarded_source_graph_invariant,
    read_flash_effect_graph,
)
from core.perception.reading_a_program import ProgramRead
from core.perception.symbolic_motion_in_flash_methods import bind_flash_methods, read_flash_method


def _action(code, payload=b''):
    return bytes((code,)) + (struct.pack('<H', len(payload)) + payload if code >= 0x80 else b'')


def _push(*values):
    payload = b''
    for value in values:
        if isinstance(value, str):
            payload += b'\0' + value.encode() + b'\0'
        elif type(value) is bool:
            payload += b'\x05' + bytes((value,))
        else:
            payload += b'\x07' + struct.pack('<i', value)
    return _action(0x96, payload)


def _assemble(parts):
    labels, offset = {}, 0
    for part in parts:
        if isinstance(part, str):
            labels[part] = offset
        else:
            offset += 5 if isinstance(part, tuple) else len(part)
    data = b''
    for part in parts:
        if isinstance(part, tuple):
            data += _action(part[0], struct.pack('<h', labels[part[1]] - len(data) - 5))
        elif not isinstance(part, str):
            data += part
    return data


def _source(body, params=()):
    header = b'\0' + struct.pack('<H', len(params))
    header += b''.join(param.encode() + b'\0' for param in params) + struct.pack('<H', len(body))
    data = _push('operate') + _action(0x9B, header) + body + b'\x1d\0'
    stream = parse_action_stream(data, source='fixture-bytes', base_offset=80)
    assert stream.complete
    return stream


def _function_literal(body, name=''):
    header = name.encode() + b'\0\0\0' + struct.pack('<H', len(body))
    return _action(0x9B, header) + body


def _binding(body, params=()):
    return bind_flash_methods((_source(body, params),)).method('operate')


@dataclass(frozen=True)
class _Object:
    number: int


def _binary(code, a, b):
    if code in (0x0A, 0x47):
        return a + b
    if code == 0x0B:
        return a - b
    if code == 0x0C:
        return a * b
    if code in (0x0E, 0x49):
        return a == b
    if code == 0x66:
        return type(a) is type(b) and a == b
    if code == 0x48:
        return a < b
    raise AssertionError(code)


def _operation(operation, memory, operands, code):
    """A small fixture's semantics; no Flash runtime or product controller."""
    memory = copy.deepcopy(memory)
    variables, objects, log, registers = memory
    if operation == 'get_variable':
        value = variables[operands[0]]
    elif operation == 'get_member':
        receiver, member = operands
        if member == 'withGetter':
            objects[receiver.number]['count'] += 1
            value = objects[receiver.number]['count']
        else:
            value = objects[receiver.number][member]
    elif operation in ('set_variable', 'define_local'):
        variables[operands[0]] = operands[1]
        value = None
    elif operation == 'set_member':
        receiver, member, value = operands
        objects[receiver.number][member] = value
    elif operation == 'get_register':
        value = registers[int(operands[0])]
    elif operation == 'set_register':
        registers[int(operands[0])] = operands[1]
        value = None
    elif operation == 'delete_member':
        receiver, member = operands
        value = member in objects[receiver.number]
        objects[receiver.number].pop(member, None)
    elif operation == 'call_function':
        name, *args = operands
        log.append((name, tuple(args)))
        if name == 'changeAlias':
            objects[variables['alias'].number]['value'] += 100
        if name == 'clobberRegister':
            registers[2] = 100
        value = len(log)
    elif operation == 'call_method':
        receiver, name, *args = operands
        log.append((receiver, name, tuple(args)))
        value = len(log)
    elif operation in ('construct_object', 'construct_method', 'init_array'):
        log.append((operation, operands))
        value = _Object(max(objects) + 1)
        objects[value.number] = {'constructor': operands}
    elif operation == 'binary':
        value = _binary(code, *operands)
    elif operation == 'unary':
        assert code == 0x12
        value = not operands[0]
    elif operation == 'truth':
        value = bool(operands[0])
    else:
        raise AssertionError(operation)
    return value, memory


def _run_graph(graph, initial, args=()):
    memo = {}

    def evaluate(index):
        if index in memo:
            return memo[index]
        node = graph.nodes[index]
        op = node.operation
        if op in ('boolean', 'number', 'string'):
            value = node.data[0]
        elif op in ('undefined', 'null'):
            value = None
        elif op == 'receiver_input':
            value = initial[0]['this']
        elif op == 'argument_input':
            value = args[node.data[0]]
        elif op == 'entry_memory':
            value = copy.deepcopy(initial)
            value[0].update(zip(node.data[0], args, strict=True))
            for register, name in node.data[1]:
                value[3][register] = value[0][name]
            if node.data[2] & 1:
                value[3][1] = value[0]['this']
        elif op == 'select':
            value = evaluate(node.inputs[1] if evaluate(node.inputs[0]) else node.inputs[2])
        elif op == 'and':
            value = evaluate(node.inputs[0]) and evaluate(node.inputs[1])
        elif op == 'or':
            value = evaluate(node.inputs[0]) or evaluate(node.inputs[1])
        elif op == 'not':
            value = not evaluate(node.inputs[0])
        elif op == 'after':
            evaluate(node.inputs[0])
            value = memo[('memory', node.inputs[0])]
        else:
            memory = evaluate(node.inputs[0])
            operands = tuple(evaluate(input_) for input_ in node.inputs[1:])
            result, after = _operation(op, memory, operands, node.data[1])
            memo[('memory', index)] = after
            value = after if op.startswith(('set_', 'define_')) else result
        memo[index] = value
        return value

    exits = [exit_ for exit_ in graph.exits if evaluate(exit_.guard)]
    assert len(exits) == 1
    exit_ = exits[0]
    return evaluate(exit_.returned), evaluate(exit_.memory)


def _run_code(binding, initial, args=()):
    """Concrete address-following VM for these fixtures, independent of joins."""
    stream = binding.function.body
    instructions = {instruction.offset: instruction for instruction in stream.instructions}
    offset, stack, pool = stream.instructions[0].offset, [], binding.constant_pool
    memory = copy.deepcopy(initial)
    memory[0].update(zip(binding.function.parameters, args, strict=True))
    for _ in range(10000):
        if offset == stream.end_offset:
            return None, memory
        instruction = instructions[offset]
        code = instruction.opcode
        if code == 0:
            return None, memory
        if code == 0x3E:
            return stack.pop(), memory
        if code == 0x99:
            offset = instruction.branch_target
            continue
        if code == 0x9D:
            offset = instruction.branch_target if stack.pop() else instruction.next_offset
            continue
        if code == 0x96:
            for operand in instruction.operands:
                stack.append(memory[3][operand.value] if operand.kind == 'register' else
                             pool[operand.value] if operand.kind == 'constant' else operand.value)
        elif code == 0x88:
            pool = tuple(operand.value for operand in instruction.operands)
        elif code == 0x87:
            memory[3][instruction.operands[0].value] = stack[-1]
        elif code == 0x17:
            stack.pop()
        elif code == 0x4C:
            stack.append(stack[-1])
        elif code == 0x4D:
            stack[-2:] = reversed(stack[-2:])
        else:
            if code == 0x1C:
                operation, operands = 'get_variable', (stack.pop(),)
            elif code in (0x4E, 0x3A):
                member, receiver = stack.pop(), stack.pop()
                operation = 'get_member' if code == 0x4E else 'delete_member'
                operands = (receiver, member)
            elif code in (0x1D, 0x3C):
                value, name = stack.pop(), stack.pop()
                operation = 'set_variable' if code == 0x1D else 'define_local'
                operands = (name, value)
            elif code == 0x4F:
                value, member, receiver = stack.pop(), stack.pop(), stack.pop()
                operation, operands = 'set_member', (receiver, member, value)
            elif code in (0x3D, 0x52, 0x40, 0x53):
                name = stack.pop()
                receiver = stack.pop() if code in (0x52, 0x53) else None
                count = stack.pop()
                operands = tuple(stack.pop() for _ in range(count))
                operation = {0x3D: 'call_function', 0x52: 'call_method', 0x40: 'construct_object', 0x53: 'construct_method'}[code]
                operands = (receiver, name, *operands) if code in (0x52, 0x53) else (name, *operands)
            elif code == 0x42:
                count = stack.pop()
                operation, operands = 'init_array', tuple(stack.pop() for _ in range(count))
            elif code == 0x12:
                operation, operands = 'unary', (stack.pop(),)
            else:
                b, a = stack.pop(), stack.pop()
                operation, operands = 'binary', (a, b)
            value, memory = _operation(operation, memory, operands, code)
            if operation not in ('set_variable', 'define_local', 'set_member'):
                stack.append(value)
        offset = instruction.next_offset
    raise AssertionError('fixture did not terminate')


def _initial(**variables):
    return ({'this': _Object(1), 'alias': _Object(1), **variables}, {1: {'value': 3, 'count': 0}}, [], {})


def _diamonds(count):
    parts = [_push('total', 0) + b'\x1d']
    for index in range(count):
        parts += [_push(f'flag{index}') + b'\x1c', (0x9D, f'yes{index}'),
                  _push('total', 'total') + b'\x1c' + _push(2) + b'\x0a\x1d', (0x99, f'end{index}'),
                  f'yes{index}', _push('total', 'total') + b'\x1c' + _push(5) + b'\x0a\x1d', f'end{index}']
    return _assemble([*parts, _push('total') + b'\x1c\x3e'])


def test_many_unknown_diamonds_share_graph_and_match_concrete_input_runs():
    binding = _binding(_diamonds(16))
    assert not read_flash_method(binding).complete
    graph = read_flash_effect_graph(binding)
    assert graph.complete and not graph.effects_composed
    assert len(graph.branches) == 16
    assert len(graph.nodes) < 1200
    randomizer = random.Random(20261016)
    for _ in range(100):
        flags = {f'flag{i}': bool(randomizer.getrandbits(1)) for i in range(16)}
        initial = _initial(**flags)
        actual = _run_graph(graph, initial)
        assert actual == _run_code(binding, initial)
        assert actual[0] == sum(5 if value else 2 for value in flags.values())
    smaller = read_flash_effect_graph(_binding(_diamonds(8)))
    assert len(graph.nodes) < 2 * len(smaller.nodes) + 10


def test_branch_stack_and_register_values_merge_under_exact_guards():
    body = _assemble([_push('flag') + b'\x1c', (0x9D, 'yes'), _push(3), (0x99, 'join'),
                      'yes', _push(9), 'join', _action(0x87, b'\x02'), b'\x17',
                      _action(0x96, b'\x04\x02'), b'\x3e'])
    binding = _binding(body)
    graph = read_flash_effect_graph(binding)
    assert graph.complete
    for flag in (False, True):
        assert _run_graph(graph, _initial(flag=flag)) == _run_code(binding, _initial(flag=flag))


def test_opaque_call_and_alias_mutation_cannot_reuse_an_earlier_load():
    read = _push('alias') + b'\x1c' + _push('value') + b'\x4e'
    body = read + _push(0, 'changeAlias') + b'\x3d\x17' + read + b'\x0b\x3e'
    binding = _binding(body)
    graph = read_flash_effect_graph(binding)
    assert graph.complete and not graph.effects_composed
    assert _run_graph(graph, _initial()) == _run_code(binding, _initial())
    assert _run_graph(graph, _initial())[0] == -100
    loads = [node for node in graph.nodes if node.operation == 'get_member']
    assert len(loads) == 2 and loads[0].inputs[0] != loads[1].inputs[0]


def test_register_load_after_opaque_effect_is_a_current_vm_state_read():
    body = (_push(3) + _action(0x87, b'\x02') + b'\x17' + _push(0, 'clobberRegister') + b'\x3d\x17'
            + _action(0x96, b'\x04\x02') + b'\x3e')
    binding = _binding(body)
    graph = read_flash_effect_graph(binding)
    assert graph.complete and not graph.effects_composed
    assert _run_graph(graph, _initial()) == _run_code(binding, _initial())
    assert _run_graph(graph, _initial())[0] == 100
    call = next(effect for effect in graph.effects if effect.operation == 'call_function')
    read = next(i for i, node in enumerate(graph.nodes) if node.operation == 'get_register')
    assert call.after in graph.slice(read)


def test_getter_memory_and_conditional_deletion_survive_joins():
    read = _push('this') + b'\x1c' + _push('withGetter') + b'\x4e'
    deletion = _push('this') + b'\x1c' + _push('value') + b'\x3a\x17'
    body = _assemble([read, b'\x17', _push('flag') + b'\x1c', (0x9D, 'join'), deletion,
                      'join', read, b'\x3e'])
    binding = _binding(body)
    graph = read_flash_effect_graph(binding)
    assert graph.complete and not graph.effects_composed
    assert [effect.operation for effect in graph.effects] == ['delete_member']
    for flag in (False, True):
        result = _run_graph(graph, _initial(flag=flag))
        assert result == _run_code(binding, _initial(flag=flag))
        assert result[0] == 2
        assert ('value' in result[1][1][1]) is flag


def test_unreached_calls_do_not_execute_when_the_graph_is_traversed():
    body = _assemble([_push('flag') + b'\x1c', (0x9D, 'done'), _push(1, 1, 'report') + b'\x3d\x17',
                      'done', _push(7) + b'\x3e'])
    binding = _binding(body)
    graph = read_flash_effect_graph(binding)
    assert graph.complete
    for flag in (False, True):
        result = _run_graph(graph, _initial(flag=flag))
        assert result == _run_code(binding, _initial(flag=flag))
        assert len(result[1][2]) == (0 if flag else 1)


def test_multiple_returns_and_same_successor_keep_one_actual_exit():
    body = _assemble([_push('a') + b'\x1c', (0x9D, 'yes'), _push(1) + b'\x3e',
                      'yes', _push('b') + b'\x1c', (0x9D, 'same'), 'same', _push(2) + b'\x3e'])
    binding = _binding(body)
    graph = read_flash_effect_graph(binding)
    assert graph.complete and len(graph.exits) == 2
    for a, b in itertools.product((False, True), repeat=2):
        assert _run_graph(graph, _initial(a=a, b=b)) == _run_code(binding, _initial(a=a, b=b))


@pytest.mark.parametrize('body, phrase', [
    (_assemble(['loop', (0x99, 'loop')]), 'loop summary'),
    (b'\x17\0', 'underflow'),
    (b'\x07\0', 'unsupported graph instruction'),
    (_action(0x96, b'\x08\0') + b'\x3e', 'constant pool'),
    (_action(0x96, b'\x04\x02') + b'\x3e', 'unbound source register'),
])
def test_unmodeled_or_invalid_source_stays_incomplete(body, phrase):
    graph = read_flash_effect_graph(_binding(body))
    assert not graph.complete and not graph.effects_composed
    assert any(phrase in issue for issue in graph.issues)


def test_incompatible_join_stack_heights_are_rejected():
    body = _assemble([_push('flag') + b'\x1c', (0x9D, 'join'), _push(2), 'join', b'\0'])
    graph = read_flash_effect_graph(_binding(body))
    assert not graph.complete and 'stack height' in graph.issues[0]


def test_conditional_pool_does_not_become_an_unconditional_literal():
    body = _assemble([_push('flag') + b'\x1c', (0x9D, 'join'),
                      _action(0x88, b'\x01\0changed\0'), 'join', _action(0x96, b'\x08\0'), b'\x3e'])
    graph = read_flash_effect_graph(replace(_binding(body), constant_pool=('original',)))
    assert not graph.complete and 'constant pool' in graph.issues[0]


def test_register_defined_on_only_one_predecessor_stays_unbound():
    body = _assemble([_push('flag') + b'\x1c', (0x9D, 'join'), _push(2), _action(0x87, b'\x02'), b'\x17',
                      'join', _action(0x96, b'\x04\x02'), b'\x3e'])
    graph = read_flash_effect_graph(_binding(body))
    assert not graph.complete and 'unbound source register' in graph.issues[0]


def test_slice_includes_guard_memory_and_exact_source_addresses():
    graph = read_flash_effect_graph(_binding(_diamonds(3)))
    assert graph.complete and graph.source == 'fixture-bytes'
    all_dependencies = graph.slice(graph.exits[0].guard, graph.exits[0].memory, graph.exits[0].returned)
    assert all(all(index < at for index in node.inputs) for at, node in enumerate(graph.nodes))
    assert all(branch.condition in all_dependencies for branch in graph.branches)
    assert all(effect.before in all_dependencies for effect in graph.effects)
    assert all(effect.offset in graph.lifted for effect in graph.effects)
    for effect in graph.effects:
        assert effect.guard in graph.slice_effect(effect.offset)
    with pytest.raises(ValueError):
        graph.slice(-1)
    with pytest.raises(ValueError):
        graph.slice_effect(-1)


def test_parameters_remain_inputs_and_receiver_is_never_inferred_from_a_method_name():
    binding = _binding(_push('choice') + b'\x1c\x3e', ('choice',))
    graph = read_flash_effect_graph(binding)
    assert graph.complete
    assert next(node for node in graph.nodes if node.operation == 'receiver_input').data == (None, 'unresolved')
    for value in (False, True, 7):
        assert _run_graph(graph, _initial(), (value,)) == _run_code(binding, _initial(), (value,))
    supplied = read_flash_effect_graph(binding, receiver='scene.actor')
    assert next(node for node in supplied.nodes if node.operation == 'receiver_input').data == ('scene.actor', 'caller supplied')
    assert not read_flash_effect_graph(binding, receiver='bad path').complete


def test_program_read_only_offers_a_uniquely_bound_complete_current_source():
    stream = _source(_diamonds(3))
    read = ProgramRead(action_sources=FlashActionSources('fixture', (stream,), ()))
    assert read.method_effects('operate').complete
    assert read.method_effects('absent') is None
    read.action_sources = FlashActionSources('fixture', (stream, stream), ())
    assert read.method_effects('operate') is None
    read.action_sources = FlashActionSources('fixture', (stream,), (ActionIssue(0, 'unread container'),))
    assert read.method_effects('operate') is None
    assert ProgramRead().method_effects('operate') is None


def test_graph_budgets_fail_without_increasing_the_path_limit(monkeypatch):
    import core.perception.guarded_effects_in_flash_methods as module

    monkeypatch.setattr(module, 'MAX_NODES', 24)
    graph = read_flash_effect_graph(_binding(_diamonds(3)))
    assert not graph.complete and 'node budget' in graph.issues[0]
    assert _guarded_source_graph_invariant() == ()


@pytest.mark.parametrize('limit, value, phrase', [
    ('MAX_INSTRUCTIONS', 4, 'instruction budget'),
    ('MAX_EFFECTS', 1, 'effect budget'),
    ('MAX_STACK', 1, 'stack or effect budget'),
])
def test_other_resource_budgets_also_preserve_incompleteness(monkeypatch, limit, value, phrase):
    import core.perception.guarded_effects_in_flash_methods as module

    monkeypatch.setattr(module, limit, value)
    graph = read_flash_effect_graph(_binding(_diamonds(3)))
    assert not graph.complete and any(phrase in issue for issue in graph.issues)


def test_a_complete_source_stream_still_refuses_unsupported_method_preloads():
    binding = _binding(b'\0')
    binding = replace(binding, function=replace(binding.function, flags=4))
    graph = read_flash_effect_graph(binding)
    assert not graph.complete and 'preloaded' in graph.issues[0]


def test_unknown_argument_count_is_never_replaced_with_zero():
    body = _push('count') + b'\x1c' + _push('report') + b'\x3d\x3e'
    graph = read_flash_effect_graph(_binding(body))
    assert not graph.complete and 'argument count' in graph.issues[0]


def test_duplicate_parameters_and_preload_conflicts_cannot_hide_input_state():
    graph = read_flash_effect_graph(_binding(b'\0', ('arg', 'arg')))
    assert not graph.complete and 'duplicate parameter' in graph.issues[0]
    binding = _binding(b'\0', ('arg',))
    conflicting = replace(binding, function=replace(binding.function, flags=1, register_parameters=((1, 'arg'),)))
    assert not read_flash_effect_graph(conflicting).complete


@pytest.mark.parametrize('code', [0x40, 0x53, 0x42])
def test_construction_keeps_argument_order_and_unknown_heap_effects(code):
    if code == 0x53:
        body = _push(7, 3, 2, 'this') + b'\x1c' + _push('factory') + bytes((code,)) + b'\x3e'
    elif code == 0x40:
        body = _push(7, 3, 2, 'Factory') + bytes((code,)) + b'\x3e'
    else:
        body = _push(7, 3, 2) + bytes((code,)) + b'\x3e'
    binding = _binding(body)
    graph = read_flash_effect_graph(binding)
    assert graph.complete and not graph.effects_composed
    result = _run_graph(graph, _initial())
    assert result == _run_code(binding, _initial())
    assert result[1][2][-1][1][-2:] == (3, 7)


def test_function_literal_keeps_its_body_and_context_without_running_it():
    nested_body = _push('nestedEffect', 19) + b'\x1d\0'
    body = (_push('this') + b'\x1c' + _push('onEnterFrame') + _function_literal(nested_body)
            + b'\x4f\0')
    binding = _binding(body)
    graph = read_flash_effect_graph(binding)
    assert graph.complete and not graph.effects_composed
    function = binding.function.body.functions[0]
    creation = next(effect for effect in graph.effects if effect.operation == 'make_function')
    node = graph.nodes[creation.result]
    assert node.operation == 'function_value'
    assert node.data[2:5] == (function.body_offset, function.end_offset, '')
    assignment = next(effect for effect in graph.effects if effect.operation == 'set_member')
    assert assignment.operands[-1] == creation.result
    assert all(effect.offset < function.body_offset or effect.offset >= function.end_offset for effect in graph.effects)
    assert creation.before in graph.slice_effect(assignment.offset)


def test_named_function_declaration_has_a_scope_effect_and_pushes_no_value():
    binding = _binding(_function_literal(b'\0', name='helper') + _push(2) + b'\x3e')
    graph = read_flash_effect_graph(binding)
    assert graph.complete
    effect = next(effect for effect in graph.effects if effect.operation == 'define_function')
    assert graph.nodes[effect.operands[0]].data == ('helper',)
    assert graph.nodes[graph.exits[0].returned].data == (2.,)


def test_function_body_must_be_owned_by_its_actual_definition_address():
    binding = _binding(_function_literal(b'\0') + b'\x17\0')
    binding = replace(binding, function=replace(binding.function, body=replace(binding.function.body, functions=())))
    graph = read_flash_effect_graph(binding)
    assert not graph.complete and 'owned body' in graph.issues[0]


def test_a_later_cycle_does_not_discard_guarded_prefix_evidence():
    body = _assemble([_push('started', 7) + b'\x1d', _push('flag') + b'\x1c', (0x9D, 'loop'),
                      _push(3) + b'\x3e', 'loop', _push('later', 19) + b'\x1d', (0x99, 'loop'), b'\0'])
    binding = _binding(body)
    graph = read_flash_effect_graph(binding)
    assert not graph.complete and not graph.effects_composed
    assert len(graph.exits) == 1
    assert 'loop summary' in graph.issues[0]
    retained_names = [graph.nodes[effect.operands[0]].data[0] for effect in graph.effects
                      if effect.operation == 'set_variable']
    assert retained_names == ['started']
    assert len(graph.lifted) < len(graph.reachable)
    assert _run_graph(graph, _initial(flag=False)) == _run_code(binding, _initial(flag=False))


def test_an_entry_cycle_never_supplies_a_completed_graph_or_an_invented_exit():
    body = _assemble(['loop', _push('later', 19) + b'\x1d', (0x99, 'loop')])
    graph = read_flash_effect_graph(_binding(body))
    assert graph.reachable and graph.lifted == () and graph.effects == () and graph.exits == ()
    assert not graph.complete and 'loop summary' in graph.issues[0]

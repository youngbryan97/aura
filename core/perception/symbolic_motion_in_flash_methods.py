"""Read bounded AVM1 method paths as arithmetic with exact branch evidence.

Only implemented arithmetic and stack operations are interpreted symbolically.
No game is executed, no source call is executed, and no expression is evaluated
with Python's eval. Unknown state splits paths; each resulting write retains
the predicates that made it reachable. Unsupported operations stop that path.
"""
from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from core.perception.flash_actions_with_source import (
    ActionFunction,
    ActionInstruction,
    ActionStream,
    parse_action_stream,
)
from core.verify import invariant

MAX_PATHS = 64
MAX_STEPS = 4096
MAX_STACK = 128
MAX_BINDINGS = 1024
MAX_WRITES = 256


@dataclass(frozen=True)
class Arithmetic:
    operation: str
    arguments: tuple[Any, ...] = ()


@dataclass(frozen=True)
class BranchEvidence:
    source: str
    offset: int
    condition: Arithmetic
    taken: bool


@dataclass(frozen=True)
class SourceInvocation:
    target: str
    receiver: Arithmetic
    receiver_source: str
    parameters: tuple[tuple[str, Arithmetic], ...]
    held_codes: frozenset[int] | None


@dataclass(frozen=True)
class SourceWrite:
    target: str
    value: Arithmetic
    source: str
    offset: int
    conditions: tuple[BranchEvidence, ...] = ()
    assumptions: tuple[tuple[str, float], ...] = ()
    invocation: SourceInvocation | None = None


@dataclass(frozen=True)
class SourceCall:
    target: str
    arguments: tuple[Arithmetic, ...]
    source: str
    offset: int
    conditions: tuple[BranchEvidence, ...] = ()
    assumptions: tuple[tuple[str, float], ...] = ()
    invocation: SourceInvocation | None = None


@dataclass(frozen=True)
class SourceDeletion:
    """An attempted property mutation; success and inherited values are unknown."""

    target: str
    result: Arithmetic
    source: str
    offset: int
    conditions: tuple[BranchEvidence, ...] = ()
    assumptions: tuple[tuple[str, float], ...] = ()
    invocation: SourceInvocation | None = None


@dataclass(frozen=True)
class MethodPath:
    conditions: tuple[BranchEvidence, ...]
    writes: tuple[SourceWrite, ...]
    calls: tuple[SourceCall, ...]
    returned: Arithmetic
    issues: tuple[str, ...] = ()
    assumptions: tuple[tuple[str, float], ...] = ()
    deletions: tuple[SourceDeletion, ...] = ()


@dataclass(frozen=True)
class FlashMethodBinding:
    target: str
    function: ActionFunction
    constant_pool: tuple[str, ...] | None


@dataclass(frozen=True)
class FlashMethodReading:
    target: str
    paths: tuple[MethodPath, ...]
    issues: tuple[str, ...] = ()
    invocation: SourceInvocation | None = None

    @property
    def complete(self) -> bool:
        return bool(self.paths) and not self.issues and all(not path.issues for path in self.paths)

    @property
    def effects_composed(self) -> bool:
        return self.complete and all(not path.calls and not path.assumptions and not path.deletions for path in self.paths)


@dataclass(frozen=True)
class FlashMethodBindings:
    methods: tuple[FlashMethodBinding, ...]
    issues: tuple[str, ...] = ()

    def method(self, target: str) -> FlashMethodBinding | None:
        candidates = [method for method in self.methods if method.target == target]
        return candidates[0] if len(candidates) == 1 else None


_UNDEFINED = Arithmetic('undefined')
_IDENTIFIER = re.compile(r'[A-Za-z_$][\w$]*\Z')


def _expression_is_bounded(value: Arithmetic) -> bool:
    pending = [(value, 0)]
    seen = {}
    while pending:
        expression, depth = pending.pop()
        if not isinstance(expression, Arithmetic) or depth > 32:
            return False
        if seen.get(id(expression), -1) >= depth:
            continue
        seen[id(expression)] = depth
        if len(seen) > 512 or len(expression.arguments) > 18:
            return False
        for argument in expression.arguments:
            if isinstance(argument, Arithmetic):
                pending.append((argument, depth + 1))
            elif isinstance(argument, str) and len(argument) > 4096:
                return False
            elif type(argument) in (float, int) and not math.isfinite(argument):
                return False
            elif not isinstance(argument, (str, float, int, bool, type(None))):
                return False
    return True


def _literal(value: Any) -> Arithmetic:
    if type(value) in (float, int):
        if not math.isfinite(value):
            raise _CannotReadError('nonfinite source number is unsupported')
        return Arithmetic('number', (float(value),))
    if isinstance(value, bool):
        return Arithmetic('boolean', (value,))
    if isinstance(value, str):
        return Arithmetic('string', (value,))
    if value is None:
        return Arithmetic('null')
    raise _CannotReadError('unsupported literal value')


def _number(value: Arithmetic) -> float | None:
    return value.arguments[0] if value.operation == 'number' else None


def _boolean(value: Arithmetic) -> bool | None:
    if value.operation in ('boolean', 'number', 'string'):
        return bool(value.arguments[0])
    if value.operation in ('null', 'undefined'):
        return False
    return None


def _calculation(operation: str, *values: Arithmetic) -> Arithmetic:
    if operation == 'not':
        truth = _boolean(values[0])
        return _literal(not truth) if truth is not None else Arithmetic(operation, values)
    numeric = operation in {'add', 'sub', 'mul', 'div', 'mod', 'neg', 'lt', 'gt',
                            'cos', 'sin', 'sqrt', 'abs', 'round', 'atan2', 'int'}
    numbers = tuple(float(value.arguments[0]) if numeric and value.operation == 'boolean'
                    else _number(value) for value in values)
    if all(value is not None for value in numbers):
        a, b = numbers[0], numbers[1] if len(numbers) > 1 else None
        choices = {'add': lambda: a + b, 'sub': lambda: a - b, 'mul': lambda: a * b,
                   'div': lambda: a / b, 'mod': lambda: math.fmod(a, b), 'neg': lambda: -a,
                   'eq': lambda: a == b, 'strict eq': lambda: a == b, 'legacy eq': lambda: a == b,
                   'lt': lambda: a < b, 'gt': lambda: a > b,
                   'cos': lambda: math.cos(a), 'sin': lambda: math.sin(a), 'sqrt': lambda: math.sqrt(a),
                   'abs': lambda: abs(a), 'round': lambda: math.floor(a + .5),
                   'atan2': lambda: math.atan2(a, b), 'int': lambda: math.trunc(a)}
        try:
            if operation in choices:
                value = choices[operation]()
                if isinstance(value, bool) or math.isfinite(value):
                    return _literal(value)
        except (ZeroDivisionError, OverflowError, ValueError):
            return Arithmetic('unsupported arithmetic', (operation,))
    primitive = {'string', 'boolean', 'null', 'undefined', 'number'}
    if operation in ('eq', 'strict eq', 'legacy eq') and all(value.operation in primitive for value in values):
        a, b = values
        if operation == 'strict eq' or a.operation == b.operation and operation != 'legacy eq':
            return _literal(a == b)
        if operation == 'eq' and {a.operation, b.operation} == {'null', 'undefined'}:
            return _literal(True)
        if {a.operation, b.operation} <= {'boolean', 'number'}:
            return _literal(float(a.arguments[0]) == float(b.arguments[0]))
        return Arithmetic('unsupported arithmetic', ('unsupported coercive equality operands',))
    return Arithmetic(operation, values)


def _pools_at(stream: ActionStream, inherited: tuple[str, ...] | None) -> dict[int, tuple[str, ...] | None]:
    """Join actual CFG predecessors; conflicting or unknown pools stay unknown."""
    if not stream.instructions:
        return {}
    instructions = {instruction.offset: instruction for instruction in stream.instructions}
    pools = {stream.instructions[0].offset: inherited}
    queue = deque([stream.instructions[0].offset])
    steps = 0
    while queue and steps < MAX_STEPS * 4:
        steps += 1
        offset = queue.popleft()
        instruction = instructions[offset]
        current = pools[offset]
        if instruction.opcode == 0x88:
            current = tuple(str(value.value) for value in instruction.operands)
        for after in instruction.successors:
            if after not in instructions:
                continue
            if after not in pools:
                pools[after] = current
                queue.append(after)
            elif pools[after] != current and pools[after] is not None:
                pools[after] = None
                queue.append(after)
    return pools if not queue else {}


def bind_flash_methods(streams: tuple[ActionStream, ...]) -> FlashMethodBindings:
    """Bind qualified method assignments and a provable defining constant pool."""
    found = []
    issues = []
    def visit(stream: ActionStream, inherited: tuple[str, ...] | None, depth: int) -> None:
        if depth > 32 or len(found) > MAX_BINDINGS:
            issues.append('method binding budget exceeded')
            return
        if not stream.complete:
            issues.append(f'incomplete action source: {stream.source}')
            return
        pools = _pools_at(stream, inherited)
        for function in stream.functions:
            pool = pools.get(function.definition_offset)
            assignments = [assignment for assignment in stream.assignments
                           if assignment.function_offset == function.definition_offset]
            for assignment in assignments:
                if (function.definition_offset not in pools or assignment.instruction_offset not in pools
                        or any(_IDENTIFIER.fullmatch(part) is None for part in assignment.target)):
                    issues.append('method binding has unknown reachability or an unsupported literal target')
                    continue
                found.append(FlashMethodBinding('.'.join(assignment.target), function, pool))
            visit(function.body, pool, depth + 1)
    for stream in streams:
        visit(stream, (), 0)
    return FlashMethodBindings(tuple(found[:MAX_BINDINGS]), tuple(dict.fromkeys(issues)))


@dataclass
class _State:
    offset: int
    stack: list[Arithmetic] = field(default_factory=list)
    names: dict[str, Arithmetic] = field(default_factory=dict)
    registers: dict[int, Arithmetic] = field(default_factory=dict)
    pool: tuple[str, ...] | None = ()
    conditions: tuple[BranchEvidence, ...] = ()
    writes: list[SourceWrite] = field(default_factory=list)
    calls: list[SourceCall] = field(default_factory=list)
    visits: dict[int, int] = field(default_factory=dict)
    steps: int = 0
    assumptions: dict[str, float] = field(default_factory=dict)
    local_names: set[str] = field(default_factory=set)
    invocation: SourceInvocation | None = None
    deletions: list[SourceDeletion] = field(default_factory=list)
    invalidated: dict[str, tuple[str, int, int]] = field(default_factory=dict)

    def copy(self) -> _State:
        return replace(self, stack=list(self.stack), names=dict(self.names), registers=dict(self.registers),
                       writes=list(self.writes), calls=list(self.calls), visits=dict(self.visits),
                       assumptions=dict(self.assumptions), local_names=set(self.local_names),
                       deletions=list(self.deletions), invalidated=dict(self.invalidated))


class _CannotReadError(ValueError):
    pass


def _pop(state: _State) -> Arithmetic:
    if not state.stack:
        raise _CannotReadError('source stack underflow')
    return state.stack.pop()


def _name(value: Arithmetic) -> str:
    if value.operation not in ('string', 'reference'):
        raise _CannotReadError('dynamic property or variable name')
    name = value.arguments[0]
    if not isinstance(name, str) or len(name) > 256:
        raise _CannotReadError('source name exceeds its bound')
    return name


def _get(state: _State, path: str, constants: Mapping[str, float]) -> Arithmetic:
    if path in state.names:
        return state.names[path]
    changed = [stamp for name, stamp in state.invalidated.items() if path == name or path.startswith(name+'.')]
    if changed:
        source, offset, ordinal = max(changed, key=lambda stamp: stamp[2])
        return Arithmetic('after deletion', (path, source, offset, ordinal))
    if path == 'Math.PI':
        return _literal(math.pi)
    if path in constants:
        value = constants[path]
        if type(value) in (int, float) and math.isfinite(value):
            state.assumptions[path] = float(value)
            return _literal(value)
    return Arithmetic('reference', (path,))


def _member(receiver: Arithmetic, member: Arithmetic) -> str:
    if receiver.operation != 'reference':
        raise _CannotReadError('property receiver is not an object reference')
    if member.operation != 'string':
        raise _CannotReadError('dynamic property name is unresolved')
    name = _name(member)
    if _IDENTIFIER.fullmatch(name) is None:
        raise _CannotReadError('literal property cannot be represented as a qualified path')
    path = receiver.arguments[0] + '.' + name
    if len(path) > 512:
        raise _CannotReadError('source property path exceeds its bound')
    return path


def _arguments(state: _State) -> tuple[Arithmetic, ...]:
    count = _number(_pop(state))
    if count is None or int(count) != count or not 0 <= count <= 16:
        raise _CannotReadError('call has an unknown or excessive argument count')
    return tuple(_pop(state) for _ in range(int(count)))


def _call(target: str, arguments: tuple[Arithmetic, ...], state: _State,
          instruction: ActionInstruction, source: str, held_codes: frozenset[int] | None) -> Arithmetic:
    arities = {'cos': 1, 'sin': 1, 'sqrt': 1, 'abs': 1, 'round': 1, 'atan2': 2}
    if target.startswith('Math.'):
        operation = target.split('.')[1]
        if arities.get(operation) != len(arguments):
            raise _CannotReadError('unsupported mathematical call')
        value = _calculation(operation, *arguments)
        if value.operation == 'unsupported arithmetic':
            raise _CannotReadError('mathematical call is outside its domain')
        return value
    if target == 'Key.isDown' and held_codes is not None and len(arguments) == 1:
        code = _number(arguments[0])
        if code is None or int(code) != code:
            raise _CannotReadError('keyboard code is unknown')
        return _literal(int(code) in held_codes)
    state.calls.append(SourceCall(target, arguments, source, instruction.offset, state.conditions,
                                  tuple(sorted(state.assumptions.items())), state.invocation))
    return Arithmetic('call', (target, *arguments))


def _write(state: _State, path: str, value: Arithmetic, instruction: ActionInstruction,
           source: str, *, local: bool = False) -> None:
    if len(state.names) >= MAX_BINDINGS or len(state.writes)+len(state.deletions) >= MAX_WRITES:
        raise _CannotReadError('symbolic binding or write budget exceeded')
    if not _expression_is_bounded(value):
        raise _CannotReadError('symbolic expression budget exceeded')
    state.names[path] = value
    if not local:
        state.writes.append(SourceWrite(path, value, source, instruction.offset, state.conditions,
                                        tuple(sorted(state.assumptions.items())), state.invocation))


def _delete(state: _State, path: str, instruction: ActionInstruction, source: str) -> Arithmetic:
    """Keep the side effect and unknown result without assuming a writable slot."""
    if len(state.writes)+len(state.deletions) >= MAX_WRITES:
        raise _CannotReadError('symbolic mutation budget exceeded')
    ordinal = len(state.deletions)+1
    result = Arithmetic('deletion result', (path, source, instruction.offset, ordinal))
    state.deletions.append(SourceDeletion(path, result, source, instruction.offset, state.conditions,
                                         tuple(sorted(state.assumptions.items())), state.invocation))
    # A failed delete can leave the old value; a successful one can reveal
    # inherited material. Neither case permits replaying a cached constant.
    state.names = {name: value for name, value in state.names.items()
                   if name != path and not name.startswith(path+'.')}
    state.invalidated[path] = source, instruction.offset, ordinal
    return result


def _method_path(state: _State, returned: Arithmetic, issues: tuple[str, ...] = ()) -> MethodPath:
    return MethodPath(state.conditions, tuple(state.writes), tuple(state.calls), returned, issues,
                      tuple(sorted(state.assumptions.items())), tuple(state.deletions))


def _instruction(state: _State, instruction: ActionInstruction, source: str,
                 constants: Mapping[str, float], held_codes: frozenset[int] | None) -> Arithmetic | None:
    """One supported, non-branch instruction. A result is a method return."""
    code = instruction.opcode
    binary = {0x0A: 'add', 0x0B: 'sub', 0x0C: 'mul', 0x0D: 'div', 0x0E: 'legacy eq', 0x0F: 'lt',
              0x3F: 'mod', 0x47: 'add', 0x48: 'lt', 0x49: 'eq', 0x66: 'strict eq', 0x67: 'gt'}
    if code == 0x88:
        state.pool = tuple(str(value.value) for value in instruction.operands)
    elif code == 0x96:
        for operand in instruction.operands:
            if operand.kind == 'register':
                value = state.registers.get(int(operand.value))
                if value is None:
                    raise _CannotReadError('unbound source register')
            elif operand.kind == 'constant':
                index = int(operand.value)
                if state.pool is None or not 0 <= index < len(state.pool):
                    raise _CannotReadError('unproven defining constant pool')
                value = _literal(state.pool[index])
            elif operand.kind in ('null', 'undefined'):
                value = Arithmetic(operand.kind)
            else:
                value = _literal(operand.value)
            state.stack.append(value)
    elif code in binary:
        b, a = _pop(state), _pop(state)
        value = _calculation(binary[code], a, b)
        if value.operation == 'unsupported arithmetic':
            raise _CannotReadError('arithmetic is outside its domain')
        state.stack.append(value)
    elif code == 0x12:
        state.stack.append(_calculation('not', _pop(state)))
    elif code == 0x1C:
        state.stack.append(_get(state, _name(_pop(state)), constants))
    elif code in (0x1D, 0x3C):
        value, path = _pop(state), _name(_pop(state))
        if code == 0x3C:
            state.local_names.add(path)
        _write(state, path, value, instruction, source, local=path in state.local_names)
    elif code == 0x41:
        name = _name(_pop(state))
        state.local_names.add(name)
        state.names[name] = _UNDEFINED
    elif code == 0x4E:
        member, receiver = _pop(state), _pop(state)
        state.stack.append(_get(state, _member(receiver, member), constants))
    elif code == 0x4F:
        value, member, receiver = _pop(state), _pop(state), _pop(state)
        _write(state, _member(receiver, member), value, instruction, source)
    elif code == 0x3A:
        member, receiver = _pop(state), _pop(state)
        state.stack.append(_delete(state, _member(receiver, member), instruction, source))
    elif code in (0x3D, 0x52):
        if code == 0x52:
            method, receiver = _pop(state), _pop(state)
            target = _member(receiver, method)
        else:
            target = _name(_pop(state))
        arguments = _arguments(state)
        state.stack.append(_call(target, arguments, state, instruction, source, held_codes))
    elif code == 0x17:
        _pop(state)
    elif code == 0x3E:
        return _pop(state)
    elif code == 0x4C:
        value = _pop(state)
        state.stack.extend((value, value))
    elif code == 0x4D:
        a, b = _pop(state), _pop(state)
        state.stack.extend((a, b))
    elif code == 0x87:
        if not state.stack:
            raise _CannotReadError('store from empty source stack')
        state.registers[int(instruction.operands[0].value)] = state.stack[-1]
    elif code in (0x50, 0x51):
        state.stack.append(_calculation('add' if code == 0x50 else 'sub', _pop(state), _literal(1)))
    elif code == 0x18:
        state.stack.append(_calculation('int', _pop(state)))
    elif code not in (0x00, 0x99, 0x9D):
        raise _CannotReadError(f'unsupported symbolic instruction {instruction.name}')
    if len(state.stack) > MAX_STACK:
        raise _CannotReadError('symbolic stack budget exceeded')
    if state.stack and not _expression_is_bounded(state.stack[-1]):
        raise _CannotReadError('symbolic expression budget exceeded')
    return None


def read_flash_method(binding: FlashMethodBinding, arguments: tuple[Any, ...] = (), *,
                      constants: Mapping[str, float] | None = None,
                      held_codes: frozenset[int] | None = None, receiver: str | None = None) -> FlashMethodReading:
    """Lift one method's supported paths; unknown guards remain explicit.

    Arguments may be literals or ``Arithmetic`` references. Calls to other
    source methods are recorded rather than executed. Their effects therefore
    still require separate composition before claiming a complete controller.
    Each effect keeps its reaching guards and input context. Later branches
    and later constant reads do not become conditions of an earlier effect.
    An explicit receiver describes this invocation only. A free function's
    absent receiver remains the symbolic "this", never its own method name.
    """
    if held_codes is not None and (not isinstance(held_codes, frozenset)
            or any(type(code) is not int or not 0 <= code <= 255 for code in held_codes)):
        return FlashMethodReading(binding.target, (), ('immutable finite keyboard codes are required',))
    function = binding.function
    if not function.body.complete or len(arguments) != len(function.parameters):
        return FlashMethodReading(binding.target, (), ('incomplete method or mismatched arguments',))
    if not function.body.instructions:
        return FlashMethodReading(binding.target, (), ('empty method body',))
    if any(isinstance(argument, Arithmetic) and not _expression_is_bounded(argument) for argument in arguments):
        return FlashMethodReading(binding.target, (), ('argument expression budget exceeded',))
    if receiver is not None and (not isinstance(receiver, str) or len(receiver) > 256
            or any(_IDENTIFIER.fullmatch(part) is None for part in receiver.split('.'))):
        return FlashMethodReading(binding.target, (), ('bounded literal receiver path is required',))
    owner = receiver if receiver is not None else binding.target.rsplit('.', 1)[0] if '.' in binding.target else 'this'
    names = {'this': Arithmetic('reference', (owner,))}
    try:
        names.update({name: value if isinstance(value, Arithmetic) else _literal(value)
                      for name, value in zip(function.parameters, arguments, strict=True)})
    except _CannotReadError as exc:
        return FlashMethodReading(binding.target, (), (str(exc),))
    registers = {register: names[name] for register, name in function.register_parameters}
    if function.flags & 0x1:
        registers.setdefault(1, names['this'])
    unsupported_preloads = function.flags & (0x4 | 0x10 | 0x40 | 0x80 | 0x100)
    if unsupported_preloads:
        return FlashMethodReading(binding.target, (), ('unsupported preloaded method environment',))
    invocation = SourceInvocation(binding.target, names['this'],
                                  'caller supplied' if receiver is not None else 'declaration owner' if '.' in binding.target else 'unresolved',
                                  tuple((name, names[name]) for name in function.parameters), held_codes)
    first = _State(function.body.instructions[0].offset, names=names, registers=registers,
                   pool=binding.constant_pool, local_names=set(function.parameters), invocation=invocation)
    pending = deque([first])
    instructions = {instruction.offset: instruction for instruction in function.body.instructions}
    paths = []
    issues = []
    while pending:
        if len(paths) + len(pending) > MAX_PATHS:
            issues.append('symbolic path budget exceeded')
            break
        state = pending.popleft()
        while state.offset in instructions:
            state.steps += 1
            state.visits[state.offset] = state.visits.get(state.offset, 0) + 1
            if state.steps > MAX_STEPS or state.visits[state.offset] > 2:
                issues.append('symbolic step or loop budget exceeded')
                break
            instruction = instructions[state.offset]
            try:
                if instruction.opcode == 0x00:
                    paths.append(_method_path(state, _UNDEFINED))
                    break
                if instruction.opcode == 0x99:
                    state.offset = instruction.branch_target
                    continue
                if instruction.opcode == 0x9D:
                    condition = _pop(state)
                    truth = _boolean(condition)
                    if truth is None:
                        if len(pending) + len(paths) + 2 > MAX_PATHS:
                            raise _CannotReadError('symbolic path budget exceeded')
                        other = state.copy()
                        other.conditions += (BranchEvidence(function.body.source, instruction.offset, condition, True),)
                        other.offset = instruction.branch_target
                        pending.append(other)
                        state.conditions += (BranchEvidence(function.body.source, instruction.offset, condition, False),)
                        state.offset = instruction.next_offset
                    else:
                        state.offset = instruction.branch_target if truth else instruction.next_offset
                    continue
                returned = _instruction(state, instruction, function.body.source, constants or {}, held_codes)
                if returned is not None:
                    paths.append(_method_path(state, returned))
                    break
                state.offset = instruction.next_offset
            except (_CannotReadError, IndexError, TypeError, ValueError, OverflowError) as exc:
                issue = f'{function.body.source}@{instruction.offset}: {exc}'
                issues.append(issue)
                paths.append(_method_path(state, _UNDEFINED, (issue,)))
                break
        else:
            if state.offset == function.body.end_offset:
                paths.append(_method_path(state, _UNDEFINED))
            else:
                issues.append('method path left its owned instruction addresses')
    return FlashMethodReading(binding.target, tuple(paths), tuple(dict.fromkeys(issues)), invocation)


@invariant('perception.source_effects_keep_their_reaching_prefix', scope='perception',
           owner='core/perception/symbolic_motion_in_flash_methods.py', observational=False)
def _source_effect_prefix_invariant() -> tuple:
    invocation = SourceInvocation('method', Arithmetic('reference', ('body',)), 'caller supplied', (), frozenset({37}))
    state = _State(0, invocation=invocation)
    instruction = ActionInstruction(0, 1, 1, 1, 0x1D, 'SetVariable', (), b'')
    _write(state, 'body.x', _literal(1), instruction, 'source')
    guard = BranchEvidence('source', 8, Arithmetic('reference', ('blocked',)), False)
    state.conditions += (guard,)
    state.assumptions['ceiling'] = 10.
    _call('notify', (), state, instruction, 'source', invocation.held_codes)
    assert state.writes[0].conditions == () and state.writes[0].assumptions == ()
    assert state.calls[0].conditions == (guard,) and state.calls[0].assumptions == (('ceiling', 10.),)
    assert state.calls[0].invocation == state.writes[0].invocation == invocation
    state.conditions = ()
    state.assumptions.clear()
    assert state.calls[0].conditions == (guard,) and state.calls[0].assumptions == (('ceiling', 10.),)
    return ()


@invariant('perception.deleted_source_properties_do_not_keep_cached_values', scope='perception',
           owner='core/perception/symbolic_motion_in_flash_methods.py', observational=False)
def _source_deletion_invariant() -> tuple:
    state = _State(0, names={'body.x': _literal(7), 'body.x.child': _literal(9)})
    instruction = ActionInstruction(8, 9, 9, 9, 0x3A, 'Delete', (), b'')
    result = _delete(state, 'body.x', instruction, 'source')
    assert _boolean(result) is None
    assert _get(state, 'body.x', {'body.x': 7.}).operation == 'after deletion'
    assert _get(state, 'body.x.child', {'body.x.child': 9.}).operation == 'after deletion'
    reading = FlashMethodReading('method', (_method_path(state, result),))
    assert reading.complete and not reading.effects_composed and reading.paths[0].deletions
    return ()


@invariant('perception.source_boolean_arithmetic_keeps_type_boundaries', scope='perception',
           owner='core/perception/symbolic_motion_in_flash_methods.py', observational=False)
def _source_boolean_arithmetic_invariant() -> tuple:
    stream = parse_action_stream(b'\x96\x07\x00\x05\x00\x07\x00\x00\x00\x00\x00')
    assert stream.complete
    false, zero = stream.instructions[0].operands
    assert false.kind == 'boolean' and type(false.value) is bool
    assert zero.kind == 'integer' and type(zero.value) is int
    assert _calculation('mul', _literal(4), _literal(True)) == _literal(4)
    assert _calculation('mul', _literal(4), _literal(False)) == _literal(0)
    assert _calculation('strict eq', _literal(False), _literal(0)) == _literal(False)
    assert _calculation('mul', _literal(4), Arithmetic('reference', ('flag',))).operation == 'mul'
    return ()

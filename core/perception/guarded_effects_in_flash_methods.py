"""Lift acyclic source control flow without enumerating execution paths.

Values and memory versions form one bounded DAG. A join selects predecessor
values under their reaching guards. Reads, coercions, calls and deletions keep
their memory order; none is replaced with a guessed value or pure function.

Completeness means supported normal bytecode control flow was lifted. This IR
does not execute AVM1, resolve aliases, compose calls, identify a visible body,
or certify a controller. Instruction addresses still refer to supplied bytes.
"""
from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass, field
from typing import Any

from core.perception.flash_actions_with_source import ActionInstruction, ActionStream
from core.perception.symbolic_motion_in_flash_methods import FlashMethodBinding
from core.verify import invariant

MAX_INSTRUCTIONS = 4096
MAX_NODES = 32768
MAX_EFFECTS = 1024
MAX_STACK = 128
_PATH = re.compile(r'[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*\Z')


@dataclass(frozen=True)
class EffectNode:
    operation: str
    inputs: tuple[int, ...] = ()
    data: tuple[Any, ...] = ()


@dataclass(frozen=True)
class GuardedEffect:
    offset: int
    guard: int
    operation: str
    operands: tuple[int, ...]
    before: int
    after: int
    result: int | None = None


@dataclass(frozen=True)
class GuardedBranch:
    offset: int
    guard: int
    condition: int
    taken: int
    untaken: int
    target: int
    fallthrough: int


@dataclass(frozen=True)
class GuardedExit:
    offset: int
    guard: int
    memory: int
    returned: int


@dataclass(frozen=True)
class FlashEffectGraph:
    source: str
    target: str
    nodes: tuple[EffectNode, ...] = ()
    effects: tuple[GuardedEffect, ...] = ()
    branches: tuple[GuardedBranch, ...] = ()
    exits: tuple[GuardedExit, ...] = ()
    lifted: tuple[int, ...] = ()
    reachable: tuple[int, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return bool(self.exits) and not self.issues and self.lifted == self.reachable

    @property
    def effects_composed(self) -> bool:
        # An ordered source IR supplies neither runtime values nor AVM semantics.
        return False

    def slice(self, *roots: int) -> tuple[int, ...]:
        """Return node dependencies; use ``slice_effect`` for a guarded event."""
        if any(type(root) is not int or not 0 <= root < len(self.nodes) for root in roots):
            raise ValueError('slice roots must belong to this graph')
        pending, found = list(roots), set()
        while pending:
            index = pending.pop()
            if index not in found:
                found.add(index)
                pending.extend(self.nodes[index].inputs)
        return tuple(sorted(found))

    def slice_effect(self, offset: int) -> tuple[int, ...]:
        """Keep the event's reaching guard beside its value and memory inputs."""
        candidates = [effect for effect in self.effects if effect.offset == offset]
        if type(offset) is not int or len(candidates) != 1:
            raise ValueError('one owned effect address is required')
        effect = candidates[0]
        roots = (effect.guard, effect.before, effect.after, *effect.operands)
        return self.slice(*roots, *((effect.result,) if effect.result is not None else ()))


class _CannotLiftError(ValueError):
    pass


class _Arena:
    def __init__(self) -> None:
        self.nodes: list[EffectNode] = []
        self.known: dict[EffectNode, int] = {}
        self.false = self.add('boolean', data=(False,))
        self.true = self.add('boolean', data=(True,))
        self.undefined = self.add('undefined')

    def add(self, operation: str, inputs: tuple[int, ...] = (), data: tuple = ()) -> int:
        node = EffectNode(operation, inputs, data)
        if node in self.known:
            return self.known[node]
        if len(self.nodes) >= MAX_NODES:
            raise _CannotLiftError('source graph node budget exceeded')
        index = len(self.nodes)
        if any(type(value) is not int or not 0 <= value < index for value in inputs):
            raise _CannotLiftError('source graph edge must precede its consumer')
        self.nodes.append(node)
        self.known[node] = index
        return index

    def literal(self, value: Any) -> int:
        if type(value) is bool:
            return self.true if value else self.false
        if type(value) in (int, float) and math.isfinite(value):
            return self.add('number', data=(float(value),))
        if isinstance(value, str) and len(value) <= 4096:
            return self.add('string', data=(value,))
        if value is None:
            return self.add('null')
        raise _CannotLiftError('unsupported or unbounded literal')

    def choose(self, guard: int, yes: int, no: int) -> int:
        if yes == no:
            return yes
        if guard in (self.true, self.false):
            return yes if guard == self.true else no
        return self.add('select', (guard, yes, no))

    def both(self, a: int, b: int) -> int:
        if self.false in (a, b):
            return self.false
        if a == b or b == self.true:
            return a
        return b if a == self.true else self.add('and', (a, b))

    def either(self, a: int, b: int) -> int:
        if self.true in (a, b):
            return self.true
        if a == b or b == self.false:
            return a
        return b if a == self.false else self.add('or', (a, b))

    def negate(self, value: int) -> int:
        if value in (self.true, self.false):
            return self.false if value == self.true else self.true
        node = self.nodes[value]
        return node.inputs[0] if node.operation == 'not' else self.add('not', (value,))


@dataclass
class _Frame:
    guard: int
    memory: int
    stack: list[int] = field(default_factory=list)
    registers: set[int] = field(default_factory=set)
    pool: tuple[str, ...] | None = ()

    def copy(self, *, guard: int) -> _Frame:
        return _Frame(guard, self.memory, list(self.stack), set(self.registers), self.pool)


def _successors(instruction: ActionInstruction, end: int) -> tuple[int, ...]:
    if instruction.offset == end or instruction.opcode in (0x00, 0x3E):
        return ()
    if instruction.opcode == 0x99:
        return (instruction.branch_target,)
    if instruction.opcode == 0x9D:
        return tuple(dict.fromkeys((instruction.branch_target, instruction.next_offset)))
    return (instruction.next_offset,)


def _ordered(stream: ActionStream) -> tuple[tuple[int, ...], dict[int, ActionInstruction]]:
    """Topological order, with a virtual exit for falling off the owned body."""
    instructions = {instruction.offset: instruction for instruction in stream.instructions}
    if len(instructions) != len(stream.instructions) or len(instructions) > MAX_INSTRUCTIONS:
        raise _CannotLiftError('duplicate addresses or source instruction budget exceeded')
    first, pending = stream.instructions[0].offset, [stream.instructions[0].offset]
    edges: dict[int, tuple[int, ...]] = {}
    while pending:
        offset = pending.pop()
        if offset in edges:
            continue
        if offset == stream.end_offset:
            edges[offset] = ()
            continue
        instruction = instructions.get(offset)
        if instruction is None:
            raise _CannotLiftError('branch leaves owned instruction addresses')
        edges[offset] = _successors(instruction, stream.end_offset)
        pending.extend(edges[offset])
    incoming = dict.fromkeys(edges, 0)
    for successors in edges.values():
        for successor in successors:
            incoming[successor] += 1
    queue = [offset for offset, count in incoming.items() if count == 0]
    heapq.heapify(queue)
    order = []
    while queue:
        offset = heapq.heappop(queue)
        order.append(offset)
        for successor in edges[offset]:
            incoming[successor] -= 1
            if not incoming[successor]:
                heapq.heappush(queue, successor)
    if len(order) != len(edges) or not order or order[0] != first:
        raise _CannotLiftError('cyclic source flow requires a separate loop summary')
    return tuple(order), instructions


def _merge(arena: _Arena, frames: list[_Frame]) -> _Frame:
    if not frames:
        raise _CannotLiftError('missing reachable predecessor state')
    result = frames[-1].copy(guard=frames[-1].guard)
    for frame in reversed(frames[:-1]):
        if len(frame.stack) != len(result.stack):
            raise _CannotLiftError('inconsistent source stack height at join')
        result.stack = [arena.choose(frame.guard, a, b) for a, b in zip(frame.stack, result.stack, strict=True)]
        # A register written on one branch alone remains unavailable after the join.
        result.registers &= frame.registers
        result.memory = arena.choose(frame.guard, frame.memory, result.memory)
        result.pool = frame.pool if frame.pool == result.pool else None
        result.guard = arena.either(frame.guard, result.guard)
    return result


def _pop(frame: _Frame) -> int:
    if not frame.stack:
        raise _CannotLiftError('source stack underflow')
    return frame.stack.pop()


def _arguments(arena: _Arena, frame: _Frame) -> tuple[int, ...]:
    count = arena.nodes[_pop(frame)]
    if count.operation != 'number' or not count.data[0].is_integer() or not 0 <= count.data[0] <= 16:
        raise _CannotLiftError('call has an unknown or excessive argument count')
    return tuple(_pop(frame) for _ in range(int(count.data[0])))


def _result(arena: _Arena, frame: _Frame, instruction: ActionInstruction, operation: str,
            operands: tuple[int, ...]) -> int:
    # A load can invoke a getter; coercion can call valueOf. Memory must advance
    # even for operations whose ordinary numeric instances happen to be pure.
    value = arena.add(operation, (frame.memory, *operands), (instruction.offset, instruction.opcode))
    frame.memory = arena.add('after', (value,))
    return value


_BINARY = {0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x13, 0x21, 0x29,
           0x3F, 0x47, 0x48, 0x49, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68}
_UNARY = {0x12, 0x14, 0x18, 0x30, 0x44, 0x4A, 0x4B, 0x50, 0x51}


def _step(arena: _Arena, frame: _Frame, instruction: ActionInstruction,
          effects: list[GuardedEffect]) -> None:
    code, before = instruction.opcode, frame.memory
    if code == 0x96:
        for operand in instruction.operands:
            if operand.kind == 'register':
                register = int(operand.value)
                if register not in frame.registers:
                    raise _CannotLiftError('unbound source register')
                value = _result(arena, frame, instruction, 'get_register', (arena.literal(register),))
            elif operand.kind == 'constant':
                index = int(operand.value)
                if frame.pool is None or not 0 <= index < len(frame.pool):
                    raise _CannotLiftError('unproven defining constant pool')
                value = arena.literal(frame.pool[index])
            elif operand.kind == 'undefined':
                value = arena.undefined
            else:
                value = arena.literal(operand.value)
            frame.stack.append(value)
    elif code == 0x88:
        frame.pool = tuple(str(operand.value) for operand in instruction.operands)
    elif code in _BINARY:
        b, a = _pop(frame), _pop(frame)
        frame.stack.append(_result(arena, frame, instruction, 'binary', (a, b)))
    elif code in _UNARY:
        frame.stack.append(_result(arena, frame, instruction, 'unary', (_pop(frame),)))
    elif code == 0x1C:
        frame.stack.append(_result(arena, frame, instruction, 'get_variable', (_pop(frame),)))
    elif code == 0x4E:
        member, receiver = _pop(frame), _pop(frame)
        frame.stack.append(_result(arena, frame, instruction, 'get_member', (receiver, member)))
    elif code in (0x1D, 0x3C, 0x4F, 0x41):
        if code == 0x4F:
            value, member, receiver = _pop(frame), _pop(frame), _pop(frame)
            operation, operands = 'set_member', (receiver, member, value)
        elif code == 0x41:
            operation, operands = 'define_local', (_pop(frame), arena.undefined)
        else:
            value, name = _pop(frame), _pop(frame)
            operation = 'define_local' if code == 0x3C else 'set_variable'
            operands = (name, value)
        frame.memory = arena.add(operation, (frame.memory, *operands), (instruction.offset, code))
        effects.append(GuardedEffect(instruction.offset, frame.guard, operation, operands, before, frame.memory))
    elif code in (0x3D, 0x52, 0x3A):
        if code in (0x52, 0x3A):
            member, receiver = _pop(frame), _pop(frame)
            operands = (receiver, member)
        else:
            operands = (_pop(frame),)
        operation = {0x3D: 'call_function', 0x52: 'call_method', 0x3A: 'delete_member'}[code]
        if code != 0x3A:
            operands += _arguments(arena, frame)
        value = _result(arena, frame, instruction, operation, operands)
        frame.stack.append(value)
        effects.append(GuardedEffect(instruction.offset, frame.guard, operation, operands, before, frame.memory, value))
    elif code == 0x17:
        _pop(frame)
    elif code == 0x4C:
        value = _pop(frame)
        frame.stack.extend((value, value))
    elif code == 0x4D:
        a, b = _pop(frame), _pop(frame)
        frame.stack.extend((a, b))
    elif code == 0x87:
        if not frame.stack:
            raise _CannotLiftError('store from empty source stack')
        register = int(instruction.operands[0].value)
        operands = (arena.literal(register), frame.stack[-1])
        frame.memory = arena.add('set_register', (frame.memory, *operands), (instruction.offset, code))
        frame.registers.add(register)
        effects.append(GuardedEffect(instruction.offset, frame.guard, 'set_register', operands, before, frame.memory))
    else:
        raise _CannotLiftError(f'unsupported graph instruction {instruction.name}')
    if len(frame.stack) > MAX_STACK or len(effects) > MAX_EFFECTS:
        raise _CannotLiftError('source graph stack or effect budget exceeded')


def read_flash_effect_graph(binding: FlashMethodBinding, *, receiver: str | None = None) -> FlashEffectGraph:
    """Lift one method on demand; parameters and the runtime heap remain inputs.

    Opaque calls are ordered source effects. Their return and memory effects
    remain unknown. The graph cannot be promoted to executable motion merely
    because every instruction was represented.
    """
    function, source = binding.function, binding.function.body.source
    base = dict(source=source, target=binding.target)
    if receiver is not None and (not isinstance(receiver, str) or len(receiver) > 256 or not _PATH.fullmatch(receiver)):
        return FlashEffectGraph(**base, issues=('bounded literal receiver path is required',))
    if not function.body.complete or not function.body.instructions:
        return FlashEffectGraph(**base, issues=('incomplete or empty method body',))
    if function.flags & (0x4 | 0x10 | 0x40 | 0x80 | 0x100):
        return FlashEffectGraph(**base, issues=('unsupported preloaded method environment',))
    if len(set(function.parameters)) != len(function.parameters):
        return FlashEffectGraph(**base, issues=('duplicate parameter names require scope resolution',))
    if function.flags & ~0x1FF or function.flags & 1 and any(register == 1 for register, _ in function.register_parameters):
        return FlashEffectGraph(**base, issues=('reserved flags or conflicting preloaded register',))
    arena, effects, branches, exits, lifted = _Arena(), [], [], [], []
    order: tuple[int, ...] = ()
    issues: list[str] = []
    try:
        order, instructions = _ordered(function.body)
        own = arena.add('receiver_input', data=(receiver, 'caller supplied' if receiver else 'unresolved'))
        arguments = {name: arena.add('argument_input', data=(index, name))
                     for index, name in enumerate(function.parameters)}
        memory = arena.add('entry_memory', (own, *arguments.values()),
                           (function.parameters, function.register_parameters, function.flags, function.register_count))
        registers = {register for register, _ in function.register_parameters}
        if function.flags & 1:
            registers.add(1)
        first = _Frame(arena.true, memory, registers=registers, pool=binding.constant_pool)
        arrivals: dict[int, list[_Frame]] = {order[0]: [first]}
        for offset in order:
            frame = _merge(arena, arrivals.get(offset, []))
            instruction = instructions.get(offset)
            if instruction is None or instruction.opcode in (0x00, 0x3E):
                returned = _pop(frame) if instruction is not None and instruction.opcode == 0x3E else arena.undefined
                exits.append(GuardedExit(offset, frame.guard, frame.memory, returned))
            elif instruction.opcode == 0x9D:
                value = _pop(frame)
                condition = _result(arena, frame, instruction, 'truth', (value,))
                yes, no = arena.both(frame.guard, condition), arena.both(frame.guard, arena.negate(condition))
                branches.append(GuardedBranch(offset, frame.guard, condition, yes, no,
                                               instruction.branch_target, instruction.next_offset))
                arrivals.setdefault(instruction.branch_target, []).append(frame.copy(guard=yes))
                arrivals.setdefault(instruction.next_offset, []).append(frame.copy(guard=no))
            elif instruction.opcode == 0x99:
                arrivals.setdefault(instruction.branch_target, []).append(frame)
            else:
                _step(arena, frame, instruction, effects)
                arrivals.setdefault(instruction.next_offset, []).append(frame)
            lifted.append(offset)
    except (_CannotLiftError, KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        issues.append(f'{source}@{offset if "offset" in locals() else function.body_offset}: {exc}')
    return FlashEffectGraph(**base, nodes=tuple(arena.nodes), effects=tuple(effects), branches=tuple(branches),
                            exits=tuple(exits), lifted=tuple(lifted), reachable=order, issues=tuple(issues))


@invariant('perception.guarded_source_graph_is_ordered_and_not_an_executable_model', scope='perception',
           owner='core/perception/guarded_effects_in_flash_methods.py', observational=False)
def _guarded_source_graph_invariant() -> tuple:
    arena = _Arena()
    a, b = arena.add('input', data=('a',)), arena.add('input', data=('b',))
    chosen = arena.choose(a, arena.literal(3), arena.literal(7))
    joined = arena.either(a, arena.both(b, arena.negate(a)))
    graph = FlashEffectGraph('source', 'method', tuple(arena.nodes),
                             exits=(GuardedExit(1, joined, b, chosen),), lifted=(1,), reachable=(1,))
    assert graph.complete and not graph.effects_composed
    assert all(all(index < at for index in node.inputs) for at, node in enumerate(graph.nodes))
    assert chosen in graph.slice(chosen) and a in graph.slice(chosen)
    return ()

"""Bounded AVM1 action records with byte addresses and function ownership.

The text reader is useful for searching a program. It cannot establish which
instructions a branch skips. This sidecar retains that evidence without running
the program. Offsets address the supplied source bytes, including nested bodies.
Consumers must require ``complete`` before drawing control-flow conclusions.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from core.verify.invariants import invariant

MAX_ACTION_BYTES = 4 * 1024 * 1024
MAX_ACTIONS = 65_536
MAX_DEPTH = 32

_NAMES = {
    0x00: "End", 0x04: "NextFrame", 0x05: "PreviousFrame", 0x06: "Play",
    0x07: "Stop", 0x08: "ToggleQuality", 0x09: "StopSounds", 0x0A: "Add",
    0x0B: "Subtract", 0x0C: "Multiply", 0x0D: "Divide", 0x0E: "Equals",
    0x0F: "Less", 0x10: "And", 0x11: "Or", 0x12: "Not", 0x13: "StringEquals",
    0x14: "StringLength", 0x15: "StringExtract", 0x17: "Pop", 0x18: "ToInteger",
    0x1C: "GetVariable", 0x1D: "SetVariable", 0x20: "SetTarget2",
    0x21: "StringAdd", 0x22: "GetProperty", 0x23: "SetProperty",
    0x24: "CloneSprite", 0x25: "RemoveSprite", 0x26: "Trace", 0x27: "StartDrag",
    0x28: "EndDrag", 0x29: "StringLess", 0x2A: "Throw", 0x2B: "CastOp",
    0x2C: "ImplementsOp", 0x30: "RandomNumber", 0x31: "MBStringLength",
    0x32: "CharToAscii", 0x33: "AsciiToChar", 0x34: "GetTime",
    0x35: "MBStringExtract", 0x36: "MBCharToAscii", 0x37: "MBAsciiToChar",
    0x3A: "Delete", 0x3B: "Delete2", 0x3C: "DefineLocal", 0x3D: "CallFunction",
    0x3E: "Return", 0x3F: "Modulo", 0x40: "NewObject", 0x41: "DefineLocal2",
    0x42: "InitArray", 0x43: "InitObject", 0x44: "TypeOf", 0x45: "TargetPath",
    0x46: "Enumerate", 0x47: "Add2", 0x48: "Less2", 0x49: "Equals2",
    0x4A: "ToNumber", 0x4B: "ToString", 0x4C: "PushDuplicate", 0x4D: "StackSwap",
    0x4E: "GetMember", 0x4F: "SetMember", 0x50: "Increment", 0x51: "Decrement",
    0x52: "CallMethod", 0x53: "NewMethod", 0x54: "InstanceOf", 0x55: "Enumerate2",
    0x60: "BitAnd", 0x61: "BitOr", 0x62: "BitXor", 0x63: "BitLShift",
    0x64: "BitRShift", 0x65: "BitURShift", 0x66: "StrictEquals",
    0x67: "Greater", 0x68: "StringGreater", 0x69: "Extends",
    0x81: "GotoFrame", 0x83: "GetURL", 0x87: "StoreRegister",
    0x88: "ConstantPool", 0x8A: "WaitForFrame", 0x8B: "SetTarget",
    0x8C: "GotoLabel", 0x8D: "WaitForFrame2", 0x8E: "DefineFunction2",
    0x96: "Push", 0x99: "Jump", 0x9A: "GetURL2", 0x9B: "DefineFunction",
    0x9D: "If", 0x9E: "Call", 0x9F: "GotoFrame2",
}


@dataclass(frozen=True)
class ActionValue:
    kind: str
    value: str | int | float | bool | None


@dataclass(frozen=True)
class ActionIssue:
    offset: int
    reason: str


@dataclass(frozen=True)
class ActionInstruction:
    offset: int
    next_offset: int
    payload_offset: int
    payload_end: int
    opcode: int
    name: str
    operands: tuple[ActionValue, ...]
    payload: bytes
    branch_target: int | None = None
    successors: tuple[int, ...] = ()


@dataclass(frozen=True)
class QualifiedFunctionAssignment:
    target: tuple[str, ...]
    function_offset: int
    instruction_offset: int


@dataclass(frozen=True)
class ActionFunction:
    definition_offset: int
    body_offset: int
    end_offset: int
    name: str
    parameters: tuple[str, ...]
    register_parameters: tuple[tuple[int, str], ...]
    register_count: int
    flags: int
    body: ActionStream


@dataclass(frozen=True)
class ActionStream:
    source: str
    start_offset: int
    end_offset: int
    instructions: tuple[ActionInstruction, ...]
    functions: tuple[ActionFunction, ...]
    assignments: tuple[QualifiedFunctionAssignment, ...]
    issues: tuple[ActionIssue, ...]

    @property
    def complete(self) -> bool:
        return not self.issues and all(function.body.complete for function in self.functions)


class _InvalidActionError(ValueError):
    pass


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.pos + count > len(self.data):
            raise _InvalidActionError("truncated operand")
        result = self.data[self.pos:self.pos + count]
        self.pos += count
        return result

    def uint8(self) -> int:
        return self.take(1)[0]

    def uint16(self) -> int:
        return int.from_bytes(self.take(2), "little")

    def string(self) -> str:
        end = self.data.find(b"\0", self.pos)
        if end < 0:
            raise _InvalidActionError("unterminated string")
        raw = self.take(end - self.pos + 1)[:-1]
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    def finished(self) -> None:
        if self.pos != len(self.data):
            raise _InvalidActionError("unexpected operand bytes")


def _operands(opcode: int, payload: bytes) -> tuple[ActionValue, ...]:
    reader = _Reader(payload)
    out: list[ActionValue] = []
    if opcode == 0x96:
        while reader.pos < len(payload):
            kind = reader.uint8()
            if kind == 0:
                out.append(ActionValue("string", reader.string()))
            elif kind == 1:
                out.append(ActionValue("float", struct.unpack("<f", reader.take(4))[0]))
            elif kind in (2, 3):
                out.append(ActionValue("null" if kind == 2 else "undefined", None))
            elif kind in (4, 5, 8):
                value = reader.uint8()
                if kind == 5 and value not in (0, 1):
                    raise _InvalidActionError("invalid boolean")
                out.append(ActionValue({4: "register", 5: "boolean", 8: "constant"}[kind], bool(value) if kind == 5 else value))
            elif kind == 6:
                raw = reader.take(8)
                out.append(ActionValue("double", struct.unpack("<d", raw[4:] + raw[:4])[0]))
            elif kind == 7:
                out.append(ActionValue("integer", struct.unpack("<i", reader.take(4))[0]))
            elif kind == 9:
                out.append(ActionValue("constant", reader.uint16()))
            else:
                raise _InvalidActionError("unsupported push type")
    elif opcode == 0x88:
        count = reader.uint16()
        for _ in range(count):
            out.append(ActionValue("string", reader.string()))
    elif opcode in (0x99, 0x9D):
        out.append(ActionValue("branch_delta", struct.unpack("<h", reader.take(2))[0]))
    elif opcode in (0x81, 0x8A):
        out.append(ActionValue("frame", reader.uint16()))
        if opcode == 0x8A:
            out.append(ActionValue("skip_count", reader.uint8()))
    elif opcode in (0x87, 0x8D, 0x9A):
        out.append(ActionValue({0x87: "register", 0x8D: "skip_count", 0x9A: "flags"}[opcode], reader.uint8()))
    elif opcode in (0x8B, 0x8C, 0x83):
        out.append(ActionValue("string", reader.string()))
        if opcode == 0x83:
            out.append(ActionValue("string", reader.string()))
    elif opcode == 0x9F:
        flags = reader.uint8()
        if flags & ~3:
            raise _InvalidActionError("reserved GotoFrame2 flags")
        out.append(ActionValue("flags", flags))
        if flags & 2:
            out.append(ActionValue("scene_bias", reader.uint16()))
    reader.finished()
    return tuple(out)


def _function_header(opcode: int, payload: bytes) -> tuple[str, tuple[str, ...], tuple[tuple[int, str], ...], int, int, int]:
    reader = _Reader(payload)
    name, count = reader.string(), reader.uint16()
    registers, flags = (reader.uint8(), reader.uint16()) if opcode == 0x8E else (0, 0)
    params: list[str] = []
    register_params: list[tuple[int, str]] = []
    for _ in range(count):
        register = reader.uint8() if opcode == 0x8E else 0
        param = reader.string()
        params.append(param)
        if register:
            if register > registers:
                raise _InvalidActionError("parameter register exceeds register count")
            register_params.append((register, param))
    size = reader.uint16()
    reader.finished()
    return name, tuple(params), tuple(register_params), registers, flags, size


def parse_action_stream(data: bytes, *, source: str = "action stream", base_offset: int = 0) -> ActionStream:
    """Parse one bounded AS1/AS2 action list; never execute its instructions.

    ``source`` names the tag or event that supplied the bytes. ``base_offset``
    is its first action's byte offset in that source. An unsupported operation
    or invalid branch makes the result incomplete and withholds assignments.
    """
    if not isinstance(data, bytes) or not isinstance(base_offset, int) or base_offset < 0:
        raise ValueError("immutable bytes and a nonnegative offset are required")
    if not isinstance(source, str) or len(source) > 1024:
        raise ValueError("a bounded source name is required")
    if len(data) > MAX_ACTION_BYTES:
        return ActionStream(source, base_offset, base_offset + len(data), (), (), (),
                            (ActionIssue(base_offset, "action byte budget exceeded"),))
    return _parse(data, source, base_offset, [MAX_ACTIONS], 0)


def _parse(data: bytes, source: str, base: int, budget: list[int], depth: int) -> ActionStream:
    instructions: list[ActionInstruction] = []
    functions: list[ActionFunction] = []
    issues: list[ActionIssue] = []
    pos = 0
    while pos < len(data):
        start = pos
        if not budget[0] or depth > MAX_DEPTH:
            issues.append(ActionIssue(base + start, "action or nesting budget exceeded"))
            break
        budget[0] -= 1
        opcode = data[pos]
        pos += 1
        try:
            if opcode not in _NAMES:
                raise _InvalidActionError(f"unsupported opcode 0x{opcode:02x}")
            length = 0
            if opcode >= 0x80:
                if pos + 2 > len(data):
                    raise _InvalidActionError("truncated action header")
                length = int.from_bytes(data[pos:pos + 2], "little")
                pos += 2
            payload_start, payload_end = pos, pos + length
            if payload_end > len(data):
                raise _InvalidActionError("truncated action payload")
            payload, pos = data[pos:payload_end], payload_end
            operands: tuple[ActionValue, ...] = ()
            if opcode in (0x8E, 0x9B):
                name, params, registers, register_count, flags, size = _function_header(opcode, payload)
                if pos + size > len(data):
                    raise _InvalidActionError("truncated function body")
                body = _parse(data[pos:pos + size], source, base + pos, budget, depth + 1)
                functions.append(ActionFunction(base + start, base + pos, base + pos + size,
                                                name, params, registers, register_count, flags, body))
                pos += size
            else:
                operands = _operands(opcode, payload)
            branch = base + payload_end + int(operands[0].value) if opcode in (0x99, 0x9D) else None
            instructions.append(ActionInstruction(base + start, base + pos, base + payload_start,
                                                   base + payload_end, opcode, _NAMES[opcode], operands,
                                                   payload, branch))
            if opcode == 0:
                if any(data[pos:]):
                    raise _InvalidActionError("nonzero bytes after End")
                break
        except _InvalidActionError as exc:
            issues.append(ActionIssue(base + start, str(exc)))
            break
    linked = _successors(instructions, base + len(data), issues)
    complete = not issues and all(function.body.complete for function in functions)
    assignments = _qualified_assignments(linked, functions) if complete else ()
    return ActionStream(source, base, base + len(data), linked, tuple(functions), assignments, tuple(issues))


def _successors(instructions: list[ActionInstruction], end: int, issues: list[ActionIssue]) -> tuple[ActionInstruction, ...]:
    from dataclasses import replace

    boundaries = {instruction.offset for instruction in instructions} | {end}
    result: list[ActionInstruction] = []
    for index, instruction in enumerate(instructions):
        opcode = instruction.opcode
        following = instruction.next_offset
        if opcode in (0, 0x3E, 0x2A):
            edges: tuple[int, ...] = ()
        elif opcode in (0x99, 0x9D):
            target = instruction.branch_target
            if target not in boundaries:
                issues.append(ActionIssue(instruction.offset, "branch target is not an instruction boundary"))
                edges = ()
            else:
                edges = (int(target),) if opcode == 0x99 else tuple(dict.fromkeys((following, int(target))))
        elif opcode in (0x8A, 0x8D):
            count = int(instruction.operands[-1].value)
            skipped = index + 1 + count
            if skipped > len(instructions):
                issues.append(ActionIssue(instruction.offset, "wait skip exceeds action list"))
                edges = ()
            else:
                target = instructions[skipped].offset if skipped < len(instructions) else end
                edges = tuple(dict.fromkeys((following, target)))
        else:
            edges = (following,)
        result.append(replace(instruction, successors=edges))
    return tuple(result)


@dataclass(frozen=True)
class _Symbol:
    kind: str = "unknown"
    value: tuple[str, ...] | str | int = ""


def _qualified_assignments(instructions: tuple[ActionInstruction, ...], functions: list[ActionFunction]) -> tuple[QualifiedFunctionAssignment, ...]:
    """Keep ownership only where a bounded straight-line stack proves it.

    Every branch destination clears the stack, registers and constant pool.
    This loses some valid ownership, but cannot carry a predecessor's values
    across a join or infer that one alternative always executes.
    """
    bodies = {function.definition_offset: function for function in functions}
    targets = {edge for instruction in instructions if len(instruction.successors) != 1 or instruction.opcode == 0x99
               for edge in instruction.successors}
    stack: list[_Symbol] = []
    registers: dict[int, _Symbol] = {}
    pool: tuple[ActionValue, ...] = ()
    assignments: list[QualifiedFunctionAssignment] = []

    def pop() -> _Symbol:
        return stack.pop() if stack else _Symbol()

    for instruction in instructions:
        if instruction.offset in targets:
            stack.clear()
            registers.clear()
            pool = ()
        opcode = instruction.opcode
        if opcode == 0x88:
            pool = instruction.operands
        elif opcode == 0x96:
            for operand in instruction.operands:
                if operand.kind == "register":
                    stack.append(registers.get(int(operand.value), _Symbol()))
                else:
                    if operand.kind == "constant":
                        index = int(operand.value)
                        operand = pool[index] if index < len(pool) else ActionValue("unknown", None)
                    stack.append(_Symbol("string", operand.value) if operand.kind == "string" else _Symbol())
        elif opcode == 0x1C:
            name = pop()
            stack.append(_Symbol("path", (str(name.value),)) if name.kind == "string" else _Symbol())
        elif opcode == 0x4E:
            member, owner = pop(), pop()
            stack.append(_Symbol("path", (*owner.value, str(member.value)))
                         if owner.kind == "path" and member.kind == "string" else _Symbol())
        elif opcode == 0x87:
            registers[int(instruction.operands[0].value)] = stack[-1] if stack else _Symbol()
        elif opcode == 0x4C:
            stack.append(stack[-1] if stack else _Symbol())
        elif opcode == 0x4D:
            first, second = pop(), pop()
            stack.extend((first, second))
        elif opcode == 0x17:
            pop()
        elif opcode in (0x8E, 0x9B):
            function = bodies[instruction.offset]
            if function.name:
                assignments.append(QualifiedFunctionAssignment((function.name,), function.definition_offset, instruction.offset))
            else:
                stack.append(_Symbol("function", function.definition_offset))
        elif opcode in (0x1D, 0x3C, 0x4F):
            value, member = pop(), pop()
            owner = pop() if opcode == 0x4F else _Symbol("path", ())
            if value.kind == "function" and member.kind == "string" and owner.kind == "path":
                assignments.append(QualifiedFunctionAssignment((*owner.value, str(member.value)), int(value.value), instruction.offset))
        else:
            # Unmodelled stack effects cannot prove a later assignment target.
            stack.clear()
            registers.clear()
        if len(stack) > 256:
            stack.clear()
            registers.clear()
    return tuple(assignments)


@invariant("perception.flash_action_branches_keep_exact_byte_boundaries", scope="perception",
           owner="core/perception/flash_actions_with_source.py", observational=False)
def _action_address_invariant() -> tuple:
    valid = parse_action_stream(b"\x9d\x02\0\x01\0\x06\x07\0", base_offset=80)
    assert valid.complete and valid.instructions[0].successors == (85, 86)
    invalid = parse_action_stream(b"\x99\x02\0\xff\xff\0")
    assert not invalid.complete and invalid.assignments == ()
    return ()

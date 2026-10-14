"""Decode declared uniform fields from reflected, versioned buffer ranges.

Offsets and strides come from public reflection. Shader structures determine
field names and array extents. Uncovered bytes and changed resource generations
stay unknown; this decoder supplies values, never object or control identity.
"""
from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from typing import Any

from core.perception.shader_position_math import _shape
from core.perception.shader_position_source import (
    PositionProgram,
    ShaderType,
    UnsupportedPositionError,
)
from core.verify import invariant

# Public GLenum values for 32-bit scalar, vector and matrix uniform storage.
_TYPES = {5126: "float", 5124: "int", 5125: "uint", 35670: "bool",
          35664: "vec2", 35665: "vec3", 35666: "vec4",
          35667: "ivec2", 35668: "ivec3", 35669: "ivec4",
          35671: "bvec2", 35672: "bvec3", 35673: "bvec4",
          36294: "uvec2", 36295: "uvec3", 36296: "uvec4",
          35674: "mat2", 35675: "mat3", 35676: "mat4",
          35685: "mat2x3", 35686: "mat2x4", 35687: "mat3x2",
          35688: "mat3x4", 35689: "mat4x2", 35690: "mat4x3"}


@dataclass(frozen=True)
class BufferRange:
    context: int
    epoch: int
    buffer: int
    content: int
    offset: int
    data: bytes
    byte_order: str


@dataclass(frozen=True)
class UniformBinding:
    context: int
    epoch: int
    buffer: int
    content: int
    start: int
    size: int
    block: str | None = None


@dataclass(frozen=True)
class UniformMember:
    name: str
    gl_type: int
    offset: int
    size: int = 1
    array_stride: int = 0
    matrix_stride: int = 0
    row_major: bool = False


@dataclass(frozen=True)
class UniformValues:
    inputs: dict[str, Any] | None
    reason: str | None = None


def _path(name: str) -> tuple[str | int, ...]:
    if not isinstance(name, str) or len(name) > 512:
        raise UnsupportedPositionError("member_name")
    parts, cursor = [], 0
    for match in re.finditer(r"[A-Za-z_][A-Za-z_0-9]*|\[\d+\]", name):
        gap = name[cursor:match.start()]
        if gap != ("." if parts and not match[0].startswith("[") else ""):
            raise UnsupportedPositionError("member_path")
        parts.append(int(match[0][1:-1]) if match[0].startswith("[") else match[0])
        cursor = match.end()
    if cursor != len(name) or not parts or not isinstance(parts[0], str) or len(parts) > 16:
        raise UnsupportedPositionError("member_path")
    return tuple(parts)


def _child(kind: ShaderType, part: str | int, structures: dict) -> ShaderType:
    if type(part) is int:
        if not 0 <= part < kind.count:
            raise UnsupportedPositionError("declared_array_extent")
        return ShaderType(kind.name)
    if kind.count or kind.name not in structures:
        raise UnsupportedPositionError("declared_structure")
    fields = dict(structures[kind.name])
    if part not in fields:
        raise UnsupportedPositionError("undeclared_field")
    return fields[part]


def _empty(kind: ShaderType, structures: dict) -> Any:
    return [None] * kind.count if kind.count else {} if kind.name in structures else None


def _store(inputs: dict, path: tuple, value: Any, program: PositionProgram, structures: dict) -> None:
    declared = dict(program.inputs)
    root = path[0]
    if root not in declared:
        raise UnsupportedPositionError("undeclared_uniform")
    kind = declared[root]
    node, key = inputs, root
    for part in path[1:]:
        missing = node[key] is None if isinstance(node, list) else key not in node
        if missing:
            node[key] = _empty(kind, structures)
        node = node[key]
        kind = _child(kind, part, structures)
        key = part
    duplicate = node[key] is not None if isinstance(node, list) else key in node
    if duplicate:
        raise UnsupportedPositionError("duplicate_uniform_bytes")
    node[key] = value


def _declared(path: tuple, program: PositionProgram, structures: dict) -> ShaderType:
    declared = dict(program.inputs)
    if path[0] not in declared:
        raise UnsupportedPositionError("undeclared_uniform")
    kind = declared[path[0]]
    for part in path[1:]:
        kind = _child(kind, part, structures)
    return kind


def _read_member(member: UniformMember, address: int, binding: UniformBinding, data: BufferRange) -> Any:
    base, shape = _shape(_TYPES[member.gl_type])
    if (type(member.matrix_stride) is not int or member.matrix_stride < 0 or member.matrix_stride % 4
            or type(member.row_major) is not bool):
        raise UnsupportedPositionError("matrix_layout")
    if len(shape) == 2:
        component_width = shape[1] if member.row_major else shape[0]
        if member.matrix_stride < component_width * 4:
            raise UnsupportedPositionError("matrix_stride")
    elif member.matrix_stride or member.row_major:
        raise UnsupportedPositionError("nonmatrix_layout")
    code = {"float": "f", "int": "i", "uint": "I", "bool": "i"}[base]
    byte_order = "<" if data.byte_order == "little" else ">"

    def read(offset):
        if not binding.start <= offset <= binding.start + binding.size - 4:
            raise UnsupportedPositionError("outside_bound_range")
        relative = offset - data.offset
        if not 0 <= relative <= len(data.data) - 4:
            raise UnsupportedPositionError("uncaptured_uniform_bytes")
        value = struct.unpack_from(byte_order + code, data.data, relative)[0]
        if base == "float" and not math.isfinite(value):
            raise UnsupportedPositionError("nonfinite_uniform")
        return bool(value) if base == "bool" else value

    if not shape:
        return read(address)
    if len(shape) == 1:
        return tuple(read(address + i * 4) for i in range(shape[0]))
    rows, columns = shape
    return tuple(tuple(read(address + (r * member.matrix_stride + c * 4 if member.row_major
                                     else c * member.matrix_stride + r * 4))
                       for c in range(columns)) for r in range(rows))


def decode_uniform_block(program: PositionProgram, binding: UniformBinding, data: BufferRange,
                         members: tuple[UniformMember, ...]) -> UniformValues:
    """A block is decoded only from its own current captured content version."""
    if not isinstance(program, PositionProgram) or not isinstance(binding, UniformBinding) or not isinstance(data, BufferRange):
        return UniformValues(None, "binding_contract")
    ids = (binding.context, binding.epoch, binding.buffer, binding.content,
           data.context, data.epoch, data.buffer, data.content)
    if (any(type(v) is not int or v < 0 for v in ids) or binding.content < 1
            or (binding.context, binding.epoch, binding.buffer, binding.content)
            != (data.context, data.epoch, data.buffer, data.content)):
        return UniformValues(None, "resource_custody")
    if (type(binding.start) is not int or type(binding.size) is not int or type(data.offset) is not int
            or min(binding.start, data.offset) < 0 or not 0 < binding.size <= 2**31 - 1
            or type(data.data) is not bytes or len(data.data) > 2 * 1024 * 1024
            or data.byte_order not in {"little", "big"}
            or type(members) is not tuple or len(members) > 1024):
        return UniformValues(None, "buffer_budget_or_layout")
    inputs, structures, work = {}, dict(program.structures), 0
    try:
        for member in members:
            if (not isinstance(member, UniformMember) or type(member.gl_type) is not int or member.gl_type not in _TYPES
                    or type(member.offset) is not int or member.offset < 0 or member.offset % 4
                    or type(member.size) is not int or not 1 <= member.size <= 256
                    or type(member.array_stride) is not int or member.array_stride < 0 or member.array_stride % 4
                    or member.size > 1 and member.array_stride == 0):
                raise UnsupportedPositionError("uniform_layout")
            path = _path(member.name)
            if binding.block is not None:
                blocks = dict(program.blocks)
                if binding.block not in blocks:
                    raise UnsupportedPositionError("undeclared_block")
                if path[0] == binding.block:
                    instance = blocks[binding.block]
                    path = path[1:] if instance is None else (instance,) + path[1:]
                    if not path:
                        raise UnsupportedPositionError("member_path")
            indices = [i for i, item in enumerate(path) if type(item) is int]
            if member.size > 1 and not indices:
                raise UnsupportedPositionError("array_member_path")
            for index in range(member.size):
                at = list(path)
                if member.size > 1:
                    at[indices[-1]] += index
                at = tuple(at)
                kind = _declared(at, program, structures)
                if kind.count or _shape(kind.name) != _shape(_TYPES[member.gl_type]):
                    raise UnsupportedPositionError("reflected_type_mismatch")
                base, shape = _shape(kind.name)
                work += 1 if not shape else shape[0] if len(shape) == 1 else shape[0] * shape[1]
                if work > 8192:
                    raise UnsupportedPositionError("decoded_value_budget")
                value = _read_member(member, binding.start + member.offset + index * member.array_stride, binding, data)
                _store(inputs, at, value, program, structures)
        return UniformValues(inputs)
    except (UnsupportedPositionError, IndexError, TypeError, KeyError, ValueError) as why:
        return UniformValues(None, str(why))


@invariant("perception.uniform_bytes_keep_resource_versions", scope="perception",
           owner="core/perception/shader_uniforms_from_buffer_bytes.py", observational=False)
def _uniform_resource_invariant() -> tuple:
    from core.perception.shader_position_source import parse_positions

    source = "#version 300 es\nuniform vec4 place;void main(){gl_Position=place;}"
    program = parse_positions(source)
    data = BufferRange(1, 2, 3, 4, 16, struct.pack("<4f", 1., 2., 3., 4.), "little")
    binding = UniformBinding(1, 2, 3, 4, 16, 16)
    layout = (UniformMember("place", 35666, 0),)
    assert decode_uniform_block(program, binding, data, layout).inputs == {"place": (1., 2., 3., 4.)}
    changed = UniformBinding(1, 2, 3, 5, 16, 16)
    assert decode_uniform_block(program, changed, data, layout).inputs is None
    return ()

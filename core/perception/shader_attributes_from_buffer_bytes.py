"""Read one declared vertex input from its current public attribute binding.

Addressing preserves stride, offset and instance divisor. Missing vector
components use the API defaults; normalization follows the stored type.
Unsupported packed formats and changed buffer versions return unknown.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from core.perception.shader_position_math import _shape
from core.perception.shader_position_source import PositionProgram, UnsupportedPositionError
from core.perception.shader_uniforms_from_buffer_bytes import BufferRange
from core.verify import invariant

# Public component types: format, byte width, integral signedness.
_FORMATS = {5120: ("b", 1, "int"), 5121: ("B", 1, "uint"),
            5122: ("h", 2, "int"), 5123: ("H", 2, "uint"),
            5124: ("i", 4, "int"), 5125: ("I", 4, "uint"),
            5126: ("f", 4, "float"), 5131: ("e", 2, "float")}


@dataclass(frozen=True)
class AttributeBinding:
    name: str
    context: int
    epoch: int
    buffer: int
    content: int
    components: int
    component_type: int
    offset: int = 0
    stride: int = 0
    divisor: int = 0
    normalized: bool = False
    integer: bool = False


@dataclass(frozen=True)
class AttributeValue:
    value: tuple | float | int | None
    reason: str | None = None


def attribute_from_bytes(program: PositionProgram, binding: AttributeBinding, data: BufferRange,
                         *, vertex: int, instance: int = 0) -> AttributeValue:
    """Decode one enabled scalar/vector input, using only its needed components."""
    if not isinstance(program, PositionProgram) or not isinstance(binding, AttributeBinding) or not isinstance(data, BufferRange):
        return AttributeValue(None, "attribute_contract")
    identities = (binding.context, binding.epoch, binding.buffer, binding.content,
                  data.context, data.epoch, data.buffer, data.content)
    if (any(type(v) is not int or v < 0 for v in identities) or binding.content < 1
            or (binding.context, binding.epoch, binding.buffer, binding.content)
            != (data.context, data.epoch, data.buffer, data.content)):
        return AttributeValue(None, "resource_custody")
    if (type(binding.component_type) is not int or binding.component_type not in _FORMATS
            or type(binding.components) is not int or not 1 <= binding.components <= 4
            or any(type(v) is not int or not 0 <= v <= 2**31 - 1
                   for v in (binding.offset, binding.stride, binding.divisor, vertex, instance))
            or type(binding.normalized) is not bool or type(binding.integer) is not bool
            or type(data.offset) is not int or data.offset < 0 or type(data.data) is not bytes
            or len(data.data) > 2 * 1024 * 1024 or data.byte_order not in {"little", "big"}):
        return AttributeValue(None, "attribute_layout_or_budget")
    try:
        kinds = dict(program.inputs)
        kind = kinds.get(binding.name)
        if kind is None or kind.count:
            raise UnsupportedPositionError("undeclared_attribute")
        base, shape = _shape(kind.name)
        if len(shape) > 1 or base == "bool":
            raise UnsupportedPositionError("attribute_shader_type")
        wanted = shape[0] if shape else 1
        code, width, stored = _FORMATS[binding.component_type]
        if (binding.integer and (stored != base or stored == "float" or binding.normalized)
                or not binding.integer and base != "float"):
            raise UnsupportedPositionError("attribute_conversion")
        stride = binding.stride or width * binding.components
        if stride % width or binding.offset % width:
            raise UnsupportedPositionError("attribute_alignment")
        index = instance // binding.divisor if binding.divisor else vertex
        address = binding.offset + stride * index
        read_count = min(wanted, binding.components)
        relative = address - data.offset
        if not 0 <= relative <= len(data.data) - read_count * width:
            raise UnsupportedPositionError("uncaptured_attribute_bytes")
        endian = "<" if data.byte_order == "little" else ">"
        values = list(struct.unpack_from(endian + code * read_count, data.data, relative))
        if stored == "float" and any(not math.isfinite(v) for v in values):
            raise UnsupportedPositionError("nonfinite_attribute")
        if base == "float":
            if binding.normalized and stored in {"int", "uint"}:
                maximum = (2 ** (8 * width - 1) - 1) if stored == "int" else 2 ** (8 * width) - 1
                values = [max(-1., float(v) / maximum) if stored == "int" else float(v) / maximum for v in values]
            else:
                values = [float(v) for v in values]
        defaults = (0., 0., 0., 1.) if base == "float" else (0, 0, 0, 1)
        values.extend(defaults[len(values):wanted])
        return AttributeValue(tuple(values) if shape else values[0])
    except (UnsupportedPositionError, KeyError, ValueError, TypeError) as why:
        return AttributeValue(None, str(why))


@invariant("perception.vertex_bytes_keep_draw_addressing", scope="perception",
           owner="core/perception/shader_attributes_from_buffer_bytes.py", observational=False)
def _attribute_addressing_invariant() -> tuple:
    from core.perception.shader_position_source import parse_positions

    program = parse_positions("#version 300 es\nin vec2 point;void main(){gl_Position=vec4(point,0.0,1.0);}")
    data = BufferRange(1, 2, 3, 4, 0, bytes((255, 0, 91, 91, 128, 64)), "little")
    attribute = AttributeBinding("point", 1, 2, 3, 4, 2, 5121, stride=4, divisor=2, normalized=True)
    result = attribute_from_bytes(program, attribute, data, vertex=90, instance=2)
    assert result.value == (128 / 255, 64 / 255)
    assert attribute_from_bytes(program, attribute, data, vertex=90, instance=4).value is None
    changed = BufferRange(1, 2, 3, 5, 0, data.data, "little")
    assert attribute_from_bytes(program, attribute, changed, vertex=0).value is None
    return ()

"""Assemble admitted public triangle draws without assigning object identity."""
from __future__ import annotations

import struct
from dataclasses import dataclass

from core.perception.shader_uniforms_from_buffer_bytes import BufferRange
from core.verify import invariant


@dataclass(frozen=True)
class ElementBinding:
    context: int
    epoch: int
    buffer: int
    content: int
    component_type: int
    offset: int


@dataclass(frozen=True)
class DrawTriangles:
    indices: tuple[int, ...] | None
    reason: str | None = None


def _assembled(values: tuple[int | None, ...], mode: int) -> tuple[int, ...]:
    out, segment = [], []
    for vertex in (*values, None):
        if vertex is not None:
            segment.append(vertex)
            continue
        if mode == 4:
            triangles = (segment[i:i + 3] for i in range(0, len(segment) - 2, 3))
        elif mode == 5:
            triangles = ((segment[i + 1], segment[i], segment[i + 2]) if i % 2
                         else (segment[i], segment[i + 1], segment[i + 2]) for i in range(len(segment) - 2))
        else:
            triangles = ((segment[0], segment[i], segment[i + 1]) for i in range(1, len(segment) - 1))
        for triangle in triangles:
            if len(set(triangle)) == 3:
                out.extend(triangle)
        segment = []
    return tuple(out)


def triangle_draw(mode: int, count: int, *, first: int = 0, binding: ElementBinding | None = None,
                  data: BufferRange | None = None, fixed_restart: bool = True) -> DrawTriangles:
    """Admit triangles, strips and fans; preserve winding and fixed restart."""
    if (type(mode) is not int or mode not in (4, 5, 6) or type(count) is not int or not 0 <= count <= 4096
            or type(first) is not int or not 0 <= first <= 2**31 - 1 or type(fixed_restart) is not bool):
        return DrawTriangles(None, "draw_mode_or_budget")
    if binding is None:
        if data is not None or first + count > 2**31:
            return DrawTriangles(None, "array_draw_contract")
        values = tuple(range(first, first + count))
    else:
        if not isinstance(binding, ElementBinding) or not isinstance(data, BufferRange):
            return DrawTriangles(None, "element_contract")
        ids = (binding.context, binding.epoch, binding.buffer, binding.content,
               data.context, data.epoch, data.buffer, data.content)
        if (any(type(v) is not int or v < 0 for v in ids) or binding.content < 1
                or ids[:4] != ids[4:]):
            return DrawTriangles(None, "resource_custody")
        formats = {5121: ("B", 1, 255), 5123: ("H", 2, 65535), 5125: ("I", 4, 2**32 - 1)}
        if (type(binding.component_type) is not int or binding.component_type not in formats
                or type(binding.offset) is not int or binding.offset < 0
                or type(data.offset) is not int or data.offset < 0 or type(data.data) is not bytes
                or len(data.data) > 2 * 1024 * 1024 or data.byte_order not in {"little", "big"}):
            return DrawTriangles(None, "element_layout")
        code, width, restart = formats[binding.component_type]
        relative = binding.offset - data.offset
        if binding.offset % width or not 0 <= relative <= len(data.data) - count * width:
            return DrawTriangles(None, "uncaptured_element_bytes")
        raw = struct.unpack_from(("<" if data.byte_order == "little" else ">") + code * count, data.data, relative)
        values = tuple(None if fixed_restart and value == restart else value for value in raw)
    return DrawTriangles(_assembled(values, mode))


@invariant("perception.draw_triangles_keep_assembly_and_versions", scope="perception",
           owner="core/perception/triangles_from_public_draws.py", observational=False)
def _triangle_assembly_invariant() -> tuple:
    data = BufferRange(1, 2, 3, 4, 0, bytes((0, 1, 2, 3, 255, 4, 5, 6)), "little")
    binding = ElementBinding(1, 2, 3, 4, 5121, 0)
    assert triangle_draw(5, 8, binding=binding, data=data).indices == (0, 1, 2, 2, 1, 3, 4, 5, 6)
    changed = BufferRange(1, 2, 3, 5, 0, data.data, "little")
    assert triangle_draw(5, 8, binding=binding, data=changed).indices is None
    assert triangle_draw(4, 2).indices == ()
    return ()
